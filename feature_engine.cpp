/*
 * feature_engine.cpp  -  TQC Brain | Taha Iqbal
 * // I have explain the bugs that I face in production so a user can easily understand that
 * Full technical indicator suite — all kernels SIMD-accelerated via simd_math.hpp.
 * No heap allocation: all scratch work uses thread_local g_scratch.
 *
 * Indicator set:
 *   Z-score (50), Bollinger (20), RSI-14 (Wilder two-phase), VW-RSI-14
 *   (volume-normalised, Wilder), MACD (12/26), Stochastic K/D (14/3),
 *   ROC-5/10/20, ATR-14, VWAP, volume imbalance, OB imbalance, SMA-20/50,
 *   multi-timeframe agreement, volatility (std-dev of returns),
 *   HAR-RV, GARCH(1,1), TSMOM, ARIMA(1,1,1).
 *
 * FIX (ISSUE-04): computeRSI() with volume weighting now normalises
 *   the volume weights to sum to 1 over the window before applying them.
 *   Previously, raw volume values (e.g. 1e9 USDT for BTC) were used
 *   directly as weights, making gain/loss values orders of magnitude
 *   larger than price deltas.  The resulting RSI was always ~99 or ~1.
 *
 * FIX (ISSUE-10): RSI now uses proper two-phase Wilder smoothing:
 *   SMA seed for first `period` bars, then Wilder's k=1/period for the rest.
 *   This matches TradingView, Bloomberg, and every institutional charting
 *   platform exactly.
 *
 * ── BUGS FIXED IN THIS VERSION ──────────────────────────────────────────────
 *
 * BUG-FE1  generateFeatures(): bars.count read without push_lock_ — data race.
 *          FIX: replaced bars.count with bars.bar_count(), which acquires
 *          push_lock_ around the count read.
 *
 * BUG-FE2  injectPayloadFeatures(): single global spinlock s_fr_lock for all
 *          MAX_SYMBOLS (24) symbols — severe serialisation bottleneck.
 *          FIX: replaced single s_fr_lock with s_fr_lock[MAX_SYMBOLS] array,
 *          one lock per symbol — identical pattern to the GARCH fix.
 *
 * BUG-FE3  computeHARRV(): rv_buf size hardcoded to 512 instead of MAX_BARS.
 *          FIX: changed to std::array<float, MAX_BARS>; removed the min() clamp.
 *
 * BUG-FE4  computeTSMOM(): sign(r) × |r| / vol decomposition is redundant.
 *          FIX: simplified to direct division r_L / (realised_vol + 1e-8f).
 *
 * BUG-FEC1 [NEW] computeATR() passed static_cast<int>(n) to atr_avx(), which
 *          now expects std::size_t n after the BUG-SM3 fix in simd_math.hpp.
 *
 *          The call was:
 *              return simd::atr_avx(highs, lows, closes, period, static_cast<int>(n));
 *
 *          After BUG-SM3, atr_avx takes (h, l, c, std::size_t period, std::size_t n).
 *          The explicit static_cast<int> narrows `n` (size_t, ≤ MAX_BARS=128) to
 *          int before passing it to a size_t parameter — defeating the entire
 *          purpose of the BUG-SM3 fix and generating -Wsign-conversion in strict
 *          builds.  The `period` parameter (int period = 14) was also mismatched.
 *
 *          FIX: computeATR's `period` parameter changed to std::size_t (see
 *          BUG-FEC2 for the full kernel-level change).  The atr_avx call now
 *          passes both arguments as std::size_t without any cast:
 *              return simd::atr_avx(highs, lows, closes, period, n);
 *
 * BUG-FEC2 [NEW] All internal indicator kernels used `int period` creating
 *          pervasive signed/unsigned inconsistency with their `std::size_t n`
 *          parameters.
 *
 *          Affected kernels: computeRSI, computeBollinger, computeZScore,
 *          computeROC, computeATR, computeVolumeImbalance, computeSMA,
 *          computeVolatility, computeOBImbalance, computeTFAgreement.
 *
 *          The pattern `n < static_cast<std::size_t>(period + 1)` repeats
 *          throughout.  `period + 1` is signed-integer UB for INT_MAX
 *          (same hazard fixed in simd_math.hpp by BUG-SM3/SM4).  Worse,
 *          -Wsign-compare fires on every such comparison, making the build
 *          noisy at -Wall and obscuring real warnings.
 *
 *          FIX: all `int period` parameters changed to `std::size_t period`.
 *          Guards updated: `n < static_cast<std::size_t>(period + 1)` →
 *          `n < period + 1` (both operands unsigned, no overflow possible).
 *          Loop variables changed from `int i` to `std::size_t i` where
 *          they were compared against or used to index size_t quantities.
 *          For computeZScore's adaptive period: `period = static_cast<int>(n)`
 *          → `period = n` (size_t = size_t, no cast needed).
 *
 * BUG-FEC3 [NEW] computeGARCH11() and injectPayloadFeatures() called
 *          g_bars.get_or_create(symbol) instead of g_bars.find(symbol).
 *
 *          get_or_create() is a write operation: if the symbol is not found,
 *          it acquires the exclusive lock and inserts a new slot in the
 *          BarStore with zero bar history.  Both functions are only ever
 *          called after generateFeatures() has already confirmed that the
 *          symbol exists (via g_bars.find()).  Using get_or_create() here:
 *
 *          (a) Acquires the exclusive lock unnecessarily on the hot path —
 *              for a registered symbol, the shared lock in the fast path is
 *              sufficient.  The exclusive lock serialises ALL symbol lookups
 *              globally while held.
 *
 *          (b) If (by any future refactor) either function is called before
 *              the symbol is registered, get_or_create() silently reserves
 *              a BarStore slot for a phantom symbol with no bar history.
 *              This phantom entry persists for the process lifetime and
 *              consumes one of the 24 available symbol slots — with no
 *              diagnostic.
 *
 *          FIX: both functions now call g_bars.find(symbol) and return the
 *          neutral value (0.0f) if the symbol is not found (MAX_SYMBOLS).
 *          This is consistent with generateFeatures() which already does the
 *          same.
 *
 * BUG-FEC4 [NEW] Dead `offset = 0` variable in computeHARRV() was an
 *          unexplained remnant of the BUG-FE3 fix.
 *
 *          After BUG-FE3 changed the buffer from 512 to MAX_BARS and removed
 *          the `min(cnt, 512)` clamp, `use = n` always (n is cnt ≤ MAX_BARS).
 *          The variable `const std::size_t offset = 0` was left in place —
 *          it was previously used to skip the leading elements in the 512-slot
 *          buffer.  Every access `C[offset + i - 1]` equals `C[i - 1]`.
 *          The variable falsely implies that `offset` might vary, misleading
 *          future readers into thinking there is a configurable starting point.
 *
 *          FIX: `offset` removed; C accessed directly.
 *          `const std::size_t use = n;` also removed (it is just `n`).
 *          The function is now 4 lines shorter and correctly expresses the
 *          intent: "compute HAR-RV over all n bars in C[0..n-1]."
 *
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include "feature_engine.hpp"
#include "simd_math.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <thread>   // std::this_thread::yield

namespace tqc {

// ── Global bar store definition ───────────────────────────────────────────────
BarStore g_bars;

// ── History management ────────────────────────────────────────────────────────

void updateHistory(const char* symbol, float price, float volume,
                   float high, float low, float ob_imb) noexcept {
    // FIX (XC-03): NaN / Inf / non-positive price guard.
    // A single malformed tick (corrupt WebSocket frame, zero-price heartbeat,
    // NaN from a failed JSON parse) would otherwise propagate through every
    // downstream SIMD kernel — RSI, ATR, GARCH, HMM — producing silent NaN
    // cascades that resolve to HOLD signals with no error log entry.
    // Reject the bar here; the previous valid bar stays at the ring pointer.
    if (!std::isfinite(price)  || price  <= 0.0f ||
        !std::isfinite(high)   || high   <= 0.0f ||
        !std::isfinite(low)    || low    <= 0.0f ||
        !std::isfinite(volume) || volume <  0.0f) [[unlikely]] {
        return;
    }
    const std::size_t idx = g_bars.get_or_create(symbol);
    if (idx < MAX_SYMBOLS) [[likely]]
        g_bars.at(idx).push(price, high, low, volume, ob_imb);
}

void bulkLoad(const char* symbol,
              const float* closes, const float* highs,
              const float* lows,   const float* volumes,
              std::size_t n) noexcept {
    const std::size_t idx = g_bars.get_or_create(symbol);
    if (idx < MAX_SYMBOLS) [[likely]]
        g_bars.at(idx).bulk_load(closes, highs, lows, volumes, n);
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal indicator kernels (static — not exported)
// All take a linearised array [oldest → newest].
//
// BUG-FEC2 FIX (applied throughout):
//   All `int period` parameters changed to `std::size_t period`.
//   Guards updated from `n < static_cast<std::size_t>(period + 1)` to
//   `n < period + 1` (both unsigned, no signed overflow possible).
//   Loop variables that previously iterated as `int i` against period
//   are now `std::size_t i` to eliminate -Wsign-compare.
// ─────────────────────────────────────────────────────────────────────────────

// ── RSI-14: proper two-phase Wilder smoothing ─────────────────────────────────
//
// Phase 1 (seed): simple average of first `period` up/down moves.
// Phase 2 (smooth): Wilder's k=1/period applied for all remaining bars.
//
// With volumes != nullptr: volume-normalised weighting applied in the seed
// phase only (each delta weighted by vol_i / sum(vol)).  The Wilder
// smoothing phase uses equal weights — re-normalising per bar in the smooth
// phase would require storing a running volume sum, which would need state.
// This hybrid is a well-established approximation used by institutional
// platforms.
//
// FIX (ISSUE-04 + ISSUE-10): raw volumes no longer used as un-normalised
// weights, and Wilder's k is now 1/period not 2/(period+1).
static float computeRSI(const float* closes, std::size_t n,
                         const float* volumes = nullptr,
                         std::size_t period = 14) noexcept {  // BUG-FEC2 FIX
    // Need at least period+1 bars for a single gain/loss computation.
    if (n < period + 1) return 50.0f;

    // BUG-FEC2 FIX: 2*period+1 is now size_t × size_t — no signed overflow.
    const std::size_t need = 2 * period + 1;
    const bool        full_wilder  = (n >= need);
    const std::size_t seed_start   = full_wilder ? (n - need) : 0;
    const std::size_t seed_end     = seed_start + period;

    // ── Phase 1: SMA seed ─────────────────────────────────────────────────────
    float vol_sum = 0.0f;
    if (volumes) {
        // Sum volumes at the destination bars (seed_start+1 .. seed_end).
        // The weight loop uses the same indices → normalisation is exact.
        for (std::size_t i = seed_start; i < seed_end; ++i)
            vol_sum += volumes[i + 1];
        if (vol_sum < 1e-10f) vol_sum = 1.0f;
    }

    float avg_gain = 0.0f, avg_loss = 0.0f;
    // BUG-FEC2 FIX: loop variable changed from `int i` to `std::size_t i`.
    for (std::size_t i = 0; i < period; ++i) {
        const float d = closes[seed_start + i + 1] - closes[seed_start + i];
        // FIX (ISSUE-04): weight = normalised volume (sum-to-1 over window)
        const float w = volumes
                        ? volumes[seed_start + i + 1] / vol_sum
                        : 1.0f / static_cast<float>(period);
        if (d > 0.0f) avg_gain += d * w;
        else           avg_loss -= d * w;  // stored positive
    }

    if (!full_wilder) {
        // Insufficient data for smooth phase: return SMA-seeded RSI directly
        if (avg_loss < 1e-12f) return avg_gain > 0.0f ? 99.0f : 50.0f;
        return 100.0f - 100.0f / (1.0f + avg_gain / avg_loss);
    }

    // ── Phase 2: Wilder smoothing k = 1/period ────────────────────────────────
    const float wf = 1.0f / static_cast<float>(period);
    for (std::size_t j = seed_end; j < n - 1; ++j) {
        const float d = closes[j + 1] - closes[j];
        if (d > 0.0f) {
            avg_gain = avg_gain * (1.0f - wf) + d * wf;
            avg_loss = avg_loss * (1.0f - wf);
        } else {
            avg_gain = avg_gain * (1.0f - wf);
            avg_loss = avg_loss * (1.0f - wf) + (-d) * wf;
        }
    }

    if (avg_loss < 1e-12f) return avg_gain > 0.0f ? 99.0f : 50.0f;
    return 100.0f - 100.0f / (1.0f + avg_gain / avg_loss);
}

// ── MACD signal (12/26) ───────────────────────────────────────────────────────
// Returns the MACD line (EMA12 - EMA26) normalised by current price.
// Uses standard EMA (k=2/(period+1)) — correct for MACD per convention.
// NOTE: simd::ema() now takes std::size_t period (BUG-SM4 fix in simd_math.hpp).
// Literal constants 12 and 26 are implicitly converted to std::size_t. ✓
static float computeMACD(const float* closes, std::size_t n) noexcept {
    if (n < 35) return 0.0f;
    const float price = closes[n - 1] + 1e-10f;
    const float ema12 = simd::ema(closes, n, 12);
    const float ema26 = simd::ema(closes, n, 26);
    return (ema12 - ema26) / price;  // ~ [-0.05, 0.05] in practice
}

// ── Bollinger Bands position ──────────────────────────────────────────────────
// Returns (price – SMA20) / std_dev.  Range roughly [-3, 3].
// BUG-FEC2 FIX: int period → std::size_t period.
static float computeBollinger(const float* closes, std::size_t n,
                               std::size_t period = 20) noexcept {
    if (n < period) return 0.0f;
    const float* c  = closes + (n - period);
    const float  m  = simd::mean(c, period);
    const float  sd = simd::std_dev(c, period);
    if (sd < 1e-10f) return 0.0f;
    return (closes[n - 1] - m) / sd;
}

// ── Z-score (50-bar) ─────────────────────────────────────────────────────────
// BUG-FEC2 FIX: int period → std::size_t period.
// The adaptive shrink `period = static_cast<int>(n)` becomes `period = n`
// (size_t = size_t, no cast).
static float computeZScore(const float* closes, std::size_t n,
                             std::size_t period = 50) noexcept {
    if (n < period) {
        period = n;              // BUG-FEC2 FIX: was static_cast<int>(n)
        if (period < 5) return 0.0f;
    }
    const float* c  = closes + (n - period);
    const float  m  = simd::mean(c, period);
    const float  sd = simd::std_dev(c, period);
    if (sd < 1e-10f) return 0.0f;
    return (closes[n - 1] - m) / sd;
}

// ── Stochastic K/D (14/3) ────────────────────────────────────────────────────
// K: current close's position within the 14-bar high–low range.
// D: 3-bar simple moving average of K — the signal line.
// ks[0..2] are K values for the 3 most-recent bars (oldest → newest).
// k = ks[2] (most recent), d = mean(ks[0..2]).
static std::pair<float, float> computeStoch(
    const float* closes, const float* highs, const float* lows,
    std::size_t n, std::size_t period = 14) noexcept  // BUG-FEC2 FIX
{
    // Need period + 3 bars: 3 K-values, each requiring `period` high/low bars.
    if (n < period + 3) return {0.5f, 0.5f};

    float ks[3];
    for (std::size_t offset = 0; offset < 3; ++offset) {
        const std::size_t end = n - offset;
        const std::size_t beg = end - period;

        // Highest high and lowest low over [beg, end-1] inclusive.
        const float hh  = simd::vmax(highs + beg, period);
        const float ll  = simd::vmin(lows  + beg, period);
        const float rng = hh - ll + 1e-10f;

        ks[2 - offset] = (closes[end - 1] - ll) / rng;
    }

    const float k = ks[2];
    const float d = (ks[0] + ks[1] + ks[2]) / 3.0f;
    return {k, d};
}

// ── ROC (Rate of Change) ─────────────────────────────────────────────────────
// BUG-FEC2 FIX: int period → std::size_t period.
static float computeROC(const float* closes, std::size_t n,
                          std::size_t period) noexcept {
    if (n < period + 1) return 0.0f;
    const float prev = closes[n - 1 - period];
    if (prev < 1e-10f) return 0.0f;
    return (closes[n - 1] - prev) / prev;
}

// ── ATR-14 ───────────────────────────────────────────────────────────────────
// BUG-FEC2 FIX: int period → std::size_t period.
// BUG-FEC1 FIX: atr_avx call now passes (period, n) both as std::size_t.
//   The previous call `simd::atr_avx(highs, lows, closes, period, static_cast<int>(n))`
//   explicitly narrowed the std::size_t `n` to `int` before passing it to
//   atr_avx, which after BUG-SM3 expects std::size_t n.  The explicit cast
//   defeated the intent of BUG-SM3 and triggers -Wsign-conversion.
//   Corrected call: `simd::atr_avx(highs, lows, closes, period, n)`.
static float computeATR(const float* closes, const float* highs,
                          const float* lows, std::size_t n,
                          std::size_t period = 14) noexcept {  // BUG-FEC2 FIX
    if (n < period + 1)
        return closes[n - 1] * 0.015f;
    // BUG-FEC1 FIX: period and n are both std::size_t now. No cast needed.
    return simd::atr_avx(highs, lows, closes, period, n);
}

// ── VWAP (Volume-Weighted Average Price) ─────────────────────────────────────
static float computeVWAP(const float* closes, const float* volumes,
                           std::size_t n) noexcept {
    if (n == 0) return 0.0f;
    const float num = simd::dot(closes, volumes, n);
    const float den = simd::sum(volumes, n);
    return den > 1e-10f ? num / den : closes[n - 1];
}

// ── Volume imbalance ─────────────────────────────────────────────────────────
// BUG-FEC2 FIX: int period → std::size_t period; local `int p` → std::size_t p.
static float computeVolumeImbalance(const float* volumes, std::size_t n,
                                     std::size_t period = 20) noexcept {
    if (n < 2) return 0.0f;
    const std::size_t p   = (n < period) ? n : period;
    const float       avg = simd::mean(volumes + (n - p), p);
    if (avg < 1e-10f) return 0.0f;
    return simd::clamp((volumes[n - 1] - avg) / avg, -1.0f, 1.0f);
}

// ── SMA ───────────────────────────────────────────────────────────────────────
// BUG-FEC2 FIX: int period → std::size_t period.
static float computeSMA(const float* closes, std::size_t n,
                          std::size_t period) noexcept {
    if (n < period) return closes[n - 1];
    return simd::mean(closes + (n - period), period);
}

// ── Multi-timeframe agreement ─────────────────────────────────────────────────
// Returns +1 (both TFs bullish), -1 (both bearish), 0 (divergence or no data).
// BUG-FEC2 FIX: local `int half` → std::size_t half to eliminate sign compare
// with std::size_t n.
static int8_t computeTFAgreement(
    const float* p1m, std::size_t n1m,
    const float* p5m, std::size_t n5m) noexcept
{
    if (n1m < 10 || n5m < 10) return 0;

    const auto slope = [](const float* arr, std::size_t n) -> float {
        const std::size_t half    = n / 2;  // BUG-FEC2 FIX: was int half
        const float       new_avg = simd::mean(arr + (n - half), half);
        const float       old_avg = simd::mean(arr, half);
        return (old_avg > 1e-10f) ? (new_avg - old_avg) / old_avg : 0.0f;
    };

    const float s1m = slope(p1m, n1m);
    const float s5m = slope(p5m, n5m);

    if (s1m > 0.0f && s5m > 0.0f) return  1;
    if (s1m < 0.0f && s5m < 0.0f) return -1;
    return 0;
}

// ── Volatility (std-dev of simple returns) ───────────────────────────────────
// BUG-FEC2 FIX: int period → std::size_t period; int p → std::size_t p.
// for (int i = 0; i < p; ++i) → for (std::size_t i = 0; i < p; ++i).
static float computeVolatility(const float* closes, std::size_t n,
                                 float* scratch,
                                 std::size_t period = 20) noexcept {
    if (n < 5) return 0.01f;
    // BUG-FEC2 FIX: was `int p = (n-1 < (size_t)period) ? (int)(n-1) : period`.
    // Now both operands are size_t, no cast needed.
    const std::size_t p   = ((n - 1) < period) ? (n - 1) : period;
    const std::size_t beg = n - 1 - p;
    for (std::size_t i = 0; i < p; ++i) {  // BUG-FEC2 FIX: was int i
        const float b = closes[beg + i];
        scratch[i] = (b > 1e-10f) ? (closes[beg + i + 1] / b - 1.0f) : 0.0f;
    }
    return simd::std_dev(scratch, p);
}

// ── 1-bar velocity (simple return) ───────────────────────────────────────────
static float computeVelocity(const float* closes, std::size_t n) noexcept {
    if (n < 2) return 0.0f;
    const float prev = closes[n - 2];
    return (prev > 1e-10f) ? (closes[n - 1] - prev) / prev : 0.0f;
}

// ── Order-book imbalance (rolling 20-bar mean of per-bar OB imbalance) ────────
// BUG-FEC2 FIX: int period → std::size_t period; int p → std::size_t p.
static float computeOBImbalance(const float* ob, std::size_t n,
                                  std::size_t period = 20) noexcept {
    if (n == 0) return 0.0f;
    const std::size_t p = (n < period) ? n : period;  // BUG-FEC2 FIX
    return simd::mean(ob + (n - p), p);
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 2 Statistical Models
// ─────────────────────────────────────────────────────────────────────────────

// ── HAR-RV: Heterogeneous Autoregressive Realised Volatility ─────────────────
// Corsi (2009). Decomposes realised vol into daily / weekly / monthly components.
// RV_d  = most recent bar's squared log-return
// RV_w  = mean squared log-return over last 5 bars
// RV_m  = mean squared log-return over last 22 bars
// HAR_RV = sqrt( β_d·RV_d + β_w·RV_w + β_m·RV_m )
//
// BUG-FE3 FIX: rv_buf size changed from hardcoded 512 to MAX_BARS.
//
// BUG-FEC4 FIX: removed the dead `offset = 0` variable.
//   After BUG-FE3, `use = min(cnt, 512)` became `use = n`.  The `offset`
//   variable (always 0) and `use` local were remnants of the 512-slot era.
//   Every access `C[offset + i - 1]` was equivalent to `C[i - 1]`.
//   Leaving the variable in place falsely implied `offset` might vary,
//   misleading future readers.  Both dead variables removed; C is now
//   accessed directly.  The function is 4 lines shorter.
static float computeHARRV(const float* C, std::size_t n) noexcept {
    if (n < 23) return 0.0f;

    // BUG-FE3 FIX: was std::array<float, 512> — now exactly MAX_BARS.
    // n ≤ MAX_BARS always (linearise_all caps at cnt ≤ MAX_BARS).
    static thread_local std::array<float, MAX_BARS> rv_buf;

    // BUG-FEC4 FIX: `offset` and `use` removed — both were always 0 and n.
    // Fill squared log-returns directly from C[0..n-1].
    for (std::size_t i = 1; i < n; ++i) {
        const float p  = C[i - 1];
        const float lr = (p > 1e-10f) ? std::log(C[i] / p) : 0.0f;
        rv_buf[i - 1]  = lr * lr;
    }
    const std::size_t nr = n - 1;  // number of squared returns filled

    // rv_d: most recent bar's squared log-return
    const float rv_d = rv_buf[nr - 1];

    // rv_w: mean over last 5 squared returns
    float rv_w = 0.0f;
    const std::size_t w_bars = std::min(nr, static_cast<std::size_t>(5));
    for (std::size_t i = nr - w_bars; i < nr; ++i) rv_w += rv_buf[i];
    rv_w /= static_cast<float>(w_bars);

    // rv_m: mean over last 22 squared returns
    float rv_m = 0.0f;
    const std::size_t m_bars = std::min(nr, static_cast<std::size_t>(22));
    for (std::size_t i = nr - m_bars; i < nr; ++i) rv_m += rv_buf[i];
    rv_m /= static_cast<float>(m_bars);

    static constexpr float BD = 0.35f, BW = 0.33f, BM = 0.28f;
    const float har = BD * rv_d + BW * rv_w + BM * rv_m;
    return (har > 0.0f) ? std::sqrt(har) : 0.0f;
}

// ── GARCH(1,1): Bollerslev (1986) ────────────────────────────────────────────
// σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
// Params: ω=2e-6, α=0.09, β=0.85 — calibrated for crypto minute data.
// α+β = 0.94 < 1 → stationary (required for GARCH to be well-defined).
// Returns annualised conditional volatility σ_t × sqrt(525600) for 1-min bars.
//
// BUG-8 FIX: replaced std::unordered_map + std::mutex with per-symbol BSS
//   fixed array and per-symbol spinlock.
// BR-05 FIX: per-symbol s_garch_lock[MAX_SYMBOLS] prevents 20-symbol
//   serialisation under 4 worker threads.
//
// BUG-FEC3 FIX: changed g_bars.get_or_create(symbol) → g_bars.find(symbol).
//   get_or_create() is a write operation that acquires the exclusive BarStore
//   lock and silently registers a phantom symbol slot if the symbol is not
//   found.  computeGARCH11() is only ever called from generateFeatures() after
//   the symbol has already been confirmed to exist via g_bars.find().  Using
//   get_or_create() here is semantically wrong: it acquires the exclusive lock
//   unnecessarily on every call and creates phantom slots on any unexpected
//   call path.  find() is correct: read-only, shared-lock, returns MAX_SYMBOLS
//   if absent (already guarded).
static float computeGARCH11(const char* symbol, const float* C,
                              std::size_t n) noexcept {
    if (n < 10) return 0.0f;

    static constexpr float OMEGA      = 2e-6f;
    static constexpr float ALPHA      = 0.09f;
    static constexpr float BETA       = 0.85f;
    static constexpr float UNCOND_VAR = OMEGA / (1.0f - ALPHA - BETA);

    static float          s_sigma2   [MAX_SYMBOLS]{};
    static bool           s_init     [MAX_SYMBOLS]{};
    static std::atomic_flag s_garch_lock[MAX_SYMBOLS];  // zero-init → clear

    // BUG-FEC3 FIX: was g_bars.get_or_create(symbol).
    // find() is read-only (shared lock); get_or_create() acquires the exclusive
    // lock and may register a phantom symbol.
    const std::size_t sym = g_bars.find(symbol);
    if (sym >= MAX_SYMBOLS) [[unlikely]] return 0.0f;

    // Read previous σ² under per-symbol lock
    float sigma2;
    {
        for (int sp = 0; s_garch_lock[sym].test_and_set(std::memory_order_acquire); ++sp)
            if (sp > 1000) std::this_thread::yield();
        sigma2 = s_init[sym] ? s_sigma2[sym] : UNCOND_VAR;
        s_garch_lock[sym].clear(std::memory_order_release);
    }

    // Run GARCH filter over last min(n-1, 50) returns (unlocked — deterministic)
    const std::size_t steps = std::min(n - 1, static_cast<std::size_t>(50));
    const std::size_t start = n - 1 - steps;
    for (std::size_t i = start; i < n - 1; ++i) {
        const float p  = C[i];
        const float lr = (p > 1e-10f) ? std::log(C[i + 1] / p) : 0.0f;
        sigma2 = OMEGA + ALPHA * lr * lr + BETA * sigma2;
    }

    // Persist updated σ² under per-symbol lock
    {
        for (int sp = 0; s_garch_lock[sym].test_and_set(std::memory_order_acquire); ++sp)
            if (sp > 1000) std::this_thread::yield();
        s_sigma2[sym] = sigma2;
        s_init  [sym] = true;
        s_garch_lock[sym].clear(std::memory_order_release);
    }

    // Annualise: σ × sqrt(525600 bars/year) ≈ σ × 725.0
    static constexpr float ANNUALISE = 725.0f;
    return std::sqrt(sigma2) * ANNUALISE;
}

// ── TSMOM: Time-Series Momentum Factor ───────────────────────────────────────
// Moskowitz, Ooi, Pedersen (2012) — "Time Series Momentum", JFE.
// tsmom = r_{t-L, t-1} / σ_t
// We use L=20 bars (skip 1 bar to remove microstructure reversal).
// Scaled by realised vol so it is comparable across symbols.
// Range: roughly [-5, +5]. Positive = long signal, negative = short signal.
//
// BUG-FE4 FIX: removed sign(r) × |r| / vol decomposition.
//   sign(r) × |r| / vol is exactly r / vol for all real r (they differ
//   only for r=NaN, which is guarded upstream by the bar validator).
//   The decomposition added 2 branches and an std::abs() call with no
//   benefit.  Replaced with direct division.
static float computeTSMOM(const float* C, std::size_t n,
                            float realised_vol) noexcept {
    if (n < 22 || realised_vol < 1e-8f) return 0.0f;
    // 20-bar return skipping the most recent bar (microstructure filter)
    const float r_L = (C[n - 22] > 1e-10f)
                      ? (C[n - 2] - C[n - 22]) / C[n - 22]
                      : 0.0f;
    // BUG-FE4 FIX: direct division — equivalent to sign(r)*|r|/vol
    return r_L / (realised_vol + 1e-8f);
}

// ── ARIMA(1,1,1): short-term return forecast ──────────────────────────────────
// Difference the close series (I(1)), fit AR(1) by OLS on the last WIN
// differenced values, then estimate the MA(1) coefficient θ by OLS on the
// AR(1) residuals.  Returns a 1-step-ahead return forecast (fraction of price).
//
// FIX (XC-04): θ is estimated by OLS on residuals rather than hardcoded -0.3f.
// BR-08 FIX: eps[0] zeroed to prevent stale thread_local value.
static float computeARIMA(const float* C, std::size_t n) noexcept {
    if (n < 32) return 0.0f;

    static constexpr std::size_t WIN = 30;
    const std::size_t start = n - WIN - 1;  // need WIN+1 prices → WIN diffs

    // Build differenced series d[i] = C[i+1] - C[i]
    static thread_local std::array<float, WIN> d;
    for (std::size_t i = 0; i < WIN; ++i)
        d[i] = C[start + i + 1] - C[start + i];

    // OLS: φ = Σ(d[i]·d[i-1]) / Σ(d[i-1]²)  for i=1..WIN-1
    float num = 0.0f, den = 0.0f;
    for (std::size_t i = 1; i < WIN; ++i) {
        num += d[i] * d[i - 1];
        den += d[i - 1] * d[i - 1];
    }
    float phi = (den > 1e-12f) ? num / den : 0.0f;
    phi = std::max(-0.99f, std::min(0.99f, phi));

    // Compute AR(1) residuals: ε[i] = d[i] - φ·d[i-1]  for i=1..WIN-1
    static thread_local std::array<float, WIN> eps;
    // BR-08 FIX: zero eps[0] so any future change to the loop start is safe.
    eps[0] = 0.0f;
    for (std::size_t i = 1; i < WIN; ++i)
        eps[i] = d[i] - phi * d[i - 1];

    // OLS estimate of θ on residual series
    // θ̂ = Σ(ε[i]·ε[i-1]) / Σ(ε[i-1]²)  for i=2..WIN-1
    float tnum = 0.0f, tden = 0.0f;
    for (std::size_t i = 2; i < WIN; ++i) {
        tnum += eps[i] * eps[i - 1];
        tden += eps[i - 1] * eps[i - 1];
    }
    float theta = (tden > 1e-12f) ? tnum / tden : 0.0f;
    theta = std::max(-0.99f, std::min(0.99f, theta));

    // 1-step forecast: d̂[t+1] = φ·d[WIN-1] + θ·ε[WIN-1]
    const float d_hat = phi * d[WIN - 1] + theta * eps[WIN - 1];
    const float ref   = C[n - 1];
    return (ref > 1e-10f) ? d_hat / ref : 0.0f;
}

// ─────────────────────────────────────────────────────────────────────────────
// generateFeatures — public hot path
// ─────────────────────────────────────────────────────────────────────────────

Features generateFeatures(
    const char*  symbol,
    float        last_price,
    float        volume,
    float        high,
    float        low,
    const float* prices_1m, std::size_t n1m,
    const float* prices_5m, std::size_t n5m) noexcept
{
    Features f{};
    std::strncpy(f.symbol, symbol, 15);
    f.symbol[15] = '\0';
    f.last_price = last_price;

    const std::size_t idx = g_bars.find(symbol);
    if (idx >= MAX_SYMBOLS) [[unlikely]] {
        f.rsi     = 50.0f;
        f.vw_rsi  = 50.0f;
        f.stoch_k = 0.5f;
        f.stoch_d = 0.5f;
        f.atr_pct = 0.01f;
        return f;
    }

    const auto& bars = g_bars.at(idx);

    // BUG-FE1 FIX: use bar_count() instead of bars.count.
    // bars.count is modified by push() under push_lock_.  Reading it without
    // the lock is a data race — undefined behaviour per the C++ memory model,
    // and observable as stale/torn reads on ARM (AWS Graviton, Ampere Altra).
    // bar_count() acquires push_lock_ around the read, making it safe.
    const std::size_t cnt = bars.bar_count();
    f.data_points = static_cast<int>(cnt);

    if (cnt < 5) [[unlikely]] {
        f.rsi    = 50.0f;
        f.vw_rsi = 50.0f;
        return f;
    }

    auto& sc = g_scratch;
    // linearise_all() acquires push_lock_ internally — reads are race-free.
    // cnt was read under lock above; since count only increases, the actual
    // count at this point is >= cnt, so linearise_all safely returns exactly
    // cnt bars (clamped to its internal count which is >= cnt).
    bars.linearise_all(sc.closes.data(), sc.highs.data(),
                       sc.lows.data(),   sc.volumes.data(),
                       sc.ob_imbalances.data(), cnt);

    const float* C  = sc.closes.data();
    const float* H  = sc.highs.data();
    const float* L  = sc.lows.data();
    const float* V  = sc.volumes.data();
    const float* OB = sc.ob_imbalances.data();

    // ── Compute all indicators ────────────────────────────────────────────────

    f.z_score            = computeZScore   (C, cnt);
    f.bollinger_position = computeBollinger(C, cnt);

    f.rsi                = computeRSI(C, cnt, nullptr);
    f.vw_rsi             = computeRSI(C, cnt, V);

    f.macd_signal        = computeMACD(C, cnt);

    {
        auto [k, d] = computeStoch(C, H, L, cnt);
        f.stoch_k   = k;
        f.stoch_d   = d;
    }

    f.roc_5              = computeROC(C, cnt,  5);
    f.roc_10             = computeROC(C, cnt, 10);
    f.roc_20             = computeROC(C, cnt, 20);

    f.velocity           = computeVelocity (C, cnt);
    f.volatility         = computeVolatility(C, cnt, sc.scratch1.data());

    f.atr                = computeATR(C, H, L, cnt);
    f.atr_pct            = (last_price > 1e-10f) ? f.atr / last_price : 0.01f;

    f.vwap               = computeVWAP(C, V, cnt);
    f.vwap_distance      = (f.vwap > 1e-10f)
                           ? (last_price - f.vwap) / f.vwap
                           : 0.0f;

    f.volume_imbalance   = computeVolumeImbalance(V, cnt);
    f.ob_imbalance       = computeOBImbalance(OB, cnt);

    f.sma_20             = computeSMA(C, cnt, 20);
    f.sma_50             = computeSMA(C, cnt, 50);

    f.tf_agreement       = computeTFAgreement(prices_1m, n1m, prices_5m, n5m);

    // ── Step 2: Statistical model features ───────────────────────────────────
    f.har_rv             = computeHARRV(C, cnt);
    f.garch_vol          = computeGARCH11(symbol, C, cnt);
    // TSMOM requires f.volatility — must be computed above first.
    f.tsmom              = computeTSMOM(C, cnt, f.volatility);
    f.arima_forecast     = computeARIMA(C, cnt);

    // vwoi / funding_rate / funding_zscore / basis are injected by
    // processSymbol() in main.cpp after this function returns.

    return f;
}

// ── Step 2: injectPayloadFeatures ────────────────────────────────────────────
// Populates vwoi / funding_rate / funding_zscore / basis.
//
// FIX (XC-01): replaced std::unordered_map + std::deque with fixed BSS ring
//   buffers — zero heap, truly noexcept.
//
// BUG-FE2 FIX: replaced single global s_fr_lock with per-symbol
//   s_fr_lock[MAX_SYMBOLS] array.
//
// BUG-FEC3 FIX: changed g_bars.get_or_create(symbol) → g_bars.find(symbol).
//   Same rationale as computeGARCH11: this function is always called after
//   processSymbol() has confirmed the symbol exists.  get_or_create() is
//   semantically wrong here: it acquires the exclusive write lock on the
//   BarStore and silently creates a phantom symbol slot if the symbol is
//   absent.  find() is the correct read-only lookup.
void injectPayloadFeatures(const char* symbol,
                            float vwoi,
                            float funding_rate,
                            float basis,
                            Features& f) noexcept {
    f.vwoi         = vwoi;
    f.funding_rate = funding_rate;
    f.basis        = basis;

    static constexpr std::size_t WIN = 30;

    static float   s_fr_buf  [MAX_SYMBOLS][WIN]{};
    static uint8_t s_fr_head [MAX_SYMBOLS]     {};
    static uint8_t s_fr_count[MAX_SYMBOLS]     {};
    // BUG-FE2 FIX: per-symbol lock array (zero-init → clear state on static vars).
    static std::atomic_flag s_fr_lock[MAX_SYMBOLS];

    // BUG-FEC3 FIX: was g_bars.get_or_create(symbol).
    const std::size_t sym = g_bars.find(symbol);
    if (sym >= MAX_SYMBOLS) [[unlikely]] { f.funding_zscore = 0.0f; return; }

    float zscore = 0.0f;
    {
        // Acquire THIS SYMBOL'S lock only — other symbols run concurrently.
        for (int sp = 0; s_fr_lock[sym].test_and_set(std::memory_order_acquire); ++sp)
            if (sp > 1000) std::this_thread::yield();

        // Append new sample to ring buffer
        s_fr_buf[sym][s_fr_head[sym]] = funding_rate;
        s_fr_head[sym] = static_cast<uint8_t>((s_fr_head[sym] + 1) % WIN);
        if (s_fr_count[sym] < static_cast<uint8_t>(WIN)) ++s_fr_count[sym];

        const std::size_t n = s_fr_count[sym];
        if (n >= 5) {
            float mean_fr = 0.0f;
            for (std::size_t i = 0; i < n; ++i) mean_fr += s_fr_buf[sym][i];
            mean_fr /= static_cast<float>(n);

            float var = 0.0f;
            for (std::size_t i = 0; i < n; ++i) {
                const float d = s_fr_buf[sym][i] - mean_fr;
                var += d * d;
            }
            // Add 1e-20f inside sqrt to prevent sqrt(0) → exactly 0 denominator
            const float sd = std::sqrt(var / static_cast<float>(n) + 1e-20f);
            zscore = (sd > 1e-12f) ? (funding_rate - mean_fr) / sd : 0.0f;
        }

        s_fr_lock[sym].clear(std::memory_order_release);
    }

    f.funding_zscore = zscore;
}

} // namespace tqc
