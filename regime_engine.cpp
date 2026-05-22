/*
 * regime_engine.cpp  - TQC Brain | Taha Iqbal
 *
 * Market regime classification: TRENDING / MEAN_REVERTING / NOISE.
 * // I have remove the numerical thresholds because they are my proprietry
 * Pipeline per call:
 *   1. hurstRS()       — Hurst exponent via R/S analysis
 *   2. adx()           — ADX trend-strength (0→1 normalised)
 *   3. regSlope()      — 20-bar linear-regression slope (normalised)
 *   4. volState()      — volatility regime (LOW/NORMAL/HIGH)
 *   5. hmmViterbi()    — 3-state HMM Viterbi decoder with online M-step
 *   6. classifyRegime()— combine all five into RegimeInfo
 *
 * ── BUGS FIXED IN THIS VERSION ──────────────────────────────────────────────
 *
 * BUG-RE1  classifyRegime(): bars.count read without push_lock_ — data race.
 *          FIX: both the guard check and the main read use bar_count(),
 *          which acquires push_lock_ around the read.
 *
 * BUG-RE2  hmmViterbi(): double forward pass — 100% redundant work.
 *          FIX: merged into one pass; halves log() call count.
 *
 * BUG-RE3  volState(): copy+abs loop is fragile and uses extra bandwidth.
 *          FIX: simd::abs_diff() writes |scratch1| into scratch2 in one pass.
 *
 * BUG-RE4  classifyRegime(): guard-site read of bars.count — same race as RE1.
 *          FIX: unified into single bar_count() call.
 *
 * BUG-RE5  [NEW] HMM observation symbol 3 ("flat high-vol") was structurally
 *          unreachable — the assignment condition was self-contradictory.
 *
 *          
 *
 *    
 *
 *       
 *
 *          
 *
 * BUG-RE6  [NEW] adx() and regSlope() used `int period` with `std::size_t n`
 *          — same BUG-FEC2 pattern fixed across feature_engine.cpp.
 *
 *       
 *
 *          
 * BUG-RE7  [NEW] `!hi_vol` in obs=1 condition was dead code masking BUG-RE5.
 *
 *         
 *
 *          F
 *
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include "regime_engine.hpp"
#include "feature_engine.hpp"
#include "bar_store.hpp"
#include "simd_math.hpp"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <thread>   // std::this_thread::yield

namespace tqc {

// ── Hurst Exponent (R/S Analysis) ────────────────────────────────────────────
// Returns H ∈ [0, 1].
//   H >  → trending (persistent)
//   H <  → mean-reverting (anti-persistent)
//   H ≈  → random walk (NOISE)
//
// Requires n ≥ ~80-100 bars for a meaningful estimate (needs ≥ 4 lag points;
// max_k = min(6, n/4) with each lag ≥ 4 requires n ≥ 80-100).
// During the first ~80 minutes after startup the function returns 0.5f
// (neutral/NOISE), which suppresses all trades during the warm-up window.
// This is intentional cold-start behaviour.
static float hurstRS(const float* prices, std::size_t n) noexcept {
    if (n < ) [[unlikely]] return f;

    auto& sc = g_scratch;
    const std::size_t nr = n - 1;
    for (std::size_t i = 0; i < nr; ++i) {
        const float p  = prices[i];
        sc.scratch1[i] = (p > 1e-10f) ? std::log(prices[i + 1] / p) : 0.0f;
    }
    const float* lr = sc.scratch1.data();

    if (nr < 8) [[unlikely]] return 0.5f;

    int lags[6]; int nlags = 0;
    const int max_k = std::min(6, static_cast<int>(n / 4));
    for (int k = 2; k <= max_k && nlags < 6; ++k) {
        const int lag = static_cast<int>(n) / k;
        if (lag >= 4) lags[nlags++] = lag;
    }
    if (nlags < 2) [[unlikely]] return 0.5f;

    float rs_log[6], lag_log[6]; int npts = 0;

    for (int li = 0; li < nlags; ++li) {
        const int lag = lags[li];
        float rs_sum  = 0.0f; int nchunks = 0;

        for (int start = 0; start + lag <= static_cast<int>(nr); start += lag) {
            const float* chunk = lr + start;
            if (lag < 4) continue;

            const float m = simd::mean(chunk, lag);

            float cum = 0.0f, cmax = -1e30f, cmin = 1e30f;
            for (int i = 0; i < lag; ++i) {
                cum  += chunk[i] - m;
                cmax  = std::max(cmax, cum);
                cmin  = std::min(cmin, cum);
            }
            const float s = simd::std_dev(chunk, lag);
            if (s > 1e-12f) { rs_sum += (cmax - cmin) / s; ++nchunks; }
        }

        if (nchunks > 0) {
            rs_log[npts]  = std::log(rs_sum / nchunks + 1e-10f);
            lag_log[npts] = std::log(static_cast<float>(lag));
            ++npts;
        }
    }

    // M-06 FIX: with only 2 data points any linear fit has R²=1.0 by definition —
    // the slope estimate has zero statistical validity.  A spurious Hurst of 0.85
    // from 2 log-RS points misclassifies a short series as strongly trending.
    // Minimum 4 lag points provides the floor for a meaningful regression.
    if (npts < 4) [[unlikely]] return 0.5f;

    float slope = 0.5f;
    {
        float sx = 0.0f, sy = 0.0f, sxy = 0.0f, sx2 = 0.0f;
        const float fn = static_cast<float>(npts);
        for (int i = 0; i < npts; ++i) {
            sx  += lag_log[i];
            sy  += rs_log[i];
            sxy += lag_log[i] * rs_log[i];
            sx2 += lag_log[i] * lag_log[i];
        }
        const float denom = fn * sx2 - sx * sx;
        if (std::abs(denom) > 1e-12f)
            slope = (fn * sxy - sx * sy) / denom;
    }

    return simd::clamp(slope, 0.0f, 1.0f);
}

// ── ADX (normalised 0→1) ──────────────────────────────────────────────────────
// ADX >  (≡ ADX >  on the 0-100 scale) → trending strength present.
//
// BUG-RE6 FIX: int period → std::size_t period.
//   Original guard `n < static_cast<std::size_t>(period + 2)` contained a
//   signed `period + 2` expression (UB for INT_MAX), -Wsign-compare on the
//   comparison, and `int i` loop variable mixed with size_t indexing.
//   All corrected: period is now size_t, guard is `n < period + 2`
//   (unsigned arithmetic, no overflow), loop is `std::size_t i`.
static float adx(const float* c, const float* h, const float* l,
                  std::size_t n, std::size_t period = 14) noexcept {  // BUG-RE6 FIX
    if (n < period + 2) [[unlikely]] return 0.0f;  // BUG-RE6 FIX: no cast needed

    const std::size_t beg = n - period - 2;
    const float* cc = c + beg;
    const float* ch = h + beg;
    const float* cl = l + beg;

    float sum_tr = 0.0f, sum_pdm = 0.0f, sum_ndm = 0.0f;
    // BUG-RE6 FIX: loop variable changed from `int i` to `std::size_t i`.
    for (std::size_t i = 1; i <= period; ++i) {
        const float tr  = std::max({ ch[i] - cl[i],
                                     std::abs(ch[i]  - cc[i-1]),
                                     std::abs(cl[i]  - cc[i-1]) });
        const float dmP = ch[i] - ch[i-1];
        const float dmN = cl[i-1] - cl[i];
        sum_tr  += tr;
        sum_pdm += (dmP > dmN && dmP > 0.0f) ? dmP : 0.0f;
        sum_ndm += (dmN > dmP && dmN > 0.0f) ? dmN : 0.0f;
    }
    const float atr_  = sum_tr / static_cast<float>(period) + 1e-10f;
    const float diP   = sum_pdm / atr_ * 100.0f;
    const float diN   = sum_ndm / atr_ * 100.0f;
    const float dx    = std::abs(diP - diN) / (diP + diN + 1e-10f) * 100.0f;
    return simd::clamp(dx / 100.0f, 0.0f, 1.0f);
}

// ── Regression slope (normalised) ────────────────────────────────────────────
// Returns slope / mean_price: price-normalised directional momentum.
// Range clamped to [, ] ( per bar momentum).
//
// BUG-RE6 FIX: int period → std::size_t period.
//   Same rationale as adx(). The signed `period` forced redundant casts
//   in the guard and generated -Wsign-compare on the loop counter.
static float regSlope(const float* closes, std::size_t n,
                       std::size_t period = 20) noexcept {  // BUG-RE6 FIX
    if (n < period) [[unlikely]] return 0.0f;
    const float* y = closes + (n - period);
    const float  m = simd::mean(y, period);
    if (m < 1e-10f) [[unlikely]] return 0.0f;

    float sx = 0.0f, sy = 0.0f, sxy = 0.0f, sx2 = 0.0f;
    const float fn = static_cast<float>(period);  // safe: period ≤ MAX_BARS ≤ 128
    // BUG-RE6 FIX: loop variable changed from `int i` to `std::size_t i`.
    for (std::size_t i = 0; i < period; ++i) {
        const float xi = static_cast<float>(i);
        sx  += xi;
        sy  += y[i];
        sxy += xi * y[i];
        sx2 += xi * xi;
    }
    const float denom = fn * sx2 - sx * sx;
    if (std::abs(denom) < 1e-12f) [[unlikely]] return 0.0f;
    const float slope = (fn * sxy - sx * sy) / denom;
    return simd::clamp(slope / m, -0.05f, 0.05f);
}

// ── Volatility regime ─────────────────────────────────────────────────────────
// Uses the distribution of absolute returns to place current vol in context.
// When sufficient history exists (nr ≥ 30), compares current std-dev of
// returns against the 25th and 75th percentile of absolute returns.
//
// BUG-RE3 FIX: replaced two-step copy+abs loop with simd::abs_diff() to
// write absolute values into scratch2 directly from scratch1.  The two
// percentile calls must remain in ascending order (p25 before p75) because
// nth_element partitions in-place; the p75 call operates on the upper
// partition left intact by the p25 call.  If the order were reversed,
// p25 would receive a partially-sorted array and could return a wrong value.
// The SIMD write also halves the memory bandwidth vs the copy+loop approach.
static VolRegime volState(const float* closes, std::size_t n) noexcept {
    if (n < ) [[unlikely]] return VolRegime::NORMAL;

    auto& sc = g_scratch;
    const std::size_t nr = n - 1;
    for (std::size_t i = 0; i < nr; ++i) {
        const float b  = closes[i];
        sc.scratch1[i] = (b > 1e-10f) ? (closes[i + 1] / b - 1.0f) : 0.0f;
    }
    const float v = simd::std_dev(sc.scratch1.data(), nr);

    if (nr >= ) {
        // BUG-RE3 FIX: write |scratch1| into scratch2 via single SIMD pass.
        // Original: std::copy() + in-place abs loop — two passes, no SIMD.
        // simd::abs_diff(src, dst, n, mean=0.0f) computes |src[i] - 0| = |src[i]|.
        simd::abs_diff(sc.scratch1.data(), sc.scratch2.data(), nr, 0.0f);

        // ORDERING REQUIREMENT: p25 MUST be called before p75.
        // nth_element partitions scratch2 in-place: elements below the pivot
        // are ≤ pivot; elements above are untouched but unordered.
        // p25 call leaves the upper 75% intact → p75 call finds the correct
        // 75th-percentile element in the untouched upper portion.
        // Reversing this order would give p25 a partially-sorted input → wrong result.
        const float p25 = simd::percentile(sc.scratch2.data(), nr, f);
        const float p75 = simd::percentile(sc.scratch2.data(), nr, f);

        if (v < p25)         return VolRegime::LOW;
        if (v > p75 * f)  return VolRegime::HIGH;
        return VolRegime::NORMAL;
    }

    // Fallback thresholds for short history (< 30 bars)
    if (v < f) return VolRegime::LOW;
    if (v > f)  return VolRegime::HIGH;
    return VolRegime::NORMAL;
}

// ── HMM: 3-State Viterbi Decoder with Online Baum-Welch M-step ───────────────
// States: 0=BEAR, 1=SIDEWAYS, 2=BULL
//
// Architecture:
//   Hidden states model the underlying market condition.
//   Observations are discrete symbols derived from each bar's return magnitude
//   and volatility level — 6 possible observation symbols total.
//
// FIX (XC-05): A and B are mutable (non-const), seeded from empirical priors.
// After every Viterbi decode, hmmMStep() re-estimates A and B from the current
// observation sequence using a hard-assignment (Viterbi path) approximation of
// the Baum-Welch M-step with exponential forgetting (λ = 0.02 per call ≈
// 50-call half-life ≈ ~50 minutes of data at one call/minute).

static constexpr int N_STATES = ;   // BEAR=0, SIDEWAYS=1, BULL=2
static constexpr int N_OBS    = ;  // Viterbi window (bars)
static constexpr int N_EMIT   = ;   // observation symbols

// Mutable A (transition) and B (emission) — updated online via hmmMStep().
// Seeded from crypto empirical priors.
static float g_A[N_STATES][N_STATES] = {
    {0.00f, 0.00f, 0.00f},   // BEAR  → BEAR / SIDEWAYS / BULL
    {0.00f, 0.00f, 0.00f},   // SIDEWAYS → ...
    {0.00f, 0.00f, 0.00f},   // BULL  → BEAR / SIDEWAYS / BULL
};
static float g_B[N_STATES][N_EMIT] = {
    //  sd_dn  wk_dn  fl_lv  fl_hv  wk_up  sd_up
    {0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f},  // BEAR
    {0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f},  // SIDEWAYS (symmetric/flat)
    {0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f},  // BULL
};
// Single spinlock guards both g_A and g_B reads and writes.
// g_A and g_B are GLOBAL (shared across all symbols — one Markov model for
// all crypto pairs).  A per-symbol lock array (as used for GARCH) would be
// incorrect here: the entire point of the M-step is to learn a single shared
// transition model from all symbols' paths.
// Held briefly: one memcpy on read (~72 bytes), one blend+normalise on write
// (~27 FMA operations) — well under 1 µs on any modern CPU.
static std::atomic_flag g_hmm_lock = ATOMIC_FLAG_INIT;

// ── Baum-Welch online M-step ──────────────────────────────────────────────────
// Hard-assignment (Viterbi path) approximation for speed.
// λ: exponential forgetting factor per call window.
// BR-09 FIX: re-normalise each row after blending to prevent floating-point
// drift of row sums over thousands of calls. A convex blend of two valid
// probability distributions is still valid, but FP rounding accumulates;
// after ~10,000 calls (≈ a week of continuous trading), row sums can drift
// by ±0.01, making the Viterbi forward pass numerically invalid.
static void hmmMStep(const int* state_seq, const int* obs_seq,
                      std::size_t T, float lambda) noexcept {
    // Accumulate hard transition and emission counts from this window.
    float cnt_A[N_STATES][N_STATES] = {};
    float cnt_B[N_STATES][N_EMIT]   = {};
    for (std::size_t t = 0; t < T; ++t) {
        const int s = state_seq[t];
        if (s < 0 || s >= N_STATES) continue;   // defensive: guard corrupt seq
        const int o = obs_seq[t];
        if (o >= 0 && o < N_EMIT) cnt_B[s][o] += 1.0f;
        if (t + 1 < T) {
            const int ns = state_seq[t + 1];
            if (ns >= 0 && ns < N_STATES) cnt_A[s][ns] += 1.0f;
        }
    }

    // BUG-2 FIX: bounded spinlock with yield after 1000 failed attempts.
    // An unbounded while(test_and_set){} can cause live-lock under heavy
    // contention (20 symbols × 4 threads, all calling hmmViterbi near-simultaneously).
    for (int sp = 0; g_hmm_lock.test_and_set(std::memory_order_acquire); ++sp)
        if (sp > 1000) std::this_thread::yield();

    for (int s = 0; s < N_STATES; ++s) {
        // Transition row blend + re-normalise.
        float row_sum = 0.0f;
        for (int d = 0; d < N_STATES; ++d) row_sum += cnt_A[s][d];
        if (row_sum > 0.5f) {
            for (int d = 0; d < N_STATES; ++d)
                g_A[s][d] = (1.0f - lambda) * g_A[s][d]
                           + lambda * (cnt_A[s][d] / row_sum);
            // BR-09 FIX: re-normalise blended row.
            float new_row = 0.0f;
            for (int d = 0; d < N_STATES; ++d) new_row += g_A[s][d];
            if (new_row > 1e-10f)
                for (int d = 0; d < N_STATES; ++d) g_A[s][d] /= new_row;
        }

        // Emission row blend + re-normalise.
        float emit_sum = 0.0f;
        for (int o = 0; o < N_EMIT; ++o) emit_sum += cnt_B[s][o];
        if (emit_sum > 0.5f) {
            for (int o = 0; o < N_EMIT; ++o)
                g_B[s][o] = (1.0f - lambda) * g_B[s][o]
                           + lambda * (cnt_B[s][o] / emit_sum);
            // BR-09 FIX: re-normalise blended row.
            float new_emit = 0.0f;
            for (int o = 0; o < N_EMIT; ++o) new_emit += g_B[s][o];
            if (new_emit > 1e-10f)
                for (int o = 0; o < N_EMIT; ++o) g_B[s][o] /= new_emit;
        }
    }

    g_hmm_lock.clear(std::memory_order_release);
}

// ── Viterbi decoder ───────────────────────────────────────────────────────────
// BUG-RE2 FIX: merged the two forward passes into one.
//
// The original code ran two separate forward passes:
//   Pass 1 (delta-only): computed final delta to find best_state.
//   Pass 2 (psi-forward): computed psi[t][s] for the backtrace.
// Both passes are deterministic on the same inputs (A_snap, B_snap, obs_seq).
// best_state can be extracted from the SECOND pass's final delta, making the
// first pass completely redundant — 50% of all log() calls wasted.
//
static uint8_t hmmViterbi(const float* C, std::size_t n) noexcept {
    if (n < 6) return 1;  // cold-start default: SIDEWAYS

    // Initial state distribution — strong SIDEWAYS prior.
    static constexpr float PI[N_STATES] = {0.00f, 0.00f, 0.00f};

    // Build discretised observation sequence from bar returns.
    const std::size_t use   = std::min(n - 1, static_cast<std::size_t>(N_OBS));
    const std::size_t start = n - 1 - use;

    static thread_local std::array<float, N_OBS> ret_buf;
    float mean_ret = 0.0f;
    for (std::size_t i = 0; i < use; ++i) {
        const float p = C[start + i];
        ret_buf[i]    = (p > f) ? (C[start + i + 1] - p) / p : 0.0f;
        mean_ret     += ret_buf[i];
    }
    mean_ret /= static_cast<float>(use);

    float var = 0.0f;
    for (std::size_t i = 0; i < use; ++i) {
        const float d = ret_buf[i] - mean_ret;
        var += d * d;
    }
    const float sigma = std::sqrt(var / static_cast<float>(use) + 1e-12f);

    // BUG-RE5 FIX: hi_vol now uses absolute sigma vs. a fixed per-bar threshold.
    //
    // ORIGINAL: `const bool hi_vol = std::abs(z) > 1.5f;`
    // The original hi_vol was z-score based.  z already divides by sigma,
    // so a flat return in a high-vol market has BOTH small z AND small sigma
    // → hi_vol = (|z| > 1.5) is false for flat returns regardless of market vol.
    // Result: obs=3 was never assigned — symbol 3 was structurally unreachable.
    //
    // FIX: hi_vol is now based on the window's actual sigma vs. a calibrated
    // threshold.  This is computed ONCE for the window and applied per-bar
    // below — it correctly distinguishes "calm flat" from "stressed flat".
    //
    // Threshold 0.005f = 0.5% per 1-min bar ≈ empirical ~75th percentile
    // for BTCUSDT/ETHUSDT.  Set once at design time; can be promoted to
    // AppConfig::hmm_vol_threshold if per-deployment tuning is required.
    const bool hi_vol = (sigma > 0.000f);  // BUG-RE5 FIX: sigma-based, not z-based

    static thread_local std::array<int, N_OBS> obs_seq;
    for (std::size_t i = 0; i < use; ++i) {
        const float z = ret_buf[i] / sigma;

        // BUG-RE5 FIX: 6-bin vocabulary with non-overlapping, exhaustive conditions.
        // BUG-RE7 FIX: removed dead `&& !hi_vol` from obs=1 condition.
        //
        if      (z < -1.0f)               obs_seq[i] = 0;  // strong down
        else if (z < -0.25f)              obs_seq[i] = ;  // weak down
        else if (std::abs(z) <= 0.25f)   obs_seq[i] = hi_vol ? 3 : 2; // flat stressed / flat calm
        else if (z < 1.0f)               obs_seq[i] = ;  // weak up
        else                              obs_seq[i] = ;  // strong up
    }

    // Snapshot A and B under lock for a consistent read.
    float A_snap[N_STATES][N_STATES];
    float B_snap[N_STATES][N_EMIT];
    for (int sp = 0; g_hmm_lock.test_and_set(std::memory_order_acquire); ++sp)
        if (sp > 1000) std::this_thread::yield();
    std::memcpy(A_snap, g_A, sizeof(g_A));
    std::memcpy(B_snap, g_B, sizeof(g_B));
    g_hmm_lock.clear(std::memory_order_release);

    // ── BUG-RE2 FIX: single merged forward pass ───────────────────────────────
    // Computes delta[t][s] (log-probability of best path ending in state s at t)
    // and psi[t][s] (argmax predecessor state) in one traversal.
    // Previous code ran two separate identical loops — 100% redundant work.
    static thread_local std::array<std::array<int,   N_STATES>, N_OBS> psi;
    static thread_local std::array<float, N_STATES> delta, delta_new;

    // t = 0: initialise from prior + first emission.
    for (int s = 0; s < N_STATES; ++s) {
        delta[s]  = std::log(PI[s] + 1e-30f)
                  + std::log(B_snap[s][obs_seq[0]] + 1e-30f);
        psi[0][s] = 0;  // no predecessor at t=0
    }

    // t = 1..use-1: forward recurrence, recording psi for backtrace.
    for (std::size_t t = 1; t < use; ++t) {
        const int ob = obs_seq[t];
        for (int s = 0; s < N_STATES; ++s) {
            float best_v = -1e30f; int best_p = 0;
            for (int p = 0; p < N_STATES; ++p) {
                const float v = delta[p] + std::log(A_snap[p][s] + 1e-30f);
                if (v > best_v) { best_v = v; best_p = p; }
            }
            delta_new[s] = best_v + std::log(B_snap[s][ob] + 1e-30f);
            psi[t][s]    = best_p;
        }
        delta = delta_new;
    }

    // Extract most-likely final state from the final delta.
    uint8_t best_state = 1;  // default SIDEWAYS
    float   best_val   = -1e30f;
    for (int s = 0; s < N_STATES; ++s) {
        if (delta[s] > best_val) { best_val = delta[s]; best_state = static_cast<uint8_t>(s); }
    }

    // Backtrace the optimal Viterbi path using psi.
    // psi[t][s] = argmax predecessor of state s at time t.
    static thread_local std::array<int, N_OBS> state_seq;
    state_seq[use - 1] = static_cast<int>(best_state);
    for (int t = static_cast<int>(use) - 2; t >= 0; --t)
        state_seq[t] = psi[static_cast<std::size_t>(t + 1)][state_seq[t + 1]];

    // Online M-step: adapt A and B from the current Viterbi path.
    // Forgetting factor λ=0.02 → ~50-call half-life ≈ ~50 min of 1-min data.
    hmmMStep(state_seq.data(), obs_seq.data(), use, 0.02f);

    return best_state;
}

// ── Public: classifyRegime ────────────────────────────────────────────────────
RegimeInfo classifyRegime(const Features& f) noexcept {
    const std::size_t idx = g_bars.find(f.symbol);
    RegimeInfo info;

    // BUG-RE1 + BUG-RE4 FIX: use bar_count() (acquires push_lock_) for BOTH
    // the guard check and the main cnt read.
    // Original code read g_bars.at(idx).count directly — a data race with
    // any concurrent push() on ARM VPS hardware.
    // A single bar_count() call provides a consistent, lock-protected value
    // for both purposes.
    if (idx >= MAX_SYMBOLS) [[unlikely]] return info;  // NOISE default

    const std::size_t cnt = g_bars.at(idx).bar_count();
    if (cnt < 20) [[unlikely]] return info;  // insufficient history → NOISE

    auto& bars = g_bars.at(idx);
    auto& sc   = g_scratch;

    // linearise_all() acquires push_lock_ internally — race-free.
    bars.linearise_all(sc.closes.data(), sc.highs.data(),
                       sc.lows.data(),   sc.volumes.data(),
                       sc.ob_imbalances.data(), cnt);

    const float* C = sc.closes.data();
    const float* H = sc.highs.data();
    const float* L = sc.lows.data();

    const float    hurst  = hurstRS(C, cnt);
    const float    adx_v  = adx(C, H, L, cnt);
    const float    slope  = regSlope(C, cnt);
    const VolRegime vol   = volState(C, cnt);

    // FIX (ISSUE-06): [[likely]] marks NOISE — the dominant crypto regime (~=%).
    // TRENDING and MEAN_REVERTING are rare → [[unlikely]].
    if (hurst > 0.00f && adx_v > 0.00f && std::abs(slope) > 0.0003f) [[unlikely]] {
        info.regime = Regime::TRENDING;
    } else if (hurst < 0.00f && adx_v < 0.00f) [[unlikely]] {
        info.regime = Regime::MEAN_REVERTING;
    } else [[likely]] {
        info.regime = Regime::NOISE;
    }

    // High volatility overrides TRENDING → NOISE.
    // Rationale: in high-vol environments, momentum signals are dominated by
    // noise and the Hurst exponent is unreliable — reversal risk is too high.
    if (vol == VolRegime::HIGH && info.regime == Regime::TRENDING) [[unlikely]]
        info.regime = Regime::NOISE;

    info.vol_regime = vol;
    // Round to 4 decimal places for stable JSON serialisation (avoids e.g.
    // 
    info.hurst     = std::round(hurst  * 10000.0f) / 10000.0f;
    info.adx       = std::round(adx_v  * 10000.0f) / 10000.0f;
    info.reg_slope = std::round(slope  * 1000000.0f) / 1000000.0f;

    // Step 2: HMM Viterbi — independent of Hurst/ADX.
    // Used by signal_engine for directional bias: BULL → prefer LONG entries,
    // BEAR → prefer SHORT entries, SIDEWAYS → require higher confidence.
    info.hmm_state = hmmViterbi(C, cnt);

    return info;
}

} // namespace tqc
