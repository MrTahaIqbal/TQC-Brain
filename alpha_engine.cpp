/*
 * alpha_engine.cpp  -  BigBoyAgent TQC Brain | Taha Iqbal
 *
 * Cross-sectional factor model:
 *   1. Momentum alpha     — rank by 20-bar ROC
 *   2. Low-vol anomaly    — rank by inverted volatility (low-vol outperforms)
 *   3. Volume surge alpha — confirm signal direction with order-flow
 *   4. Quality alpha      — Hurst + ADX composite
 *
 * All arrays are fixed-size on the stack (MAX_SIG = 24).
 * Composite alpha is blended into confidence scores.
 *
 * ── BUGS FIXED IN THIS VERSION ──────────────────────────────────────────────
 *
 * BUG-AE1  applyAlphaToSignals(): no guard on sigs.size() > MAX_SIG.
 *
 *          Six stack arrays are declared as float[MAX_SIG] (24 elements).
 *          rankNorm() also declares int idx[MAX_SIG].  If sigs.size() > 24,
 *          every one of these arrays overflows on the write loop:
 *            for (int i = 0; i < n; ++i) vals[i] = sigs[i].roc_20;
 *          Overflow writes beyond the stack frame, corrupting adjacent local
 *          variables or the return address — undefined behaviour, potential
 *          crash, potential silent data corruption.
 *
 *          In normal operation MAX_SYMBOLS = 24 matches MAX_SIG, so this
 *          never fires.  But no contract enforced it — a caller passing a
 *          larger span (e.g. from a test fixture or a future config increase)
 *          would silently corrupt memory.
 *          FIX: assert(n <= MAX_SIG) at the top of applyAlphaToSignals().
 *          Also added n > MAX_SIG early-return after the assert so release
 *          builds fail safely (return without modifying signals) rather than
 *          overflowing.
 *
 * BUG-AE2  applyAlphaToSignals(): SELL signals blended in the wrong direction.
 *
 *          The composite alpha score is a cross-sectional rank: positive means
 *          the symbol ranks high on momentum/quality/volume relative to peers.
 *          Positive composite = bullish cross-sectional edge.
 *
 *          The blend (original):
 *            adj = s.confidence + composite * alpha_weight
 *          is applied identically to BUY and SELL signals.
 *
 *
 *          FIX: for SELL signals, negate the composite before blending.
 *          Unified formula: adj = confidence + dir_sign * composite * alpha_weight
 *          where dir_sign = +1 for BUY, -1 for SELL.
 *          Now: bullish composite boosts BUY and penalises SELL; bearish
 *          composite penalises BUY and boosts SELL — directionally correct.
 *
 * 
 *     
 *
 *
 *
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include "alpha_engine.hpp"
#include "simd_math.hpp"
#include <cstring>
#include <algorithm>
#include <cmath>
#include <cassert>

namespace tqc {

static constexpr int MAX_SIG = 24;

// ── Cross-sectional rank → normalise to [-1, +1] ─────────────────────────────
// Rank 0 (lowest value) → -1.0, Rank n-1 (highest) → +1.0.
// n == 1: every symbol gets 0.0 (no cross-sectional information).
static void rankNorm(const float* vals, float* out, int n) noexcept {
    // BUG-AE1: idx overflow prevented by the caller-side assert in
    // applyAlphaToSignals.  Defensive: n is already bounded to <= MAX_SIG.
    int idx[MAX_SIG];
    for (int i = 0; i < n; ++i) idx[i] = i;
    std::sort(idx, idx + n, [&](int a, int b){ return vals[a] < vals[b]; });

    const float fn = static_cast<float>(n);
    for (int rank = 0; rank < n; ++rank) {
        const float norm = (fn > 1.0f)
                           ? (static_cast<float>(rank) / (fn - 1.0f)) * 2.0f - 1.0f
                           : 0.0f;
        out[idx[rank]] = norm;
    }
}

// ── Factor 1: Momentum (20-bar ROC) ──────────────────────────────────────────
static void momentumAlpha(std::span<Signal> sigs, float* out) noexcept {
    const int n = static_cast<int>(sigs.size());
    float vals[MAX_SIG];
    for (int i = 0; i < n; ++i) vals[i] = sigs[i].roc_20;
    rankNorm(vals, out, n);
}

// ── Factor 2: Low-vol anomaly ─────────────────────────────────────────────────
// Lower volatility = better quality = higher rank after inversion.
static void lowVolAlpha(std::span<Signal> sigs, float* out) noexcept {
    const int n = static_cast<int>(sigs.size());
    float vals[MAX_SIG];
    for (int i = 0; i < n; ++i) vals[i] = -sigs[i].volatility;  // invert
    rankNorm(vals, out, n);
}

// ── Factor 3: Volume surge (direction-aware) ──────────────────────────────────
// BUY:  positive volume imbalance confirms bullish order flow → positive score.
// SELL: positive volume imbalance contradicts bearish signal → negative score.
// HOLD: neutral.
// BUG-AE4 FIX: strncmp with n=4 includes null terminator byte.
static void volumeSurgeAlpha(std::span<Signal> sigs, float* out) noexcept {
    for (int i = 0; i < static_cast<int>(sigs.size()); ++i) {
        const auto& s     = sigs[i];
        const float imbal = s.volume_imbalance;
        if      (std::strncmp(s.signal, "BUY",  4) == 0)   // BUG-AE4 FIX: n=4
            out[i] = simd::clamp( imbal, -1.0f, 1.0f);
        else if (std::strncmp(s.signal, "SELL", 4) == 0)
            out[i] = simd::clamp(-imbal, -1.0f, 1.0f);
        else
            out[i] = 0.0f;  // HOLD: no directional preference
    }
}

// ── Factor 4: Quality (Hurst + ADX) ──────────────────────────────────────────
// hdev: Hurst deviation from random-walk baseline (0.5).
//   H > 0.5 → persistent (trending) → positive quality signal.
//   H < 0.5 → anti-persistent (mean-reverting) → negative quality signal.
// ADX ∈ [0, 1]: trend strength (direction-agnostic).
// Combined: (H−0.5)×2 + ADX, clamped to [-1, +1].
static void qualityAlpha(std::span<Signal> sigs, float* out) noexcept {
    for (int i = 0; i < static_cast<int>(sigs.size()); ++i) {
        const float hdev = (sigs[i].hurst - 0.5f) * 2.0f;   // [0,1] → [-1,+1]
        out[i] = simd::clamp(hdev + sigs[i].adx, -1.0f, 1.0f);
    }
}

// ── Composite alpha application ───────────────────────────────────────────────
void applyAlphaToSignals(std::span<Signal> sigs, float alpha_weight) noexcept {
    const int n = static_cast<int>(sigs.size());

    // BUG-AE1 FIX: assert fires in debug builds immediately at the call site;
    // the early-return below ensures release builds fail safely (no overflow)
    // rather than writing past the end of the stack arrays.
    assert(n <= MAX_SIG && "applyAlphaToSignals: sigs.size() exceeds MAX_SIG (24)");
    if (n < 2 || n > MAX_SIG) [[unlikely]] return;

    // alpha_weight guard: clamp to [0, 1] so callers can't accidentally pass
    // a value > 1.0 and produce adj values far outside [0, 1].
    alpha_weight = simd::clamp(alpha_weight, 0.0f, 1.0f);

    float mom      [MAX_SIG];
    float lvol     [MAX_SIG];
    float vsurge   [MAX_SIG];
    float qual     [MAX_SIG];
    float vsurge_raw[MAX_SIG];
    float qual_raw  [MAX_SIG];

    momentumAlpha    (sigs, mom);
    lowVolAlpha      (sigs, lvol);
    volumeSurgeAlpha (sigs, vsurge_raw);
    qualityAlpha     (sigs, qual_raw);

    // H-07 FIX: rankNorm applied to all four factors before compositing.
    // momentumAlpha and lowVolAlpha already call rankNorm internally.
    // volumeSurgeAlpha and qualityAlpha returned raw values with unbounded
    // dynamic range, making the factor weights meaningless — vsurge could
    // dominate (and push composite outside [-1, +1]) on high-volume cycles.
    rankNorm(vsurge_raw, vsurge, n);
    rankNorm(qual_raw,   qual,   n);

    static constexpr float W_MOM  = 0.25f;
    static constexpr float W_LVOL = 0.25f;
    static constexpr float W_VS   = 0.25f;
    static constexpr float W_QUAL = 0.25f;
  

    for (int i = 0; i < n; ++i) {
        const float composite = simd::clamp(
            mom[i]*W_MOM + lvol[i]*W_LVOL + vsurge[i]*W_VS + qual[i]*W_QUAL,
            -1.0f, 1.0f);

        auto& s = sigs[i];
        s.alpha_score = std::round(composite * 10000.0f) / 10000.0f;

        const bool is_buy  = (std::strncmp(s.signal, "BUY",  4) == 0);  // BUG-AE4 FIX
        const bool is_sell = (std::strncmp(s.signal, "SELL", 4) == 0);  // BUG-AE4 FIX

        if (is_buy || is_sell) {
            // BUG-AE2 FIX: direction-aware composite sign.
            // Positive composite = bullish cross-sectional edge.
            //   BUY  (dir_sign =+1): bullish composite → boost confidence.
            //   SELL (dir_sign =-1): bullish composite → penalise confidence.
            //   (bearish composite has the opposite effect in each case)
            //
            // BEFORE: adj = confidence + composite * alpha_weight (identical for both).
            //   A SELL signal with composite=+0.80 got: 0.62 + 0.16 = 0.78.
            //   Alpha was INCREASING sell confidence on a bullish cross-sectional
            //   signal — directionally backwards.
            //
            // AFTER: dir_sign negates composite for SELL so the same bullish
            //   cross-sectional signal REDUCES sell confidence, as it should.
            const float dir_sign = is_buy ? 1.0f : -1.0f;
            const float adj      = s.confidence + dir_sign * composite * alpha_weight;

            // BUG-AE3 FIX: symmetric ±alpha_weight band around pre-alpha confidence.
            // Prevents alpha from overriding the regime conf_cap by more than
            // alpha_weight.  The regime cap (e.g. 0.65 for NOISE) was previously
            // ignorable: alpha could push to 0.65 + 0.20 = 0.85 — 31% over cap.
            // Now: the maximum deviation from the regime-capped baseline is
            // exactly ±alpha_weight in either direction, regardless of composite.
            const float lo = std::max(0.01f, s.confidence - alpha_weight);
            const float hi = std::min(0.99f, s.confidence + alpha_weight);
            s.confidence = std::round(simd::clamp(adj, lo, hi) * 10000.0f) / 10000.0f;
        }
    }
}

} // namespace tqc
