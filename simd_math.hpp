#pragma once
/*
 * simd_math.hpp  -  BigBoyAgent TQC Brain | Taha Iqbal
 * ============================================================
 * HARDWARE REQUIREMENT: SIMD VECTORISATION
 *
 * Why SIMD matters:
 *   Scalar: 1 float/cycle.  AVX2: 8 floats/cycle.
 *   RSI on 120 bars: scalar = 120 ops, AVX2 = 15 ops. (~8× throughput)
 *
 * AVX2 register: __m256 = 8 × float32 (256 bits).
 *   _mm256_add_ps(a,b)     — a[i]+b[i] for i=0..7 in ONE instruction
 *   _mm256_fmadd_ps(a,b,c) — a[i]*b[i]+c[i]  (FMA: fused multiply-add)
 *   _mm256_max_ps(a,b)     — max(a[i],b[i])
 *
 * Scalar fallback: all functions compile without AVX2 (CI, ARM, VPS, etc.)
 *
 * EMA VARIANTS (ISSUE-10 FIX — two separate functions):
 *   ema()        — standard EMA  k = 2/(period+1)  → used by MACD
 *   wilder_ema() — Wilder's smoothing k = 1/period  → used by RSI, ATR
 *   These are NOT interchangeable.  Standard EMA for period=14:
 *     k = 0.1333.  Wilder's for period=14: k = 0.0714.
 *   Using standard EMA for RSI produces a 2–4 point divergence vs every
 *   major charting platform.  Always use wilder_ema() for RSI and ATR.
 *
 * EMA SEEDING NOTE:
 *   Both ema() and wilder_ema() seed with data[0] and apply the recursive
 *   formula from index 1.  This is the minimal-look-back convention: the
 *   caller is expected to pass a full window (≥ 3×period bars) so the seed
 *   bias decays to < 1% before the most-recent value is used for trading.
 *   If n < period, the result is mathematically valid but has a high seed
 *   bias — callers should gate on Features::data_points >= minimum_bars
 *   (enforced in feature_engine.cpp) before using EMA-derived indicators.
 *
 * ── BUGS FIXED IN THIS VERSION ──────────────────────────────────────────────
 *
 * BUG-SM1  wilder_ema() — division by zero when period == 0.
 *          k = 1.0f / static_cast<float>(0) = +inf (IEEE 754).
 *          Every subsequent fma produces NaN or inf in the output, which
 *          silently propagates through RSI, ATR, and Stochastic D, producing
 *          a NaN confidence score that compares false to every threshold —
 *          meaning the Brain emits HOLD for every symbol with period=0.
 *          No assertion or log entry was generated; the failure was invisible.
 *          FIX: early-return guard for period == 0.  Returns the last data
 *          point (data[n-1]) as the best available estimate, which is correct
 *          for a degenerate 1-bar window.
 *
 * BUG-SM2  ema() — degenerate multiplier when period == 0.
 *          k = 2.0f / (0 + 1) = 2.0f.  With k > 1.0f the EMA formula becomes
 *          v = data[i] * 2.0f + v * (1.0f - 2.0f) = 2*data[i] - v.
 *          This is an oscillating series, not a smoothing function.  For
 *          MACD the result would be a rapidly alternating pseudo-signal that
 *          fires BUY and SELL on consecutive bars regardless of trend.
 *          FIX: same early-return guard as BUG-SM1.
 *
 * BUG-SM3  [NEW] atr_avx() used `int n, int period` while every other function
 *          in this header uses `std::size_t` for array sizes.  Two concrete
 *          failure modes:
 *
 *          (a) Signed integer overflow in the guard condition.
 *              The guard `n < period + 1` has undefined behaviour when
 *              `period = INT_MAX` because `period + 1` wraps to `INT_MIN`
 *              (signed overflow is UB in C++17; optimising compilers are
 *              permitted to eliminate the branch entirely on the basis that
 *              UB cannot occur).  On GCC 12+ with -O2, this optimisation IS
 *              applied: the guard disappears and `off = n - period - 1` runs
 *              with a negative result, leading to a pointer-arithmetic OOB
 *              access several hundred megabytes before the arrays.
 *
 *          (b) Silent truncation at the call site.
 *              Callers hold bar counts as `std::size_t` (the type of
 *              `std::vector::size()`).  Passing a `size_t n` to an `int`
 *              parameter silently truncates any value > INT_MAX to a negative
 *              or wrong positive — no warning unless -Wsign-conversion is
 *              enabled.  With n now negative (from the caller's perspective)
 *              the guard `n < period + 1` would again fail for large periods.
 *
 *          FIX: both parameters changed to `std::size_t`.  The guard is
 *          updated from `period <= 0` (impossible for size_t) to
 *          `period == 0`.  The inner loop counter changed to `std::size_t`
 *          for consistency; `std::ptrdiff_t` is used for `off` since it
 *          represents a signed displacement into the array.  The guard
 *          now reads `n < period + 1 || period == 0` which is equivalent to
 *          the original intent with no overflow possible.
 *
 * BUG-SM4  [NEW] ema() and wilder_ema() used `int period` while `n` is
 *          `std::size_t`.  The degenerate guard `period <= 0` correctly catches
 *          negatives, but:
 *          (a) Any compiler with -Wsign-compare enabled warns on comparisons
 *              between `n` (size_t) and any expression involving `period`
 *              (int) — e.g. inside a caller that checks `n >= (size_t)period`.
 *          (b) `static_cast<float>(period + 1)` overflows to UB if the caller
 *              passes `period = INT_MAX`.  Unreachable in practice, but the
 *              UB is technically present and can be exploited by aggressive
 *              LTO passes.
 *          FIX: `period` changed to `std::size_t` in both functions.  Guard
 *          updated from `period <= 0` to `period == 0`.
 *          The cast `static_cast<float>(period + 1)` is now safe for all
 *          representable size_t values that fit in float (up to ~2^24 periods,
 *          i.e. any realistic EMA window).
 *
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <algorithm>
#include <limits>   // std::numeric_limits — used by vmax/vmin sentinel seeds
#include <span>

#if defined(__AVX2__)
  #include <immintrin.h>
  #define TQC_AVX2 1
#elif defined(__AVX__)
  #include <immintrin.h>
  #define TQC_AVX  1
#else
  #define TQC_SCALAR 1
#endif

namespace tqc::simd {

// ── Horizontal sum of 8 lanes in __m256 ──────────────────────────────────────
#ifdef TQC_AVX2
[[nodiscard]] inline float hsum_f32(__m256 v) noexcept {
    __m128 lo   = _mm256_castps256_ps128(v);
    __m128 hi   = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    lo = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, lo);
    lo   = _mm_add_ss(lo, shuf);
    return _mm_cvtss_f32(lo);
}
#endif

// ── Sum ───────────────────────────────────────────────────────────────────────
[[nodiscard]] inline float sum(const float* __restrict__ data,
                                std::size_t n) noexcept {
#ifdef TQC_AVX2
    __m256 acc = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8)
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(data + i));
    float total = hsum_f32(acc);
    for (; i < n; ++i) total += data[i];
    return total;
#else
    return std::accumulate(data, data + n, 0.0f);
#endif
}

// ── Mean ──────────────────────────────────────────────────────────────────────
[[nodiscard]] inline float mean(const float* data, std::size_t n) noexcept {
    if (n == 0) [[unlikely]] return 0.0f;
    return sum(data, n) / static_cast<float>(n);
}

// ── Variance (population: divides by n, not n-1) ─────────────────────────────
// NOTE: population variance (÷n) is used throughout this system.  For
// indicators that feed into signal scoring (Bollinger bands, z-score,
// volatility normalisation), the distinction between population and sample
// variance matters only when n is small (< ~30 bars).  The minimum_bars
// gate in feature_engine.cpp ensures n ≥ 60 before any indicator is used
// for trading decisions, making the ~1/n vs 1/(n-1) difference < 2%.
// Callers that require unbiased sample variance must divide the result
// of variance() by n and multiply by n-1 themselves.
[[nodiscard]] inline float variance(const float* __restrict__ data,
                                     std::size_t n) noexcept {
    if (n < 2) [[unlikely]] return 0.0f;
    const float m = mean(data, n);

#ifdef TQC_AVX2
    __m256 mean_v = _mm256_set1_ps(m);
    __m256 sq_acc = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x  = _mm256_loadu_ps(data + i);
        __m256 d  = _mm256_sub_ps(x, mean_v);
        sq_acc    = _mm256_fmadd_ps(d, d, sq_acc);
    }
    float sq_sum = hsum_f32(sq_acc);
    for (; i < n; ++i) { float d = data[i] - m; sq_sum += d * d; }
    return sq_sum / static_cast<float>(n);
#else
    float sq_sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        float d = data[i] - m;
        sq_sum += d * d;
    }
    return sq_sum / static_cast<float>(n);
#endif
}

[[nodiscard]] inline float std_dev(const float* data, std::size_t n) noexcept {
    return std::sqrt(variance(data, n));
}

// ── Max of N floats ───────────────────────────────────────────────────────────
// Returns 0.0f for empty input. Callers should guard n > 0 if 0 is ambiguous.
[[nodiscard]] inline float vmax(const float* __restrict__ data,
                                 std::size_t n) noexcept {
    if (n == 0) [[unlikely]] return 0.0f;
#ifdef TQC_AVX2
    __m256 acc = _mm256_set1_ps(std::numeric_limits<float>::lowest());
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8)
        acc = _mm256_max_ps(acc, _mm256_loadu_ps(data + i));
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_max_ps(lo, hi);
    lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_max_ps(lo, _mm_movehdup_ps(lo));
    float mx = _mm_cvtss_f32(lo);
    for (; i < n; ++i) mx = std::max(mx, data[i]);
    return mx;
#else
    return *std::max_element(data, data + n);
#endif
}

// ── Min of N floats ───────────────────────────────────────────────────────────
// Returns 0.0f for empty input. Callers should guard n > 0 if 0 is ambiguous.
[[nodiscard]] inline float vmin(const float* __restrict__ data,
                                 std::size_t n) noexcept {
    if (n == 0) [[unlikely]] return 0.0f;
#ifdef TQC_AVX2
    __m256 acc = _mm256_set1_ps(std::numeric_limits<float>::max());
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8)
        acc = _mm256_min_ps(acc, _mm256_loadu_ps(data + i));
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_min_ps(lo, hi);
    lo = _mm_min_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_min_ps(lo, _mm_movehdup_ps(lo));
    float mn = _mm_cvtss_f32(lo);
    for (; i < n; ++i) mn = std::min(mn, data[i]);
    return mn;
#else
    return *std::min_element(data, data + n);
#endif
}

// ── ATR (True Range average) — vectorised ────────────────────────────────────
//
// FIX (ATR-OOB): Added explicit `n` parameter (total array length).
// The function computes ATR over the last `period` bars within an array of
// `n` elements, starting at offset (n - period - 1) so the maximum index
// accessed is h[n-1] — always within bounds.
//
// REQUIRES: n >= period + 1  (need period true-range bars → period+1 prices)
//           period > 0
//
// BUG-SM3 FIX: changed `int n, int period` → `std::size_t n, std::size_t period`.
//
// Callers pass bar counts as std::size_t (size of std::vector / std::array).
// The original `int` parameters required an implicit narrowing conversion that
// silently truncates values > INT_MAX and triggers -Wsign-conversion warnings.
// More critically, the guard `n < period + 1` with `int period = INT_MAX`
// caused signed-integer overflow (UB), which GCC 12+ exploits to eliminate
// the guard entirely at -O2.
//
// `off` is declared as `std::ptrdiff_t` (signed) because it represents an
// array displacement from the start of the input.  The guard ensures n >=
// period + 1, so off = n - period - 1 >= 0 always holds after the guard.
// The cast to ptrdiff_t is safe since both n and period fit in ptrdiff_t
// for any realistic bar count (< 2^62 bars).
[[nodiscard]] inline float atr_avx(const float* __restrict__ h,
                                    const float* __restrict__ l,
                                    const float* __restrict__ c,
                                    std::size_t period,    // BUG-SM3 FIX: was int
                                    std::size_t n          // BUG-SM3 FIX: was int
                                    ) noexcept {
    // BUG-SM3 FIX: guard updated from `period <= 0` (impossible for size_t)
    // to `period == 0`.  The first condition (`n < period + 1`) is safe from
    // overflow because both operands are now unsigned (size_t arithmetic wraps
    // predictably and the compiler does not treat wrap as UB for unsigned types).
    if (period == 0 || n < period + 1) [[unlikely]] return 0.0f;

    // off >= 0 guaranteed by the guard above.
    const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(n - period - 1);
    const float* H = h + off;
    const float* L = l + off;
    const float* C = c + off;

#ifdef TQC_AVX2
    __m256 tr_sum = _mm256_setzero_ps();
    std::size_t i = 0;                      // BUG-SM3 FIX: was int
    for (; i + 8 <= period; i += 8) {
        __m256 hi_v = _mm256_loadu_ps(H + i + 1);
        __m256 lo_v = _mm256_loadu_ps(L + i + 1);
        __m256 pc   = _mm256_loadu_ps(C + i);
        __m256 hl   = _mm256_sub_ps(hi_v, lo_v);
        // abs(high - prev_close) and abs(low - prev_close) via sign-bit clear
        const __m256 sign_bit = _mm256_set1_ps(-0.0f);
        __m256 hpc  = _mm256_andnot_ps(sign_bit, _mm256_sub_ps(hi_v, pc));
        __m256 lpc  = _mm256_andnot_ps(sign_bit, _mm256_sub_ps(lo_v, pc));
        __m256 tr   = _mm256_max_ps(hl, _mm256_max_ps(hpc, lpc));
        tr_sum      = _mm256_add_ps(tr_sum, tr);
    }
    float total = hsum_f32(tr_sum);
    for (; i < period; ++i) {
        float hl  = H[i+1] - L[i+1];
        float hpc = std::abs(H[i+1] - C[i]);
        float lpc = std::abs(L[i+1] - C[i]);
        total += std::max({hl, hpc, lpc});
    }
    return total / static_cast<float>(period);
#else
    float total = 0.0f;
    for (std::size_t i = 0; i < period; ++i) {   // BUG-SM3 FIX: was int
        float hl  = H[i+1] - L[i+1];
        float hpc = std::abs(H[i+1] - C[i]);
        float lpc = std::abs(L[i+1] - C[i]);
        total += std::max({hl, hpc, lpc});
    }
    return total / static_cast<float>(period);
#endif
}

// ── Dot product: sum(a[i] * b[i]) ────────────────────────────────────────────
[[nodiscard]] inline float dot(const float* __restrict__ a,
                                const float* __restrict__ b,
                                std::size_t n) noexcept {
#ifdef TQC_AVX2
    __m256 acc = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),
                              _mm256_loadu_ps(b + i), acc);
    float total = hsum_f32(acc);
    for (; i < n; ++i) total += a[i] * b[i];
    return total;
#else
    float total = 0.0f;
    for (std::size_t i = 0; i < n; ++i) total += a[i] * b[i];
    return total;
#endif
}

// ── Absolute differences |src[i] - m| → dst[i] ───────────────────────────────
inline void abs_diff(const float* __restrict__ src,
                           float* __restrict__ dst,
                     std::size_t n, float m) noexcept {
#ifdef TQC_AVX2
    __m256 mean_v   = _mm256_set1_ps(m);
    __m256 sign_bit = _mm256_set1_ps(-0.0f);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x  = _mm256_loadu_ps(src + i);
        __m256 d  = _mm256_sub_ps(x, mean_v);
        __m256 ab = _mm256_andnot_ps(sign_bit, d);
        _mm256_storeu_ps(dst + i, ab);
    }
    for (; i < n; ++i) dst[i] = std::abs(src[i] - m);
#else
    for (std::size_t i = 0; i < n; ++i) dst[i] = std::abs(src[i] - m);
#endif
}

// ── Standard EMA — k = 2/(period+1) — for MACD ───────────────────────────────
// Seeds with data[0]; applies recursive EMA from index 1.
// Caller should pass at least 3×period bars to allow seed bias to decay.
//
// BUG-SM2 / BUG-SM4 FIX: period changed from `int` to `std::size_t`.
//   Old `int period` caused:
//   - Degenerate case: k = 2/(0+1) = 2.0 → oscillating series (BUG-SM2).
//     Guard was `period <= 0`; now `period == 0` (size_t cannot be negative).
//   - Signed-integer overflow: `period + 1` UB for INT_MAX (BUG-SM4).
//   - -Wsign-compare warnings when n (size_t) and period (int) are compared.
//   For realistic period values (≤ 200 bars), `static_cast<float>(period + 1)`
//   is exact and within float's 24-bit mantissa.
[[nodiscard]] inline float ema(const float* data, std::size_t n,
                                std::size_t period) noexcept {   // BUG-SM4 FIX
    if (n == 0)      [[unlikely]] return 0.0f;
    if (period == 0) [[unlikely]] return data[n - 1];  // degenerate; BUG-SM2 FIX
    const float k = 2.0f / static_cast<float>(period + 1);
    float v = data[0];
    for (std::size_t i = 1; i < n; ++i)
        v = std::fma(data[i], k, v * (1.0f - k));
    return v;
}

// ── Wilder's Smoothing — k = 1/period — for RSI and ATR ──────────────────────
// DIFFERENT from standard EMA (k = 2/(period+1)).
// For period=14: Wilder k=0.0714 vs standard EMA k=0.1333.
// Using standard EMA for RSI causes ~2–4 point divergence vs every
// major charting platform.  Always use this function for RSI and ATR.
//
// BUG-SM1 / BUG-SM4 FIX: period changed from `int` to `std::size_t`.
//   Old `int period = 0`: k = 1/0 = +inf → NaN cascade through RSI (BUG-SM1).
//   Guard updated from `period <= 0` to `period == 0` (size_t is unsigned).
//   Signed-integer overflow hazard from `int period` also eliminated (BUG-SM4).
[[nodiscard]] inline float wilder_ema(const float* data, std::size_t n,
                                       std::size_t period) noexcept {  // BUG-SM4 FIX
    if (n == 0)      [[unlikely]] return 0.0f;
    if (period == 0) [[unlikely]] return data[n - 1];  // BUG-SM1 FIX
    const float k = 1.0f / static_cast<float>(period);
    float v = data[0];
    for (std::size_t i = 1; i < n; ++i)
        v = std::fma(data[i], k, v * (1.0f - k));
    return v;
}

// ── Clamp (branchless, using std::min/max) ────────────────────────────────────
[[nodiscard]] inline float clamp(float v, float lo, float hi) noexcept {
    return std::max(lo, std::min(hi, v));
}

// ── Percentile (nth_element, O(n) average) ────────────────────────────────────
// Declared here; defined in simd_math.cpp (out-of-line — uses a static
// thread_local scratch buffer to avoid per-call heap allocation).
// p must be in [0, 100]; values outside this range are clamped.
// Returns 0.0f for empty input.
// NOT re-entrant on the same thread (uses a shared thread_local buffer).
// PRECONDITION: n <= 512 (MAX_N in simd_math.cpp). Violating this triggers
//               an assert in debug builds and a stderr warning in release builds.
//               All call sites in this codebase guarantee n <= 512.
[[nodiscard]] float percentile(const float* data, std::size_t n, float p) noexcept;

} // namespace tqc::simd
