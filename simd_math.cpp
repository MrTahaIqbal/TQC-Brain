/*
 * simd_math.cpp  -  TQC Brain | Taha Iqbal
 * // Note I have mentioned all the bugs that came in production so a user can understand that
 * Implements the out-of-line percentile() declared in simd_math.hpp.
 * All other functions in simd_math.hpp are inline — this file is minimal.
 *
 * FIX (ISSUE-16): The original implementation used std::vector<float> for
 *   the working copy, which calls std::malloc internally.  std::malloc can
 *   invoke the OS allocator and block the calling thread for microseconds,
 *   making the noexcept declaration a lie — any OOM condition would call
 *   std::terminate() with no recovery path and no useful error message.
 *
 *   Fix: replaced with a fixed-size static thread_local array [MAX_N = 512].
 *   - No heap allocation — truly noexcept.
 *   - Allocated once per thread at first call, reused for all subsequent
 *     calls on that thread.  Zero per-call stack or heap cost.
 *   - All call sites in this codebase guarantee n ≤ 512:
 *       risk_engine::histVaR()   → n = return buffer (≤ 512)
 *       LatencyTracker::stats()  → n = sample window (≤ 500)
 *   - Violations are caught at function entry (see BUG-SC2 FIX below).
 *
 * NON-REENTRANCY NOTE:
 *   The thread_local buffer is shared across all calls on the same thread.
 *   Do NOT call percentile() recursively or from a signal handler on the
 *   same thread — the buffer would be in an indeterminate state mid-sort.
 *
 * WHY NOT INLINE:
 *   nth_element requires a mutable copy of the data (it partitions in-place).
 *   n is runtime-variable.  A static thread_local array must have compile-time
 *   size (MAX_N = 512 = 2 KiB), which is too large to duplicate at every
 *   inline call site.  Out-of-line: the 2 KiB lives once per thread.
 *
 * ── BUGS FIXED IN THIS VERSION ──────────────────────────────────────────────
 *
 * BUG-SC1  percentile() — p out of range causes undefined behaviour.
 *          If the caller passes p < 0:
 *            idx_f = (negative / 100) * (n-1)  — negative float
 *            lo = static_cast<size_t>(floor(negative_float))
 *          In C++17 and earlier, converting a negative float to size_t is
 *          implementation-defined.  On x86 with the default FPU state it
 *          wraps to a very large value (~2^64), so:
 *            std::nth_element(tmp, tmp + huge_offset, tmp + n)
 *          is called with nth far outside [first, last) — undefined behaviour,
 *          typically a segfault or silent heap corruption.
 *          The lo/hi >= n clamps at lines 63-64 DO NOT catch this because
 *          static_cast<size_t>(negative) wraps to a value much larger than n,
 *          which is then clamped to n-1 — producing a 99th-percentile result
 *          for any negative p, with no error.
 *
 *          Similarly, p > 100 produces idx_f > n-1, which the clamps catch
 *          correctly — but clamping to n-1 silently promotes any p > 100
 *          to the 100th percentile with no diagnostic.
 *
 *          FIX: clamp p to [0.0f, 100.0f] at function entry before any
 *          arithmetic.  Out-of-range p is now defined behaviour (clamped),
 *          and the result is the nearest valid percentile.
 *
 * BUG-SC2  [NEW] percentile() silently truncated n > MAX_N with no diagnostic.
 *
 *          The original guard:
 *              if (n > MAX_N) n = MAX_N;
 *          silently computed the percentile of the first 512 elements instead
 *          of the full array.  The comment acknowledged this was a "safety
 *          guard" that "should never trigger", but provided no mechanism to
 *          detect or report a violation.
 *
 *          Any future call site that passes n = 600 (e.g., risk_engine
 *          expanding its return history to 600 bars) receives a quietly
 *          incorrect answer — the 95th percentile of a 512-element prefix,
 *          not the 95th percentile of the full 600-value distribution.
 *
 *          This is the most dangerous class of bug: numerically plausible
 *          but systematically wrong output with no error signal.  For
 *          histVaR(), a low-biased 95th-percentile loss estimate means the
 *          risk engine under-sizes stop-losses, producing positions that
 *          lose more than the model predicts under tail events.
 *
 *          FIX:
 *          - In debug builds (NDEBUG not defined): assert(n <= MAX_N) fires
 *            immediately at the call site with a clear message, stopping the
 *            test run at the exact violating call.
 *          - In release builds: a single stderr warning is printed and n is
 *            clamped (preserving the previous behaviour for resilience in
 *            live trading) — but the warning is unconditional and will appear
 *            in every log rotation until the call site is fixed.
 *          - MAX_N is exposed as a public constant in the header comment so
 *            future call sites can static_assert their buffer sizes.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include "simd_math.hpp"

#include <algorithm>
#include <cmath>
#include <cassert>
#include <cstdio>   // std::fprintf for release-build warning

namespace tqc::simd {

// MAX_N must match the value documented in simd_math.hpp's percentile() comment.
// All call sites in this codebase are required to guarantee n <= MAX_N.
// If a new call site cannot guarantee this, increase MAX_N and recompile.
static constexpr std::size_t MAX_N = 512;

float percentile(const float* data, std::size_t n, float p) noexcept {
    if (n == 0) [[unlikely]] return 0.0f;
    if (n == 1) [[unlikely]] return data[0];

    // BUG-SC1 FIX: clamp p before any arithmetic.
    // Without this, p < 0 produces idx_f < 0, and
    // static_cast<size_t>(floor(negative)) wraps to a huge offset →
    // nth_element with nth outside [first, last) → undefined behaviour.
    if (p < 0.0f)   p = 0.0f;
    if (p > 100.0f) p = 100.0f;

    // BUG-SC2 FIX: detect n > MAX_N violations at the call site rather than
    // silently producing a wrong answer.
    //
    // Debug builds: assert fires immediately, stopping the test run here with
    // a clear message.  No silent truncation ever occurs in testing.
    //
    // Release builds: print an unconditional stderr warning (visible in all
    // log outputs) and clamp n so the process remains alive in live trading.
    // The warning will appear every call cycle until the call site is fixed,
    // making it impossible to overlook.
    if (n > MAX_N) [[unlikely]] {
#ifdef NDEBUG
        std::fprintf(stderr,
            "[SIMD] WARNING: percentile() called with n=%zu > MAX_N=%zu.\n"
            "       Result is the percentile of the first %zu elements ONLY.\n"
            "       This is NUMERICALLY INCORRECT for the full %zu-element "
            "distribution.\n"
            "       Increase MAX_N in simd_math.cpp and recompile to fix.\n",
            n, MAX_N, MAX_N, n);
        n = MAX_N;   // clamp for resilience — process stays alive
#else
        // Debug: fire immediately so the test run stops here, not at a
        // downstream NaN or wrong VaR value that is hard to trace back.
        assert(n <= MAX_N &&
               "percentile(): n exceeds MAX_N — result would be computed on "
               "a truncated prefix.  Increase MAX_N in simd_math.cpp.");
#endif
    }

    // Thread-local scratch buffer: allocated once per thread, reused for all
    // subsequent calls.  No heap allocation, no per-call stack cost, noexcept.
    static thread_local float tmp[MAX_N];
    for (std::size_t i = 0; i < n; ++i) tmp[i] = data[i];

    // Map percentile [0,100] to a fractional index in [0, n-1].
    const float     idx_f = (p / 100.0f) * static_cast<float>(n - 1);
    std::size_t     lo    = static_cast<std::size_t>(std::floor(idx_f));
    std::size_t     hi    = static_cast<std::size_t>(std::ceil(idx_f));

    // Defensive clamps — p is now guaranteed in [0,100] so these should not
    // fire, but they protect against floating-point rounding at p == 100.0f
    // producing idx_f = n-1+epsilon → lo = n (one past the last element).
    if (lo >= n) lo = n - 1;
    if (hi >= n) hi = n - 1;

    // nth_element partitions in O(n) average: elements before lo are ≤ tmp[lo],
    // elements after lo are ≥ tmp[lo].  tmp[lo] is the exact lo-th order statistic.
    std::nth_element(tmp, tmp + lo, tmp + n);
    const float v_lo = tmp[lo];
    if (lo == hi) return v_lo;

    // Linear interpolation: partition the upper sub-range to find the hi-th
    // order statistic, then interpolate between lo and hi.
    std::nth_element(tmp + lo + 1, tmp + hi, tmp + n);
    const float v_hi = tmp[hi];
    const float frac = idx_f - static_cast<float>(lo);
    return v_lo + frac * (v_hi - v_lo);
}

} // namespace tqc::simd
