#pragma once
/*
 * bar_store.hpp  -  TQC Brain | Taha Iqbal
 * ============================================================
 * DATA-ORIENTED DESIGN: Struct-of-Arrays (SoA) layout.
 *
 *   AoS (what beginners write):
 *     struct Bar { float open, high, low, close, volume; };
 *     Bar bars[128];
 *     // RSI touches close[i] for all i — but the CPU loads
 *     // [open,high,low,CLOSE,vol] each iteration → 80% wasted bandwidth.
 *
 *   SoA (what quant desks write):
 *     float closes[128];   // [c0,c1,...,c127] — contiguous
 *     float highs[128];
 *     // RSI: loads closes[0..127] in 8 cache lines → pre-fetcher fires.
 *
 * BarStore uses fixed-size std::array so:
 *   1. No heap allocation (pool+stack allocator requirement)
 *   2. Contiguous memory (SIMD requirement)
 *   3. Compile-time stride → compiler auto-vectorises indicator loops
 *
 * ── BUGS FIXED IN THIS VERSION ──────────────────────────────────────────────
 *
 * BUG-BS1  SymbolBars::bulk_load() — no lock held during write.
 *          The HTTP handler calls bulk_load() on first request for a symbol.
 *          A second concurrent request for the same symbol (from a different
 *          worker thread) could call push() simultaneously.  push() acquires
 *          push_lock_; bulk_load() did not — giving it unsynchronised access
 *          to the same arrays, head, and count.  This is a data race:
 *          undefined behaviour under the C++ memory model, and manifests as
 *          a garbled price history on any machine with weak cache coherency
 *          (ARM VPS, any multi-socket x86).
 *          FIX: bulk_load() acquires push_lock_ for the duration of the write,
 *          the same spinlock that push() uses.
 *
 * BUG-BS2  SymbolBars::linearise_closes() and linearise_all() — no lock held
 *          during reads.
 *          The feature engine calls linearise_all() on every /predict request
 *          to copy the ring buffer into flat scratch arrays for SIMD processing.
 *          Meanwhile, the HTTP handler calls push() to append new bars.
 *          Without synchronisation, the read of `head`, `count`, and the five
 *          float arrays in linearise_all() can race with a concurrent push().
 *
 *          On x86, naturally-aligned 4-byte stores are torn-free at the hardware
 *          level — so individual float values won't be half-old/half-new.  But:
 *            (a) The C++ memory model does NOT guarantee this; it is implementation-
 *                specific.  Any ARM-based VPS (AWS Graviton, Ampere Altra) makes
 *                this race observable with standard C++ tools.
 *            (b) Even on x86, the compound read of head + count + 5 arrays is not
 *                atomic.  A push() between the `head` read and the array read
 *                causes linearise to see inconsistent state: correct head but stale
 *                array contents or vice-versa.
 *          FIX: linearise_closes() and linearise_all() acquire push_lock_ for the
 *          duration of the read.  Lock hold time is small (copy 5 × 128 floats =
 *          2560 bytes, ~100 ns on L1-warm data), well within the acceptable window.
 *          push_lock_ is mutable so const member functions can acquire it.
 *
 * BUG-BS3  BarStore::at() — no bounds check.
 *          The hot-path accessor at(idx) returns bars_[idx] without checking
 *          that idx < MAX_SYMBOLS.  An off-by-one in get_or_create() returning
 *          MAX_SYMBOLS (the "not found" sentinel) that is then passed to at()
 *          would access bars_[24] — one slot past the end of the array — with
 *          no error.  On typical implementations this reads into the mtx_ padding
 *          bytes and returns a reference to the lock, which the caller then
 *          overwrites with a bar push.  The result is a corrupted mutex.
 *          FIX: assert(idx < MAX_SYMBOLS) in both const and non-const overloads.
 *          Debug builds catch this immediately; release builds have zero overhead.
 *
 * BUG-BS4  [NEW] [[likely]] annotation on equality match in get_or_create()
 *          and find() is semantically inverted.
 *
 *          Both methods perform a linear scan:
 *            for (std::size_t i = 0; i < symbol_count_; ++i)
 *                if (std::string_view(bars_[i].name) == sym) [[likely]]
 *                    return i;
 *
 *          The [[likely]] attribute on the if-body tells the CPU branch
 *          predictor "this condition is usually true."  During a linear scan
 *          of up to 24 symbol names, the condition is FALSE on every iteration
 *          except the single matching one.  Annotating the match as [[likely]]
 *          inverts the predictor: it predicts a branch to `return i` on every
 *          iteration, mispredicts on every non-matching slot, and correctly
 *          predicts only on the one slot that actually matches.
 *
 *          Net effect: maximum misprediction rate (up to 23 out of 24) in
 *          the common case where the symbol is near the end of the list.
 *          On a modern CPU, each branch misprediction costs ~15–20 cycles.
 *          For a 20-symbol scan with [[likely]] on the match, the worst case
 *          (symbol at slot 20) is 19 mispredictions = ~380 wasted cycles on
 *          every /predict request, every cycle, for every symbol.
 *
 *          FIX: removed [[likely]] from both equality-match branches in
 *          get_or_create() and find().  No hint is the correct choice for
 *          a linear search where the match position is not predictable.
 *          The CPU's own branch history hardware adapts to the actual
 *          access pattern far more accurately than a static hint that is
 *          wrong 95% of the time.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <array>
#include <cstring>
#include <cstddef>
#include <string_view>
#include <atomic>
#include <shared_mutex>
#include <mutex>        // std::unique_lock
#include <thread>       // std::this_thread::yield
#include <cassert>

namespace tqc {

// ── Constants ─────────────────────────────────────────────────────────────────
inline constexpr std::size_t MAX_BARS    = 128; // power-of-2 → branchless mask
inline constexpr std::size_t MAX_SYMBOLS = 24;
inline constexpr std::size_t SYM_NAME    = 16;

// ── Per-symbol SoA bar storage ────────────────────────────────────────────────
// Each array is 64-byte aligned → AVX2 aligned loads, no straddle penalty.
// Total per symbol: 5 × 128 × 4 bytes = 2.5 KiB → fits in L1 cache.
struct alignas(64) SymbolBars {
    alignas(64) std::array<float, MAX_BARS> closes      {};
    alignas(64) std::array<float, MAX_BARS> highs       {};
    alignas(64) std::array<float, MAX_BARS> lows        {};
    alignas(64) std::array<float, MAX_BARS> volumes     {};
    // Order-book bid/ask size imbalance per bar: (bid − ask) / (bid + ask) ∈ [-1,+1].
    // Stored as 0.0f when no L2 WebSocket data is available (neutral).
    alignas(64) std::array<float, MAX_BARS> ob_imbalances{};

    std::size_t head  = 0;   // ring buffer write pointer (always increases)
    std::size_t count = 0;   // valid bars clamped at MAX_BARS

    // Per-symbol spinlock.
    // Protects ALL reads and writes to head, count, and the five arrays.
    // Acquired exclusively by: push(), bulk_load().
    // Acquired exclusively by: linearise_closes(), linearise_all() — see BUG-BS2.
    // push_lock_ is mutable so const members (linearise_*) can acquire it.
    mutable std::atomic_flag push_lock_ = ATOMIC_FLAG_INIT;

    char name[SYM_NAME] = {};

    // ── Push one new bar ──────────────────────────────────────────────────────
    // O(1): ring pointer advance, no shift.
    // ob_imb: order-book imbalance for this bar; 0.0f when L2 data unavailable.
    void push(float c, float h, float l, float v, float ob_imb = 0.0f) noexcept {
        // Exponential backoff: spin briefly, then yield to avoid pegging a core.
        for (int spin = 0; push_lock_.test_and_set(std::memory_order_acquire); ++spin)
            if (spin > 64) std::this_thread::yield();

        const std::size_t slot = head & (MAX_BARS - 1);
        closes       [slot] = c;
        highs        [slot] = h;
        lows         [slot] = l;
        volumes      [slot] = v;
        ob_imbalances[slot] = ob_imb;
        ++head;
        if (count < MAX_BARS) ++count;

        push_lock_.clear(std::memory_order_release);
    }

    // ── Bulk-load historical data (cold-start warmup) ─────────────────────────
    // Replaces the entire ring buffer with the most-recent `n` bars.
    // Called once per symbol on the first /predict request.
    //
    // BUG-BS1 FIX: acquire push_lock_ before writing arrays/head/count.
    // Concurrent push() from a second worker thread for the same symbol
    // is now safely blocked until bulk_load() completes.
    void bulk_load(const float* c, const float* h,
                   const float* l, const float* v,
                   std::size_t n) noexcept {
        // BUG-BS1 FIX: lock before touching any shared state.
        for (int spin = 0; push_lock_.test_and_set(std::memory_order_acquire); ++spin)
            if (spin > 64) std::this_thread::yield();

        const std::size_t take = (n > MAX_BARS) ? MAX_BARS : n;
        const std::size_t skip = n - take;
        for (std::size_t i = 0; i < take; ++i) {
            closes       [i] = c[skip + i];
            highs        [i] = h ? h[skip + i] : c[skip + i];
            lows         [i] = l ? l[skip + i] : c[skip + i];
            volumes      [i] = v ? v[skip + i] : 0.0f;
            ob_imbalances[i] = 0.0f;  // historical data has no live OB imbalance
        }
        head  = take;
        count = take;

        push_lock_.clear(std::memory_order_release);
    }

    // ── Linearise most-recent n closes into out[] ─────────────────────────────
    // The ring buffer may wrap — copies to a contiguous scratch array.
    // BUG-BS2 FIX: acquire push_lock_ to prevent racing with a concurrent push().
    // Declared const; push_lock_ is mutable so this is valid.
    void linearise_closes(float* __restrict__ out, std::size_t n) const noexcept {
        // BUG-BS2 FIX: take lock before reading head/count/closes.
        for (int spin = 0; push_lock_.test_and_set(std::memory_order_acquire); ++spin)
            if (spin > 64) std::this_thread::yield();

        n = (n < count) ? n : count;
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t slot = (head - 1 - (n - 1 - i)) & (MAX_BARS - 1);
            out[i] = closes[slot];
        }

        push_lock_.clear(std::memory_order_release);
    }

    // ── Linearise most-recent n bars (all five arrays) into out* ─────────────
    // BUG-BS2 FIX: acquire push_lock_ for the full duration of the copy.
    void linearise_all(float* __restrict__ oc, float* __restrict__ oh,
                       float* __restrict__ ol, float* __restrict__ ov,
                       float* __restrict__ oob,
                       std::size_t n) const noexcept {
        // BUG-BS2 FIX: take lock before reading head/count/arrays.
        for (int spin = 0; push_lock_.test_and_set(std::memory_order_acquire); ++spin)
            if (spin > 64) std::this_thread::yield();

        n = (n < count) ? n : count;
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t slot = (head - 1 - (n - 1 - i)) & (MAX_BARS - 1);
            oc [i] = closes       [slot];
            oh [i] = highs        [slot];
            ol [i] = lows         [slot];
            ov [i] = volumes      [slot];
            oob[i] = ob_imbalances[slot];
        }

        push_lock_.clear(std::memory_order_release);
    }

    // ── Count of valid bars (advisory) ───────────────────────────────────────
    [[nodiscard]] std::size_t bar_count() const noexcept {
        for (int spin = 0; push_lock_.test_and_set(std::memory_order_acquire); ++spin)
            if (spin > 64) std::this_thread::yield();
        const std::size_t c = count;
        push_lock_.clear(std::memory_order_release);
        return c;
    }
};

// ── Global symbol table ───────────────────────────────────────────────────────
// Fixed-size array of SymbolBars pre-allocated once at startup.
// shared_mutex protects symbol registration only (rare slow path).
// Hot-path reads use a pre-looked-up index with no global lock.
class BarStore {
public:
    BarStore() noexcept : symbol_count_(0) {}

    // Find or register a symbol slot.
    // Returns MAX_SYMBOLS if the table is full.
    [[nodiscard]] std::size_t get_or_create(std::string_view sym) noexcept {
        // Fast path: already registered (shared read lock).
        // BUG-BS4 FIX: [[likely]] removed from the equality check.
        // In a linear scan of up to 24 names, the condition is false on every
        // iteration except the one matching slot.  Annotating the match as
        // [[likely]] inverts branch prediction: the predictor mispredicts on
        // every non-matching slot (up to 23 of 24) — maximum misprediction
        // rate.  No hint lets the CPU's own branch history hardware adapt.
        {
            std::shared_lock lk(mtx_);
            for (std::size_t i = 0; i < symbol_count_; ++i)
                if (std::string_view(bars_[i].name) == sym)
                    return i;
        }
        // Slow path: register new symbol (exclusive write lock).
        // Double-check after acquiring write lock — another thread may have
        // registered this symbol between releasing the shared lock and acquiring
        // the exclusive lock.
        std::unique_lock lk(mtx_);
        for (std::size_t i = 0; i < symbol_count_; ++i)
            if (std::string_view(bars_[i].name) == sym) return i;
        if (symbol_count_ >= MAX_SYMBOLS) [[unlikely]] return MAX_SYMBOLS;
        const std::size_t idx = symbol_count_++;
        auto& b = bars_[idx];
        std::strncpy(b.name, sym.data(), SYM_NAME - 1);
        b.name[SYM_NAME - 1] = '\0';
        return idx;
    }

    // BUG-BS4 FIX: [[likely]] also removed from find() for the same reason.
    [[nodiscard]] std::size_t find(std::string_view sym) const noexcept {
        std::shared_lock lk(mtx_);
        for (std::size_t i = 0; i < symbol_count_; ++i)
            if (std::string_view(bars_[i].name) == sym)
                return i;
        return MAX_SYMBOLS;
    }

    // Direct indexed access — no global lock, no search.
    // Hot path: caller must have validated idx via get_or_create() first.
    //
    // BUG-BS3 FIX: assert that idx is within the static array bounds.
    // at(MAX_SYMBOLS) — the "not found" sentinel from find() — would previously
    // silently access bars_[24], one past the array end, corrupting mtx_.
    // The assert fires in debug builds, producing an immediate, actionable error.
    [[nodiscard]] SymbolBars& at(std::size_t idx) noexcept {
        assert(idx < MAX_SYMBOLS && "BarStore::at(): idx out of bounds. "
               "Did you check for the MAX_SYMBOLS sentinel before calling at()?");
        return bars_[idx];
    }
    [[nodiscard]] const SymbolBars& at(std::size_t idx) const noexcept {
        assert(idx < MAX_SYMBOLS && "BarStore::at(): idx out of bounds.");
        return bars_[idx];
    }

    [[nodiscard]] std::size_t symbol_count() const noexcept {
        std::shared_lock lk(mtx_);
        return symbol_count_;
    }

private:
    // Total size: 24 × 2.5 KiB ≈ 60 KiB — fits in L2 cache (256 KiB typical).
    alignas(64) std::array<SymbolBars, MAX_SYMBOLS> bars_{};
    std::size_t           symbol_count_{0};
    mutable std::shared_mutex mtx_;
};

// ── Per-thread scratch buffers ────────────────────────────────────────────────
// Each HTTP worker thread gets its own buffers — zero contention, zero allocation.
// linearise_all() copies ring buffer data here before SIMD indicator computation.
struct alignas(64) ScratchBuffers {
    std::array<float, MAX_BARS> closes      {};
    std::array<float, MAX_BARS> highs       {};
    std::array<float, MAX_BARS> lows        {};
    std::array<float, MAX_BARS> volumes     {};
    std::array<float, MAX_BARS> ob_imbalances{};
    std::array<float, MAX_BARS> scratch1    {};
    std::array<float, MAX_BARS> scratch2    {};
};

// Inline thread_local: C++17 guarantees one definition per translation unit
// with the linker merging them to a single per-thread instance.
inline thread_local ScratchBuffers g_scratch{};

} // namespace tqc
