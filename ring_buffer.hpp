#pragma once
/*
 * ring_buffer.hpp  -  TQC Brain | Taha Iqbal
 * ============================================================
 * HARDWARE REQUIREMENT: LOCK-FREE IPC
 *
 * std::mutex in a trading system is dangerous:
 *   If the Brain thread holds a mutex and the OS pre-empts it,
 *   the executor thread blocks until rescheduled (10–50 µs typical).
 *   That is longer than the entire fill window on a crypto exchange.
 *
 * SPSC (Single-Producer Single-Consumer) ring buffer:
 *   - Producer only writes head_.  Consumer only reads tail_.
 *   - They NEVER contend on the same atomic. Zero lock overhead.
 *   - std::memory_order_release/acquire provides the required fence.
 *     On x86-64 this compiles to zero extra instructions (TSO model).
 *   - Capacity must be a power of 2:  index = pos & (N-1)  (one AND)
 *
 * CAPACITY NOTE (ISSUE-01 FIX):
 *   The SPSC sentinel design reserves one slot to distinguish full from
 *   empty: "full" = ((head+1) & Mask) == tail.  Therefore the effective
 *   usable capacity is N-1 slots, not N.  Capacity is documented as N-1
 *   to avoid misleading callers.  N must still be a power of 2 for the
 *   branchless mask to work.
 *
 * MPSC ring buffer:
 *   - Used when multiple HTTP threads publish to one consumer.
 *   - Head advances under a CAS loop.  Tail is single-consumer.
 *
 * Cache-line padding on head_/tail_ prevents false sharing
 * (without it, updating head_ on CPU0 invalidates tail_'s cache line
 * on CPU1, causing ~100 ns stalls on every write).
 *
 * ── BUGS FIXED IN THIS VERSION ──────────────────────────────────────────────
 *
 * BUG-RB1  SPSC::pop() — [[likely]] hint on the wrong (empty) branch.
 *          The original code marked the "buffer is empty → return nullopt"
 *          branch as [[likely]].  This is correct only for a polling consumer
 *          that calls pop() in a tight idle loop.  In this system the consumer
 *          calls pop() exactly when it expects data (event-driven HTTP handler
 *          or signal pipeline), so the buffer is NOT empty when pop() is called.
 *          The misplaced [[likely]] trains the CPU branch predictor to predict
 *          "empty" and mispredict on every productive call — the exact opposite
 *          of the intent.
 *          FIX: removed the [[likely]] hint entirely.  No hint is the safest
 *          choice for a mixed-usage buffer; the processor's own branch history
 *          will adapt to whichever call site pattern is in play.
 *
 * BUG-RB2  MPSC::pop() — identical [[likely]] misplacement as BUG-RB1.
 *          FIX: removed the [[likely]] hint from MPSCRingBuffer::pop().
 *
 * BUG-RB3  MPSCRingBuffer — copy and move not explicitly deleted.
 *          SPSCRingBuffer inherits the implicit delete of copy/move from its
 *          atomic members (atomics are not copyable/movable).  However
 *          attempting to copy or move MPSCRingBuffer at a call site produces
 *          a deep compiler error inside the atomic internals rather than a
 *          clean, immediate diagnostic at the point of misuse.
 *          An explicit delete gives the caller a direct, actionable error:
 *          "use of deleted function MPSCRingBuffer(MPSCRingBuffer&&)".
 *          FIX: added explicit = delete for copy-construct, copy-assign,
 *          move-construct, and move-assign on MPSCRingBuffer.
 *
 * BUG-RB4  [NEW] MPSC::push() CAS failure ordering is memory_order_relaxed —
 *          must be memory_order_acquire on non-x86 architectures.
 *
 *          On a failed compare_exchange_weak(), the `head` variable is updated
 *          with the current observed value of head_.  With memory_order_relaxed
 *          as the failure ordering, this observation is made without an acquire
 *          fence.  On x86-64 (TSO — Total Store Order), all loads observe the
 *          most recently committed store regardless of ordering annotation, so
 *          the relaxed failure is harmless in practice on x86.
 *
 *          However this system is required to run correctly on Linux on ANY
 *          hardware.  ARM Graviton 2/3 (common AWS instance types), POWER9/10,
 *          and RISC-V all use weakly-ordered memory models.  On these ISAs,
 *          a relaxed load can observe a stale value that does not reflect prior
 *          CAS successes from other producer threads.  The retry loop then:
 *            (a) computes `next` from a stale `head`, and
 *            (b) re-evaluates the full check using a tail_ snapshot that may
 *                also be stale (loaded acquire, but from a call site that
 *                observed stale head — the two form an inconsistent view).
 *          The net effect is spurious extra CAS iterations or, in pathological
 *          reordering scenarios, a false "full" return that discards a push
 *          that could have succeeded.  No slot double-claim is possible (the
 *          CAS itself is atomic), but throughput and latency are degraded
 *          under high concurrency on non-x86 hardware.
 *          FIX: failure ordering changed from memory_order_relaxed to
 *          memory_order_acquire.  On x86-64 this compiles to exactly the same
 *          instruction (LOCK CMPXCHG already has full ordering).  On ARM it
 *          emits LDAXR/STLXR which are required for correctness.
 *
 * BUG-RB5  [NEW] Unconditional noexcept on push/pop is incorrect for
 *          non-trivially-movable T — silent std::terminate on throwing move.
 *
 *          All push() and pop() overloads in both SPSCRingBuffer and
 *          MPSCRingBuffer are declared noexcept.  However they internally
 *          perform operations that CAN throw if T is not nothrow-move/copy-
 *          constructible:
 *            SPSC::push(const T&) : storage_[head] = item   (copy-assign)
 *            SPSC::push(T&&)      : storage_[head] = move   (move-assign)
 *            SPSC::pop()          : T item = move(storage_) (move-construct)
 *            MPSC::push(const T&) : T copy = item           (copy-construct)
 *            MPSC::pop()          : T item = move(storage_) (move-construct)
 *          If T's constructor or assignment operator throws, the C++ runtime
 *          calls std::terminate() immediately — no stack unwind, no log entry,
 *          the process dies with no actionable diagnostic.
 *
 *          For the types currently used in this system (Signal, Bar, OHLCBar —
 *          all trivially copyable, all nothrow everything), this is not a live
 *          bug today.  But as a reusable template instantiated over arbitrary T,
 *          the unconditional noexcept is a hidden trap that will fire the first
 *          time a non-trivial payload type is used without any compiler warning.
 *
 *          FIX: noexcept specifications changed to conditional form:
 *            push(const T&) : noexcept(nothrow_copy_and_move<T>)
 *            push(T&&)      : noexcept(nothrow_move<T>)
 *            pop()          : noexcept(nothrow_move<T>)
 *          where the helper aliases are defined at the top of each class.
 *          For the current trivially-copyable T, all conditions evaluate to
 *          true at compile time, so the generated code is IDENTICAL to the
 *          original unconditional noexcept — zero runtime cost.
 *
 * BUG-RB6  [NEW / DOCUMENTED TRADE-OFF] ready_ array causes false sharing
 *          between producers on wide machines.
 *          std::atomic<bool> is 1 byte on all major implementations.  For
 *          N=8 (typical HTTP worker count), all 8 ready flags occupy 8 bytes
 *          — comfortably inside one 64-byte cache line.  Two producers
 *          claiming adjacent slots simultaneously both write to this cache line
 *          (ready_[0] and ready_[1]), triggering a coherency invalidation
 *          round-trip between their cores.  On high-core-count ARM servers
 *          (Graviton 3: 64 cores) this adds measurable latency under sustained
 *          concurrent push load.
 *
 *          The production fix is to pad each flag to a full cache line:
 *            struct alignas(CACHE_LINE) PaddedFlag { std::atomic<bool> v{}; };
 *            std::array<PaddedFlag, N> ready_{};
 *          This eliminates all producer-producer false sharing at the cost of
 *          N × 64 bytes of additional memory.  For N=8: 512 bytes — acceptable.
 *          For N=256: 16 KB — may exceed L1 and cause its own cache pressure.
 *
 *          NOT FIXED in this pass: the HTTP-rate use case (≪ 1M pushes/sec,
 *          small N) makes the false sharing impact negligible.  If this buffer
 *          is ever used on a high-rate path, apply the PaddedFlag fix and
 *          profile.  The trade-off is documented here for future maintainers.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <array>
#include <atomic>
#include <cstddef>
#include <optional>
#include <new>
#include <type_traits>   // BUG-RB5 FIX: for is_nothrow_* traits

namespace tqc {

// Portable cache-line size
#ifdef __cpp_lib_hardware_interference_size
    inline constexpr std::size_t CACHE_LINE =
        std::hardware_destructive_interference_size;
#else
    inline constexpr std::size_t CACHE_LINE = 64;
#endif

// ── SPSC Ring Buffer ──────────────────────────────────────────────────────────
// Single-Producer, Single-Consumer.
// Thread safety: push() and pop() may each be called from exactly ONE thread.
// empty(), size(), and clear() are advisory only — safe to call from any thread
// but the result may be stale by the time the caller acts on it.
template<typename T, std::size_t N>
class SPSCRingBuffer {
    static_assert((N & (N - 1)) == 0, "N must be a power of 2");
    static_assert(N >= 4,             "Minimum N is 4 (effective capacity N-1 >= 3)");

    // BUG-RB5 FIX: compile-time noexcept predicates for T.
    // For the current Signal/Bar types (trivially copyable), both evaluate to
    // true and the generated code is identical to unconditional noexcept.
    // For a future non-trivial T, the compiler will correctly mark push/pop
    // as potentially-throwing, enabling proper exception handling at call sites.
    static constexpr bool k_nothrow_move =
        std::is_nothrow_move_constructible_v<T> &&
        std::is_nothrow_move_assignable_v<T>;

    static constexpr bool k_nothrow_copy_and_move =
        std::is_nothrow_copy_constructible_v<T> &&
        k_nothrow_move;

public:
    // Sentinel design: one slot reserved to distinguish full from empty.
    // Effective usable capacity is N-1, not N.
    static constexpr std::size_t Capacity = N - 1;
    static constexpr std::size_t Mask     = N - 1;

    // Non-copyable, non-movable (atomics + fixed storage addresses).
    SPSCRingBuffer()                                  = default;
    SPSCRingBuffer(const SPSCRingBuffer&)             = delete;
    SPSCRingBuffer& operator=(const SPSCRingBuffer&) = delete;
    SPSCRingBuffer(SPSCRingBuffer&&)                  = delete;
    SPSCRingBuffer& operator=(SPSCRingBuffer&&)       = delete;

    // ── push (producer only) ─────────────────────────────────────────────────
    // BUG-RB5 FIX: conditional noexcept — copy path requires nothrow copy + move.
    [[nodiscard]] bool push(const T& item) noexcept(k_nothrow_copy_and_move) {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        const std::size_t next = (head + 1) & Mask;
        if (next == tail_.load(std::memory_order_acquire)) [[unlikely]]
            return false;  // full
        storage_[head] = item;
        head_.store(next, std::memory_order_release);
        return true;
    }

    // BUG-RB5 FIX: conditional noexcept — move path requires nothrow move.
    [[nodiscard]] bool push(T&& item) noexcept(k_nothrow_move) {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        const std::size_t next = (head + 1) & Mask;
        if (next == tail_.load(std::memory_order_acquire)) [[unlikely]]
            return false;
        storage_[head] = std::move(item);
        head_.store(next, std::memory_order_release);
        return true;
    }

    // ── pop (consumer only) ──────────────────────────────────────────────────
    // BUG-RB1 FIX: [[likely]] removed from the empty-return branch.
    // The hint was backwards for an event-driven consumer that is only called
    // when data is expected.  No hint lets the CPU branch predictor adapt to
    // whichever usage pattern is actually in play at each call site.
    //
    // BUG-RB5 FIX: conditional noexcept — move-construct of return value.
    [[nodiscard]] std::optional<T> pop() noexcept(k_nothrow_move) {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire))
            return std::nullopt;  // empty
        T item = std::move(storage_[tail]);
        tail_.store((tail + 1) & Mask, std::memory_order_release);
        return item;
    }

    // ── Utilities (advisory — may be stale) ──────────────────────────────────
    [[nodiscard]] bool empty() const noexcept {
        return head_.load(std::memory_order_acquire)
            == tail_.load(std::memory_order_acquire);
    }

    [[nodiscard]] std::size_t size() const noexcept {
        const std::size_t h = head_.load(std::memory_order_acquire);
        const std::size_t t = tail_.load(std::memory_order_acquire);
        return (h >= t) ? (h - t) : (N - t + h);
    }

    // CAUTION: only call clear() when no concurrent producer or consumer is active.
    // Setting tail_ = head_ from outside the consumer thread is a data race if
    // the consumer is concurrently inside pop() on another thread.
    void clear() noexcept {
        tail_.store(head_.load(std::memory_order_acquire),
                    std::memory_order_release);
    }

private:
    // head_ and tail_ on separate cache lines — prevents false sharing.
    alignas(CACHE_LINE) std::atomic<std::size_t> head_{0};
    alignas(CACHE_LINE) std::atomic<std::size_t> tail_{0};
    alignas(CACHE_LINE) std::array<T, N>         storage_{};
};

// ── MPSC Ring Buffer (Multiple-Producer, Single-Consumer) ─────────────────────
// Head advances under a CAS loop. Tail is single-consumer (no contention).
//
// FIX (MPSC-Race): The original implementation advanced head_ via CAS *before*
// writing the payload into storage_[head].  A consumer racing on the same slot
// could observe the new head_ value and pop a slot that had not yet been written
// by the producer — returning a default-constructed or stale T with no error.
//
// Root cause: CAS establishes the slot's ownership but NOT its readiness.
// Two operations are needed, not one:
//   1. CAS: claim ownership of slot `head` (advance the shared head counter).
//   2. Write: copy/move item into storage_[head].
//   3. Commit: mark the slot ready for the consumer.
//
// Fix: a per-slot commit flag array (ready_[N], each atomic<bool>).
//   Producer: after CAS succeeds and item is written, sets ready_[head] = true
//             with release semantics.
//   Consumer: before reading storage_[t], spins on ready_[t] (acquire).
//             Resets ready_[t] = false after consuming.
//
// The spin in pop() is bounded: the window between CAS and the store is O(1)
// instructions (one move/copy + one atomic store).  Typical spin count = 0.
// For the HTTP-acceptor use case (one acceptor + N workers, low-rate),
// this is negligible.  For a high-rate critical path, prefer SPSC instead.
template<typename T, std::size_t N>
class MPSCRingBuffer {
    static_assert((N & (N - 1)) == 0, "N must be a power of 2");
    static_assert(N >= 4,             "Minimum N is 4");

    // BUG-RB5 FIX: compile-time noexcept predicates (same rationale as SPSC).
    static constexpr bool k_nothrow_move =
        std::is_nothrow_move_constructible_v<T> &&
        std::is_nothrow_move_assignable_v<T>;

    static constexpr bool k_nothrow_copy_and_move =
        std::is_nothrow_copy_constructible_v<T> &&
        k_nothrow_move;

public:
    static constexpr std::size_t Capacity = N - 1; // consistent with SPSC

    MPSCRingBuffer() noexcept {
        // Explicitly initialise every ready flag to false.
        // std::atomic<bool> default-constructor does NOT guarantee
        // zero-initialisation in C++17 (only in C++20).  The member-initialiser
        // ready_{} value-initialises each atomic but the standard only guarantees
        // a defined initial value in C++20.  The explicit loop here is correct
        // for all C++ versions this codebase targets (C++17 and later).
        for (std::size_t i = 0; i < N; ++i)
            ready_[i].store(false, std::memory_order_relaxed);
    }

    // BUG-RB3 FIX: explicitly delete copy and move.
    // Implicit deletion from atomic members produces a confusing error deep in
    // the atomic implementation.  Explicit deletion gives a direct diagnostic.
    MPSCRingBuffer(const MPSCRingBuffer&)             = delete;
    MPSCRingBuffer& operator=(const MPSCRingBuffer&) = delete;
    MPSCRingBuffer(MPSCRingBuffer&&)                  = delete;
    MPSCRingBuffer& operator=(MPSCRingBuffer&&)       = delete;

    // BUG-RB5 FIX: conditional noexcept — move path.
    // BUG-RB4 FIX: CAS failure ordering changed from memory_order_relaxed to
    //              memory_order_acquire (see header comment for full rationale).
    //              Summary: on ARM/POWER, relaxed failure gives a stale `head`
    //              value to the retry loop.  acquire failure ensures the updated
    //              head_ reflects all prior CAS successes from other producers,
    //              preventing spurious extra retries and false "full" returns.
    //              On x86-64 this change compiles to the same LOCK CMPXCHG
    //              instruction — zero performance impact on x86.
    [[nodiscard]] bool push(T&& item) noexcept(k_nothrow_move) {
        std::size_t head = head_.load(std::memory_order_relaxed);
        std::size_t next;
        do {
            next = (head + 1) & (N - 1);
            if (next == tail_.load(std::memory_order_acquire)) [[unlikely]]
                return false;  // full
        } while (!head_.compare_exchange_weak(
                     head, next,
                     std::memory_order_acq_rel,  // success: publish new head (release)
                                                 //          + see prior pops (acquire)
                     std::memory_order_acquire   // BUG-RB4 FIX: was relaxed.
                                                 // failure: observe the true current
                                                 // head_ from other producers' CAS
                                                 // successes on ARM/POWER/RISC-V.
                                                 // On x86-64: compiles identically.
                 ));

        // Slot `head` is now exclusively ours.  Write payload, then commit.
        storage_[head] = std::move(item);
        ready_[head].store(true, std::memory_order_release); // publish to consumer
        return true;
    }

    // push() overload for lvalue references — copies into a temporary then moves.
    // BUG-RB5 FIX: conditional noexcept — copy path requires nothrow copy + move.
    [[nodiscard]] bool push(const T& item) noexcept(k_nothrow_copy_and_move) {
        T copy = item;
        return push(std::move(copy));
    }

    // BUG-RB2 FIX: [[likely]] removed from the empty-return branch.
    // Same reasoning as BUG-RB1 in SPSC — wrong hint for event-driven consumers.
    //
    // BUG-RB5 FIX: conditional noexcept — move-construct of return value.
    [[nodiscard]] std::optional<T> pop() noexcept(k_nothrow_move) {
        const std::size_t t = tail_.load(std::memory_order_relaxed);
        if (t == head_.load(std::memory_order_acquire))
            return std::nullopt;  // empty

        // Spin until the producer that claimed this slot has committed the write.
        // Typical spin count = 0 (producer is O(1) instructions ahead).
        // The [[unlikely]] hint on the while body (C++20) tells the branch
        // predictor that the spin body is rarely executed — correct for this case.
        while (!ready_[t].load(std::memory_order_acquire)) [[unlikely]] {
            // Intentionally tight — the producer is at most one store away.
            // A yield() here would add ~100 ns latency for a 1-instruction window.
            //
            // KNOWN RISK: if the producer thread is preempted by the OS between
            // its CAS success and its ready_[head] store (e.g., SIGSTOP, long GC,
            // scheduler eviction), this loop spins for the full OS quantum (~10 ms)
            // before the producer is rescheduled.  For a single-threaded event loop
            // where the consumer IS the scheduler, this can cause a deadlock.
            // Mitigation: use SPSC if the producer is also on the critical path,
            // or add a spin-count limit with std::this_thread::yield() fallback
            // if producer-preemption risk is unacceptable for your deployment.
        }

        T item = std::move(storage_[t]);
        ready_[t].store(false, std::memory_order_relaxed); // reset for reuse
        // Release on tail_ advance ensures ready_[t]=false is visible to any
        // producer that later claims slot t (via their acquire on tail_).
        tail_.store((t + 1) & (N - 1), std::memory_order_release);
        return item;
    }

    [[nodiscard]] bool empty() const noexcept {
        return head_.load(std::memory_order_acquire)
            == tail_.load(std::memory_order_acquire);
    }

private:
    alignas(CACHE_LINE) std::atomic<std::size_t> head_{0};
    alignas(CACHE_LINE) std::atomic<std::size_t> tail_{0};
    alignas(CACHE_LINE) std::array<T, N>         storage_{};

    // Per-slot commit flags — one cache line per 64 bools.
    // For N<=64 all flags fit in one cache line.
    //
    // NOTE (BUG-RB6 trade-off): multiple producers writing adjacent ready_[i]
    // flags share cache lines, causing producer-producer false sharing.  For
    // the HTTP-rate use case (small N, low push rate) this is negligible.
    // If this buffer is used on a high-rate path, replace with:
    //   struct alignas(CACHE_LINE) PaddedFlag { std::atomic<bool> v{}; };
    //   std::array<PaddedFlag, N> ready_{};
    // and update all ready_[i] accesses to ready_[i].v.
    alignas(CACHE_LINE) std::array<std::atomic<bool>, N> ready_{};
};

} // namespace tqc
