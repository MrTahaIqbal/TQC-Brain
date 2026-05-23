#pragma once
/*
 * object_pool.hpp  - TQC Brain | Taha Iqbal
 * ============================================================
 * HARDWARE REQUIREMENT: NO new/delete IN HOT PATH
 *
 * std::malloc() calls the OS.  The OS can pause your thread for
 * microseconds while it searches for a free memory block (jitter).
 * In live trading, jitter = missed entries.
 *
 * Solution: allocate ALL memory once at startup.
 *   ObjectPool<T, N>:
 *     - N raw-storage slots in a char array  (BSS segment — no heap)
 *     - Free-list tracked with a lock-free CAS atomic
 *     - acquire() and release() are O(1) with zero OS calls
 *     - ABA problem prevented by an embedded version counter
 *
 * Layout of the 64-bit CAS word:
 *   [version: upper 48 bits | free_list_head: lower 16 bits]
 *   The version counter increments on every push/pop, making ABA impossible.
 *
 * OBJECT LIFETIME CONTRACT:
 *   Raw storage — no T is constructed until the caller placement-news it.
 *   This contract is SYMMETRIC across every acquire/release cycle:
 *     acquire()  → returns uninitialised storage; caller MUST placement-new.
 *     release()  → calls ~T(); resets slot to uninitialised.
 *   The previous implementation (std::array<T,N> storage_{}) broke this
 *   symmetry: default-constructing all N T objects at pool creation meant
 *   the first cycle of acquire+placement-new destroyed a live object (the
 *   one constructed by std::array<T,N>{}). See BUG-OP3 for full detail.
 *
 * ── BUGS FIXED IN THIS VERSION ──────────────────────────────────────────────
 *
 * BUG-OP1  ObjectPool::release() — ptr->~T() called AFTER the CAS that
 *          returns the slot to the free list.  This is a use-after-free race.
 *
 *          Scenario (two threads, any multi-core machine):
 *            T1 (release):  CAS succeeds → slot `idx` is now the free-list head
 *            T2 (acquire):  loads top_, unpacks idx, CAS succeeds → gets &storage_[idx]
 *            T2 (caller):   new(&storage_[idx]) T(args...)  ← constructs new object
 *            T1 (release):  ptr->~T()  ← DESTROYS the object T2 just constructed!
 *
 *          The window (CAS to ~T()) is only a few instructions, but on a busy
 *          trading system running 24/7 this WILL eventually manifest as silent
 *          memory corruption — a pool slot whose destructor runs on a live object.
 *
 *          Root cause: the previous comment ("destroy after CAS to avoid
 *          spinning on an already-destroyed object") is based on a faulty
 *          premise.  The CAS loop in release() spins on top_ (the global free-
 *          list head), not on the slot being released.  The slot being released
 *          (storage_[idx]) is NEVER visible to other threads until the CAS
 *          succeeds, so it is exclusively owned by the releasing thread at all
 *          times before the CAS.  Calling ~T() before the CAS is completely safe.
 *
 *          FIX: moved ptr->~T() to BEFORE the CAS loop — while the slot is still
 *          exclusively owned by the releasing thread.  After the CAS, the slot is
 *          on the free list and could be acquired+placement-new'd at any instant.
 *
 * BUG-OP2  PoolHandle — missing move assignment operator.
 *          The move constructor is defined, but without a matching move assignment,
 *          the compiler synthesises a defaulted one that:
 *            (a) Does NOT call pool_->release(ptr_) on the overwritten handle.
 *                This leaks the old slot — it's never returned to the pool.
 *            (b) Does NOT null out the source's ptr_.
 *                The source's destructor then calls release() again on the same
 *                pointer the destination now owns → double-release.
 *          In a pool with ABA-protected CAS this causes incorrect available()
 *          counts and potentially gives the same slot to two callers simultaneously.
 *          FIX: added explicit move assignment operator that:
 *            - releases the existing ptr_ if non-null,
 *            - steals pool_ and ptr_ from the source,
 *            - nulls out the source's ptr_.
 *
 * BUG-OP3  [NEW] ObjectPool used std::array<T, N> storage_{} — this
 *          default-constructs all N T objects at pool creation, breaking the
 *          symmetric acquire/release lifetime contract.
 *
 *          For non-trivial T (any type with a non-trivial default constructor,
 *          e.g. a Signal struct with std::string fields, a T owning a file
 *          descriptor or a network socket), default-construction at pool startup
 *          allocates resources that are never released before the first
 *          release() call:
 *
 *          Pool startup:   std::array<T,N>{}  → calls T() for all N slots.
 *          acquire():      returns &storage_[idx]  (a live T object).
 *          Caller:         new(&storage_[idx]) T(args...)
 *                          ← calls placement-new on a LIVE object.
 *                          ← the existing object's resources are overwritten
 *                             without calling ~T() first.  Resource leak.
 *
 *          For the current Signal and Bar types (trivially copyable, no
 *          owned resources), this is not a live bug.  But as a generic
 *          template instantiated over arbitrary T, the error is a trap
 *          waiting to fire when the pool is used with any richer payload.
 *
 *          FIX: storage_ changed from std::array<T, N> to raw aligned byte
 *          storage (alignas(T) char raw_[sizeof(T) * N]).  A slot(i) helper
 *          uses std::launder to obtain a valid T* from the raw storage.
 *          No T is ever constructed until the caller placement-news it after
 *          acquire().  Every release() destructs exactly the T that was
 *          constructed.  The contract is now SYMMETRIC across all cycles,
 *          including the first.
 *
 *          IMPORTANT: After this change, the pool ObjectPool<T,N> no longer
 *          zero-initialises slots.  Callers MUST placement-new before reading
 *          any field.  The existing acquire() docstring already required this;
 *          this fix makes the requirement enforceable (reading an
 *          unconstructed slot is now formally UB, which sanitisers detect).
 *
 * BUG-OP4  [NEW] PoolHandle::operator->() and operator*() dereference ptr_
 *          unconditionally with no null guard.
 *
 *          A PoolHandle constructed from an exhausted pool:
 *              auto h = PoolHandle(pool, pool.acquire());
 *          has ptr_ == nullptr when pool.acquire() returns nullptr (pool full).
 *          valid() correctly returns false, but operator->() and operator*()
 *          dereference ptr_ regardless, producing a null-pointer dereference
 *          with no actionable diagnostic — the segfault occurs at the
 *          dereference site, which may be several stack frames away from the
 *          point where the handle was constructed.
 *
 *          In a busy trading cycle where the pool is occasionally exhausted
 *          (e.g., 20 concurrent signals with N=16 slots), any call site that
 *          omits the `if (h.valid())` guard hits this path silently.
 *
 *          FIX: assert(ptr_ != nullptr) added at the top of both operators.
 *          - Debug builds: assert fires immediately at the dereference site
 *            with a clear message and a full stack trace in most debuggers.
 *            This catches missing valid() guards during development.
 *          - Release builds (NDEBUG defined): assert is a no-op; the
 *            dereference proceeds as before.  Zero runtime cost in production.
 *          The assert does NOT add a branch in release builds and does NOT
 *          change the return type or calling convention — pure safety annotation.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cassert>
#include <new>      // std::launder — BUG-OP3 FIX
#include <utility>  // std::pair

namespace tqc {

template<typename T, std::size_t N>
class ObjectPool {
    static_assert(N > 0 && N <= 65534, "Pool size must be 1–65534");

public:
    static constexpr std::size_t Capacity = N;

    // Sentinel value for "list is empty"
    static constexpr uint16_t NIL = static_cast<uint16_t>(N);

    // BUG-OP3 FIX: constructor no longer calls T() for any slot.
    // next_[] free-list is initialised (trivial uint16_t, no construction),
    // top_ and available_ are set.  No T objects exist until acquire()+new().
    ObjectPool() noexcept {
        for (std::size_t i = 0; i < N; ++i)
            next_[i] = static_cast<uint16_t>(i + 1);
        next_[N - 1] = NIL;
        top_.store(pack(0, 0), std::memory_order_relaxed);
        available_.store(N, std::memory_order_relaxed);
    }

    // BUG-OP3 FIX: destructor must NOT call T destructors on raw_ storage
    // because acquire()d slots may still be in use by callers.
    // The pool is designed for static or global lifetime — it should outlive
    // all PoolHandles.  Leaking constructed T objects at shutdown is
    // acceptable for this use case; the OS reclaims all memory on process exit.
    // If graceful shutdown is required, all handles must be explicitly reset()
    // before the pool is destroyed.
    ~ObjectPool() noexcept = default;

    // Non-copyable, non-movable (fixed addresses required for release()).
    ObjectPool(const ObjectPool&)            = delete;
    ObjectPool& operator=(const ObjectPool&) = delete;
    ObjectPool(ObjectPool&&)                 = delete;
    ObjectPool& operator=(ObjectPool&&)      = delete;

    // ── acquire: O(1), lock-free ──────────────────────────────────────────────
    // Returns raw pointer to an uninitialised T-slot, or nullptr if exhausted.
    // Caller MUST placement-new before first use:
    //   T* p = pool.acquire();
    //   if (p) new(p) T(args...);
    // This contract is now enforced by the raw storage design (BUG-OP3 FIX):
    // reading any field of *p before placement-new is formally UB and will be
    // caught by AddressSanitizer and MemorySanitizer in debug builds.
    [[nodiscard]] T* acquire() noexcept {
        uint64_t old = top_.load(std::memory_order_acquire);
        while (true) {
            auto [idx, ver] = unpack(old);
            if (idx >= N) [[unlikely]] return nullptr;  // pool exhausted

            const uint16_t nxt     = next_[idx];
            const uint64_t desired = pack(nxt, ver + 1);

            if (top_.compare_exchange_weak(old, desired,
                    std::memory_order_release,
                    std::memory_order_acquire)) {
                available_.fetch_sub(1, std::memory_order_relaxed);
                return slot(idx);   // BUG-OP3 FIX: returns laundered raw pointer
            }
        }
    }

    // ── release: O(1), lock-free ──────────────────────────────────────────────
    // BUG-OP1 FIX: ptr->~T() is now called BEFORE the CAS loop.
    //
    // WHY THIS IS SAFE:
    //   Until the CAS succeeds, slot `idx` is NOT on the free list.  No other
    //   thread can acquire() it.  The slot is exclusively owned by this release()
    //   call regardless of how many times the CAS retries.  Calling ~T() here
    //   is safe and correct — it mirrors the acquire() contract which requires
    //   the caller to placement-new before use (the destroyed slot is ready for
    //   re-construction by the next acquirer after the CAS completes).
    //
    // WHY THE PREVIOUS APPROACH WAS WRONG:
    //   After the CAS succeeds, the slot is publicly visible on the free list.
    //   Any thread can acquire() it and placement-new a new T into it.  The
    //   original code called ~T() after the CAS — potentially destroying the
    //   freshly constructed object from the concurrent acquire().  This is a
    //   classic lock-free use-after-free race condition.
    void release(T* ptr) noexcept {
        assert(ptr >= slot(0) && ptr < slot(N) &&
               "PoolHandle::release(): pointer does not belong to this pool.");
        const uint16_t idx = static_cast<uint16_t>(ptr - slot(0));

        // Destroy FIRST — slot is not on the free list yet, exclusively ours.
        // After the CAS below, the slot is publicly visible; ~T() would race
        // with a concurrent acquire()+placement-new (BUG-OP1 FIX).
        ptr->~T();

        uint64_t old = top_.load(std::memory_order_acquire);
        while (true) {
            auto [top_idx, ver] = unpack(old);
            next_[idx] = static_cast<uint16_t>(top_idx);
            const uint64_t desired = pack(idx, ver + 1);

            if (top_.compare_exchange_weak(old, desired,
                    std::memory_order_release,
                    std::memory_order_acquire)) {
                available_.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }
    }

    [[nodiscard]] std::size_t available() const noexcept {
        return available_.load(std::memory_order_relaxed);
    }

private:
    // Layout: [ver:48 | idx:16]
    // L-01 NOTE: The 48-bit version counter wraps after ~2.8×10^14 CAS operations
    // (~8.9 years at 1M ops/sec). ABA prevention is theoretically broken at wrap,
    // but this is unreachable in normal trading operation.
    static uint64_t pack(uint16_t idx, uint64_t ver) noexcept {
        return (ver << 16) | idx;
    }
    static std::pair<uint16_t, uint64_t> unpack(uint64_t v) noexcept {
        return { static_cast<uint16_t>(v & 0xFFFF), v >> 16 };
    }

    // BUG-OP3 FIX: raw aligned byte storage replaces std::array<T, N>.
    //
    // std::array<T, N> storage_{} called T() for every slot at pool construction,
    // breaking the symmetric lifetime contract: the first placement-new after
    // acquire() would overwrite a live T object without calling its destructor.
    //
    // Raw char storage defers all T construction to explicit placement-new by
    // the caller.  std::launder in slot() is required to obtain a T* from the
    // raw bytes without violating strict aliasing (C++17 [ptr.launder]).
    //
    // Alignment: alignas(T) guarantees that slot(i) satisfies T's alignment
    // requirement.  The manual offset arithmetic (sizeof(T) * i) is guaranteed
    // correct because sizeof(T) already includes any tail padding T requires.
    alignas(T) char raw_[sizeof(T) * N];

    // Free-list next pointers and metadata — trivially initialised.
    alignas(64) std::array<uint16_t, N> next_{};
    std::atomic<uint64_t>    top_{0};
    std::atomic<std::size_t> available_{N};

    // Helper: obtain a valid T* for slot index i from raw storage.
    // std::launder is required for correct strict-aliasing semantics when
    // accessing T objects through a char* buffer in C++17 and later.
    [[nodiscard]] T* slot(std::size_t i) noexcept {
        return std::launder(reinterpret_cast<T*>(raw_ + sizeof(T) * i));
    }
    [[nodiscard]] const T* slot(std::size_t i) const noexcept {
        return std::launder(reinterpret_cast<const T*>(raw_ + sizeof(T) * i));
    }
};

// ── RAII scoped handle ────────────────────────────────────────────────────────
// Calls release() automatically on scope exit.
// Usage:
//   auto h = PoolHandle(pool, pool.acquire());
//   if (h.valid()) {
//       new(h.get()) T(args...);  // placement-new REQUIRED before first use
//       h->field = value;
//   }
// ALWAYS check h.valid() after construction.  If the pool is exhausted,
// acquire() returns nullptr and h.valid() returns false.  Calling h->field
// on an invalid handle is undefined behaviour (caught by assert in debug
// builds — see BUG-OP4 FIX below).
template<typename T, std::size_t N>
class PoolHandle {
public:
    PoolHandle(ObjectPool<T, N>& pool, T* ptr) noexcept
        : pool_(&pool), ptr_(ptr) {}

    ~PoolHandle() noexcept {
        if (ptr_) [[likely]] pool_->release(ptr_);
    }

    // Non-copyable.
    PoolHandle(const PoolHandle&)            = delete;
    PoolHandle& operator=(const PoolHandle&) = delete;

    // Move constructor: steal ownership, null source.
    PoolHandle(PoolHandle&& o) noexcept
        : pool_(o.pool_), ptr_(o.ptr_) {
        o.ptr_ = nullptr;
    }

    // BUG-OP2 FIX: explicit move assignment operator.
    // Without this, the compiler-generated version does NOT release the
    // overwritten handle (slot leak) and does NOT null the source's ptr_
    // (double-release on destruction).
    PoolHandle& operator=(PoolHandle&& o) noexcept {
        if (this != &o) {
            // Release the currently held slot (if any) before stealing the new one.
            if (ptr_) pool_->release(ptr_);
            pool_  = o.pool_;
            ptr_   = o.ptr_;
            o.ptr_ = nullptr;   // source no longer owns the slot
        }
        return *this;
    }

    // BUG-OP4 FIX: assert ptr_ != nullptr before dereferencing.
    //
    // A PoolHandle constructed from an exhausted pool has ptr_ == nullptr.
    // valid() returns false in this case, but operator->() and operator*()
    // previously dereferenced ptr_ unconditionally — a null dereference with
    // no diagnostic, typically manifesting as a segfault several frames away.
    //
    // The assert fires immediately at the dereference site in debug builds,
    // making the missing valid() check trivially discoverable.
    // In release builds (NDEBUG defined), assert is a no-op — zero overhead.
    [[nodiscard]] T* operator->() noexcept {
        assert(ptr_ != nullptr &&
               "PoolHandle::operator->(): handle is invalid (pool was exhausted). "
               "Always check valid() before dereferencing.");
        return ptr_;
    }
    [[nodiscard]] T& operator*() noexcept {
        assert(ptr_ != nullptr &&
               "PoolHandle::operator*(): handle is invalid (pool was exhausted). "
               "Always check valid() before dereferencing.");
        return *ptr_;
    }
    [[nodiscard]] const T* operator->() const noexcept {
        assert(ptr_ != nullptr &&
               "PoolHandle::operator->() const: handle is invalid.");
        return ptr_;
    }
    [[nodiscard]] const T& operator*() const noexcept {
        assert(ptr_ != nullptr &&
               "PoolHandle::operator*() const: handle is invalid.");
        return *ptr_;
    }

    // get(): returns raw pointer without asserting — safe to call even when
    // invalid.  Use this when you need the pointer for a nullptr check rather
    // than for access.  Prefer valid() for boolean checks.
    [[nodiscard]] T*   get()   const noexcept { return ptr_; }
    [[nodiscard]] bool valid() const noexcept { return ptr_ != nullptr; }

    // Explicit release before scope exit (optional — destructor handles it too).
    void reset() noexcept {
        if (ptr_) {
            pool_->release(ptr_);
            ptr_ = nullptr;
        }
    }

private:
    ObjectPool<T, N>* pool_;
    T*                ptr_;
};

} // namespace tqc
