#pragma once
/*
 * tracking.hpp  -  BigBoyAgent TQC Brain | Taha Iqbal
 *
 * Four stateful trackers in one translation unit:
 *   1. Analytics       — per-symbol P&L, Sharpe, Sortino, Calmar
 *   2. PortfolioEngine — simulated positions with leverage, fees, slippage
 *   3. PositionManager — disk-persisted brain-side simulation state
 *   4. LatencyTracker  — rolling latency stats for /predict endpoint
 *
 * ALL state lives in fixed-size arrays — zero heap allocation in any class.
 *
 * Thread safety:
 *   - Analytics      uses per-symbol spinlocks (SymAnalytics::lock_) + reg_lock_.
 *   - PortfolioEngine uses a single bounded spinlock.
 *   - PositionManager uses a bounded spinlock + a persistent std::jthread saver.
 *   - LatencyTracker  uses a single bounded spinlock.
 *
 * BUG-TR10 FIX: all three spinLock() methods in PortfolioEngine,
 *   PositionManager, and LatencyTracker were unbounded busy-spins
 *   (while(lock_.test_and_set(...));) — identical to the B-08 live-lock
 *   risk fixed in SpinGuard and SymAnalytics::push().  Under a HuggingFace
 *   cold filesystem, saveToDisk_impl() can hold PositionManager::lock_ for
 *   tens of milliseconds.  An unbounded spin on an HTTP worker thread for
 *   that duration is a guaranteed stall on every request during a disk flush.
 *   FIX: all three spinLock() methods now yield after 1000 iterations.
 */

#include "types.hpp"
#include <array>
#include <atomic>
#include <cstddef>
#include <chrono>
#include <thread>   // std::jthread (C++20), std::this_thread::yield

namespace tqc {

// ── 1. Analytics ──────────────────────────────────────────────────────────────

class Analytics {
public:
    void logTrade(const char* symbol, float net_pnl,
                  float balance_after, float entry_size) noexcept;

    struct Metrics {
        int   total_trades  = 0;
        float net_pnl_usd   = 0.0f;
        float win_rate      = 0.0f;
        float profit_factor = 0.0f;
        float avg_pnl       = 0.0f;
        float sharpe        = 0.0f;
        float sortino       = 0.0f;
        float max_drawdown  = 0.0f;
        float calmar        = 0.0f;
    };

    [[nodiscard]] Metrics computeMetrics(const char* symbol,
                                          int window = 0) const noexcept;

    int allMetricsJSON(char* buf, int buf_size) const noexcept;

private:
    static constexpr int MAX_HIST = 256;
    static constexpr int MAX_SYMS = 24;

    struct SymAnalytics {
        char  name[16]{};
        float pnls  [MAX_HIST]{};
        float rets  [MAX_HIST]{};
        float equity[MAX_HIST]{};
        std::size_t head  = 0;
        std::size_t count = 0;
        mutable std::atomic_flag lock = ATOMIC_FLAG_INIT;

        void push(float pnl, float ret, float eq) noexcept {
            for (int sp = 0; lock.test_and_set(std::memory_order_acquire); ++sp)
                if (sp > 1000) std::this_thread::yield();
            pnls  [head & (MAX_HIST - 1)] = pnl;
            rets  [head & (MAX_HIST - 1)] = ret;
            equity[head & (MAX_HIST - 1)] = eq;
            ++head;
            if (count < MAX_HIST) ++count;
            lock.clear(std::memory_order_release);
        }

        void linearise(float* op, float* or_, float* oeq,
                       std::size_t n) const noexcept {
            n = (n < count) ? n : count;
            for (std::size_t i = 0; i < n; ++i) {
                const std::size_t slot = (head - 1 - (n - 1 - i)) & (MAX_HIST - 1);
                op [i] = pnls  [slot];
                or_[i] = rets  [slot];
                oeq[i] = equity[slot];
            }
        }
    };

    std::array<SymAnalytics, MAX_SYMS> syms_{};
    std::atomic<int>  nsyms_{0};
    mutable std::atomic_flag reg_lock_ = ATOMIC_FLAG_INIT;

    int findOrCreate(const char* sym) noexcept;
    [[nodiscard]] int find(const char* sym) const noexcept;
};

[[nodiscard]] Analytics& globalAnalytics() noexcept;

// ── 2. Portfolio Engine ───────────────────────────────────────────────────────

struct SimPosition {
    char  symbol[16]{};
    char  side[8]{};
    float entry_price = 0.0f;
    float margin_usd  = 0.0f;
    float notional    = 0.0f;
    float quantity    = 0.0f;
    float fee_open    = 0.0f;
    bool  active      = false;
};

class PortfolioEngine {
public:
    void openPosition(const char* symbol, const char* side,
                      float price, float margin_usd) noexcept;

    struct CloseResult {
        float net_pnl    = 0.0f;
        float gross_pnl  = 0.0f;
        float fees       = 0.0f;
        float exit_price = 0.0f;
        float size_usd   = 0.0f;
        bool  closed     = false;
    };
    [[nodiscard]] CloseResult closePosition(const char* symbol,
                                             float current_price) noexcept;

    [[nodiscard]] float getUnrealizedPnL(const char* symbol,
                                          float current_price) const noexcept;
    [[nodiscard]] bool  isOpen(const char* symbol) const noexcept;

private:
    static constexpr int MAX_POS = 8;
    std::array<SimPosition, MAX_POS> positions_{};
    mutable std::atomic_flag lock_ = ATOMIC_FLAG_INIT;

    [[nodiscard]] int find(const char* sym) const noexcept;

    // BUG-TR10 FIX: bounded spin — yield after 1000 iterations.
    // Was: while(lock_.test_and_set(...));  — unbounded live-lock risk.
    void spinLock() const noexcept {
        for (int sp = 0; lock_.test_and_set(std::memory_order_acquire); ++sp)
            if (sp > 1000) std::this_thread::yield();
    }
    void spinUnlock() const noexcept { lock_.clear(std::memory_order_release); }
};

[[nodiscard]] PortfolioEngine& globalPortfolio() noexcept;

// ── 3. Position Manager ───────────────────────────────────────────────────────

struct BrainPosition {
    char   symbol[16]{};
    char   side[8]{};
    float  entry_price = 0.0f;
    float  margin_usd  = 0.0f;
    double open_time   = 0.0;
    bool   active      = false;
};

class PositionManager {
public:
    PositionManager() noexcept;
    ~PositionManager() noexcept = default;

    PositionManager(const PositionManager&) = delete;
    PositionManager& operator=(const PositionManager&) = delete;

    void openSim (const char* symbol, const char* side,
                  float price, float margin) noexcept;
    bool closeSim(const char* symbol) noexcept;
    [[nodiscard]] bool getSim(const char* symbol,
                               BrainPosition& out) const noexcept;
    [[nodiscard]] bool isOpen(const char* symbol) const noexcept;

    void syncFromLaptop(const char* const* open_syms, int n) noexcept;
    [[nodiscard]] int count() const noexcept;

private:
    static constexpr int MAX_POS   = 8;
    static constexpr int MAX_AGE_S = 600;

    std::array<BrainPosition, MAX_POS> pos_{};
    mutable std::atomic_flag lock_ = ATOMIC_FLAG_INIT;

    std::atomic<bool> save_needed_{false};
    std::jthread      saver_;

    // BUG-TR10 FIX: bounded spin — CRITICAL for PositionManager.
    // saveToDisk_impl() holds lock_ via SpinGuard during disk I/O (can be
    // 10–100 ms on HuggingFace cold filesystem).  An unbounded spinLock()
    // in openSim/closeSim/syncFromLaptop would busy-spin an HTTP worker
    // thread for the full disk I/O duration — stalling every concurrent request.
    void spinLock() const noexcept {
        for (int sp = 0; lock_.test_and_set(std::memory_order_acquire); ++sp)
            if (sp > 1000) std::this_thread::yield();
    }
    void spinUnlock() const noexcept { lock_.clear(std::memory_order_release); }

    void requestSave() noexcept {
        save_needed_.store(true, std::memory_order_release);
    }

    void saveToDisk_impl() const noexcept;
    void loadFromDisk() noexcept;
    [[nodiscard]] int findSlot(const char* sym) const noexcept;
};

[[nodiscard]] PositionManager& globalPositionManager() noexcept;

// ── 4. Latency Tracker ────────────────────────────────────────────────────────

class LatencyTracker {
public:
    float record(std::chrono::steady_clock::time_point start) noexcept;
    [[nodiscard]] LatencyStats stats() const noexcept;

private:
    static constexpr int WINDOW = 500;

    mutable std::atomic_flag lock_ = ATOMIC_FLAG_INIT;
    float samples_[WINDOW]{};
    int   head_  = 0;
    int   count_ = 0;

    // BUG-TR10 FIX: bounded spin.
    void spinLock() const noexcept {
        for (int sp = 0; lock_.test_and_set(std::memory_order_acquire); ++sp)
            if (sp > 1000) std::this_thread::yield();
    }
    void spinUnlock() const noexcept { lock_.clear(std::memory_order_release); }
};

[[nodiscard]] LatencyTracker& globalLatency() noexcept;

[[nodiscard]] float getDailyStart(float current_balance) noexcept;

} // namespace tqc
