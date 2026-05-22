/*
 * tracking.cpp  -  BigBoyAgent TQC Brain | Taha Iqbal
 *
 * ── BUGS FIXED IN THIS VERSION ──────────────────────────────────────────────
 *
 * BUG-TR1  Analytics::findOrCreate() fast path: strncmp(15) misses byte 15.
 *          FIX: strncmp(16) — compare full 16-byte name field.
 *          (Same class of bug as BUG-RK1, BUG-SE6, BUG-AE4.)
 *
 * BUG-TR2  Analytics::findOrCreate() double-check: strncmp(15).
 *          FIX: strncmp(16).
 *
 * BUG-TR3  Analytics::find(): strncmp(15).
 *          FIX: strncmp(16).
 *
 * BUG-TR4  computeMetrics(): Sharpe annualisation uses 3650.0f.
 *          For 1-min bars: 365 × 24 × 60 = 525,600 bars/year.
 *          3650.0f implies 10 bars/day — off by 144×.
 *          FIX: BARS_PER_YEAR = 525600.0f  (consistent with risk_engine.cpp).
 *
 * BUG-TR5  computeMetrics(): Sortino denominator uses nn (count of negatives)
 *          instead of n (total), AND uses mean-centred std-dev of the negative
 *          subsample instead of the correct full-sample downside deviation.
 *
 *          simd::std_dev(neg_rets, nn) computes:
 *            sqrt( Σ(r - mean_neg)² / nn )
 *          The Sortino formula requires:
 *            sqrt( Σ min(r, 0)² / n_total )
 *          Two errors:
 *            (a) divides by nn (negatives count), not n (total)
 *            (b) subtracts mean_neg instead of 0 (the target return threshold)
 *          Both inflate dsd, producing a systematically understated Sortino.
 *          The effect compounds with BUG-TR4 (wrong annualisation factor).
 *          FIX: direct downside-squared accumulation divided by n_total —
 *          identical fix to BUG-RK4 in risk_engine.cpp.
 *          Also removed the dead neg_rets[MAX_HIST] stack array (redundant
 *          after the fix, same as the (void)neg situation in risk_engine.cpp).
 *
 * BUG-TR6  computeMetrics(): Calmar annualisation uses 3650.0f.
 *          Same as BUG-TR4 — understated by 144×.
 *          FIX: BARS_PER_YEAR = 525600.0f.
 *
 * BUG-TR7  PortfolioEngine::find(): strncmp(15).
 *          FIX: strncmp(16).
 *
 * BUG-TR8  PositionManager::findSlot(): strncmp(15).
 *          FIX: strncmp(16).
 *
 * BUG-TR9  PositionManager::syncFromLaptop(): strncmp(15).
 *          FIX: strncmp(16).
 *
 * BUG-TR10 All three spinLock() methods: unbounded busy-spin.
 *          Documented and fixed in tracking.hpp — yield after 1000 iterations.
 *          Critical for PositionManager::spinLock(): saveToDisk_impl() holds
 *          lock_ via SpinGuard during disk I/O (10–100 ms on cold filesystem).
 *          An unbounded spin on an HTTP worker thread stalls every request.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include "tracking.hpp"
#include "config.hpp"
#include "simd_math.hpp"
#include "json.hpp"

#include <cstring>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <thread>
#include <bit>      // std::bit_cast (C++20)

using json = nlohmann::json;

namespace tqc {

// Annualisation for 1-min bars: 365 × 24 × 60 = 525,600.
// BUG-TR4/TR6 FIX: was 3650.0f (≈ 10 bars/day) — off by 144×.
static constexpr float BARS_PER_YEAR = 525600.0f;

// ── RAII spinlock guard ───────────────────────────────────────────────────────
struct SpinGuard {
    std::atomic_flag& flag;
    explicit SpinGuard(std::atomic_flag& f) noexcept : flag(f) {
        for (int sp = 0; flag.test_and_set(std::memory_order_acquire); ++sp)
            if (sp > 1000) std::this_thread::yield();
    }
    ~SpinGuard() noexcept { flag.clear(std::memory_order_release); }
    SpinGuard(const SpinGuard&) = delete;
};

// ════════════════════════════════════════════════════════════════════════════
// 1. Analytics
// ════════════════════════════════════════════════════════════════════════════

static Analytics g_analytics{};
Analytics& globalAnalytics() noexcept { return g_analytics; }

int Analytics::findOrCreate(const char* sym) noexcept {
    // Fast path — no lock; slot names are immutable once written.
    const int n = nsyms_.load(std::memory_order_acquire);
    for (int i = 0; i < n; ++i)
        // BUG-TR1 FIX: compare 16 bytes (was 15 — missed null-terminator byte).
        if (std::strncmp(syms_[i].name, sym, 16) == 0) return i;

    // Slow path — double-checked registration under reg_lock_.
    for (int sp = 0; reg_lock_.test_and_set(std::memory_order_acquire); ++sp)
        if (sp > 1000) std::this_thread::yield();

    const int n2 = nsyms_.load(std::memory_order_relaxed);
    for (int i = 0; i < n2; ++i) {
        // BUG-TR2 FIX: compare 16 bytes.
        if (std::strncmp(syms_[i].name, sym, 16) == 0) {
            reg_lock_.clear(std::memory_order_release);
            return i;
        }
    }
    if (n2 >= MAX_SYMS) {
        reg_lock_.clear(std::memory_order_release);
        return -1;
    }
    std::strncpy(syms_[n2].name, sym, 15);
    syms_[n2].name[15] = '\0';
    nsyms_.store(n2 + 1, std::memory_order_release);
    reg_lock_.clear(std::memory_order_release);
    return n2;
}

int Analytics::find(const char* sym) const noexcept {
    const int n = nsyms_.load(std::memory_order_acquire);
    for (int i = 0; i < n; ++i)
        // BUG-TR3 FIX: compare 16 bytes.
        if (std::strncmp(syms_[i].name, sym, 16) == 0) return i;
    return -1;
}

void Analytics::logTrade(const char* sym, float pnl,
                          float bal_after, float size) noexcept {
    const int idx = findOrCreate(sym);
    if (idx < 0) [[unlikely]] return;
    const float ret = (size > 1e-10f) ? pnl / size : 0.0f;
    syms_[idx].push(pnl, ret, bal_after);
}

Analytics::Metrics Analytics::computeMetrics(const char* sym,
                                              int window) const noexcept {
    const int idx = find(sym);
    if (idx < 0) return {};

    const auto& s = syms_[idx];
    const std::size_t n = (window > 0)
                          ? std::min(static_cast<std::size_t>(window), s.count)
                          : s.count;
    if (n == 0) return {};

    static thread_local float pnl_buf[MAX_HIST];
    static thread_local float ret_buf[MAX_HIST];
    static thread_local float eq_buf [MAX_HIST];
    s.linearise(pnl_buf, ret_buf, eq_buf, n);

    float sum_pos = 0.0f, sum_neg = 0.0f;
    int   n_wins  = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (pnl_buf[i] > 0.0f) { sum_pos += pnl_buf[i]; ++n_wins; }
        else                    { sum_neg += std::abs(pnl_buf[i]); }
    }

    Metrics m;
    m.total_trades  = static_cast<int>(n);
    m.net_pnl_usd   = std::round(simd::sum(pnl_buf, n) * 10000.0f) / 10000.0f;
    m.win_rate      = static_cast<float>(n_wins) / static_cast<float>(n);
    m.profit_factor = (sum_neg > 1e-10f) ? sum_pos / sum_neg : 99.0f;
    m.avg_pnl       = simd::mean(pnl_buf, n);

    // BUG-TR4 FIX: Sharpe uses BARS_PER_YEAR (525600), not 3650.
    const float sd = simd::std_dev(ret_buf, n);
    m.sharpe = (sd > 1e-10f)
               ? simd::mean(ret_buf, n) / sd * std::sqrt(BARS_PER_YEAR)
               : 0.0f;

    // BUG-TR5 FIX: correct full-sample downside deviation.
    // BEFORE:
    //   float neg_rets[MAX_HIST]; int nn = 0;
    //   for each r < 0: neg_rets[nn++] = r;
    //   dsd = simd::std_dev(neg_rets, nn)  ← divides by nn, subtracts mean_neg
    // Both errors inflate dsd → understated Sortino.
    //
    // AFTER: Σ min(r,0)² / n_total — the correct Sortino denominator.
    // No heap allocation, no dead array, no suppress-warning pragma.
    {
        float down_sq = 0.0f;
        for (std::size_t i = 0; i < n; ++i)
            if (ret_buf[i] < 0.0f) down_sq += ret_buf[i] * ret_buf[i];
        const float dsd = std::sqrt(down_sq / static_cast<float>(n) + 1e-20f);
        m.sortino = simd::mean(ret_buf, n) / (dsd + 1e-10f) * std::sqrt(BARS_PER_YEAR);
    }

    float peak = eq_buf[0], mdd = 0.0f;
    for (std::size_t i = 1; i < n; ++i) {
        if (eq_buf[i] > peak) peak = eq_buf[i];
        const float dd = (peak - eq_buf[i]) / (peak + 1e-10f);
        if (dd > mdd) mdd = dd;
    }
    m.max_drawdown = mdd;

    // BUG-TR6 FIX: Calmar uses BARS_PER_YEAR (525600), not 3650.
    // Calmar = mean_bar_return × bars_per_year / max_drawdown.
    // With 3650: annual return understated 144×; every symbol looked unprofitable.
    const float ann = simd::mean(ret_buf, n) * BARS_PER_YEAR;
    m.calmar = (mdd > 1e-10f) ? ann / mdd : 0.0f;

    return m;
}

int Analytics::allMetricsJSON(char* buf, int bufsz) const noexcept {
    const int n = nsyms_.load(std::memory_order_acquire);
    int written = 0;
    written += std::snprintf(buf + written, bufsz - written, "[");
    for (int i = 0; i < n && written < bufsz - 4; ++i) {
        if (i > 0) written += std::snprintf(buf + written, bufsz - written, ",");
        const Metrics all = computeMetrics(syms_[i].name, 0);
        const Metrics r50 = computeMetrics(syms_[i].name, 50);
        written += std::snprintf(buf + written, bufsz - written,
            "{"
            "\"symbol\":\"%s\","
            "\"all_time\":{\"trades\":%d,\"net_pnl\":%.4f,\"win_rate\":%.4f,"
            "\"sharpe\":%.3f,\"sortino\":%.3f,\"max_drawdown\":%.4f},"
            "\"recent_50\":{\"trades\":%d,\"net_pnl\":%.4f,\"win_rate\":%.4f,"
            "\"sharpe\":%.3f}"
            "}",
            syms_[i].name,
            all.total_trades, all.net_pnl_usd, all.win_rate,
            all.sharpe, all.sortino, all.max_drawdown,
            r50.total_trades, r50.net_pnl_usd, r50.win_rate, r50.sharpe);
    }
    written += std::snprintf(buf + written, bufsz - written, "]");
    return written;
}

// ════════════════════════════════════════════════════════════════════════════
// 2. Portfolio Engine
// ════════════════════════════════════════════════════════════════════════════

static PortfolioEngine g_portfolio{};
PortfolioEngine& globalPortfolio() noexcept { return g_portfolio; }

int PortfolioEngine::find(const char* sym) const noexcept {
    for (int i = 0; i < MAX_POS; ++i)
        if (positions_[i].active &&
            // BUG-TR7 FIX: compare 16 bytes (was 15).
            std::strncmp(positions_[i].symbol, sym, 16) == 0)
            return i;
    return -1;
}

void PortfolioEngine::openPosition(const char* sym, const char* side,
                                    float price, float margin_usd) noexcept {
    spinLock();
    if (find(sym) >= 0) { spinUnlock(); return; }

    int slot = -1;
    for (int i = 0; i < MAX_POS; ++i)
        if (!positions_[i].active) { slot = i; break; }
    if (slot < 0) { spinUnlock(); return; }

    const auto& cfg  = globalConfig();
    const bool is_long = (std::strncmp(side, "LONG", 4) == 0 ||
                          std::strncmp(side, "BUY",  4) == 0);
    const float slip    = cfg.slip_pct;
    const float entry   = is_long ? price * (1.0f + slip)
                                  : price * (1.0f - slip);
    const float notional = margin_usd * static_cast<float>(cfg.leverage);
    const float qty      = notional / (entry + 1e-10f);
    const float fee_open = notional * cfg.fee_pct;

    auto& p = positions_[slot];
    std::strncpy(p.symbol, sym, 15);  p.symbol[15] = '\0';
    std::strncpy(p.side, is_long ? "LONG" : "SHORT", 7); p.side[7] = '\0';
    p.entry_price = entry;
    p.margin_usd  = margin_usd;
    p.notional    = notional;
    p.quantity    = qty;
    p.fee_open    = fee_open;
    p.active      = true;
    spinUnlock();
}

PortfolioEngine::CloseResult
PortfolioEngine::closePosition(const char* sym, float current_price) noexcept {
    spinLock();
    const int i = find(sym);
    if (i < 0) { spinUnlock(); return {}; }
    const SimPosition pos = positions_[i];
    positions_[i].active = false;
    spinUnlock();

    const auto& cfg  = globalConfig();
    const bool is_long = (std::strncmp(pos.side, "LONG", 4) == 0);
    const float slip    = cfg.slip_pct;
    const float exit_px = is_long ? current_price * (1.0f - slip)
                                  : current_price * (1.0f + slip);
    const float gross   = is_long
                          ? (exit_px - pos.entry_price) * pos.quantity
                          : (pos.entry_price - exit_px) * pos.quantity;
    const float fee_close = pos.notional * cfg.fee_pct;
    const float net       = gross - pos.fee_open - fee_close;

    return {net, gross, pos.fee_open + fee_close, exit_px, pos.margin_usd, true};
}

float PortfolioEngine::getUnrealizedPnL(const char* sym,
                                         float px) const noexcept {
    spinLock();
    const int i = find(sym);
    if (i < 0) { spinUnlock(); return 0.0f; }
    const SimPosition pos = positions_[i];
    spinUnlock();

    const bool is_long = (std::strncmp(pos.side, "LONG", 4) == 0);
    const float gross = is_long
                        ? (px - pos.entry_price) * pos.quantity
                        : (pos.entry_price - px) * pos.quantity;
    return gross - pos.fee_open;
}

bool PortfolioEngine::isOpen(const char* sym) const noexcept {
    spinLock();
    const bool r = (find(sym) >= 0);
    spinUnlock();
    return r;
}

// ════════════════════════════════════════════════════════════════════════════
// 3. Position Manager
// ════════════════════════════════════════════════════════════════════════════

static PositionManager g_pos_mgr{};
PositionManager& globalPositionManager() noexcept { return g_pos_mgr; }

static const char* const POS_FILE = "/tmp/brain_state/positions.json";

PositionManager::PositionManager() noexcept
    : saver_([this](std::stop_token st) {
        while (!st.stop_requested()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            if (save_needed_.exchange(false, std::memory_order_acq_rel))
                saveToDisk_impl();
        }
        if (save_needed_.load(std::memory_order_acquire))
            saveToDisk_impl();
    })
{
    loadFromDisk();
}

int PositionManager::findSlot(const char* sym) const noexcept {
    for (int i = 0; i < MAX_POS; ++i)
        if (pos_[i].active &&
            // BUG-TR8 FIX: compare 16 bytes (was 15).
            std::strncmp(pos_[i].symbol, sym, 16) == 0)
            return i;
    return -1;
}

void PositionManager::openSim(const char* sym, const char* side,
                               float price, float margin) noexcept {
    spinLock();
    if (findSlot(sym) >= 0) { spinUnlock(); return; }
    int slot = -1;
    for (int i = 0; i < MAX_POS; ++i)
        if (!pos_[i].active) { slot = i; break; }
    if (slot < 0) { spinUnlock(); return; }

    const bool is_long = (std::strncmp(side, "BUY",  4) == 0 ||
                          std::strncmp(side, "LONG", 4) == 0);
    auto& p = pos_[slot];
    std::strncpy(p.symbol, sym, 15);                         p.symbol[15] = '\0';
    std::strncpy(p.side, is_long ? "LONG" : "SHORT", 7);    p.side[7]    = '\0';
    p.entry_price = price;
    p.margin_usd  = margin;
    p.open_time   = static_cast<double>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    p.active = true;
    spinUnlock();
    requestSave();
}

bool PositionManager::closeSim(const char* sym) noexcept {
    spinLock();
    const int i = findSlot(sym);
    if (i < 0) { spinUnlock(); return false; }
    pos_[i].active = false;
    spinUnlock();
    requestSave();
    return true;
}

bool PositionManager::getSim(const char* sym, BrainPosition& out) const noexcept {
    spinLock();
    const int i = findSlot(sym);
    if (i >= 0) out = pos_[i];
    spinUnlock();
    return (i >= 0);
}

bool PositionManager::isOpen(const char* sym) const noexcept {
    spinLock();
    const bool r = (findSlot(sym) >= 0);
    spinUnlock();
    return r;
}

int PositionManager::count() const noexcept {
    spinLock();
    int c = 0;
    for (int i = 0; i < MAX_POS; ++i) if (pos_[i].active) ++c;
    spinUnlock();
    return c;
}

void PositionManager::syncFromLaptop(const char* const* syms, int n) noexcept {
    spinLock();
    for (int i = 0; i < MAX_POS; ++i) {
        if (!pos_[i].active) continue;
        bool found = false;
        for (int j = 0; j < n; ++j)
            // BUG-TR9 FIX: compare 16 bytes (was 15).
            if (syms[j] && std::strncmp(pos_[i].symbol, syms[j], 16) == 0) {
                found = true; break;
            }
        if (!found) pos_[i].active = false;
    }
    spinUnlock();
    requestSave();
}

void PositionManager::saveToDisk_impl() const noexcept {
    try {
        json arr = json::array();
        {
            SpinGuard g(lock_);
            for (int i = 0; i < MAX_POS; ++i) {
                if (!pos_[i].active) continue;
                arr.push_back({
                    {"symbol",      pos_[i].symbol},
                    {"side",        pos_[i].side},
                    {"entry_price", pos_[i].entry_price},
                    {"margin_usd",  pos_[i].margin_usd},
                    {"open_time",   pos_[i].open_time},
                });
            }
        }
        const double ts = static_cast<double>(
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
        const json doc = {{"positions", arr}, {"ts", ts}};
        std::ofstream f(POS_FILE);
        if (f.is_open()) f << doc.dump();
    } catch (...) {}
}

void PositionManager::loadFromDisk() noexcept {
    try {
        std::ifstream f(POS_FILE);
        if (!f.is_open()) return;
        json doc; f >> doc;

        const double now = static_cast<double>(
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
        const double ts = doc.value("ts", 0.0);
        if (now - ts > MAX_AGE_S) return;

        SpinGuard g(lock_);
        int slot = 0;
        for (auto& p : doc["positions"]) {
            if (slot >= MAX_POS) break;
            const std::string sym  = p.value("symbol", "");
            const std::string side = p.value("side",   "LONG");
            std::strncpy(pos_[slot].symbol, sym.c_str(),  15);
            std::strncpy(pos_[slot].side,   side.c_str(),  7);
            pos_[slot].symbol[15] = '\0';
            pos_[slot].side[7]    = '\0';
            pos_[slot].entry_price = static_cast<float>(p.value("entry_price", 0.0));
            pos_[slot].margin_usd  = static_cast<float>(p.value("margin_usd",  0.0));
            pos_[slot].open_time   = p.value("open_time", 0.0);
            pos_[slot].active      = true;
            ++slot;
        }
    } catch (...) {}
}

// ════════════════════════════════════════════════════════════════════════════
// 4. Latency Tracker
// ════════════════════════════════════════════════════════════════════════════

static LatencyTracker g_latency{};
LatencyTracker& globalLatency() noexcept { return g_latency; }

float LatencyTracker::record(std::chrono::steady_clock::time_point start) noexcept {
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start).count();
    const float ms = static_cast<float>(us) / 1000.0f;

    spinLock();
    samples_[head_ % WINDOW] = ms;
    ++head_;
    if (count_ < WINDOW) ++count_;
    spinUnlock();

    return std::round(ms * 10000.0f) / 10000.0f;
}

LatencyStats LatencyTracker::stats() const noexcept {
    spinLock();
    const int n    = count_;
    const int head = head_;
    static thread_local float tmp[WINDOW];
    for (int i = 0; i < n; ++i)
        tmp[i] = samples_[(head - 1 - (n - 1 - i) + WINDOW) % WINDOW];
    spinUnlock();

    if (n == 0) return {};

    LatencyStats s;
    s.samples = n;
    s.avg_ms  = std::round(simd::mean(tmp, n) * 10000.0f) / 10000.0f;
    s.max_ms  = std::round(simd::vmax(tmp, n) * 10000.0f) / 10000.0f;

    // percentile() uses nth_element which partitions in-place.
    // Must copy tmp → ptmp before each call (p95 corrupts the array for p99).
    static thread_local float ptmp[WINDOW];
    std::copy(tmp, tmp + n, ptmp);
    s.p95_ms = std::round(simd::percentile(ptmp, n, 95.0f) * 10000.0f) / 10000.0f;
    std::copy(tmp, tmp + n, ptmp);
    s.p99_ms = std::round(simd::percentile(ptmp, n, 99.0f) * 10000.0f) / 10000.0f;
    return s;
}

// ════════════════════════════════════════════════════════════════════════════
// 5. Daily start balance
// ════════════════════════════════════════════════════════════════════════════

float getDailyStart(float balance) noexcept {
    static std::atomic<uint32_t> daily_bits{0};
    static std::atomic<int>      daily_day{-1};

    const time_t now_t = std::time(nullptr);
    struct tm    utc{};
    gmtime_r(&now_t, &utc);   // POSIX — thread-safe (no shared static struct)
    const int today = utc.tm_yday + utc.tm_year * 400;

    int stored_day = daily_day.load(std::memory_order_acquire);
    if (stored_day != today) {
        if (daily_day.compare_exchange_strong(stored_day, today,
                std::memory_order_release, std::memory_order_acquire)) {
            daily_bits.store(std::bit_cast<uint32_t>(balance),
                             std::memory_order_release);
        }
    }

    const uint32_t bits = daily_bits.load(std::memory_order_acquire);
    const float ds = std::bit_cast<float>(bits);
    return (ds > 0.0f) ? ds : balance;
}

} // namespace tqc
