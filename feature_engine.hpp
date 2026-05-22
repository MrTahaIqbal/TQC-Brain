#pragma once
/*
 * feature_engine.hpp  -  BigBoyAgent TQC Brain | Taha Iqbal
 * ============================================================
 * Computes all technical indicators for one symbol per cycle.
 *
 * Pipeline:
 *   updateHistory()    — push one new bar into BarStore (called every cycle)
 *   bulkLoad()         — warm-start with N historical bars (called once)
 *   generateFeatures() — run all SIMD indicator kernels, return Features
 *
 * Design guarantees:
 *   - Zero heap allocation in any of these functions.
 *   - All scratch work uses thread_local g_scratch (bar_store.hpp).
 *   - g_bars is the global BarStore, shared across all translation units.
 *
 * Thread safety model:
 *   - updateHistory() and bulkLoad() are fully thread-safe: they acquire
 *     SymbolBars::push_lock_ internally before modifying bar arrays.
 *   - generateFeatures() is thread-safe per the same spinlock:
 *       (a) bar_count() acquires push_lock_ to read count atomically.
 *       (b) linearise_all() acquires push_lock_ for the full array copy.
 *     Two calls for DIFFERENT symbols are fully concurrent (per-symbol lock).
 *     Two calls for the SAME symbol are serialised at the spinlock.
 *
 * BUG-FH1 FIX: The previous comment said "thread-safe: no lock needed —
 *   we use the pre-looked-up index."  Both halves were wrong:
 *   (a) generateFeatures() calls g_bars.find() INTERNALLY — no index is
 *       pre-looked-up by the caller.
 *   (b) A lock IS needed: bars.count was read without push_lock_, a data
 *       race with any concurrent push() on the same symbol.  Fixed in
 *       feature_engine.cpp: bar_count() is now used instead.
 *
 * Indicator set (9 signal inputs + regime/risk metadata):
 *   Z-score (50-bar), Bollinger position (20-bar), RSI-14 (Wilder),
 *   VW-RSI-14 (volume-normalised, Wilder), MACD (12/26/9),
 *   Stochastic K/D (14/3), ROC-5/10/20, ATR-14, VWAP,
 *   volume imbalance, order-book imbalance, SMA-20/50, TF agreement,
 *   HAR-RV, GARCH(1,1), TSMOM, ARIMA(1,1,1).
 * ============================================================
 */

#include "types.hpp"
#include "bar_store.hpp"
#include <cstddef>
#include <cstring>

namespace tqc {

// ── Global bar store ──────────────────────────────────────────────────────────
// Defined in feature_engine.cpp. Shared by regime_engine, signal_engine, etc.
extern BarStore g_bars;

// ── Core API ──────────────────────────────────────────────────────────────────

// Push one new OHLCV bar into BarStore for a symbol.
// Called every cycle before generateFeatures().
// ob_imb: order-book bid/ask size imbalance for this bar = (bid − ask)/(bid + ask).
//         Pass 0.0f when L2 WebSocket data is not yet available (neutral, not suppressing).
void updateHistory(const char* symbol, float price, float volume,
                   float high, float low, float ob_imb = 0.0f) noexcept;

// Warm-start: replace ring buffer history with N historical bars.
// Called once per symbol on the first /predict request (bulk_prices field).
void bulkLoad(const char* symbol,
              const float* closes,
              const float* highs,
              const float* lows,
              const float* volumes,
              std::size_t  n) noexcept;

// Compute all technical features for a symbol.
// Thread-safe: internally acquires SymbolBars::push_lock_ twice —
//   once for bar_count() and once for linearise_all().
// Two concurrent calls for different symbols run fully in parallel.
// Two concurrent calls for the same symbol serialise at the spinlock.
[[nodiscard]] Features generateFeatures(
    const char*  symbol,
    float        last_price,
    float        volume,
    float        high,
    float        low,
    const float* prices_1m, std::size_t n1m,   // optional 1-min bar prices
    const float* prices_5m, std::size_t n5m    // optional 5-min bar prices
) noexcept;

// ── main.cpp compatibility shim ───────────────────────────────────────────────
// Accepts raw bid/ask sizes and computes OB imbalance before forwarding.
// bid_sz / ask_sz: total resting size at top-N levels of the order book.
// When L2 WebSocket data is not available, pass 0.0f / 0.0f — updateHistory
// will store 0.0 (neutral) for that bar without suppressing any indicator.
//
// The four-initializer-list form in the original call site:
//   UpdateHistory(sym, price, vol, &h, &l, {}, {}, {}, {})
// is preserved as an overload below so existing call sites compile unchanged.

inline void UpdateHistory(
    const char* sym, float price, float vol,
    const float* h, const float* l,
    float bid_sz, float ask_sz) noexcept
{
    const float hi    = h ? *h : price;
    const float lo    = l ? *l : price;
    const float total = bid_sz + ask_sz;
    const float ob_imb = (total > 1e-10f) ? (bid_sz - ask_sz) / total : 0.0f;
    updateHistory(sym, price, vol, hi, lo, ob_imb);
}

// Backward-compatible overload used by existing call sites that pass empty
// initializer_lists — maps to neutral OB imbalance (0.0f).
inline void UpdateHistory(
    const char* sym, float price, float vol,
    const float* h, const float* l,
    std::initializer_list<float>,   // bid levels (unused until WS depth live)
    std::initializer_list<float>,   // ask levels
    std::initializer_list<float>,   // reserved
    std::initializer_list<float>) noexcept
{
    const float hi = h ? *h : price;
    const float lo = l ? *l : price;
    updateHistory(sym, price, vol, hi, lo, 0.0f);
}

[[nodiscard]] inline Features GenerateFeatures(
    const char* sym, float price, float vol,
    const float* h, const float* l,
    const float* p1m, std::size_t n1m,
    const float* p5m, std::size_t n5m) noexcept
{
    const float hi = h ? *h : price;
    const float lo = l ? *l : price;
    return generateFeatures(sym, price, vol, hi, lo, p1m, n1m, p5m, n5m);
}

// ── Step 2: Payload feature injection ────────────────────────────────────────
// Called in main.cpp's processSymbol() AFTER GenerateFeatures() returns.
// Populates the four fields that come from the executor's real-time
// WebSocket/REST payload (not bar history):
//   vwoi           — aggTrade buyer/seller imbalance [-1, +1]
//   funding_rate   — Binance 8hr perpetual funding rate (raw)
//   funding_zscore — z-score vs 30-bar rolling window of funding_rate history
//   basis          — (mark_price - index_price) / index_price
//
// funding_zscore uses a per-symbol rolling buffer maintained here.
// Thread-safe: per-symbol atomic_flag (BUG-FE2 FIX — was one global flag).
void injectPayloadFeatures(const char* symbol,
                            float vwoi,
                            float funding_rate,
                            float basis,
                            Features& f) noexcept;

} // namespace tqc
