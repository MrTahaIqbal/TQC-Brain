#pragma once
/*
 * risk_engine.hpp  -  BigBoyAgent TQC Brain | Taha Iqbal
 *
 * Institutional position sizing: Kelly criterion + Historical VaR +
 * volatility targeting.  All state in fixed-size circular buffers — no heap.
 *
 * Position sizing pipeline (calculatePosition):
 *   Step 1: Kelly fraction from win-rate history
 *   Step 2: Volatility scalar (vol-targeting toward 1.5% daily vol)
 *   Step 3: VaR scalar (95th-percentile loss caps size)
 *   Step 4: Autonomous tier selection (confidence + HMM + GARCH + funding)
 *   Step 5: Final risk % = Kelly × vol_scalar × var_scalar (capped by tier)
 *   Step 6: ATR-based stop-loss distance (1.5 × ATR14)
 *   Step 7: Margin = risk_usd / (leverage × sl_pct)
 *   Step 8: TP = SL × RR ratio (default 2:1)
 *
 * Thread safety:
 *   All mutable state (g_syms, g_equity, g_eq_head, g_eq_cnt) is protected
 *   by g_risk_mu (std::mutex).  findOrCreate() acquires and releases g_risk_mu
 *   independently; callers must NOT hold g_risk_mu when calling it.
 *   findOnly() requires the caller to already hold g_risk_mu.
 *
 * BUG-RK3 (Calmar annualisation) is fixed in this version:
 *   getStats() used 3650.0f (10 bars/day) for Calmar while sharpe() and
 *   sortino() had already been corrected to 525600.0f (1-min bars, BR-01 fix).
 *   Calmar now uses 525600.0f consistently.
 *
 * FIX (XC-10): confidence is an explicit parameter (not stuffed into Features).
 *   The old pattern violated const-correctness at the call site.
 */

#include "types.hpp"

namespace tqc {

// Per-symbol performance statistics.
[[nodiscard]] TradeStats getStats(const char* symbol) noexcept;

// Portfolio-level aggregated performance.
void getPortfolioPerformance(float& out_pnl, float& out_wr,
                              float& out_sharpe, float& out_sortino,
                              float& out_mdd) noexcept;

// Record a closed trade (updates circular return buffers).
void recordTrade(const char* symbol, float net_pnl,
                 float entry_size, float balance) noexcept;

// Full position sizing — returns a PositionParams with margin, SL, TP.
[[nodiscard]] PositionParams calculatePosition(
    float             balance,
    float             entry_price,
    const Features&   features,
    const RegimeInfo& regime,
    SignalDir         dir,
    float             confidence) noexcept;

// Portfolio-level risk gate.
// NOTE: `symbol` is accepted for API symmetry and future per-symbol tracking
// but is not used in the current implementation — all checks are portfolio-level.
// Per-symbol daily drawdown tracking is the caller's (main.cpp) responsibility.
[[nodiscard]] RiskCheckResult checkRisk(
    const char* symbol,
    float       balance,
    int         current_positions,
    float       daily_start_bal) noexcept;

} // namespace tqc
