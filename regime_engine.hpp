#pragma once
/*
 * regime_engine.hpp  -  TQC Brain | Taha Iqbal
 *
 * Market regime classification: TRENDING / MEAN_REVERTING / NOISE.
 *
 * Uses three independent signals:
 *   1. Hurst exponent (R/S analysis)     — memory / persistence of series
 *   2. ADX (Average Directional Index)   — trend strength
 *   3. Regression slope (20-bar linear)  — current directional momentum
 *   4. Volatility percentile             — vol regime gate (HIGH kills TRENDING)
 *
 * On crypto markets, empirical observation shows NOISE is the dominant
 * regime ~60-70% of the time (choppy, low-Hurst markets between momentum
 * bursts).  Branch prediction hints are set accordingly.
 *
 * Thread safety:
 *   classifyRegime() acquires SymbolBars::push_lock_ internally (via
 *   bar_count() and linearise_all()) and g_hmm_lock for HMM matrix reads.
 *   Concurrent calls for different symbols are fully parallel.
 */

#include "types.hpp"

namespace tqc {

// Classify the market regime for a symbol given its computed features.
// Uses the global g_bars store — no additional data needed.
[[nodiscard]] RegimeInfo classifyRegime(const Features& features) noexcept;

} // namespace tqc
