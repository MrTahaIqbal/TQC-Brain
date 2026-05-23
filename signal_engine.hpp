#pragma once
/*
 * signal_engine.hpp  -  TQC Brain | Taha Iqbal
 *
 * Converts normalised features + regime into a BUY/SELL/HOLD signal.
 *
 * Adaptive weights: per-indicator accuracy is tracked via lock-free
 * atomics and used to scale the base regime weights at runtime.
 *
 * Post-trade feedback loop:
 *   updateAdaptiveWeights() is called after every simulated trade close.
 *   It records whether each indicator's direction agreed with the outcome.
 *
 * Thread safety:
 *   generateSignal()         — fully re-entrant; internal norm cache
 *                              write protected by per-slot atomic_flag.
 *   recordIndicators()       — protected by same per-slot atomic_flag.
 *   updateAdaptiveWeights()  — protected by same per-slot atomic_flag;
 *                              AccuracyEntry updates are lock-free CAS.
 *   findSymSlot()            — double-check locking via shared_mutex.
 *
 * BUG-SE2 / BUG-SE8 FIX: g_last_norm[] was written by generateSignal()
 *   and recordIndicators(), and read by updateAdaptiveWeights(), with no
 *   protection beyond the g_norm_valid acquire/release flag.  That pattern
 *   is only safe for a SINGLE writer; multiple HTTP threads can call
 *   generateSignal() for the same symbol simultaneously, creating a torn
 *   write race on g_last_norm[slot] (plain std::array, not atomic).
 *   Fixed with per-slot atomic_flag g_norm_lock[MAX_SYMS_ADAPT] that
 *   serialises all three accesses to the norm cache.
 */

#include "types.hpp"
#include "config.hpp"
#include <array>

namespace tqc {

// Generate a trading signal for a symbol.
[[nodiscard]] SignalResult generateSignal(const Features&   features,
                                           const RegimeInfo& regime) noexcept;

// Store the last normalised indicator scores for a symbol.
// Called automatically inside generateSignal(); exposed for testing.
void recordIndicators(const char* symbol,
                      const std::array<float, N_INDICATORS>& norm) noexcept;

// Update adaptive weights based on whether the last trade was a win.
// Called after every simulated trade close in main.cpp.
void updateAdaptiveWeights(const char* symbol, bool trade_won) noexcept;

} // namespace tqc
