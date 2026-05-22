#pragma once
/*
 * alpha_engine.hpp  -  BigBoyAgent TQC Brain | Taha Iqbal
 *
 * Cross-sectional alpha factors applied across the full signal universe.
 * Factors: momentum rank, low-vol anomaly, volume surge, quality (Hurst+ADX).
 * All computation uses stack arrays — zero heap allocation.
 *
 * PRECONDITION: sigs.size() <= 24 (MAX_SYMBOLS from bar_store.hpp).
 * Enforced by assert() inside applyAlphaToSignals — checked in debug builds.
 */

#include "types.hpp"
#include <span>

namespace tqc {

// Apply composite alpha to all signals in-place.
// Adjusts confidence scores based on cross-sectional rankings.
// alpha_weight: blend fraction [0, 1] — default 0.20 (20% alpha modulation).
// PRECONDITION: sigs.size() <= 24.
void applyAlphaToSignals(std::span<Signal> signals,
                          float alpha_weight = 0.20f) noexcept;

} // namespace tqc
