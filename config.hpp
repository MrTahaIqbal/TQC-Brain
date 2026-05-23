#pragma once
/*
 * config.hpp  -  TQC Brain | Taha Iqbal
 * // Note the risk parameters are set by me according to my plan structure
 * HARDWARE REQUIREMENT 3: ZERO-LATENCY LOGIC
 *
 * Template specialisation resolves regime-specific weights at compile time.
 * Instead of runtime branching, getRegimeParams() is constexpr — the
 * compiler hard-wires the correct weight array with a direct jump table.
 *
 * ── BUGS FIXED IN THIS VERSION ──────────────────────────────────────────────
 *
 * BUG-C1  AppConfig::validate() was missing entirely.
 *         
 *
 * BUG-C2  AppConfig::has_pair() iterated the full pairs array even after
 *         finding the symbol, because it used range-based iteration over
 *         all MAX_PAIRS slots rather than stopping at num_pairs.
 *         With a pair list of 20 symbols and MAX_PAIRS = 24, the last 4
 *         slots are zero-filled, so strcmp against "" never matches a real
 *         pair — correct by accident, but wasteful.
 *         FIX: loop bound changed from MAX_PAIRS-implicit to num_pairs
 *         (was already correct in the original; confirmed and documented).
 *
 * BUG-C3  CapitalTier in-class member initializers used brace-init syntax
 *        
 *
 * 
 *
 * BUG-C5  [NEW] FINDING-S5 (audit) not implemented: vol_target_pct and
 *         kelly_coldstart were hardcoded constants inside risk_engine.cpp.
 *         The audit explicitly required both to be promoted to AppConfig so
 *         they could be adjusted from settings.json without recompiling.
 *         A user trading low-volatility instruments or a differently-sized
 *         account could not tune the vol-scalar target or the cold-start
 *         Kelly fraction without a code change.  These values are as
 *         operationally significant as leverage or risk_per_trade_pct.
 *         FIX: vol_target_pct (default 0.015 = 1.5% daily vol target) and
 *         kelly_coldstart (default 0.0105 = 1.05% cold-start fraction) are
 *         now fields in AppConfig.  Both are validated in validate() and
 *         loaded from the CAPITAL_TIERS block of settings.json in
 *         config.cpp.  risk_engine.cpp must be updated to read these from
 *         globalConfig() instead of local constexpr values.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include "types.hpp"
#include <array>
#include <string_view>
#include <cstddef>
#include <cstring>
#include <cstdio>   // FILE* for validate() error output

namespace tqc {

// ── Regime weights: 9 indicators ─────────────────────────────────────────────
// Order: z_score, rsi_c, vw_rsi_c, macd, roc_5, roc_20, bb, stoch_k, vwap_d
inline constexpr std::size_t N_INDICATORS = 9;
using WeightArray = std::array<float, N_INDICATORS>;

// ── Primary template (never instantiated directly) ───────────────────────────
template<Regime R> struct RegimeWeights;

template<> struct RegimeWeights<Regime::TRENDING> {
    static constexpr WeightArray value = {
        0.12f, 0.10f, 0.12f, 0.22f, 0.14f, 0.10f, 0.08f, 0.06f, 0.06f
    };
    // Sum: 0.12+0.10+0.12+0.22+0.14+0.10+0.08+0.06+0.06 = 1.00 ✓
    static constexpr float buy_thresh  = 0.62f;
    static constexpr float sell_thresh = 0.38f;
    static constexpr float sigmoid_k   = 5.0f;
    static constexpr float conf_cap    = 0.99f;
};

template<> struct RegimeWeights<Regime::MEAN_REVERTING> {
    static constexpr WeightArray value = {
        0.22f, 0.14f, 0.10f, 0.06f, 0.06f, 0.04f, 0.22f, 0.10f, 0.06f
    };
    // Sum: 0.22+0.14+0.10+0.06+0.06+0.04+0.22+0.10+0.06 = 1.00 ✓
    static constexpr float buy_thresh  = 0.60f;
    static constexpr float sell_thresh = 0.40f;
    static constexpr float sigmoid_k   = 4.5f;
    static constexpr float conf_cap    = 0.95f;
};

template<> struct RegimeWeights<Regime::NOISE> {
    static constexpr WeightArray value = {
        0.18f, 0.12f, 0.14f, 0.12f, 0.10f, 0.06f, 0.10f, 0.10f, 0.08f
    };
    // Sum: 0.18+0.12+0.14+0.12+0.10+0.06+0.10+0.10+0.08 = 1.00 ✓
    //
    // NOTE: conf_cap (0.65) intentionally falls below buy_thresh (0.72).
    // This creates a structural double-gate: NOISE regime can never produce
    // a BUY or SELL signal because the maximum achievable confidence (0.65)
    // is less than the minimum required threshold (0.72 / 1-0.28=0.72).
    // This complements the pos_valid=false gate in app.py as defense-in-depth.
    static constexpr float buy_thresh  = 0.72f;
    static constexpr float sell_thresh = 0.28f;
    static constexpr float sigmoid_k   = 6.0f;
    static constexpr float conf_cap    = 0.65f;
};

// ── Runtime-dispatchable parameter bundle ────────────────────────────────────
struct RegimeParams {
    WeightArray weights;
    float       buy_thresh;
    float       sell_thresh;
    float       sigmoid_k;
    float       conf_cap;
};

// BUG-C4 FIX: removed redundant `inline`; constexpr is implicitly inline in C++17.
[[nodiscard]] constexpr RegimeParams getRegimeParams(Regime r) noexcept {
    switch (r) {
        case Regime::TRENDING:
            return { RegimeWeights<Regime::TRENDING>::value,
                     RegimeWeights<Regime::TRENDING>::buy_thresh,
                     RegimeWeights<Regime::TRENDING>::sell_thresh,
                     RegimeWeights<Regime::TRENDING>::sigmoid_k,
                     RegimeWeights<Regime::TRENDING>::conf_cap };
        case Regime::MEAN_REVERTING:
            return { RegimeWeights<Regime::MEAN_REVERTING>::value,
                     RegimeWeights<Regime::MEAN_REVERTING>::buy_thresh,
                     RegimeWeights<Regime::MEAN_REVERTING>::sell_thresh,
                     RegimeWeights<Regime::MEAN_REVERTING>::sigmoid_k,
                     RegimeWeights<Regime::MEAN_REVERTING>::conf_cap };
        default: // NOISE
            return { RegimeWeights<Regime::NOISE>::value,
                     RegimeWeights<Regime::NOISE>::buy_thresh,
                     RegimeWeights<Regime::NOISE>::sell_thresh,
                     RegimeWeights<Regime::NOISE>::sigmoid_k,
                     RegimeWeights<Regime::NOISE>::conf_cap };
    }
}

// ── Capital tier ──────────────────────────────────────────────────────────────
// The system selects a tier autonomously per-trade based on:
//   confidence × hmm_state × garch_vol × var95
// Tiers replace the hardcoded [0.003, 0.030] clamp in risk_engine.cpp.
// All values are fractions of account balance (e.g. 0.005 = 0.5%).
//
// BUG-C3 FIX: explicit default member values instead of brace-init in AppConfig
// to avoid compiler-specific aggregate-initialisation warnings.
struct CapitalTier {
    float risk_pct       = 0.005f;  // base risk fraction of balance
    float max_margin_pct = 0.15f;   // max margin as fraction of balance
};

// ── Global application config ─────────────────────────────────────────────────
struct AppConfig {
    static constexpr std::size_t MAX_PAIRS = 24;

    // Trading pairs
    char        pairs[MAX_PAIRS][16]{};
    std::size_t num_pairs = 0;

    // Risk parameters
    float risk_per_trade_pct = 0.0105f;
    int   leverage           = 3;
    // NOTE: was 0.80f — a dangerously large fallback that could consume 80%
    // of balance if any code path bypassed the tier selector.  Clamped to
    // 0.15f (conservative tier ceiling) so any accidental fallthrough still
    // produces the safest possible size, not a near-full-balance trade.
    float max_margin_pct     = 0.15f;
    float max_daily_drawdown = 0.05f;
    float min_notional_usd   = 20.0f;
    int   max_open_positions = 3;
    float fee_pct            = 0.0004f;  // Binance futures taker ~0.04%
    float slip_pct           = 0.0002f;  // estimated slippage per side
    float tp_rr_ratio        = 2.0f;
    float hard_floor_usd     = 15.0f;

    // ── Autonomous capital tiers ──────────────────────────────────────────────
    // Conservative: high vol / BEAR HMM / low confidence
    // Standard:     normal conditions
    // Aggressive:   high confidence / BULL HMM / low GARCH vol
    // The system selects between tiers in real time — no human input required.
    //
    // BUG-C3 FIX: using named field assignment instead of aggregate brace-init
    // to avoid MSVC/older-Clang "initializer-list for non-aggregate" warnings.
    CapitalTier tier_conservative;  // populated in AppConfig() constructor below
    CapitalTier tier_standard;
    CapitalTier tier_aggressive;

    // Hard bounds — the Kelly/VaR output is clamped to these regardless of tier.
    // These are the only remaining hardcoded limits, and they're now in config.
    float min_risk_pct  = 0.003f;  // floor:   never risk less than 0.3%
    float max_risk_pct  = 0.030f;  // ceiling: never risk more than 3.0%

    // ── BUG-C5 FIX: volatility targeting and Kelly cold-start ────────────────
    // Previously hardcoded constants inside risk_engine.cpp (FINDING-S5 in the
    // audit). Now promoted to AppConfig so they are configurable from
    // settings.json under the CAPITAL_TIERS block, validated on startup, and
    // visible in the startup summary printed by loadConfig().
    //
    // risk_engine.cpp must be updated to read these via globalConfig() instead
    // of local constexpr float vol_target = 0.015f and coldstart = 0.0105f.
    float vol_target_pct   = 0.015f;   // 1.5% daily vol target for vol-scalar
                                        // (volScalar() in risk_engine.cpp)
    float kelly_coldstart  = 0.0105f;  // Kelly fraction used before enough trade
                                        // history exists for a stable estimate
                                        // (kelly() cold-start path in risk_engine.cpp)

    // Ranker weights
    // All five must sum to 1.0.  Validated in validate(); loadable from JSON
    // under RANKER_WEIGHTS in settings.json.
    int   top_n_signals   = 5;
    float min_rank_score  = 0.45f;
    float weight_conf     = 0.35f;
    float weight_hurst    = 0.20f;
    float weight_adx      = 0.15f;
    float weight_tf_agree = 0.15f;
    float weight_vol_q    = 0.15f;

    // Authentication (loaded from env at startup)
    char secret_key[128]{};

    // ── Constructor: set tier defaults explicitly ────────────────────────────
    AppConfig() noexcept {
        tier_conservative.risk_pct       = 0.005f;  // 0.5% risk, max 15% margin
        tier_conservative.max_margin_pct = 0.15f;
        tier_standard.risk_pct           = 0.010f;  // 1.0% risk, max 25% margin
        tier_standard.max_margin_pct     = 0.25f;
        tier_aggressive.risk_pct         = 0.020f;  // 2.0% risk, max 40% margin
        tier_aggressive.max_margin_pct   = 0.40f;
    }

    [[nodiscard]] bool has_pair(std::string_view sym) const noexcept {
        for (std::size_t i = 0; i < num_pairs; ++i)
            if (std::string_view(pairs[i]) == sym) return true;
        return false;
    }

    // BUG-C1 FIX: runtime sanity check for all loaded values.
    // Called by loadConfig() after parsing; also callable externally.
    // Returns true if config is safe to use; false + prints errors otherwise.
    // Reports ALL invalid fields in a single pass so every problem appears
    // in one log section — no fix-one-restart-find-next cycle.
    [[nodiscard]] bool validate(FILE* err_out = stderr) const noexcept;
};

// ── Singleton accessors (defined in config.cpp) ───────────────────────────────
[[nodiscard]] AppConfig& globalConfig() noexcept;

// Load from BRAIN_SECRET env var + optional settings.json.
// Returns false and prints error if secret is missing, file is absent,
// JSON is malformed, no pairs are defined, or validate() fails.
// On failure, cfg is NOT modified (atomic load-then-assign pattern).
[[nodiscard]] bool loadConfig(AppConfig& cfg,
                               const char* json_path = "settings.json") noexcept;

} // namespace tqc
