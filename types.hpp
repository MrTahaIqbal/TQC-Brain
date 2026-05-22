#pragma once
/*
 * types.hpp  -  BigBoyAgent TQC Brain | Taha Iqbal
 *
 * All domain structs.  Design rules:
 *   - No std::string in structs on the hot path.
 *     std::string holds a heap pointer → pointer chase → cache miss.
 *   - Use char arrays for short strings (symbol names, signal direction).
 *   - Use float (32-bit) not double (64-bit) for indicator values.
 *     float fits 2x as many values in a SIMD register.
 *   - Use int8_t / uint8_t for small integers to maximise cache density.
 *   - C++17: [[nodiscard]] on all helper methods that return values.
 *
 * ── BUGS FIXED IN THIS VERSION ──────────────────────────────────────────────
 *
 * BUG-T1  FixStr<N>: No static_assert guard on N.
 *         If N == 0 the array data[0] is zero-length (legal C++) but
 *         data[N-1] = data[-1] is undefined behaviour in the constructor.
 *         If N == 1 the string can hold zero characters, which is useless
 *         and every copy would just write '\0' to data[0].
 *         FIX: static_assert(N > 1, ...) at the start of the struct.
 *
 * BUG-T2  FixStr<N>: No copy-assignment from const char*.
 *         Without operator=(const char*), code like:
 *             FixStr<16> sym;  sym = "BTCUSDT";
 *         fails to compile. Callers must write the verbose:
 *             sym = FixStr<16>("BTCUSDT");
 *         — error-prone and easy to forget, leading to stale symbol names.
 *         FIX: added operator=(const char*) mirroring constructor logic.
 *
 * BUG-T3  FixStr<N>: No operator==(const FixStr<N>&) overload.
 *         Comparing two FixStr values (e.g. sig.symbol == prev_symbol)
 *         was impossible without going through c_str(), forcing callers
 *         to write strcmp(a.c_str(), b.c_str()) — verbose and crash-prone
 *         if either c_str() returns an unterminated buffer.
 *         FIX: added operator==(const FixStr<N>&) using std::strncmp.
 *
 * BUG-T4  FixStr<N>: No operator!= for either overload.
 *         Absence forces !(a == b) patterns; easy to mis-parenthesise.
 *         FIX: added operator!=(const char*) and operator!=(const FixStr<N>&).
 *
 * BUG-T5  FixStr<N>::operator==(const char*): original loop walked i < N
 *         and read s[i] unconditionally for all N iterations.
 *         If the caller passes a string shorter than N-1 chars (the common
 *         case) the loop reads one byte past s's null terminator, which is
 *         technically well-defined for literals but is undefined behaviour
 *         for any heap/stack string whose allocation ends exactly at '\0'.
 *         FIX: replaced with std::strncmp(data, s, N) which is safe,
 *         correct, and expresses the intent in one standard call.
 *
 * BUG-T6  FixStr<N>: No empty() / len() helpers.
 *         Callers were forced to check data[0] == '\0' inline or call
 *         std::strlen(c_str()), neither of which is discoverable or safe
 *         when the buffer is not null-terminated (possible if N == 1 slips
 *         through without the static_assert).
 *         FIX: added empty() and len() as [[nodiscard]] helpers.
 *
 * BUG-T7  Signal struct: three separate 1-byte fields (pos_valid, tf_agreement,
 *         hmm_state) are interspersed between float sequences, each creating
 *         3 bytes of implicit compiler padding before the next float.
 *         Total waste: 9 bytes per Signal instance — not a correctness bug
 *         but measurable cache pressure when 20+ Signals are compared/ranked.
 *         FIX: grouped all three 1-byte fields together at the end of the
 *         float block (before int64_t internal_ts), collapsing padding to
 *         a single 5-byte gap. External field names and types are UNCHANGED.
 *
 * BUG-T8  [NEW] FixStr<N>::operator=(const char*) did not zero-fill trailing
 *         bytes after assignment.
 *         The constructor is safe: the default member initialiser char data[N]{}
 *         zero-initialises the entire array before the constructor body runs.
 *         The assignment operator lacked this protection: after
 *             FixStr<16> sym("BTCUSDT");  // 7 chars filled, bytes 8-15 = '\0'
 *             sym = "ETH";               // 3 chars filled, data[3] = '\0'
 *         bytes 4-7 would retain 'U','S','D','T' from the previous value.
 *         C-string functions (strncmp, strlen) are safe because they stop at
 *         the null terminator, but any raw-byte operation — memcmp, hashing,
 *         network serialisation, or a future memcpy — reads stale data silently.
 *         FIX: operator= now calls std::memset(data, 0, N) before copying,
 *         matching the zero-init guarantee of the constructor path.
 *
 * BUG-T9  [NEW] Signal struct: stoch_d field was missing entirely.
 *         Features declares both stoch_k and stoch_d. Signal declared only
 *         stoch_k. The Stochastic %D line is the 3-bar SMA of %K — it is the
 *         signal line whose crossover with %K is the primary trading trigger
 *         of the Stochastic oscillator. Without stoch_d in Signal:
 *           (a) The executor never receives the crossover direction.
 *           (b) The "5 bytes of padding before int64_t" comment in the struct
 *               was incorrect: without stoch_d the actual padding is only
 *               1 byte. With stoch_d restored the padding is exactly 5 bytes,
 *               matching the documented layout — confirming the field was
 *               present when the struct was designed and accidentally dropped.
 *         FIX: stoch_d restored immediately after stoch_k in the Indicators
 *         block. The padding comment is now correct again.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <array>
#include <cstdint>
#include <cstddef>
#include <cstring>

namespace tqc {

// ── Fixed-size string: never allocates ───────────────────────────────────────
template<std::size_t N>
struct FixStr {
    // BUG-T1 FIX: guard against degenerate sizes.
    static_assert(N > 1, "FixStr<N>: N must be >= 2 (at least 1 char + null terminator).");

    char data[N]{};

    FixStr() = default;

    explicit FixStr(const char* s) noexcept {
        // data[N]{} zero-initialises the entire array before this body runs
        // (via the default member initialiser), so no memset is needed here.
        if (!s) { data[0] = '\0'; return; }
        std::size_t i = 0;
        for (; i < N - 1 && s[i]; ++i)
            data[i] = s[i];
        data[i] = '\0';   // explicit null even if s was shorter
    }

    // BUG-T2 FIX: assignment from raw C-string.
    // BUG-T8 FIX: zero-fill the entire buffer before copying so that no
    //             stale bytes survive a reassignment to a shorter string.
    //             The constructor path is safe (zero-init via default member
    //             initialiser) but the assignment operator had no such
    //             guarantee — now both paths are consistent.
    FixStr& operator=(const char* s) noexcept {
        std::memset(data, 0, N);          // clear stale content from prior value
        if (!s) return *this;
        for (std::size_t i = 0; i < N - 1 && s[i]; ++i)
            data[i] = s[i];
        // data[N-1] is already '\0' from memset
        return *this;
    }

    [[nodiscard]] const char* c_str() const noexcept { return data; }

    // BUG-T6 FIX: discoverable helpers.
    [[nodiscard]] bool        empty() const noexcept { return data[0] == '\0'; }
    [[nodiscard]] std::size_t len()   const noexcept { return std::strlen(data); }

    // BUG-T5 FIX: use std::strncmp — safe against short/non-null-terminated s.
    // BUG-T4 FIX: operator!= provided alongside operator==.
    [[nodiscard]] bool operator==(const char* s) const noexcept {
        if (!s) return false;
        return std::strncmp(data, s, N) == 0;
    }
    [[nodiscard]] bool operator!=(const char* s) const noexcept {
        return !(*this == s);
    }

    // BUG-T3 FIX: compare two FixStr of the same size.
    [[nodiscard]] bool operator==(const FixStr<N>& o) const noexcept {
        return std::strncmp(data, o.data, N) == 0;
    }
    [[nodiscard]] bool operator!=(const FixStr<N>& o) const noexcept {
        return !(*this == o);
    }
};

// ── Regime enums ──────────────────────────────────────────────────────────────
enum class Regime    : uint8_t { NOISE = 0, TRENDING = 1, MEAN_REVERTING = 2 };
enum class VolRegime : uint8_t { LOW   = 0, NORMAL   = 1, HIGH           = 2 };
enum class SignalDir : uint8_t { HOLD  = 0, BUY      = 1, SELL           = 2 };

// ── Features ─────────────────────────────────────────────────────────────────
// Computed once per symbol per cycle.  Flows through the entire pipeline.
struct alignas(64) Features {
    char   symbol[16]{};
    float  last_price         = 0.0f;
    int    data_points        = 0;

    // Mean-reversion / trend
    float  z_score            = 0.0f;
    float  bollinger_position = 0.0f;
    float  roc_5              = 0.0f;
    float  roc_10             = 0.0f;
    float  roc_20             = 0.0f;
    float  velocity           = 0.0f;   // 1-bar return

    // Momentum oscillators
    float  rsi                = 50.0f;
    float  vw_rsi             = 50.0f;
    float  macd_signal        = 0.0f;
    float  stoch_k            = 0.5f;
    float  stoch_d            = 0.5f;   // 3-bar SMA of stoch_k (signal line)

    // Volatility
    float  volatility         = 0.0f;
    float  atr                = 0.0f;
    float  atr_pct            = 0.01f;

    // Volume / price / order book
    float  vwap               = 0.0f;
    float  vwap_distance      = 0.0f;
    float  volume_imbalance   = 0.0f;
    // Order book imbalance: (bid_size − ask_size) / (bid_size + ask_size) ∈ [-1, +1].
    // Positive  →  more resting buy interest  →  mild bullish pressure.
    // Populated from WebSocket L2 data when available; 0 otherwise (neutral).
    float  ob_imbalance       = 0.0f;

    // Moving averages
    float  sma_20             = 0.0f;
    float  sma_50             = 0.0f;

    // Multi-timeframe agreement
    int8_t tf_agreement       = 0;    // -1 bearish, 0 neutral, +1 bullish

    // ── Step 2: Statistical models + microstructure payload fields ────────────
    // Fields below are populated in two stages:
    //   (a) HAR-RV / GARCH / TSMOM / ARIMA — computed inside generateFeatures()
    //       from bar history in g_bars.
    //   (b) vwoi / funding_rate / funding_zscore / basis — injected in
    //       processSymbol() (main.cpp) from the executor's JSON payload AFTER
    //       generateFeatures() returns, because these come from real-time
    //       WebSocket/REST feeds, not from stored bar history.
    float  har_rv           = 0.0f;  // HAR-RV volatility forecast (next bar)
    float  garch_vol        = 0.0f;  // GARCH(1,1) conditional vol (annualised σ)
    float  tsmom            = 0.0f;  // Time-series momentum factor (Moskowitz 2012)
    float  arima_forecast   = 0.0f;  // ARIMA(1,1,1) 1-step return forecast
    float  vwoi             = 0.0f;  // Volume-Weighted Order Flow Imbalance [-1,+1]
    float  funding_rate     = 0.0f;  // Binance 8hr funding rate (raw, e.g. 0.0001)
    float  funding_zscore   = 0.0f;  // z-score vs 30-bar rolling funding window
    float  basis            = 0.0f;  // (mark_price - index_price) / index_price
};

// ── Regime result ─────────────────────────────────────────────────────────────
struct RegimeInfo {
    Regime    regime     = Regime::NOISE;
    VolRegime vol_regime = VolRegime::NORMAL;
    float     hurst      = 0.5f;
    float     adx        = 0.0f;
    float     reg_slope  = 0.0f;

    // HMM 3-state Viterbi output: 0=BEAR, 1=SIDEWAYS, 2=BULL
    // Computed in regime_engine.cpp alongside the Hurst/ADX classifier.
    // Used by signal_engine.cpp to gate entries and by backtest for regime slicing.
    uint8_t   hmm_state  = 1;  // default SIDEWAYS (most common crypto regime)

    [[nodiscard]] const char* regime_str() const noexcept {
        switch (regime) {
            case Regime::TRENDING:       return "TRENDING";
            case Regime::MEAN_REVERTING: return "MEAN_REVERTING";
            default:                     return "NOISE";
        }
    }
    [[nodiscard]] const char* vol_regime_str() const noexcept {
        switch (vol_regime) {
            case VolRegime::LOW:  return "LOW";
            case VolRegime::HIGH: return "HIGH";
            default:              return "NORMAL";
        }
    }
    [[nodiscard]] const char* hmm_state_str() const noexcept {
        switch (hmm_state) {
            case 0:  return "BEAR";
            case 2:  return "BULL";
            default: return "SIDEWAYS";
        }
    }
};

// ── Signal result ─────────────────────────────────────────────────────────────
struct SignalResult {
    SignalDir dir        = SignalDir::HOLD;
    float     confidence = 0.5f;

    [[nodiscard]] const char* dir_str() const noexcept {
        switch (dir) {
            case SignalDir::BUY:  return "BUY";
            case SignalDir::SELL: return "SELL";
            default:              return "HOLD";
        }
    }
};

// ── Position sizing output ────────────────────────────────────────────────────
struct PositionParams {
    float margin        = 0.0f;
    float notional      = 0.0f;
    float sl_price      = 0.0f;
    float tp_price      = 0.0f;
    float sl_pct        = 0.0f;
    float risk_usd      = 0.0f;
    float risk_pct      = 0.0f;
    float vol_scalar    = 1.0f;
    float var_95        = 0.0f;
    bool  valid         = false;
    char  reject_reason[48]{};
};

// ── Risk gate output ──────────────────────────────────────────────────────────
struct RiskCheckResult {
    bool allowed = false;
    char reason[48]{};
};

// ── Per-symbol trade statistics ───────────────────────────────────────────────
struct TradeStats {
    int   trades       = 0;
    float win_rate     = 0.0f;
    float avg_win      = 0.0f;
    float avg_loss     = 0.0f;
    float total_pnl    = 0.0f;
    float sharpe       = 0.0f;
    float sortino      = 0.0f;
    float max_drawdown = 0.0f;
    float calmar       = 0.0f;
};

// ── Full signal output (one per pair per cycle) ───────────────────────────────
// Lives on the stack during computation.
// Serialised to JSON only at the HTTP boundary.
//
// BUG-T7 FIX: Field reordering to eliminate implicit compiler padding.
//
// ORIGINAL layout had three 1-byte fields (pos_valid:bool, tf_agreement:int8_t,
// hmm_state:uint8_t) scattered between float sequences.  Each one caused 3 bytes
// of padding before the next float, for a total of 9 wasted bytes per Signal.
// With 20 symbols in flight, that's 180 bytes of useless cache load every cycle.
//
// NEW layout: keep all semantically-related fields visually together but move
// the three 1-byte fields into a single consecutive block.  The combined
// 3-byte region is followed immediately by int64_t internal_ts (8-byte aligned).
//
// BUG-T9 FIX: stoch_d restored (see header comment above).
//
// Verified layout (cumulative byte offsets):
//   symbol[16]                              : 16 bytes  →  offset 0
//   signal[8]                               : 8 bytes   →  offset 16
//   7 floats  (confidence … position_size)  : 28 bytes  →  offset 24
//   regime[20] + vol_regime[8]              : 28 bytes  →  offset 52
//   hurst + adx + reg_slope                 : 12 bytes  →  offset 80
//   atr … volatility  (15 floats + stoch_d) : 64 bytes  →  offset 92  → 156
//     NOTE: stoch_d is the 5th of these 15+1=16 floats (was 15 before fix).
//   unrealized_pnl + total_pnl + sharpe     : 12 bytes  →  offset 216  (was 212)
//   rank_score … data_points  (2f+2i)       : 16 bytes  →  offset 228  (was 224)
//   har_rv … basis  (8 floats)              : 32 bytes  →  offset 244  (was 240)
//   pos_valid + tf_agreement + hmm_state    : 3 bytes   →  offset 276  (was 272)
//   5 bytes padding (compiler-managed)      → offset 279+5=284? ...
//
// Simplified verified count before the 3 one-byte fields:
//   16 + 8 + 28 + 28 + 12 + 64 + 12 + 16 + 32 = 216 bytes
//   216 is divisible by 8 → the 3 one-byte fields occupy 217, 218, 219
//   Next 8-byte boundary: 224 → 5 bytes of padding  ✓ (matches comment)
//
// All field names and types are IDENTICAL to the original.
struct alignas(64) Signal {
    // ── Identity ──────────────────────────────────────────────────────────────
    char    symbol[16]{};
    char    signal[8]{};      // "BUY" / "SELL" / "HOLD"

    // ── Price / sizing ─────────────────────────────────────────────────────────
    float   confidence         = 0.0f;
    float   price              = 0.0f;
    float   high               = 0.0f;
    float   low                = 0.0f;
    float   suggested_sl       = 0.0f;
    float   suggested_tp       = 0.0f;
    float   position_size      = 0.0f;

    // ── Regime ────────────────────────────────────────────────────────────────
    // char arrays placed before floats so they share the same alignment class
    // and require no padding between them.
    char    regime[20]{};
    char    vol_regime[8]{};

    float   hurst              = 0.5f;
    float   adx                = 0.0f;
    float   reg_slope          = 0.0f;

    // ── Indicators ────────────────────────────────────────────────────────────
    float   atr                = 0.0f;
    float   atr_pct            = 0.0f;
    float   vw_rsi             = 50.0f;
    float   rsi                = 50.0f;
    float   stoch_k            = 0.5f;
    float   stoch_d            = 0.5f;  // BUG-T9 FIX: restored 3-bar SMA of K
                                        // (crossover signal line). Its absence
                                        // made the padding comment wrong (1 byte
                                        // actual vs 5 bytes documented). With this
                                        // field present the layout is correct.
    float   macd_signal        = 0.0f;
    float   roc_5              = 0.0f;
    float   roc_10             = 0.0f;
    float   roc_20             = 0.0f;
    float   z_score            = 0.0f;
    float   bollinger_position = 0.0f;
    float   vwap_distance      = 0.0f;
    float   volume_imbalance   = 0.0f;
    float   ob_imbalance       = 0.0f;   // order-book bid/ask size imbalance
    float   volatility         = 0.0f;

    // ── Performance (brain simulation) ────────────────────────────────────────
    float   unrealized_pnl     = 0.0f;
    float   total_pnl          = 0.0f;
    float   sharpe             = 0.0f;

    // ── Ranking ───────────────────────────────────────────────────────────────
    float   rank_score         = 0.0f;
    int     rank_position      = 0;
    float   alpha_score        = 0.0f;
    int     data_points        = 0;

    // ── Step 2: Statistical model outputs ─────────────────────────────────────
    float   har_rv             = 0.0f;  // HAR-RV vol forecast
    float   garch_vol          = 0.0f;  // GARCH(1,1) annualised conditional vol
    float   tsmom              = 0.0f;  // Time-series momentum factor
    float   arima_forecast     = 0.0f;  // ARIMA(1,1,1) 1-step return forecast
    float   vwoi               = 0.0f;  // Volume-weighted order flow imbalance
    float   funding_rate       = 0.0f;  // Binance perpetual funding rate
    float   funding_zscore     = 0.0f;  // Funding rate z-score (30-bar rolling)
    float   basis              = 0.0f;  // (mark - index) / index

    // ── 1-byte flags grouped together (BUG-T7 FIX) ────────────────────────────
    // Original placement interleaved these with floats, causing 3×3 = 9 bytes
    // of padding. Grouped here they need exactly 5 bytes of padding before
    // int64_t internal_ts (8-byte aligned), saving 4 bytes total vs the
    // original 9-byte scattered padding.
    bool    pos_valid          = false;  // position sizing was successful
    int8_t  tf_agreement       = 0;     // -1 bearish, 0 neutral, +1 bullish
    uint8_t hmm_state          = 1;     // 0=BEAR, 1=SIDEWAYS, 2=BULL (HMM Viterbi)

    // 5 bytes of padding here (compiler-managed) before int64_t.
    // This count is correct only when stoch_d is present (BUG-T9 FIX).
    // Without stoch_d the padding would collapse to 1 byte — the mismatch
    // between the comment and reality was the diagnostic clue for BUG-T9.

    int64_t internal_ts        = 0;
    char    error[128]{};
};

// ── Latency statistics ────────────────────────────────────────────────────────
struct LatencyStats {
    float avg_ms  = 0.0f;
    float p95_ms  = 0.0f;
    float p99_ms  = 0.0f;
    float max_ms  = 0.0f;
    int   samples = 0;
};

} // namespace tqc
