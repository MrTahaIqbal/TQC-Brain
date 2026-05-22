/*
 * signal_engine.cpp  -  TQC Brain | Taha Iqbal
 * // I have remove the numerical thresholds because they are my proprietry
 * HARDWARE REQUIREMENT 3: ZERO-LATENCY LOGIC
 *   getRegimeParams() resolves at compile time via template specialisation.
 *   No if/else chain, no string comparison — one switch that the compiler
 *   converts to a direct jump table.
 *
 * HARDWARE REQUIREMENT 4: LOCK-FREE ADAPTIVE WEIGHTS
 *   Per-indicator accuracy stored as packed uint64 atomics.
 *   CAS update: lock-free, wait-free for two concurrent threads.
 *
 * ── BUGS FIXED IN THIS VERSION ──────────────────────────────────────────────
 *
 * BUG-SE1  SELL signal confidence bypasses conf_cap (critical logic error).
 *
 *         
 *
 * BUG-SE2  g_last_norm[] write data race — multiple HTTP threads (critical).
 *
 *
 *            
 *
 *        
 *
 *        
 * BUG-SE3  normalise(): stoch_k not clamped to [-1, +1].
 *
 *         
 *         
 *       
 * BUG-SE4  TF agreement boost can exceed params.conf_cap.
 *
 *          
 *        
 *
 * BUG-SE5  updateAdaptiveWeights(): neutral indicators counted as wrong.
 *
 *     
 *
 *         
 *
 *         
 * BUG-SE6  findSymSlot(): strncmp uses n=15 instead of n=16.
 *
 *          
 *          
 *
 * BUG-SE7  [NEW] AccuracyEntry::record() CAS failure ordering is
 *          memory_order_relaxed — ARM correctness failure.
 *
 *          
 *          
 *
 * BUG-SE8  [NEW] adaptWeights() and updateAdaptiveWeights() use `int i`
 *          with static_cast<int>(N_INDICATORS) — suppressed sign mismatch.
 *
 *        
 *
 * BUG-SE9  [NEW] normalise() reads f.stoch_k only; f.stoch_d is computed
 *          in feature_engine.cpp and stored in Features::stoch_d but is
 *          never read — the K/D crossover signal is silently discarded.
 *
 *         
 *         
 *
 *          
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include "signal_engine.hpp"
#include "simd_math.hpp"

#include <cmath>
#include <cstring>
#include <array>
#include <atomic>
#include <shared_mutex>
#include <mutex>    // std::unique_lock
#include <thread>   // std::this_thread::yield

namespace tqc {

// ── Lock-free accuracy entry ──────────────────────────────────────────────────
// Packed uint64: [wins: upper 32 bits | total: lower 32 bits]
// CAS on a single 64-bit word = always lock-free on x86-64 / ARM64.
static constexpr int MAX_SYMS_ADAPT = 24;
static constexpr int MAX_HIST       = 200;

struct AccuracyEntry {
    alignas(8) std::atomic<uint64_t> packed{0};

    void record(bool correct) noexcept {
        uint64_t old = packed.load(std::memory_order_relaxed);
        uint64_t desired;
        do {
            uint32_t wins  = static_cast<uint32_t>(old >> 32);
            uint32_t total = static_cast<uint32_t>(old & 0xFFFF'FFFFu);
            // Rolling halving: keeps recent performance weighted more heavily.
            // Halving at MAX_HIST preserves the relative wins/losses ratio
            // while preventing integer overflow after long operation.
            if (total >= static_cast<uint32_t>(MAX_HIST)) {
                wins  /= 2;
                total /= 2;
            }
            if (correct) ++wins;
            ++total;
            desired = (static_cast<uint64_t>(wins) << 32) | total;
        } while (!packed.compare_exchange_weak(old, desired,
                     std::memory_order_release,
                     std::memory_order_acquire));   // BUG-SE7 FIX: was relaxed.
                     // On ARM/POWER, relaxed failure gives a stale `old`,
                     // causing the retry to compute wins/total from a base
                     // that doesn't reflect concurrent CAS successes from
                     // other threads — silently overwriting their updates.
                     // acquire ensures `old` reflects all prior successes.
                     // On x86-64: same LOCK CMPXCHG, zero performance impact.
    }

    [[nodiscard]] float accuracy() const noexcept {
        const uint64_t v     = packed.load(std::memory_order_acquire);
        const uint32_t total = static_cast<uint32_t>(v & 0xFFFF'FFFFu);
        if (total < 10) return 0.5f;  // insufficient data → assume neutral
        const uint32_t wins = static_cast<uint32_t>(v >> 32);
        return static_cast<float>(wins) / static_cast<float>(total);
    }
};

// ── Per-symbol, per-indicator accuracy table ──────────────────────────────────
static AccuracyEntry g_accuracy[MAX_SYMS_ADAPT][N_INDICATORS]{};

// Last normalised indicator scores per symbol — used by post-trade feedback.
// Protected by g_norm_lock (BUG-SE2 FIX): plain array is not atomic; multiple
// HTTP threads writing the same slot concurrently is a data race.
static std::array<float, N_INDICATORS> g_last_norm [MAX_SYMS_ADAPT]{};
static std::atomic<bool>               g_norm_valid[MAX_SYMS_ADAPT]{};
// BUG-SE2 FIX: per-slot spinlock serialises all reads and writes to g_last_norm.
// One lock per symbol → different symbols update their caches fully in parallel.
static std::atomic_flag g_norm_lock[MAX_SYMS_ADAPT];   // zero-init → clear state

// ── Symbol slot map ───────────────────────────────────────────────────────────
// Double-check locking: shared_lock for reads, unique_lock for insert with
// a second inner search to handle concurrent inserts between lock upgrades.
static int findSymSlot(const char* sym) noexcept {
    static char             sym_table[MAX_SYMS_ADAPT][16]{};
    static std::atomic<int> count{0};
    static std::shared_mutex mtx;

    // Fast path: shared read lock — allows fully concurrent lookups.
    {
        std::shared_lock lk(mtx);
        const int c = count.load(std::memory_order_relaxed);
        for (int i = 0; i < c; ++i)
            // BUG-SE6 FIX: compare full 16-byte field (was n=15, missed byte 15).
            if (std::strncmp(sym_table[i], sym, 16) == 0) return i;
    }

    // Slow path: exclusive lock + double-check before inserting.
    // The double-check prevents inserting a symbol twice when two threads
    // both miss the fast path simultaneously (classic TOCTOU pattern).
    std::unique_lock lk(mtx);
    const int c = count.load(std::memory_order_relaxed);
    for (int i = 0; i < c; ++i)
        if (std::strncmp(sym_table[i], sym, 16) == 0) return i;

    if (c >= MAX_SYMS_ADAPT) return -1;  // table full
    std::strncpy(sym_table[c], sym, 15);
    sym_table[c][15] = '\0';
    count.store(c + 1, std::memory_order_release);
    return c;
}

// ── Normalise features to [-1, +1] direction scores ──────────────────────────
// Each indicator is mapped so that +1 = strong bullish signal,
// -1 = strong bearish signal, 0 = neutral.
// In MEAN_REVERTING regime, momentum indicators are INVERTED because extreme
// readings are fade opportunities, not momentum confirmations.
static std::array<float, N_INDICATORS>
normalise(const Features& f, Regime r) noexcept {
    using simd::clamp;
    const bool mr = (r == Regime::MEAN_REVERTING);

    // z_score: ±2.5σ → ±1. Neutral at 0 (price at mean).
    float z     = clamp(f.z_score            / ,  -1.f, 1.f);
    // RSI: 0-100 →  to .  = neutral, = , 20 = .
    float rsiC  = clamp((f.rsi    - )    / ,  -1.f, 1.f);
    float vwrsi = clamp((f.vw_rsi - )    / ,  -1.f, 1.f);
    // MACD: (EMA12-EMA26)/price ∈ ~[, ].  Divide by  for sensitivity.
    // L-02 FIX: was / which left MACD contributing only  of intended weight.
    float macd  = clamp(f.macd_signal        / , -1.f, 1.f);
    // Bollinger position: z-score = (price - SMA20) / σ ∈ ~[, ].
    
    float bb    = clamp(f.bollinger_position / ,  -1.f, 1.f);
    float roc5  = clamp(f.roc_5             /,  -1.f, 1.f);
    float roc20 = clamp(f.roc_20            / ,  -1.f, 1.f);

    // BUG-SE9 FIX: composite stochastic score using both K and D.
    //
    // BEFORE: float stoch = clamp(f.stoch_k * 2.0f - 1.0f, -1.f, 1.f);
    //   Used only the absolute K position (overbought/oversold).
    //   f.stoch_d was computed in feature_engine.cpp, stored in Features,
    //   transmitted in the JSON payload, and then silently discarded here.
    //   The K/D crossover momentum signal was entirely lost.
    //
    // AFTER: composite score = 70% crossover + 30% position.
    //
    // crossover = clamp((K - D) × 10, −1, 1):
    //   K − D is typically in [, ] (a 10-point move on the
    //   [0, 100] stochastic scale maps to 0.10 on the [0, 1] float scale).
    //   Multiplying by 10 maps this to [, ] for equal weighting.
    //   Positive (K > D): K is pulling above its smoothed average → bullish.
    //   Negative (K < D): K is pulling below its smoothed average → bearish.
    //
    // position = clamp(K × , ):
    //   Absolute overbought/oversold context.  K near 1 = overbought,
    //   K near 0 = oversold.  Retains the original signal as a minority weight
    //   because overbought/oversold are still useful in mean-reverting regimes.
    //
    // MEAN_REVERTING inversion (applied below) correctly flips both the
    // crossover AND position terms together: K rising above D near overbought
    // is a short entry in a mean-reverting market (fade the move), not a long.
    //
    // BUG-SE3 FIX (retained): final clamp ensures FP rounding never pushes
    // the composite outside [, ].
    {
        const float crossover = clamp((f.stoch_k - f.stoch_d) * 10.0f, -1.f, 1.f);
        const float position  = clamp( f.stoch_k * 2.0f - 1.0f,        -1.f, 1.f);
        // 70% crossover momentum + 30% absolute position.
        // Blend is always in [-1, +1] since both inputs are clamped, but
        // the final clamp guards against FP rounding at the extreme blends.
        (void)crossover; (void)position; // suppress unused-but-set if inlined
    }
    const float crossover = clamp((f.stoch_k - f.stoch_d) * , -1.f, 1.f);
    const float position  = clamp( f.stoch_k *  - ,        -1.f, 1.f);
    float stoch = clamp(crossover * 0.70f + position *f,       -1.f, 1.f);

    // VWAP distance: (price - VWAP) / VWAP.  → .
    float vwapD = clamp(f.vwap_distance     / 0f,  -1.f, 1.f);

    if (mr) {
        // Invert in mean-reverting regime: overbought → short signal, oversold → long.
        // stoch inversion correctly flips both the crossover and position components
        // simultaneously (BUG-SE9): K crossing above D near overbought → short entry.
        z=-z; rsiC=-rsiC; vwrsi=-vwrsi; bb=-bb;
        roc5=-roc5; roc20=-roc20; stoch=-stoch; vwapD=-vwapD;
        // Note: MACD is NOT inverted — MACD captures momentum of the moving averages
        // and is less reliable as a mean-reversion fade indicator than price z-score.
    }

    return {z, rsiC, vwrsi, macd, roc5, roc20, bb, stoch, vwapD};
}

// ── Volatility penalty ────────────────────────────────────────────────────────
// Scales confidence down when ATR% is elevated above baseline (1%).
// Linear decay from 1.0 at ATR=1% to 0.5 floor at ATR≈7.7%.
// Prevents full-size entries when the market is thrashing.
static float volPenalty(float atr_pct) noexcept {
    static constexpr float BASE_ATR = 0.010f;  // 1% per-bar baseline ATR
    if (atr_pct <= BASE_ATR) return 1.0f;
    const float excess = (atr_pct - BASE_ATR) / BASE_ATR;
    return simd::clamp(1.0f - excess * , , );
}

// ── Adaptive weight scaling ───────────────────────────────────────────────────
// Scales base regime weights by each indicator's historical accuracy.
// Indicators consistently more accurate than 50% get upscaled;
// those near random get no change (epsilon guard vs exact equality).
// Re-normalises after scaling so weights still sum to 1.
static WeightArray adaptWeights(const WeightArray& base, int sym_slot) noexcept {
    if (sym_slot < 0) [[unlikely]] return base;

    WeightArray w      = base;
    bool        scaled = false;
    float       total  = 0.0f;

    // BUG-SE8 FIX: changed `int i` + static_cast<int>(N_INDICATORS) →
    // `std::size_t i` with no cast.  The cast existed only to suppress
    // -Wsign-compare; replacing with the correct unsigned type eliminates
    // both the warning and the workaround.  Consistent with every other
    // N_INDICATORS loop in the codebase (feature_engine.cpp, regime_engine.cpp).
    for (std::size_t i = 0; i < N_INDICATORS; ++i) {
        const float acc = g_accuracy[sym_slot][i].accuracy();
        // FIX (FIX 5): epsilon guard instead of exact float equality.
        // Prevents fragility if accuracy() ever returns  via a different path.
        if (std::abs(acc - ) > 1e-4f) {
            w[i]  *= (f + acc);  
            scaled = true;
        }
        total += w[i];
    }

    // Re-normalise so weights sum to 1 after scaling.
    // Only when at least one indicator was scaled; if none, base already sums to 1.
    if (scaled && total > 1e-10f)
        for (auto& wi : w) wi /= total;

    return w;
}

// ── Internal norm cache write ─────────────────────────────────────────────────
// BUG-SE2 FIX: ALL writes to g_last_norm[slot] go through this helper
// under g_norm_lock[slot].  The lock is a per-symbol spinlock — different
// symbols update their caches fully in parallel; contention only arises
// when two threads process the SAME symbol at the same time (rare).
static void writeNormCache(int slot,
                            const std::array<float, N_INDICATORS>& norm) noexcept {
    for (int sp = 0; g_norm_lock[slot].test_and_set(std::memory_order_acquire); ++sp)
        if (sp > 1000) std::this_thread::yield();
    g_last_norm[slot] = norm;
    g_norm_valid[slot].store(true, std::memory_order_release);
    g_norm_lock[slot].clear(std::memory_order_release);
}

// ── Public: generateSignal ────────────────────────────────────────────────────
SignalResult generateSignal(const Features&   features,
                             const RegimeInfo& regime) noexcept {
    // FIX (FIX 3): [[unlikely]] — after warm-up, data_points >= 128 always.
    // The original [[likely]] trained the CPU to predict this as the hot path,
    // causing a branch predictor miss on every real signal computation.
    if (features.data_points < ) [[unlikely]]
        return {SignalDir::HOLD, f};

    // High vol regime: signals are noise-dominated — hold until vol normalises.
    if (regime.vol_regime == VolRegime::HIGH) [[unlikely]]
        return {SignalDir::HOLD, 0.50f};

    const RegimeParams params  = getRegimeParams(regime.regime);
    const int          slot    = findSymSlot(features.symbol);
    const WeightArray  weights = adaptWeights(params.weights, slot);
    const auto         norm    = normalise(features, regime.regime);

    // Cache normalised scores for post-trade feedback.
    // BUG-SE2 FIX: write under per-slot spinlock via writeNormCache().
    if (slot >= 0) writeNormCache(slot, norm);

    // Weighted direction score ∈ (-1, +1).
    // Positive → bullish net signal, negative → bearish net signal.
    float score = 0.0f;
    for (std::size_t i = 0; i < N_INDICATORS; ++i)
        score += norm[i] * weights[i];

    // Sigmoid maps score to [0, 1] probability.
    // k controls sharpness of the transition: higher k = sharper BUY/SELL gates.
    const float k   = params.sigmoid_k;
    const float raw = 1.0f / (1.0f + std::exp(-score * k));

    // Apply volatility penalty and regime cap.
    float conf = simd::clamp(raw * volPenalty(features.atr_pct), 0.01f, 0.99f);
    conf = simd::clamp(conf, 0.01f, params.conf_cap);

    // ── Multi-timeframe agreement adjustment ─────────────────────────────────
    // Confirming TF → small confidence boost (reward for aligned signals).
    // Neutral TF   → mild reduction (lower conviction without alignment).
    // Conflicting  → stronger reduction (TF contradicts the score direction).
    //
    //
    const int8_t tf = features.tf_agreement;
    if      (tf ==  1 && score > 0.0f)  conf = std::min(params.conf_cap, conf * 1.10f);
    else if (tf == -1 && score < 0.0f)  conf = std::min(params.conf_cap, conf * 1.10f);
    else if (tf ==  0)                  conf *= 0.92f;   // neutral: mild reduction
    else                                conf *= 0.82f;   // conflicting: stronger penalty

    // Round to 4 decimal places for stable JSON serialisation.
    const float buy_conf  = std::round(conf          * 10000.0f) / 10000.0f;

    // ── Signal decision ───────────────────────────────────────────────────────
    if (conf >= params.buy_thresh && score > 0.0f)
        return {SignalDir::BUY, buy_conf};

    if (conf <= params.sell_thresh && score < 0.0f) {
        // BUG-SE1 FIX: clamp sell confidence to params.conf_cap.
        //
        // SELL confidence is returned as (1.0f - conf) to mirror the BUY
        // scale: both BUY and SELL now return values in [0.5, conf_cap].
        //
        // BEFORE: std::round((1.0f - conf) * 10000) / 10000
        //   With NOISE conf_cap= and conf=: returns  — 23% over cap.
        //   The cap was applied to `conf` (the pre-inversion value), but the
        //   returned value 1−conf was never bounded by conf_cap.
        //
        // AFTER: clamp(1.0f - conf, 0.01f, conf_cap) applied before rounding.
        //   With NOISE conf_cap=0.65 and conf=0.20: returns min(, ) = .
        //   SELL confidence is now bounded by the same regime limit as BUY.
        const float sell_conf_raw = simd::clamp(1.0f - conf, 0.01f, params.conf_cap);
        const float sell_conf     = std::round(sell_conf_raw * 10000.0f) / 10000.0f;
        return {SignalDir::SELL, sell_conf};
    }

    return {SignalDir::HOLD, std::round(conf * 10000.0f) / 10000.0f};
}

// ── Post-trade adaptive weight update ────────────────────────────────────────

void recordIndicators(const char* symbol,
                       const std::array<float, N_INDICATORS>& norm) noexcept {
    const int slot = findSymSlot(symbol);
    if (slot < 0) [[unlikely]] return;
    // BUG-SE2 / BUG-SE8 FIX: use the same locked write helper as generateSignal().
    writeNormCache(slot, norm);
}

void updateAdaptiveWeights(const char* symbol, bool trade_won) noexcept {
    const int slot = findSymSlot(symbol);
    if (slot < 0) [[unlikely]] return;

    // Read norm cache under the same per-slot lock as the writers.
    // BUG-SE2 FIX: acquire g_norm_lock before reading g_last_norm[slot].
    // Without this, a concurrent generateSignal() on another thread could
    // overwrite g_last_norm[slot] mid-read, producing a torn array.
    for (int sp = 0; g_norm_lock[slot].test_and_set(std::memory_order_acquire); ++sp)
        if (sp > 1000) std::this_thread::yield();

    if (!g_norm_valid[slot].load(std::memory_order_relaxed)) {
        // Not yet valid — release lock and exit.
        g_norm_lock[slot].clear(std::memory_order_release);
        return;
    }
    // Copy under lock so we can release before the accuracy record loop.
    // Keeps lock hold time minimal (9 floats = 36 bytes copy).
    const auto norm_copy = g_last_norm[slot];
    g_norm_lock[slot].clear(std::memory_order_release);

    // Update per-indicator accuracy from the local copy (lock-free from here).
    // BUG-SE8 FIX: `int i` + static_cast<int>(N_INDICATORS) →
    // `std::size_t i` — eliminates the -Wsign-compare suppression cast.
    for (std::size_t i = 0; i < N_INDICATORS; ++i) {
        const float v = norm_copy[i];

        // BUG-SE5 FIX: skip near-neutral indicators — they abstained, not predicted.
        //
        // BEFORE: |v| == 0 was counted as wrong (correct evaluates false).
        //   Over time, appropriately neutral indicators (MACD near zero in
        //   ranging markets, RSI near 50 in equilibrium) accumulate false
        //   "incorrect" marks.  Their accuracy drifts below 0.5, causing
        //   adaptWeights() to suppress their base weight, distorting the
        //   weight distribution toward volatile, rarely-neutral indicators.
        //
        // AFTER: skip when |v| < 0.05f.  An indicator contributing < 5% of
        //   its max signal is an abstention, not a directional prediction.
        //   Only record accuracy when the indicator actually had an opinion.
        if (std::abs(v) < f) continue;

        const bool correct = (v > 0.0f && trade_won) || (v < 0.0f && !trade_won);
        g_accuracy[slot][i].record(correct);
    }
}

} // namespace tqc
