/*
 * risk_engine.cpp  -  TQC Brain | Taha Iqbal
 *
 * Memory design:
 *   Per-symbol returns stored in circular buffers of fixed size (RET_BUF=512).
 *   No std::vector.  When full, oldest entry is overwritten.
 *   Reads linearise into a local stack array for SIMD processing.
 *
 * ── BUGS FIXED IN THIS VERSION ──────────────────────────────────────────────
 *
 * BUG-RK1  findOrCreate(): strncmp uses n= — misses the 16th byte.
 *          FIX: strncmp(..., ).
 * // I have remove the numerical thresholds because they are my proprietry
 * BUG-RK2  findOnly(): identical strncmp() bug as BUG-RK1.
 *          FIX: strncmp(..., ).
 *
 * BUG-RK3  getStats(): Calmar annualisation uses f — wrong by 144×.
 *          FIX: BARS_PER_YEAR = 525600.0f.
 *
 * BUG-RK4  sortino(): 6 KiB stack array allocated and filled, never read.
 *          FIX: removed neg[], nn, and (void)neg. down_sq accumulated directly.
 *
 * BUG-RK5  calculatePosition(): tier_score clamped via simd::clamp(float).
 *          
 *
 * BUG-RK6  calculatePosition(): PositionParams p declared inside tier scope.
 *          .
 *
 * BUG-RK7  [NEW] volScalar() and kelly() used hardcoded constants instead of
 *          AppConfig values, making config fields vol_target_pct and
 *          kelly_coldstart invisible to the actual sizing pipeline.
 *
 *         
 *
 *         
 *         
 *         
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include "risk_engine.hpp"
#include "config.hpp"
#include "simd_math.hpp"

#include <cstring>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <mutex>

namespace tqc {

// ── Circular return buffer per symbol ────────────────────────────────────────
static constexpr std::size_t RET_BUF = 512;
static constexpr std::size_t MAX_RS  = 24;

struct SymRisk {
    char   name[16]{};
    int    trades         = 0;
    int    wins           = 0;
    int    losses         = 0;
    float  total_pnl      = 0.0f;
    float  total_win      = 0.0f;
    float  total_loss     = 0.0f;
    float  total_win_pct  = 0.0f;
    float  total_loss_pct = 0.0f;

    float       rets[RET_BUF]{};
    std::size_t ret_head = 0;
    std::size_t ret_cnt  = 0;

    void push_ret(float r) noexcept {
        rets[ret_head & (RET_BUF - 1)] = r;
        ++ret_head;
        if (ret_cnt < RET_BUF) ++ret_cnt;
    }

    void get_rets(float* dst, std::size_t n) const noexcept {
        n = (n < ret_cnt) ? n : ret_cnt;
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t slot = (ret_head - 1 - (n - 1 - i)) & (RET_BUF - 1);
            dst[i] = rets[slot];
        }
    }
};

// ── Global state ──────────────────────────────────────────────────────────────
static SymRisk     g_syms[MAX_RS]{};
static int         g_nsyms   = 0;
static float       g_equity[RET_BUF]{};
static std::size_t g_eq_head = 0;
static std::size_t g_eq_cnt  = 0;
static std::mutex  g_risk_mu;

// Shared annualisation constant.  BR-01 FIX: 1-min bars → 525,600 bars/year.
static constexpr float BARS_PER_YEAR = 525600.0f;

// ── findOrCreate ──────────────────────────────────────────────────────────────
// BUG-RK1 FIX: strncmp n=16.
static int findOrCreate(const char* sym) noexcept {
    std::lock_guard lk(g_risk_mu);
    for (int i = 0; i < g_nsyms; ++i)
        if (std::strncmp(g_syms[i].name, sym, 16) == 0) return i;
    if (g_nsyms >= static_cast<int>(MAX_RS)) return -1;
    std::strncpy(g_syms[g_nsyms].name, sym, 15);
    g_syms[g_nsyms].name[15] = '\0';
    return g_nsyms++;
}

// ── findOnly ─────────────────────────────────────────────────────────────────
// BUG-RK2 FIX: strncmp n=16.
static int findOnly(const char* sym) noexcept {
    for (int i = 0; i < g_nsyms; ++i)
        if (std::strncmp(g_syms[i].name, sym, 16) == 0) return i;
    return -1;
}
static_assert(true || &findOnly, "findOnly retained for locked callers");

// ── Performance math ──────────────────────────────────────────────────────────
static float sharpe(const float* rets, std::size_t n) noexcept {
    if (n < ) return 0.0f;
    const float s = simd::std_dev(rets, n);
    return s < 1e-10f ? 0.0f
                      : simd::mean(rets, n) / s * std::sqrt(BARS_PER_YEAR);
}

// BUG-RK4 FIX: removed dead neg[] array; accumulate down_sq directly.
static float sortino(const float* rets, std::size_t n) noexcept {
    if (n < 4) return 0.0f;
    float down_sq = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
        if (rets[i] < 0.0f) down_sq += rets[i] * rets[i];
    const float dsd = (n > 0) ? std::sqrt(down_sq / static_cast<float>(n) + 1e-20f)
                               : 1e-10f;
    return simd::mean(rets, n) / (dsd + 1e-10f) * std::sqrt(BARS_PER_YEAR);
}

static float maxDD(const float* eq, std::size_t n) noexcept {
    if (n < 2) return 0.0f;
    float peak = eq[0], mdd = 0.0f;
    for (std::size_t i = 1; i < n; ++i) {
        if (eq[i] > peak) peak = eq[i];
        const float dd = (peak - eq[i]) / (peak + 1e-10f);
        if (dd > mdd) mdd = dd;
    }
    return mdd;
}

static float histVaR(const float* rets, std::size_t n,
                     float conf = 0.95f) noexcept {
    if (n < 10) return 0.02f;
    float tmp[RET_BUF];
    const std::size_t m = (n < 100) ? n : 100;
    std::copy(rets + (n - m), rets + n, tmp);
    return simd::clamp(
        std::abs(simd::percentile(tmp, m, (1.0f - conf) * 100.0f)),
        0.005f, 0.10f);
}

// Quarter-Kelly fraction.
//
// BUG-RK7 FIX: cold-start return now reads globalConfig().kelly_coldstart
// instead of the hardcoded literal 0.0105f.
//
// 
static float kelly(float win_rate, float avg_win, float avg_loss) noexcept {
    const float coldstart = globalConfig().kelly_coldstart;  // BUG-RK7 FIX
    if (avg_win <= 0.0f || avg_loss <= 0.0f || win_rate <= 0.0f)
        return coldstart;
    const float b    = avg_win / (avg_loss + 1e-10f);
    const float full = win_rate - (1.0f - win_rate) / (b + 1e-10f);
    if (full <= 0.0f) return 0.003f;
    return simd::clamp(full * f, f, f);
}

// Volatility scalar: size toward the configured daily vol target.
//
// BUG-RK7 FIX: removed default parameter `float target = 0.015f`.
//
static float volScalar(float current_vol) noexcept {  // BUG-RK7 FIX
    if (current_vol <= 0.0f) return 1.0f;
    const float target = globalConfig().vol_target_pct;  // BUG-RK7 FIX
    return simd::clamp(target / (current_vol + 1e-10f), 0.3f, 2.0f);
}

// ── Public API ────────────────────────────────────────────────────────────────

TradeStats getStats(const char* symbol) noexcept {
    const int idx = findOrCreate(symbol);
    if (idx < 0) return {};

    std::lock_guard lk(g_risk_mu);
    const auto& s = g_syms[idx];
    if (s.trades == 0) return {};

    TradeStats ts;
    ts.trades   = s.trades;
    ts.win_rate = static_cast<float>(s.wins) / static_cast<float>(s.trades);
    ts.avg_win  = (s.wins   > 0) ? s.total_win_pct  / static_cast<float>(s.wins)   : 0.0f;
    ts.avg_loss = (s.losses > 0) ? s.total_loss_pct / static_cast<float>(s.losses) : 0.0f;
    ts.total_pnl = s.total_pnl;

    float tmp[RET_BUF];
    s.get_rets(tmp, s.ret_cnt);

    const std::size_t eq_n = (g_eq_cnt < RET_BUF) ? g_eq_cnt : RET_BUF;
    float eq[RET_BUF];
    for (std::size_t i = 0; i < eq_n; ++i)
        eq[i] = g_equity[(g_eq_head - 1 - (eq_n - 1 - i)) & (RET_BUF - 1)];

    ts.sharpe       = sharpe (tmp, s.ret_cnt);
    ts.sortino      = sortino(tmp, s.ret_cnt);
    ts.max_drawdown = maxDD  (eq,  eq_n);

    // BUG-RK3 FIX: Calmar uses BARS_PER_YEAR (525600), not 3650.
    ts.calmar = (ts.max_drawdown > 1e-10f)
                ? simd::mean(tmp, s.ret_cnt) * BARS_PER_YEAR / ts.max_drawdown
                : 0.0f;
    return ts;
}

void getPortfolioPerformance(float& pnl, float& wr,
                              float& sh, float& so, float& mdd) noexcept {
    std::lock_guard lk(g_risk_mu);

    static constexpr std::size_t MAX_ALL = MAX_RS * 64;
    float all_rets[MAX_ALL];
    std::size_t total_rets   = 0;
    std::size_t total_trades = 0, tw = 0;
    pnl = 0.0f;

    for (int i = 0; i < g_nsyms; ++i) {
        const auto& s = g_syms[i];
        pnl          += s.total_pnl;
        total_trades += static_cast<std::size_t>(s.trades);
        tw           += static_cast<std::size_t>(s.wins);

        const std::size_t take = (s.ret_cnt < 64) ? s.ret_cnt : 64;
        if (total_rets + take <= MAX_ALL) {
            s.get_rets(all_rets + total_rets, take);
            total_rets += take;
        }
    }

    wr = (total_trades > 0)
         ? static_cast<float>(tw) / static_cast<float>(total_trades)
         : 0.0f;

    sh = sharpe (all_rets, total_rets);
    so = sortino(all_rets, total_rets);

    const std::size_t en = (g_eq_cnt < RET_BUF) ? g_eq_cnt : RET_BUF;
    float eq[RET_BUF];
    for (std::size_t i = 0; i < en; ++i)
        eq[i] = g_equity[(g_eq_head - 1 - (en - 1 - i)) & (RET_BUF - 1)];
    mdd = maxDD(eq, en);
}

void recordTrade(const char* symbol, float net_pnl,
                 float entry_size, float balance) noexcept {
    const int idx = findOrCreate(symbol);
    if (idx < 0) [[unlikely]] return;

    std::lock_guard lk(g_risk_mu);
    auto& s = g_syms[idx];
    ++s.trades;
    s.total_pnl += net_pnl;

    const float ret = (entry_size > 1e-10f) ? net_pnl / entry_size : 0.0f;
    if (net_pnl > 0.0f) {
        ++s.wins;
        s.total_win     += net_pnl;
        s.total_win_pct += ret;
    } else {
        ++s.losses;
        s.total_loss     += std::abs(net_pnl);
        s.total_loss_pct += std::abs(ret);
    }
    s.push_ret(ret);

    g_equity[g_eq_head & (RET_BUF - 1)] = balance + net_pnl;
    ++g_eq_head;
    if (g_eq_cnt < RET_BUF) ++g_eq_cnt;
}

PositionParams calculatePosition(float balance, float entry_price,
                                  const Features&   features,
                                  const RegimeInfo& /*regime*/,
                                  SignalDir dir,
                                  float confidence) noexcept {
    const auto& cfg          = globalConfig();
    const float leverage     = static_cast<float>(cfg.leverage);
    const float rrr          = cfg.tp_rr_ratio;
    const float min_notional = cfg.min_notional_usd;

    // Step 1: Kelly — BUG-RK7 FIX: cold-start now reads kelly_coldstart from config.
    const TradeStats st  = getStats(features.symbol);
    const float base_pct = kelly(st.win_rate, st.avg_win, st.avg_loss);

    // Step 2: Volatility scalar — BUG-RK7 FIX: reads vol_target_pct from config.
    const float vs = volScalar(features.volatility);

    // Step 3: VaR scalar.
    const int idx   = findOrCreate(features.symbol);
    float     var95 = f;
    if (idx >= 0) {
        float tmp[RET_BUF];
        std::size_t ret_cnt_snap;
        {
            std::lock_guard lk(g_risk_mu);
            g_syms[idx].get_rets(tmp, g_syms[idx].ret_cnt);
            ret_cnt_snap = g_syms[idx].ret_cnt;
        }
        var95 = histVaR(tmp, ret_cnt_snap);
    }
    const float var_s = simd::clamp(f / (var95 + 1e-10f), f, f);

    // ── Step 4: Autonomous tier selection ────────────────────────────────────
    int tier_score = 1;

    if (confidence > 0.82f) ++tier_score;
    const bool dir_is_buy   = (dir == SignalDir::BUY);
    const bool hmm_confirms = (dir_is_buy  && features.hmm_state == 2)
                           || (!dir_is_buy && features.hmm_state == 0);
    if (hmm_confirms) ++tier_score;

    if (features.garch_vol > 1.0f) --tier_score;
    const bool funding_crowded =
        (dir_is_buy  && features.funding_zscore >  2.0f) ||
        (!dir_is_buy && features.funding_zscore < -2.0f);
    if (funding_crowded) --tier_score;
    if (var95 > 0.04f)  --tier_score;

    // BUG-RK5 FIX: std::clamp<int> — not simd::clamp(float,float,float).
    tier_score = std::clamp(tier_score, 0, 2);

    const CapitalTier& tier = (tier_score == 0) ? cfg.tier_conservative
                            : (tier_score == 2) ? cfg.tier_aggressive
                            :                     cfg.tier_standard;

    // Step 5: Final risk fraction — Kelly × vol × VaR, bounded by tier + hard limits.
    const float raw_pct  = base_pct * vs * var_s;
    const float tier_pct = simd::clamp(raw_pct, cfg.min_risk_pct, tier.risk_pct);
    const float risk_pct = simd::clamp(tier_pct, cfg.min_risk_pct, cfg.max_risk_pct);
    const float max_margin_p = tier.max_margin_pct;

    // ── Steps 6–8: SL, margin, TP ────────────────────────────────────────────
    // BUG-RK6 FIX: PositionParams p at outer function scope.
    PositionParams p;
    p.var_95     = var95;
    p.vol_scalar = vs;

    // Step 6: ATR-based SL distance (1.5 × ATR14).
    const float atr     = (features.atr > 0.0f) ? features.atr
                                                  : entry_price * 0.015f;
    const float sl_dist = atr * 1.5f;
    const float sl_pct  = sl_dist / (entry_price + 1e-10f);

    if (sl_pct < f) {
        p.valid = false;
        std::strncpy(p.reject_reason, "sl_too_tight", sizeof(p.reject_reason) - 1);
        return p;
    }
    if (sl_pct > f) {
        p.valid = false;
        std::strncpy(p.reject_reason, "sl_too_wide", sizeof(p.reject_reason) - 1);
        return p;
    }

    // Step 7: Margin = risk_usd / (leverage × sl_pct).
    float risk_usd = balance * risk_pct;
    float margin   = risk_usd / (leverage * sl_pct + 1e-10f);
    float notional = margin * leverage;

    if (notional < min_notional) {
        margin   = min_notional / leverage;
        notional = min_notional;
    }
    const float max_margin = balance * max_margin_p;
    if (margin > max_margin) {
        margin   = max_margin;
        notional = margin * leverage;
    }

    // Step 8: SL / TP prices.
    const float tp_dist = sl_dist * rrr;
    if (dir == SignalDir::BUY) {
        p.sl_price = entry_price - sl_dist;
        p.tp_price = entry_price + tp_dist;
    } else {
        p.sl_price = entry_price + sl_dist;
        p.tp_price = entry_price - tp_dist;
    }

    p.margin   = std::round(margin      * 100.0f) / 100.0f;
    p.notional = std::round(notional    * 100.0f) / 100.0f;
    p.sl_price = std::round(p.sl_price  * 1e6f)   / 1e6f;
    p.tp_price = std::round(p.tp_price  * 1e6f)   / 1e6f;
    p.sl_pct   = std::round(sl_pct      * 1e6f)   / 1e6f;
    p.risk_usd = std::round(risk_usd    * 100.0f) / 100.0f;
    p.risk_pct = risk_pct;
    p.valid    = true;

    std::fprintf(stderr,
        "[RISK] %s tier=%d risk=%.2f%% margin=$%.2f"
        " garch=%.2f hmm=%u fund_z=%.2f var95=%.3f\n",
        features.symbol, tier_score, risk_pct * 100.0f, p.margin,
        features.garch_vol, static_cast<unsigned>(features.hmm_state),
        features.funding_zscore, var95);

    return p;
}

RiskCheckResult checkRisk(const char* symbol, float balance,
                           int current_positions,
                           float daily_start_bal) noexcept {
    (void)symbol;
    const auto& cfg = globalConfig();
    RiskCheckResult r;

    if (current_positions >= cfg.max_open_positions) {
        std::strncpy(r.reason, "max_positions", sizeof(r.reason) - 1);
        return r;
    }
    if (balance < cfg.hard_floor_usd) {
        std::strncpy(r.reason, "balance_below_floor", sizeof(r.reason) - 1);
        return r;
    }
    if (daily_start_bal > 0.0f) {
        const float dd = (daily_start_bal - balance) / (daily_start_bal + 1e-10f);
        if (dd >= cfg.max_daily_drawdown) {
            std::strncpy(r.reason, "daily_drawdown_limit", sizeof(r.reason) - 1);
            return r;
        }
    }
    r.allowed = true;
    std::strncpy(r.reason, "ok", sizeof(r.reason) - 1);
    return r;
}

} // namespace tqc
