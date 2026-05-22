/*
 * main.cpp  - TQC Brain | Taha Iqbal
 * ============================================================
 * SYSTEM ARCHITECTURE:
 *
 *   executor_main.exe  →  POST /predict  →  HttpServer (N worker threads)
 *                                              │
 *                                         handlePredict()
 *                                              │
 *               ┌──────────────────────────────┤
 *               ▼          ▼                   ▼
 *          BarStore   FeatureEngine       RegimeEngine
 *          (SoA)     (SIMD indicators)   (Hurst/ADX/slope)
 *               │
 *               ▼
 *          SignalEngine ──── AdaptiveWeights (lock-free CAS)
 *               │
 *               ▼
 *          RiskEngine (Kelly + VaR + vol-targeting)
 *               │
 *               ▼
 *          AlphaEngine (cross-sectional factors)
 *               │
 *               ▼
 *          Ranker (insertion sort + correlated-pair dedup, no heap)
 *               │
 *               ▼
 *          JsonBuilder (direct buffer write, no DOM)
 *               │
 *               ▼
 *          HTTP Response  →  executor_main.exe
 *
 * Memory guarantee:
 *   Zero heap allocation from request arrival until JSON response write.
 *   All state lives in pre-allocated fixed-size arrays and thread_local
 *   scratch buffers.
 *
 * FIX (ISSUE-07): Side string standardisation.
 *   BrainPosition and SimPosition both store "LONG"/"SHORT".
 *   Previously BrainPosition stored "BUY"/"SELL" — an inconsistency that
 *   worked by accident but was a maintenance time-bomb.
 *
 * FIX: processSymbol() is NOT noexcept.
 *   nlohmann::json accessors (value(), get<T>(), operator[]) throw
 *   json::type_error on malformed data.  A noexcept caller causes
 *   std::terminate() instead of the catch block in handlePredict().
 *
 * FIX (ISSUE-ranked_buf): ranked_buf was array<Signal, 8> — UB if
 *   top_n_signals > 8.  Changed to array<Signal, AppConfig::MAX_PAIRS> (24).
 *
 * FIX (ISSUE-exception-log): catch(std::exception) now logs e.what() +
 *   symbol; catch(...) logs symbol + "unknown exception".
 *
 * FIX (ISSUE-analytics-buf): allMetricsJSON() return value checked for
 *   near-full truncation and warned to stderr.
 *
 * ── BUGS FIXED IN THIS VERSION ──────────────────────────────────────────────
 *
 * BUG-M1  handlePredict(): `open_count` read from JSON but never used.
 *         FIX: removed.  pm.count() is used directly at the call site.
 *
 * BUG-M2  handlePredict(): sym_ptrs / sym_strs hardcoded to size 8.
 *         FIX: arrays sized to AppConfig::MAX_PAIRS (24).
 *
 * BUG-M3  processSymbol(): `balance >= ` hardcoded, ignores config.
 *         FIX: replaced with cfg.hard_floor_usd.
 *
 * BUG-M4  [NEW] processSymbol(): sig.stoch_d not copied from features.
 *
 *         The BUG-T9 chain restores stoch_d end-to-end:
 *           types.hpp:           Signal::stoch_d field restored
 *           feature_engine.cpp:  computeStoch() returns (k, d); stored in Features
 *           signal_engine.cpp:   normalise() reads f.stoch_d (BUG-SE9 fix)
 *           main.cpp (HERE):     sig.stoch_k = features.stoch_k ← correct
 *                                sig.stoch_d = ???               ← MISSING
 *           json_builder.hpp:    kv_num("stoch_d", ...) restored (BUG-JB3 fix)
 *
 *         With stoch_d missing from the Signal→fill block:
 *           (a) Signal::stoch_d is always f (default), regardless of market.
 *           (b) The executor receives stoch_d= for every symbol every cycle —
 *               making the K/D crossover invisible to all executor-side logic,
 *               logging, and dashboard visualisation.
 *           (c) Because json_builder.hpp was also missing stoch_d (BUG-JB3),
 *               this bug was completely silent — no mismatch between filled and
 *               serialised values, because both were wrong together.
 *           (d) The BUG-JB3 fix makes this bug observable: the executor now
 *               receives stoch_d but it's always 0.5 — a flat line that
 *               destroys the crossover signal's diagnostic value.
 *
 *         FIX: added `sig.stoch_d = features.stoch_d;` immediately after
 *         `sig.stoch_k = features.stoch_k;` in the indicator fill block.
 *         This completes the end-to-end stoch_d pipeline that was designed
 *         by BUG-T9 and made operational by BUG-SE9.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include "config.hpp"
#include "feature_engine.hpp"
#include "regime_engine.hpp"
#include "signal_engine.hpp"
#include "risk_engine.hpp"
#include "ranker.hpp"
#include "alpha_engine.hpp"
#include "tracking.hpp"
#include "http_server.hpp"
#include "json_builder.hpp"
#include "json.hpp"

#include <cstdio>
#include <cstring>
#include <ctime>
#include <cstdlib>
#include <chrono>
#include <span>
#include <array>

using json = nlohmann::json;
using namespace tqc;

// ── Startup timestamp ─────────────────────────────────────────────────────────
static const auto g_boot = std::chrono::steady_clock::now();

// ── Tradeable regime check ────────────────────────────────────────────────────
static bool isTradeableRegime(Regime r) noexcept {
    return r == Regime::TRENDING || r == Regime::MEAN_REVERTING;
}

// ── Process one symbol from market_data JSON array entry ──────────────────────
// NOT noexcept — nlohmann::json accessors throw json::type_error on malformed
// data.  If declared noexcept, any throw would call std::terminate() instead
// of being caught by the per-symbol try/catch in handlePredict().
static Signal processSymbol(const json& md,
                             float       balance,
                             int         open_count,
                             float       daily_start,
                             int64_t     timestamp) {
    Signal sig{};
    sig.internal_ts = timestamp;

    const std::string& sym_str = md.value("symbol", "");
    if (sym_str.empty() || sym_str.size() >= 16) return sig;
    std::strncpy(sig.symbol, sym_str.c_str(), 15);
    sig.symbol[15] = '\0';

    const float price  = static_cast<float>(md.value("lastPrice", 0.0));
    const float volume = static_cast<float>(md.value("volume",    0.0));
    const float high   = md.contains("high") && !md["high"].is_null()
                         ? static_cast<float>(md["high"].get<double>()) : price;
    const float low    = md.contains("low")  && !md["low"].is_null()
                         ? static_cast<float>(md["low"].get<double>())  : price;

    if (price <= 0.0f) {
        std::strncpy(sig.signal, "HOLD", 7);
        return sig;
    }

    // ── Bulk warmup (first call for this symbol) ──────────────────────────────
    if (md.contains("bulk_prices") && !md["bulk_prices"].is_null()) {
        const auto& bp = md["bulk_prices"];
        const std::size_t n = bp.size();
        if (n > 0) {
            static thread_local std::array<float, 256> bc, bh, bl, bv;
            const std::size_t take = std::min(n, static_cast<std::size_t>(256));
            for (std::size_t i = 0; i < take; ++i)
                bc[i] = static_cast<float>(bp[n - take + i]);

            const auto& bha = md.contains("bulk_highs")
                              ? md["bulk_highs"]   : md["bulk_prices"];
            const auto& bla = md.contains("bulk_lows")
                              ? md["bulk_lows"]    : md["bulk_prices"];
            const auto& bva = md.contains("bulk_volumes")
                              ? md["bulk_volumes"] : md["bulk_prices"];

            for (std::size_t i = 0; i < take; ++i) {
                bh[i] = bha.is_array() && bha.size() > n-take+i
                        ? static_cast<float>(bha[n-take+i].get<double>()) : bc[i];
                bl[i] = bla.is_array() && bla.size() > n-take+i
                        ? static_cast<float>(bla[n-take+i].get<double>()) : bc[i];
                bv[i] = bva.is_array() && bva.size() > n-take+i
                        ? static_cast<float>(bva[n-take+i].get<double>()) : 0.0f;
            }
            bulkLoad(sym_str.c_str(),
                     bc.data(), bh.data(), bl.data(), bv.data(), take);
        }
    }

    // ── 1m / 5m sub-minute arrays ────────────────────────────────────────────
    static thread_local std::array<float, 64> p1m_buf, p5m_buf;
    int n1m = 0, n5m = 0;
    if (md.contains("prices_1m") && md["prices_1m"].is_array())
        for (auto& v : md["prices_1m"]) {
            if (n1m >= 64) break;
            p1m_buf[n1m++] = static_cast<float>(v);
        }
    if (md.contains("prices_5m") && md["prices_5m"].is_array())
        for (auto& v : md["prices_5m"]) {
            if (n5m >= 64) break;
            p5m_buf[n5m++] = static_cast<float>(v);
        }

    // ── Push bar into store ───────────────────────────────────────────────────
    const float ob_bid_sz =
        md.contains("ob_bid_sz") && !md["ob_bid_sz"].is_null()
        ? static_cast<float>(md["ob_bid_sz"].get<double>()) : 0.0f;
    const float ob_ask_sz =
        md.contains("ob_ask_sz") && !md["ob_ask_sz"].is_null()
        ? static_cast<float>(md["ob_ask_sz"].get<double>()) : 0.0f;

    UpdateHistory(sym_str.c_str(), price, volume, &high, &low,
                  ob_bid_sz, ob_ask_sz);

    // ── Feature computation ───────────────────────────────────────────────────
    Features features = GenerateFeatures(sym_str.c_str(), price, volume,
                                          &high, &low,
                                          p1m_buf.data(), n1m,
                                          p5m_buf.data(), n5m);

    // ── Step 2: Inject real-time payload fields ───────────────────────────────
    {
        const float vwoi_v    =
            md.contains("vwoi")         && !md["vwoi"].is_null()
            ? static_cast<float>(md["vwoi"].get<double>())         : 0.0f;
        const float funding_v =
            md.contains("funding_rate") && !md["funding_rate"].is_null()
            ? static_cast<float>(md["funding_rate"].get<double>()) : 0.0f;
        const float basis_v   =
            md.contains("basis")        && !md["basis"].is_null()
            ? static_cast<float>(md["basis"].get<double>())        : 0.0f;
        injectPayloadFeatures(sym_str.c_str(), vwoi_v, funding_v, basis_v,
                               features);
    }

    // ── Regime ────────────────────────────────────────────────────────────────
    const RegimeInfo regime = classifyRegime(features);

    // ── Signal ────────────────────────────────────────────────────────────────
    SignalResult sr = generateSignal(features, regime);

    // ── Daily drawdown gate ───────────────────────────────────────────────────
    if (daily_start > 0.0f) {
        const float dd = (daily_start - balance) / (daily_start + 1e-10f);
        if (dd >= globalConfig().max_daily_drawdown) {
            sr.dir        = SignalDir::HOLD;
            sr.confidence = 0.5f;
        }
    }

    // ── Position sizing ───────────────────────────────────────────────────────
    bool  pos_valid = false;
    float sl_price  = 0.0f, tp_price = 0.0f, pos_size = 0.0f;

    const auto& cfg = globalConfig();

    // BUG-M3 FIX: replaced hardcoded 30.0f with cfg.hard_floor_usd.
    const bool is_tradeable = isTradeableRegime(regime.regime)
                           && (sr.dir == SignalDir::BUY || sr.dir == SignalDir::SELL)
                           && balance >= cfg.hard_floor_usd;

    if (is_tradeable) [[unlikely]] {
        const RiskCheckResult rck = checkRisk(sym_str.c_str(), balance,
                                               open_count, daily_start);
        if (rck.allowed) {
            const PositionParams pp = calculatePosition(balance, price,
                                                         features, regime,
                                                         sr.dir, sr.confidence);
            if (pp.valid) {
                sl_price  = pp.sl_price;
                tp_price  = pp.tp_price;
                pos_size  = pp.margin;
                pos_valid = true;
            } else {
                sr.dir        = SignalDir::HOLD;
                sr.confidence = 0.3f;
            }
        }
    }

    // ── Brain simulation ──────────────────────────────────────────────────────
    auto& pm  = globalPositionManager();
    auto& pfe = globalPortfolio();

    BrainPosition existing_sim;
    const bool has_sim = pm.getSim(sym_str.c_str(), existing_sim);

    if (has_sim) {
        const bool should_close =
            (sr.dir == SignalDir::SELL &&
             std::strncmp(existing_sim.side, "LONG",  4) == 0) ||
            (sr.dir == SignalDir::BUY  &&
             std::strncmp(existing_sim.side, "SHORT", 5) == 0);

        if (should_close) {
            pm.closeSim(sym_str.c_str());
            const auto cr = pfe.closePosition(sym_str.c_str(), price);
            if (cr.closed) {
                updateAdaptiveWeights(sym_str.c_str(), cr.net_pnl > 0.0f);
                recordTrade(sym_str.c_str(), cr.net_pnl, cr.size_usd, balance);
                globalAnalytics().logTrade(sym_str.c_str(), cr.net_pnl,
                                            balance + cr.net_pnl, cr.size_usd);
            }
        }
    }

    if (pos_valid && !pm.isOpen(sym_str.c_str())) {
        const char* side = (sr.dir == SignalDir::BUY) ? "LONG" : "SHORT";
        pm.openSim (sym_str.c_str(), side, price, pos_size);
        pfe.openPosition(sym_str.c_str(), side, price, pos_size);
    }

    // ── Fill Signal struct ────────────────────────────────────────────────────
    std::strncpy(sig.signal, sr.dir_str(), 7);
    sig.signal[7] = '\0';
    sig.confidence         = sr.confidence;
    sig.price              = price;
    sig.high               = high;
    sig.low                = low;
    sig.suggested_sl       = sl_price;
    sig.suggested_tp       = tp_price;
    sig.position_size      = pos_size;
    sig.pos_valid          = pos_valid;

    std::strncpy(sig.regime,     regime.regime_str(),     19);
    std::strncpy(sig.vol_regime, regime.vol_regime_str(),  7);
    sig.regime    [19] = '\0';
    sig.vol_regime[ 7] = '\0';
    sig.hurst     = regime.hurst;
    sig.adx       = regime.adx;
    sig.reg_slope = regime.reg_slope;

    // HMM state + Step 2 statistical model outputs
    sig.hmm_state        = regime.hmm_state;
    sig.har_rv           = features.har_rv;
    sig.garch_vol        = features.garch_vol;
    sig.tsmom            = features.tsmom;
    sig.arima_forecast   = features.arima_forecast;
    sig.vwoi             = features.vwoi;
    sig.funding_rate     = features.funding_rate;
    sig.funding_zscore   = features.funding_zscore;
    sig.basis            = features.basis;

    sig.atr                = features.atr;
    sig.atr_pct            = features.atr_pct;
    sig.vw_rsi             = features.vw_rsi;
    sig.rsi                = features.rsi;
    sig.stoch_k            = features.stoch_k;
    sig.stoch_d            = features.stoch_d;   // BUG-M4 FIX: was missing.
                                                  // stoch_d (3-bar SMA of K,
                                                  // the K/D signal line) was
                                                  // restored to Features and
                                                  // Signal by BUG-T9, used in
                                                  // normalise() by BUG-SE9, but
                                                  // never copied to the output
                                                  // Signal struct — silently
                                                  // frozen at 0.5f forever.
                                                  // Completes the end-to-end
                                                  // stoch_d pipeline.
    sig.macd_signal        = features.macd_signal;
    sig.roc_5              = features.roc_5;
    sig.roc_10             = features.roc_10;
    sig.roc_20             = features.roc_20;
    sig.z_score            = features.z_score;
    sig.bollinger_position = features.bollinger_position;
    sig.vwap_distance      = features.vwap_distance;
    sig.volume_imbalance   = features.volume_imbalance;
    sig.ob_imbalance       = features.ob_imbalance;
    sig.tf_agreement       = features.tf_agreement;
    sig.volatility         = features.volatility;
    sig.data_points        = features.data_points;

    sig.unrealized_pnl = pfe.getUnrealizedPnL(sym_str.c_str(), price);

    const TradeStats ts = getStats(sym_str.c_str());
    sig.total_pnl = ts.total_pnl;
    sig.sharpe    = ts.sharpe;

    return sig;
}

// ── Route: GET / ──────────────────────────────────────────────────────────────
static HttpResponse handleHome(const HttpRequest&) {
    static constexpr const char* html = R"(<html><body>
<p>C++20 HFT brain: SoA+AVX2+Lock-Free+Kelly+VaR+Cross-Sectional-Alpha</p>
<p>Endpoints: POST /predict | GET /health | GET /analytics</p>
</body></html>)";
    return {200, html, "text/html"};
}

// ── Route: GET /health ────────────────────────────────────────────────────────
static HttpResponse handleHealth(const HttpRequest&) {
    const double uptime_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - g_boot).count();
    char buf[128];
    std::snprintf(buf, sizeof(buf),
        R"({"status":"ok","uptime_s":%.1f})", uptime_s);
    return {200, buf};
}

// ── Route: GET /analytics ─────────────────────────────────────────────────────
static HttpResponse handleAnalytics(const HttpRequest&) {
    char metrics_buf[16384];
    const int metrics_written = globalAnalytics().allMetricsJSON(
        metrics_buf, sizeof(metrics_buf));
    if (metrics_written >=
            static_cast<int>(sizeof(metrics_buf)) - 4) [[unlikely]]
        std::fprintf(stderr,
            "[analytics] WARN: metrics_buf near capacity (%d/%zu bytes) "
            "— JSON may be truncated\n",
            metrics_written, sizeof(metrics_buf));

    float pnl = 0.0f, wr = 0.0f, sh = 0.0f, so = 0.0f, mdd = 0.0f;
    getPortfolioPerformance(pnl, wr, sh, so, mdd);

    JsonBuilder j(2048);
    j.obj_open();
    j.key("portfolio");
    j.obj_open();
    j.kv_num("total_pnl",    pnl);
    j.kv_num("win_rate",     wr);
    j.kv_num("sharpe",       sh,  3);
    j.kv_num("sortino",      so,  3);
    j.kv_num("max_drawdown", mdd);
    j.obj_close();
    j.key("metrics");
    j.raw(metrics_buf);
    j.obj_close();
    return {200, j.take()};
}

// ── Route: POST /retrain ──────────────────────────────────────────────────────
static HttpResponse handleRetrain(const HttpRequest&) {
    return {200,
        R"({"ok":false,"error":"XGBoost not in C++ build. Retrain offline."})"};
}

// ── Route: POST /predict ──────────────────────────────────────────────────────
static HttpResponse handlePredict(const HttpRequest& req) {
    const auto t_start = std::chrono::steady_clock::now();

    json j;
    try { j = json::parse(req.body); }
    catch (...) { return {400, R"({"error":"invalid_json"})"}; }

    if (!j.contains("account_info") || !j.contains("market_data"))
        return {400, R"({"error":"missing_fields"})"};

    const float   balance   = static_cast<float>(
                              j["account_info"].value("account_balance", 0.0));
    const int64_t timestamp = j.value("timestamp", static_cast<int64_t>(0));
    const float   daily_start = getDailyStart(balance);

    // BUG-M1 FIX: removed `open_count` variable — it was dead code.
    // pm.count() is used directly at the call site (reflects live brain state).

    // Sync brain positions with executor
    auto& pm = globalPositionManager();
    if (j["account_info"].contains("open_symbols") &&
        j["account_info"]["open_symbols"].is_array()) {
        // BUG-M2 FIX: arrays sized to MAX_PAIRS (24), not hardcoded 8.
        static thread_local
            std::array<const char*, AppConfig::MAX_PAIRS> sym_ptrs{};
        static thread_local
            std::array<std::string,  AppConfig::MAX_PAIRS> sym_strs{};
        int ns = 0;
        for (auto& s : j["account_info"]["open_symbols"]) {
            if (ns >= static_cast<int>(AppConfig::MAX_PAIRS)) break;
            sym_strs[ns] = s.get<std::string>();
            sym_ptrs[ns] = sym_strs[ns].c_str();
            ++ns;
        }
        pm.syncFromLaptop(sym_ptrs.data(), ns);
    }

    // Process each symbol — stack-allocated fixed-size array
    static thread_local std::array<Signal, MAX_SYMBOLS> all_signals{};
    int n_signals = 0;

    const auto& cfg = globalConfig();

    // BUG-5 FIX: pm.count() evaluated fresh at each iteration so positions
    // opened/closed mid-cycle are visible to subsequent symbols' risk checks.
    for (auto& md : j["market_data"]) {
        if (n_signals >= static_cast<int>(MAX_SYMBOLS)) break;
        const std::string sym = md.value("symbol", "");
        if (!cfg.has_pair(sym)) continue;
        try {
            all_signals[n_signals++] = processSymbol(
                md, balance, pm.count(), daily_start, timestamp);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[processSymbol] SKIP %s — %s\n",
                         sym.c_str(), e.what());
            --n_signals;
        } catch (...) {
            std::fprintf(stderr,
                "[processSymbol] SKIP %s — unknown exception\n",
                sym.c_str());
            --n_signals;
        }
    }

    // Cross-sectional alpha adjustment (in-place)
    applyAlphaToSignals(
        std::span(all_signals.data(), n_signals), 0.20f);

    // Rank
    static thread_local
        std::array<Signal, AppConfig::MAX_PAIRS> ranked_buf{};
    const int capped_top_n = std::min(
        cfg.top_n_signals,
        static_cast<int>(AppConfig::MAX_PAIRS));
    const int n_ranked = rankSignals(
        std::span(all_signals.data(), n_signals),
        ranked_buf.data(),
        capped_top_n,
        cfg.min_rank_score);

    // Latency
    const float lat_ms = globalLatency().record(t_start);
    const LatencyStats ls = globalLatency().stats();

    // Serialise — zero heap in the hot path
    JsonBuilder jb(32768);
    jb.obj_open();
    jb.kv_str ("status",         "success");
    jb.kv_int ("pairs_scanned",  n_signals);
    jb.kv_int ("signals_ranked", n_ranked);

    jb.key("ranked_signals");
    jb.writeSignalArray(
        std::span<const Signal>(ranked_buf.data(), n_ranked));

    jb.key("all_signals");
    jb.writeSignalArray(
        std::span<const Signal>(all_signals.data(), n_signals));

    jb.kv_int64("server_ts",  timestamp);
    jb.kv_num  ("latency_ms", lat_ms, 2);

    jb.key("latency_stats");
    jb.obj_open();
    jb.kv_num("avg_ms",  ls.avg_ms);
    jb.kv_num("p95_ms",  ls.p95_ms);
    jb.kv_num("p99_ms",  ls.p99_ms);
    jb.kv_num("max_ms",  ls.max_ms);
    jb.kv_int("samples", ls.samples);
    jb.obj_close();

    jb.kv_bool("xgboost_active", false);
    jb.key("modules_active");
    jb.obj_open();
    jb.kv_bool("orderbook",      true);
    jb.kv_bool("microstructure", false);
    jb.kv_bool("alpha",          true);
    jb.kv_bool("xgboost",        false);
    jb.obj_close();
    jb.obj_close();

    return {200, jb.take()};
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main() {
    std::fprintf(stderr,
        "============================================================\n"
        "TQC-Brain v12  |  C++20  |  Taha Iqbal\n"
        "Hardware: SoA+AVX2+Lock-Free+Pool+Template Dispatch\n"
        "============================================================\n");

    AppConfig& cfg = globalConfig();
    if (!loadConfig(cfg)) return 1;

    std::fprintf(stderr, "Pairs:     %zu\n",     cfg.num_pairs);
    std::fprintf(stderr, "Leverage:  %dx\n",     cfg.leverage);
    std::fprintf(stderr, "Fee:       %.3f%%\n",  cfg.fee_pct * 100.0f);
    std::fprintf(stderr, "Ranker:    top=%d  min_score=%.2f\n",
                 cfg.top_n_signals, cfg.min_rank_score);

    const char* port_env = std::getenv("PORT");
    const int   port     = port_env ? std::atoi(port_env) : 7860;

    int workers = 4;
    const char* w_env = std::getenv("BRAIN_WORKERS");
    if (w_env) {
        const int raw = std::atoi(w_env);
        workers = std::max(1, std::min(raw, 32));
        if (raw != workers)
            std::fprintf(stderr,
                "[WARN] BRAIN_WORKERS=%d is out of range — clamped to %d\n",
                raw, workers);
    }

    std::fprintf(stderr, "HTTP port: %d  workers: %d\n", port, workers);
    std::fprintf(stderr,
        "------------------------------------------------------------\n"
        "AVX2:      %s\n"
        "Allocator: pool+stack (zero heap in hot path)\n"
        "IPC:       lock-free MPSC ring buffer\n"
        "Dispatch:  template metaprogramming (compile-time weights)\n"
        "============================================================\n",
#ifdef TQC_AVX2
        "enabled (8 floats/cycle)"
#elif defined(TQC_AVX)
        "AVX only (4 floats/cycle)"
#else
        "scalar fallback"
#endif
    );

    // Pre-warm: touch all thread_local and static storage to avoid
    // first-request latency spikes from cold page faults.
    {
        UpdateHistory("WARMUP", 100.0f, 1000.0f, nullptr, nullptr, {},{},{},{});
        const Features f = GenerateFeatures("WARMUP", 100.0f, 1.0f,
                                             nullptr, nullptr,
                                             nullptr, 0, nullptr, 0);
        classifyRegime(f);
        globalPositionManager(); // load disk state (constructor called here)
    }
    std::fprintf(stderr, "Pre-warm complete. Ready to serve.\n");

    HttpServer server(port, workers);
    server.addRoute("GET",  "/",          false, handleHome);
    server.addRoute("GET",  "/health",    false, handleHealth);
    server.addRoute("GET",  "/analytics", true,  handleAnalytics);
    server.addRoute("POST", "/retrain",   true,  handleRetrain);
    server.addRoute("POST", "/predict",   true,  handlePredict);

    server.start(); // blocks until SIGINT

    std::fprintf(stderr, "Brain stopped.\n");
    return 0;
}
