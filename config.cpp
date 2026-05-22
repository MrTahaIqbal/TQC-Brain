/*
 * config.cpp  -  BigBoyAgent TQC Brain | Taha Iqbal
 *
 * Implements globalConfig() singleton, loadConfig(), and AppConfig::validate().
 * Reads BRAIN_SECRET from the environment and all trading parameters
 * from settings.json (nlohmann/json).
 *
 * ── BUGS FIXED IN THIS VERSION ──────────────────────────────────────────────
 *
 * BUG-CC1  loadConfig() returned true after a JSON parse exception.
 *          If settings.json existed but was malformed (e.g. a trailing comma,
 *          unclosed brace, or invalid UTF-8 from a text editor saving with
 *          BOM), nlohmann::json::parse_error was thrown, caught, a warning
 *          was printed, and then the function returned true with cfg.num_pairs
 *          == 0.  The Brain would start, find no symbols to process, and
 *          silently produce empty signal arrays every cycle — no trades, no
 *          errors, just silence.
 *          FIX: changed catch block to return false on any parse exception.
 *          Caller (main.cpp) must treat this as a fatal startup error.
 *
 * BUG-CC2  loadConfig() returned true even when cfg.num_pairs == 0.
 *          This could happen if: (a) settings.json had no "PAIRS" key,
 *          (b) "PAIRS" was an empty array [], or (c) BUG-CC1 above.
 *          In all cases the Brain starts with an empty symbol universe,
 *          processes nothing, and reports "success" to the executor.
 *          FIX: added an explicit num_pairs == 0 check before returning true;
 *          prints a clear error and returns false.
 *
 * BUG-CC3  Error message on JSON failure said "using defaults" which was
 *          misleading — there are no hardcoded default pairs.  The message
 *          implied partial operation was possible when it was not.
 *          FIX: error message now says "settings.json is required — cannot
 *          start Brain." and returns false (see BUG-CC1 above).
 *
 * BUG-CC4  No validation of loaded numeric values.
 *          A settings.json with "LEVERAGE": 0 would pass loadConfig()
 *          successfully, then cause division-by-zero in risk_engine.cpp
 *          when computing margin = notional / leverage.  Similarly
 *          "RISK_PCT": -0.05 would produce negative risk amounts that
 *          silently flip long/short sizing logic.
 *          FIX: added AppConfig::validate() which is called at the end of
 *          loadConfig() and checks all critical numeric bounds.
 *
 * BUG-CC5  Capital tier ordering was never validated.
 *          If settings.json accidentally set tier_aggressive.risk_pct <
 *          tier_conservative.risk_pct, the tier selector in risk_engine.cpp
 *          would assign higher risk to "safe" regimes and lower risk to
 *          high-confidence setups — the opposite of intended behaviour.
 *          FIX: validate() checks conservative ≤ standard ≤ aggressive
 *          for both risk_pct and max_margin_pct.
 *
 * BUG-CC6  Hard risk bounds were never validated against tiers.
 *          If min_risk_pct > tier_conservative.risk_pct, the clamp would
 *          always override the tier — meaning the "conservative" tier was
 *          actually riskier than its definition.
 *          FIX: validate() checks min_risk_pct ≤ tier_conservative.risk_pct
 *          and max_risk_pct ≥ tier_aggressive.risk_pct.
 *
 * BUG-CC7  Secret key was not trimmed of whitespace.
 *          On some Linux shells, exporting BRAIN_SECRET with a trailing
 *          newline (e.g. export BRAIN_SECRET=$(cat secret.txt) where the
 *          file has a trailing newline) stores the newline in the env var.
 *          strncpy then copies the newline into cfg.secret_key, causing all
 *          HMAC-SHA256 authentication checks to fail at runtime.
 *          FIX: rtrim_secret() strips trailing whitespace/newlines.
 *
 * BUG-CC8  [NEW] slip_pct, hard_floor_usd, ranker weights, vol_target_pct,
 *          and kelly_coldstart were never loaded from JSON.
 *          All five ranker weight fields exist in AppConfig and all affect
 *          signal selection, yet none appeared in the JSON parsing loop.
 *          vol_target_pct and kelly_coldstart (FINDING-S5 fix) were also
 *          absent despite being declared in config.hpp.  All fields were
 *          permanently pinned to their in-code defaults with no way to
 *          override from settings.json.
 *          FIX: slip_pct, hard_floor_usd, and all five ranker weights are
 *          now loaded from the top-level JSON object.  vol_target_pct and
 *          kelly_coldstart are loaded from the CAPITAL_TIERS block.
 *          Startup summary extended to log their resolved values.
 *
 * BUG-CC9  [NEW] validate() was missing checks for slip_pct, hard_floor_usd,
 *          ranker weight sum, vol_target_pct, and kelly_coldstart.
 *          A negative slip_pct inflates simulated PnL on every entry.
 *          Ranker weights that don't sum to 1.0 silently scale all rank
 *          scores up or down, making min_rank_score comparisons meaningless.
 *          vol_target_pct <= 0 produces a division-by-zero in volScalar().
 *          kelly_coldstart <= 0 makes cold-start Kelly sizing zero on every
 *          new symbol, effectively disabling trading on unknown assets.
 *          FIX: validate() now checks all five new fields and verifies the
 *          weight sum is within 1e-4f of 1.0f.
 *
 * BUG-CC10 [NEW] Individual pair strings not validated before loading.
 *          The PAIRS parsing loop copied every array element without checking
 *          that the string was non-empty and alphanumeric.  A config entry
 *          like "PAIRS": ["BTCUSDT", "", "INVALID PAIR"] loaded an empty
 *          slot (which has_pair("") may spuriously match) and a symbol with
 *          an embedded space (which causes Binance API failures at runtime
 *          with a cryptic HTTP 400 error and no clear log pointing back to
 *          the config).  Errors should be caught at load time, not at the
 *          exchange API boundary minutes later during the first trade cycle.
 *          FIX: each symbol in PAIRS is validated: non-empty, all
 *          alphanumeric, and length within [3, 15] chars.  Invalid symbols
 *          are skipped with a WARNING; if all symbols are invalid the load
 *          fails via the existing num_pairs == 0 guard (BUG-CC2 FIX).
 *
 * BUG-CC11 [NEW] loadConfig() mutated cfg before validate() succeeded;
 *          caller received partial state on validation failure.
 *          If validate() failed, loadConfig() returned false but cfg (which
 *          is typically g_config — the global singleton) had already been
 *          partially written with the new values.  Any code path that ignored
 *          the return value, or that read globalConfig() after a failed reload,
 *          would operate on a corrupted configuration object silently.
 *          FIX: all JSON values are loaded into a local AppConfig tmp{}.
 *          Only after tmp.validate() succeeds is cfg = tmp performed as an
 *          atomic overwrite.  The caller's cfg object is never modified on
 *          any failure path.
 *
 * BUG-CC12 [NEW] fee_pct upper bound in validate() was too permissive and
 *          self-contradictory.
 *          The original bound (fee_pct > 0.01f, i.e. 1%) is 25× the Binance
 *          futures taker rate (~0.04%).  A user who typed "0.05" intending
 *          "0.05%" (= 0.0005) would pass validation with a fee 50× too large,
 *          silently distorting all PnL simulations and Kelly sizing.  The
 *          error message compounded this by listing both "1% max" and "Check
 *          FUTURES taker fee ~0.04%" — two numbers that contradict each other.
 *          FIX: upper bound tightened to 0.002f (0.2%), which provides
 *          generous headroom for exotic fee structures while catching obvious
 *          mistakes.  Error message updated to be consistent.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include "config.hpp"
#include "json.hpp"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <cctype>    // std::isspace, std::isalnum
#include <cmath>     // std::fabs

using json = nlohmann::json;

namespace tqc {

// ── Singleton ─────────────────────────────────────────────────────────────────
static AppConfig g_config;
AppConfig& globalConfig() noexcept { return g_config; }

// ── Internal helpers ──────────────────────────────────────────────────────────

// BUG-CC7 FIX: strip trailing whitespace from secret key copied from env.
static void rtrim_secret(char* buf, std::size_t buf_size) noexcept {
    if (!buf || buf_size == 0) return;
    std::size_t len = std::strlen(buf);
    while (len > 0 && (std::isspace(static_cast<unsigned char>(buf[len - 1])))) {
        buf[--len] = '\0';
    }
}

// BUG-CC10 FIX: validate a single symbol string from the PAIRS array.
// Returns true if the symbol is safe to use as a Binance instrument name.
// Accepts: 3–15 characters, all alphanumeric (A-Z, 0-9), non-empty.
// Rejects: empty strings, strings with spaces/punctuation, excessively long
// strings that would overflow the pairs[i][16] buffer.
static bool is_valid_symbol(const char* s, std::size_t len) noexcept {
    if (len < 3 || len > 15) return false;
    for (std::size_t i = 0; i < len; ++i) {
        if (!std::isalnum(static_cast<unsigned char>(s[i]))) return false;
    }
    return true;
}

// ── AppConfig::validate() ─────────────────────────────────────────────────────
// BUG-CC4 / BUG-CC5 / BUG-CC6 FIX: full sanity check on all loaded values.
// Reports ALL invalid fields in one pass so the user can fix settings.json
// in a single edit-restart cycle.
bool AppConfig::validate(FILE* err_out) const noexcept {
    bool ok = true;

    auto fail = [&](const char* msg) noexcept {
        std::fprintf(err_out, "[CONFIG] INVALID: %s\n", msg);
        ok = false;
    };

    // ── Pairs ─────────────────────────────────────────────────────────────────
    if (num_pairs == 0)
        fail("No trading pairs defined — add at least one symbol to PAIRS.");
    if (num_pairs > MAX_PAIRS)
        fail("num_pairs exceeds MAX_PAIRS (24) — reduce the PAIRS list.");

    // ── Risk scalars ──────────────────────────────────────────────────────────
    if (leverage <= 0)
        fail("LEVERAGE must be > 0 (e.g. 3). Value 0 causes division-by-zero.");
    if (leverage > 125)
        fail("LEVERAGE > 125 exceeds Binance Futures maximum. Use <= 20 for safety.");
    if (risk_per_trade_pct <= 0.0f || risk_per_trade_pct > 0.10f)
        fail("RISK_PCT must be in (0, 0.10]. Values > 10% are dangerously large.");
    if (max_margin_pct <= 0.0f || max_margin_pct > 1.0f)
        fail("max_margin_pct must be in (0, 1.0].");
    if (max_daily_drawdown <= 0.0f || max_daily_drawdown > 0.50f)
        fail("MAX_DAILY_DD must be in (0, 0.50]. Values > 50% are irrecoverable.");
    if (min_notional_usd < 5.0f)
        fail("MIN_NOTIONAL must be >= 5 USD (Binance minimum is ~5 USD).");
    if (max_open_positions <= 0 || max_open_positions > 20)
        fail("MAX_OPEN_POSITIONS must be in [1, 20].");

    // BUG-CC12 FIX: tightened fee_pct upper bound from 0.01 (1%) to 0.002
    // (0.2%).  Binance futures taker is ~0.04%; 0.2% provides generous
    // headroom while catching obvious input errors (e.g. typing 0.05 when
    // intending 0.05% = 0.0005).
    if (fee_pct < 0.0f || fee_pct > 0.002f)
        fail("FEE_PCT must be in [0, 0.002] (0.2% max). "
             "Binance futures taker fee is ~0.0004 (0.04%). "
             "Did you accidentally enter a percentage instead of a fraction?");

    if (tp_rr_ratio < 1.0f)
        fail("TP_RR_RATIO must be >= 1.0 (reward must cover risk + fees).");
    if (hard_floor_usd < 0.0f)
        fail("hard_floor_usd must be >= 0.");

    // BUG-CC9 FIX: slip_pct bounds check (was entirely absent).
    // Negative slip_pct inflates simulated PnL by modelling a fill that is
    // better than the order price — unrealistic and silently optimistic.
    if (slip_pct < 0.0f || slip_pct > 0.005f)
        fail("slip_pct must be in [0, 0.005] (0.5% max). "
             "Typical crypto futures slippage is 0.01–0.05%.");

    // ── Hard risk bounds ──────────────────────────────────────────────────────
    if (min_risk_pct <= 0.0f)
        fail("min_risk_pct must be > 0.");
    if (max_risk_pct <= min_risk_pct)
        fail("max_risk_pct must be > min_risk_pct.");
    if (max_risk_pct > 0.10f)
        fail("max_risk_pct > 10% is dangerously large. Use <= 0.05 (5%).");

    // ── Capital tier ordering (BUG-CC5 FIX) ──────────────────────────────────
    if (tier_standard.risk_pct < tier_conservative.risk_pct)
        fail("tier_standard.risk_pct must be >= tier_conservative.risk_pct.");
    if (tier_aggressive.risk_pct < tier_standard.risk_pct)
        fail("tier_aggressive.risk_pct must be >= tier_standard.risk_pct.");
    if (tier_standard.max_margin_pct < tier_conservative.max_margin_pct)
        fail("tier_standard.max_margin_pct must be >= tier_conservative.max_margin_pct.");
    if (tier_aggressive.max_margin_pct < tier_standard.max_margin_pct)
        fail("tier_aggressive.max_margin_pct must be >= tier_standard.max_margin_pct.");

    // ── Tier vs hard bound consistency (BUG-CC6 FIX) ─────────────────────────
    if (min_risk_pct > tier_conservative.risk_pct)
        fail("min_risk_pct > tier_conservative.risk_pct: "
             "hard floor overrides conservative tier — conservative tier is meaningless.");
    if (max_risk_pct < tier_aggressive.risk_pct)
        fail("max_risk_pct < tier_aggressive.risk_pct: "
             "hard ceiling overrides aggressive tier — aggressive tier is meaningless.");

    // ── Vol target and Kelly cold-start (BUG-CC9 FIX, per FINDING-S5) ────────
    // vol_target_pct: volScalar() divides by this value.  Zero or negative
    // produces division-by-zero or an inverted scalar (more vol → more size).
    if (vol_target_pct <= 0.0f || vol_target_pct > 0.20f)
        fail("vol_target_pct must be in (0, 0.20]. "
             "Default 0.015 = 1.5% daily vol target. "
             "Values > 20% disable the volatility scalar effectively.");

    // kelly_coldstart: used as the Kelly fraction before enough trade history
    // exists for a reliable estimate. Zero makes new-symbol sizing always zero.
    // Must also respect the hard risk bounds.
    if (kelly_coldstart <= 0.0f)
        fail("kelly_coldstart must be > 0. "
             "A zero value disables position sizing on all new symbols.");
    if (kelly_coldstart < min_risk_pct)
        fail("kelly_coldstart < min_risk_pct: cold-start Kelly would be clipped "
             "to the hard floor on every use — kelly_coldstart is meaningless.");
    if (kelly_coldstart > max_risk_pct)
        fail("kelly_coldstart > max_risk_pct: cold-start Kelly would be clipped "
             "to the hard ceiling on every use — set kelly_coldstart <= max_risk_pct.");

    // ── Ranker ────────────────────────────────────────────────────────────────
    if (top_n_signals <= 0)
        fail("TOP_N_SIGNALS must be >= 1.");
    if (top_n_signals > static_cast<int>(num_pairs))
        fail("TOP_N_SIGNALS > num_pairs: would rank more signals than symbols available.");
    if (min_rank_score < 0.0f || min_rank_score > 1.0f)
        fail("MIN_RANK_SCORE must be in [0, 1].");

    // BUG-CC9 FIX: ranker weight sum must be 1.0 ± tolerance.
    // If weights sum to anything other than 1.0, all rank scores are
    // systematically scaled up or down, making the min_rank_score threshold
    // compare against a mis-scaled value.  The tolerance (1e-4f) accounts for
    // the accumulated float rounding error across five addition operations.
    {
        float wsum = weight_conf + weight_hurst + weight_adx
                   + weight_tf_agree + weight_vol_q;
        if (std::fabs(wsum - 1.0f) > 1e-4f) {
            std::fprintf(err_out,
                "[CONFIG] INVALID: Ranker weights must sum to 1.0 "
                "(got %.6f = %.4f + %.4f + %.4f + %.4f + %.4f). "
                "Adjust RANKER_WEIGHTS in settings.json.\n",
                wsum,
                weight_conf, weight_hurst, weight_adx,
                weight_tf_agree, weight_vol_q);
            ok = false;
        }
        if (weight_conf     < 0.0f) fail("weight_conf must be >= 0.");
        if (weight_hurst    < 0.0f) fail("weight_hurst must be >= 0.");
        if (weight_adx      < 0.0f) fail("weight_adx must be >= 0.");
        if (weight_tf_agree < 0.0f) fail("weight_tf_agree must be >= 0.");
        if (weight_vol_q    < 0.0f) fail("weight_vol_q must be >= 0.");
    }

    // ── Secret key ────────────────────────────────────────────────────────────
    if (secret_key[0] == '\0')
        fail("secret_key is empty — BRAIN_SECRET env var was not loaded.");

    return ok;
}

// ── loadConfig() ──────────────────────────────────────────────────────────────
bool loadConfig(AppConfig& cfg, const char* json_path) noexcept {

    // BUG-CC11 FIX: load into a temporary AppConfig first.
    // cfg (typically g_config) is only overwritten after validate() succeeds.
    // This ensures the caller's config object is never left in a partially-
    // modified state on any failure path — loadConfig() is atomic w.r.t. cfg.
    AppConfig tmp{};

    // ── 1. Secret key from environment ────────────────────────────────────────
    const char* secret = std::getenv("BRAIN_SECRET");
    if (!secret || secret[0] == '\0') {
        std::fprintf(stderr,
            "[CONFIG] ERROR: BRAIN_SECRET environment variable is not set.\n"
            "         Set it to the shared API key used by executor_main.\n");
        return false;
    }
    std::strncpy(tmp.secret_key, secret, sizeof(tmp.secret_key) - 1);
    tmp.secret_key[sizeof(tmp.secret_key) - 1] = '\0';

    // BUG-CC7 FIX: strip trailing newline/whitespace from the env var.
    rtrim_secret(tmp.secret_key, sizeof(tmp.secret_key));
    if (tmp.secret_key[0] == '\0') {
        std::fprintf(stderr,
            "[CONFIG] ERROR: BRAIN_SECRET is set but contains only whitespace.\n");
        return false;
    }

    // ── 2. Settings from JSON ────────────────────────────────────────────────
    std::ifstream f(json_path);
    if (!f.is_open()) {
        std::fprintf(stderr,
            "[CONFIG] ERROR: %s not found — cannot start Brain.\n"
            "         Create settings.json with the correct PAIRS and CAPITAL_TIERS\n"
            "         matching the executor's configuration before restarting.\n",
            json_path);
        return false;
    }

    // BUG-CC1 / BUG-CC3 FIX: JSON parse failure is fatal, not a recoverable
    // warning.  Return false; the temporary is discarded, cfg is unchanged.
    try {
        json j;
        f >> j;

        // ── PAIRS ─────────────────────────────────────────────────────────────
        if (j.contains("PAIRS") && j["PAIRS"].is_array()) {
            tmp.num_pairs = 0;
            for (auto& p : j["PAIRS"]) {
                if (tmp.num_pairs >= AppConfig::MAX_PAIRS) {
                    std::fprintf(stderr,
                        "[CONFIG] WARNING: PAIRS list exceeds MAX_PAIRS (%zu); "
                        "extra entries ignored.\n",
                        AppConfig::MAX_PAIRS);
                    break;
                }

                // BUG-CC10 FIX: validate each symbol before accepting it.
                std::string s = p.get<std::string>();
                if (!is_valid_symbol(s.c_str(), s.size())) {
                    std::fprintf(stderr,
                        "[CONFIG] WARNING: Skipping invalid symbol \"%s\" in PAIRS. "
                        "Symbols must be 3-15 alphanumeric characters (e.g. BTCUSDT).\n",
                        s.c_str());
                    continue;
                }

                std::strncpy(tmp.pairs[tmp.num_pairs], s.c_str(), 15);
                tmp.pairs[tmp.num_pairs][15] = '\0';
                ++tmp.num_pairs;
            }
        }

        // ── Scalar risk parameters ────────────────────────────────────────────
        if (j.contains("LEVERAGE"))          tmp.leverage           = j["LEVERAGE"];
        if (j.contains("RISK_PCT"))          tmp.risk_per_trade_pct = j["RISK_PCT"];
        if (j.contains("MAX_DAILY_DD"))      tmp.max_daily_drawdown = j["MAX_DAILY_DD"];
        if (j.contains("MIN_NOTIONAL"))      tmp.min_notional_usd   = j["MIN_NOTIONAL"];
        if (j.contains("MAX_OPEN_POSITIONS"))tmp.max_open_positions = j["MAX_OPEN_POSITIONS"];
        if (j.contains("FEE_PCT"))           tmp.fee_pct            = j["FEE_PCT"];
        if (j.contains("TP_RR_RATIO"))       tmp.tp_rr_ratio        = j["TP_RR_RATIO"];

        // BUG-CC8 FIX: slip_pct and hard_floor_usd were not loaded from JSON.
        if (j.contains("SLIP_PCT"))          tmp.slip_pct           = j["SLIP_PCT"];
        if (j.contains("HARD_FLOOR_USD"))    tmp.hard_floor_usd     = j["HARD_FLOOR_USD"];

        // ── Ranker ────────────────────────────────────────────────────────────
        if (j.contains("TOP_N_SIGNALS"))     tmp.top_n_signals      = j["TOP_N_SIGNALS"];
        if (j.contains("MIN_RANK_SCORE"))    tmp.min_rank_score     = j["MIN_RANK_SCORE"];

        // BUG-CC8 FIX: ranker weights were not loaded from JSON.
        // Grouped under optional RANKER_WEIGHTS object in settings.json:
        //   "RANKER_WEIGHTS": {
        //       "confidence": 0.35,
        //       "hurst":      0.20,
        //       "adx":        0.15,
        //       "tf_agree":   0.15,
        //       "vol_q":      0.15
        //   }
        if (j.contains("RANKER_WEIGHTS")) {
            const auto& rw = j["RANKER_WEIGHTS"];
            if (rw.contains("confidence")) tmp.weight_conf     = rw["confidence"];
            if (rw.contains("hurst"))      tmp.weight_hurst    = rw["hurst"];
            if (rw.contains("adx"))        tmp.weight_adx      = rw["adx"];
            if (rw.contains("tf_agree"))   tmp.weight_tf_agree = rw["tf_agree"];
            if (rw.contains("vol_q"))      tmp.weight_vol_q    = rw["vol_q"];
        }

        // ── Autonomous capital tiers ──────────────────────────────────────────
        if (j.contains("CAPITAL_TIERS")) {
            const auto& ct = j["CAPITAL_TIERS"];

            auto load_tier = [&](const char* name, CapitalTier& tier) {
                if (!ct.contains(name)) return;
                const auto& t = ct[name];
                if (t.contains("risk_pct"))       tier.risk_pct       = t["risk_pct"];
                if (t.contains("max_margin_pct")) tier.max_margin_pct = t["max_margin_pct"];
            };

            load_tier("conservative", tmp.tier_conservative);
            load_tier("standard",     tmp.tier_standard);
            load_tier("aggressive",   tmp.tier_aggressive);

            // Hard global bounds
            if (ct.contains("min_risk_pct")) tmp.min_risk_pct = ct["min_risk_pct"];
            if (ct.contains("max_risk_pct")) tmp.max_risk_pct = ct["max_risk_pct"];

            // BUG-CC8 FIX (FINDING-S5): vol_target_pct and kelly_coldstart
            // are now loaded from the CAPITAL_TIERS block.
            // settings.json example:
            //   "CAPITAL_TIERS": {
            //       "vol_target_pct":  0.015,
            //       "kelly_coldstart": 0.0105,
            //       ...
            //   }
            if (ct.contains("vol_target_pct"))  tmp.vol_target_pct  = ct["vol_target_pct"];
            if (ct.contains("kelly_coldstart")) tmp.kelly_coldstart = ct["kelly_coldstart"];
        }

    } catch (const std::exception& e) {
        // BUG-CC1 FIX: fatal — cfg is NOT modified (tmp is on the stack).
        std::fprintf(stderr,
            "[CONFIG] ERROR: Failed to parse %s: %s\n"
            "         settings.json is required — cannot start Brain.\n"
            "         Validate your JSON at https://jsonlint.com before retrying.\n",
            json_path, e.what());
        return false;
    }

    // BUG-CC2 FIX: treat zero pairs as a fatal error after successful parse.
    if (tmp.num_pairs == 0) {
        std::fprintf(stderr,
            "[CONFIG] ERROR: No valid trading pairs found in %s.\n"
            "         Add a PAIRS array with at least one valid symbol, e.g.:\n"
            "         \"PAIRS\": [\"BTCUSDT\", \"ETHUSDT\"]\n"
            "         Symbols must be 3-15 alphanumeric characters.\n",
            json_path);
        return false;
    }

    // ── 3. Validate all loaded values ─────────────────────────────────────────
    // BUG-CC4 / CC5 / CC6 / CC9 FIX: full sanity check before committing.
    // tmp is validated; cfg is not touched until this succeeds.
    if (!tmp.validate(stderr)) {
        std::fprintf(stderr,
            "[CONFIG] ERROR: Configuration failed validation. "
            "Fix the issues listed above in %s before restarting.\n",
            json_path);
        return false;
    }

    // ── 4. Atomic commit (BUG-CC11 FIX) ─────────────────────────────────────
    // All checks passed.  Overwrite the caller's config in one assignment.
    cfg = tmp;

    // ── 5. Startup summary ───────────────────────────────────────────────────
    std::fprintf(stderr, "[CONFIG] Loaded %zu pair(s) from %s:\n",
                 cfg.num_pairs, json_path);
    for (std::size_t i = 0; i < cfg.num_pairs; ++i)
        std::fprintf(stderr, "         [%zu] %s\n", i + 1, cfg.pairs[i]);

    std::fprintf(stderr,
        "[CONFIG] leverage=%d  risk_pct=%.4f  max_dd=%.3f  "
        "max_positions=%d  top_n=%d\n",
        cfg.leverage,
        cfg.risk_per_trade_pct,
        cfg.max_daily_drawdown,
        cfg.max_open_positions,
        cfg.top_n_signals);

    std::fprintf(stderr,
        "[CONFIG] Fees — fee_pct=%.4f%%  slip_pct=%.4f%%  "
        "hard_floor=%.2f USD\n",
        cfg.fee_pct    * 100.0f,
        cfg.slip_pct   * 100.0f,
        cfg.hard_floor_usd);

    std::fprintf(stderr,
        "[CONFIG] Tiers — conservative: %.3f%%/%.0f%%  "
        "standard: %.3f%%/%.0f%%  aggressive: %.3f%%/%.0f%%\n",
        cfg.tier_conservative.risk_pct  * 100.0f,
        cfg.tier_conservative.max_margin_pct * 100.0f,
        cfg.tier_standard.risk_pct      * 100.0f,
        cfg.tier_standard.max_margin_pct * 100.0f,
        cfg.tier_aggressive.risk_pct    * 100.0f,
        cfg.tier_aggressive.max_margin_pct * 100.0f);

    std::fprintf(stderr,
        "[CONFIG] Hard bounds — min_risk=%.3f%%  max_risk=%.3f%%\n",
        cfg.min_risk_pct * 100.0f,
        cfg.max_risk_pct * 100.0f);

    // BUG-CC8 FIX: log the two previously-invisible risk engine constants.
    std::fprintf(stderr,
        "[CONFIG] Risk engine — vol_target=%.3f%%  kelly_coldstart=%.4f%%\n",
        cfg.vol_target_pct   * 100.0f,
        cfg.kelly_coldstart  * 100.0f);

    std::fprintf(stderr,
        "[CONFIG] Ranker weights — conf=%.2f  hurst=%.2f  adx=%.2f  "
        "tf_agree=%.2f  vol_q=%.2f  (sum=%.4f)\n",
        cfg.weight_conf,
        cfg.weight_hurst,
        cfg.weight_adx,
        cfg.weight_tf_agree,
        cfg.weight_vol_q,
        cfg.weight_conf + cfg.weight_hurst + cfg.weight_adx
        + cfg.weight_tf_agree + cfg.weight_vol_q);

    return true;
}

} // namespace tqc
