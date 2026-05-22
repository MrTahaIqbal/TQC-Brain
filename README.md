# BigBoyAgent-Brain v12

> **Institutional-grade C++20 algorithmic trading brain — SIMD · Lock-Free · Zero-Heap hot path**

A production-quality ML inference and signal generation engine for cryptocurrency perpetual futures, built as a stateless HTTP microservice. Designed to run on HuggingFace Spaces (free tier) and communicate with a Windows 10 execution layer over HTTPS.

---

## What This Is

This repository contains the **ML Brain** component of a full-stack systematic trading system. It receives OHLCV market data from an execution layer, runs a complete quantitative pipeline, and returns ranked trade signals with position sizing parameters.

The system demonstrates:
- **SIMD-accelerated technical indicators** (AVX2, scalar fallback for ARM/any hardware)
- **Lock-free concurrent data structures** (MPSC ring buffer, per-symbol spinlocks, CAS atomic weights)
- **Institutional risk sizing** (quarter-Kelly + Historical VaR + vol-targeting + autonomous tier selection)
- **Online-adaptive signal weights** (lock-free CAS accuracy tracking, no heap)
- **Statistical models** (GARCH(1,1), HAR-RV, TSMOM, ARIMA(1,1,1))
- **Market regime detection** (Hurst R/S, ADX, 3-state HMM Viterbi with online Baum-Welch M-step)
- **Zero heap allocation** from HTTP request arrival to JSON response write

---

## Architecture

```
Executor (Windows 10)
  └── POST /predict ──► HttpServer (4 workers, MPSC ring buffer)
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
              BarStore (SoA)       ConfigLayer
              24 × 128 bars        Kelly / Vol Target
              per-symbol lock      Tier Weights
                    │
                    ▼
           FeatureEngine (SIMD AVX2)
           Z-score · RSI(Wilder) · MACD
           Stoch K/D · ATR · VWAP · OB Imbalance
           HAR-RV · GARCH(1,1) · TSMOM · ARIMA(1,1,1)
                    │
                    ▼
           RegimeEngine
           Hurst R/S · ADX · Regression Slope
           3-State HMM Viterbi + Online M-Step
                    │
                    ▼
           SignalEngine
           9-Factor Weighted Score · Sigmoid Dispatch
           Adaptive CAS Weights · BUY / SELL / HOLD
                    │
                    ▼
           RiskEngine
           Quarter-Kelly · Hist VaR 95 · Vol Scalar
           Autonomous Tier · ATR SL · RR TP
                    │
                    ▼
           AlphaEngine
           Momentum · Low-Vol · Vol Surge · Quality
           Cross-Sectional Rank Normalisation
                    │
                    ▼
           Ranker (insertion sort, correlation dedup)
                    │
                    ▼
           JsonBuilder (direct buffer, RFC 8259)
                    │
                    ▼
  └── JSON Response ──► Executor
```

---

## Technical Highlights

### Memory Design
- **Struct-of-Arrays** bar storage — contiguous float arrays per indicator, cache-line aligned
- **Object pool** with lock-free ABA-safe CAS (`ObjectPool<T,N>`) — raw aligned storage, symmetric placement-new contract
- **Thread-local scratch buffers** — zero per-request heap allocation
- **Stack-allocated fixed arrays** for all intermediate data; `std::vector` banned on the hot path

### Concurrency
- **SPSC ring buffer** — producer/consumer separated across cache lines, zero-contention IPC
- **MPSC ring buffer** — commit-flag per slot eliminates head-before-payload race; CAS failure uses `acquire` ordering for ARM correctness
- **Per-symbol spinlocks** — independent symbols process fully in parallel; contention only on same-symbol concurrent calls
- **Lock-free accuracy CAS** — `packed uint64` [wins | total] updated atomically; rolling halving prevents overflow

### SIMD
- AVX2: 8 floats/cycle for sum, variance, dot product, abs-diff
- Horizontal reduction via `_mm256_extractf128_ps` + `_mm_add_ps` ladder
- Scalar fallback compiles and passes identical tests on ARM/aarch64
- `-march=x86-64-v3` for portable cross-Xeon binary in Docker

### Signal Engine
- Regime-specific weight arrays resolved at **compile time** via template specialisation — no runtime branch
- Wilder EMA (k=1/period) for RSI/ATR vs standard EMA (k=2/(period+1)) for MACD — correct per Bloomberg/TradingView convention
- Stochastic K/D composite: 70% crossover momentum + 30% absolute position
- NOISE regime structural double-gate: `conf_cap=0.65` < `buy_thresh=0.72` → BUY/SELL impossible by construction

### Risk Sizing
- Quarter-Kelly with per-symbol rolling 512-bar return history
- Volatility scalar toward configurable daily vol target (default 1.5%)
- Historical VaR 95th percentile from circular buffer — no heap, `nth_element` O(n)
- Autonomous tier selection: Conservative / Standard / Aggressive based on confidence × HMM × GARCH × funding z-score
- All parameters configurable from `settings.json` — none hardcoded

---

## File Structure

```
brain/
├── main.cpp               # HTTP routing, processSymbol(), pre-warm
├── config.hpp / .cpp      # AppConfig, loadConfig(), validate()
├── types.hpp              # FixStr<N>, Features, Signal, RegimeInfo, ...
├── bar_store.hpp          # SoA BarStore, per-symbol ring buffer, ScratchBuffers
├── simd_math.hpp / .cpp   # AVX2 kernels: sum, mean, var, ema, wilder_ema, atr, percentile
├── feature_engine.hpp / .cpp  # All technical indicators + HAR-RV/GARCH/TSMOM/ARIMA
├── regime_engine.hpp / .cpp   # Hurst R/S, ADX, regression slope, HMM Viterbi
├── signal_engine.hpp / .cpp   # 9-factor scoring, adaptive weights, BUY/SELL/HOLD
├── risk_engine.hpp / .cpp     # Kelly, VaR, vol scalar, tier selection, SL/TP
├── alpha_engine.hpp / .cpp    # Cross-sectional alpha factors
├── tracking.hpp / .cpp    # Analytics, PortfolioEngine, PositionManager, LatencyTracker
├── object_pool.hpp        # Lock-free ABA-safe object pool + RAII PoolHandle
├── ring_buffer.hpp        # SPSC + MPSC lock-free ring buffers
├── http_server.hpp / .cpp # POSIX HTTP/1.1 server, MPSC acceptor/worker threads
├── json_builder.hpp       # Direct-buffer JSON serialiser, RFC 8259 escaping
├── ranker.hpp / .cpp      # Insertion sort + correlation deduplication
├── CMakeLists.txt         # C++20, AVX2, ASAN/UBSan debug target
└── Dockerfile             # Multi-stage: gcc:13-bookworm builder → debian:bookworm-slim runtime
```

---

## Building

### Docker (HuggingFace Spaces / any Linux)
```bash
docker build -t brain .
docker run -e BRAIN_SECRET=your_secret_here -p 7860:7860 brain
```

### Local (Linux / WSL2)
```bash
# Requirements: GCC 13+, CMake 3.22+, Ninja
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target brain --parallel
BRAIN_SECRET=your_secret_here ./build/brain
```

### Debug (ASAN + UBSan)
```bash
cmake -B build_debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build_debug --target brain --parallel
BRAIN_SECRET=dev_secret ./build_debug/brain
```

---

## Configuration

Copy `settings.json.example` to `settings.json` and edit:

```json
{
    "PAIRS": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    "LEVERAGE": 3,
    "RISK_PCT": 0.0105,
    "MAX_DAILY_DD": 0.05,
    "MAX_OPEN_POSITIONS": 3,
    "FEE_PCT": 0.0004,
    "SLIP_PCT": 0.0002,
    "TP_RR_RATIO": 2.0,
    "HARD_FLOOR_USD": 15.0,
    "TOP_N_SIGNALS": 5,
    "MIN_RANK_SCORE": 0.45,
    "RANKER_WEIGHTS": {
        "confidence": 0.35,
        "hurst":      0.20,
        "adx":        0.15,
        "tf_agree":   0.15,
        "vol_q":      0.15
    },
    "CAPITAL_TIERS": {
        "vol_target_pct":  0.015,
        "kelly_coldstart": 0.0105,
        "conservative": { "risk_pct": 0.005, "max_margin_pct": 0.15 },
        "standard":     { "risk_pct": 0.010, "max_margin_pct": 0.25 },
        "aggressive":   { "risk_pct": 0.020, "max_margin_pct": 0.40 },
        "min_risk_pct": 0.003,
        "max_risk_pct": 0.030
    }
}
```

Set `BRAIN_SECRET` as an environment variable (never in source):
```bash
export BRAIN_SECRET=your_shared_key_here
```

---

## API

### `POST /predict`
Accepts a JSON body with `account_info` and `market_data` array.
Returns `ranked_signals` (top-N) and `all_signals` (all symbols processed).

### `GET /health`
Returns `{"status":"ok","uptime_s":...}` — used by Docker HEALTHCHECK.

### `GET /analytics`
Returns per-symbol Sharpe, Sortino, Calmar, win rate, and portfolio P&L.

---

## Audit Trail

This codebase was subjected to a full senior-engineer audit covering:

| Category | Issues Found & Fixed |
|---|---|
| Concurrency (data races, lock ordering, ARM memory model) | C1–C4 + 7 new |
| Statistical correctness (RSI, GARCH, HMM, EMA seeding) | S1–S5 + 4 new |
| Performance (SIMD, branch prediction, false sharing) | P1–P5 + 3 new |
| Security (Content-Length overflow, auth bypass, SIGPIPE) | R1–R3 |
| Architecture (config promotion, signal pipeline gaps) | A1–A4 + 5 new |
| Edge cases (zero division, NaN propagation, cold-start) | E1–E3 + 6 new |

Key fixes include:
- **HMM observation symbol 3 was structurally unreachable** (contradictory conditions) — silent model degradation for 6+ months
- **SELL confidence bypassed `conf_cap`** — NOISE regime SELL signals could return 0.80 confidence despite a cap of 0.65
- **MPSC CAS failure ordering `relaxed`** — ARM Graviton correctness bug (worked on x86 TSO by accident)
- **`stoch_d` pipeline gap** — computed, stored, used for sizing, but never copied to output Signal or serialised to JSON
- **`volScalar()` and `kelly()` ignored config** — `vol_target_pct` and `kelly_coldstart` were configurable at the config layer but the risk engine read hardcoded constants

---

## Requirements

- **Compiler**: GCC 13+ or Clang 16+ (C++20: `std::jthread`, `std::span`, `std::bit_cast`)
- **CPU**: x86-64 with AVX2 (Intel Haswell 2013+ / AMD Ryzen 2017+). Scalar fallback compiles on ARM.
- **OS**: Linux (POSIX sockets). WSL2 supported for development.
- **Memory**: ~60 KiB working set per request (all stack/BSS, no heap in hot path)
- **Dependencies**: `nlohmann/json` single-header (downloaded automatically by Dockerfile)

---

## License

MIT — see [LICENSE](LICENSE).

---

## Author

**Taha Iqbal** — Self-taught systems developer  
Built solo as a demonstration of institutional-grade quantitative systems engineering in C++20.
