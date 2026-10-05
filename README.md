# State-Dependent Neural Gating of VaR/ES Forecasts Trained on FZ Loss

Research code for a study of forecast combination for Value-at-Risk (VaR) and Expected Shortfall (ES) on Indian equities. A small neural gate maps the observable market state to combination weights over six econometric risk models. It is trained directly on the FZ0 joint VaR/ES loss.

> **Status:** Plans 1 and 2 of 3 complete (data, base models, combination baselines, gate and ablations). Plan 3 (significance tests and backtests) is not built yet, so the comparison below has no significance tests.

## Research question

Existing VaR/ES combination methods use static weights or weights based only on recent performance ([Taylor 2020](https://doi.org/10.1016/j.ijforecast.2019.05.014); [Storti & Wang 2023](https://doi.org/10.1002/for.2972); [Taylor & Wang 2025](https://arxiv.org/abs/2508.16919)). We ask:

> Do combination weights that depend on the market state, including implied volatility and the option-expiry calendar, and that are trained on a strictly consistent joint VaR/ES loss, improve tail-risk forecasts over static and performance-based combinations? If so, when?

## Method in brief

1. **Base models (6):** EWMA (filtered historical simulation tails), GARCH-t, EGARCH-t, GJR-GARCH-t, HAR on range-based variance, and an India VIX implied-volatility model. Each one produces a VaR/ES pair every day. Models are re-estimated every 21 trading days on an expanding window.
2. **Gate:** one MLP pooled across all assets. Inputs are trailing model losses, realized volatility, vol-of-vol, India VIX features, NSE expiry-calendar features, lagged returns, and an asset embedding. Outputs are softmax weights over the base models and a bounded scale factor:

   (VaR_t, ES_t) = exp(b_t) · Σ_i w_{i,t} · (VaR_{i,t}, ES_{i,t})

3. **Loss:** FZ0 ([Fissler & Ziegel 2016](https://doi.org/10.1214/16-AOS1439); [Patton, Ziegel & Chen 2019](https://doi.org/10.1016/j.jeconom.2018.10.008)), with shrinkage toward equal weights.
4. **Baselines:** each base model alone, equal-weight mean and median, and the Taylor (2020) combination methods.
5. **Ablations:** variance-level combination, and the gate without VIX features, without expiry features, without the scale head, and trained on QLIKE.
6. **Evaluation:** walk-forward test from 2023-01 to 2026-09, retraining the gate yearly. Methods are compared with mean FZ0 loss, Diebold–Mariano tests, and the Model Confidence Set. VaR is backtested with Kupiec, Christoffersen and DQ tests, and ES with the McNeil–Frey test. The primary tail level is α = 2.5%, with 1% and 5% as robustness checks.

## Data

Daily OHLC data for 2015-01-01 to 2026-09-30 from Yahoo Finance:

| Ticker | Name | Group |
|---|---|---|
| `^NSEI` | NIFTY 50 | Index |
| `^NSEBANK` | NIFTY Bank | Index |
| `ADANIENT.NS` | Adani Enterprises | High beta |
| `TATASTEEL.NS` | Tata Steel | High beta |
| `DLF.NS` | DLF | High beta |
| `HINDUNILVR.NS` | Hindustan Unilever | Defensive |
| `NESTLEIND.NS` | Nestle India | Defensive |
| `SUNPHARMA.NS` | Sun Pharma | Defensive |
| `^INDIAVIX` | India VIX | Implied volatility |

The raw downloads are frozen with a SHA-256 manifest. Every daily return larger than 10% in absolute value is audited for corporate actions, and the audit table is published with the results.

## Implementation status

| Component | Status |
|---|---|
| Data download, audit, expiry calendar | Done (Plan 1) |
| Base models and VaR/ES | Done (Plan 1) |
| FZ0 loss | Done (Plan 1) |
| Baseline combinations (equal, median, Taylor 2020) | Done (Plan 2) |
| Neural gate and walk-forward training | Done (Plan 2) |
| DM, MCS, VaR/ES backtests | Planned |
| Expiry and state analyses | Planned |

## Reproducing

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -q   # unit and leakage tests
make download      # optional: re-download raw data (overwrites the frozen files in data/raw/)
make all           # prepare panel -> walk-forward base-model forecasts
```

`make all` uses the frozen raw files committed in `data/raw/` and stops if their SHA-256 hashes do not match `data/raw/MANIFEST.csv`. It writes processed data to `data/processed/` (not committed) and tables to `results/tables/`.

## Current outputs

- `results/tables/data_summary.csv`: rows per asset, dropped no-trade rows, India VIX fills.
- `results/tables/outlier_audit.csv`: every daily return above 10% in absolute value, with the review decision and evidence. One row is removed: Adani Enterprises 2015-06-03 (unadjusted demerger).
- `results/tables/base_model_summary.csv`: for each asset, base model and tail level, the out-of-sample VaR hit rate, mean FZ0 loss, and number of refits that fell back to the previous parameters.
- `configs/expiry_rules.yaml`: NSE expiry rules with a source for each rule.

## Preliminary results (no significance tests yet)

Pooled mean FZ0 loss over 2023-01 to 2026-09, all 8 assets, from `results/tables/test_fz0.csv` (lower is better):

| Method | α = 1% | α = 2.5% | α = 5% |
|---|---|---|---|
| Taylor relative score | -2.971 | -3.256 | -3.485 |
| Taylor minimum score | -2.967 | -3.258 | -3.487 |
| Previous best | -2.959 | -3.256 | -3.472 |
| Equal-weight mean | -2.955 | -3.249 | -3.476 |
| Gate (main) | -2.919 | -3.219 | -3.466 |
| Best single model (HAR) | -2.957 | -3.250 | -3.476 |

At α = 2.5% the gate ablations score: QLIKE-trained variance gate -3.262 (lowest of all methods), no scale head -3.252, variance combination -3.244, no expiry features -3.240, no VIX features -3.224.

What this shows so far:

- The main gate does worse than equal weights and than Taylor's (2020) combinations at all three tail levels.
- Most of the loss comes from one fold. The gate validated on 2023, a calm year, learned a scale factor of about 0.82 (VaR about 18% too small) and then under-predicted risk in 2024: VaR hit rate 3.8% against 2.4% for equal weights. Without the scale head the gate stays close to equal weights in every fold.
- Taylor's minimum and relative score combinations are the strongest baselines.
- These are point estimates. Plan 3 adds Diebold–Mariano tests, the Model Confidence Set and backtests; differences of this size may not be significant.

## Repository layout

```
configs/               run configuration and NSE expiry rules
data/raw/              frozen Yahoo Finance downloads + MANIFEST.csv
data/audit/            outlier review decisions
src/volgate/risk/      FZ0 loss, Student-t and empirical VaR/ES
src/volgate/data/      download, panel construction, expiry calendar
src/volgate/models/    EWMA, GARCH-t, EGARCH-t, GJR-t, HAR, India VIX model, walk-forward engine
scripts/               numbered pipeline stages
tests/                 pytest suite, including look-ahead (leakage) tests
docs/superpowers/      design spec and implementation plans
```

## Design document

The full design is in [docs/superpowers/specs/2026-10-05-fz-gate-design.md](docs/superpowers/specs/2026-10-05-fz-gate-design.md).

## Predecessor

This project extends [volatality-modelling](https://github.com/RushiMMehta/volatality-modelling). That project used a QLIKE-trained gate over variance forecasts with parameters frozen after the in-sample period.

## License

To be decided by the authors.
