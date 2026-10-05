# State-Dependent Neural Gating of VaR/ES Forecasts Trained on FZ Loss

Research code for a study of forecast combination for Value-at-Risk (VaR) and Expected Shortfall (ES) on Indian equities. A small neural gate maps the observable market state to combination weights over six econometric risk models. It is trained directly on the FZ0 joint VaR/ES loss.

> **Status:** All three implementation plans are complete: data and base models, combination baselines and the gate, and the statistical evaluation. The main hypothesis is **not** supported on this data; see Results.

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
| DM, MCS, VaR/ES backtests | Done (Plan 3) |
| Expiry and state analyses | Done (Plan 3) |

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

## Results

Test period 2023-01 to 2026-09, 8 assets, about 920 days each. All numbers come from files in `results/tables/`; LaTeX versions are in `results/tables/tex/`.

### Pooled FZ0 loss (lower is better)

From `results/tables/main_results.csv`. DM compares the main gate with each method on the cross-asset average loss; a positive statistic means the gate has higher loss.

| Method | α = 1% | α = 2.5% | α = 5% | MCS p (2.5%) | DM gate vs method (2.5%) | DM p |
|---|---|---|---|---|---|---|
| Gate, QLIKE-trained variance (ablation) | | -3.2618 | | 1.000 | 2.27 | 0.023 |
| Taylor minimum score | -2.9668 | -3.2578 | -3.4868 | 0.951 | 1.42 | 0.157 |
| Taylor relative score | -2.9707 | -3.2560 | -3.4844 | 0.904 | 1.29 | 0.199 |
| Previous best | -2.9592 | -3.2560 | -3.4720 | 0.951 | 1.18 | 0.237 |
| Gate without scale head (ablation) | | -3.2517 | | 0.821 | 1.30 | 0.192 |
| HAR | -2.9564 | -3.2502 | -3.4762 | 0.766 | 1.40 | 0.162 |
| Equal-weight mean | -2.9550 | -3.2488 | -3.4761 | 0.766 | 1.10 | 0.273 |
| Gate (main) | -2.9187 | -3.2189 | -3.4656 | 0.343 | | |

### What the evidence says

- **Main hypothesis not supported.** The FZ-trained state-dependent gate has higher mean FZ0 loss than equal weights and Taylor's (2020) combinations at all three tail levels.
- **But no method is clearly better than another.** At α = 2.5% the gate stays in the 10% Model Confidence Set pooled and for every one of the 8 assets. Its DM statistics against equal weights and against Taylor's methods are below 2 in absolute value pooled and for every asset. The only significant pooled difference is the QLIKE-trained variance gate beating the main gate (DM 2.27, p = 0.023).
- **Why the main gate loses:** its scale head. In the fold tested on 2024 it learned g ≈ 0.82 from 2020–22 data, so VaR was too small (hit rate 3.8% against 2.4% for equal weights). Without the scale head the gate tracks equal weights.
- **When the gate helps** (`results/tables/state_regression.csv`, HAC errors). The loss difference (gate minus equal weights) is lower, meaning the gate does better, when volatility-of-volatility is high (coefficient -0.073 per standard deviation, p = 0.0005) and when the variance risk premium is high (-0.077, p = 0.0002). Model-loss dispersion is borderline (-0.039, p = 0.052).
- **Expiry effects: none detected.** Difference-in-differences around the 2025-09-01 move to Tuesday expiry finds no significant expiry, post-switch or interaction effect on the gate's relative loss, for NIFTY or for the average stock (`results/tables/expiry_did.csv`; all p > 0.15).
- **Backtests at α = 2.5%** (`results/tables/backtest_rejections.csv`, counts of assets rejecting at 5%): main gate hit rate 2.7%; Kupiec rejects for 2 assets, conditional coverage for 1, DQ for 3, McNeil–Frey for none. Equal weights: hit rate 2.0%, rejections 1/1/0/0. The India VIX model alone is the worst calibrated (4/2/3/2).
- Figure: `results/figures/gate_weights.png` shows the gate's monthly mean VaR weights and scale factor.

### Caveats

- The test period is one market (8 NSE assets), about 3.75 years, with roughly 23 VaR exceptions per asset at α = 2.5%. Power to separate methods is low.
- The gate's design and hyperparameters were fixed before the test years were seen. Changing it now (for example removing or regularising the scale head) would be tuned on the test period and needs a fresh hold-out or a pre-registered protocol.
- McNeil–Frey uses residuals scaled by |ES| because pair combinations have no σ.

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
