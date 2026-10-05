# Design: State-Dependent Neural Gate for VaR/ES Forecast Combination Trained on FZ0 Loss

Date: 2026-10-05
Status: Draft for review
Target venue: IEEE CIFEr (conference paper, 6–8 pages)
Predecessor: https://github.com/RushiMMehta/volatality-modelling

## 1. Goal and claim

Prior work on combining Value-at-Risk (VaR) and Expected Shortfall (ES) forecasts estimates combination weights that are static or depend only on recent performance (Taylor 2020, IJF; Storti & Wang 2023, J. Forecasting; Taylor & Wang 2025, arXiv 2508.16919). This project learns combination weights that depend on the market state. A small neural gate maps observable state variables (trailing model losses, realized volatility, India VIX, option-expiry calendar) to convex weights and a scale correction, and is trained directly on the strictly consistent FZ0 joint VaR/ES loss (Fissler & Ziegel 2016; Patton, Ziegel & Chen 2019).

Paper claim (one sentence): a state-dependent, FZ0-trained gate over econometric VaR/ES forecasts improves joint VaR/ES accuracy over static and performance-based combinations for Indian equities, and we characterize when it does and does not.

Success criteria:
- Every number in the paper is produced by code in this repo from frozen raw data.
- The primary comparison (gate vs. equal weights and vs. Taylor 2020 combinations) is tested with Diebold–Mariano and Model Confidence Set procedures.
- VaR/ES backtests (Kupiec, Christoffersen, DQ, McNeil–Frey) are reported for every method.
- Results are reported honestly, including assets and periods where the gate does not win.

Non-goals (conference scope): LSTM direct forecaster, SHAP analysis, transfer to unseen assets, intraday realized measures. The code should allow adding these later for a journal extension.

## 2. Data

- Assets: NIFTY 50 (`^NSEI`), NIFTY Bank (`^NSEBANK`), Adani Enterprises (`ADANIENT.NS`), Tata Steel (`TATASTEEL.NS`), DLF (`DLF.NS`), Hindustan Unilever (`HINDUNILVR.NS`), Nestle India (`NESTLEIND.NS`), Sun Pharma (`SUNPHARMA.NS`), plus India VIX (`^INDIAVIX`).
- Range: 2015-01-01 to 2026-09-30 (fixed cutoff). Source: Yahoo Finance via `yfinance`, downloaded by `scripts/01_download.py`.
- Raw downloads are stored once under `data/raw/` with a SHA-256 manifest. All later stages read only the frozen raw files.
- Returns: daily log returns from adjusted close.
- Variance proxy for secondary evaluation: Garman–Klass range estimator from OHLC.
- Outlier audit: every |r_t| > 10% is listed and checked against Yahoo split/dividend actions and news. Each is labelled `real` or `corporate_action`. Corporate-action returns are corrected (set from adjusted series or removed with a documented rule). The audit table is saved to `results/tables/outlier_audit.csv` and reported in the paper appendix. Known case: Adani Enterprises 2015-06-03 (−49%).
- Calendar alignment: assets are aligned to NSE trading days; India VIX is forward-filled by at most one day and the fill count is reported.

### 2.1 Expiry calendar (rule-based)

Built in `src/volgate/data/expiry.py` from documented NSE/SEBI rules, with holiday adjustment to the previous trading day:
- NIFTY weekly: Thursday until 2025-08-31; Tuesday from 2025-09-01.
- NIFTY Bank weekly: Thursday (Wednesday from Sep 2023) until weekly contracts were discontinued on 2024-11-20; monthly thereafter.
- Single-stock and monthly index: last Thursday of the month until 2025-08-31; last Tuesday from 2025-09-01.

Exact historical rule changes are verified against NSE circulars before use; each rule has a source citation in code comments.

Features: `own_is_expiry`, `own_days_to_expiry`, `nifty_is_expiry`, `nifty_days_to_expiry`.

## 3. Base models

Each base model i produces, for each asset and day t, using information up to t−1: a variance forecast σ²_{i,t} and a VaR/ES pair (VaR_{i,t}, ES_{i,t}) at tail level α, with ES ≤ VaR < 0.

1. EWMA, λ = 0.94. Tails by filtered historical simulation on standardized residuals from the estimation window.
2. GARCH(1,1), Student-t innovations.
3. EGARCH(1,1), Student-t innovations.
4. GJR-GARCH(1,1), Student-t innovations.
5. HAR on Garman–Klass variance (daily, weekly, monthly lags), estimated by OLS on log variance. Tails from empirical quantiles of standardized returns in the estimation window.
6. Implied-volatility model: σ²_t = β · (VIX_{t−1}/100)² / 252, with β estimated by regression on the estimation window. For stocks, β absorbs the stock's relative volatility. Tails as in model 5.

Estimation: expanding window starting 2015-01-01, re-estimated every 21 trading days. Parameters estimated at refit date d are used for forecasts on days after d only. Base-model out-of-sample forecasts start 2020-01-01. Fit failures fall back to the previous parameter set, and every fallback is logged and counted in the results.

Implementation: `arch` package for GARCH-family models, `statsmodels`/NumPy for HAR and VIX regression.

## 4. Combination methods

### 4.1 Proposed: state-dependent FZ gate (method B)

Gate input x_t (all measurable at t−1):
- Trailing 60-day average FZ0 loss of each of the 6 base models (6 features).
- Garman–Klass realized volatility over 5, 22, 66 days (3).
- Volatility-of-volatility: standard deviation of 5-day realized volatility over the last 30 days (1).
- India VIX level, 5-day change in VIX, variance risk premium VIX²/252 − RV_22 (3).
- Expiry features from section 2.1 (4).
- Lagged return sign and lagged absolute return (2).
- Asset embedding, dimension 4 (learned).

Features are standardized with statistics from the training window only.

Network: MLP with one hidden layer of 16 units, ReLU, dropout. Two heads:
- Softmax weights w_t ∈ Δ^5 over the 6 base models.
- Scale b_t = 0.5 · tanh(·), so exp(b_t) ∈ [0.61, 1.65].

Forecast: (VaR_t, ES_t) = exp(b_t) · Σ_i w_{i,t} (VaR_{i,t}, ES_{i,t}). Because each base pair satisfies ES ≤ VaR < 0 and the combination is convex with a positive scale, the combined pair also satisfies it, as FZ0 requires.

Loss: mean FZ0 loss over training rows, plus λ_eq · ||w_t − 1/6||² (shrinkage toward equal weights), plus weight decay. λ_eq is selected on the validation year from a small grid including 0.

Training: one pooled gate across all 8 assets. Adam, early stopping on validation FZ0. Final forecast is the average of 5 seeds.

### 4.2 Baselines

- Each of the 6 base models alone.
- Equal-weight mean and median of the 6 pairs.
- Taylor (2020) combinations, re-implemented from the paper text: minimum score combining (separate convex weights for VaR and for the ES−VaR spacing, fitted by minimising mean FZ0) and relative score combining (w_i ∝ exp(−λ·S_i), λ fitted). Plus previous-best model selection. Weights are re-estimated every 21 trading days on an expanding window of all earlier out-of-sample forecasts, starting once 250 rows exist. (Revised 2026-10-06 during Plan 2; the draft said a rolling 250-day window.)

### 4.3 Ablations

- A: variance combination. Gate weights combine σ²_{i,t}; VaR/ES come from a single Student-t with ν estimated on the training window; no scale head.
- Gate without VIX features.
- Gate without expiry features.
- Gate without the scale head.
- QLIKE-trained variance gate (the predecessor repo's approach), evaluated on the same backtests.

### 4.4 Tail levels

Primary α = 2.5% (Basel FRTB ES level). Robustness at α = 1% and 5%. One gate is trained per α.

## 5. Walk-forward protocol

The gate is retrained once per test year on an expanding window:

| Fold | Train | Validation | Test |
|---|---|---|---|
| 1 | 2020–2021 | 2022 | 2023 |
| 2 | 2020–2022 | 2023 | 2024 |
| 3 | 2020–2023 | 2024 | 2025 |
| 4 | 2020–2024 | 2025 | 2026-01-01 to 2026-09-30 |

The combined test period is 2023-01-01 to 2026-09-30. The gate's effective training start is 2020-04-08: base forecasts begin 2020-01-01 and the 60-day trailing-loss features need a warm-up. All rolling baselines use only data before each forecast date. Hyperparameters (hidden size, dropout, λ_eq grid) are fixed before fold 1 is tested and are identical across folds.

## 6. Evaluation

Primary:
- Mean FZ0 loss per method, per asset and pooled.
- Diebold–Mariano tests (Newey–West HAC, Harvey–Leybourne–Newbold correction) of the gate against every baseline.
- Model Confidence Set at 10% (`arch.bootstrap.MCS`) over all methods.

Backtests (per method, per asset):
- VaR: Kupiec unconditional coverage, Christoffersen conditional coverage, Engle–Manganelli dynamic quantile (DQ) test.
- ES: McNeil–Frey exceedance residual test (bootstrap).

Secondary:
- QLIKE and MSE of combined variance against the Garman–Klass proxy, for methods that produce a variance.

Analyses:
- Expiry: compare gate weights and loss differentials on expiry vs. non-expiry days, before vs. after 2025-09-01 (difference-in-differences style regression with HAC errors).
- When the gate helps: regress the loss differential (gate − equal weight) on state variables (vol-of-vol, VIX level, VRP, recent loss dispersion across models) with HAC errors.

## 7. Repository layout

```
pyproject.toml            # uv-managed, Python 3.12
configs/default.yaml      # assets, dates, alpha levels, hyperparameters
src/volgate/
  data/                   # download, audit, alignment, expiry calendar
  features/               # gate state features
  models/                 # base models: ewma, garch_t, egarch_t, gjr_t, har, vix
  risk/                   # VaR/ES from distributions, FHS, FZ0 loss
  combine/                # equal, median, Taylor (2020) methods
  gate/                   # network, training, walk-forward driver
  evaluate/               # DM, MCS, backtests, analyses, tables
scripts/01_download.py ... 07_report.py
tests/                    # pytest suite
results/                  # generated tables (CSV + LaTeX) and figures
data/raw/                 # frozen raw downloads + SHA-256 manifest
Makefile                  # `make all` runs the pipeline end to end
.github/workflows/tests.yml
```

## 8. Testing and integrity

- Unit tests for FZ0 loss (known values, ordering constraint, consistency check by simulation), VaR/ES formulas for Student-t and FHS, and backtest statistics against known reference values.
- Leakage tests at every stage: perturb returns at t+k (k ≥ 0 for the target row, k ≥ 1 for features) and assert that every feature, base forecast, combination weight and combined forecast dated ≤ t is unchanged. These tests perturb real inputs, not unused columns.
- Determinism test: two runs with the same seed give identical gate forecasts.
- No report contains hardcoded status strings. Every reported value is computed.
- CI runs the test suite on every push.

## 9. Risks

- Few tail events: at α = 2.5% each asset-year has about 6 exceedances. Mitigations: pooled gate, shrinkage toward equal weights, seed averaging, reporting α = 5%.
- Yahoo data quality: mitigated by the outlier audit and the frozen raw-data manifest; reported as a limitation.
- Gate may not beat equal weights. The "when it helps" analysis and honest reporting keep the paper publishable in that case.
- Expiry rule history errors: each rule is cited to an NSE/SEBI circular.

## 10. Open items to verify during implementation

- Exact definitions of Taylor (2020) combination methods.
- Exact dates of NSE expiry-day changes (Bank Nifty Wednesday switch, weekly discontinuation, Tuesday switch).
- Whether `arch.bootstrap.MCS` supports the loss matrix shape needed; otherwise implement MCS directly.
