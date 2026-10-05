# State-Dependent Neural Gating of VaR/ES Forecasts Trained on FZ Loss

Research code for a study of forecast combination for Value-at-Risk (VaR) and Expected Shortfall (ES) on Indian equities. A small neural gate maps the observable market state to combination weights over six econometric risk models. It is trained directly on the FZ0 joint VaR/ES loss.

> **Status:** design stage. No results yet. This README describes what the code will do; the status table below is updated as each part is implemented and tested.

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
| Data download, audit, expiry calendar | Planned |
| Base models and VaR/ES | Planned |
| FZ0 loss | Planned |
| Baseline combinations (equal, median, Taylor 2020) | Planned |
| Neural gate and walk-forward training | Planned |
| DM, MCS, VaR/ES backtests | Planned |
| Expiry and state analyses | Planned |

## Reproducing

Planned commands, not yet available:

```bash
uv sync
make all        # download -> audit -> base models -> combinations -> gate -> evaluation -> tables
uv run pytest   # unit, leakage, and determinism tests
```

## Design document

The full design is in [docs/superpowers/specs/2026-10-05-fz-gate-design.md](docs/superpowers/specs/2026-10-05-fz-gate-design.md).

## Predecessor

This project extends [volatality-modelling](https://github.com/RushiMMehta/volatality-modelling). That project used a QLIKE-trained gate over variance forecasts with parameters frozen after the in-sample period.

## License

To be decided by the authors.
