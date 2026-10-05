# Plan 2: Combination Baselines and the FZ-Trained Gate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce out-of-sample VaR/ES forecasts for 2023-01-01 to 2026-09-30 from the combination baselines (mean, median, Taylor 2020 minimum-score and relative-score combining, previous best) and from the state-dependent gate and its ablations, at α ∈ {1%, 2.5%, 5%}.

**Architecture:** `volgate.combine` holds data loading and the rolling baselines. `volgate.gate` holds the gate features, a NumPy MLP with hand-written gradients (checked against finite differences), Adam training with early stopping, and the yearly walk-forward driver. Scripts 04–06 run baselines, the gate, and a first FZ0 comparison table.

**Tech Stack:** Python 3.12, NumPy, pandas, SciPy. No deep-learning framework (see Ruling below).

**Spec:** `docs/superpowers/specs/2026-10-05-fz-gate-design.md` (sections 4, 5). Builds on Plan 1 outputs in `data/processed/panel/` and `data/processed/base_forecasts/`.

**Rulings carried from planning:**
- Gate in NumPy, not PyTorch: PyTorch ≥2.3 has no wheels for macOS x86_64, and 2.2.2 is built against NumPy 1.x while the project uses NumPy 2.5. Gradients are verified by finite-difference tests.
- Taylor (2020) definitions read from the paper (Oxford ORA preprint, sections 3.1–3.2): minimum score combining uses separate convex weights for VaR and for the ES−VaR spacing; relative score combining uses w_i ∝ exp(−λ·S_i) with λ estimated. "Previous best" is not in Taylor (2020); it stays as an extra simple baseline.
- The gate follows Taylor's spacing structure: VaR = g·Σ w^Q_i VaR_i and ES = g·(Σ w^Q_i VaR_i + Σ w^S_i (ES_i − VaR_i)), g = exp(0.5·tanh(·)). It is the state-dependent version of minimum score combining. Heads start at zero, so an untrained gate equals the equal-weight combination.

## Global Constraints

- Base models, in this order everywhere: `ewma, garch_t, egarch_t, gjr_t, har, vix`.
- Tail levels: α ∈ {0.01, 0.025, 0.05}; tags `a010, a025, a050`. All gate variants run at α = 0.025; only the main gate runs at 0.01 and 0.05.
- Baseline weights re-estimated every 21 rows on all rows before the block, after a minimum of 250 training rows.
- Gate folds (train starts at the first usable row):

  | Fold | Train end | Validation | Test |
  |---|---|---|---|
  | 1 | 2021-12-31 | 2022 | 2023 |
  | 2 | 2022-12-31 | 2023 | 2024 |
  | 3 | 2023-12-31 | 2024 | 2025 |
  | 4 | 2024-12-31 | 2025 | 2026-01-01 to 2026-09-30 |

- Gate hyperparameters fixed before any test year is seen: hidden 16, embedding 4, dropout 0.1, weight decay 1e-4, Adam lr 1e-3, batch 256, max 300 epochs, patience 20, λ_eq grid {0, 0.1, 1, 10} chosen on validation with seed 0, final forecast = mean of seeds 0–4.
- ES ≤ VaR < 0 for every combined forecast.
- Any feature for day t uses only data before t, except the expiry calendar (known in advance).

## Review Focus

1. A training batch where every row is a VaR hit or none is → gradients stay finite; covered by the gradient check with mixed hits (Task 4).
2. A base model with NaN forecasts on some rows (VIX gaps) → rows dropped from baselines' training and from the pooled gate data, never propagated as NaN weights (Tasks 2, 5).
3. Validation loss never improves after epoch 0 → the epoch-0 parameters are restored, not the last ones (Task 4 test).
4. Two assets with different date sets → pooled frame keeps each asset's own dates; predictions are per asset and date (Task 5 test).
5. A feature constant in the training window (zero std) → standardization uses std 1, no division by zero (Task 5 test).

---

### Task 1: Combination data access

**Files:** Create `src/volgate/combine/__init__.py` (empty), `src/volgate/combine/data.py`; extend `tests/conftest.py`; test `tests/test_combine_data.py`.

**Interfaces:**
- Produces: `MODELS: list[str]`; `load_asset(processed: Path, asset: str) -> pd.DataFrame` (panel inner-joined with base forecasts); `pairs(df, alpha) -> (V, E)` arrays (n, 6); `variances(df) -> S2` (n, 6); `model_losses(df, alpha) -> pd.DataFrame` (FZ0 per model, NaN where a forecast is NaN).
- Test helper `make_combo_frame(n=700, seed=0) -> pd.DataFrame` in `tests/conftest.py`: `make_panel` plus base-forecast columns for all six models (model i variance = true variance × factor_i, normal VaR/ES) and zero expiry columns.

- [ ] **Step 1: Add helper to `tests/conftest.py`**

```python
from scipy import stats

from volgate.combine.data import MODELS
from volgate.risk.dist import alpha_tag

_FACTORS = [1.0, 0.6, 1.5, 0.9, 1.2, 2.0]


def make_combo_frame(n: int = 700, seed: int = 0) -> pd.DataFrame:
    df = make_panel(n, seed)
    true_s2 = df["log_return"].rolling(20, min_periods=1).var().bfill().to_numpy() + 1e-6
    for m, f in zip(MODELS, _FACTORS):
        s2 = true_s2 * f
        df[f"{m}__sigma2"] = s2
        df[f"{m}__fallback"] = False
        for a in (0.01, 0.025, 0.05):
            q = stats.norm.ppf(a)
            df[f"{m}__var_{alpha_tag(a)}"] = np.sqrt(s2) * q
            df[f"{m}__es_{alpha_tag(a)}"] = -np.sqrt(s2) * stats.norm.pdf(q) / a
    for c in ["own_is_expiry", "own_days_to_expiry", "nifty_is_expiry", "nifty_days_to_expiry"]:
        df[c] = 0.0
    return df


@pytest.fixture
def combo_frame() -> pd.DataFrame:
    return make_combo_frame()
```

- [ ] **Step 2: Write failing test `tests/test_combine_data.py`**

```python
import numpy as np
import pandas as pd

from volgate.combine.data import MODELS, load_asset, model_losses, pairs, variances
from volgate.risk.fz import fz0_loss


def test_pairs_and_variances_shapes(combo_frame):
    V, E = pairs(combo_frame, 0.025)
    assert V.shape == E.shape == (len(combo_frame), 6)
    assert np.all(E <= V)
    assert variances(combo_frame).shape == (len(combo_frame), 6)


def test_model_losses_match_fz0_and_propagate_nan(combo_frame):
    df = combo_frame.copy()
    df.iloc[5, df.columns.get_loc("vix__var_a025")] = np.nan
    L = model_losses(df, 0.025)
    assert list(L.columns) == MODELS
    y = df["log_return"].iloc[3]
    expected = fz0_loss(y, df["ewma__var_a025"].iloc[3], df["ewma__es_a025"].iloc[3], 0.025)
    assert L["ewma"].iloc[3] == expected
    assert np.isnan(L["vix"].iloc[5]) and np.isfinite(L["ewma"].iloc[5])


def test_load_asset_inner_joins(tmp_path, combo_frame):
    (tmp_path / "panel").mkdir()
    (tmp_path / "base_forecasts").mkdir()
    panel_cols = [c for c in combo_frame.columns if "__" not in c]
    base_cols = [c for c in combo_frame.columns if "__" in c]
    combo_frame[panel_cols].to_csv(tmp_path / "panel" / "x.csv")
    combo_frame[base_cols].iloc[100:].to_csv(tmp_path / "base_forecasts" / "x.csv")
    df = load_asset(tmp_path, "x")
    assert len(df) == len(combo_frame) - 100
    assert isinstance(df.index, pd.DatetimeIndex)
```

- [ ] **Step 3: Run** `uv run pytest tests/test_combine_data.py -q` — Expected: collection error, `No module named 'volgate.combine'`.

- [ ] **Step 4: Implement `src/volgate/combine/data.py`**

```python
from pathlib import Path

import numpy as np
import pandas as pd

from volgate.risk.dist import alpha_tag
from volgate.risk.fz import fz0_loss

MODELS = ["ewma", "garch_t", "egarch_t", "gjr_t", "har", "vix"]


def load_asset(processed: Path, asset: str) -> pd.DataFrame:
    processed = Path(processed)
    panel = pd.read_csv(processed / "panel" / f"{asset}.csv", index_col="date", parse_dates=True)
    base = pd.read_csv(processed / "base_forecasts" / f"{asset}.csv", index_col="date",
                       parse_dates=True)
    return panel.join(base, how="inner")


def pairs(df: pd.DataFrame, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    t = alpha_tag(alpha)
    V = df[[f"{m}__var_{t}" for m in MODELS]].to_numpy(dtype=float)
    E = df[[f"{m}__es_{t}" for m in MODELS]].to_numpy(dtype=float)
    return V, E


def variances(df: pd.DataFrame) -> np.ndarray:
    return df[[f"{m}__sigma2" for m in MODELS]].to_numpy(dtype=float)


def model_losses(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    V, E = pairs(df, alpha)
    y = np.broadcast_to(df["log_return"].to_numpy(dtype=float)[:, None], V.shape)
    ok = np.isfinite(V) & np.isfinite(E) & np.isfinite(y)
    L = np.full(V.shape, np.nan)
    L[ok] = fz0_loss(y[ok], V[ok], E[ok], alpha)
    return pd.DataFrame(L, index=df.index, columns=MODELS)
```

- [ ] **Step 5: Run** `uv run pytest tests/test_combine_data.py -q` — Expected: 3 passed. Then `uv run pytest -q` — all pass.

- [ ] **Step 6: Commit** `git add src/volgate/combine tests/conftest.py tests/test_combine_data.py && git commit -m "feat: add combination data access"`

---

### Task 2: Rolling combination baselines

**Files:** Create `src/volgate/combine/baselines.py`; test `tests/test_baselines.py`.

**Interfaces:**
- Consumes: `fz0_loss`.
- Produces:
  - `combine_pairs(V, E, wq, ws) -> (var, es)`; weights shape (M,) or (n, M)
  - `mean_combo(V, E)`, `median_combo(V, E) -> (var, es)`
  - `fit_min_score(y, V, E, alpha, init=None, n_random=3, seed=0) -> (wq, ws, theta)`
  - `fit_relative_score(y, V, E, alpha) -> (w, lam)`
  - `fit_previous_best(y, V, E, alpha) -> w`
  - `rolling_combination(y, V, E, alpha, method, refit_every=21, min_train=250) -> (var, es, WQ)`; `method` in `{"min_score", "relative_score", "previous_best"}`; rows before `min_train` are NaN.

- [ ] **Step 1: Write failing test `tests/test_baselines.py`**

```python
import numpy as np
import pytest
from scipy import stats

from volgate.combine.baselines import (combine_pairs, fit_min_score, fit_previous_best,
                                       fit_relative_score, mean_combo, median_combo,
                                       rolling_combination)

A = 0.025


def _toy(n=3000, seed=0):
    """Model 0 is exact for N(0, 1); models 1, 2 are mis-scaled."""
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n)
    q = stats.norm.ppf(A)
    v0, e0 = q, -stats.norm.pdf(q) / A
    scales = np.array([1.0, 0.5, 2.0])
    V = np.tile(v0 * scales, (n, 1))
    E = np.tile(e0 * scales, (n, 1))
    return y, V, E


def test_combine_pairs_mean_equals_mean_es():
    _, V, E = _toy(10)
    v, e = mean_combo(V, E)
    np.testing.assert_allclose(e, E.mean(axis=1))
    np.testing.assert_allclose(v, V.mean(axis=1))
    v2, e2 = combine_pairs(V, E, np.array([0.2, 0.3, 0.5]), np.array([1.0, 0.0, 0.0]))
    assert np.all(e2 <= v2)


def test_median_keeps_ordering():
    _, V, E = _toy(10)
    v, e = median_combo(V, E)
    assert np.all(e <= v)


def test_min_score_finds_correct_model():
    y, V, E = _toy()
    wq, ws, _ = fit_min_score(y, V, E, A)
    assert wq.sum() == pytest.approx(1) and ws.sum() == pytest.approx(1)
    v, e = combine_pairs(V, E, wq, ws)
    assert v[0] == pytest.approx(V[0, 0], rel=0.1)
    assert e[0] == pytest.approx(E[0, 0], rel=0.1)


def test_relative_score_favours_best_and_previous_best_is_one_hot():
    y, V, E = _toy()
    w, lam = fit_relative_score(y, V, E, A)
    assert np.argmax(w) == 0 and lam > 0
    assert fit_previous_best(y, V, E, A).tolist() == [1.0, 0.0, 0.0]


def test_rolling_combination_is_causal_and_skips_nan_rows():
    y, V, E = _toy(600)
    V = V.copy()
    V[300, 1] = np.nan
    base = rolling_combination(y, V, E, A, "relative_score", refit_every=50, min_train=250)
    assert np.all(np.isnan(base[0][:250])) and np.all(np.isfinite(base[0][250:]) | (np.arange(600)[250:] == 300))
    y2 = y.copy()
    y2[400] = -50.0
    pert = rolling_combination(y2, V, E, A, "relative_score", refit_every=50, min_train=250)
    np.testing.assert_array_equal(base[0][:401], pert[0][:401])
    np.testing.assert_array_equal(base[1][:401], pert[1][:401])


def test_unknown_method_raises():
    y, V, E = _toy(300)
    with pytest.raises(ValueError, match="method"):
        rolling_combination(y, V, E, A, "bogus")
```

- [ ] **Step 2: Run** `uv run pytest tests/test_baselines.py -q` — Expected: collection error, module missing.

- [ ] **Step 3: Implement `src/volgate/combine/baselines.py`**

```python
import numpy as np
from scipy.optimize import minimize

from volgate.risk.fz import fz0_loss


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def combine_pairs(V, E, wq, ws) -> tuple[np.ndarray, np.ndarray]:
    """Taylor (2020) form: VaR = sum wq*VaR_i; ES = VaR + sum ws*(ES_i - VaR_i)."""
    var = np.sum(V * wq, axis=-1)
    return var, var + np.sum((E - V) * ws, axis=-1)


def mean_combo(V, E):
    M = V.shape[1]
    w = np.full(M, 1.0 / M)
    return combine_pairs(V, E, w, w)


def median_combo(V, E):
    return np.median(V, axis=1), np.median(E, axis=1)


def _score(y, V, E, alpha):
    return np.array([fz0_loss(y, V[:, i], E[:, i], alpha).mean() for i in range(V.shape[1])])


def fit_min_score(y, V, E, alpha, init=None, n_random=3, seed=0):
    M = V.shape[1]

    def obj(theta):
        v, e = combine_pairs(V, E, _softmax(theta[:M]), _softmax(theta[M:]))
        return fz0_loss(y, v, e, alpha).mean()

    rng = np.random.default_rng(seed)
    starts = [np.zeros(2 * M)] + ([init] if init is not None else [])
    starts += [rng.normal(0, 1, 2 * M) for _ in range(n_random)]
    best = min((minimize(obj, s, method="Powell") for s in starts), key=lambda r: r.fun)
    return _softmax(best.x[:M]), _softmax(best.x[M:]), best.x


def fit_relative_score(y, V, E, alpha):
    S = _score(y, V, E, alpha)
    grid = np.concatenate([[0.0], np.logspace(-2, 4, 400)])
    best = (np.inf, 0.0, None)
    for lam in grid:
        w = _softmax(-lam * S)
        v, e = combine_pairs(V, E, w, w)
        s = fz0_loss(y, v, e, alpha).mean()
        if s < best[0]:
            best = (s, lam, w)
    return best[2], best[1]


def fit_previous_best(y, V, E, alpha):
    w = np.zeros(V.shape[1])
    w[np.argmin(_score(y, V, E, alpha))] = 1.0
    return w


def rolling_combination(y, V, E, alpha, method, refit_every=21, min_train=250):
    if method not in ("min_score", "relative_score", "previous_best"):
        raise ValueError(f"unknown method {method!r}")
    n, M = V.shape
    var, es = np.full(n, np.nan), np.full(n, np.nan)
    WQ = np.full((n, M), np.nan)
    ok = np.isfinite(y) & np.isfinite(V).all(axis=1) & np.isfinite(E).all(axis=1)
    theta = None
    for k in range(min_train, n, refit_every):
        tr = ok[:k]
        yt, Vt, Et = y[:k][tr], V[:k][tr], E[:k][tr]
        if method == "min_score":
            wq, ws, theta = fit_min_score(yt, Vt, Et, alpha, init=theta,
                                          n_random=3 if theta is None else 0)
        elif method == "relative_score":
            wq, _ = fit_relative_score(yt, Vt, Et, alpha)
            ws = wq
        else:
            wq = ws = fit_previous_best(yt, Vt, Et, alpha)
        end = min(k + refit_every, n)
        var[k:end], es[k:end] = combine_pairs(V[k:end], E[k:end], wq, ws)
        WQ[k:end] = wq
    return var, es, WQ
```

- [ ] **Step 4: Run** `uv run pytest tests/test_baselines.py -q` — Expected: 6 passed. `uv run pytest -q` — all pass.

- [ ] **Step 5: Commit** `git commit -m "feat: add mean, median, Taylor (2020) and previous-best combinations"`

---

### Task 3: Gate features

**Files:** Create `src/volgate/gate/__init__.py` (empty), `src/volgate/gate/features.py`; test `tests/test_gate_features.py`.

**Interfaces:**
- Consumes: `MODELS`, `model_losses`.
- Produces: `gate_features(df, alpha, window=60) -> pd.DataFrame`; constants `LOSS_FEATURES`, `RV_FEATURES`, `VIX_FEATURES`, `EXPIRY_FEATURES`, `RETURN_FEATURES`; `feature_columns(use_vix: bool, use_expiry: bool) -> list[str]`.

- [ ] **Step 1: Write failing test**

```python
import numpy as np
import pandas as pd

from volgate.gate.features import feature_columns, gate_features


def test_columns_and_selection(combo_frame):
    f = gate_features(combo_frame, 0.025)
    assert set(feature_columns(True, True)) == set(f.columns)
    assert "vix" not in feature_columns(False, True)
    assert "own_is_expiry" not in feature_columns(True, False)
    assert f.iloc[100:].notna().all().all()


def test_no_lookahead(combo_frame):
    base = gate_features(combo_frame, 0.025)
    j = 400
    pert = combo_frame.copy()
    cols = [c for c in pert.columns if c in ("log_return", "gk_var", "vix") or "__" in c]
    pert.iloc[j, pert.columns.get_indexer(cols)] *= 3
    out = gate_features(pert, 0.025)
    pd.testing.assert_frame_equal(base.iloc[: j + 1], out.iloc[: j + 1])
    assert not base.iloc[j + 1].equals(out.iloc[j + 1])
```

Note: base forecast columns at row j are perturbed too; row j's features must still not change, because features use losses and data only up to j−1.

- [ ] **Step 2: Run** — Expected: module missing.

- [ ] **Step 3: Implement `src/volgate/gate/features.py`**

```python
import numpy as np
import pandas as pd

from volgate.combine.data import MODELS, model_losses

LOSS_FEATURES = [f"fz_{m}" for m in MODELS]
RV_FEATURES = ["rv5", "rv22", "rv66", "vov30"]
VIX_FEATURES = ["vix", "vix_chg5", "vrp"]
EXPIRY_FEATURES = ["own_is_expiry", "own_days_to_expiry", "nifty_is_expiry", "nifty_days_to_expiry"]
RETURN_FEATURES = ["ret_neg_lag1", "absret_lag1"]


def feature_columns(use_vix: bool, use_expiry: bool) -> list[str]:
    cols = LOSS_FEATURES + RV_FEATURES + RETURN_FEATURES
    if use_vix:
        cols = cols + VIX_FEATURES
    if use_expiry:
        cols = cols + EXPIRY_FEATURES
    return cols


def gate_features(df: pd.DataFrame, alpha: float, window: int = 60) -> pd.DataFrame:
    """State features for day t built from data before t (expiry calendar is known ahead)."""
    f = pd.DataFrame(index=df.index)
    L = model_losses(df, alpha)
    for m in MODELS:
        f[f"fz_{m}"] = L[m].shift(1).rolling(window, min_periods=window).mean()
    gk = df["gk_var"].shift(1)
    for w in (5, 22, 66):
        f[f"rv{w}"] = np.sqrt(gk.rolling(w).mean())
    f["vov30"] = f["rv5"].rolling(30).std()
    r1 = df["log_return"].shift(1)
    f["ret_neg_lag1"] = (r1 < 0).astype(float).where(r1.notna())
    f["absret_lag1"] = r1.abs()
    vix = df["vix"].shift(1)
    f["vix"] = vix
    f["vix_chg5"] = vix - vix.shift(5)
    f["vrp"] = (vix / 100.0) ** 2 / 252.0 - gk.rolling(22).mean()
    for c in EXPIRY_FEATURES:
        f[c] = df[c].astype(float)
    return f
```

- [ ] **Step 4: Run** — Expected: 2 passed; full suite passes.

- [ ] **Step 5: Commit** `git commit -m "feat: add causal gate state features"`

---

### Task 4: Gate network, gradients, training

**Files:** Create `src/volgate/gate/net.py`; test `tests/test_gate_net.py`.

**Interfaces:**
- Consumes: `student_t_var_es`.
- Produces:
  - `GateNet(n_features, n_assets, n_models, hidden=16, emb=4, dropout=0.1, mode="pairs", loss="fz0", use_scale=True, alpha=0.025, nu=None, lam_eq=0.0, wd=1e-4, seed=0)`; attributes `p: dict[str, np.ndarray]`
  - `GateNet.forward(b, train=False, rng=None) -> dict` with keys `var, es, wq` (+ `ws, g` in pairs mode)
  - `GateNet.loss_and_grad(b, train=False, rng=None) -> (float, dict)`
  - `GateNet.data_loss(b) -> float` (mean training loss without penalties, no dropout)
  - `train_gate(net, tr, va, lr=1e-3, epochs=300, batch=256, patience=20, seed=0) -> dict` (`best_val, best_epoch, history`); restores best parameters
  - batch dict `b`: `X (n,F) float, a (n,) int, V, E, S2 (n,M) float, y (n,) float`

- [ ] **Step 1: Write failing test**

```python
import numpy as np
import pytest
from scipy import stats

from volgate.gate.net import GateNet, train_gate

A = 0.025


def _batch(n=64, F=5, M=3, n_assets=2, seed=0):
    rng = np.random.default_rng(seed)
    s = np.exp(rng.normal(-4, 0.3, (n, M)))
    q = stats.norm.ppf(A)
    return {"X": rng.normal(size=(n, F)), "a": rng.integers(0, n_assets, n),
            "V": s * q, "E": -s * stats.norm.pdf(q) / A, "S2": s**2,
            "y": rng.normal(0, 0.02, n)}


CONFIGS = [dict(mode="pairs", use_scale=True), dict(mode="pairs", use_scale=False),
           dict(mode="variance", use_scale=False, nu=6.0),
           dict(mode="variance", loss="qlike", use_scale=False, nu=6.0)]


@pytest.mark.parametrize("cfg", CONFIGS)
def test_gradients_match_finite_differences(cfg):
    b = _batch()
    net = GateNet(5, 2, 3, hidden=4, emb=2, dropout=0.0, lam_eq=0.5, wd=1e-3, seed=1, **cfg)
    rng = np.random.default_rng(2)
    for k in net.p:  # move heads off zero so every path is exercised
        net.p[k] = net.p[k] + rng.normal(0, 0.3, net.p[k].shape)
    _, g = net.loss_and_grad(b)
    h = 1e-6
    for k, v in net.p.items():
        for idx in list(np.ndindex(v.shape))[:6]:
            old = v[idx]
            v[idx] = old + h
            lp, _ = net.loss_and_grad(b)
            v[idx] = old - h
            lm, _ = net.loss_and_grad(b)
            v[idx] = old
            assert g[k][idx] == pytest.approx((lp - lm) / (2 * h), rel=1e-4, abs=1e-6), (k, idx)


def test_untrained_gate_equals_equal_weights():
    b = _batch()
    net = GateNet(5, 2, 3, seed=0)
    out = net.forward(b)
    np.testing.assert_allclose(out["var"], b["V"].mean(axis=1))
    np.testing.assert_allclose(out["es"], b["E"].mean(axis=1))


def test_es_below_var_after_random_params():
    b = _batch()
    net = GateNet(5, 2, 3, seed=0)
    rng = np.random.default_rng(3)
    for k in net.p:
        net.p[k] = net.p[k] + rng.normal(0, 2, net.p[k].shape)
    out = net.forward(b)
    assert np.all(out["es"] <= out["var"]) and np.all(out["var"] < 0)


def test_gate_learns_state_dependent_weights():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.integers(0, 2, n).astype(float)
    sd = np.where(x == 1, 0.01, 0.03)
    y = rng.normal(0, sd)
    q, es = stats.norm.ppf(A), -stats.norm.pdf(stats.norm.ppf(A)) / A
    V = np.column_stack([np.full(n, 0.01 * q), np.full(n, 0.03 * q)])
    E = np.column_stack([np.full(n, 0.01 * es), np.full(n, 0.03 * es)])
    b = {"X": x[:, None], "a": np.zeros(n, int), "V": V, "E": E, "S2": (V / q) ** 2, "y": y}
    tr = {k: v[:3000] for k, v in b.items()}
    va = {k: v[3000:] for k, v in b.items()}
    net = GateNet(1, 1, 2, hidden=8, emb=0, dropout=0.0, seed=0)
    eq = net.data_loss(va)
    info = train_gate(net, tr, va, lr=1e-2, epochs=100, patience=20, seed=0)
    assert info["best_val"] < eq - 0.05
    w = net.forward(va)["wq"]
    assert w[va["X"][:, 0] == 1, 0].mean() > 0.8


def test_train_restores_best_epoch(monkeypatch):
    b = _batch(200)
    net = GateNet(5, 2, 3, seed=0)
    start = {k: v.copy() for k, v in net.p.items()}
    calls = {"n": 0}

    def worse_each_time(self, batch):
        calls["n"] += 1
        return float(calls["n"])  # epoch 0 is best

    monkeypatch.setattr(GateNet, "data_loss", worse_each_time)
    info = train_gate(net, b, b, epochs=10, patience=3, seed=0)
    assert info["best_epoch"] == 0
    assert not all(np.array_equal(start[k], net.p[k]) for k in net.p)  # trained one epoch first


def test_training_is_deterministic():
    b = _batch(300)
    outs = []
    for _ in range(2):
        net = GateNet(5, 2, 3, seed=4)
        train_gate(net, b, b, epochs=5, seed=4)
        outs.append(net.forward(b)["var"])
    np.testing.assert_array_equal(outs[0], outs[1])


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        GateNet(5, 2, 3, mode="pairs", loss="qlike")
    with pytest.raises(ValueError):
        GateNet(5, 2, 3, mode="variance", nu=None)
```

- [ ] **Step 2: Run** — Expected: module missing.

- [ ] **Step 3: Implement `src/volgate/gate/net.py`**

```python
import numpy as np

from volgate.risk.dist import student_t_var_es

_DECAYED = ("W1", "Wq", "Ws", "wb", "emb")


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _softmax_back(w, dw):
    return w * (dw - np.sum(dw * w, axis=1, keepdims=True))


class GateNet:
    """One-hidden-layer gate over M base models with hand-written gradients.

    pairs mode: VaR = g*sum(wq*V), ES = g*(sum(wq*V) + sum(ws*(E-V))), g = exp(0.5*tanh(.))
    variance mode: s2 = sum(wq*S2); VaR/ES = Student-t(nu) quantile/ES * sqrt(s2)
    """

    def __init__(self, n_features, n_assets, n_models, hidden=16, emb=4, dropout=0.1,
                 mode="pairs", loss="fz0", use_scale=True, alpha=0.025, nu=None,
                 lam_eq=0.0, wd=1e-4, seed=0):
        if mode not in ("pairs", "variance"):
            raise ValueError(f"bad mode {mode!r}")
        if loss not in ("fz0", "qlike"):
            raise ValueError(f"bad loss {loss!r}")
        if loss == "qlike" and mode != "variance":
            raise ValueError("qlike loss requires variance mode")
        if mode == "variance" and nu is None:
            raise ValueError("variance mode needs nu")
        self.F, self.M = n_features, n_models
        self.mode, self.loss, self.alpha = mode, loss, alpha
        self.use_scale = use_scale and mode == "pairs"
        self.dropout, self.lam_eq, self.wd = dropout, lam_eq, wd
        rng = np.random.default_rng(seed)
        d = n_features + emb
        self.p = {
            "W1": rng.normal(0, np.sqrt(2.0 / d), (d, hidden)), "b1": np.zeros(hidden),
            "emb": rng.normal(0, 0.1, (n_assets, emb)),
            "Wq": np.zeros((hidden, n_models)), "bq": np.zeros(n_models),
        }
        if mode == "pairs":
            self.p["Ws"] = np.zeros((hidden, n_models))
            self.p["bs"] = np.zeros(n_models)
            if self.use_scale:
                self.p["wb"] = np.zeros(hidden)
                self.p["bb"] = np.zeros(1)
        else:
            qv, qe = student_t_var_es(1.0, nu, alpha)
            self.qv, self.qe = float(qv), float(qe)

    def forward(self, b, train=False, rng=None):
        p = self.p
        z = np.concatenate([b["X"], p["emb"][b["a"]]], axis=1)
        a1 = z @ p["W1"] + p["b1"]
        h = np.maximum(a1, 0.0)
        if train and self.dropout > 0:
            mask = (rng.random(h.shape) >= self.dropout) / (1.0 - self.dropout)
        else:
            mask = np.ones_like(h)
        hd = h * mask
        c = {"z": z, "a1": a1, "mask": mask, "hd": hd,
             "wq": _softmax(hd @ p["Wq"] + p["bq"])}
        if self.mode == "pairs":
            c["ws"] = _softmax(hd @ p["Ws"] + p["bs"])
            if self.use_scale:
                c["th"] = np.tanh(hd @ p["wb"] + p["bb"][0])
                c["g"] = np.exp(0.5 * c["th"])
            else:
                c["g"] = np.ones(len(hd))
            c["v0"] = np.sum(b["V"] * c["wq"], axis=1)
            c["sp"] = np.sum((b["E"] - b["V"]) * c["ws"], axis=1)
            c["var"] = c["g"] * c["v0"]
            c["es"] = c["g"] * (c["v0"] + c["sp"])
        else:
            c["s2"] = np.sum(b["S2"] * c["wq"], axis=1)
            c["sd"] = np.sqrt(c["s2"])
            c["var"], c["es"] = self.qv * c["sd"], self.qe * c["sd"]
        return c

    def _data_terms(self, c, y):
        n = len(y)
        if self.loss == "fz0":
            v, e = c["var"], c["es"]
            hit = (y <= v).astype(float)
            loss = np.mean(-hit * (v - y) / (self.alpha * e) + v / e + np.log(-e) - 1.0)
            dv = (-hit / (self.alpha * e) + 1.0 / e) / n
            de = (hit * (v - y) / (self.alpha * e**2) - v / e**2 + 1.0 / e) / n
            return loss, dv, de, None
        y2 = np.maximum(y**2, 1e-12)
        ratio = y2 / c["s2"]
        return np.mean(ratio - np.log(ratio) - 1.0), None, None, (-y2 / c["s2"]**2 + 1.0 / c["s2"]) / n

    def data_loss(self, b):
        return float(self._data_terms(self.forward(b), b["y"])[0])

    def loss_and_grad(self, b, train=False, rng=None):
        p, M = self.p, self.M
        c = self.forward(b, train, rng)
        n = len(b["y"])
        loss, dv, de, ds2 = self._data_terms(c, b["y"])
        g = {k: np.zeros_like(v) for k, v in p.items()}
        eq = 1.0 / M
        pen = np.mean(np.sum((c["wq"] - eq) ** 2, axis=1))
        dwq = 2.0 * self.lam_eq * (c["wq"] - eq) / n
        dhd = np.zeros_like(c["hd"])
        if self.mode == "pairs":
            pen += np.mean(np.sum((c["ws"] - eq) ** 2, axis=1))
            gg = c["g"]
            dwq = dwq + ((dv + de) * gg)[:, None] * b["V"]
            dws = 2.0 * self.lam_eq * (c["ws"] - eq) / n + (de * gg)[:, None] * (b["E"] - b["V"])
            dls = _softmax_back(c["ws"], dws)
            g["Ws"], g["bs"] = c["hd"].T @ dls, dls.sum(axis=0)
            dhd = dhd + dls @ p["Ws"].T
            if self.use_scale:
                dgg = dv * c["v0"] + de * (c["v0"] + c["sp"])
                dpre = dgg * gg * 0.5 * (1.0 - c["th"] ** 2)
                g["wb"], g["bb"] = c["hd"].T @ dpre, np.array([dpre.sum()])
                dhd = dhd + dpre[:, None] * p["wb"][None, :]
        else:
            if self.loss == "fz0":
                ds2 = (dv * self.qv + de * self.qe) / (2.0 * c["sd"])
            dwq = dwq + ds2[:, None] * b["S2"]
        dlq = _softmax_back(c["wq"], dwq)
        g["Wq"], g["bq"] = c["hd"].T @ dlq, dlq.sum(axis=0)
        dhd = dhd + dlq @ p["Wq"].T
        da1 = dhd * c["mask"] * (c["a1"] > 0)
        g["W1"], g["b1"] = c["z"].T @ da1, da1.sum(axis=0)
        np.add.at(g["emb"], b["a"], (da1 @ p["W1"].T)[:, self.F:])
        reg = 0.0
        for k in _DECAYED:
            if k in p:
                reg += 0.5 * self.wd * np.sum(p[k] ** 2)
                g[k] = g[k] + self.wd * p[k]
        return float(loss + self.lam_eq * pen + reg), g


def _take(b, idx):
    return {k: v[idx] for k, v in b.items()}


def train_gate(net, tr, va, lr=1e-3, epochs=300, batch=256, patience=20, seed=0):
    rng = np.random.default_rng(seed)
    m1 = {k: np.zeros_like(v) for k, v in net.p.items()}
    m2 = {k: np.zeros_like(v) for k, v in net.p.items()}
    t, n = 0, len(tr["y"])
    best_val, best_p, best_ep, hist = np.inf, None, -1, []
    for ep in range(epochs):
        order = rng.permutation(n)
        for i in range(0, n, batch):
            _, g = net.loss_and_grad(_take(tr, order[i:i + batch]), train=True, rng=rng)
            t += 1
            for k in net.p:
                m1[k] = 0.9 * m1[k] + 0.1 * g[k]
                m2[k] = 0.999 * m2[k] + 0.001 * g[k] ** 2
                net.p[k] = net.p[k] - lr * (m1[k] / (1 - 0.9**t)) / (np.sqrt(m2[k] / (1 - 0.999**t)) + 1e-8)
        val = net.data_loss(va)
        hist.append(val)
        if val < best_val:
            best_val, best_ep = val, ep
            best_p = {k: v.copy() for k, v in net.p.items()}
        elif ep - best_ep >= patience:
            break
    net.p = best_p
    return {"best_val": best_val, "best_epoch": best_ep, "history": hist}
```

Note on `test_train_restores_best_epoch`: parameters after one epoch differ from initial ones, and the restored set is the epoch-0 set, not the initial or last one.

- [ ] **Step 4: Run** `uv run pytest tests/test_gate_net.py -q` — Expected: 10 passed; full suite passes.

- [ ] **Step 5: Commit** `git commit -m "feat: add NumPy gate network with verified gradients and Adam training"`

---

### Task 5: Pooled data and walk-forward gate driver

**Files:** Create `src/volgate/gate/walkforward.py`; test `tests/test_gate_walkforward.py`.

**Interfaces:**
- Consumes: `gate_features`, `feature_columns`, `pairs`, `variances`, `MODELS`, `GateNet`, `train_gate`.
- Produces:
  - `GateConfig` frozen dataclass (fields as below) and `VARIANTS: dict[str, GateConfig]` with keys `gate, gate_novix, gate_noexpiry, gate_noscale, gate_variance, gate_qlike`
  - `Fold` frozen dataclass `(train_end, val_start, val_end, test_start, test_end)`; `FOLDS: list[Fold]`
  - `build_pooled(frames: dict[str, pd.DataFrame], alpha: float) -> pd.DataFrame` long frame: columns `asset, date, y,` all feature columns, `V__{m}, E__{m}, S2__{m}`; rows with any NaN dropped
  - `run_variant(long, cfg, alpha, folds, lam_grid=(0.0, 0.1, 1.0, 10.0), seeds=(0, 1, 2, 3, 4)) -> (pred, selection)`; `pred` columns `asset, date, fold, var, es, lam, wq_{m}..., g`; `selection` columns `fold, lam, val_loss`

- [ ] **Step 1: Write failing test**

```python
import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_combo_frame
from volgate.gate.walkforward import FOLDS, Fold, GateConfig, VARIANTS, build_pooled, run_variant

FAST = dict(epochs=3, patience=2, hidden=4)
TOY_FOLDS = [Fold("2019-06-30", "2019-07-01", "2019-12-31", "2020-01-01", "2020-06-30")]


def _frames():
    a = make_combo_frame(650, seed=0)
    b = make_combo_frame(650, seed=1).iloc[5:]  # different date set
    return {"a": a, "b": b}


def test_variants_and_folds():
    assert set(VARIANTS) == {"gate", "gate_novix", "gate_noexpiry", "gate_noscale",
                             "gate_variance", "gate_qlike"}
    assert FOLDS[-1].test_end == "2026-09-30"


def test_build_pooled_keeps_asset_dates_and_drops_nan():
    frames = _frames()
    long = build_pooled(frames, 0.025)
    assert set(long["asset"]) == {"a", "b"}
    assert not long.isna().any().any()
    assert set(long.loc[long.asset == "b", "date"]) <= set(frames["b"].index)


def test_run_variant_outputs_test_rows_only_and_valid_pairs():
    long = build_pooled(_frames(), 0.025)
    cfg = GateConfig("t", **FAST)
    pred, sel = run_variant(long, cfg, 0.025, TOY_FOLDS, lam_grid=(0.0, 1.0), seeds=(0, 1))
    assert pred["date"].min() >= pd.Timestamp("2020-01-01")
    assert pred["date"].max() <= pd.Timestamp("2020-06-30")
    assert (pred["es"] <= pred["var"]).all() and (pred["var"] < 0).all()
    assert len(sel) == 2


def test_constant_feature_does_not_break_scaling():
    long = build_pooled(_frames(), 0.025)
    long["own_is_expiry"] = 0.0  # zero std in training
    pred, _ = run_variant(long, GateConfig("t", **FAST), 0.025, TOY_FOLDS, lam_grid=(0.0,), seeds=(0,))
    assert np.isfinite(pred["var"]).all()


def test_test_period_returns_do_not_change_earlier_predictions():
    frames = _frames()
    cfg = GateConfig("t", **FAST)
    base, _ = run_variant(build_pooled(frames, 0.025), cfg, 0.025, TOY_FOLDS, (0.0,), (0,))
    j = frames["a"].index.get_loc(pd.Timestamp(base["date"].iloc[len(base) // 2]))
    pert = {k: v.copy() for k, v in frames.items()}
    pert["a"].iloc[j, pert["a"].columns.get_loc("log_return")] = -0.5
    out, _ = run_variant(build_pooled(pert, 0.025), cfg, 0.025, TOY_FOLDS, (0.0,), (0,))
    cut = frames["a"].index[j]
    m = (base.asset == "a") & (base.date <= cut)
    pd.testing.assert_frame_equal(base[m].reset_index(drop=True), out[m].reset_index(drop=True))
```

The last test perturbs a test-period return: the model is trained on train/validation rows only, and features for days up to and including that date do not use it, so those predictions must be identical.

- [ ] **Step 2: Run** — Expected: module missing.

- [ ] **Step 3: Implement `src/volgate/gate/walkforward.py`**

```python
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from volgate.combine.data import MODELS, pairs, variances
from volgate.gate.features import feature_columns, gate_features
from volgate.gate.net import GateNet, train_gate


@dataclass(frozen=True)
class GateConfig:
    name: str
    mode: str = "pairs"
    loss: str = "fz0"
    use_scale: bool = True
    use_vix: bool = True
    use_expiry: bool = True
    hidden: int = 16
    emb: int = 4
    dropout: float = 0.1
    wd: float = 1e-4
    lr: float = 1e-3
    epochs: int = 300
    batch: int = 256
    patience: int = 20


VARIANTS = {c.name: c for c in [
    GateConfig("gate"),
    GateConfig("gate_novix", use_vix=False),
    GateConfig("gate_noexpiry", use_expiry=False),
    GateConfig("gate_noscale", use_scale=False),
    GateConfig("gate_variance", mode="variance", use_scale=False),
    GateConfig("gate_qlike", mode="variance", loss="qlike", use_scale=False),
]}


@dataclass(frozen=True)
class Fold:
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str


FOLDS = [
    Fold("2021-12-31", "2022-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    Fold("2022-12-31", "2023-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    Fold("2023-12-31", "2024-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    Fold("2024-12-31", "2025-01-01", "2025-12-31", "2026-01-01", "2026-09-30"),
]


def build_pooled(frames: dict[str, pd.DataFrame], alpha: float) -> pd.DataFrame:
    parts = []
    for asset, df in frames.items():
        f = gate_features(df, alpha)
        V, E = pairs(df, alpha)
        S2 = variances(df)
        for i, m in enumerate(MODELS):
            f[f"V__{m}"], f[f"E__{m}"], f[f"S2__{m}"] = V[:, i], E[:, i], S2[:, i]
        f["y"] = df["log_return"].to_numpy()
        f.insert(0, "date", df.index)
        f.insert(0, "asset", asset)
        parts.append(f.reset_index(drop=True))
    return pd.concat(parts, ignore_index=True).dropna().reset_index(drop=True)


def _batch(rows, cols, mean, std, assets):
    return {
        "X": ((rows[cols].to_numpy(float) - mean) / std),
        "a": rows["asset"].map(assets).to_numpy(int),
        "V": rows[[f"V__{m}" for m in MODELS]].to_numpy(float),
        "E": rows[[f"E__{m}" for m in MODELS]].to_numpy(float),
        "S2": rows[[f"S2__{m}" for m in MODELS]].to_numpy(float),
        "y": rows["y"].to_numpy(float),
    }


def _fit_nu(rows) -> float:
    s2 = rows[[f"S2__{m}" for m in MODELS]].to_numpy(float).mean(axis=1)
    z = rows["y"].to_numpy(float) / np.sqrt(s2)
    return float(max(stats.t.fit(z, floc=0)[0], 2.5))


def _net(cfg, n_feat, n_assets, alpha, nu, lam, seed):
    return GateNet(n_feat, n_assets, len(MODELS), hidden=cfg.hidden, emb=cfg.emb,
                   dropout=cfg.dropout, mode=cfg.mode, loss=cfg.loss, use_scale=cfg.use_scale,
                   alpha=alpha, nu=nu, lam_eq=lam, wd=cfg.wd, seed=seed)


def run_variant(long, cfg, alpha, folds, lam_grid=(0.0, 0.1, 1.0, 10.0), seeds=(0, 1, 2, 3, 4)):
    cols = feature_columns(cfg.use_vix, cfg.use_expiry)
    assets = {a: i for i, a in enumerate(sorted(long["asset"].unique()))}
    preds, selection = [], []
    for k, fold in enumerate(folds, start=1):
        d = long["date"]
        tr_rows = long[d <= fold.train_end]
        va_rows = long[(d >= fold.val_start) & (d <= fold.val_end)]
        te_rows = long[(d >= fold.test_start) & (d <= fold.test_end)]
        mean = tr_rows[cols].to_numpy(float).mean(axis=0)
        std = tr_rows[cols].to_numpy(float).std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        tr, va, te = (_batch(r, cols, mean, std, assets) for r in (tr_rows, va_rows, te_rows))
        nu = _fit_nu(tr_rows) if cfg.mode == "variance" else None
        best_lam, best_val = None, np.inf
        for lam in lam_grid:
            net = _net(cfg, len(cols), len(assets), alpha, nu, lam, seeds[0])
            info = train_gate(net, tr, va, lr=cfg.lr, epochs=cfg.epochs, batch=cfg.batch,
                              patience=cfg.patience, seed=seeds[0])
            selection.append({"fold": k, "lam": lam, "val_loss": info["best_val"]})
            if info["best_val"] < best_val:
                best_lam, best_val = lam, info["best_val"]
        outs = []
        for s in seeds:
            net = _net(cfg, len(cols), len(assets), alpha, nu, best_lam, s)
            train_gate(net, tr, va, lr=cfg.lr, epochs=cfg.epochs, batch=cfg.batch,
                       patience=cfg.patience, seed=s)
            outs.append(net.forward(te))
        out = pd.DataFrame({"asset": te_rows["asset"].to_numpy(), "date": te_rows["date"].to_numpy(),
                            "fold": k, "lam": best_lam,
                            "var": np.mean([o["var"] for o in outs], axis=0),
                            "es": np.mean([o["es"] for o in outs], axis=0)})
        wq = np.mean([o["wq"] for o in outs], axis=0)
        for i, m in enumerate(MODELS):
            out[f"wq_{m}"] = wq[:, i]
        out["g"] = np.mean([o.get("g", np.ones(len(te_rows))) for o in outs], axis=0)
        preds.append(out)
    return pd.concat(preds, ignore_index=True), pd.DataFrame(selection)
```

- [ ] **Step 4: Run** `uv run pytest tests/test_gate_walkforward.py -q` — Expected: 5 passed; full suite passes.

- [ ] **Step 5: Commit** `git commit -m "feat: add pooled walk-forward gate driver with variants"`

---

### Task 6: Pipeline scripts and first comparison table

**Files:** Create `scripts/04_combinations.py`, `scripts/05_gate.py`, `scripts/06_fz0_table.py`; modify `Makefile`, `README.md`.

**Interfaces:**
- Produces:
  - `data/processed/combos/{tag}/{asset}.csv`: index `date`; columns `{method}__var, {method}__es` for `mean, median, min_score, relative_score, previous_best`, plus `min_score__wq_{m}` and `relative_score__wq_{m}`
  - `data/processed/gate/{tag}/{variant}.csv`: rows from `run_variant`
  - `results/tables/gate_selection.csv`: `alpha, variant, fold, lam, val_loss`
  - `results/tables/test_fz0.csv`: `alpha, asset, method, n, mean_fz0` over common test dates 2023-01-01..2026-09-30 for the 6 base models, 5 baselines and every gate variant run at that alpha; `asset = "ALL"` rows pool all assets

- [ ] **Step 1: Write `scripts/04_combinations.py`**

```python
import pandas as pd

from volgate.combine.baselines import mean_combo, median_combo, rolling_combination
from volgate.combine.data import MODELS, load_asset, pairs
from volgate.config import REPO_ROOT, load_config
from volgate.risk.dist import alpha_tag

cfg = load_config()
P = {k: REPO_ROOT / v for k, v in cfg["paths"].items()}
for a in cfg["alphas"]:
    out_dir = P["processed"] / "combos" / alpha_tag(a)
    out_dir.mkdir(parents=True, exist_ok=True)
    for asset in cfg["assets"]:
        df = load_asset(P["processed"], asset)
        V, E = pairs(df, a)
        y = df["log_return"].to_numpy()
        out = pd.DataFrame(index=df.index)
        out["mean__var"], out["mean__es"] = mean_combo(V, E)
        out["median__var"], out["median__es"] = median_combo(V, E)
        for method in ("min_score", "relative_score", "previous_best"):
            v, e, W = rolling_combination(y, V, E, a, method, cfg["refit_every"], 250)
            out[f"{method}__var"], out[f"{method}__es"] = v, e
            if method != "previous_best":
                for i, m in enumerate(MODELS):
                    out[f"{method}__wq_{m}"] = W[:, i]
        out.to_csv(out_dir / f"{asset}.csv", date_format="%Y-%m-%d")
        print(f"{alpha_tag(a)} {asset} done")
```

- [ ] **Step 2: Write `scripts/05_gate.py`**

```python
import time

import pandas as pd

from volgate.combine.data import load_asset
from volgate.config import REPO_ROOT, load_config
from volgate.gate.walkforward import FOLDS, VARIANTS, build_pooled, run_variant
from volgate.risk.dist import alpha_tag

cfg = load_config()
P = {k: REPO_ROOT / v for k, v in cfg["paths"].items()}
frames = {a: load_asset(P["processed"], a) for a in cfg["assets"]}
runs = [(0.025, v) for v in VARIANTS] + [(0.01, "gate"), (0.05, "gate")]
selections = []
for a, name in runs:
    t0 = time.time()
    out_dir = P["processed"] / "gate" / alpha_tag(a)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred, sel = run_variant(build_pooled(frames, a), VARIANTS[name], a, FOLDS)
    pred.to_csv(out_dir / f"{name}.csv", index=False, date_format="%Y-%m-%d")
    selections.append(sel.assign(alpha=a, variant=name))
    print(f"{alpha_tag(a)} {name} done in {time.time() - t0:.0f}s")
pd.concat(selections)[["alpha", "variant", "fold", "lam", "val_loss"]].to_csv(
    P["tables"] / "gate_selection.csv", index=False)
```

- [ ] **Step 3: Write `scripts/06_fz0_table.py`**

```python
from pathlib import Path

import numpy as np
import pandas as pd

from volgate.combine.data import MODELS, load_asset
from volgate.config import REPO_ROOT, load_config
from volgate.risk.dist import alpha_tag
from volgate.risk.fz import fz0_loss

TEST_START, TEST_END = "2023-01-01", "2026-09-30"
cfg = load_config()
P = {k: REPO_ROOT / v for k, v in cfg["paths"].items()}
rows = []
for a in cfg["alphas"]:
    t = alpha_tag(a)
    gate_files = sorted((P["processed"] / "gate" / t).glob("*.csv"))
    gates = {f.stem: pd.read_csv(f, parse_dates=["date"]) for f in gate_files}
    pooled = {}
    for asset in cfg["assets"]:
        df = load_asset(P["processed"], asset).loc[TEST_START:TEST_END]
        combos = pd.read_csv(P["processed"] / "combos" / t / f"{asset}.csv", index_col="date",
                             parse_dates=True).loc[TEST_START:TEST_END]
        fc = {m: (df[f"{m}__var_{t}"], df[f"{m}__es_{t}"]) for m in MODELS}
        for c in ("mean", "median", "min_score", "relative_score", "previous_best"):
            fc[c] = (combos[f"{c}__var"], combos[f"{c}__es"])
        for name, g in gates.items():
            gi = g[g.asset == asset].set_index("date")
            fc[name] = (gi["var"], gi["es"])
        common = df.index
        for v, _ in fc.values():
            common = common.intersection(v.dropna().index)
        y = df.loc[common, "log_return"].to_numpy()
        for name, (v, e) in fc.items():
            L = fz0_loss(y, v.loc[common].to_numpy(), e.loc[common].to_numpy(), a)
            rows.append({"alpha": a, "asset": asset, "method": name, "n": len(L), "mean_fz0": L.mean()})
            pooled.setdefault(name, []).append(L)
    for name, Ls in pooled.items():
        L = np.concatenate(Ls)
        rows.append({"alpha": a, "asset": "ALL", "method": name, "n": len(L), "mean_fz0": L.mean()})
table = pd.DataFrame(rows)
table.to_csv(P["tables"] / "test_fz0.csv", index=False)
print(table[table.asset == "ALL"].pivot(index="method", columns="alpha", values="mean_fz0").round(4))
```

- [ ] **Step 4: Update `Makefile`** — `all: prepare base combos gate table`, with targets `combos: uv run python scripts/04_combinations.py`, `gate: uv run python scripts/05_gate.py`, `table: uv run python scripts/06_fz0_table.py`.

- [ ] **Step 5: Run** `make combos gate table`. Expected: per-alpha/asset `done` lines; 8 gate runs with timings; a pooled FZ0 table with one row per method. Sanity checks: every method has the same `n` per asset and alpha; mean FZ0 values are within ±0.5 of the base-model values; gate rows exist for all four folds.

- [ ] **Step 6: Update README** status rows for "Baseline combinations" and "Neural gate and walk-forward training" to `Done (Plan 2)`; add the test FZ0 table location and a one-paragraph factual description of the pooled result (which methods rank lowest), without claims of significance (that is Plan 3).

- [ ] **Step 7: Run** `uv run pytest -q` — all pass. **Commit** scripts, Makefile, README, `results/tables/gate_selection.csv`, `results/tables/test_fz0.csv`.
