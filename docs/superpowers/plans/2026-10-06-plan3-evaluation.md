# Plan 3: Evaluation, Statistical Tests and Paper Tables — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Plan 2 forecasts into the paper's evidence: Diebold–Mariano tests, Model Confidence Sets, VaR/ES backtests, the expiry before/after analysis, the "when does the gate help" regression, LaTeX tables and one weights figure.

**Architecture:** `volgate.evaluate.stats` holds the tests (pure functions on arrays). `volgate.evaluate.analysis` holds HAC regressions. `volgate.evaluate.losses` assembles aligned test-period losses for every method. `scripts/07_evaluate.py` writes CSV and LaTeX tables to `results/tables/` and a figure to `results/figures/`.

**Tech Stack:** NumPy, SciPy, pandas, statsmodels (HAC OLS), arch (`arch.bootstrap.MCS`), matplotlib (figure only).

**Spec:** `docs/superpowers/specs/2026-10-05-fz-gate-design.md` section 6. Consumes Plan 2 outputs in `data/processed/combos/` and `data/processed/gate/`.

## Global Constraints

- Test period 2023-01-01 to 2026-09-30; every comparison uses dates common to all methods for that asset and α.
- Primary α = 0.025; DM and MCS also reported at 0.01 and 0.05 for the main gate and baselines.
- DM: Newey–West variance with Bartlett lag floor(4(n/100)^(2/9)), Harvey–Leybourne–Newbold correction, t(n−1) reference. Negative statistic means the first method has lower loss.
- Pooled ("ALL") tests use the cross-asset average loss per date over dates where all assets have forecasts.
- MCS: `arch.bootstrap.MCS`, size 0.10, 2000 reps, stationary bootstrap, method "R", seed 0.
- McNeil–Frey variant: exceedance residuals scaled by |ES| (no σ is available for pair combinations); one-sided bootstrap test of mean zero against mean below zero; this variant is stated in the table note.
- Tables never contain hardcoded status strings; every value is computed.

## Review Focus

1. A method with zero VaR hits in an asset's test period → Kupiec, Christoffersen and McNeil–Frey return finite statistics or NaN with n_hits = 0, never crash (Task 1 test).
2. Two methods with identical losses → DM returns NaN, not a division-by-zero warning turned into inf (Task 1 test).
3. Asset forecasts missing on some dates for one method → that asset uses the intersection of dates; pooled series only uses dates where every asset is present (Task 2 test).
4. HAC regression with a constant regressor column (e.g. no expiry days post-switch for an asset) → column dropped with a note, no singular-matrix crash (Task 3 test).
5. LaTeX writer with NaN values → renders "--" (Task 4 test).

---

### Task 1: Statistical tests

**Files:** Create `src/volgate/evaluate/__init__.py` (empty), `src/volgate/evaluate/stats.py`; test `tests/test_eval_stats.py`; add `statsmodels>=0.14` and `matplotlib>=3.8` to `pyproject.toml` dependencies.

**Interfaces:**
- Produces: `nw_variance(d, lags=None) -> float`; `dm_test(l1, l2, h=1) -> (stat, p)`; `kupiec(hits, alpha) -> (lr, p)`; `christoffersen(hits, alpha) -> dict(lr_ind, p_ind, lr_cc, p_cc)`; `dq_test(hits, var, alpha, lags=4) -> (stat, p)`; `mcneil_frey(y, var, es, reps=2000, seed=0) -> (mean_resid, p, n_hits)`; `mcs_table(losses: pd.DataFrame, size=0.10, reps=2000, seed=0) -> pd.DataFrame` (index method; columns `pvalue, in_mcs`).

- [ ] **Step 1: Write failing test**

```python
import numpy as np
import pandas as pd
import pytest

from volgate.evaluate.stats import (christoffersen, dm_test, dq_test, kupiec, mcneil_frey,
                                    mcs_table, nw_variance)


def test_nw_variance_iid_close_to_sample_variance():
    d = np.random.default_rng(0).normal(size=20000)
    assert nw_variance(d) == pytest.approx(1.0, rel=0.05)


def test_dm_detects_lower_loss_and_handles_identical():
    rng = np.random.default_rng(1)
    l2 = rng.normal(size=1000)
    stat, p = dm_test(l2 - 0.2 + rng.normal(0, 0.5, 1000), l2)
    assert stat < 0 and p < 0.01
    s, p = dm_test(l2, l2)
    assert np.isnan(s) and np.isnan(p)


def test_kupiec_exact_rate_and_zero_hits():
    hits = np.zeros(400, bool)
    hits[::40] = True  # 10 hits = 2.5%
    lr, p = kupiec(hits, 0.025)
    assert lr == pytest.approx(0.0, abs=1e-9) and p == pytest.approx(1.0)
    lr0, p0 = kupiec(np.zeros(400, bool), 0.025)
    assert np.isfinite(lr0) and p0 < 0.001


def test_christoffersen_flags_clustering():
    clustered = np.zeros(400, bool)
    clustered[100:110] = True
    spread = np.zeros(400, bool)
    spread[::40] = True
    assert christoffersen(clustered, 0.025)["p_ind"] < 0.01
    assert christoffersen(spread, 0.025)["p_ind"] > 0.1
    assert np.isfinite(christoffersen(np.zeros(400, bool), 0.025)["lr_cc"])


def test_dq_flags_clustered_hits():
    var = np.full(500, -0.02)
    clustered = np.zeros(500, bool)
    clustered[200:213] = True
    assert dq_test(clustered, var, 0.025)[1] < 0.01
    iid = np.random.default_rng(3).random(500) < 0.025
    assert dq_test(iid, var, 0.025)[1] > 0.01


def test_mcneil_frey_detects_too_shallow_es():
    rng = np.random.default_rng(4)
    y = rng.standard_normal(5000)
    var = np.full(5000, -1.96)
    good, bad = np.full(5000, -2.338), np.full(5000, -2.1)
    assert mcneil_frey(y, var, good)[1] > 0.05
    assert mcneil_frey(y, var, bad)[1] < 0.01
    m, p, n = mcneil_frey(y, np.full(5000, -10.0), np.full(5000, -11.0))
    assert n == 0 and np.isnan(m) and np.isnan(p)


def test_mcs_table_excludes_clearly_worse():
    rng = np.random.default_rng(5)
    L = pd.DataFrame({"a": rng.normal(0, 1, 800), "b": rng.normal(1.0, 1, 800)})
    t = mcs_table(L, reps=300)
    assert bool(t.loc["a", "in_mcs"]) and not bool(t.loc["b", "in_mcs"])
```

- [ ] **Step 2: Run** `uv run pytest tests/test_eval_stats.py -q` — Expected: module missing.

- [ ] **Step 3: Implement `src/volgate/evaluate/stats.py`**

```python
import numpy as np
import pandas as pd
from arch.bootstrap import MCS
from scipy import stats


def _bartlett_lags(n: int) -> int:
    return int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


def nw_variance(d, lags=None) -> float:
    d = np.asarray(d, dtype=float)
    n = len(d)
    lags = _bartlett_lags(n) if lags is None else lags
    u = d - d.mean()
    v = u @ u / n
    for k in range(1, lags + 1):
        v += 2.0 * (1.0 - k / (lags + 1.0)) * (u[k:] @ u[:-k]) / n
    return float(v)


def dm_test(l1, l2, h: int = 1):
    """Diebold-Mariano with HLN correction; negative stat means l1 has lower mean loss."""
    d = np.asarray(l1, dtype=float) - np.asarray(l2, dtype=float)
    n = len(d)
    v = nw_variance(d)
    if v <= 0:
        return np.nan, np.nan
    stat = d.mean() / np.sqrt(v / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(stat), float(2 * stats.t.sf(abs(stat), df=n - 1))


def _xlogy(x, y):
    return x * np.log(y) if x > 0 else 0.0


def kupiec(hits, alpha):
    hits = np.asarray(hits, dtype=bool)
    n, x = len(hits), int(hits.sum())
    ll0 = _xlogy(n - x, 1 - alpha) + _xlogy(x, alpha)
    ll1 = _xlogy(n - x, 1 - x / n) + _xlogy(x, x / n)
    lr = -2.0 * (ll0 - ll1)
    return float(lr), float(stats.chi2.sf(lr, 1))


def christoffersen(hits, alpha) -> dict:
    h = np.asarray(hits, dtype=int)
    prev, cur = h[:-1], h[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll1 = _xlogy(n00, 1 - pi01) + _xlogy(n01, pi01) + _xlogy(n10, 1 - pi11) + _xlogy(n11, pi11)
    ll0 = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    lr_ind = -2.0 * (ll0 - ll1)
    lr_cc = lr_ind + kupiec(hits, alpha)[0]
    return {"lr_ind": float(lr_ind), "p_ind": float(stats.chi2.sf(lr_ind, 1)),
            "lr_cc": float(lr_cc), "p_cc": float(stats.chi2.sf(lr_cc, 2))}


def dq_test(hits, var, alpha, lags: int = 4):
    """Engle-Manganelli dynamic quantile test: regress Hit-alpha on lags and VaR."""
    h = np.asarray(hits, dtype=float) - alpha
    var = np.asarray(var, dtype=float)
    n = len(h)
    X = np.column_stack([np.ones(n - lags)] + [h[lags - k:n - k] for k in range(1, lags + 1)]
                        + [var[lags:]])
    y = h[lags:]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    dq = float(beta @ X.T @ X @ beta / (alpha * (1 - alpha)))
    return dq, float(stats.chi2.sf(dq, X.shape[1]))


def mcneil_frey(y, var, es, reps: int = 2000, seed: int = 0):
    """Mean of (y - ES)/|ES| on VaR exceedances; one-sided bootstrap p for mean < 0."""
    y, var, es = (np.asarray(a, dtype=float) for a in (y, var, es))
    hit = y <= var
    r = (y[hit] - es[hit]) / np.abs(es[hit])
    if r.size < 2:
        return np.nan, np.nan, int(r.size)
    rng = np.random.default_rng(seed)
    c = r - r.mean()
    boot = c[rng.integers(0, r.size, (reps, r.size))].mean(axis=1)
    return float(r.mean()), float(np.mean(boot <= r.mean())), int(r.size)


def mcs_table(losses: pd.DataFrame, size: float = 0.10, reps: int = 2000, seed: int = 0) -> pd.DataFrame:
    m = MCS(losses, size=size, reps=reps, method="R", bootstrap="stationary", seed=seed)
    m.compute()
    p = m.pvalues["Pvalue"].reindex(losses.columns)
    return pd.DataFrame({"pvalue": p, "in_mcs": p.index.isin(m.included)})
```

- [ ] **Step 4: Run** — Expected: 7 passed; full suite passes.

- [ ] **Step 5: Commit** `git commit -m "feat: add DM, MCS, Kupiec, Christoffersen, DQ and McNeil-Frey tests"`

---

### Task 2: Aligned test-period losses

**Files:** Create `src/volgate/evaluate/losses.py`; test `tests/test_eval_losses.py`.

**Interfaces:**
- Consumes: `load_asset`, `MODELS`, `fz0_loss`.
- Produces:
  - `BASELINES = ["mean", "median", "min_score", "relative_score", "previous_best"]`
  - `collect_forecasts(processed, asset, alpha, start, end) -> dict[str, pd.DataFrame]` each with columns `var, es` indexed by date, for base models, baselines and every gate file present
  - `aligned(fc: dict, y: pd.Series) -> (dates, y_arr, {name: (var, es)})` on common non-NaN dates
  - `loss_frame(fc_aligned, y, alpha) -> pd.DataFrame` (FZ0 per method, index dates)
  - `pooled_average(per_asset: dict[str, pd.DataFrame]) -> pd.DataFrame` (mean across assets on dates present for all assets)

- [ ] **Step 1: Write failing test**

```python
import numpy as np
import pandas as pd

from volgate.evaluate.losses import aligned, loss_frame, pooled_average


def test_aligned_uses_common_dates():
    idx = pd.bdate_range("2023-01-02", periods=5)
    y = pd.Series(np.linspace(-0.02, 0.02, 5), index=idx)
    a = pd.DataFrame({"var": -0.03, "es": -0.04}, index=idx)
    b = a.copy()
    b.iloc[1] = np.nan
    dates, yv, fc = aligned({"a": a, "b": b}, y)
    assert len(dates) == 4 and idx[1] not in dates
    L = loss_frame(fc, yv, 0.025)
    assert list(L.columns) == ["a", "b"] and len(L) == 4


def test_pooled_average_requires_all_assets():
    i1 = pd.bdate_range("2023-01-02", periods=4)
    x = pd.DataFrame({"m": [1.0, 2.0, 3.0, 4.0]}, index=i1)
    y = pd.DataFrame({"m": [3.0, 4.0, 5.0]}, index=i1[1:])
    p = pooled_average({"x": x, "y": y})
    assert list(p.index) == list(i1[1:])
    assert p["m"].tolist() == [2.5, 3.5, 4.5]
```

- [ ] **Step 2: Run** — Expected: module missing.

- [ ] **Step 3: Implement `src/volgate/evaluate/losses.py`**

```python
from pathlib import Path

import pandas as pd

from volgate.combine.data import MODELS, load_asset
from volgate.risk.dist import alpha_tag
from volgate.risk.fz import fz0_loss

BASELINES = ["mean", "median", "min_score", "relative_score", "previous_best"]


def collect_forecasts(processed: Path, asset: str, alpha: float, start: str, end: str):
    processed = Path(processed)
    t = alpha_tag(alpha)
    df = load_asset(processed, asset).loc[start:end]
    fc = {m: df[[f"{m}__var_{t}", f"{m}__es_{t}"]].set_axis(["var", "es"], axis=1) for m in MODELS}
    combos = pd.read_csv(processed / "combos" / t / f"{asset}.csv", index_col="date",
                         parse_dates=True).loc[start:end]
    for c in BASELINES:
        fc[c] = combos[[f"{c}__var", f"{c}__es"]].set_axis(["var", "es"], axis=1)
    for f in sorted((processed / "gate" / t).glob("*.csv")):
        g = pd.read_csv(f, parse_dates=["date"])
        fc[f.stem] = g[g["asset"] == asset].set_index("date")[["var", "es"]]
    return fc, df


def aligned(fc: dict, y: pd.Series):
    dates = y.dropna().index
    for v in fc.values():
        dates = dates.intersection(v.dropna().index)
    return dates, y.loc[dates].to_numpy(), {k: (v.loc[dates, "var"].to_numpy(),
                                               v.loc[dates, "es"].to_numpy()) for k, v in fc.items()}


def loss_frame(fc_aligned: dict, y, alpha: float, dates=None) -> pd.DataFrame:
    data = {k: fz0_loss(y, v, e, alpha) for k, (v, e) in fc_aligned.items()}
    return pd.DataFrame(data, index=dates)


def pooled_average(per_asset: dict[str, pd.DataFrame]) -> pd.DataFrame:
    common = None
    for L in per_asset.values():
        common = L.index if common is None else common.intersection(L.index)
    return sum(L.loc[common] for L in per_asset.values()) / len(per_asset)
```

Note: `loss_frame` gets `dates=dates` from `aligned` in the script; the test checks only length and columns.

- [ ] **Step 4: Run** — Expected: 2 passed; full suite passes.

- [ ] **Step 5: Commit** `git commit -m "feat: add aligned test-period loss assembly"`

---

### Task 3: HAC regressions for the expiry and state analyses

**Files:** Create `src/volgate/evaluate/analysis.py`; test `tests/test_eval_analysis.py`.

**Interfaces:**
- Produces: `hac_ols(y, X: pd.DataFrame) -> pd.DataFrame` (index regressor incl. `const`; columns `coef, se, t, p`; constant regressor columns other than `const` dropped); `expiry_did(d: pd.Series, is_expiry: pd.Series, switch_date="2025-09-01") -> pd.DataFrame`; `state_regression(d: pd.Series, states: pd.DataFrame) -> pd.DataFrame` (states standardized).

- [ ] **Step 1: Write failing test**

```python
import numpy as np
import pandas as pd
import pytest

from volgate.evaluate.analysis import expiry_did, hac_ols, state_regression


def test_hac_ols_recovers_coefficients_and_drops_constant_columns():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=2000), "flat": 1.0})
    y = 0.5 + 2.0 * X["x"] + rng.normal(0, 0.1, 2000)
    t = hac_ols(y, X)
    assert "flat" not in t.index
    assert t.loc["x", "coef"] == pytest.approx(2.0, abs=0.02)
    assert t.loc["const", "coef"] == pytest.approx(0.5, abs=0.02)


def test_expiry_did_picks_up_interaction():
    idx = pd.bdate_range("2025-01-01", "2026-06-30")
    e = pd.Series((idx.dayofweek == 1).astype(float), index=idx)
    post = idx >= pd.Timestamp("2025-09-01")
    rng = np.random.default_rng(1)
    d = pd.Series(rng.normal(0, 0.1, len(idx)) - 0.5 * e.to_numpy() * post, index=idx)
    t = expiry_did(d, e)
    assert t.loc["expiry_x_post", "coef"] == pytest.approx(-0.5, abs=0.1)
    assert t.loc["expiry_x_post", "p"] < 0.01


def test_state_regression_standardizes():
    idx = pd.bdate_range("2023-01-02", periods=500)
    rng = np.random.default_rng(2)
    s = pd.DataFrame({"vix": rng.normal(15, 5, 500)}, index=idx)
    d = pd.Series(0.1 * (s["vix"] - 15) / 5 + rng.normal(0, 0.05, 500), index=idx)
    t = state_regression(d, s)
    assert t.loc["vix", "coef"] == pytest.approx(0.1, abs=0.02)
```

- [ ] **Step 2: Run** — Expected: module missing.

- [ ] **Step 3: Implement `src/volgate/evaluate/analysis.py`**

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm


def hac_ols(y, X: pd.DataFrame) -> pd.DataFrame:
    X = X.loc[:, X.std() > 0]
    n = len(X)
    lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    res = sm.OLS(np.asarray(y, dtype=float), sm.add_constant(X, has_constant="add")).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags})
    return pd.DataFrame({"coef": res.params, "se": res.bse, "t": res.tvalues, "p": res.pvalues})


def expiry_did(d: pd.Series, is_expiry: pd.Series, switch_date: str = "2025-09-01") -> pd.DataFrame:
    post = (d.index >= pd.Timestamp(switch_date)).astype(float)
    e = is_expiry.reindex(d.index).astype(float).to_numpy()
    X = pd.DataFrame({"expiry": e, "post": post, "expiry_x_post": e * post}, index=d.index)
    return hac_ols(d.to_numpy(), X)


def state_regression(d: pd.Series, states: pd.DataFrame) -> pd.DataFrame:
    s = states.reindex(d.index)
    z = (s - s.mean()) / s.std()
    return hac_ols(d.to_numpy(), z)
```

- [ ] **Step 4: Run** — Expected: 3 passed; full suite passes.

- [ ] **Step 5: Commit** `git commit -m "feat: add HAC regressions for expiry and state analyses"`

---

### Task 4: LaTeX table writer

**Files:** Create `src/volgate/evaluate/latex.py`; test `tests/test_eval_latex.py`.

**Interfaces:** `to_latex(df: pd.DataFrame, caption: str, label: str, digits: int = 3, note: str | None = None) -> str` — booktabs table; NaN rendered `--`; index becomes first column.

- [ ] **Step 1: Write failing test**

```python
import numpy as np
import pandas as pd

from volgate.evaluate.latex import to_latex


def test_latex_renders_nan_and_structure():
    df = pd.DataFrame({"a": [1.23456, np.nan], "b": ["x_y", "z"]}, index=["m1", "m2"])
    s = to_latex(df, "Cap", "tab:x", digits=2, note="Note text.")
    assert "\\toprule" in s and "\\bottomrule" in s and "\\label{tab:x}" in s
    assert "1.23" in s and "--" in s and "x\\_y" in s and "Note text." in s
```

- [ ] **Step 2: Run** — Expected: module missing.

- [ ] **Step 3: Implement `src/volgate/evaluate/latex.py`**

```python
import numpy as np
import pandas as pd


def _cell(v, digits):
    if isinstance(v, (float, np.floating)):
        return "--" if np.isnan(v) else f"{v:.{digits}f}"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    return str(v).replace("_", "\\_")


def to_latex(df: pd.DataFrame, caption: str, label: str, digits: int = 3,
             note: str | None = None) -> str:
    cols = [str(df.index.name or "")] + [str(c) for c in df.columns]
    lines = ["\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}",
             f"\\label{{{label}}}", "\\begin{tabular}{l" + "r" * len(df.columns) + "}",
             "\\toprule", " & ".join(_cell(c, digits) for c in cols) + " \\\\", "\\midrule"]
    for idx, row in df.iterrows():
        lines.append(" & ".join([_cell(idx, digits)] + [_cell(v, digits) for v in row]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if note:
        lines.append(f"\\par\\footnotesize {note}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run** — Expected: 1 passed; full suite passes.

- [ ] **Step 5: Commit** `git commit -m "feat: add LaTeX table writer"`

---

### Task 5: Evaluation script, tables, figure, README

**Files:** Create `scripts/07_evaluate.py`; modify `Makefile` (`evaluate` target, added to `all`), `README.md`.

**Outputs** (CSV in `results/tables/`, LaTeX in `results/tables/tex/`, figure in `results/figures/`):
- `dm_gate.csv`: alpha, asset (8 + ALL), other method, DM stat, p — gate vs every other method
- `mcs.csv`: alpha, asset, method, pvalue, in_mcs
- `backtests.csv`: alpha, asset, method, n, hit_rate, kupiec_p, cc_p, dq_p, mf_mean, mf_p, mf_n
- `expiry_did.csv`: regression of (gate − mean) FZ0 differential on own expiry, post-switch and interaction, NIFTY and pooled stocks, α = 0.025
- `state_regression.csv`: pooled (gate − mean) differential on standardized vov30, vix, vrp, loss dispersion, α = 0.025
- `gate_weights_expiry.csv`: mean gate weights by model on expiry vs non-expiry days, pre vs post switch, α = 0.025
- `tex/main_results.tex`: pooled mean FZ0 (α = 1%, 2.5%, 5%), MCS membership and DM vs gate at 2.5%
- `tex/backtests.tex`: pooled-over-assets count of rejections at 5% per test for each method, α = 0.025
- `figures/gate_weights.png`: monthly mean gate VaR weights by model (stacked area) and mean scale g, α = 0.025

- [ ] **Step 1: Write `scripts/07_evaluate.py`**

```python
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from volgate.combine.data import MODELS, load_asset  # noqa: E402
from volgate.config import REPO_ROOT, load_config  # noqa: E402
from volgate.evaluate.analysis import expiry_did, state_regression  # noqa: E402
from volgate.evaluate.latex import to_latex  # noqa: E402
from volgate.evaluate.losses import aligned, collect_forecasts, loss_frame, pooled_average  # noqa: E402
from volgate.evaluate.stats import (christoffersen, dm_test, dq_test, kupiec,  # noqa: E402
                                    mcneil_frey, mcs_table)
from volgate.gate.features import gate_features  # noqa: E402
from volgate.risk.dist import alpha_tag  # noqa: E402

START, END, MAIN = "2023-01-01", "2026-09-30", 0.025
cfg = load_config()
P = {k: REPO_ROOT / v for k, v in cfg["paths"].items()}
TEX, FIG = P["tables"] / "tex", REPO_ROOT / "results" / "figures"
TEX.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

dm_rows, mcs_rows, bt_rows, mean_rows = [], [], [], []
losses_main = {}
for a in cfg["alphas"]:
    per_asset = {}
    for asset in cfg["assets"]:
        fc, df = collect_forecasts(P["processed"], asset, a, START, END)
        dates, y, fca = aligned(fc, df["log_return"])
        L = loss_frame(fca, y, a, dates)
        per_asset[asset] = L
        for name, (v, e) in fca.items():
            hits = y <= v
            cc = christoffersen(hits, a)
            mf = mcneil_frey(y, v, e)
            bt_rows.append({"alpha": a, "asset": asset, "method": name, "n": len(y),
                            "hit_rate": hits.mean(), "kupiec_p": kupiec(hits, a)[1],
                            "cc_p": cc["p_cc"], "dq_p": dq_test(hits, v, a)[1],
                            "mf_mean": mf[0], "mf_p": mf[1], "mf_n": mf[2]})
    per_asset["ALL"] = pooled_average({k: v for k, v in per_asset.items()})
    if a == MAIN:
        losses_main = per_asset
    for asset, L in per_asset.items():
        for other in L.columns.drop("gate"):
            s, p = dm_test(L["gate"], L[other])
            dm_rows.append({"alpha": a, "asset": asset, "other": other, "dm_stat": s, "p": p})
        t = mcs_table(L)
        for name, r in t.iterrows():
            mcs_rows.append({"alpha": a, "asset": asset, "method": name,
                             "pvalue": r["pvalue"], "in_mcs": bool(r["in_mcs"])})
        for name in L.columns:
            mean_rows.append({"alpha": a, "asset": asset, "method": name, "mean_fz0": L[name].mean()})

dm, mcs, bt, means = (pd.DataFrame(r) for r in (dm_rows, mcs_rows, bt_rows, mean_rows))
for name, t in [("dm_gate", dm), ("mcs", mcs), ("backtests", bt)]:
    t.to_csv(P["tables"] / f"{name}.csv", index=False)

# Expiry and state analyses at the main alpha (gate minus equal-weight mean).
did_rows, panels, weights_rows = [], [], []
gate = pd.read_csv(P["processed"] / "gate" / alpha_tag(MAIN) / "gate.csv", parse_dates=["date"])
for asset in cfg["assets"]:
    L = losses_main[asset]
    df = load_asset(P["processed"], asset).reindex(L.index)
    d = L["gate"] - L["mean"]
    st = gate_features(load_asset(P["processed"], asset), MAIN).reindex(L.index)
    disp = st[[f"fz_{m}" for m in MODELS]].std(axis=1)
    panels.append(pd.DataFrame({"d": d, "expiry": df["own_is_expiry"], "asset": asset,
                                "vov30": st["vov30"], "vix": st["vix"], "vrp": st["vrp"],
                                "loss_dispersion": disp}))
    g = gate[gate.asset == asset].set_index("date").reindex(L.index)
    g["expiry"] = df["own_is_expiry"].to_numpy()
    g["post"] = g.index >= pd.Timestamp("2025-09-01")
    w = g.groupby(["post", "expiry"])[[f"wq_{m}" for m in MODELS] + ["g"]].mean()
    weights_rows.append(w.assign(asset=asset).reset_index())
panel = pd.concat(panels)
nifty = panel[panel.asset == "nifty50"]
stocks = panel[~panel.asset.isin(["nifty50", "banknifty"])].groupby(level=0).mean(numeric_only=True)
for name, p in [("nifty50", nifty), ("stocks_avg", stocks)]:
    t = expiry_did(p["d"], p["expiry"]).assign(sample=name)
    did_rows.append(t.reset_index(names="term"))
pd.concat(did_rows).to_csv(P["tables"] / "expiry_did.csv", index=False)
avg = panel.groupby(level=0).mean(numeric_only=True)
state_regression(avg["d"], avg[["vov30", "vix", "vrp", "loss_dispersion"]]).reset_index(
    names="term").to_csv(P["tables"] / "state_regression.csv", index=False)
pd.concat(weights_rows).to_csv(P["tables"] / "gate_weights_expiry.csv", index=False)

# LaTeX: main results (pooled).
pooled = means[means.asset == "ALL"].pivot(index="method", columns="alpha", values="mean_fz0")
pooled.columns = [f"FZ0 a={c:g}" for c in pooled.columns]
m25 = mcs[(mcs.asset == "ALL") & (mcs.alpha == MAIN)].set_index("method")
d25 = dm[(dm.asset == "ALL") & (dm.alpha == MAIN)].set_index("other")
pooled["MCS p (2.5%)"] = m25["pvalue"]
pooled["DM gate vs (2.5%)"] = d25["dm_stat"]
pooled["DM p (2.5%)"] = d25["p"]
pooled.index.name = "method"
(TEX / "main_results.tex").write_text(to_latex(
    pooled.sort_values("FZ0 a=0.025"), "Pooled test-period FZ0 loss, 2023-01 to 2026-09",
    "tab:main", digits=4,
    note="Lower is better. DM: gate minus row method, HLN-corrected; negative favours the gate. "
         "Pooled series is the cross-asset average loss per date."))
b25 = bt[bt.alpha == MAIN].assign(
    kupiec=lambda x: x.kupiec_p < 0.05, cc=lambda x: x.cc_p < 0.05,
    dq=lambda x: x.dq_p < 0.05, mf=lambda x: x.mf_p < 0.05)
rej = b25.groupby("method")[["hit_rate", "kupiec", "cc", "dq", "mf"]].agg(
    {"hit_rate": "mean", "kupiec": "sum", "cc": "sum", "dq": "sum", "mf": "sum"})
rej.index.name = "method"
(TEX / "backtests.tex").write_text(to_latex(
    rej, "Backtest rejections at 5\\% across 8 assets, alpha = 2.5\\%", "tab:backtests", digits=3,
    note="Counts of assets where each test rejects. McNeil-Frey uses residuals scaled by |ES|."))

# Figure: monthly mean gate weights and scale.
gm = gate.set_index("date")[[f"wq_{m}" for m in MODELS] + ["g"]].resample("ME").mean()
fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
ax[0].stackplot(gm.index, gm[[f"wq_{m}" for m in MODELS]].T.to_numpy(), labels=MODELS)
ax[0].set_ylabel("VaR weight")
ax[0].legend(loc="upper left", ncol=3, fontsize=8)
ax[1].plot(gm.index, gm["g"])
ax[1].set_ylabel("scale g")
fig.tight_layout()
fig.savefig(FIG / "gate_weights.png", dpi=200)
print(pooled.sort_values("FZ0 a=0.025").round(4).to_string())
```

- [ ] **Step 2: Update `Makefile`**: add `evaluate: uv run python scripts/07_evaluate.py` and append `evaluate` to `all`.

- [ ] **Step 3: Run** `make evaluate`. Expected: printed pooled table; all listed files exist; `dm_gate.csv` has one row per (alpha, asset incl. ALL, other method); every p-value in [0, 1] or NaN.

- [ ] **Step 4: Update README**: status rows for "DM, MCS, VaR/ES backtests" and "Expiry and state analyses" → `Done (Plan 3)`; a "Results" section reporting, from the generated tables only, the pooled FZ0 ranking at α = 2.5%, whether the gate is in the MCS, the DM statistics against equal weights and Taylor's methods, backtest rejection counts, and the expiry and state regression coefficients with p-values. State results that do not favour the gate as plainly as those that do.

- [ ] **Step 5: Run** `uv run pytest -q`; commit scripts, Makefile, README, `results/tables/*.csv`, `results/tables/tex/*.tex`, `results/figures/gate_weights.png`.
