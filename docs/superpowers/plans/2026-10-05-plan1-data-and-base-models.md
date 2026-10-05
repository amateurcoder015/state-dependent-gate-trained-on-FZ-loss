# Plan 1: Data Pipeline and Base Risk Models — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce frozen raw data, a cleaned and audited daily panel with expiry features, and walk-forward out-of-sample VaR/ES forecasts from six base models for eight NSE assets.

**Architecture:** A `volgate` Python package under `src/`. Small modules for risk math (FZ0 loss, VaR/ES), data (download, panel, expiry calendar), and models (six base models sharing one `fit`/`filter` interface). A single walk-forward engine refits every 21 trading days on an expanding window. Numbered scripts run the stages; a Makefile chains them.

**Tech Stack:** Python 3.12, uv, numpy, pandas, scipy, arch, yfinance, pyyaml, pytest.

**Spec:** `docs/superpowers/specs/2026-10-05-fz-gate-design.md` (sections 2, 3, 7, 8). Plan 2 (combinations and gate) and Plan 3 (evaluation and tables) follow this plan.

## Global Constraints

- Python `>=3.12,<3.13`, managed with uv.
- Data range: 2015-01-01 to 2026-09-30 inclusive. Base-model out-of-sample forecasts start 2020-01-01.
- Refit every 21 trading days on an expanding window starting 2015-01-01; parameters fitted with rows strictly before the first row of a block.
- Tail levels: α ∈ {0.01, 0.025, 0.05}; primary 0.025. Column tags: `a010`, `a025`, `a050`.
- Returns: daily log returns from adjusted close. `arch` models are fitted on returns × 100.
- ES ≤ VaR < 0 for every forecast.
- Raw downloads are frozen with a SHA-256 manifest; later stages read only frozen files and verify the manifest first.
- No report or table contains hardcoded status strings; every value is computed.
- Generated data under `data/processed/` is not committed. `data/raw/`, `data/audit/`, `configs/`, and `results/tables/` are committed.

## Review Focus

1. Yahoo returns flat no-trade rows (open = high = low = close, volume 0) for stocks on some holidays → these rows must be dropped, not turned into zero-return days. Test in Task 5.
2. A day with high = low, or a negative Garman–Klass value → variance floored at 1e-10 so HAR logs stay finite. Tests in Task 5 and Task 7.
3. A model fit raises or does not converge → the walk-forward engine reuses the previous fit and flags the block; a failed first fit raises. Test in Task 9.
4. India VIX missing for two or more consecutive asset trading days → only one day is forward-filled, the rest stay NaN, and the VIX model returns NaN for those days without crashing. Tests in Task 5 and Task 7.
5. A computed expiry date falls on an exchange holiday → expiry moves to the previous trading day. Test in Task 6.

---

### Task 1: Project scaffold, config, CI

**Files:**
- Create: `pyproject.toml`, `src/volgate/__init__.py`, `src/volgate/config.py`, `configs/default.yaml`, `tests/test_config.py`, `.github/workflows/tests.yml`, `Makefile`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `volgate.config.load_config(path: str | Path | None = None) -> dict`, `volgate.config.REPO_ROOT: Path`

- [ ] **Step 1: Create empty `src/volgate/__init__.py` and write `pyproject.toml`**

```toml
[project]
name = "volgate"
version = "0.1.0"
description = "State-dependent neural gating of VaR/ES forecasts trained on FZ loss"
requires-python = ">=3.12,<3.13"
dependencies = [
    "numpy>=1.26",
    "pandas>=2.2",
    "scipy>=1.13",
    "arch>=7.0",
    "yfinance>=0.2.54",
    "pyyaml>=6.0",
]

[dependency-groups]
dev = ["pytest>=8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/volgate"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Write `configs/default.yaml`**

```yaml
start: "2015-01-01"
end: "2026-09-30"
oos_start: "2020-01-01"
refit_every: 21
alphas: [0.01, 0.025, 0.05]
outlier_threshold: 0.10
assets:
  nifty50: "^NSEI"
  banknifty: "^NSEBANK"
  adanient: "ADANIENT.NS"
  tatasteel: "TATASTEEL.NS"
  dlf: "DLF.NS"
  hindunilvr: "HINDUNILVR.NS"
  nestleind: "NESTLEIND.NS"
  sunpharma: "SUNPHARMA.NS"
vix: "^INDIAVIX"
contracts:
  nifty50: NIFTY
  banknifty: BANKNIFTY
  adanient: STOCK
  tatasteel: STOCK
  dlf: STOCK
  hindunilvr: STOCK
  nestleind: STOCK
  sunpharma: STOCK
paths:
  raw: data/raw
  audit: data/audit
  processed: data/processed
  tables: results/tables
  expiry_rules: configs/expiry_rules.yaml
```

- [ ] **Step 3: Write the failing test `tests/test_config.py`**

```python
import pytest
import yaml

from volgate.config import load_config


def test_default_config_loads():
    cfg = load_config()
    assert 0.025 in cfg["alphas"]
    assert cfg["refit_every"] == 21
    assert set(cfg["contracts"]) == set(cfg["assets"])


def test_missing_key_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"start": "2015-01-01"}))
    with pytest.raises(ValueError, match="missing keys"):
        load_config(p)
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv sync && uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'volgate.config'` (the empty package from Step 1 lets `uv sync` build)

- [ ] **Step 5: Write `src/volgate/config.py`**

```python
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

_REQUIRED = {
    "start", "end", "oos_start", "refit_every", "alphas", "outlier_threshold",
    "assets", "vix", "contracts", "paths",
}


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path is not None else REPO_ROOT / "configs" / "default.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    missing = _REQUIRED - set(cfg)
    if missing:
        raise ValueError(f"config missing keys: {sorted(missing)}")
    return cfg
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: 2 passed

- [ ] **Step 7: Write `.github/workflows/tests.yml`, `Makefile`, update `.gitignore`**

```yaml
name: tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync
      - run: uv run pytest -q
```

```makefile
.PHONY: all download prepare base test

all: prepare base

download:
	uv run python scripts/01_download.py

prepare:
	uv run python scripts/02_prepare.py

base:
	uv run python scripts/03_base_forecasts.py

test:
	uv run pytest -q
```

Append to `.gitignore`:

```
data/processed/
```

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock src configs tests .github Makefile .gitignore
git commit -m "build: scaffold volgate package, config loader, CI"
```

---

### Task 2: FZ0 loss

**Files:**
- Create: `src/volgate/risk/__init__.py` (empty), `src/volgate/risk/fz.py`
- Test: `tests/test_fz.py`

**Interfaces:**
- Produces: `volgate.risk.fz.fz0_loss(y, var, es, alpha: float) -> np.ndarray` (elementwise; raises `ValueError` if any ES ≥ 0)

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest
from scipy import stats

from volgate.risk.fz import fz0_loss


def test_known_value_with_hit():
    # hit: y <= var. Loss = -(v - y)/(a*e) + v/e + log(-e) - 1
    loss = fz0_loss(-0.05, -0.03, -0.04, 0.025)
    assert loss == pytest.approx(20 + 0.75 + np.log(0.04) - 1, rel=1e-12)


def test_known_value_without_hit():
    loss = fz0_loss(0.01, -0.03, -0.04, 0.025)
    assert loss == pytest.approx(0.75 + np.log(0.04) - 1, rel=1e-12)


def test_nonnegative_es_raises():
    with pytest.raises(ValueError, match="ES must be negative"):
        fz0_loss(np.array([0.0]), np.array([-0.01]), np.array([0.0]), 0.025)


def test_true_quantile_and_es_minimise_expected_loss():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(400_000)
    a = 0.025
    v = stats.norm.ppf(a)
    e = -stats.norm.pdf(v) / a
    true = fz0_loss(y, v, e, a).mean()
    for dv, de in [(1.1, 1.0), (0.9, 1.0), (1.0, 1.1), (1.0, 0.9)]:
        assert fz0_loss(y, v * dv, e * de, a).mean() > true
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_fz.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'volgate.risk'`

- [ ] **Step 3: Implement `src/volgate/risk/fz.py`**

```python
import numpy as np


def fz0_loss(y, var, es, alpha: float) -> np.ndarray:
    """FZ0 joint VaR/ES loss (Patton, Ziegel & Chen 2019), lower tail.

    L = -1/(alpha*es) * 1{y <= var} * (var - y) + var/es + log(-es) - 1
    """
    y = np.asarray(y, dtype=float)
    var = np.asarray(var, dtype=float)
    es = np.asarray(es, dtype=float)
    if np.any(es >= 0):
        raise ValueError("ES must be negative")
    hit = (y <= var).astype(float)
    return -hit * (var - y) / (alpha * es) + var / es + np.log(-es) - 1.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_fz.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/volgate/risk tests/test_fz.py
git commit -m "feat: add FZ0 joint VaR/ES loss"
```

---

### Task 3: VaR/ES from Student-t and empirical residuals

**Files:**
- Create: `src/volgate/risk/dist.py`
- Test: `tests/test_dist.py`

**Interfaces:**
- Produces:
  - `student_t_var_es(sigma, nu: float, alpha: float) -> tuple[np.ndarray, np.ndarray]` (unit-variance t; raises `ValueError` if nu ≤ 2)
  - `empirical_var_es(z, sigma, alpha: float) -> tuple[np.ndarray, np.ndarray]` (raises `ValueError` if fewer than `2*ceil(1/alpha)` finite residuals)
  - `alpha_tag(alpha: float) -> str` (0.025 → `"a025"`)

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest

from volgate.risk.dist import alpha_tag, empirical_var_es, student_t_var_es


def test_alpha_tag():
    assert alpha_tag(0.025) == "a025"
    assert alpha_tag(0.01) == "a010"
    assert alpha_tag(0.05) == "a050"


def test_student_t_large_nu_matches_normal():
    v, e = student_t_var_es(np.array([2.0]), 1e7, 0.025)
    assert v[0] == pytest.approx(2 * -1.959964, abs=1e-4)
    assert e[0] == pytest.approx(2 * -2.337803, abs=1e-4)


def test_student_t_matches_monte_carlo():
    nu, a = 5.0, 0.025
    rng = np.random.default_rng(1)
    x = rng.standard_t(nu, 2_000_000) * np.sqrt((nu - 2) / nu)
    q = np.quantile(x, a)
    v, e = student_t_var_es(1.0, nu, a)
    assert float(v) == pytest.approx(q, rel=0.01)
    assert float(e) == pytest.approx(x[x <= q].mean(), rel=0.01)


def test_student_t_rejects_small_nu():
    with pytest.raises(ValueError, match="nu"):
        student_t_var_es(1.0, 2.0, 0.025)


def test_empirical_var_es_known_values():
    z = np.linspace(-1, 1, 1001)
    v, e = empirical_var_es(z, np.array([2.0]), 0.025)
    assert v[0] == pytest.approx(2 * -0.95)
    assert e[0] == pytest.approx(2 * -0.975)


def test_empirical_ignores_nan_and_requires_enough_points():
    z = np.concatenate([np.linspace(-1, 1, 1001), [np.nan]])
    empirical_var_es(z, 1.0, 0.025)
    with pytest.raises(ValueError, match="residuals"):
        empirical_var_es(np.linspace(-1, 1, 50), 1.0, 0.025)


def test_es_below_var():
    v, e = student_t_var_es(np.array([0.01, 0.02]), 6.0, 0.01)
    assert np.all(e <= v) and np.all(v < 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dist.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'volgate.risk.dist'`

- [ ] **Step 3: Implement `src/volgate/risk/dist.py`**

```python
import numpy as np
from scipy import stats


def alpha_tag(alpha: float) -> str:
    return f"a{round(alpha * 1000):03d}"


def student_t_var_es(sigma, nu: float, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Lower-tail VaR and ES for sigma * (unit-variance Student-t with nu dof)."""
    if nu <= 2:
        raise ValueError(f"nu must exceed 2, got {nu}")
    q = stats.t.ppf(alpha, nu)
    scale = np.sqrt((nu - 2.0) / nu)
    es_std = -(nu + q**2) / (nu - 1.0) * stats.t.pdf(q, nu) / alpha
    sigma = np.asarray(sigma, dtype=float)
    return sigma * q * scale, sigma * es_std * scale


def empirical_var_es(z, sigma, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Lower-tail VaR and ES for sigma * z, z drawn from the empirical residuals."""
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    needed = 2 * int(np.ceil(1.0 / alpha))
    if z.size < needed:
        raise ValueError(f"need at least {needed} residuals, got {z.size}")
    q = np.quantile(z, alpha)
    es = z[z <= q].mean()
    sigma = np.asarray(sigma, dtype=float)
    return sigma * q, sigma * es
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dist.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/volgate/risk/dist.py tests/test_dist.py
git commit -m "feat: add Student-t and empirical VaR/ES"
```

---

### Task 4: Raw data download with SHA-256 manifest

**Files:**
- Create: `src/volgate/data/__init__.py` (empty), `src/volgate/data/download.py`, `scripts/01_download.py`
- Test: `tests/test_download.py`

**Interfaces:**
- Produces:
  - `sha256_file(path: Path) -> str`
  - `download_raw(tickers: dict[str, str], start: str, end: str, out_dir: Path) -> pd.DataFrame` — writes `{key}.csv` (index `date`; columns `open, high, low, close, adj_close, volume`) and `MANIFEST.csv` (columns `key, ticker, file, rows, first, last, sha256`). `end` is inclusive.
  - `verify_manifest(out_dir: Path) -> list[str]` — names of files whose hash does not match; empty list means intact.

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd
import pytest

from volgate.data import download as dl


def _fake_download(ticker, start, end, **kwargs):
    idx = pd.bdate_range(start, end, inclusive="left")
    return pd.DataFrame(
        {"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5, "Adj Close": 1.4, "Volume": 10},
        index=idx,
    )


def test_download_writes_files_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(dl.yf, "download", _fake_download)
    manifest = dl.download_raw({"x": "X.NS"}, "2024-01-01", "2024-01-05", tmp_path)
    df = pd.read_csv(tmp_path / "x.csv", index_col="date", parse_dates=True)
    assert list(df.columns) == ["open", "high", "low", "close", "adj_close", "volume"]
    assert df.index.max() == pd.Timestamp("2024-01-05")  # end is inclusive
    assert manifest.loc[0, "rows"] == 5
    assert dl.verify_manifest(tmp_path) == []


def test_verify_manifest_detects_change(tmp_path, monkeypatch):
    monkeypatch.setattr(dl.yf, "download", _fake_download)
    dl.download_raw({"x": "X.NS"}, "2024-01-01", "2024-01-05", tmp_path)
    with open(tmp_path / "x.csv", "a") as f:
        f.write("tampered\n")
    assert dl.verify_manifest(tmp_path) == ["x.csv"]


def test_empty_download_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(dl.yf, "download", lambda *a, **k: pd.DataFrame())
    with pytest.raises(RuntimeError, match="no data"):
        dl.download_raw({"x": "X.NS"}, "2024-01-01", "2024-01-05", tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_download.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'volgate.data'`

- [ ] **Step 3: Implement `src/volgate/data/download.py`**

```python
import hashlib
from pathlib import Path

import pandas as pd
import yfinance as yf

_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download_raw(tickers: dict[str, str], start: str, end: str, out_dir: Path) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    rows = []
    for key, ticker in tickers.items():
        df = yf.download(
            ticker, start=start, end=end_exclusive, auto_adjust=False,
            actions=False, progress=False, multi_level_index=False,
        )
        if df is None or df.empty:
            raise RuntimeError(f"no data for {ticker}")
        df = df.rename(columns=lambda c: c.lower().replace(" ", "_"))[_COLUMNS]
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        df.index.name = "date"
        path = out_dir / f"{key}.csv"
        df.to_csv(path, date_format="%Y-%m-%d")
        rows.append({
            "key": key, "ticker": ticker, "file": path.name, "rows": len(df),
            "first": df.index.min().strftime("%Y-%m-%d"),
            "last": df.index.max().strftime("%Y-%m-%d"),
            "sha256": sha256_file(path),
        })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "MANIFEST.csv", index=False)
    return manifest


def verify_manifest(out_dir: Path) -> list[str]:
    out_dir = Path(out_dir)
    manifest = pd.read_csv(out_dir / "MANIFEST.csv")
    return [
        row.file for row in manifest.itertuples()
        if sha256_file(out_dir / row.file) != row.sha256
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_download.py -v`
Expected: 3 passed

- [ ] **Step 5: Write `scripts/01_download.py`**

```python
from volgate.config import REPO_ROOT, load_config
from volgate.data.download import download_raw

cfg = load_config()
tickers = {**cfg["assets"], "india_vix": cfg["vix"]}
manifest = download_raw(tickers, cfg["start"], cfg["end"], REPO_ROOT / cfg["paths"]["raw"])
print(manifest[["key", "rows", "first", "last"]].to_string(index=False))
```

- [ ] **Step 6: Commit**

```bash
git add src/volgate/data tests/test_download.py scripts/01_download.py
git commit -m "feat: add frozen raw data download with SHA-256 manifest"
```

---

### Task 5: Panel construction, cleaning, outlier audit

**Files:**
- Create: `src/volgate/data/panel.py`, `data/audit/corrections.csv`
- Test: `tests/test_panel.py`

**Interfaces:**
- Consumes: raw CSV format from Task 4.
- Produces:
  - `load_raw(path: Path) -> pd.DataFrame` (DatetimeIndex `date`)
  - `gk_variance(df: pd.DataFrame) -> pd.Series` (Garman–Klass, floored at 1e-10)
  - `build_asset_frame(raw: pd.DataFrame, vix_close: pd.Series, drop_flat: bool = True) -> tuple[pd.DataFrame, dict]` — frame columns `open, high, low, close, adj_close, volume, log_return, gk_var, vix`; stats keys `rows, flat_dropped, vix_filled, vix_missing`
  - `outlier_table(frames: dict[str, pd.DataFrame], threshold: float) -> pd.DataFrame` (columns `asset, date, log_return, close_return`)
  - `apply_corrections(frames: dict[str, pd.DataFrame], corrections: pd.DataFrame) -> dict[str, pd.DataFrame]` (corrections columns `asset, date, action, note`; action `keep` or `drop`)

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pandas as pd
import pytest

from volgate.data.panel import apply_corrections, build_asset_frame, gk_variance, outlier_table


def _raw(n=6):
    idx = pd.bdate_range("2024-01-01", periods=n, name="date")
    close = 100 * np.exp(np.arange(n) * 0.01)
    return pd.DataFrame({
        "open": close * 0.99, "high": close * 1.02, "low": close * 0.97,
        "close": close, "adj_close": close, "volume": 1000,
    }, index=idx)


def test_gk_known_value_and_floor():
    df = pd.DataFrame({"open": [100.0, 100.0], "high": [110.0, 100.0],
                       "low": [100.0, 100.0], "close": [105.0, 101.0]})
    gk = gk_variance(df)
    assert gk.iloc[0] == pytest.approx(0.0036224, rel=1e-4)
    assert gk.iloc[1] == 1e-10  # high == low gives negative GK -> floored


def test_build_frame_uses_adj_close_and_drops_first_row():
    raw = _raw()
    raw["adj_close"] = raw["close"] * 0.5  # constant factor: returns unchanged
    vix = pd.Series(15.0, index=raw.index)
    df, stats = build_asset_frame(raw, vix)
    assert len(df) == 5 and stats["rows"] == 5
    assert df["log_return"].iloc[0] == pytest.approx(0.01)


def test_flat_no_trade_rows_are_dropped():
    raw = _raw()
    d = raw.index[3]
    raw.loc[d, ["open", "high", "low", "close", "adj_close"]] = raw.loc[raw.index[2], "close"]
    raw.loc[d, "volume"] = 0
    df, stats = build_asset_frame(raw, pd.Series(15.0, index=raw.index))
    assert stats["flat_dropped"] == 1
    assert d not in df.index
    assert (df["log_return"] != 0).all()


def test_vix_forward_fill_limited_to_one_day():
    raw = _raw(6)
    vix = pd.Series([10.0, 11.0, 12.0], index=raw.index[:3])  # missing last three days
    df, stats = build_asset_frame(raw, vix)
    assert df.loc[raw.index[3], "vix"] == 12.0  # filled once
    assert np.isnan(df.loc[raw.index[4], "vix"])
    assert stats["vix_filled"] == 1 and stats["vix_missing"] == 2


def test_outlier_table_and_corrections():
    raw = _raw()
    raw.loc[raw.index[3]:, ["adj_close", "close"]] *= 0.5  # -69% jump
    df, _ = build_asset_frame(raw, pd.Series(15.0, index=raw.index))
    table = outlier_table({"x": df}, 0.10)
    assert list(table["date"]) == [raw.index[3].strftime("%Y-%m-%d")]
    corr = pd.DataFrame({"asset": ["x"], "date": table["date"], "action": ["drop"], "note": ["test"]})
    fixed = apply_corrections({"x": df}, corr)
    assert raw.index[3] not in fixed["x"].index
    with pytest.raises(ValueError, match="unknown action"):
        apply_corrections({"x": df}, corr.assign(action="fix"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_panel.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'volgate.data.panel'`

- [ ] **Step 3: Implement `src/volgate/data/panel.py`**

```python
from pathlib import Path

import numpy as np
import pandas as pd

_GK_FLOOR = 1e-10


def load_raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col="date", parse_dates=True).sort_index()


def gk_variance(df: pd.DataFrame) -> pd.Series:
    hl = np.log(df["high"] / df["low"])
    co = np.log(df["close"] / df["open"])
    return (0.5 * hl**2 - (2 * np.log(2) - 1) * co**2).clip(lower=_GK_FLOOR)


def build_asset_frame(raw: pd.DataFrame, vix_close: pd.Series,
                      drop_flat: bool = True) -> tuple[pd.DataFrame, dict]:
    df = raw.sort_index().dropna(subset=["open", "high", "low", "close", "adj_close"]).copy()
    n_flat = 0
    if drop_flat:
        flat = ((df["high"] == df["low"]) & (df["open"] == df["close"])
                & (df["high"] == df["close"]) & (df["volume"] == 0))
        n_flat = int(flat.sum())
        df = df[~flat]
    df["log_return"] = np.log(df["adj_close"]).diff()
    df["gk_var"] = gk_variance(df)
    vix_close = vix_close.dropna()
    union = df.index.union(vix_close.index)
    df["vix"] = vix_close.reindex(union).ffill(limit=1).reindex(df.index)
    df = df.iloc[1:]
    stats = {
        "rows": len(df),
        "flat_dropped": n_flat,
        "vix_filled": int((~df.index.isin(vix_close.index) & df["vix"].notna()).sum()),
        "vix_missing": int(df["vix"].isna().sum()),
    }
    return df, stats


def outlier_table(frames: dict[str, pd.DataFrame], threshold: float) -> pd.DataFrame:
    rows = []
    for asset, df in frames.items():
        close_ret = np.log(df["close"]).diff()
        for d in df.index[df["log_return"].abs() > threshold]:
            rows.append({"asset": asset, "date": d.strftime("%Y-%m-%d"),
                         "log_return": df.at[d, "log_return"], "close_return": close_ret.get(d)})
    return pd.DataFrame(rows, columns=["asset", "date", "log_return", "close_return"])


def apply_corrections(frames: dict[str, pd.DataFrame],
                      corrections: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {k: v.copy() for k, v in frames.items()}
    for row in corrections.itertuples(index=False):
        if row.action == "keep":
            continue
        if row.action != "drop":
            raise ValueError(f"unknown action {row.action!r}")
        d = pd.Timestamp(row.date)
        if d not in out[row.asset].index:
            raise KeyError(f"{row.asset} {row.date} not in data")
        out[row.asset] = out[row.asset].drop(index=d)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_panel.py -v`
Expected: 5 passed

- [ ] **Step 5: Create `data/audit/corrections.csv` with header only**

```
asset,date,action,note
```

- [ ] **Step 6: Commit**

```bash
git add src/volgate/data/panel.py tests/test_panel.py data/audit/corrections.csv
git commit -m "feat: add panel builder with cleaning, GK variance, VIX alignment, outlier audit"
```

---

### Task 6: Rule-based expiry calendar

**Files:**
- Create: `src/volgate/data/expiry.py`, `configs/expiry_rules.yaml`
- Test: `tests/test_expiry.py`

**Interfaces:**
- Produces:
  - `ExpiryRule` frozen dataclass: `kind: str` (`weekly`|`monthly`), `weekday: int` (0=Mon), `start: pd.Timestamp`, `end: pd.Timestamp | None`, `source: str`
  - `load_rules(path: Path) -> dict[str, list[ExpiryRule]]`
  - `unverified(rules: dict[str, list[ExpiryRule]]) -> list[str]` — descriptions of rules whose source is `UNVERIFIED`
  - `expiry_dates(rules: list[ExpiryRule], trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex`
  - `expiry_features(dates: pd.DatetimeIndex, expiries: pd.DatetimeIndex, prefix: str) -> pd.DataFrame` — columns `{prefix}_is_expiry` (int 0/1), `{prefix}_days_to_expiry` (trading days to next expiry, 0 on expiry day)

- [ ] **Step 1: Write `configs/expiry_rules.yaml`** (initial rules; Task 10 verifies each against NSE/SEBI circulars and replaces `UNVERIFIED` with a source URL)

```yaml
NIFTY:
  - {kind: monthly, weekday: THU, start: "2015-01-01", end: "2025-08-31", source: UNVERIFIED}
  - {kind: weekly,  weekday: THU, start: "2019-02-11", end: "2025-08-31", source: UNVERIFIED}
  - {kind: weekly,  weekday: TUE, start: "2025-09-01", end: null,         source: UNVERIFIED}
  - {kind: monthly, weekday: TUE, start: "2025-09-01", end: null,         source: UNVERIFIED}
BANKNIFTY:
  - {kind: weekly,  weekday: THU, start: "2016-05-27", end: "2023-09-03", source: UNVERIFIED}
  - {kind: weekly,  weekday: WED, start: "2023-09-04", end: "2024-11-19", source: UNVERIFIED}
  - {kind: monthly, weekday: THU, start: "2015-01-01", end: "2023-08-31", source: UNVERIFIED}
  - {kind: monthly, weekday: WED, start: "2023-09-01", end: "2024-12-31", source: UNVERIFIED}
  - {kind: monthly, weekday: THU, start: "2025-01-01", end: "2025-08-31", source: UNVERIFIED}
  - {kind: monthly, weekday: TUE, start: "2025-09-01", end: null,         source: UNVERIFIED}
STOCK:
  - {kind: monthly, weekday: THU, start: "2015-01-01", end: "2025-08-31", source: UNVERIFIED}
  - {kind: monthly, weekday: TUE, start: "2025-09-01", end: null,         source: UNVERIFIED}
```

- [ ] **Step 2: Write the failing test**

```python
import pandas as pd
import pytest
import yaml

from volgate.data.expiry import ExpiryRule, expiry_dates, expiry_features, load_rules, unverified


def _rule(kind, wd, start, end=None):
    return ExpiryRule(kind, wd, pd.Timestamp(start), pd.Timestamp(end) if end else None, "test")


def test_weekly_thursday_moves_to_previous_day_on_holiday():
    days = pd.bdate_range("2024-01-01", "2024-01-31").drop(pd.Timestamp("2024-01-25"))
    exp = expiry_dates([_rule("weekly", 3, "2024-01-01", "2024-01-31")], days)
    assert list(exp.strftime("%Y-%m-%d")) == ["2024-01-04", "2024-01-11", "2024-01-18", "2024-01-24"]


def test_monthly_last_tuesday():
    days = pd.bdate_range("2025-09-01", "2025-10-31")
    exp = expiry_dates([_rule("monthly", 1, "2025-09-01")], days)
    assert "2025-09-30" in exp.strftime("%Y-%m-%d")
    assert "2025-10-28" in exp.strftime("%Y-%m-%d")


def test_features_days_to_expiry():
    days = pd.bdate_range("2024-01-01", "2024-01-12")
    exp = expiry_dates([_rule("weekly", 3, "2024-01-01")], days)
    f = expiry_features(days, exp, "own")
    assert f.loc["2024-01-04", "own_is_expiry"] == 1
    assert f.loc["2024-01-04", "own_days_to_expiry"] == 0
    assert f.loc["2024-01-03", "own_days_to_expiry"] == 1
    assert f.loc["2024-01-05", "own_days_to_expiry"] == 4  # Fri -> next Thu
    assert f.loc["2024-01-12", "own_days_to_expiry"] == 4  # counts business days after sample end
    assert f["own_days_to_expiry"].notna().all()  # beyond-sample expiry still found


def test_load_rules_and_unverified(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump({"X": [
        {"kind": "weekly", "weekday": "THU", "start": "2024-01-01", "end": None, "source": "UNVERIFIED"},
        {"kind": "monthly", "weekday": "TUE", "start": "2024-01-01", "end": None, "source": "https://x"},
    ]}))
    rules = load_rules(p)
    assert rules["X"][0].weekday == 3
    assert len(unverified(rules)) == 1


def test_load_rules_rejects_bad_weekday(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump({"X": [
        {"kind": "weekly", "weekday": "SAT", "start": "2024-01-01", "end": None, "source": "s"}]}))
    with pytest.raises(ValueError, match="weekday"):
        load_rules(p)


def test_repo_rules_file_loads():
    from volgate.config import REPO_ROOT
    rules = load_rules(REPO_ROOT / "configs" / "expiry_rules.yaml")
    assert set(rules) == {"NIFTY", "BANKNIFTY", "STOCK"}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_expiry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'volgate.data.expiry'`

- [ ] **Step 4: Implement `src/volgate/data/expiry.py`**

```python
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

WEEKDAYS = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4}
_FREQ = {v: f"W-{k}" for k, v in WEEKDAYS.items()}


@dataclass(frozen=True)
class ExpiryRule:
    kind: str
    weekday: int
    start: pd.Timestamp
    end: pd.Timestamp | None
    source: str


def load_rules(path: Path) -> dict[str, list[ExpiryRule]]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    out = {}
    for contract, items in raw.items():
        rules = []
        for it in items:
            if it["weekday"] not in WEEKDAYS:
                raise ValueError(f"bad weekday {it['weekday']!r} in {contract}")
            if it["kind"] not in ("weekly", "monthly"):
                raise ValueError(f"bad kind {it['kind']!r} in {contract}")
            if not it.get("source"):
                raise ValueError(f"missing source in {contract}")
            rules.append(ExpiryRule(
                it["kind"], WEEKDAYS[it["weekday"]], pd.Timestamp(it["start"]),
                pd.Timestamp(it["end"]) if it["end"] else None, it["source"]))
        out[contract] = rules
    return out


def unverified(rules: dict[str, list[ExpiryRule]]) -> list[str]:
    return [f"{c}: {r.kind} {r.start.date()}..{r.end.date() if r.end else 'open'}"
            for c, rs in rules.items() for r in rs if r.source == "UNVERIFIED"]


def _candidates(rule: ExpiryRule, lo: pd.Timestamp, hi: pd.Timestamp) -> list[pd.Timestamp]:
    if rule.kind == "weekly":
        return list(pd.date_range(lo, hi, freq=_FREQ[rule.weekday]))
    out = []
    for period in pd.period_range(lo, hi, freq="M"):
        month_end = period.end_time.normalize()
        d = month_end - pd.Timedelta(days=(month_end.weekday() - rule.weekday) % 7)
        if lo <= d <= hi:
            out.append(d)
    return out


def expiry_dates(rules: list[ExpiryRule], trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Scheduled expiries; a non-trading expiry moves to the previous trading day.

    Days after the last trading day are approximated by business days so that
    days-to-expiry is defined at the end of the sample.
    """
    future = pd.bdate_range(trading_days[-1] + pd.Timedelta(days=1), periods=40)
    cal = trading_days.union(future)
    out = set()
    for rule in rules:
        lo = max(rule.start, cal[0])
        hi = min(rule.end if rule.end is not None else cal[-1], cal[-1])
        for d in _candidates(rule, lo, hi):
            pos = cal.searchsorted(d, side="right") - 1
            if pos >= 0:
                out.add(cal[pos])
    return pd.DatetimeIndex(sorted(out))


def expiry_features(dates: pd.DatetimeIndex, expiries: pd.DatetimeIndex, prefix: str) -> pd.DataFrame:
    future = pd.bdate_range(dates[-1] + pd.Timedelta(days=1), periods=40)
    cal = dates.union(future).union(expiries)
    pos_dates = cal.get_indexer(dates)
    exp_pos = np.sort(cal.get_indexer(expiries))
    nxt = np.searchsorted(exp_pos, pos_dates, side="left")
    has_next = nxt < len(exp_pos)
    dte = np.where(has_next, exp_pos[np.minimum(nxt, len(exp_pos) - 1)] - pos_dates, np.nan)
    return pd.DataFrame({
        f"{prefix}_is_expiry": dates.isin(expiries).astype(int),
        f"{prefix}_days_to_expiry": dte,
    }, index=dates)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_expiry.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add src/volgate/data/expiry.py configs/expiry_rules.yaml tests/test_expiry.py
git commit -m "feat: add rule-based NSE expiry calendar and features"
```

---

### Task 7: Model interface and non-GARCH base models (EWMA, HAR, VIX)

**Files:**
- Create: `src/volgate/models/__init__.py` (empty), `src/volgate/models/base.py`, `src/volgate/models/simple.py`, `tests/conftest.py`
- Test: `tests/test_simple_models.py`

**Interfaces:**
- Consumes: `empirical_var_es`, `student_t_var_es` (Task 3); `gk_variance` (Task 5); panel frame columns `log_return, gk_var, vix` (Task 5).
- Produces:
  - `FitResult` dataclass: `params: dict`, `z: np.ndarray | None`, `nu: float | None`, `converged: bool`
  - `tail_var_es(fit: FitResult, sigma: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]` — Student-t if `fit.nu` is set, else empirical on `fit.z`
  - Model protocol: attribute `name: str`; `fit(train: pd.DataFrame) -> FitResult`; `filter(fit: FitResult, data: pd.DataFrame) -> np.ndarray` where element t uses only rows before t
  - Classes `EWMA(lam=0.94)` name `"ewma"`, `HAR()` name `"har"`, `VIXModel()` name `"vix"`
  - Test helper `tests/conftest.py::make_panel(n=600, seed=0) -> pd.DataFrame` and fixture `panel`

- [ ] **Step 1: Write `tests/conftest.py`**

```python
import numpy as np
import pandas as pd
import pytest

from volgate.data.panel import gk_variance


def make_panel(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Synthetic GARCH(1,1)-t returns with OHLC, GK variance and a VIX-like series."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n, name="date")
    omega, a, b = 2e-6, 0.08, 0.90
    z = rng.standard_t(6, n) / np.sqrt(6 / 4)
    s2 = np.empty(n)
    r = np.empty(n)
    s2[0] = omega / (1 - a - b)
    for t in range(n):
        if t > 0:
            s2[t] = omega + a * r[t - 1] ** 2 + b * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * z[t]
    close = 100 * np.exp(np.cumsum(r))
    open_ = close * np.exp(rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.005, n)))
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                       "adj_close": close, "volume": 1000, "log_return": r}, index=dates)
    df["gk_var"] = gk_variance(df)
    df["vix"] = 100 * np.sqrt(252 * s2) * np.exp(rng.normal(0, 0.05, n))
    return df


@pytest.fixture
def panel() -> pd.DataFrame:
    return make_panel()
```

- [ ] **Step 2: Write the failing test `tests/test_simple_models.py`**

```python
import numpy as np
import pytest

from volgate.models.base import FitResult, tail_var_es
from volgate.models.simple import EWMA, HAR, VIXModel


@pytest.mark.parametrize("model", [EWMA(), HAR(), VIXModel()])
def test_fit_and_filter_shapes(panel, model):
    fit = model.fit(panel.iloc[:400])
    s2 = model.filter(fit, panel)
    assert s2.shape == (len(panel),)
    assert np.all(s2[100:] > 0)
    v, e = tail_var_es(fit, np.sqrt(s2[400:]), 0.025)
    assert np.all(e <= v) and np.all(v < 0)


@pytest.mark.parametrize("model", [EWMA(), HAR(), VIXModel()])
def test_filter_row_t_ignores_row_t(panel, model):
    fit = model.fit(panel.iloc[:400])
    base = model.filter(fit, panel)
    pert = panel.copy()
    pert.iloc[450, pert.columns.get_indexer(["log_return", "gk_var", "vix"])] *= 10
    out = model.filter(fit, pert)
    np.testing.assert_array_equal(base[:451], out[:451])
    assert base[451] != out[451]


def test_ewma_recursion_value(panel):
    m = EWMA(lam=0.94)
    s2 = m.filter(m.fit(panel.iloc[:400]), panel)
    r = panel["log_return"].to_numpy()
    assert s2[10] == pytest.approx(0.94 * s2[9] + 0.06 * r[9] ** 2)


def test_har_finite_with_zero_range_day(panel):
    p = panel.copy()
    p.iloc[300, p.columns.get_loc("gk_var")] = 1e-10  # floored zero-range day
    m = HAR()
    s2 = m.filter(m.fit(p.iloc[:400]), p)
    assert np.all(np.isfinite(s2[30:]))


def test_vix_model_nan_vix_gives_nan_not_error(panel):
    p = panel.copy()
    p.iloc[500, p.columns.get_loc("vix")] = np.nan
    m = VIXModel()
    s2 = m.filter(m.fit(p.iloc[:400]), p)
    assert np.isnan(s2[501]) and np.isfinite(s2[502])


def test_tail_uses_t_when_nu_set():
    fit = FitResult(params={}, z=None, nu=1e7, converged=True)
    v, _ = tail_var_es(fit, np.array([1.0]), 0.025)
    assert v[0] == pytest.approx(-1.959964, abs=1e-4)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_simple_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'volgate.models'`

- [ ] **Step 4: Implement `src/volgate/models/base.py`**

```python
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from volgate.risk.dist import empirical_var_es, student_t_var_es


@dataclass
class FitResult:
    params: dict
    z: np.ndarray | None
    nu: float | None
    converged: bool


class VolModel(Protocol):
    name: str

    def fit(self, train: pd.DataFrame) -> FitResult: ...

    def filter(self, fit: FitResult, data: pd.DataFrame) -> np.ndarray: ...


def tail_var_es(fit: FitResult, sigma: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    if fit.nu is not None:
        return student_t_var_es(sigma, fit.nu, alpha)
    return empirical_var_es(fit.z, sigma, alpha)
```

- [ ] **Step 5: Implement `src/volgate/models/simple.py`**

```python
import numpy as np
import pandas as pd

from volgate.models.base import FitResult

_FLOOR = 1e-10


def _std_resid(r: np.ndarray, s2: np.ndarray, burn: int) -> np.ndarray:
    z = r[burn:] / np.sqrt(s2[burn:])
    return z[np.isfinite(z)]


class EWMA:
    """RiskMetrics EWMA; tails by filtered historical simulation."""

    name = "ewma"

    def __init__(self, lam: float = 0.94, init_var: float = 1e-4, burn: int = 22):
        self.lam, self.init_var, self.burn = lam, init_var, burn

    def _recursion(self, r: np.ndarray) -> np.ndarray:
        s2 = np.empty(len(r))
        s2[0] = self.init_var  # fixed prior: uses no data
        for t in range(1, len(r)):
            s2[t] = self.lam * s2[t - 1] + (1 - self.lam) * r[t - 1] ** 2
        return s2

    def fit(self, train: pd.DataFrame) -> FitResult:
        r = train["log_return"].to_numpy()
        z = _std_resid(r, self._recursion(r), self.burn)
        return FitResult(params={"lam": self.lam}, z=z, nu=None, converged=True)

    def filter(self, fit: FitResult, data: pd.DataFrame) -> np.ndarray:
        return self._recursion(data["log_return"].to_numpy())


class HAR:
    """HAR on log Garman-Klass variance; tails from empirical standardized returns."""

    name = "har"

    @staticmethod
    def _design(gk: pd.Series) -> np.ndarray:
        g = gk.clip(lower=_FLOOR)
        return np.column_stack([
            np.ones(len(g)),
            np.log(g).shift(1),
            np.log(g.rolling(5).mean()).shift(1),
            np.log(g.rolling(22).mean()).shift(1),
        ])

    def fit(self, train: pd.DataFrame) -> FitResult:
        X = self._design(train["gk_var"])
        y = np.log(train["gk_var"].clip(lower=_FLOOR)).to_numpy()
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
        beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        s2e = float(np.var(y[ok] - X[ok] @ beta, ddof=X.shape[1]))
        fit = FitResult(params={"beta": beta, "s2e": s2e}, z=None, nu=None, converged=True)
        s2 = self.filter(fit, train)
        fit.z = _std_resid(train["log_return"].to_numpy(), s2, 0)
        return fit

    def filter(self, fit: FitResult, data: pd.DataFrame) -> np.ndarray:
        X = self._design(data["gk_var"])
        return np.exp(X @ fit.params["beta"] + 0.5 * fit.params["s2e"])


class VIXModel:
    """sigma2_t = beta * (VIX_{t-1}/100)^2 / 252, beta by OLS through the origin on r^2."""

    name = "vix"

    @staticmethod
    def _implied(data: pd.DataFrame) -> np.ndarray:
        return ((data["vix"].shift(1) / 100.0) ** 2 / 252.0).to_numpy()

    def fit(self, train: pd.DataFrame) -> FitResult:
        x = self._implied(train)
        r = train["log_return"].to_numpy()
        ok = np.isfinite(x) & np.isfinite(r)
        beta = max(float(np.sum(x[ok] * r[ok] ** 2) / np.sum(x[ok] ** 2)), 1e-6)
        z = r[ok] / np.sqrt(beta * x[ok])
        return FitResult(params={"beta": beta}, z=z, nu=None, converged=True)

    def filter(self, fit: FitResult, data: pd.DataFrame) -> np.ndarray:
        return fit.params["beta"] * self._implied(data)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_simple_models.py -v`
Expected: 10 passed

- [ ] **Step 7: Commit**

```bash
git add src/volgate/models tests/conftest.py tests/test_simple_models.py
git commit -m "feat: add model interface and EWMA, HAR, VIX base models"
```

---

### Task 8: GARCH-family models with Student-t errors

**Files:**
- Create: `src/volgate/models/garch.py`
- Test: `tests/test_garch.py`

**Interfaces:**
- Consumes: `FitResult` (Task 7).
- Produces: `GarchT(name: str)` with `name` in `{"garch_t", "egarch_t", "gjr_t"}`; `fit` returns `FitResult(params=<dict of arch params>, z=None, nu=params["nu"], converged=<bool>)`; `filter` returns decimal-scale variance where element t uses rows before t.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest

from volgate.models.garch import GarchT


@pytest.mark.parametrize("name", ["garch_t", "egarch_t", "gjr_t"])
def test_fit_filter_and_causality(panel, name):
    m = GarchT(name)
    fit = m.fit(panel.iloc[:400])
    assert fit.converged and fit.nu > 2
    base = m.filter(fit, panel)
    assert base.shape == (len(panel),) and np.all(base > 0)
    pert = panel.copy()
    pert.iloc[450, pert.columns.get_loc("log_return")] *= 10
    out = m.filter(fit, pert)
    np.testing.assert_array_equal(base[:451], out[:451])
    assert base[451] != out[451]


def test_garch_recovers_simulated_persistence(panel):
    fit = GarchT("garch_t").fit(panel)
    p = fit.params
    assert 0.8 < p["alpha[1]"] + p["beta[1]"] < 1.0


def test_unknown_name_raises():
    with pytest.raises(KeyError):
        GarchT("figarch")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_garch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'volgate.models.garch'`

- [ ] **Step 3: Implement `src/volgate/models/garch.py`**

```python
import numpy as np
import pandas as pd
from arch import arch_model

from volgate.models.base import FitResult

_SPECS = {
    "garch_t": {"vol": "GARCH", "p": 1, "o": 0, "q": 1},
    "gjr_t": {"vol": "GARCH", "p": 1, "o": 1, "q": 1},
    "egarch_t": {"vol": "EGARCH", "p": 1, "o": 1, "q": 1},
}


class GarchT:
    """GARCH-family model with Student-t errors, fitted on returns x 100."""

    def __init__(self, name: str):
        self.spec = _SPECS[name]
        self.name = name

    def _model(self, r: np.ndarray):
        return arch_model(r * 100.0, mean="Zero", dist="t", rescale=False, **self.spec)

    def fit(self, train: pd.DataFrame) -> FitResult:
        res = self._model(train["log_return"].to_numpy()).fit(disp="off", show_warning=False)
        params = res.params.to_dict()
        converged = res.convergence_flag == 0 and params["nu"] > 2.0
        return FitResult(params=params, z=None, nu=float(params["nu"]), converged=bool(converged))

    def filter(self, fit: FitResult, data: pd.DataFrame) -> np.ndarray:
        fixed = self._model(data["log_return"].to_numpy()).fix(np.array(list(fit.params.values())))
        return (np.asarray(fixed.conditional_volatility) / 100.0) ** 2
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_garch.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/volgate/models/garch.py tests/test_garch.py
git commit -m "feat: add GARCH, EGARCH, GJR-GARCH with Student-t errors"
```

---

### Task 9: Walk-forward engine with fallback and leakage tests

**Files:**
- Create: `src/volgate/models/walkforward.py`
- Test: `tests/test_walkforward.py`

**Interfaces:**
- Consumes: `VolModel`, `FitResult`, `tail_var_es` (Task 7); `GarchT` (Task 8); `alpha_tag` (Task 3).
- Produces: `walk_forward(data: pd.DataFrame, model, oos_start: str, refit_every: int, alphas: list[float]) -> pd.DataFrame` indexed by OOS dates with columns `sigma2, refit_date, fallback, var_{tag}, es_{tag}` for each alpha.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pandas as pd
import pytest

from volgate.models.base import FitResult
from volgate.models.garch import GarchT
from volgate.models.simple import EWMA, HAR, VIXModel
from volgate.models.walkforward import walk_forward

ALPHAS = [0.01, 0.025]


class Flaky:
    name = "flaky"

    def __init__(self, fail_on=(), unconverged_on=()):
        self.calls = 0
        self.fail_on, self.unconverged_on = set(fail_on), set(unconverged_on)

    def fit(self, train):
        self.calls += 1
        if self.calls in self.fail_on:
            raise RuntimeError("boom")
        return FitResult(params={"n": len(train)}, z=np.linspace(-3, 3, 401), nu=None,
                         converged=self.calls not in self.unconverged_on)

    def filter(self, fit, data):
        return np.full(len(data), 1e-4 * fit.params["n"] / 400)


def test_blocks_columns_and_refit_dates(panel):
    oos = panel.index[400].strftime("%Y-%m-%d")
    out = walk_forward(panel, Flaky(), oos, 21, ALPHAS)
    assert out.index[0] == panel.index[400] and out.index[-1] == panel.index[-1]
    assert {"sigma2", "refit_date", "fallback", "var_a010", "es_a025"} <= set(out.columns)
    assert out["refit_date"].nunique() == int(np.ceil(200 / 21))
    assert (out["es_a025"] <= out["var_a025"]).all()


@pytest.mark.parametrize("kw", [{"fail_on": [2]}, {"unconverged_on": [2]}])
def test_failed_or_unconverged_fit_falls_back(panel, kw):
    oos = panel.index[400].strftime("%Y-%m-%d")
    out = walk_forward(panel, Flaky(**kw), oos, 21, ALPHAS)
    blocks = out.groupby("refit_date")
    flags = blocks["fallback"].first().to_numpy()
    assert flags.tolist() == [False, True] + [False] * (len(flags) - 2)
    s = blocks["sigma2"].first().to_numpy()
    assert s[1] == s[0]  # block 2 reuses block 1 params


def test_first_fit_failure_raises(panel):
    with pytest.raises(RuntimeError, match="first fit"):
        walk_forward(panel, Flaky(fail_on=[1]), panel.index[400].strftime("%Y-%m-%d"), 21, ALPHAS)


@pytest.mark.parametrize("model", [EWMA(), HAR(), VIXModel(), GarchT("garch_t"),
                                   GarchT("egarch_t"), GarchT("gjr_t")])
def test_no_lookahead(panel, model):
    oos = panel.index[400].strftime("%Y-%m-%d")
    base = walk_forward(panel, model, oos, 21, ALPHAS)
    j = 450
    pert = panel.copy()
    cols = pert.columns.get_indexer(["log_return", "gk_var", "vix", "open", "high", "low", "close"])
    pert.iloc[j, cols] *= 3
    out = walk_forward(pert, model, oos, 21, ALPHAS)
    dj = panel.index[j]
    pd.testing.assert_frame_equal(base.loc[:dj], out.loc[:dj])
    assert not base.loc[panel.index[j + 1]].equals(out.loc[panel.index[j + 1]])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_walkforward.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'volgate.models.walkforward'`

- [ ] **Step 3: Implement `src/volgate/models/walkforward.py`**

```python
import numpy as np
import pandas as pd

from volgate.models.base import tail_var_es
from volgate.risk.dist import alpha_tag


def walk_forward(data: pd.DataFrame, model, oos_start: str, refit_every: int,
                 alphas: list[float]) -> pd.DataFrame:
    """Expanding-window refit every `refit_every` rows from `oos_start`.

    A block starting at row k is fitted on rows < k. Forecast for row t uses
    rows < t through the model's filter. A failed or unconverged fit reuses
    the previous fit and sets `fallback`.
    """
    data = data.sort_index()
    oos = np.flatnonzero(data.index >= pd.Timestamp(oos_start))
    if oos.size == 0:
        raise ValueError(f"no rows on or after {oos_start}")
    blocks, prev = [], None
    for k in range(oos[0], len(data), refit_every):
        end = min(k + refit_every, len(data))
        fallback = False
        try:
            fit = model.fit(data.iloc[:k])
            if not fit.converged:
                raise RuntimeError("fit did not converge")
        except Exception as exc:
            if prev is None:
                raise RuntimeError(f"{model.name}: first fit failed") from exc
            fit, fallback = prev, True
        prev = fit
        s2 = model.filter(fit, data.iloc[:end])[k:end]
        block = pd.DataFrame({"sigma2": s2}, index=data.index[k:end])
        block["refit_date"] = data.index[k]
        block["fallback"] = fallback
        sigma = np.sqrt(s2)
        for a in alphas:
            v, e = tail_var_es(fit, sigma, a)
            block[f"var_{alpha_tag(a)}"] = v
            block[f"es_{alpha_tag(a)}"] = e
        blocks.append(block)
    return pd.concat(blocks)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_walkforward.py -v`
Expected: 10 passed

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add src/volgate/models/walkforward.py tests/test_walkforward.py
git commit -m "feat: add walk-forward engine with fallback and leakage tests"
```

---

### Task 10: Pipeline scripts, real-data run, audit, rule verification

**Files:**
- Create: `scripts/02_prepare.py`, `scripts/03_base_forecasts.py`
- Modify: `data/audit/corrections.csv`, `configs/expiry_rules.yaml`, `README.md`
- Generated and committed: `data/raw/*`, `results/tables/outlier_audit.csv`, `results/tables/data_summary.csv`, `results/tables/base_model_summary.csv`

**Interfaces:**
- Consumes: everything from Tasks 1–9.
- Produces (for Plan 2):
  - `data/processed/panel/{asset}.csv`: index `date`; columns `open, high, low, close, adj_close, volume, log_return, gk_var, vix, own_is_expiry, own_days_to_expiry, nifty_is_expiry, nifty_days_to_expiry`
  - `data/processed/base_forecasts/{asset}.csv`: index `date`; for each model `m` in `ewma, garch_t, egarch_t, gjr_t, har, vix`: columns `{m}__sigma2, {m}__fallback, {m}__var_{tag}, {m}__es_{tag}`

- [ ] **Step 1: Write `scripts/02_prepare.py`**

```python
import warnings

import pandas as pd

from volgate.config import REPO_ROOT, load_config
from volgate.data.download import verify_manifest
from volgate.data.expiry import expiry_dates, expiry_features, load_rules, unverified
from volgate.data.panel import apply_corrections, build_asset_frame, load_raw, outlier_table

cfg = load_config()
P = {k: REPO_ROOT / v for k, v in cfg["paths"].items()}
bad = verify_manifest(P["raw"])
if bad:
    raise SystemExit(f"raw data changed since download: {bad}")

vix = load_raw(P["raw"] / "india_vix.csv")["close"]
frames, stats = {}, []
for asset in cfg["assets"]:
    frames[asset], s = build_asset_frame(load_raw(P["raw"] / f"{asset}.csv"), vix)
    stats.append({"asset": asset, **s})

P["tables"].mkdir(parents=True, exist_ok=True)
corrections = pd.read_csv(P["audit"] / "corrections.csv", dtype=str)
audit = outlier_table(frames, cfg["outlier_threshold"]).merge(
    corrections, on=["asset", "date"], how="left")
audit["action"] = audit["action"].fillna("unreviewed")
audit.to_csv(P["tables"] / "outlier_audit.csv", index=False)
n_unreviewed = int((audit["action"] == "unreviewed").sum())
if n_unreviewed:
    warnings.warn(f"{n_unreviewed} outliers not reviewed; see results/tables/outlier_audit.csv")
frames = apply_corrections(frames, corrections)

rules = load_rules(P["expiry_rules"])
for item in unverified(rules):
    warnings.warn(f"unverified expiry rule: {item}")

out_dir = P["processed"] / "panel"
out_dir.mkdir(parents=True, exist_ok=True)
nifty_dates = frames["nifty50"].index
nifty_exp = expiry_dates(rules["NIFTY"], nifty_dates)
for asset, df in frames.items():
    own_exp = expiry_dates(rules[cfg["contracts"][asset]], df.index)
    df = df.join(expiry_features(df.index, own_exp, "own"))
    df = df.join(expiry_features(df.index, nifty_exp, "nifty"))
    df.to_csv(out_dir / f"{asset}.csv", date_format="%Y-%m-%d")

summary = pd.DataFrame(stats)
summary["rows_after_corrections"] = [len(frames[a]) for a in summary["asset"]]
summary.to_csv(P["tables"] / "data_summary.csv", index=False)
print(summary.to_string(index=False))
```

- [ ] **Step 2: Write `scripts/03_base_forecasts.py`**

```python
import numpy as np
import pandas as pd

from volgate.config import REPO_ROOT, load_config
from volgate.models.garch import GarchT
from volgate.models.simple import EWMA, HAR, VIXModel
from volgate.models.walkforward import walk_forward
from volgate.risk.dist import alpha_tag
from volgate.risk.fz import fz0_loss

cfg = load_config()
P = {k: REPO_ROOT / v for k, v in cfg["paths"].items()}
out_dir = P["processed"] / "base_forecasts"
out_dir.mkdir(parents=True, exist_ok=True)
models = [EWMA(), GarchT("garch_t"), GarchT("egarch_t"), GarchT("gjr_t"), HAR(), VIXModel()]

rows = []
for asset in cfg["assets"]:
    panel = pd.read_csv(P["processed"] / "panel" / f"{asset}.csv", index_col="date", parse_dates=True)
    parts = []
    for m in models:
        wf = walk_forward(panel, m, cfg["oos_start"], cfg["refit_every"], cfg["alphas"])
        y = panel["log_return"].reindex(wf.index).to_numpy()
        for a in cfg["alphas"]:
            t = alpha_tag(a)
            ok = np.isfinite(wf[f"var_{t}"]) & np.isfinite(wf[f"es_{t}"])
            rows.append({
                "asset": asset, "model": m.name, "alpha": a, "n": int(ok.sum()),
                "hit_rate": float(np.mean(y[ok] <= wf[f"var_{t}"][ok])),
                "mean_fz0": float(fz0_loss(y[ok], wf[f"var_{t}"][ok], wf[f"es_{t}"][ok], a).mean()),
                "fallback_blocks": int(wf.groupby("refit_date")["fallback"].first().sum()),
            })
        parts.append(wf.drop(columns="refit_date").add_prefix(f"{m.name}__"))
        print(f"{asset} {m.name} done")
    pd.concat(parts, axis=1).to_csv(out_dir / f"{asset}.csv", date_format="%Y-%m-%d")

summary = pd.DataFrame(rows)
summary.to_csv(P["tables"] / "base_model_summary.csv", index=False)
print(summary[summary["alpha"] == 0.025].to_string(index=False))
```

- [ ] **Step 3: Download real data**

Run: `make download`
Expected: a table with 9 keys; `last` is the last trading day on or before 2026-09-30 for every key. Commit `data/raw/` in Step 9.

- [ ] **Step 4: First prepare run and outlier review**

Run: `make prepare`
Expected: data summary printed; warnings for unreviewed outliers and unverified expiry rules.

For every row in `results/tables/outlier_audit.csv`:
- Check corporate actions with `uv run python -c "import yfinance as yf; print(yf.Ticker('ADANIENT.NS').actions)"` (or the `yfinance` MCP tool `get_stock_actions`), and search news for that date.
- A large gap between `log_return` and `close_return` signals an adjustment by Yahoo; a jump present in both with no market news signals an unadjusted corporate action.
- Append one row per outlier to `data/audit/corrections.csv`: `asset,date,keep|drop,<one-line evidence with source URL>`. Adani Enterprises 2015-06-03 is expected to be `drop` (demerger) if the evidence confirms it.

- [ ] **Step 5: Verify expiry rules**

For each rule in `configs/expiry_rules.yaml`, find the NSE circular or SEBI circular that sets its weekday and dates (web search: "NSE circular Bank Nifty weekly expiry Wednesday 2023", "SEBI circular one weekly expiry per exchange 2024", "NSE expiry Tuesday September 2025 circular", "NSE monthly expiry Bank Nifty last Wednesday 2024"). Correct `start`, `end`, and `weekday` where the circular differs, and replace `UNVERIFIED` with the circular URL.

- [ ] **Step 6: Re-run prepare**

Run: `make prepare`
Expected: no warnings. `results/tables/outlier_audit.csv` has no `unreviewed` rows.

- [ ] **Step 7: Run base forecasts**

Run: `make base`
Expected: 48 `done` lines, then a table for α = 0.025. Sanity checks on `results/tables/base_model_summary.csv`:
- `n` ≥ 1600 for every row except `vix` (which may be lower only by `vix_missing` days).
- `hit_rate` between 0.005 and 0.06 for α = 0.025 on every row. Investigate any row outside this range before continuing (likely data or scaling error), and record findings in the commit message.
- `fallback_blocks` is reported; more than 5 for any GARCH row needs investigation.

- [ ] **Step 8: Update README status table**

In `README.md`, change these rows of the "Implementation status" table:

```markdown
| Data download, audit, expiry calendar | Done (Plan 1) |
| Base models and VaR/ES | Done (Plan 1) |
| FZ0 loss | Done (Plan 1) |
```

- [ ] **Step 9: Run full test suite and commit**

Run: `uv run pytest -q`
Expected: all pass

```bash
git add scripts data/raw data/audit configs/expiry_rules.yaml results/tables README.md
git commit -m "feat: run data pipeline and walk-forward base forecasts on NSE data"
```
