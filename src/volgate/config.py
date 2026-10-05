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
