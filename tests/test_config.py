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
