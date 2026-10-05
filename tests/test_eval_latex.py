import numpy as np
import pandas as pd

from volgate.evaluate.latex import to_latex


def test_latex_renders_nan_and_structure():
    df = pd.DataFrame({"a": [1.23456, np.nan], "b": ["x_y", "z"]}, index=["m1", "m2"])
    s = to_latex(df, "Cap", "tab:x", digits=2, note="Note text.")
    assert "\\toprule" in s and "\\bottomrule" in s and "\\label{tab:x}" in s
    assert "1.23" in s and "--" in s and "x\\_y" in s and "Note text." in s
