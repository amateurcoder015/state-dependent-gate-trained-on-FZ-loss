import numpy as np
import pandas as pd


def _cell(v, digits):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (float, np.floating)):
        return "--" if np.isnan(v) else f"{v:.{digits}f}"
    return str(v).replace("_", "\\_")


def to_latex(df: pd.DataFrame, caption: str, label: str, digits: int = 3,
             note: str | None = None) -> str:
    """Booktabs table; NaN shows as --, underscores escaped, index is the first column."""
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
