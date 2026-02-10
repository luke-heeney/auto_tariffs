from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _resolve_cfg_path() -> Path:
    cfg_path = Path("results_config.json")
    if not cfg_path.exists():
        cfg_path = Path("post_est") / "results_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError("results_config.json not found in repo root or post_est/.")
    return cfg_path


def _resolve_owner_mapping_path(cfg_path: Path) -> Path:
    cfg = json.loads(cfg_path.read_text())
    owner_mapping_path = cfg.get("owner_mapping_path")
    if not owner_mapping_path:
        raise ValueError("owner_mapping_path not set in results_config.json.")
    p = Path(owner_mapping_path)
    if not p.is_absolute():
        p = (cfg_path.parent / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Owner mapping file not found: {p}")
    return p


def _escape_tex(s: object) -> str:
    text = str(s)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def build_table(owner_mapping_path: Path, out_path: Path) -> None:
    df = pd.read_excel(owner_mapping_path, sheet_name="brands")
    col_map = {c.strip().lower(): c for c in df.columns}
    needed = ["brand", "owner", "pricer"]
    missing = [c for c in needed if c not in col_map]
    if missing:
        raise ValueError(f"Missing required columns in brands sheet: {missing}")

    df = df[[col_map["brand"], col_map["owner"], col_map["pricer"]]].copy()
    df.columns = ["brand", "owner", "pricer"]
    df = df.dropna(subset=["brand", "owner", "pricer"])
    df = df.sort_values("brand").reset_index(drop=True)

    lines: list[str] = []
    lines.append("\\begin{table}[!htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Brand ownership and pricing groups}")
    lines.append("\\label{tab:ownership_matrix}")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{6pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.1}")
    lines.append("\\begin{adjustbox}{max width=\\textwidth}")
    lines.append("\\begin{tabular}{lll}")
    lines.append("\\toprule")
    lines.append("Brand & Owner & Pricing group (pricer) \\\\")
    lines.append("\\midrule")
    for _, row in df.iterrows():
        lines.append(
            f"{_escape_tex(row['brand'])} & {_escape_tex(row['owner'])} & {_escape_tex(row['pricer'])} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{adjustbox}")
    lines.append("\\end{table}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    cfg_path = _resolve_cfg_path()
    owner_mapping_path = _resolve_owner_mapping_path(cfg_path)

    out_dir = cfg_path.parent / "outputs"
    out_path = out_dir / "ownership_matrix.tex"
    build_table(owner_mapping_path, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
