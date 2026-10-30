"""Write the paper's LaTeX tables from CSVs under data/processed/.

    python paper/tables.py

Each function reads one CSV and writes one ``tables/<name>.tex``. The parameter
table reads ``data/processed/horizon/parameters.csv``, which
``scripts/analyze_extras.py`` writes from ``experiments/base.yaml`` and the job
list, so its values are the configuration's and not typed here.
"""

from __future__ import annotations

import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
H = ROOT / "data" / "processed" / "horizon"
RIGOR = ROOT / "data" / "processed" / "rigor"
OUT = pathlib.Path(__file__).resolve().parent / "tables"

CELL_NAMES = {
    "shipped": "shipped", "taste_sigma_0.2": r"$\sigma_\tau = 0.2$",
    "taste_sigma_1.0": r"$\sigma_\tau = 1.0$", "shock_scale_0.5": r"$\sigma_L = 0.5$",
    "shock_scale_3.0": r"$\sigma_L = 3.0$", "beta_alpha_1.0": r"$\beta/\alpha = 1$",
    "beta_alpha_3.0": r"$\beta/\alpha = 3$", "delta_0.0": r"$\delta = 0$",
    "delta_1.0": r"$\delta = 1$",
}


def write(name: str, lines: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote paper/tables/{name}")


def tex(text: str) -> str:
    return str(text).replace("_", r"\_").replace("%", r"\%")


def yes(flag: bool) -> str:
    return "yes" if bool(flag) else r"\textbf{no}"


def table_params() -> None:
    p = pd.read_csv(H / "parameters.csv")
    p = p.fillna("")  # an empty symbol reads back from the CSV as NaN
    lines = [r"\small", r"\begin{tabular}{llll}", r"\hline",
             r"Symbol & Meaning & Shipped & Swept \\", r"\hline"]
    group = None
    for _, r in p.iterrows():
        if r["group"] != group:
            if group is not None:
                lines.append(r"\hline")
            group = r["group"]
        lines.append(f"{r['symbol']} & {r['meaning']} & {r['shipped']} & {r['swept']} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    write("params.tex", lines)


def table_experiments() -> None:
    idx = pd.read_csv(H / "run_index.csv")
    # Wall time is in run_index.csv but not printed: one batch ran while the
    # machine was suspended, so per-job wall time there is not compute time.
    lines = [r"\small", r"\begin{tabular}{llrrl}", r"\hline",
             r"Experiment & Question & Seeds & Rounds & Commit \\", r"\hline"]
    for _, r in idx.iterrows():
        q = tex(r["question"]) + (r" (post hoc)" if r["post_hoc"] else "")
        lines.append(f"{r['experiment']} & {q} & {r['n_seeds']} & {r['rounds']} & "
                     f"\\texttt{{{r['commits']}}} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    write("experiments.tex", lines)


def table_predictions() -> None:
    p = pd.read_csv(H / "predictions.csv")
    lines = [r"\footnotesize", r"\begin{tabular}{lp{6.2cm}p{4.4cm}l}", r"\hline",
             r"Experiment & Registered prediction & Measured & Held \\", r"\hline"]
    for _, r in p.iterrows():
        lines.append(f"{r['experiment']} & {tex(r['prediction'])} & {tex(r['measured'])} & "
                     f"{yes(r['held'])} \\\\")
    lines += [r"\hline", rf"\multicolumn{{3}}{{l}}{{Held}} & {int(p['held'].sum())} of {len(p)} \\",
              r"\hline", r"\end{tabular}"]
    write("predictions.tex", lines)


def censored_median(entries: pd.Series, censored: pd.Series) -> float:
    v = sorted(float("inf") if c else float(e) for e, c in zip(entries, censored))
    k = len(v)
    mid = (v[(k - 1) // 2] + v[k // 2]) / 2
    if mid == float("inf"):
        raise ValueError("median censored")
    return mid


def table_sweep() -> None:
    c = pd.read_csv(H / "e11_sweep" / "analysis" / "cells.csv").set_index("cell")
    per = pd.read_csv(H / "e11_sweep" / "analysis" / "per_seed.csv")
    order = ["shipped", "taste_sigma_0.2", "taste_sigma_1.0", "shock_scale_0.5",
             "shock_scale_3.0", "beta_alpha_1.0", "beta_alpha_3.0", "delta_0.0", "delta_1.0"]
    lines = [r"\small", r"\begin{tabular}{lrrrrccccc}", r"\hline",
             r"Cell & Peak gap & Entry & Censored & Tail $\times 10^3$ & C1t & C2t & C3t & C4t & C5t \\",
             r"\hline"]
    for cell in order:
        r = c.loc[cell]
        g = per[per.cell == cell]
        med = censored_median(g["entry_round"].fillna(0), g["censored"])
        lines.append(
            f"{CELL_NAMES[cell]} & {r['peak_mean']:.3f} & {med:.0f} & {int(r['n_censored'])} & "
            f"{1000 * r['tail_mean']:+.2f} & {yes(r['C1t'])} & {yes(r['C2t'])} & "
            f"{yes(r['C3t'])} & {yes(r['C4t'])} & {'yes' if r['C5t'] else 'no'} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    write("sweep.tex", lines)


def table_units() -> None:
    u = pd.read_csv(H / "units.csv")
    lines = [r"\footnotesize", r"\begin{tabular}{llrrrr}", r"\hline",
             r"Experiment & Condition & Rounds & Rows per agent & Optimizer steps & Censored \\",
             r"\hline"]
    for _, r in u.iterrows():
        rows = "not logged" if pd.isna(r["T05_rows"]) else \
            f"{int(round(r['T05_rows'])):,}".replace(",", "{,}")
        steps = f"{int(round(r['T05_steps'])):,}".replace(",", "{,}")
        cond = r["condition"]
        cond = CELL_NAMES.get(cond, tex(cond))
        lines.append(f"{r['experiment']} & {cond} & {r['T05_rounds']:.0f} & {rows} & {steps} & "
                     f"{int(r['n_censored'])} of {int(r['n_seeds'])} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    write("units.tex", lines)


def table_warm() -> None:
    w = pd.read_csv(H / "e13d_warm" / "analysis" / "summary.csv").set_index(["set", "quantity"])
    lines = [r"\small", r"\begin{tabular}{llrr}", r"\hline",
             r"Quantity & Basis & Confirmation (100--119) & Exploratory (42--61) \\", r"\hline"]
    for q, basis, label in (("gap_round0_F299", "round 0", "gap"),
                            ("level_round0_F299", "round 0", "level"),
                            ("gap_peak_F299", "own peak", "gap"),
                            ("level_peak_F299", "own peak", "level")):
        cells = []
        for st in ("confirmation", "exploratory"):
            r = w.loc[(st, q)]
            cells.append(f"{100 * r['mean']:.1f}\\% [{100 * r['lo_t95']:.1f}, "
                         f"{100 * r['hi_t95']:.1f}]")
        primary = r" (primary)" if q == "gap_round0_F299" else ""
        lines.append(f"{label}{primary} & {basis} & {cells[0]} & {cells[1]} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    write("warm.tex", lines)


def table_validation() -> None:
    rows = pd.read_csv(RIGOR / "validation.csv")
    lines = [r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lllll}", r"\hline",
             r"Check & $N$ & Measured & Evidence & Verdict \\", r"\hline"]
    for _, r in rows.iterrows():
        lines.append(f"{r['check']} & {r['n']} & {r['measured']} & {r['evidence']} & "
                     f"{r['verdict']} \\\\")
    lines += [r"\hline", r"\end{tabular}%", "}"]
    write("validation.tex", lines)


def main() -> None:
    for fn in (table_params, table_experiments, table_predictions, table_sweep, table_units,
               table_warm, table_validation):
        fn()


if __name__ == "__main__":
    main()
