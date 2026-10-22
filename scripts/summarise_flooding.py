"""Summarise data/processed/flooding/ into the tables the H2 report quotes.

Reads coefficients.csv, cells.csv and panel.parquet written by run_flooding.py and
writes summary_coefficients.csv, summary_cells.csv and decomposition.csv beside
them. Every number is a mean +/- sd over the three seeds; nothing is pooled across
seeds before averaging.

The decomposition answers (b) of the session prompt: when the flooding slope
changes between c = 0 and c = 1, how much is the volume term's dispersion falling
(fewer proposers, so less spread in V/V̄) and how much is received intent rising?
At w > 0 the flooding increment is w * (V/V̄ - 1) * (1 - τ̄), recorded per agent, so
its mean absolute size and its two factors are read directly off the panel.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

CELL = ["top_k", "cost", "w"]


def mean_sd(df: pd.DataFrame, by: list[str], cols: list[str]) -> pd.DataFrame:
    g = df.groupby(by)[cols]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_sd"))
    return out[[c for pair in zip(out.columns[: len(cols)], out.columns[len(cols):]) for c in pair]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default="data/processed/flooding")
    args = ap.parse_args()
    d = pathlib.Path(args.dir)

    coefs = pd.read_csv(d / "coefficients.csv")
    cells = pd.read_csv(d / "cells.csv")
    panel = pd.read_parquet(d / "panel.parquet")

    coef_cols = ["b_theta", "b_theta_active", "b_vrel", "b_recv_intent", "b_inter", "n",
                 "n_dropped_zero_V", "share_active"]
    summary = mean_sd(coefs, [*CELL, "side", "window", "model"], coef_cols).reset_index()
    summary.to_csv(d / "summary_coefficients.csv", index=False)

    cell_cols = [c for c in cells.columns if c not in ("seed", *CELL)]
    mean_sd(cells, CELL, cell_cols).reset_index().to_csv(d / "summary_cells.csv", index=False)

    # Flooding term, W side, by cell and seed: its factors and its realised size,
    # over the open late epochs only. A closed epoch has V = 0 everywhere, no V̄,
    # and a flooding term of exactly zero; it is a different market, not a small
    # value of this one. Cells with no open late epoch get NaN, which is the answer.
    w = panel[(panel["side"] == "W") & (panel["w"] > 0)].copy()
    w["pool"] = 1.0 - w["recv_intent"]
    rows = []
    for key, g in w.groupby([*CELL, "seed"]):
        late = g[(g["epoch"] >= 5) & g["vrel"].notna()]
        recv = late[late["volume"] > 0]
        rows.append(dict(zip([*CELL, "seed"], key),
                         n_open_late=int(late["epoch"].nunique()),
                         sd_vrel=float(late.groupby("epoch")["vrel"].std().mean()),
                         mean_pool=float(recv["pool"].mean()),
                         mean_abs_flood=float(late["flood"].abs().mean()),
                         corr_flood_vrel=(float(np.corrcoef(late["flood"], late["vrel"])[0, 1])
                                          if len(late) > 2 else float("nan")),
                         share_flood_up=float((late["flood"] > 0).mean())))
    dec = pd.DataFrame(rows)
    mean_sd(dec, CELL, ["n_open_late", "sd_vrel", "mean_pool", "mean_abs_flood",
                        "corr_flood_vrel", "share_flood_up"]
            ).reset_index().to_csv(d / "decomposition.csv", index=False)


if __name__ == "__main__":
    main()
