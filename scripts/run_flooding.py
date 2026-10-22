"""H2: which signal does θ track, and does a flooding term survive the yield rule?

What this produces (data/processed/flooding/):

    panel.parquet       one row per (cell, seed, epoch, agent): θ_t, Δθ, received
                        positive swipes V, their mean proposer intent τ̄, DA
                        received proposals, matches, the flooding increment.
    coefficients.csv    per (cell, seed, side, window, model): OLS coefficients of
                        Δθ on θ_t, V/V̄ and τ̄, with standard errors.
    cells.csv           per (cell, seed): realised volume, realised intent, who
                        proposed at all, the gate-binding share, mean Δθ, sd(θ_0).

The cell grid is ncf.top_k x hypergamy.signal_cost x hypergamy.flooding_weight.
Part 1 of the H2 session is the (top_k=20, c=0, w=0) cell of the same data.

Received volume V is counted from the swipe log's label-1 rows *by receiver*,
column 1; column 0 is the agent who swiped. That is H2's "received proposal
volume". It is not what update_thresholds reads: the shipped rule reads DA's
received count, which for W comes from the NCF lists, not the log. Both are kept.

Every model but ``net`` controls for the yield step's own θ term (see ``fit``).
θ starts at a utility quantile of order 4 and the yield target is in [0, 1], so
-ηθ is most of Δθ for any agent the step touches. Epochs 0-4 and 5-19 are fitted
separately because at φ = 0.02 the gate binds for about four epochs and then
relaxes, and a pooled fit is mostly that transient.

The cost axis is {0, 0.05, 1.0}. With p carried forward after an empty log, the
market stays open at c = 0.05 (proposer intent rises from 0.51 to 0.58) and closes
for good between 0.05 and 0.07; c = 1.0 is kept to show the closure, and its
cells hold one open epoch. Cell-level volume and intent are taken over *open*
epochs only — an epoch with no proposal has V = 0 everywhere and τ̄ undefined,
and averaging it in would report a market that does not exist.

Run from the repo root:

    python scripts/run_flooding.py
"""

from __future__ import annotations

import argparse
import itertools
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.python.config import SimConfig  # noqa: E402
from src.python.orchestrator import (  # noqa: E402
    Simulation,
    admissible_prefs,
    received_proposals,
)

SEEDS = (42, 43, 44)
TOP_K = (20, 40)
COSTS = (0.0, 0.05, 1.0)
WEIGHTS = (0.0, 0.05, 0.2)
N_EPOCHS = 20
WINDOWS = {"early": (0, 4), "late": (5, 19)}


def run_cell(base: dict, seed: int, top_k: int, cost: float, weight: float) -> tuple[pd.DataFrame, dict]:
    cfg = SimConfig.coerce(base).to_dict()
    cfg["population"]["seed"] = seed
    cfg["ncf"]["top_k"] = top_k
    cfg["hypergamy"]["signal_cost"] = cost
    cfg["hypergamy"]["flooding_weight"] = weight
    sim = Simulation(cfg, recommender="ncf")
    sim.record_states = True
    sim.record_market = True
    frame = sim.run(N_EPOCHS)

    side = np.where(sim.states[:, 1] == 0.0, "M", "W")
    parts = []
    truncated = []
    for t, (seen, rec) in enumerate(zip(sim.states_seen, sim.market_log)):
        log = rec["swipe_log"]
        pos = log[log[:, 2] == 1, :2] if log is not None else np.zeros((0, 2), np.int64)
        volume, intent = received_proposals(seen, pos)
        swiped = np.bincount(pos[:, 0], minlength=sim.n)
        theta = seen[:, 6]
        da = rec["da_proposals"]
        matched = rec["matches"]
        yld = np.where(da > 0, matched / np.maximum(da, 1), np.nan)
        flood = rec["flood"] if rec["flood"] is not None else np.zeros(sim.n)
        parts.append(pd.DataFrame({
            "epoch": t, "row": np.arange(sim.n), "side": side,
            "theta": theta, "dtheta": rec["theta_after"] - theta,
            "volume": volume, "recv_intent": intent, "da_received": da,
            "matches": matched, "yield": yld, "swiped": swiped,
            "intent": seen[:, 11], "flood": flood,
        }))
        prop_adm, acc_adm = admissible_prefs(seen, cfg)
        lens = np.array([len(x) for x in prop_adm] + [len(x) for x in acc_adm])
        truncated.append(float(np.mean(lens < top_k)))

    panel = pd.concat(parts, ignore_index=True)
    panel.insert(0, "seed", seed)
    panel.insert(1, "top_k", top_k)
    panel.insert(2, "cost", cost)
    panel.insert(3, "w", weight)
    # V/V̄ per (epoch, side): the scale-free volume the flooding term reads.
    vbar = panel.groupby(["epoch", "side"])["volume"].transform("mean")
    panel["vrel"] = np.where(vbar > 0, panel["volume"] / vbar.where(vbar > 0, 1.0), np.nan)

    # An epoch is open when anyone proposed. Volume and intent are cell properties
    # of the open market; a closed epoch is counted, not averaged in.
    open_epoch = panel.groupby("epoch")["volume"].transform("sum") > 0
    w_rows = (panel["side"] == "W") & open_epoch
    m_rows = (panel["side"] == "M") & open_epoch
    cell = {
        "n_open_epochs": int(panel.loc[open_epoch, "epoch"].nunique()),
        "seed": seed, "top_k": top_k, "cost": cost, "w": weight,
        "sd_theta0": float(sim.states_seen[0][:, 6].std(ddof=1)),
        "mean_V_W": float(panel.loc[w_rows, "volume"].mean()),
        "mean_V_M": float(panel.loc[m_rows, "volume"].mean()),
        "cv_V_W": float((panel.loc[w_rows].groupby("epoch")["volume"].std()
                         / panel.loc[w_rows].groupby("epoch")["volume"].mean()).mean()),
        "mean_recv_intent_W": float(panel.loc[w_rows, "recv_intent"].mean()),
        "mean_proposal_intent": float(frame["mean_proposal_intent"].dropna().mean()),
        "share_M_proposing": float((panel.loc[m_rows, "swiped"] > 0).mean()),
        "share_W_received": float((panel.loc[w_rows, "volume"] > 0).mean()),
        "mean_da_received_W": float(panel.loc[w_rows, "da_received"].mean()),
        "truncated_share_early": float(np.mean(truncated[:5])),
        "truncated_share_late": float(np.mean(truncated[5:])),
        "mean_dtheta_W": float(panel.loc[panel["side"] == "W", "dtheta"].mean()),
        "mean_dtheta_M": float(panel.loc[panel["side"] == "M", "dtheta"].mean()),
        "n_matched_mean": float(frame["n_matched"].mean()),
    }
    return panel, cell


def ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coefficients and classical standard errors; X already carries the intercept."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(1, X.shape[0] - X.shape[1])
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    return beta, np.sqrt(np.diag(cov))


def fit(panel: pd.DataFrame) -> list[dict]:
    """Δθ on V/V̄ and τ̄, with the yield step's own θ term held fixed.

    The θ control is θ_t * 1[DA proposals > 0], with 1[DA > 0] as its own
    regressor, not θ_t alone: update_threshold skips an agent with no DA proposals,
    so -ηθ acts only on active agents, and on the acceptor side most agents are not
    active. A bare θ_t control would be averaged over the two regimes and absorb
    neither.

    Four models:
      net            V/V̄ only, no controls, every row: the slope an observer of
                     (V, Δθ) alone would see, yield step and all.
      volume         V/V̄ with the θ control, every row.
      volume_intent  V/V̄ + τ̄, receivers only (τ̄ is undefined at V = 0).
      interaction    V/V̄ + τ̄ + (V/V̄ - 1)(1 - τ̄), receivers only.

    The interaction is the flooding term's own shape and is the coefficient that
    separates the accounts. τ̄'s main effect does not: ∂/∂τ̄ of the flooding term is
    -w(V/V̄ - 1), positive for every below-average receiver, so its sign in a
    linear fit is set by where τ̄ happens to vary — and τ̄ varies most among
    low-volume agents, whose mean is over few proposals.
    """
    out = []
    keys = ["seed", "top_k", "cost", "w"]
    panel = panel.assign(
        active=(panel["da_received"] > 0).astype(np.float64),
        theta_active=panel["theta"] * (panel["da_received"] > 0),
        inter=(panel["vrel"] - 1.0) * (1.0 - panel["recv_intent"]),
    )
    for vals, g in panel.groupby([*keys, "side"]):
        for win, (lo, hi) in WINDOWS.items():
            sub = g[(g["epoch"] >= lo) & (g["epoch"] <= hi) & g["vrel"].notna()]
            recv = sub[sub["volume"] > 0]
            models = {
                "net": (sub, []),
                "volume": (sub, ["vrel"]),
                "volume_intent": (recv, ["vrel", "recv_intent"]),
                "interaction": (recv, ["vrel", "recv_intent", "inter"]),
            }
            for name, (d, cols) in models.items():
                if name == "net":
                    cols = ["vrel"]
                else:
                    controls = (["theta_active", "active"] if 0.0 < d["active"].mean() < 1.0
                                else ["theta"])
                    cols = controls + cols
                if len(d) < len(cols) + 2:
                    continue
                X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy() for c in cols])
                beta, se = ols(d["dtheta"].to_numpy(), X)
                row = dict(zip([*keys, "side"], vals))
                row.update(window=win, model=name, n=len(d),
                           n_dropped_zero_V=int((sub["volume"] == 0).sum()) if d is recv else 0,
                           share_active=float(d["active"].mean()))
                for c, b, s in zip(["const", *cols], beta, se):
                    row[f"b_{c}"] = float(b)
                    row[f"se_{c}"] = float(s)
                out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="experiments/base.yaml")
    ap.add_argument("--out", default="data/processed/flooding")
    ap.add_argument("--refit", action="store_true",
                    help="re-run the regressions on an existing panel.parquet; no simulation")
    args = ap.parse_args()

    base = SimConfig.from_yaml(args.config).to_dict()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.refit:
        panel = pd.read_parquet(out / "panel.parquet")
        pd.DataFrame(fit(panel)).to_csv(out / "coefficients.csv", index=False)
        return

    panels, cells, coefs = [], [], []
    for top_k, cost, weight, seed in itertools.product(TOP_K, COSTS, WEIGHTS, SEEDS):
        t0 = time.time()
        panel, cell = run_cell(base, seed, top_k, cost, weight)
        panels.append(panel)
        cells.append(cell)
        coefs.extend(fit(panel))
        print(f"k={top_k} c={cost} w={weight} seed={seed}: {time.time() - t0:.1f}s", flush=True)

    pd.concat(panels, ignore_index=True).to_parquet(out / "panel.parquet", index=False)
    pd.DataFrame(cells).to_csv(out / "cells.csv", index=False)
    pd.DataFrame(coefs).to_csv(out / "coefficients.csv", index=False)


if __name__ == "__main__":
    main()
