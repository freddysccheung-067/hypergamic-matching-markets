"""The H1 sweep behind the separating-curve figure (THEORY.md §10.2).

For each seed and each signal cost c, one epoch-0 swipe log on the shipped weights
and the measured quantities the figure needs:

    mean_tau        E[tau | proposal], the mean intent of the proposers in the log
    tau_star        c / (Vbar * 0.5): the analytic cut with the p the decision uses
    tau_star_phat   c / (Vbar * phat): the same cut with the measured accept rate
    tau_trunc       mean of tau over agents above tau_star (what a pure cut predicts
                    E[tau | proposal] to be, from the population's own tau column)
    phat            share of positive swipes the other side reciprocated

``Vbar`` is the pooled mean over theta-admissible candidates, as in the orchestrator
(``_mean_admissible_utility_by_side``) but pooled over both sides because the figure
has one axis. Nothing here is reported at a single seed.

Output: data/processed/separating/h1.csv
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "build" / "Release"))

COST_GRID = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
SEEDS = [42, 43, 44, 45, 46]
DECISION_P = 0.5  # the epoch-0 prior the C++ decision is handed (orchestrator)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--output", default="data/processed/separating/h1.csv")
    args = parser.parse_args()

    import _hypergamic  # noqa: PLC0415

    with open(ROOT / "experiments" / "base.yaml", encoding="utf-8") as fh:
        base = yaml.safe_load(fh)

    rows = []
    for seed in SEEDS:
        cfg0 = yaml.safe_load(yaml.safe_dump(base))
        cfg0["population"]["seed"] = seed
        states = _hypergamic.sample_population(args.n, cfg0, seed)
        tau = states[:, 11]

        means, counts = _hypergamic.mean_admissible_utility(states, cfg0)
        ok = np.isfinite(means) & (counts > 0)
        v_bar = float(np.sum(means[ok] * counts[ok]) / np.sum(counts[ok]))

        for cost in COST_GRID:
            cfg = yaml.safe_load(yaml.safe_dump(cfg0))
            cfg["hypergamy"]["signal_cost"] = float(cost)
            cfg["signal"] = {"accept_prob": DECISION_P}
            log = _hypergamic.generate_swipes(states, cfg, 0)
            pos = log[log[:, 2] == 1]
            n_pos = int(pos.shape[0])
            if n_pos:
                u = pos[:, 0].astype(np.int64)
                v = pos[:, 1].astype(np.int64)
                mutual = np.isin(v * args.n + u, u * args.n + v)
                mean_tau = float(tau[u].mean())
                phat = float(mutual.mean())
            else:
                mean_tau, phat = float("nan"), float("nan")

            t_star = cost / (v_bar * DECISION_P)
            t_star_hat = cost / (v_bar * phat) if n_pos and phat > 0 else float("nan")
            above = tau[tau > t_star]
            rows.append({
                "signal_cost": cost, "seed": seed, "n_proposals": n_pos,
                "mean_tau": mean_tau, "v_bar": v_bar, "p_decision": DECISION_P,
                "phat": phat, "tau_star": t_star, "tau_star_phat": t_star_hat,
                "tau_trunc": float(above.mean()) if above.size else float("nan"),
            })
            print(f"seed {seed} c={cost:<5} n={n_pos:<6} E[tau|prop]={mean_tau:.4f} "
                  f"tau*={t_star:.4f} phat={phat:.4f}")

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
