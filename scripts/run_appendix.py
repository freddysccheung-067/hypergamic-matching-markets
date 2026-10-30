"""R2-H1r and R2-H5r: the appendix hypotheses re-measured at the shipped gate.

    python scripts/run_appendix.py h1 --seeds 100-109 --output-dir data/processed/horizon/h1r
    python scripts/run_appendix.py h5 --seed 100 --output-dir data/processed/horizon/h5r

``h1`` repeats ``scripts/run_separating.py`` (one round-0 swipe log per seed and
signal cost, no training) at the configuration as shipped, writing ``h1.csv``.
``h5`` runs the factorial of ``src/python/factorial.py`` for one seed and writes
``seed<seed>.csv``. Both write a provenance record beside their data.
"""

from __future__ import annotations

import argparse
import copy
import pathlib
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "build" / "Release"))

from src.python.provenance import JobRecord, write_csv_atomic  # noqa: E402

COST_GRID = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
DECISION_P = 0.5


def seeds_from(text: str) -> list[int]:
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi) + 1)) if hi else [int(s) for s in text.split(",")]


def h1(base: dict, seeds: list[int]) -> pd.DataFrame:
    import _hypergamic  # noqa: PLC0415

    rows = []
    for seed in seeds:
        cfg0 = copy.deepcopy(base)
        cfg0["population"]["seed"] = seed
        n = int(cfg0["population"]["n"])
        states = _hypergamic.sample_population(n, cfg0, seed)
        tau = states[:, 11]
        means, counts = _hypergamic.mean_admissible_utility(states, cfg0)
        ok = np.isfinite(means) & (counts > 0)
        v_bar = float(np.sum(means[ok] * counts[ok]) / np.sum(counts[ok]))
        for cost in COST_GRID:
            cfg = copy.deepcopy(cfg0)
            cfg["hypergamy"]["signal_cost"] = float(cost)
            cfg["signal"] = {"accept_prob": DECISION_P}
            log = _hypergamic.generate_swipes(states, cfg, 0)
            pos = log[log[:, 2] == 1]
            u = pos[:, 0].astype(np.int64)
            v = pos[:, 1].astype(np.int64)
            t_star = cost / (v_bar * DECISION_P)
            above = tau[tau > t_star]
            rows.append({
                "seed": seed, "signal_cost": cost, "n_proposals": int(pos.shape[0]),
                "mean_tau": float(tau[u].mean()) if pos.shape[0] else float("nan"),
                "phat": float(np.isin(v * n + u, u * n + v).mean()) if pos.shape[0]
                else float("nan"),
                "v_bar": v_bar, "tau_star": t_star,
                "tau_trunc": float(above.mean()) if above.size else float("nan"),
            })
    return pd.DataFrame(rows)


def h5(base: dict, seed: int, gamma: float, rounds: int, window: int) -> pd.DataFrame:
    from src.python.factorial import run_factorial  # noqa: PLC0415

    cfg = copy.deepcopy(base)
    cfg["population"]["seed"] = seed
    res = run_factorial(cfg, gamma, rounds, window)
    return pd.DataFrame([{"seed": seed, "gamma": res.gamma,
                          **{f"W_{k}": v for k, v in res.cells.items()},
                          "exogenous": res.exogenous, "algorithmic": res.algorithmic,
                          "interaction": res.interaction}])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("which", choices=("h1", "h5"))
    parser.add_argument("--seeds", default="100-109")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=0.02)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "experiments" / "base.yaml", encoding="utf-8") as fh:
        base = yaml.safe_load(fh)
    from src.python.config import SimConfig  # noqa: PLC0415
    resolved = SimConfig.coerce(copy.deepcopy(base)).to_dict()
    if args.which == "h1":
        seeds = seeds_from(args.seeds)
        done, data = out_dir / "h1.json", out_dir / "h1.csv"
        record = JobRecord(ROOT, "R2-H1r", seeds[0], resolved, sys.argv, args.allow_dirty)
        write_csv_atomic(h1(base, seeds), data)
        record.finish(done, seeds=seeds)
    else:
        done, data = out_dir / f"seed{args.seed}.json", out_dir / f"seed{args.seed}.csv"
        record = JobRecord(ROOT, "R2-H5r", args.seed, resolved, sys.argv, args.allow_dirty)
        write_csv_atomic(h5(base, args.seed, args.gamma, args.rounds, args.window), data)
        record.finish(done, gamma=args.gamma, rounds=args.rounds, window=args.window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
