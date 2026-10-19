"""Run the four ablation arms across seeds (docs/SPRINT_PROMPTS.md Session 3-C).

What this produces, and what each output is for:

    data/processed/ablation/<arm>_seed<seed>.csv   one long-format metrics table
                                                   per (arm, seed): all epochs, so
                                                   the trajectory is recoverable
                                                   and the Lorenz plot can show
                                                   it rather than only the end.
    data/processed/ablation/summary.csv            per-arm mean and sd of every
                                                   metric over seeds.
    data/processed/ablation/gaps.csv               the three per-seed gaps, with
                                                   their spread.

The gaps are computed **per seed and then averaged**, never as a difference of
arm means. Two arms sharing a seed share a population, so their difference at
that seed is one observation of the ablation effect; averaging the arms first
and subtracting afterwards drops the pairing and inflates the interval.

There is deliberately no "gini_excess_over_random" headline. The random arm has
the *lowest* Gini of the four — uniform exposure is near-equal by construction —
so `Gini(<arm>) - Gini(random)` mostly measures distance from uniformity. The
quantity that carries the argument is `gini(ncf) - gini(oracle)`: how much the
recommender's estimation error costs relative to a ranker that is perfect at the
same task. See the preamble to Session 3-C, point 4.

The noise-to-signal ratio is recorded per row. Nothing in this table means
anything without it (docs/THEORY.md §2.3): the oracle arm contains unlearnable
pairwise chemistry, so even a perfect recommender cannot reach it, and how far
short it falls is a function of that ratio rather than of model quality.
"""

from __future__ import annotations

import argparse
import copy
import pathlib
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

ARMS = ("ncf", "random", "popularity", "oracle")

# Fixed column order of every per-(arm, seed) CSV. `noise_ratio` is constant
# within a (arm, seed) — it is a property of the population, not of the epoch —
# and is repeated on every row so a row is self-describing when a reader only
# has one line of a file in front of them.
COLUMNS = [
    "arm", "seed", "epoch", "gini", "reach", "top1_share",
    "mcr", "rsi", "welfare_alg", "noise_ratio",
]

# The three gaps the session preamble specifies, as (label, left, right) over
# arm names. `lowrank` in the preamble's notation is this project's `ncf` arm:
# the NCF *is* the low-rank approximation, and the preamble's "rank-8 replica"
# was standing in for it.
GAPS = [
    ("gini_ncf_minus_oracle", "ncf", "oracle"),
    ("gini_oracle_minus_random", "oracle", "random"),
    ("gini_popularity_minus_random", "popularity", "random"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/base.yaml")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--epochs", type=int, default=None,
                        help="default: the config's simulation.n_epochs")
    parser.add_argument("--output-dir", default="data/processed/ablation")
    parser.add_argument("--arms", default=",".join(ARMS))
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        base_config = yaml.safe_load(fh)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if len(seeds) < 5:
        print(f"warning: {len(seeds)} seeds; the session asks for at least 5",
              file=sys.stderr)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise SystemExit(f"unknown arm(s) {sorted(unknown)}; expected {ARMS}")

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Imported here so the C++ extension is already built when this runs.
    from src.python.orchestrator import Simulation  # noqa: PLC0415

    frames: dict[str, pd.DataFrame] = {}
    per_seed_gaps: dict[str, list[float]] = {label: [] for label, _, _ in GAPS}

    for seed in seeds:
        seed_end: dict[str, float] = {}
        for arm in arms:
            cfg = copy.deepcopy(base_config)
            cfg["population"]["seed"] = seed
            n_epochs = args.epochs if args.epochs is not None \
                else int(cfg["simulation"]["n_epochs"])

            sim = Simulation(config=cfg, recommender=arm)
            metrics = sim.run(n_epochs=n_epochs)

            table = pd.DataFrame({
                "arm": arm,
                "seed": seed,
                "epoch": metrics["epoch"].to_numpy(),
                "gini": metrics["gini"].to_numpy(),
                "reach": metrics["reach"].to_numpy(),
                "top1_share": metrics["top1_share"].to_numpy(),
                "mcr": metrics["mcr"].to_numpy(),
                "rsi": metrics["rsi"].to_numpy(),
                "welfare_alg": metrics["welfare_alg"].to_numpy(),
                # Per-epoch noise ratio: the population's chemistry scale over
                # its own utility spread. Constant within a run by construction.
                "noise_ratio": np.full(len(metrics), sim.noise_ratio),
            })[COLUMNS]

            path = out_dir / f"{arm}_seed{seed}.csv"
            table.to_csv(path, index=False)
            frames.setdefault(arm, pd.DataFrame())
            frames[arm] = pd.concat([frames[arm], table], ignore_index=True)

            # The final epoch is the arm's steady state: the run's own history has
            # been fed back through θ by then, which is the object worth comparing.
            seed_end[arm] = float(table["gini"].iloc[-1])
            print(f"seed {seed} arm {arm:<11} gini={seed_end[arm]:.4f} "
                  f"reach={table['reach'].iloc[-1]:.3f} "
                  f"mcr={table['mcr'].iloc[-1]:.4f} "
                  f"nsr={sim.noise_ratio:.3f} -> {path.name}")

        # Paired by seed: both arms in a difference ran on the same population.
        for label, left, right in GAPS:
            if left in seed_end and right in seed_end:
                per_seed_gaps[label].append(seed_end[left] - seed_end[right])

    # ── summary ───────────────────────────────────────────────────────────────
    metrics = ["gini", "reach", "top1_share", "mcr", "rsi", "welfare_alg",
               "noise_ratio"]
    summary_rows: list[dict] = []
    for arm, frame in frames.items():
        # One observation per seed, not per epoch: epochs within a run are a
        # trajectory, not independent draws, and pooling them would shrink the sd
        # by roughly sqrt(n_epochs) for no reason.
        final = frame.groupby("seed", as_index=False).tail(1)
        for metric in metrics:
            series = final[metric].to_numpy(dtype=float)
            series = series[np.isfinite(series)]
            summary_rows.append({
                "arm": arm,
                "metric": metric,
                "n_seeds": int(series.size),
                "mean": float(np.mean(series)) if series.size else float("nan"),
                "sd": float(np.std(series, ddof=1)) if series.size > 1 else float("nan"),
            })
    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    gap_rows = []
    for label, _, _ in GAPS:
        values = np.asarray(per_seed_gaps[label], dtype=float)
        if values.size == 0:
            continue
        gap_rows.append({
            "gap": label,
            "n_seeds": int(values.size),
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
        })
    gaps = pd.DataFrame(gap_rows)
    gaps_path = out_dir / "gaps.csv"
    gaps.to_csv(gaps_path, index=False)

    # ── console report ────────────────────────────────────────────────────────
    print("\nFinal-epoch Gini by arm (mean ± sd over seeds):")
    for arm in arms:
        row = summary[(summary.arm == arm) & (summary.metric == "gini")].iloc[0]
        print(f"  {arm:<11} {row['mean']:.4f} ± {row['sd']:.4f}")

    print("\nPer-seed gaps (paired by seed):")
    for row in gap_rows:
        print(f"  {row['gap']:<28} {row['mean']:+.4f} ± {row['sd']:.4f}"
              f"   ({row['n_seeds']} seeds)")

    print(f"\nwrote {len(frames)} arm tables, {summary_path.name}, {gaps_path.name} "
          f"to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
