"""Run the four ablation arms keeping every metric column, over an arbitrary seed list.

`scripts/run_ablation.py` writes a trimmed column set — nine metrics, chosen so
the headline table reads cleanly. The statistical work needs the columns that
were trimmed: `algorithmic_loss` and `welfare_full` in particular are computed on
every epoch and thrown away at write time, and no committed directory carries
them. So this script runs the same arms over the same config and keeps the whole
`Simulation.run()` frame.

Two jobs read this output:

    §1  intervals at n = 5   — seeds 42-46, the same five populations
                                `data/processed/ablation_h100/` used
    §2  power at n = 20      — the same five plus fifteen more

One directory serves both: n is chosen by the reader of the CSVs, not by the
run, which is what lets §2 compare 5 seeds against 20 on an identical regime
rather than across two.

Usage (one seed per process, so it parallelises by seed):

    python scripts/run_rigor_base.py --seed 42 --output-dir data/processed/rigor/base
    python scripts/run_rigor_base.py --all --jobs 8

The per-seed file is `metrics_seed<seed>.csv`, holding one row per (arm, epoch)
and every column of INTERFACES.md §6, for all four arms. Keeping the arms in one
file per seed makes the pairing explicit: the arms in a row set share a
population, so a gap is a row-wise subtraction and never a difference of means.
"""

from __future__ import annotations

import argparse
import concurrent.futures as _futures
import copy
import pathlib
import sys

import pandas as pd
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# The four arms, in the order a table should read them: the two that carry the
# argument first, then the two baselines the reader checks it against.
ARMS = ("ncf", "oracle", "random", "popularity")

# Columns that are constant within a (seed, arm) run by construction. Written on
# every row so a row is self-describing, exactly as run_ablation.py does it.
PASS_THROUGH = ("delta", "seed", "generation")


def run_one(seed: int, base_config: dict, epochs: int, out_dir: pathlib.Path) -> str:
    """Run every arm at one seed and write the combined frame. Returns a status line."""
    from src.python.orchestrator import Simulation  # noqa: PLC0415

    frames = []
    for arm in ARMS:
        cfg = copy.deepcopy(base_config)
        cfg["population"]["seed"] = int(seed)
        sim = Simulation(config=cfg, recommender=arm)
        metrics = sim.run(n_epochs=epochs)
        metrics = metrics.copy()
        metrics["arm"] = arm
        # noise_ratio is a property of the population, not the epoch, and it is
        # what makes rows from different heterogeneity settings comparable.
        metrics["noise_ratio"] = float(sim.noise_ratio)
        frames.append(metrics)

    table = pd.concat(frames, ignore_index=True)
    path = out_dir / f"metrics_seed{seed}.csv"
    table.to_csv(path, index=False)

    final = table[table.epoch == table.epoch.max()].set_index("arm")
    gap = final.loc["ncf", "gini"] - final.loc["oracle", "gini"]
    loss = final.loc["ncf", "algorithmic_loss"]
    return (f"seed {seed:>3}: gini ncf={final.loc['ncf', 'gini']:.4f} "
            f"oracle={final.loc['oracle', 'gini']:.4f} gap={gap:+.4f} "
            f"alg_loss={loss:.4f} nsr={sim.noise_ratio:.3f} -> {path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/base.yaml")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--n-seeds", type=int, default=None,
                        help="run seeds 42..42+n-1; overrides --seeds")
    parser.add_argument("--all", action="store_true",
                        help="alias for --n-seeds 20")
    parser.add_argument("--epochs", type=int, default=None,
                        help="default: the config's simulation.n_epochs")
    parser.add_argument("--output-dir", default="data/processed/rigor/base")
    parser.add_argument("--jobs", type=int, default=1,
                        help="worker processes, one seed each")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        base_config = yaml.safe_load(fh)

    if args.all:
        seeds = list(range(42, 62))
    elif args.n_seeds is not None:
        seeds = list(range(42, 42 + args.n_seeds))
    else:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    epochs = args.epochs if args.epochs is not None \
        else int(base_config["simulation"]["n_epochs"])

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.jobs <= 1:
        for seed in seeds:
            print(run_one(seed, base_config, epochs, out_dir), flush=True)
    else:
        # One process per seed. The simulation is already OpenMP-parallel inside
        # one epoch, so jobs * threads is capped elsewhere (OMP_NUM_THREADS in
        # the caller); this only decides how many seeds are in flight.
        with _futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run_one, s, base_config, epochs, out_dir)
                       for s in seeds]
            for fut in _futures.as_completed(futures):
                print(fut.result(), flush=True)

    print(f"\nwrote {len(seeds)} seed files to {out_dir} "
          f"({epochs} epochs, {len(ARMS)} arms)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
