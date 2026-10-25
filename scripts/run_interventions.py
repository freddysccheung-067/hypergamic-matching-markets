"""C3: manipulate reach and see whether the exposure gap still closes.

The paper's weakest claim is its mechanism. The gap closes while coverage climbs,
so coverage learning is offered as the reason — but reach was never manipulated,
and a correlation over a trajectory is not a mechanism. This script runs two
interventions against a control in the same session, at one thread count.

    python scripts/run_interventions.py --seeds 42,43,44,45,46 --jobs 5

Conditions, all at ``experiments/base.yaml`` with ``ncf.num_threads = 1``:

  control      the shipped loop, 100 epochs
  frozen       train to the peak (epoch 6) then stop training, keep the market
               running to epoch 100. If the gap is a *consequence* of continued
               training it must stay at its peak; if it decays anyway, something
               else closes it.
  warm         pretrain on the epoch-0 swipe log for ``--warm-passes`` passes,
               then run the shipped loop. If the gap is a *consequence* of
               starting from ignorance, a ranker that starts informed should
               begin near zero and have little left to decay.

Predictions and fail conditions are registered in ``docs/PREREGISTRATION.md`` E3.
They were committed before this script was first run.

**Paired, always.** All three conditions at one seed share a population, so every
reported difference is a per-seed difference against the control, never a
difference of means against a committed directory. The committed directories were
generated at 24 threads and are not comparable to these runs at 1
(``docs/INTERFACES.md`` §3.1(b)).

Output, per seed: ``interventions_seed<seed>.csv`` with one row per (condition,
epoch) over the columns the headline quantities need, plus
``interventions_summary.csv`` with the per-seed endpoint gaps and the paired
differences.
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

COLUMNS = [
    "condition", "seed", "epoch", "gini", "reach", "mcr", "rsi",
    "top1_share", "welfare_alg", "noise_ratio",
]

# The peak epoch, from the committed long-horizon series: the five-seed mean gap
# is highest at epoch 6. `freeze_after` is the last epoch on which the ranker
# trains, so 6 means epochs 0..6 train and epoch 7 onward runs on frozen weights.
PEAK_EPOCH = 6


def _conditions(warm_passes: int) -> dict[str, dict]:
    return {
        "control": {},
        "frozen": {"freeze_after": PEAK_EPOCH},
        "warm": {"warm_start_epochs": int(warm_passes)},
    }


def run_one(seed: int, base_config: dict, epochs: int, warm_passes: int,
            out_dir: pathlib.Path) -> str:
    from src.python.orchestrator import Simulation  # noqa: PLC0415

    frames = []
    for name, overrides in _conditions(warm_passes).items():
        cfg = copy.deepcopy(base_config)
        cfg["population"]["seed"] = int(seed)
        cfg["ncf"].update(overrides)
        sim = Simulation(config=cfg, recommender="ncf")
        metrics = sim.run(n_epochs=epochs).copy()
        metrics["condition"] = name
        metrics["seed"] = int(seed)
        metrics["noise_ratio"] = float(sim.noise_ratio)
        frames.append(metrics[COLUMNS])

    table = pd.concat(frames, ignore_index=True)
    path = out_dir / f"interventions_seed{seed}.csv"
    table.to_csv(path, index=False)

    final = table[table.epoch == table.epoch.max()].set_index("condition")
    peak = table[table.epoch == PEAK_EPOCH].set_index("condition")
    return (f"seed {seed:>3}: gap control={final.loc['control', 'gini']:.4f} "
            f"frozen={final.loc['frozen', 'gini']:.4f} warm={final.loc['warm', 'gini']:.4f} "
            f"| peak(6) control={peak.loc['control', 'gini']:.4f} "
            f"frozen={peak.loc['frozen', 'gini']:.4f} "
            f"warm={peak.loc['warm', 'gini']:.4f} -> {path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/base.yaml")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warm-passes", type=int, default=20,
                        help="passes over the epoch-0 log before the loop starts")
    parser.add_argument("--output-dir", default="data/processed/rigor/interventions")
    parser.add_argument("--jobs", type=int, default=1, help="worker processes, one seed each")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        base_config = yaml.safe_load(fh)

    # Pinned here rather than trusted from the file: these runs are compared
    # against each other and against the fresh-seed set, and both are at 1.
    base_config["ncf"]["num_threads"] = 1

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.jobs <= 1:
        for seed in seeds:
            print(run_one(seed, base_config, args.epochs, args.warm_passes, out_dir),
                  flush=True)
    else:
        with _futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run_one, s, base_config, args.epochs,
                                   args.warm_passes, out_dir) for s in seeds]
            for fut in _futures.as_completed(futures):
                print(fut.result(), flush=True)

    print(f"\nwrote {len(seeds)} seed files to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
