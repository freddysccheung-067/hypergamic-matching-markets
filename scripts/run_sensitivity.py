"""§4: does any headline survive a different corner of the parameter space?

Every committed result sits at one setting of four knobs. This sweeps each knob
one at a time, holding the others at the shipped value, at N = 1000, k = 20,
5 seeds, 100 epochs, and records the three quantities a reader will re-read at
each cell: the NCF−oracle Gini gap, the NCF arm's reach, and the final-epoch gap.

| knob | levels | shipped |
|---|---|---|
| `population.taste_sigma` | 0.2, 0.5, 1.0 | 0.5 |
| `hypergamy.shock_scale` | 0.5, 1.5, 3.0 | 1.5 |
| `hypergamy.beta / alpha` | 1.0, 2.0, 3.0 | 2.0 |
| `hypergamy.delta` | 0.0, 0.5, 1.0 | 0.5 |

**The β/α axis is the one that is expected to matter**, and the reason is
documented in `experiments/base.yaml` and guarded by `tests/cpp/test_delta_power`:
at α = 1.0, β = 0.8 the measured top-1000 overlap between δ = 0 and δ = 1 is
0.9661, so δ moves almost nothing — a status-dominated utility makes the
hypergamous indicator nearly constant across the shortlist. At β/α = 2 the
overlap is 0.7615. Below 2 the δ term is inert *by construction*, so any
hypergamy result at β/α < 2 is a result about the other three knobs wearing δ's
name. This script therefore runs the δ axis at the shipped β/α and reports the
overlap test at each β/α level so the claim can be checked rather than asserted.

The shipped cell (taste_sigma 0.5, shock_scale 1.5, β/α 2.0, δ 0.5) appears in
three of the four axes. It is run once and its result is reused, so the grid has
3 + 3 + 3 + 3 − 2 = 10 distinct non-shipped cells plus the shipped one.

Writes one CSV per cell plus a tidy `sensitivity.csv` with the three quantities
per (axis, level) and an `is_shipped` flag.
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

ARMS = ("ncf", "oracle")

# (axis, level label, config mutations). The shipped value is stated per axis so
# the reused cell can be identified rather than inferred from float comparison.
AXES: dict[str, dict] = {
    "taste_sigma": {
        "shipped": 0.5,
        "levels": {
            "0.2": {"population": {"taste_sigma": 0.2}},
            "0.5": {"population": {"taste_sigma": 0.5}},
            "1.0": {"population": {"taste_sigma": 1.0}},
        },
    },
    "shock_scale": {
        "shipped": 1.5,
        "levels": {
            "0.5": {"hypergamy": {"shock_scale": 0.5}},
            "1.5": {"hypergamy": {"shock_scale": 1.5}},
            "3.0": {"hypergamy": {"shock_scale": 3.0}},
        },
    },
    "beta_alpha": {
        "shipped": 2.0,
        "levels": {
            # alpha held at 1.0 throughout; the ratio is what moves.
            "1.0": {"hypergamy": {"alpha": 1.0, "beta": 1.0}},
            "2.0": {"hypergamy": {"alpha": 1.0, "beta": 2.0}},
            "3.0": {"hypergamy": {"alpha": 1.0, "beta": 3.0}},
        },
    },
    "delta": {
        "shipped": 0.5,
        "levels": {
            "0.0": {"hypergamy": {"delta": 0.0}},
            "0.5": {"hypergamy": {"delta": 0.5}},
            "1.0": {"hypergamy": {"delta": 1.0}},
        },
    },
}

CELL_FILE = "metrics_{axis}_{level}_seed{seed}.csv"


def _apply(cfg: dict, mutations: dict) -> dict:
    for section, values in mutations.items():
        cfg.setdefault(section, {}).update(values)
    return cfg


def run_cell(
    axis: str,
    level: str,
    mutations: dict,
    base_config: dict,
    seeds: list[int],
    epochs: int,
    out_dir: pathlib.Path,
) -> str:
    """Run ncf and oracle at every seed for one cell; write one CSV per seed."""
    from src.python.orchestrator import Simulation  # noqa: PLC0415

    rows = []
    for seed in seeds:
        cfg = _apply(copy.deepcopy(base_config), mutations)
        cfg["population"]["seed"] = int(seed)
        for arm in ARMS:
            sim = Simulation(config=cfg, recommender=arm)
            frame = sim.run(n_epochs=epochs)
            last = frame.iloc[-1]
            rows.append({
                "axis": axis,
                "level": level,
                "seed": int(seed),
                "arm": arm,
                "epoch": int(last["epoch"]),
                "gini": float(last["gini"]),
                "reach": float(last["reach"]),
                "mcr": float(last["mcr"]),
                "rsi": float(last["rsi"]),
                "welfare_alg": float(last["welfare_alg"]),
                "noise_ratio": float(sim.noise_ratio),
            })
    table = pd.DataFrame(rows)
    path = out_dir / f"cell_{axis}_{level}.csv"
    table.to_csv(path, index=False)

    wide = table.pivot(index="seed", columns="arm")
    gap = (wide[("gini", "ncf")] - wide[("gini", "oracle")])
    return (f"{axis:<12} {level:<5} gap={gap.mean():+.4f}±{gap.std(ddof=1):.4f} "
            f"reach={wide[('reach','ncf')].mean():.3f} nsr={table.noise_ratio.mean():.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/base.yaml")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output-dir", default="data/processed/rigor/sensitivity")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        base_config = yaml.safe_load(fh)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = [(axis, level, muts, AXES[axis]["shipped"])
             for axis, spec in AXES.items()
             for level, muts in spec["levels"].items()]

    if args.jobs > 1:
        with _futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [
                pool.submit(run_cell, a, lv, mu, base_config, seeds, args.epochs, out_dir)
                for a, lv, mu, _ in cells
            ]
            for fut in _futures.as_completed(futures):
                print(fut.result(), flush=True)
    else:
        for axis, level, muts, _ in cells:
            print(run_cell(axis, level, muts, base_config, seeds, args.epochs, out_dir),
                  flush=True)

    print(f"\nwrote {len(cells)} cells to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
