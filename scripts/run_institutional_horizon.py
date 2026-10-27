"""R2-E15: the institutional comparison, measured after the transient.

    python scripts/run_institutional_horizon.py --seed 100 --rounds 300 \
        --window-start 250 --output-dir data/processed/horizon/e15_institutional

For one seed: run each Environment A arm (oracle A and NCF A) once, then every
(c_B, size_fraction) cell of Environment B, and pair B with each A over the
window ``[window-start, rounds)``. Environment A reads neither c_B nor the
subgraph size, so one A run per arm serves all forty cells; that is checked
rather than assumed, by comparing A's welfare with an A run built from the first
cell's config.

Writes ``seed<seed>.csv``, one row per (A arm, c_B, size_fraction), and the
provenance record ``seed<seed>.json``. The welfare difference is the residual
one of ``scripts/run_institutional.py``: B's common-basis welfare with A re-run on
the agents B leaves behind, minus A's own welfare, averaged over the window.
"""

from __future__ import annotations

import argparse
import copy
import logging
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.python.provenance import JobRecord, write_csv_atomic  # noqa: E402

SIZE_GRID = [0.02, 0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 1.00]
COST_GRID = [0.01, 0.05, 0.10, 0.25]


def apply_set(cfg: dict, pairs: list[str]) -> dict:
    for item in pairs:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def run_a(cfg: dict, arm: str, rounds: int, start: int):
    from src.python.orchestrator import Simulation  # noqa: PLC0415

    sim = Simulation(config=copy.deepcopy(cfg), recommender=arm)
    sim.record_outcomes = True
    sim.record_outcomes_from = start
    frame = sim.run(rounds)
    return sim, frame


def pair_cell(sim_a, frame_a, sim_b, frame_b, start: int) -> dict[str, float]:
    from src.python.comparison import welfare_accounts  # noqa: PLC0415

    env = sim_b.config.require_environment_b()
    residual = []
    for oa, ob in zip(sim_a.outcomes, sim_b.outcomes):
        acc = welfare_accounts(ob, oa, sim_a.config_dict, env)
        residual.append(acc.welfare_common_residual)
    wa = frame_a.loc[frame_a.epoch >= start, "welfare_total"].to_numpy()
    if len(wa) != len(residual):
        raise RuntimeError("A and B windows differ in length")
    fb = frame_b[frame_b.epoch >= start]
    elig = sim_b.subgraph.n_eligible
    return {
        "welfare_a": float(wa.mean()),
        "welfare_common_residual_b": float(np.mean(residual)),
        "welfare_diff_residual": float(np.mean(np.asarray(residual) - wa)),
        "clearance_a": float(frame_a.loc[frame_a.epoch >= start, "mcr"].mean()),
        "clearance_b": float(fb["mcr"].mean() * len(sim_b.states) / elig),
        "gini_a": float(frame_a.loc[frame_a.epoch >= start, "gini"].mean()),
        "gini_b": float(fb["gini"].mean()),
        "n_eligible": float(elig),
    }

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default="R2-E15")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--window-start", type=int, default=250)
    parser.add_argument("--a-arms", default="oracle,ncf")
    parser.add_argument("--costs", default=",".join(map(str, COST_GRID)))
    parser.add_argument("--sizes", default=",".join(map(str, SIZE_GRID)))
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = out_dir / f"seed{args.seed}.json"
    if done.exists():
        print(f"{done} exists; skipping")
        return 0
    if not 0 <= args.window_start < args.rounds:
        raise SystemExit("--window-start must be inside [0, rounds)")

    with open(ROOT / "experiments" / "base.yaml", encoding="utf-8") as fh:
        base = apply_set(yaml.safe_load(fh), args.set)
    base["population"]["seed"] = int(args.seed)

    from src.python.config import SimConfig  # noqa: PLC0415
    from src.python.orchestrator import Simulation  # noqa: PLC0415
    resolved = SimConfig.coerce(base).to_dict()
    if int(resolved["ncf"]["num_threads"]) != 1:
        raise SystemExit("every quoted run is at ncf.num_threads = 1")
    record = JobRecord(ROOT, args.experiment, args.seed, resolved, sys.argv,
                       allow_dirty=args.allow_dirty)

    costs = [float(c) for c in args.costs.split(",")]
    sizes = [float(s) for s in args.sizes.split(",")]
    a_runs = {}
    for arm in args.a_arms.split(","):
        t0 = time.perf_counter()
        a_runs[arm] = run_a(base, arm, args.rounds, args.window_start)
        record.time(f"A_{arm}", time.perf_counter() - t0)

    # A must not depend on the cell. Checked on three rounds of the oracle arm:
    # the B block of the first cell's config, run as A, gives A's frame exactly.
    probe = copy.deepcopy(base)
    probe["environment_b"].update(size_fraction=sizes[0], signal_cost=costs[0])
    a_probe = Simulation(config=probe, recommender="oracle").run(3)
    a_base = Simulation(config=copy.deepcopy(base), recommender="oracle").run(3)
    if not a_probe["welfare_total"].equals(a_base["welfare_total"]):
        raise RuntimeError("Environment A depends on the B block; one A run cannot serve all cells")

    rows = []
    t0 = time.perf_counter()
    for cost in costs:
        for size in sizes:
            cfg = copy.deepcopy(base)
            cfg["environment_b"].update(size_fraction=size, signal_cost=cost)
            sim_b = Simulation(config=cfg, recommender="complete")
            sim_b.record_outcomes = True
            sim_b.record_outcomes_from = args.window_start
            frame_b = sim_b.run(args.rounds)
            for arm, (sim_a, frame_a) in a_runs.items():
                rows.append({"seed": args.seed, "a_arm": arm, "signal_cost": cost,
                             "size_fraction": size,
                             **pair_cell(sim_a, frame_a, sim_b, frame_b, args.window_start)})
            del sim_b, frame_b
        print(f"seed {args.seed} c={cost} done ({time.perf_counter() - t0:.0f}s)", flush=True)
    record.time("B_cells", time.perf_counter() - t0)

    write_csv_atomic(pd.DataFrame(rows), out_dir / f"seed{args.seed}.csv")
    record.finish(done, rounds=args.rounds, window=[args.window_start, args.rounds - 1],
                  a_arms=list(a_runs), costs=costs, sizes=sizes, overrides=args.set)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
