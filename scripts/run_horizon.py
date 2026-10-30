"""One seed of one second-revision experiment: every arm, every round, one file.

    python scripts/run_horizon.py --experiment R2-E9 --seed 100 --rounds 300 \
        --arms ncf,oracle,random,popularity --diagnostics \
        --output-dir data/processed/horizon/e09_canonical

Writes ``seed<seed>.csv`` (one row per arm and round) and ``seed<seed>.json``
(the provenance record of ``src/python/provenance.py``: commit, cleanliness,
config and its hash, versions, wall time). The JSON is written last and is the
job's done-marker: a job whose JSON exists is complete and is skipped on rerun.

Every arm on a seed shares the population, the utility parameters and the
arrival stream, so any difference between two arms' rows at one seed and round
is a paired observation.

Options that change the design go through ``--set key=value`` on the config
(``--set ncf.passes_per_round=2``, ``--set population.n=500``), so the change is
inside the config hash. Options that only change what is recorded
(``--diagnostics``, ``--shadow``, ``--prefix``, ``--cohorts``) are not part of the
design and are listed in the record instead.

Columns, per arm and round:

    gini reach top1_share mcr rsi algorithmic_loss welfare_alg n_matched
        the §6 metrics of that round
    mean_swipes_per_agent grad_steps
        data and optimizer steps the market's ranker consumed (0 for an arm that
        does not learn); grad_steps is cumulative
    spearman_* item_share* modal_share* identical_share* jaccard_oracle*
        R2-E12 diagnostics, NCF arm with --diagnostics
    shadow_gini shadow_reach shadow_grad_steps
        R2-E13c, learning-free arms with --shadow
    gini_prefix
        exposure Gini over the first P entries of every list, --prefix P
    cohort_<band>_pop cohort_<band>_exposure
        R2-E14, with --cohorts: share of agents in a tenure band and share of
        that round's exposure they received
"""

from __future__ import annotations

import argparse
import copy
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.python.provenance import JobRecord, write_csv_atomic  # noqa: E402

FRAME_COLUMNS = ["gini", "reach", "top1_share", "mcr", "rsi", "algorithmic_loss",
                 "welfare_alg", "n_matched", "mean_swipes_per_agent"]
# Tenure bands for the open market, in rounds since arrival: [lo, hi).
COHORTS = (("t000_009", 0, 10), ("t010_049", 10, 50), ("t050_up", 50, 10**9))


def parse_set(pairs: list[str]) -> list[tuple[list[str], object]]:
    out = []
    for item in pairs:
        key, _, raw = item.partition("=")
        if not key or not _:
            raise SystemExit(f"--set expects key=value; got {item!r}")
        out.append((key.split("."), yaml.safe_load(raw)))
    return out


def apply_set(cfg: dict, updates: list[tuple[list[str], object]]) -> dict:
    for path, value in updates:
        node = cfg
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = value
    return cfg


def exposure_from_lists(prop: np.ndarray, acc: np.ndarray, prop_rows: np.ndarray,
                        acc_rows: np.ndarray, n: int, prefix: int | None = None) -> np.ndarray:
    """Row-indexed exposure from padded dense-slot lists, optionally a prefix only."""
    if prefix is not None:
        prop, acc = prop[:, :prefix], acc[:, :prefix]
    exposure = np.zeros(n, dtype=np.int64)
    flat = prop[prop >= 0]
    np.add.at(exposure, acc_rows[flat], 1)
    flat = acc[acc >= 0]
    np.add.at(exposure, prop_rows[flat], 1)
    return exposure


def cohort_rows(sim, rounds: list[int]) -> list[dict[str, float]]:
    """Per round, each tenure band's share of agents and of exposure."""
    arrival = np.zeros(sim.n, dtype=np.int64)
    events = iter(sim.arrivals_log)
    pending = next(events, None)
    out = []
    for t, (prop, acc) in zip(rounds, sim.lists_log):
        while pending is not None and pending["epoch"] <= t:
            arrival[pending["rows"]] = pending["epoch"]
            pending = next(events, None)
        tenure = t - arrival
        exposure = exposure_from_lists(prop, acc, sim.prop_rows, sim.acc_rows, sim.n)
        total = exposure.sum()
        row: dict[str, float] = {}
        for name, lo, hi in COHORTS:
            mask = (tenure >= lo) & (tenure < hi)
            row[f"cohort_{name}_pop"] = float(mask.mean())
            row[f"cohort_{name}_exposure"] = float(exposure[mask].sum() / total) if total else np.nan
        out.append(row)
    return out


def run_arm(cfg: dict, arm: str, rounds: int, args) -> pd.DataFrame:
    from src.python.metrics import gini  # noqa: PLC0415
    from src.python.orchestrator import Simulation  # noqa: PLC0415

    sim = Simulation(config=copy.deepcopy(cfg), recommender=arm,
                     shadow_ranker=bool(args.shadow and arm != "ncf"))
    sim.record_diagnostics = bool(args.diagnostics and arm == "ncf")
    sim.record_lists = args.prefix is not None or args.cohorts
    frame = sim.run(rounds)

    table = frame[["epoch", *FRAME_COLUMNS]].rename(columns={"epoch": "round"}).copy()
    log = pd.DataFrame(sim.round_log).rename(columns={"epoch": "round"})
    table = table.merge(log.drop(columns=["n_swipes"]), on="round", how="left")
    if args.prefix is not None:
        table["gini_prefix"] = [
            gini(exposure_from_lists(p, a, sim.prop_rows, sim.acc_rows, sim.n, args.prefix))
            for p, a in sim.lists_log
        ]
    if args.cohorts:
        table = pd.concat([table, pd.DataFrame(cohort_rows(sim, list(table["round"])))],
                          axis=1)
    table.insert(0, "arm", arm)
    table.insert(1, "seed", int(cfg["population"]["seed"]))
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--arms", default="ncf,oracle")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--config", default="experiments/base.yaml")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--shadow", action="store_true")
    parser.add_argument("--prefix", type=int, default=None)
    parser.add_argument("--cohorts", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-dirty", action="store_true",
                        help="run from an uncommitted tree; for smoke tests only")
    parser.add_argument("--force", action="store_true", help="rerun a completed seed")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = out_dir / f"seed{args.seed}.json"
    if done.exists() and not args.force:
        print(f"{done} exists; skipping")
        return 0

    with open(ROOT / args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg = apply_set(cfg, parse_set(args.set))
    cfg["population"]["seed"] = int(args.seed)

    from src.python.config import SimConfig  # noqa: PLC0415
    resolved = SimConfig.coerce(cfg).to_dict()
    if int(resolved["ncf"]["num_threads"]) != 1:
        raise SystemExit("every quoted run is at ncf.num_threads = 1 (docs/INTERFACES.md §3)")
    if args.cohorts and float(resolved["simulation"].get("arrival_rate", 0.0)) == 0.0:
        raise SystemExit("--cohorts needs simulation.arrival_rate > 0")

    record = JobRecord(ROOT, args.experiment, args.seed, resolved, sys.argv,
                       allow_dirty=args.allow_dirty)
    tables = []
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        t0 = time.perf_counter()
        tables.append(run_arm(cfg, arm, args.rounds, args))
        record.time(arm, time.perf_counter() - t0)
        last = tables[-1].iloc[-1]
        print(f"{args.experiment} seed {args.seed} {arm:<10} round {int(last['round'])} "
              f"gini {last['gini']:.4f} reach {last['reach']:.3f} "
              f"({record.record['wall_seconds'][arm]:.0f}s)", flush=True)

    table = pd.concat(tables, ignore_index=True)
    write_csv_atomic(table, out_dir / f"seed{args.seed}.csv")
    record.finish(done, rounds=args.rounds, arms=args.arms.split(","), overrides=args.set,
                  recorders={"diagnostics": args.diagnostics, "shadow": args.shadow,
                             "prefix": args.prefix, "cohorts": args.cohorts})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
