"""The institutional comparison of THEORY.md §10.4-§10.6 (Session 5-E).

Three outputs, one per question, each over seeds 42-46 by default:

    data/processed/institutional/grid.csv       (size_fraction x signal_cost), values_min
                                                fixed at the shipped 0.3
    data/processed/institutional/values_min.csv (values_min), size_fraction and c_B fixed
    data/processed/institutional/factorial.csv  the four cells of §10.6 and the three terms

plus a ``*_summary.csv`` beside the first two with mean and sd over seeds. Nothing
is reported at a single seed.

**Why values_min is its own axis.** It interacts with size_fraction through the
compatibility CDF (THEORY.md §10.5): a stricter filter thins B in count and raises
its mean alignment at the same time, so a joint grid would confound the two. The
values_min axis is run at one size_fraction, and the interaction is described in
THEORY.md rather than estimated here.

**Per cell, per seed:** A and B are run on one population (comparison.paired_run,
which raises if they are not), for ``--epochs`` epochs, and each welfare number is
the mean over those epochs. Under the oracle and complete arms the matching does
not move between epochs at these settings (measured: identical n_matched at every
epoch), so the epochs differ only in the defection draw and the mean is over that.
A is the oracle arm, so the A-B difference is between institutions and not
inflated by the NCF's estimation error.

The difference is formed per seed and then summarised, never as a difference of
means: two environments at one seed share a population.
"""

from __future__ import annotations

import argparse
import copy
import logging
import pathlib
import sys
import time

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "build" / "Release"))

SIZE_GRID = [0.02, 0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 1.00]
COST_GRID = [0.01, 0.05, 0.10, 0.25]
VALUES_MIN_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

METRICS = [
    "welfare_common_a", "welfare_common_b", "welfare_diff",
    "welfare_common_residual_b", "welfare_diff_residual",
    "welfare_selected_b", "welfare_selection_artifact",
    "clearance_a", "clearance_b", "gini_a", "gini_b",
    "value_created_b", "signal_cost_b", "defection_loss_b",
    "n_eligible", "n_double_booked",
]


def _with(base: dict, seed: int, **env_b) -> dict:
    d = copy.deepcopy(base)
    d["population"]["seed"] = int(seed)
    d["environment_b"].update(env_b)
    return d


def _cell(cfg: dict, epochs: int) -> dict[str, float]:
    from src.python.comparison import paired_run  # noqa: PLC0415

    r = paired_run(cfg, epochs)
    fa, fb = r.frame_a, r.frame_b
    elig = r.sim_b.subgraph.n_eligible
    return {
        "welfare_common_a": float(fa.welfare_common.mean()),
        "welfare_common_b": float(fb.welfare_common.mean()),
        "welfare_diff": float((fb.welfare_common - fa.welfare_common).mean()),
        "welfare_common_residual_b": float(r.welfare_common_residual.mean()),
        "welfare_diff_residual": float((r.welfare_common_residual - fa.welfare_common).mean()),
        "welfare_selected_b": float(fb.welfare_selected.mean()),
        "welfare_selection_artifact": float(fb.welfare_selection_artifact.mean()),
        # Clearance over each market's own members: A's mcr is over N, B's over |B|.
        # B's frame mcr is over N (an excluded agent counts as unmatched there), so it
        # is rescaled here to the share of *entrants* matched.
        "clearance_a": float(fa.mcr.mean()),
        "clearance_b": float(fb.mcr.mean() * len(r.sim_b.states) / elig),
        "gini_a": float(fa.gini.mean()),
        "gini_b": float(fb.gini.mean()),
        "value_created_b": float(fb.value_created.mean()),
        "signal_cost_b": float(fb.proposal_cost_total.mean()),
        "defection_loss_b": float(fb.defection_loss.mean()),
        "n_eligible": float(elig),
        "n_double_booked": float(r.n_double_booked.mean()),
    }


def summarise(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    g = frame.groupby(keys)[METRICS]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_sd"))
    out["n_seeds"] = frame.groupby(keys).size()
    return out.reset_index()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--epochs", type=int, default=5,
                        help="epochs per paired run; welfare is averaged over them")
    parser.add_argument("--factorial-epochs", type=int, default=40)
    parser.add_argument("--factorial-window", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.02,
                        help="status drift per epoch, in epoch-0 status sd")
    parser.add_argument("--values-min-size", type=float, default=0.5)
    parser.add_argument("--values-min-cost", type=float, default=0.01)
    parser.add_argument("--only", choices=("grid", "values_min", "factorial"), default=None)
    parser.add_argument("--output-dir", default="data/processed/institutional")
    args = parser.parse_args()

    # The balancing warning fires once per paired run; the drop is in n_eligible.
    logging.basicConfig(level=logging.ERROR)

    from src.python.config import SimConfig  # noqa: PLC0415
    from src.python.factorial import run_factorial  # noqa: PLC0415

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if len(seeds) < 5:
        print(f"warning: {len(seeds)} seeds; the summaries below need at least five",
              file=sys.stderr)
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = SimConfig.from_yaml().to_dict()
    t0 = time.time()

    if args.only in (None, "grid"):
        rows = []
        for c in COST_GRID:
            for sf in SIZE_GRID:
                for seed in seeds:
                    rows.append({"signal_cost": c, "size_fraction": sf, "seed": seed,
                                 **_cell(_with(base, seed, size_fraction=sf, signal_cost=c),
                                         args.epochs)})
            print(f"grid c={c} done ({time.time() - t0:.0f}s)", flush=True)
        grid = pd.DataFrame(rows)
        grid.to_csv(out / "grid.csv", index=False)
        summarise(grid, ["signal_cost", "size_fraction"]).to_csv(out / "grid_summary.csv",
                                                                index=False)

    if args.only in (None, "values_min"):
        rows = []
        for vm in VALUES_MIN_GRID:
            for seed in seeds:
                cfg = _with(base, seed, size_fraction=args.values_min_size,
                            signal_cost=args.values_min_cost, values_min=vm)
                rows.append({"values_min": vm, "seed": seed, **_cell(cfg, args.epochs)})
        vmf = pd.DataFrame(rows)
        vmf.to_csv(out / "values_min.csv", index=False)
        summarise(vmf, ["values_min"]).to_csv(out / "values_min_summary.csv", index=False)
        print(f"values_min done ({time.time() - t0:.0f}s)", flush=True)

    if args.only in (None, "factorial"):
        rows = []
        for seed in seeds:
            cfg = copy.deepcopy(base)
            cfg["population"]["seed"] = seed
            res = run_factorial(cfg, args.gamma, args.factorial_epochs, args.factorial_window)
            rows.append({"seed": seed, "gamma": res.gamma,
                         **{f"W_{k}": v for k, v in res.cells.items()},
                         "exogenous": res.exogenous, "algorithmic": res.algorithmic,
                         "interaction": res.interaction})
            print(f"factorial seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
        fac = pd.DataFrame(rows)
        fac.to_csv(out / "factorial.csv", index=False)
        terms = fac[["W_baseline", "W_drift", "W_ncf", "W_both",
                     "exogenous", "algorithmic", "interaction"]]
        print(pd.DataFrame({"mean": terms.mean(), "sd": terms.std(ddof=1),
                            "min": terms.min(), "max": terms.max()}).round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
