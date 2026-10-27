"""Write the job list for one tier of the second revision's experiments.

    python scripts/make_jobs.py --tier 1 --worktree D:/hm_runs/wt_480cd6c
        --python D:/hypergamic-matching-markets/.venv/Scripts/python.exe
        --data-root D:/hypergamic-matching-markets/data/processed/horizon

Prints one JSON line per job, in the format ``scripts/run_queue.py`` reads. Every
job runs from ``--worktree``, a checkout of the commit its experiment was
registered at, and writes into ``--data-root``. The designs are those of
``docs/PREREGISTRATION.md`` R2-E9 to R2-E17; a change to one is a deviation and
is logged there first.
"""

from __future__ import annotations

import argparse
import json

EXPLORATORY = list(range(42, 62))
CONFIRMATION = list(range(100, 120))
TEN = list(range(100, 110))

# R2-E11: the eight off-shipped cells, as config overrides.
SWEEP = {
    "taste_sigma_0.2": ["population.taste_sigma=0.2"],
    "taste_sigma_1.0": ["population.taste_sigma=1.0"],
    "shock_scale_0.5": ["hypergamy.shock_scale=0.5"],
    "shock_scale_3.0": ["hypergamy.shock_scale=3.0"],
    "beta_alpha_1.0": ["hypergamy.alpha=1.0", "hypergamy.beta=1.0"],
    "beta_alpha_3.0": ["hypergamy.alpha=1.0", "hypergamy.beta=3.0"],
    "delta_0.0": ["hypergamy.delta=0.0"],
    "delta_1.0": ["hypergamy.delta=1.0"],
}

def horizon_job(args, tier: int, exp: str, sub: str, seed: int, rounds: int, arms: str,
                sets: list[str] | None = None, flags: list[str] | None = None) -> dict:
    out = f"{args.data_root}/{sub}"
    cmd = [args.python, "scripts/run_horizon.py", "--experiment", exp, "--seed", str(seed),
           "--rounds", str(rounds), "--arms", arms, "--output-dir", out]
    for s in sets or []:
        cmd += ["--set", s]
    cmd += flags or []
    return {"id": f"{sub.replace('/', '-')}-s{seed}", "cwd": args.worktree, "cmd": cmd,
            "done": f"{out}/seed{seed}.json", "priority": tier}


def tier1(args) -> list[dict]:
    jobs = []
    # R2-E9 first: everything else in the tier is compared against it.
    for seed in CONFIRMATION + EXPLORATORY:
        jobs.append(horizon_job(args, 1, "R2-E9", "e09_canonical", seed, 300,
                                "ncf,oracle,random,popularity", flags=["--diagnostics"]))
    for seed in CONFIRMATION + EXPLORATORY:
        jobs.append(horizon_job(args, 1, "R2-E13d", "e13d_warm", seed, 300, "ncf",
                                sets=["ncf.warm_start_epochs=50"]))
    for cell, sets in SWEEP.items():
        for seed in TEN:
            jobs.append(horizon_job(args, 1, "R2-E11", f"e11_sweep/{cell}", seed, 300,
                                    "ncf,oracle", sets=sets))
    return jobs


def tier2(args) -> list[dict]:
    jobs = []
    for seed in TEN:
        jobs.append(horizon_job(args, 2, "R2-E10", "e10_long", seed, 1000, "ncf,oracle"))
    for dose in ("0.5", "2", "4"):
        for seed in TEN:
            jobs.append(horizon_job(args, 2, "R2-E13a", f"e13a_dose/d{dose}", seed, 300, "ncf",
                                    sets=[f"ncf.passes_per_round={dose}"]))
    for share, prefix in (("0.05", 19), ("0.2", 16)):
        for seed in TEN:
            jobs.append(horizon_job(args, 2, "R2-E13b", f"e13b_explore/e{share}", seed, 300,
                                    "ncf", sets=[f"ncf.explore_share={share}"],
                                    flags=["--prefix", str(prefix)]))
            jobs.append(horizon_job(args, 2, "R2-E13b", f"e13b_explore/oracle_p{prefix}", seed,
                                    300, "oracle", flags=["--prefix", str(prefix)]))
    for seed in TEN:
        jobs.append(horizon_job(args, 2, "R2-E13c", "e13c_offpolicy", seed, 300, "oracle,random",
                                flags=["--shadow"]))
    root = args.data_root
    for seed in TEN:
        out = f"{root}/e15_institutional"
        jobs.append({"id": f"e15_institutional-s{seed}", "cwd": args.worktree, "priority": 2,
                     "cmd": [args.python, "scripts/run_institutional_horizon.py", "--seed",
                             str(seed), "--rounds", "300", "--window-start", "250",
                             "--output-dir", out],
                     "done": f"{out}/seed{seed}.json"})
        out = f"{root}/e15i_inert"
        jobs.append({"id": f"e15i_inert-s{seed}", "cwd": args.worktree, "priority": 2,
                     "cmd": [args.python, "scripts/run_institutional_horizon.py", "--experiment",
                             "R2-E15i", "--seed", str(seed), "--rounds", "5", "--window-start",
                             "0", "--a-arms", "oracle", "--costs", "0.01", "--set",
                             "population.admissible_fraction=0.2", "--output-dir", out],
                     "done": f"{out}/seed{seed}.json"})
        out = f"{root}/h5r"
        jobs.append({"id": f"h5r-s{seed}", "cwd": args.worktree, "priority": 2,
                     "cmd": [args.python, "scripts/run_appendix.py", "h5", "--seed", str(seed),
                             "--output-dir", out],
                     "done": f"{out}/seed{seed}.json"})
    out = f"{root}/h1r"
    jobs.append({"id": "h1r", "cwd": args.worktree, "priority": 2,
                 "cmd": [args.python, "scripts/run_appendix.py", "h1", "--seeds", "100-109",
                         "--output-dir", out],
                 "done": f"{out}/h1.json"})
    return jobs


def tier3(args) -> list[dict]:
    jobs = []
    for rate in ("0.005", "0.01", "0.02"):
        for seed in TEN:
            jobs.append(horizon_job(args, 3, "R2-E14", f"e14_arrivals/r{rate}", seed, 600,
                                    "ncf,oracle", flags=["--cohorts"],
                                    sets=[f"simulation.arrival_rate={rate}",
                                          f"simulation.departure_rate={rate}"]))
    for p in range(1, 11):
        for seed in range(100, 105):
            jobs.append(horizon_job(args, 3, "R2-E16", f"e16_fp/p{p:02d}", seed, 300, "ncf",
                                    sets=["ncf.init_perturbation=1e-7",
                                          f"ncf.perturbation_seed={p}"]))
    for n in (500, 2000):
        for seed in TEN:
            jobs.append(horizon_job(args, 3, "R2-E17", f"e17_size/n{n}", seed, 300, "ncf,oracle",
                                    sets=[f"population.n={n}",
                                          f"population.admissible_fraction={20 / n}"]))
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", type=int, required=True)
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    build = {1: tier1, 2: tier2, 3: tier3}[args.tier]
    for job in build(args):
        print(json.dumps(job))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
