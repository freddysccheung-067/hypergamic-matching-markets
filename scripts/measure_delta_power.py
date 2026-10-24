"""Measure the delta-power overlap at each beta/alpha level actually swept.

The claim "below beta/alpha = 2 the delta term is inert by construction" is
documented in experiments/base.yaml and pinned by tests/cpp/test_delta_power.cpp
at a handful of settings. The sensitivity grid sweeps beta/alpha at 1.0, 2.0 and
3.0, so this script measures the overlap those cells were run at, on the same
population, rather than quoting a number from a different table.

The statistic is the top-k overlap of a woman's shortlist between delta = 0 and
delta = 1, holding everything else fixed. Overlap near 1 means delta reorders
nothing -- the hypergamous indicator is a constant across the shortlist, so a
result attributed to delta is really a result about the other knobs. The rule of
thumb the config states is that delta has power when beta/alpha >= 2.

Reads the built extension directly, so it needs build/Release on the path (the
orchestrator's loader does this).
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
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "build" / "Release"))

from _hypergamic import sample_population  # noqa: E402
from src.python.environment_b import utility_table  # noqa: E402


def overlap_at(cfg: dict, delta: float, top_k: int, side: str = "W") -> float:
    """Share of the top-k shortlist retained when delta moves from 0 to `delta`.

    Uses the utility table the simulation itself uses, so this measures the
    ordering the market would act on rather than a re-derivation of it.
    """
    base = copy.deepcopy(cfg)
    base["hypergamy"]["delta"] = 0.0
    other = copy.deepcopy(cfg)
    other["hypergamy"]["delta"] = delta

    n = int(base['population']['n'])
    seed = int(base['population']['seed'])
    states = sample_population(n, base, seed)
    u_mw0, u_wm0 = utility_table(states, base)
    u_mw1, u_wm1 = utility_table(states, other)

    # The W side is the acceptor side and that is where the hypergamous
    # indicator lives: U_wm[a, p] is acceptor a's utility toward proposer p, so
    # row a is one woman's ranking over men.
    if side == "W":
        m0, m1 = np.asarray(u_wm0, dtype=float), np.asarray(u_wm1, dtype=float)
    else:
        m0, m1 = np.asarray(u_mw0, dtype=float), np.asarray(u_mw1, dtype=float)
    n = min(m0.shape[0], m1.shape[0])
    k = min(top_k, m0.shape[1])
    overlaps = []
    for i in range(n):
        a = set(np.argsort(-m0[i])[:k].tolist())
        b = set(np.argsort(-m1[i])[:k].tolist())
        overlaps.append(len(a & b) / k)
    return float(np.mean(overlaps))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/base.yaml")
    parser.add_argument("--output", default="data/processed/rigor/sensitivity/delta_power.csv")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        base = yaml.safe_load(fh)

    rows = []
    for ratio in (1.0, 2.0, 3.0):
        cfg = copy.deepcopy(base)
        cfg["hypergamy"]["alpha"] = 1.0
        cfg["hypergamy"]["beta"] = float(ratio)
        for delta in (0.5, 1.0):
            ov = overlap_at(cfg, delta, args.top_k)
            rows.append({
                "beta_alpha": ratio,
                "alpha": 1.0,
                "beta": float(ratio),
                "delta": delta,
                "top_k": args.top_k,
                "overlap": ov,
                "delta_has_power": bool(ov < 0.90),
            })
            print(f"beta/alpha={ratio:.1f} delta={delta:.1f} overlap={ov:.4f} "
                  f"power={ov < 0.90}")

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
