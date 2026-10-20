"""Thin-market curve for Environment B (THEORY.md §10.5).

Writes ``data/processed/thin_market/curve_seed<seed>.csv`` per seed and prints the
table with the sizes at which clearance crosses 0.5. Crossings are linearly
interpolated between grid points, and reported as both a size fraction and |B|.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "build" / "Release"))

GRID = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30,
        0.35, 0.40, 0.50, 0.60, 0.80, 1.00]


def crossings(frame: pd.DataFrame, level: float = 0.5) -> list[tuple[str, float, float]]:
    """(direction, size_fraction, eligible_n) where clearance_prob crosses ``level``."""
    x = frame["size_fraction"].to_numpy()
    n = frame["eligible_n"].to_numpy()
    y = frame["clearance_prob"].to_numpy()
    out = []
    for i in range(len(y) - 1):
        if (y[i] - level) * (y[i + 1] - level) < 0:
            t = (level - y[i]) / (y[i + 1] - y[i])
            out.append(("up" if y[i + 1] > y[i] else "down",
                        float(x[i] + t * (x[i + 1] - x[i])),
                        float(n[i] + t * (n[i + 1] - n[i]))))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--bar-quantile", type=float, default=None)
    parser.add_argument("--output-dir", default="data/processed/thin_market")
    args = parser.parse_args()

    from src.python.config import SimConfig  # noqa: PLC0415
    from src.python.environment_b import thin_market_curve  # noqa: PLC0415
    from src.python.orchestrator import Simulation  # noqa: PLC0415

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = SimConfig.from_yaml().to_dict()

    frames = []
    for seed in (int(s) for s in args.seeds.split(",") if s.strip()):
        d = {**base, "population": {**base["population"], "seed": seed}}
        cfg = SimConfig.from_dict(d)
        sim = Simulation(config=cfg, recommender="complete")
        curve = thin_market_curve(
            sim.states, sim.config_dict, cfg.require_environment_b(), GRID,
            n_replicates=args.replicates, table=sim._utility_table,
            bar_quantile=args.bar_quantile,
        )
        curve.insert(0, "seed", seed)
        curve.to_csv(out_dir / f"curve_seed{seed}.csv", index=False)
        frames.append(curve)
        print(f"seed {seed}: " + "; ".join(
            f"{d_} at frac {f:.3f} (|B|={n:.0f})" for d_, f, n in crossings(curve)))

    allc = pd.concat(frames)
    mean = allc.groupby("size_fraction").mean(numeric_only=True).reset_index()
    sd = allc.groupby("size_fraction")["clearance_prob"].std().to_numpy()
    mean["clearance_sd"] = sd
    pd.set_option("display.width", 220)
    print(mean.drop(columns="seed").round(4).to_string(index=False))
    print("mean-curve crossings of 0.5:",
          "; ".join(f"{d_} at frac {f:.3f} (|B|={n:.0f})" for d_, f, n in crossings(mean)))
    mean.to_csv(out_dir / "curve_mean.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
