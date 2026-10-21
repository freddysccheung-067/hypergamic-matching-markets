"""Parameter sweep over hypergamy coefficient δ, and optionally over heterogeneity.

THEORY.md §2.3 reports results as a function of taste_sigma and shock_scale, never
at one setting, so both are sweep axes. Each defaults to the config's value; a
filename gains a suffix only for an axis that was actually swept, so a δ-only
sweep writes the same files it always did.
"""
import argparse
import copy
import itertools
import pathlib
import sys

import yaml
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


def _floats(text):
    return [float(x) for x in text.split(",")] if text else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/base.yaml")
    parser.add_argument("--delta-values", default="0.0,0.5,1.0")
    parser.add_argument("--taste-sigma", default=None,
                        help="comma-separated population.taste_sigma values")
    parser.add_argument("--shock-scale", default=None,
                        help="comma-separated hypergamy.shock_scale values")
    parser.add_argument("--output-dir", default="data/processed/")
    args = parser.parse_args()

    # utf-8: base.yaml contains em dashes, which the Windows default codec rejects.
    with open(args.config, encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    deltas = [float(x) for x in args.delta_values.split(",")]
    sigmas = _floats(args.taste_sigma)
    shocks = _floats(args.shock_scale)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = list(itertools.product(deltas, sigmas or [None], shocks or [None]))
    for delta, sigma, shock in tqdm(grid, desc="Sweep"):
        cfg = copy.deepcopy(base_config)
        cfg["hypergamy"]["delta"] = delta
        name = f"sweep_delta_{delta:.2f}"
        if sigma is not None:
            cfg["population"]["taste_sigma"] = sigma
            name += f"_sigma_{sigma:.2f}"
        if shock is not None:
            cfg["hypergamy"]["shock_scale"] = shock
            name += f"_shock_{shock:.2f}"

        # Import here so C++ extension is already built when this runs
        from src.python.orchestrator import Simulation  # noqa: PLC0415

        sim = Simulation(config=cfg)
        metrics = sim.run(n_epochs=cfg["simulation"]["n_epochs"])

        # The heterogeneity setting and the noise-to-signal ratio it implies at this
        # population, on every row: shock_scale is in absolute utility units, so
        # the ratio is what makes rows from different alpha/beta comparable.
        metrics["taste_sigma"] = float(cfg["population"].get("taste_sigma", 0.0))
        metrics["shock_scale"] = float(cfg["hypergamy"].get("shock_scale", 0.0))
        metrics["noise_ratio"] = sim.noise_ratio

        out_path = output_dir / f"{name}.csv"
        metrics.to_csv(out_path, index=False)
        tqdm.write(f"{name} (nsr={sim.noise_ratio:.3f}) → {out_path}")


if __name__ == "__main__":
    main()
