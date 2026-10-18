"""Parameter sweep over hypergamy coefficient δ."""
import argparse
import copy
import pathlib
import sys

import yaml
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/base.yaml")
    parser.add_argument("--delta-values", default="0.0,0.5,1.0")
    parser.add_argument("--output-dir", default="data/processed/")
    args = parser.parse_args()

    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    deltas = [float(x) for x in args.delta_values.split(",")]
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for delta in tqdm(deltas, desc="Sweep"):
        cfg = copy.deepcopy(base_config)
        cfg["hypergamy"]["delta"] = delta

        # Import here so C++ extension is already built when this runs
        from src.python.orchestrator import Simulation  # noqa: PLC0415

        sim = Simulation(config=cfg)
        metrics = sim.run(n_epochs=cfg["simulation"]["n_epochs"])

        out_path = output_dir / f"sweep_delta_{delta:.2f}.csv"
        metrics.to_csv(out_path, index=False)
        tqdm.write(f"delta={delta:.2f} → {out_path}")


if __name__ == "__main__":
    main()
