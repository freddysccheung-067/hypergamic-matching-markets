"""Prepare the E7 comparison: committed vs regenerated 300-epoch trajectory.

Run once `data/processed/rigor/long_horizon_1t/` is complete. Prints the numbers
`docs/PREREGISTRATION.md` E7's decision rule needs, and the rule's verdict, so the
comparison is mechanical rather than a matter of reading two series side by side.

    python scripts/_compare_long_horizon.py
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent
RIGOR = ROOT / "data" / "processed" / "rigor"
SEEDS = (42, 43, 44, 45, 46)
TAIL = range(250, 300)


def read(path: Path) -> dict[str, dict[int, float]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return {r["arm"]: {int(r["epoch"]): float(r["gini"]) for r in rows} for r in rows}


def gaps(directory: str) -> dict[int, list[float]]:
    """Per-epoch gap for each seed: {seed: [gap by epoch]}."""
    out: dict[int, list[float]] = {}
    for seed in SEEDS:
        ncf = read(RIGOR / directory / f"ncf_seed{seed}.csv")["ncf"]
        oracle = read(RIGOR / directory / f"oracle_seed{seed}.csv")["oracle"]
        out[seed] = [ncf[e] - oracle[e] for e in sorted(ncf)]
    return out


def summarise(name: str, per_seed: dict[int, list[float]]) -> dict[str, float]:
    n = len(next(iter(per_seed.values())))
    tail_means = [mean(per_seed[s][e] for e in TAIL) for s in SEEDS]
    final = [per_seed[s][n - 1] for s in SEEDS]
    at99 = [per_seed[s][99] for s in SEEDS]
    peak_epoch = max(range(n), key=lambda e: mean(per_seed[s][e] for s in SEEDS))
    print(f"\n{name}: {n} epochs, {len(SEEDS)} seeds")
    print(f"  peak epoch {peak_epoch}, mean gap {mean(per_seed[s][peak_epoch] for s in SEEDS):+.5f}")
    print(f"  epoch 99   mean {mean(at99):+.5f}  sd {stdev(at99):.5f}")
    print(f"  final      mean {mean(final):+.5f}  sd {stdev(final):.5f}")
    print(f"  tail 250-299 mean {mean(tail_means):+.6f}  sd over seeds {stdev(tail_means):.6f}")
    return {
        "epoch99": mean(at99),
        "epoch99_sd": stdev(at99),
        "final": mean(final),
        "final_sd": stdev(final),
        "tail": mean(tail_means),
        "tail_sd": stdev(tail_means),
    }


def main() -> int:
    committed = summarise("committed rigor/long_horizon", gaps("long_horizon"))
    regen = summarise("regenerated rigor/long_horizon_1t", gaps("long_horizon_1t"))

    print("\n=== E7 decision rule ===")
    tail_diff = abs(regen["tail"] - committed["tail"])
    print(f"  |tail difference| = {tail_diff:.6f}   (< 0.001 required for branch 1)")
    sd = committed["epoch99_sd"]
    print(f"  |epoch-99 difference| = {abs(regen['epoch99'] - committed['epoch99']):.6f}")
    print(f"  committed epoch-99 seed sd = {sd:.6f}; one sd holds if the difference < {sd:.6f}")

    if tail_diff < 0.001 and abs(regen["epoch99"] - committed["epoch99"]) < sd:
        print("\n  BRANCH 1: the two agree at the resolution the paper claims.")
        print("  Switch the quoted trajectory to long_horizon_1t and say the")
        print("  committed directory differs.")
    else:
        print("\n  BRANCH 2: the two disagree materially.")
        print("  Report long_horizon_1t as the measurement, state that the")
        print("  committed directory cannot be reproduced and disagrees, and do")
        print("  not average them or choose the one that supports a conclusion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
