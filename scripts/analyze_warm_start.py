"""Decay fractions for the warm-start intervention, computed four ways, per seed.

The warm-start arm (docs/PREREGISTRATION.md E3(ii)) pretrains the ranker before
the loop starts. The question it answers is how much of the control arm's decay
is still there when the ranker starts informed. "How much of the decay" has four
defensible readings, and an earlier revision printed one of them under the name
of another, so all four are computed here and written side by side:

    level, round-0 basis   (warm_0 - warm_F) / (control_0 - control_F)
    level, peak basis      (warm_P - warm_F) / (control_P - control_F)
    gap,   round-0 basis   the same on NCF minus oracle Gini
    gap,   peak basis      the same on NCF minus oracle Gini, each arm's own peak

A level is the NCF arm's exposure Gini. A gap is that level minus the oracle
arm's Gini on the same population, seed and round. F is the final round and P is
the round at which that arm's own series is highest.

The primary measure is the gap on the round-0 basis, per seed, with a Student-t
interval over seeds (docs/PREREGISTRATION.md E3d). Each ratio is computed per
seed and then averaged. The ratio of the five-seed mean series is written as
well, because an earlier draft quoted that construction, and the two differ in
the third decimal.

    python scripts/analyze_warm_start.py

Reads  data/processed/rigor/interventions/interventions_seed<seed>.csv
       data/processed/rigor/exploratory_1t/metrics_seed<seed>.csv  (oracle arm)
Writes data/processed/rigor/interventions/decay_fractions.csv          (per seed)
       data/processed/rigor/interventions/decay_fractions_summary.csv  (over seeds)

The control condition and the exploratory NCF arm are the same run: both are
the shipped loop at one thread on the same seed, and their Gini series agree to
every printed digit. The oracle Gini is read from the exploratory directory for
that reason; no oracle arm was run inside the intervention script.
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import statistics

from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parent.parent
RIGOR = ROOT / "data" / "processed" / "rigor"

MEASURES = ("level_round0", "level_peak", "gap_round0", "gap_peak")


def _series(path: pathlib.Path, key: str, value: str, column: str = "gini") -> dict[int, float]:
    out: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("condition", row.get("arm")) == value:
                out[int(row["epoch"])] = float(row[column])
    if not out:
        raise ValueError(f"{path.name}: no rows with {key} == {value!r}")
    return out


def per_seed(seed: int, interventions: pathlib.Path, oracle_dir: pathlib.Path) -> dict[str, float]:
    path = interventions / f"interventions_seed{seed}.csv"
    control = _series(path, "condition", "control")
    warm = _series(path, "condition", "warm")
    oracle = _series(oracle_dir / f"metrics_seed{seed}.csv", "arm", "oracle")
    final = max(control)
    if max(warm) != final or not set(control) <= set(oracle):
        raise ValueError(f"seed {seed}: the arms do not cover the same rounds")

    c_gap = {e: control[e] - oracle[e] for e in control}
    w_gap = {e: warm[e] - oracle[e] for e in warm}

    def peak(series: dict[int, float]) -> int:
        # Earliest round attaining the maximum, so ties resolve the same way on
        # every platform.
        top = max(series.values())
        return min(e for e, v in series.items() if v == top)

    row: dict[str, float] = {"seed": seed, "final_round": final}
    for name, c, w in (("level", control, warm), ("gap", c_gap, w_gap)):
        cp, wp = peak(c), peak(w)
        row.update({
            f"control_{name}_0": c[0], f"control_{name}_final": c[final],
            f"control_{name}_peak": c[cp], f"control_{name}_peak_round": cp,
            f"warm_{name}_0": w[0], f"warm_{name}_final": w[final],
            f"warm_{name}_peak": w[wp], f"warm_{name}_peak_round": wp,
            f"{name}_round0": (w[0] - w[final]) / (c[0] - c[final]),
            f"{name}_peak": (w[wp] - w[final]) / (c[cp] - c[final]),
        })
    row["oracle_0"] = oracle[0]
    row["oracle_final"] = oracle[final]
    row["terminal_diff"] = warm[final] - control[final]
    return row


def summarize(rows: list[dict[str, float]], columns: list[str]) -> list[dict[str, float]]:
    n = len(rows)
    tcrit = float(stats.t.ppf(0.975, n - 1))
    out = []
    for col in columns:
        values = [float(r[col]) for r in rows]
        mean = statistics.fmean(values)
        sd = statistics.stdev(values)
        half = tcrit * sd / math.sqrt(n)
        out.append({
            "quantity": col, "n_seeds": n, "mean": mean, "sd": sd,
            "lo_t95": mean - half, "hi_t95": mean + half,
            "t_stat": mean / (sd / math.sqrt(n)) if sd > 0 else float("nan"),
            "min": min(values), "max": max(values),
        })
    return out


def pooled(rows: list[dict[str, float]], interventions: pathlib.Path,
           oracle_dir: pathlib.Path) -> list[dict[str, float]]:
    """The four ratios on the seed-mean series rather than per seed.

    This is the construction an earlier draft used for the level, peak-basis
    figure. It is kept so the difference between it and the per-seed mean is on
    the record, not because it is preferred.
    """
    seeds = [int(r["seed"]) for r in rows]
    by_cond: dict[str, dict[int, list[float]]] = {"control": {}, "warm": {}, "oracle": {}}
    for seed in seeds:
        path = interventions / f"interventions_seed{seed}.csv"
        for cond in ("control", "warm"):
            for e, v in _series(path, "condition", cond).items():
                by_cond[cond].setdefault(e, []).append(v)
        for e, v in _series(oracle_dir / f"metrics_seed{seed}.csv", "arm", "oracle").items():
            by_cond["oracle"].setdefault(e, []).append(v)
    mean = {k: {e: statistics.fmean(v) for e, v in d.items()} for k, d in by_cond.items()}
    final = max(mean["control"])
    rounds = sorted(mean["control"])
    out = []
    for name in ("level", "gap"):
        def s(cond: str, e: int) -> float:
            base = mean[cond][e]
            return base - mean["oracle"][e] if name == "gap" else base
        c = {e: s("control", e) for e in rounds}
        w = {e: s("warm", e) for e in rounds}
        cp = max(c, key=c.get)
        wp = max(w, key=w.get)
        out.append({"quantity": f"{name}_round0_pooled",
                    "value": (w[0] - w[final]) / (c[0] - c[final])})
        out.append({"quantity": f"{name}_peak_pooled",
                    "value": (w[wp] - w[final]) / (c[cp] - c[final])})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--interventions", default=str(RIGOR / "interventions"))
    parser.add_argument("--oracle-dir", default=str(RIGOR / "exploratory_1t"))
    args = parser.parse_args()

    interventions = pathlib.Path(args.interventions)
    oracle_dir = pathlib.Path(args.oracle_dir)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rows = [per_seed(s, interventions, oracle_dir) for s in seeds]

    per_seed_path = interventions / "decay_fractions.csv"
    with per_seed_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_cols = [*MEASURES,
                    "control_level_0", "warm_level_0", "control_level_final", "warm_level_final",
                    "control_gap_0", "warm_gap_0", "control_gap_final", "warm_gap_final",
                    "control_gap_peak", "warm_gap_peak", "control_level_peak", "warm_level_peak",
                    "oracle_0", "oracle_final", "terminal_diff"]
    summary = summarize(rows, summary_cols)
    summary_path = interventions / "decay_fractions_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
        # Appended as rows of the same file, with only `quantity` and `mean` set,
        # so the pooled construction is beside the per-seed one it differs from.
        for p in pooled(rows, interventions, oracle_dir):
            writer.writerow({"quantity": p["quantity"], "n_seeds": len(rows), "mean": p["value"]})

    for r in summary:
        if r["quantity"] in MEASURES:
            print(f"{r['quantity']:<14} mean {r['mean']:.4f}  sd {r['sd']:.4f}  "
                  f"95% t [{r['lo_t95']:.4f}, {r['hi_t95']:.4f}]  "
                  f"range [{r['min']:.4f}, {r['max']:.4f}]")
    for p in pooled(rows, interventions, oracle_dir):
        print(f"{p['quantity']:<22} {p['value']:.4f}")
    print(f"wrote {per_seed_path.name}, {summary_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
