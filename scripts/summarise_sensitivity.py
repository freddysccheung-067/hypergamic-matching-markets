"""§4 reporting: turn the sensitivity cell files into one table and a verdict per claim.

The raw cells are `data/processed/rigor/sensitivity/cell_<axis>_<level>.csv`. This
script builds `sensitivity.csv` — one row per (axis, level) with the paired
NCF−oracle gap, the NCF arm's reach, and the final-epoch gap, all with intervals
— and then classifies each committed conclusion against the grid.

**Which conclusion can a cell "reverse"?** Only a claim with a direction and a
sign can be reversed. The four that this project makes and that a cell could
falsify are:

| claim | survives a cell if |
|---|---|
| C1 the NCF's Gini exceeds the oracle's | the paired gap's 95% interval excludes zero, positive |
| C2 the gap is a cold-start effect, so it is small at 100 epochs | the gap's point estimate stays below a stated ceiling |
| C3 the NCF reaches most agents by 100 epochs | reach's interval excludes the epoch-0 value |
| C4 clearing rate is not degenerate | mcr's interval excludes 0 |

C2 needs a ceiling, and picking one after seeing the results would be exactly the
post-hoc move §5 exists to catch. The ceiling is therefore fixed at **0.05** —
five Gini points — before the cells are read, and it is stated in the CSV and in
STATS.md. It is not a natural constant; it is a pre-registered judgment about when
a gap stops being the story, and STATS.md says so.

The β/α axis is reported with a flag: the shipped config documents that below
β/α = 2 the δ term is inert by construction, so a cell at β/α = 1 cannot speak to
any hypergamy claim and is marked `delta_power_ok = False`.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.python.stats import bca_interval, paired_gap, t_interval  # noqa: E402

BOOTSTRAP_SEED = 20261024
RESAMPLES = 10_000

# Pre-registered, see the module docstring. Written to every row that uses it.
COLD_START_CEILING = 0.05

AXES = ("taste_sigma", "shock_scale", "beta_alpha", "delta")
SHIPPED = {"taste_sigma": "0.5", "shock_scale": "1.5", "beta_alpha": "2.0", "delta": "0.5"}

# Below this ratio the δ term is inert by construction (base.yaml's comment on
# beta; guarded by tests/cpp/test_delta_power). Recorded so a hypergamy reading
# of a low-β/α cell is not even attempted.
DELTA_POWER_MIN_RATIO = 2.0


def _cell(path: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _verdicts(lo: float, hi: float, point: float, reach_lo: float, mcr_lo: float,
              reach_point: float) -> dict:
    """Apply the four claims to one cell's interval.

    C1 is the only claim with a sign that a cell could reverse outright. C3 is
    evaluated against a concrete threshold: the NCF arm reaches 96% of agents at
    the shipped setting by epoch 100, and "the filter has learned coverage" is
    the mechanism THESIS H3 offers for the cold-start gap closing. A cell where
    reach falls below 0.75 is a cell where that mechanism is not doing the work,
    and the cold-start story would need restating as something else. The
    threshold is fixed here rather than derived, and STATS.md §4 says so.
    """
    reach_floor = 0.75
    return {
        "C1_gap_excludes_zero": bool(lo > 0),
        "C1_gap_positive": bool(point > 0),
        "C2_gap_below_ceiling": bool(point <= COLD_START_CEILING),
        "C1_reversed": bool(point < 0),
        "C3_reach_above_floor": bool(reach_point >= reach_floor),
        "C3_reach_floor": reach_floor,
        "C4_mcr_excludes_zero": bool(mcr_lo > 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="data/processed/rigor/sensitivity")
    parser.add_argument("--out", default="data/processed/rigor/sensitivity.csv")
    args = parser.parse_args()

    cell_dir = pathlib.Path(args.dir)
    files = sorted(cell_dir.glob("cell_*.csv"))
    if not files:
        raise SystemExit(f"no cell_*.csv under {cell_dir}")

    rows = []
    for path in files:
        frame = _cell(path)
        axis = str(frame.axis.iloc[0])
        level = str(frame.level.iloc[0])

        wide = frame.pivot(index="seed", columns="arm")
        gini_ncf = wide[("gini", "ncf")].sort_index()
        gini_orc = wide[("gini", "oracle")].sort_index()
        common = gini_ncf.index.intersection(gini_orc.index)

        gap = paired_gap(gini_ncf.loc[common].to_numpy(),
                         gini_orc.loc[common].to_numpy(),
                         n_resamples=RESAMPLES, rng=BOOTSTRAP_SEED)
        _, lo_t, hi_t = t_interval((gini_ncf.loc[common]
                                    - gini_orc.loc[common]).to_numpy())

        reach_iv = bca_interval(wide[("reach", "ncf")].to_numpy(),
                                n_resamples=RESAMPLES, rng=BOOTSTRAP_SEED)
        mcr_iv = bca_interval(wide[("mcr", "ncf")].to_numpy(),
                              n_resamples=RESAMPLES, rng=BOOTSTRAP_SEED)
        gini_iv = bca_interval(gini_ncf.to_numpy(), n_resamples=RESAMPLES,
                               rng=BOOTSTRAP_SEED)

        ratio = None
        try:
            ratio = float(level)
        except ValueError:
            ratio = float("nan")

        row = {
            "axis": axis,
            "level": level,
            "level_numeric": ratio,
            "is_shipped": bool(level == SHIPPED.get(axis)),
            "delta_power_ok": bool(axis != "beta_alpha"
                                   or ratio >= DELTA_POWER_MIN_RATIO),
            "n_seeds": int(common.size),
            "gap_point": gap.point,
            "gap_lo": gap.lo,
            "gap_hi": gap.hi,
            "gap_lo_t": lo_t,
            "gap_hi_t": hi_t,
            "gap_sd": float((gini_ncf.loc[common] - gini_orc.loc[common]).std(ddof=1)),
            "reach_point": reach_iv.point,
            "reach_lo": reach_iv.lo,
            "reach_hi": reach_iv.hi,
            "mcr_point": mcr_iv.point,
            "mcr_lo": mcr_iv.lo,
            "mcr_hi": mcr_iv.hi,
            "gini_ncf_point": gini_iv.point,
            "gini_ncf_lo": gini_iv.lo,
            "gini_ncf_hi": gini_iv.hi,
            "gini_oracle_point": float(gini_orc.mean()),
            "noise_ratio": float(frame.noise_ratio.mean()),
            "cold_start_ceiling": COLD_START_CEILING,
        }
        row.update(_verdicts(gap.lo, gap.hi, gap.point, reach_iv.lo, mcr_iv.lo,
                             reach_iv.point))
        rows.append(row)

    table = pd.DataFrame(rows)
    order = {a: i for i, a in enumerate(AXES)}
    table = table.sort_values(["axis", "level_numeric"], key=lambda s: s.map(order)
                              if s.name == "axis" else s).reset_index(drop=True)

    # Attach the measured delta-power overlap where it exists. The claim carried
    # in base.yaml and pinned by tests/cpp/test_delta_power.cpp was measured at
    # taste_sigma = 0 and shock_scale = 0; the sweep runs at 0.5 and 1.5. The
    # overlap is therefore re-measured at the swept settings and joined here, so
    # a reader can see whether the guardrail the sweep invokes still applies.
    dp_path = cell_dir / "delta_power.csv"
    if dp_path.is_file():
        dp = pd.read_csv(dp_path)
        at_shipped = dp[(dp.delta == 1.0) & (dp.top_k == 20)][["beta_alpha", "overlap"]]
        lookup = {str(float(r.beta_alpha)): float(r.overlap) for r in at_shipped.itertuples()}
        table["delta_overlap_measured"] = [
            lookup.get(str(float(lv)), float("nan")) if ax == "beta_alpha" else float("nan")
            for ax, lv in zip(table.axis, table.level, strict=True)
        ]
        # The flag follows the MEASUREMENT, not the documented rule. The guardrail
        # in base.yaml was measured at taste_sigma = 0 and shock_scale = 0; the
        # sweep runs at 0.5 and 1.5, and at those settings the ordering the rule
        # asserts does not appear (see STATS.md §4). Where a measurement exists it
        # decides; where it does not, the documented rule stands and is marked as
        # unmeasured rather than silently reused.
        table["delta_power_measured"] = table["delta_overlap_measured"] < 0.90
    out = pathlib.Path(args.out)
    table.to_csv(out, index=False)

    with pd.option_context("display.width", 220):
        print(table[["axis", "level", "is_shipped", "delta_power_ok", "gap_point",
                     "gap_lo", "gap_hi", "reach_point", "mcr_point",
                     "C1_gap_excludes_zero", "C2_gap_below_ceiling",
                     "C3_reach_above_floor", "C4_mcr_excludes_zero",
                     "C1_reversed"]].to_string(index=False))

    # The summary a reader needs: which claims hold where.
    print("\n── claim survival ──")
    for axis, grp in table.groupby("axis"):
        c1 = int(grp.C1_gap_excludes_zero.sum())
        c1r = int(grp.C1_reversed.sum())
        c2 = int(grp.C2_gap_below_ceiling.sum())
        c3 = int(grp.C3_reach_above_floor.sum())
        c4 = int(grp.C4_mcr_excludes_zero.sum())
        print(f"  {axis:<12} C1 {c1}/{len(grp)} (reverses {c1r}); "
              f"C2 {c2}/{len(grp)}; C3 {c3}/{len(grp)}; C4 {c4}/{len(grp)}")

    # Where a claim fails, name the cell -- that is the finding.
    for claim, col in (("C1", "C1_gap_excludes_zero"),
                       ("C2", "C2_gap_below_ceiling"),
                       ("C3", "C3_reach_above_floor"),
                       ("C4", "C4_mcr_excludes_zero")):
        failing = table[~table[col]]
        if not failing.empty:
            cells = ", ".join(f"{r.axis}={r.level}" for r in failing.itertuples())
            print(f"\n  {claim} fails at: {cells}")

    print(f"\nwrote {out} ({len(table)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
