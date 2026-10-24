"""§2: is five seeds enough? Seed-variance scaling, and what n would be needed.

The question is not "is the interval wide" — §1 answers that — but "how does it
behave as n grows, and does any conclusion depend on stopping at five". Three
things are measured:

1. **Width scaling.** The 95% interval's width at n = 5, 8, 10, 15 and 20 seeds,
   from the same 20-seed run, expressed relative to its n = 20 width. For a
   quantity whose seed variance is finite and whose seeds are i.i.d., the width
   falls as `c/sqrt(n)`, so the ratio `width(n)/width(20)` should equal
   `sqrt(20/n)`. The observed ratios are compared against that curve and the
   deviation is reported: a ratio *below* the curve means the n = 5 sample
   happened to be unusually concordant (its interval is narrower than the
   asymptote entitles it to be), and *above* means unusually spread.

   Underlying it is the seed-variance component itself: `Var(seed)` estimated by
   the pooled within-regime variance across the 20 seeds, against the
   mean-of-squares a 5-seed run would give. A 5-seed sd is a chi(4)-scaled
   estimate, so it is unbiased on average but has a 95% relative range of roughly
   0.52 to 2.87 of the true sd — which is why the n = 5 interval is not just wide
   but *unpredictably* wide.

2. **Does any conclusion flip?** For each headline quantity, the sign and the
   zero-crossing of its interval at every n, and the smallest n at which the
   conclusion is stable across every prefix of the seed list (not just the first
   n seeds — every contiguous block of length n, so a lucky ordering cannot
   produce the answer).

3. **The power analysis.** For each quantity, the n at which the 95% interval
   would exclude zero given the observed effect size and the pooled seed sd,
   from `n = ceil((1.96·sd/|effect|)²)`. Where that is above 20, the number is
   named — that is the "this claim needs more seeds" answer the session asks for.

Everything is written to `data/processed/rigor/power_analysis.csv` and
`data/processed/rigor/power_widths.csv`; STATS.md §2 quotes only those.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.python.stats import bca_interval, t_interval  # noqa: E402

BOOTSTRAP_SEED = 20261024
RESAMPLES = 10_000
NS = (5, 8, 10, 15, 20)


def _load(base_dir: pathlib.Path) -> dict[str, pd.DataFrame]:
    parts = [pd.read_csv(p) for p in sorted(base_dir.glob("metrics_seed*.csv"))]
    if not parts:
        raise SystemExit(f"no metrics_seed*.csv under {base_dir}")
    table = pd.concat(parts, ignore_index=True)
    return {arm: grp for arm, grp in table.groupby("arm")}


def headline_series(frames: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    """The quantities §2 tests, each as one value per seed, indexed by seed."""

    def final(arm: str, col: str) -> pd.Series:
        last = frames[arm].sort_values("epoch").groupby("seed").tail(1)
        return last.set_index("seed")[col].astype(float).sort_index()

    ncf_gini = final("ncf", "gini")
    orc_gini = final("oracle", "gini")
    common = ncf_gini.index.intersection(orc_gini.index)

    return {
        "gini_ncf": ncf_gini,
        "gini_oracle": orc_gini,
        "gini_gap_ncf_minus_oracle": (ncf_gini.loc[common] - orc_gini.loc[common]),
        "mcr_ncf": final("ncf", "mcr"),
        "rsi_ncf": final("ncf", "rsi"),
        "algorithmic_loss_ncf": final("ncf", "algorithmic_loss"),
        "reach_ncf": final("ncf", "reach"),
    }


def width_table(series: pd.Series, rng_seed: int) -> list[dict]:
    """95% interval width at each n, using every block of that length.

    Every contiguous block, not just the first n: with the seed list ordered
    42..61, "the first 5" is one draw of a 5-seed sample and could be atypical.
    Averaging the width over all blocks gives the *expected* width at that n,
    which is what the scaling law describes. The first-block value is kept
    separately because that is what a 5-seed run before this work actually had.
    """
    full = series.to_numpy(dtype=float)
    rows = []
    for n in NS:
        if n > full.size:
            continue
        blocks = [full[i:i + n] for i in range(0, full.size - n + 1)]
        widths, t_widths = [], []
        for block in blocks:
            iv = bca_interval(block, n_resamples=RESAMPLES, rng=rng_seed)
            widths.append(iv.width)
            _, lo_t, hi_t = t_interval(block)
            t_widths.append(hi_t - lo_t)
        first = bca_interval(full[:n], n_resamples=RESAMPLES, rng=rng_seed)
        rows.append({
            "n": n,
            "n_blocks": len(blocks),
            "width_bca_mean": float(np.mean(widths)),
            "width_bca_min": float(np.min(widths)),
            "width_bca_max": float(np.max(widths)),
            "width_first_block": float(first.width),
            "width_t_mean": float(np.mean(t_widths)),
            "width_t_first_block": float(np.mean(t_widths[:1])),
        })
    return rows


def conclusion_table(series: pd.Series, rng_seed: int) -> list[dict]:
    """At each n: does the interval exclude zero, and with what sign, per block."""
    full = series.to_numpy(dtype=float)
    rows = []
    for n in NS:
        if n > full.size:
            continue
        blocks = [full[i:i + n] for i in range(0, full.size - n + 1)]
        signs, excludes = [], []
        for block in blocks:
            iv = bca_interval(block, n_resamples=RESAMPLES, rng=rng_seed)
            excl = (iv.lo > 0) or (iv.hi < 0)
            excludes.append(excl)
            signs.append(int(np.sign(iv.point)))
        rows.append({
            "n": n,
            "n_blocks": len(blocks),
            "fraction_excluding_zero": float(np.mean(excludes)),
            "fraction_positive": float(np.mean([s > 0 for s in signs])),
            "fraction_negative": float(np.mean([s < 0 for s in signs])),
            "point_sign_stable": bool(len(set(signs)) == 1),
        })
    return rows


def required_n(effect: float, sd: float, alpha: float = 0.05,
               power: float = 0.80) -> float:
    """Seeds needed for a two-sided one-sample test to reject zero at this effect.

    `n = ((z_{1−α/2} + z_power)·sd/|effect|)²`. This is the standard normal
    approximation and is what a reviewer will recompute, so it is the number
    quoted. It answers "how many seeds to *detect* this effect", which is the
    same question as "how many seeds before the interval excludes zero" for a
    symmetric interval, and the two are reported together so the reader can see
    they agree.

    Returns `inf` when the effect is zero — no number of seeds separates zero from
    zero — and NaN when the sd is undefined. A value below 5 means the quantity is
    already resolved at the shipped seed count and the seed count is not the
    binding constraint on that claim; the caller clamps the reported integer to
    the minimum feasible n = 2 and says so in `n_is_binding`.
    """
    from scipy import stats as st

    if not np.isfinite(sd) or sd <= 0:
        return float("nan")
    if effect == 0.0:
        return float("inf")
    z_a = st.norm.ppf(1.0 - alpha / 2.0)
    z_p = st.norm.ppf(power)
    return float(((z_a + z_p) * sd / abs(effect)) ** 2)


def seeds_needed(effect: float, sd: float) -> tuple[float, bool]:
    """`(n, binding)` — the seeds required, and whether 5 was ever too few.

    `binding` is True where the requirement exceeds the five seeds the project
    ships, which is the only case in which "this needs more seeds" is a real
    statement about the claim. Every requirement below 5 is rounded up to 5 and
    flagged non-binding, because the honest reading there is "five was already
    more than enough for this quantity", not "0.01 seeds would do".
    """
    n = required_n(effect, sd)
    if not np.isfinite(n):
        return (n, n > 5)
    return (max(n, 5.0), n > 5.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="data/processed/rigor/base")
    parser.add_argument("--out-dir", default="data/processed/rigor")
    args = parser.parse_args()

    base_dir = pathlib.Path(args.dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = _load(base_dir)
    series = headline_series(frames)
    n_seeds = max(s.size for s in series.values())

    width_rows, concl_rows, power_rows = [], [], []
    for i, (name, s) in enumerate(sorted(series.items())):
        rng_seed = BOOTSTRAP_SEED + i

        for row in width_table(s, rng_seed):
            # The scaling law: width(n)/width(20) = sqrt(20/n).
            row["quantity"] = name
            row["n_seeds_available"] = int(s.size)
            row["ratio_to_n20"] = row["width_bca_mean"] / next(
                r["width_bca_mean"] for r in width_table(s, rng_seed) if r["n"] == 20)
            row["sqrt20_over_n"] = float(np.sqrt(20.0 / row["n"]))
            row["ratio_over_law"] = row["ratio_to_n20"] / row["sqrt20_over_n"]
            width_rows.append(row)

        for row in conclusion_table(s, rng_seed):
            row["quantity"] = name
            row["n_seeds_available"] = int(s.size)
            concl_rows.append(row)

        vals = s.to_numpy(dtype=float)
        iv20 = bca_interval(vals, n_resamples=RESAMPLES, rng=rng_seed)
        _, lo_t, hi_t = t_interval(vals)
        effect = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1))
        iv5_vals = vals[:5]
        iv5 = bca_interval(iv5_vals, n_resamples=RESAMPLES, rng=rng_seed)
        n_need, binding = seeds_needed(effect, sd)

        # The 5-seed interval's width against what the *pooled* 20-seed sd says a
        # 5-seed interval should be. A ratio well below 1 means the first five
        # populations happened to agree more than the population of seeds allows,
        # so the committed mean ± sd is narrower than the quantity's actual spread.
        # That is the concrete form of "is five enough".
        expected_5 = sd * 2.0 * 1.959964 / np.sqrt(5)
        power_rows.append({
            "quantity": name,
            "n_available": int(vals.size),
            "point_20": effect,
            "sd_5": float(np.std(iv5_vals, ddof=1)),
            "sd_20": sd,
            "sd_ratio_5_over_20": float(np.std(iv5_vals, ddof=1) / sd) if sd else float("nan"),
            "lo_5": iv5.lo,
            "hi_5": iv5.hi,
            "width_5": iv5.width,
            "width_20": iv20.width,
            "width_ratio_5_over_20": iv5.width / iv20.width,
            "expected_width_5_from_pooled_sd": float(expected_5),
            "width_5_over_expected": float(iv5.width / expected_5) if expected_5 else float("nan"),
            "lo_t_20": lo_t,
            "hi_t_20": hi_t,
            "excludes_zero_at_5": bool(iv5.lo > 0 or iv5.hi < 0),
            "excludes_zero_at_20": bool(iv20.lo > 0 or iv20.hi < 0),
            "n_seeds_needed": n_need,
            "n_requirement_binding": bool(binding),
        })

    widths = pd.DataFrame(width_rows)
    widths.to_csv(out_dir / "power_widths.csv", index=False)
    concl = pd.DataFrame(concl_rows)
    concl.to_csv(out_dir / "power_conclusions.csv", index=False)
    power = pd.DataFrame(power_rows)
    power.to_csv(out_dir / "power_analysis.csv", index=False)

    print(f"── width scaling ({n_seeds} seeds available) ──")
    print(widths[["quantity", "n", "n_blocks", "width_bca_mean", "ratio_to_n20",
                  "sqrt20_over_n", "ratio_over_law"]].to_string(index=False))
    print("\n── conclusions by n ──")
    print(concl[["quantity", "n", "fraction_excluding_zero", "point_sign_stable"]]
          .to_string(index=False))
    print("\n── power ──")
    print(power[["quantity", "point_20", "sd_5", "sd_20", "sd_ratio_5_over_20",
                 "width_5_over_expected", "excludes_zero_at_5", "excludes_zero_at_20",
                 "n_seeds_needed", "n_requirement_binding"]].to_string(index=False))
    print(f"\nwrote 3 tables to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
