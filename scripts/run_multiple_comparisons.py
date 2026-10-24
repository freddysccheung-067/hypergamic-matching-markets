"""§5: the institutional optimum was selected from 200 cells. Does it survive?

`data/processed/institutional/grid.csv` holds 10 size_fraction levels x 4
signal_cost levels x 5 seeds = 200 (cell, seed) rows. The claim "the optimum is
at (c = 0.01, size_fraction = 0.65) with +64.62 ± 9.86" was picked by looking at
all of them. That is a selection over 40 cells (the 200 rows are 40 distinct
cells times 5 seeds), and the number quoted carries no correction for it.

What this script does:

1. **The family.** Count the cells tested. The family is the 40 distinct
   (signal_cost, size_fraction) cells — a cell is the unit that was selected
   over, and its five seeds are replicates inside it, not five more chances to
   win. The family-wise error rate at α = 0.05 uncorrected is
   `1 − (1 − α)^40` if the tests were independent; the cells are not
   independent (adjacent size_fractions share populations), so that number is
   reported as the nominal upper bound and the realised rate is not claimed.

2. **Per-cell test.** For each cell, a one-sample t test of
   `welfare_diff_residual` against zero, over its five seeds. This is the test
   that says "does B beat A in this cell", which is the quantity the optimum was
   selected on.

3. **Benjamini–Hochberg** q-values over the 40 cell p-values, and the set that
   survives at q ≤ 0.05.

4. **The optimum cell itself**, with the correction's effect on its interval:
   the point estimate is unchanged (BH adjusts decisions, not estimates), but
   the interval that can be quoted after selecting the maximum of 40 cells is
   the *simultaneous* one — Bonferroni at 40 tests, or equivalently the
   (1 − α/40) percentile of the bootstrap. Both are reported so the reader sees
   how much the selection costs.

5. **The H5 interaction**, the same treatment on the factorial's single term.

Outputs:
    sensitivity_summary.csv      per-cell estimate, t, p, q, rank
    multiple_comparisons.csv     the scalar results STATS.md §5 quotes
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.python.stats import bca_interval, t_interval  # noqa: E402

BOOTSTRAP_SEED = 20261024
RESAMPLES = 10_000
ALPHA = 0.05


def benjamini_hochberg(pvals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """BH step-up: returns (q_values, rejected_at_alpha) in the input order.

    The step-up works on ranks: sort ascending, find the largest k with
    `p_(k) ≤ k/m · α`, reject everything up to k, then enforce monotonicity on the
    q-values (`q_(i) = min over j ≥ i`). Enforcing monotonicity is not cosmetic —
    without it a cell with a smaller p can end up with a larger q, which is not a
    valid adjusted p-value.
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    q_sorted = ranked * m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q = np.empty(m)
    q[order] = q_sorted
    rejected = q <= ALPHA
    return q, rejected


def holm(pvals: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Holm step-down rejection flags, for comparison against BH.

    Reported because the two answer different questions: BH controls the false
    discovery rate (the expected share of quoted cells that are wrong), Holm
    controls the family-wise error rate (the chance of any wrong quoted cell).
    For a paper that will name one optimum, FWER is the stricter and more
    appropriate control, so both are given.
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresholds = alpha / np.arange(m, 0, -1)
    below = ranked <= thresholds
    # Step-down: reject the first k for the largest k with all of 1..k below.
    k = 0
    for i in range(m):
        if below[i]:
            k = i + 1
        else:
            break
    flags = np.zeros(m, dtype=bool)
    flags[order[:k]] = True
    return flags


def per_cell(grid: pd.DataFrame, value: str) -> pd.DataFrame:
    """One row per (signal_cost, size_fraction): mean, sd, t, p over its seeds."""
    rows = []
    for (cost, size), grp in grid.groupby(["signal_cost", "size_fraction"]):
        x = grp[value].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        n = x.size
        if n < 2:
            rows.append(dict(signal_cost=cost, size_fraction=size, n_seeds=n,
                             mean=float(x.mean()) if n else np.nan,
                             sd=np.nan, se=np.nan, t=np.nan, p=np.nan,
                             df=np.nan))
            continue
        m = float(x.mean())
        sd = float(x.std(ddof=1))
        se = sd / np.sqrt(n)
        t = m / se if se > 0 else np.nan
        p = float(2.0 * st.t.sf(abs(t), df=n - 1)) if np.isfinite(t) else np.nan
        rows.append(dict(signal_cost=cost, size_fraction=size, n_seeds=n,
                         mean=m, sd=sd, se=se, t=t, p=p, df=n - 1))
    return pd.DataFrame(rows).sort_values(["signal_cost", "size_fraction"],
                                          ignore_index=True)


def simultaneous_interval(values: np.ndarray, m: int) -> dict:
    """Bonferroni-adjusted interval: the bootstrap at (1 − α/m) percentile.

    After choosing the maximum of m cells, an interval that is still valid for
    "this is the best cell" has to cover with probability 1 − α across all m.
    The point estimate does not move — the selection does not bias the *value*
    at the selected cell if that cell is genuinely the best — but the interval
    around it must widen, and this is how much.
    """
    p_family = ALPHA / m
    iv = bca_interval(values, n_resamples=RESAMPLES, alpha=p_family,
                      rng=BOOTSTRAP_SEED)
    _, lo_t, hi_t = t_interval(values, alpha=p_family)
    return {
        "point": iv.point,
        "lo_bonferroni": iv.lo,
        "hi_bonferroni": iv.hi,
        "lo_t_bonferroni": lo_t,
        "hi_t_bonferroni": hi_t,
        "family_alpha": p_family,
        "n_seeds": iv.n_seeds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", default="data/processed/institutional/grid.csv")
    parser.add_argument("--factorial", default="data/processed/institutional/factorial.csv")
    parser.add_argument("--output-dir", default="data/processed/rigor")
    parser.add_argument("--value", default="welfare_diff_residual")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = pd.read_csv(args.grid)
    cells = per_cell(grid, args.value)
    m = len(cells)

    q, rejected_bh = benjamini_hochberg(cells["p"].to_numpy())
    rejected_holm = holm(cells["p"].to_numpy())
    cells["q_bh"] = q
    cells["reject_bh"] = rejected_bh
    cells["reject_holm"] = rejected_holm
    cells["rank_by_mean"] = cells["mean"].rank(ascending=False, method="min").astype(int)
    cells.to_csv(out_dir / "sensitivity_summary.csv", index=False)

    # The cell the thesis names. Selected by maximum of the 40 means.
    best_by_mean = cells.loc[cells["mean"].idxmax()]
    named = cells[(cells.signal_cost == 0.01) & (cells.size_fraction == 0.65)]
    if named.empty:
        raise SystemExit("the named optimum cell (0.01, 0.65) is not in the grid")
    named = named.iloc[0]

    named_values = grid[(grid.signal_cost == 0.01)
                        & (grid.size_fraction == 0.65)][args.value].to_numpy(dtype=float)
    named_plain = bca_interval(named_values, n_resamples=RESAMPLES, rng=BOOTSTRAP_SEED)
    named_sim = simultaneous_interval(named_values, m)

    # Nominal FWER if the m tests were independent. The cells share populations,
    # so this is an upper bound on the dependence-adjusted rate, and it is
    # labelled as such wherever it is quoted.
    fwer_indep = 1.0 - (1.0 - ALPHA) ** m

    scalar_rows = [
        ("n_cells_tested", float(m)),
        ("n_seed_replicates_total", float(len(grid))),
        ("alpha", ALPHA),
        ("fwer_uncorrected_independent", fwer_indep),
        ("n_reject_bh_q05", float(rejected_bh.sum())),
        ("n_reject_holm_05", float(rejected_holm.sum())),
        ("optimum_signal_cost", float(best_by_mean.signal_cost)),
        ("optimum_size_fraction", float(best_by_mean.size_fraction)),
        ("optimum_mean", float(best_by_mean["mean"])),
        ("optimum_sd", float(best_by_mean["sd"])),
        ("optimum_t", float(best_by_mean["t"])),
        ("optimum_p", float(best_by_mean["p"])),
        ("optimum_q_bh", float(best_by_mean["q_bh"])),
        ("optimum_reject_bh", float(best_by_mean["reject_bh"])),
        ("named_cell_mean", float(named["mean"])),
        ("named_cell_sd", float(named["sd"])),
        ("named_cell_t", float(named["t"])),
        ("named_cell_p", float(named["p"])),
        ("named_cell_q_bh", float(named["q_bh"])),
        ("named_cell_reject_bh", float(named["reject_bh"])),
        ("named_cell_reject_holm", float(named["reject_holm"])),
        ("named_cell_rank_by_mean", float(named["rank_by_mean"])),
        ("named_cell_lo_bca", named_plain.lo),
        ("named_cell_hi_bca", named_plain.hi),
        ("named_cell_lo_bonferroni", named_sim["lo_bonferroni"]),
        ("named_cell_hi_bonferroni", named_sim["hi_bonferroni"]),
        ("named_cell_lo_t_bonferroni", named_sim["lo_t_bonferroni"]),
        ("named_cell_hi_t_bonferroni", named_sim["hi_t_bonferroni"]),
    ]

    # H5's interaction, same treatment. n = 5 seeds is one test, so no correction
    # within it; the family here is the three decomposition terms if all three
    # are quoted, which is what BH is applied over.
    fac_path = pathlib.Path(args.factorial)
    if fac_path.is_file():
        fac = pd.read_csv(fac_path)
        terms = ["exogenous", "algorithmic", "interaction"]
        pvals = []
        for term in terms:
            x = fac[term].to_numpy(dtype=float)
            t_res = st.ttest_1samp(x, 0.0)
            pvals.append(float(t_res.pvalue))
        pvals_arr = np.asarray(pvals)
        q_terms, rej_terms = benjamini_hochberg(pvals_arr)
        for i, term in enumerate(terms):
            x = fac[term].to_numpy(dtype=float)
            iv = bca_interval(x, n_resamples=RESAMPLES, rng=BOOTSTRAP_SEED)
            _, lo_t, hi_t = t_interval(x)
            t_res = st.ttest_1samp(x, 0.0)
            scalar_rows += [
                (f"{term}_mean", float(x.mean())),
                (f"{term}_sd", float(x.std(ddof=1))),
                (f"{term}_t", float(t_res.statistic)),
                (f"{term}_p", float(t_res.pvalue)),
                (f"{term}_q_bh", float(q_terms[i])),
                (f"{term}_reject_bh", float(rej_terms[i])),
                (f"{term}_lo_bca", iv.lo),
                (f"{term}_hi_bca", iv.hi),
                (f"{term}_lo_t", lo_t),
                (f"{term}_hi_t", hi_t),
            ]

    scalar = pd.DataFrame(scalar_rows, columns=["quantity", "value"])
    scalar.to_csv(out_dir / "multiple_comparisons.csv", index=False)

    print(f"── {m} cells, {len(grid)} rows ──")
    print(f"FWER uncorrected (independent upper bound): {fwer_indep:.4f}")
    print(f"BH q<=0.05 rejects {rejected_bh.sum()} of {m}; "
          f"Holm rejects {rejected_holm.sum()}")
    print(f"\nbest cell by mean: c={best_by_mean.signal_cost} "
          f"sf={best_by_mean.size_fraction} mean={best_by_mean['mean']:+.2f} "
          f"t={best_by_mean['t']:.2f} p={best_by_mean['p']:.2g} "
          f"q={best_by_mean['q_bh']:.2g}")
    print(f"named cell: mean={named['mean']:+.2f} ± {named['sd']:.2f} "
          f"t={named['t']:.2f} p={named['p']:.3g} q={named['q_bh']:.3g} "
          f"reject_bh={named['reject_bh']} rank={int(named['rank_by_mean'])}")
    print(f"  BCa interval        [{named_plain.lo:+.2f}, {named_plain.hi:+.2f}]")
    print(f"  Bonferroni interval [{named_sim['lo_bonferroni']:+.2f}, "
          f"{named_sim['hi_bonferroni']:+.2f}]")
    print(f"\nwrote {out_dir / 'sensitivity_summary.csv'} and "
          f"{out_dir / 'multiple_comparisons.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
