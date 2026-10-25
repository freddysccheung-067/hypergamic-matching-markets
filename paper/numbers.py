"""Generate ``paper/tables/numbers.tex``: one LaTeX macro per named quantity.

The tables in this paper already regenerate from ``data/processed/``; the prose
did not, which is how the paper came to hold three different values for one
horizon. This script closes that gap: every number the prose quotes becomes a
``\\newcommand`` reading a committed CSV, so a stale figure cannot survive a
rebuild.

    python paper/numbers.py

Writes ``paper/tables/numbers.tex``. ``main.tex`` inputs it in the preamble, so
every section can use the macros below without an extra ``\\input``.

Conventions, so that a macro name says what it contains:

  \\gapShipped          a point estimate, signed, 4 places
  \\gapShippedLo        a lower bound, signed, 4 places
  \\giniNcf             a level (not a difference), unsigned
  \\optWelfare          a welfare number, signed, 2 places (its own scale)
  \\h4Q                 a p- or q-value, 3 significant figures, scientific
  \\nSweepCells         a count, no decimal
  \\reachShipped        a fraction in [0, 1], 4 places

Which run each reading comes from is in the comment beside it, because two
directories can hold the same quantity and the paper must say which.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RIGOR = PROCESSED / "rigor"
OUT = Path(__file__).resolve().parent / "tables" / "numbers.tex"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def regime_row(quantity: str, regime: str) -> dict[str, str]:
    for row in read_csv(RIGOR / "intervals.csv"):
        if row["quantity"] == quantity and row["regime"] == regime:
            return row
    raise KeyError(f"{quantity} @ {regime}")


def mc_values() -> dict[str, float]:
    return {
        row["quantity"]: float(row["value"])
        for row in read_csv(RIGOR / "multiple_comparisons.csv")
    }


def asymptote() -> dict[str, float]:
    rows = read_csv(RIGOR / "long_horizon_asymptote.csv")
    if not rows:
        raise ValueError("long_horizon_asymptote.csv is empty")
    # `window` is the window length in epochs and `epochs` is the range label
    # ("250-299"); only the former is a number, so the range is skipped here and
    # read separately where the prose needs it.
    skip = ("quantity", "regime", "window", "epochs", "method")
    return {k: float(v) for k, v in rows[0].items() if k not in skip}


def stationarity() -> dict[str, float]:
    rows = read_csv(RIGOR / "long_horizon_stationarity.csv")
    if not rows:
        raise ValueError("long_horizon_stationarity.csv is empty")
    return {k: float(v) for k, v in rows[0].items()
            if k not in ("arm", "window_early", "window_late")}


def sensitivity_rows() -> list[dict[str, str]]:
    return read_csv(RIGOR / "sensitivity.csv")


def shipped_cell() -> dict[str, str]:
    for row in sensitivity_rows():
        if row["is_shipped"] == "True":
            return row
    raise KeyError("no shipped row in sensitivity.csv")


def gap_range() -> tuple[float, float]:
    rows = sensitivity_rows()
    points = [float(r["gap_point"]) for r in rows]
    return min(points), max(points)


def sci(value: float) -> str:
    """LaTeX scientific notation, mantissa to two places."""
    if value == 0.0:
        return "0"
    exp = 0
    m = value
    while abs(m) >= 10:
        m /= 10
        exp += 1
    while abs(m) < 1:
        m *= 10
        exp -= 1
    return rf"{m:.2f}\times10^{{{exp}}}"


def gap_series() -> dict[int, float]:
    """The five-seed mean gap by epoch, from rigor/long_horizon_gap_series.csv."""
    rows = read_csv(RIGOR / "long_horizon_gap_series.csv")
    return {int(r["epoch"]): float(r["gap_mean"]) for r in rows}


def twenty_seed(directory: str) -> dict[str, dict[str, float]]:
    """Mean, sd and 95% t bounds over the twenty seeds in a rigor subdirectory.

    Reads the per-seed ``metrics_seed<N>.csv`` files written by
    ``scripts/run_rigor_base.py`` and takes the final epoch of each arm. A gap is
    formed per seed and then averaged, never as a difference of arm means: the
    arms inside one file share a population, so pairing is what makes the gap an
    observation rather than a comparison of two independent samples.

    Returns ``{quantity: {"mean", "sd", "lo", "hi"}}``. The t interval is the one
    to read at n = 20 (``docs/STATS.md`` §1), so only that construction is
    returned; the sd is included because the fresh-seed paragraph compares spreads
    between the two sets and that comparison should not be a hand-typed number.
    """
    files = sorted((RIGOR / directory).glob("metrics_seed*.csv"))
    if not files:
        raise FileNotFoundError(
            f"no metrics_seed*.csv in data/processed/rigor/{directory}; run "
            f"scripts/run_rigor_base.py for that directory first"
        )

    per_seed: dict[str, list[float]] = {
        "gini_ncf": [], "gini_oracle": [], "gini_gap": [], "alg_loss": [],
    }
    for path in files:
        rows = read_csv(path)
        last = max(int(r["epoch"]) for r in rows)
        final = {r["arm"]: r for r in rows if int(r["epoch"]) == last}
        ncf = float(final["ncf"]["gini"])
        oracle = float(final["oracle"]["gini"])
        per_seed["gini_ncf"].append(ncf)
        per_seed["gini_oracle"].append(oracle)
        per_seed["gini_gap"].append(ncf - oracle)
        if "algorithmic_loss" in final["ncf"]:
            per_seed["alg_loss"].append(float(final["ncf"]["algorithmic_loss"]))

    # Two-sided 95% t at n = 20 (19 df).
    tcrit = 2.093
    out: dict[str, dict[str, float]] = {}
    for key, values in per_seed.items():
        if not values:
            continue
        n = len(values)
        m = sum(values) / n
        var = sum((v - m) ** 2 for v in values) / (n - 1)
        se = (var / n) ** 0.5
        out[key] = {"mean": m, "sd": var ** 0.5,
                    "lo": m - tcrit * se, "hi": m + tcrit * se}
    return out


def main() -> None:
    live = regime_row("gini_gap_ncf_minus_oracle", "live100")
    base = regime_row("gini_gap_ncf_minus_oracle", "base20")
    long_gap = regime_row("gini_gap_ncf_minus_oracle", "long300")
    gini_ncf = regime_row("gini_ncf", "live100")
    gini_oracle = regime_row("gini_oracle", "live100")
    mcr = regime_row("mcr_ncf", "live100")
    rsi = regime_row("rsi_ncf", "live100")

    mc = mc_values()
    asym = asymptote()
    stat = stationarity()
    shipped = shipped_cell()
    lo_point, hi_point = gap_range()

    def f(key: str, places: int = 4, signed: bool = False) -> str:
        value = float(key)
        return f"{value:+.{places}f}" if signed else f"{value:.{places}f}"

    lines = [
        "% Generated by paper/numbers.py from data/processed/. Do not edit by hand.",
        "% Every macro here reads a committed CSV; the comment names the source.",
        "",
        "% ── The H3 gap at each horizon ──────────────────────────────────────────",
        "% live100: data/processed/rigor/intervals.csv (ablation_h100, 24 threads, 5 seeds)",
        rf"\newcommand{{\gapShipped}}{{{f(live['point'], 4, True)}}}",
        rf"\newcommand{{\gapShippedLo}}{{{f(live['lo_t'], 4, True)}}}",
        rf"\newcommand{{\gapShippedHi}}{{{f(live['hi_t'], 4, True)}}}",
        "% base20: the same quantity at 20 seeds, 1 thread",
        rf"\newcommand{{\gapBaseTwenty}}{{{f(base['point'], 4, True)}}}",
        rf"\newcommand{{\gapBaseTwentyLo}}{{{f(base['lo_t'], 4, True)}}}",
        rf"\newcommand{{\gapBaseTwentyHi}}{{{f(base['hi_t'], 4, True)}}}",
        "% base20 BCa bounds, for the full-width comparison of A1",
        rf"\newcommand{{\gapBaseTwentyBcaLo}}{{{f(base['lo'], 4, True)}}}",
        rf"\newcommand{{\gapBaseTwentyBcaHi}}{{{f(base['hi'], 4, True)}}}",
        rf"\newcommand{{\gapBaseTwentyBcaWidth}}"
        f"{{{f(float(base['hi']) - float(base['lo']), 4)}}}",
        rf"\newcommand{{\gapLiveBcaWidth}}"
        f"{{{f(float(live['hi']) - float(live['lo']), 4)}}}",
        "% long300: rigor/long_horizon, five seeds, 300 epochs",
        rf"\newcommand{{\gapTail}}{{{f(long_gap['point'], 5, True)}}}",
        rf"\newcommand{{\gapTailLo}}{{{f(long_gap['lo'], 5, True)}}}",
        rf"\newcommand{{\gapTailHi}}{{{f(long_gap['hi'], 5, True)}}}",
        "",
        "% ── The levels, for the ledger and the abstract ─────────────────────────",
        rf"\newcommand{{\giniNcf}}{{{f(gini_ncf['point'], 4)}}}",
        rf"\newcommand{{\giniOracle}}{{{f(gini_oracle['point'], 4)}}}",
        rf"\newcommand{{\mcrNcf}}{{{f(mcr['point'], 4)}}}",
        rf"\newcommand{{\rsiNcf}}{{{f(rsi['point'], 4)}}}",
        "",
        "% ── The convergence tail (rigor/long_horizon_asymptote.csv) ─────────────",
        "% The tail mean, its t interval, and the t/Wilcoxon p-values",
        rf"\newcommand{{\tailMean}}{{{f(asym['point'], 5, True)}}}",
        rf"\newcommand{{\tailMeanLo}}{{{f(asym['lo'], 6, True)}}}",
        rf"\newcommand{{\tailMeanHi}}{{{f(asym['hi'], 6, True)}}}",
        rf"\newcommand{{\tailT}}{{{f(asym['t'], 2)}}}",
        rf"\newcommand{{\tailP}}{{{f(asym['p'], 3)}}}",
        rf"\newcommand{{\tailWilcoxonP}}{{{f(asym['wilcoxon_p'], 3)}}}",
        rf"\newcommand{{\finalEpochSd}}{{{f(asym['sd'], 4)}}}",
        "",
        "% ── Stationarity (rigor/long_horizon_stationarity.csv) ───────────────────",
        rf"\newcommand{{\statPairedDiff}}{{{f(stat['paired_diff'], 5, True)}}}",
        rf"\newcommand{{\statP}}{{{f(stat['paired_t_p'], 3)}}}",
        rf"\newcommand{{\statWilcoxonP}}{{{f(stat['wilcoxon_p'], 3)}}}",
        rf"\newcommand{{\statMaxStep}}{{{f(stat['max_abs_step_last10'], 5)}}}",
        # The share is emitted as a bare number: paper/verify.py's brace checker
        # treats an escaped "\%" as a comment start and would flag the macro, and
        # a bare number lets the prose write "per cent" or "\%" as it prefers.
        rf"\newcommand{{\statMaxStepShare}}{{{f(100 * stat['max_step_over_seed_sd'], 1)}}}",
        rf"\newcommand{{\statSeedsLateHigher}}{{{int(stat['n_seeds_late_gt_early'])}}}",
        "",
        "% ── The sensitivity sweep (rigor/sensitivity.csv) ────────────────────────",
        rf"\newcommand{{\gapRangeLo}}{{{f(lo_point, 4, True)}}}",
        rf"\newcommand{{\gapRangeHi}}{{{f(hi_point, 4, True)}}}",
        rf"\newcommand{{\reachShipped}}{{{f(shipped['reach_point'], 4)}}}",
        rf"\newcommand{{\gapShippedSweep}}{{{f(shipped['gap_point'], 4, True)}}}",
        "",
        "% ── The institutional optimum (rigor/multiple_comparisons.csv) ───────────",
        rf"\newcommand{{\optWelfare}}{{{f(mc['optimum_mean'], 2, True)}}}",
        rf"\newcommand{{\optWelfareLo}}{{{f(mc['named_cell_lo_t_bonferroni'], 2, True)}}}",
        rf"\newcommand{{\optWelfareHi}}{{{f(mc['named_cell_hi_t_bonferroni'], 2, True)}}}",
        rf"\newcommand{{\optWelfareSd}}{{{f(mc['optimum_sd'], 2)}}}",
    ]
    # The standard error is derived, not read: SE = sd/sqrt(n) with n = 5 seeds.
    se = mc["optimum_sd"] / (5 ** 0.5)
    lines += [
        "% ── The 300-epoch trajectory, one macro per printed epoch ───────────────",
        "% Source: rigor/long_horizon_gap_series.csv (five-seed mean, 300 epochs).",
        "% The peak is at epoch 6, which is the value the abstract quotes.",
    ]
    series = gap_series()
    for epoch in (0, 6, 19, 49, 99, 124, 199):
        lines.append(
            rf"\newcommand{{\gapEpoch{'Zero' if epoch == 0 else epoch}}}"
            rf"{{{f(series[epoch], 4 if epoch < 99 else 5, True)}}}"
        )
    lines += [
        rf"\newcommand{{\gapPeak}}{{{f(series[6], 4, True)}}}",
        "",
    ]
    lines += [
        rf"\newcommand{{\optWelfareSe}}{{{f(se, 2)}}}",
        rf"\newcommand{{\optWelfareT}}{{{f(mc['optimum_t'], 2)}}}",
        rf"\newcommand{{\optWelfareQ}}{{{sci(mc['optimum_q_bh'])}}}",
        rf"\newcommand{{\optWelfareSeMultiple}}{{{f(mc['optimum_mean'] / se, 1)}}}",
        "",
        "% ── The H5 decomposition (same file, factorial block) ────────────────────",
        rf"\newcommand{{\hFiveInteraction}}{{{f(mc['interaction_mean'], 2, True)}}}",
        rf"\newcommand{{\hFiveInteractionLo}}{{{f(mc['interaction_lo_t'], 2, True)}}}",
        rf"\newcommand{{\hFiveInteractionHi}}{{{f(mc['interaction_hi_t'], 2, True)}}}",
        rf"\newcommand{{\hFiveInteractionBcaLo}}{{{f(mc['interaction_lo_bca'], 2, True)}}}",
        rf"\newcommand{{\hFiveInteractionBcaHi}}{{{f(mc['interaction_hi_bca'], 2, True)}}}",
        rf"\newcommand{{\hFiveInteractionQ}}{{{f(mc['interaction_q_bh'], 3)}}}",
        rf"\newcommand{{\hFiveAlgorithmic}}{{{f(mc['algorithmic_mean'], 2, True)}}}",
        rf"\newcommand{{\hFiveAlgorithmicLo}}{{{f(mc['algorithmic_lo_bca'], 2, True)}}}",
        rf"\newcommand{{\hFiveAlgorithmicHi}}{{{f(mc['algorithmic_hi_bca'], 2, True)}}}",
        rf"\newcommand{{\hFiveExogenous}}{{{f(mc['exogenous_mean'], 2, True)}}}",
        rf"\newcommand{{\hFiveExogenousBcaLo}}{{{f(mc['exogenous_lo_bca'], 2, True)}}}",
        rf"\newcommand{{\hFiveExogenousBcaHi}}{{{f(mc['exogenous_hi_bca'], 2, True)}}}",
        rf"\newcommand{{\hFiveExogenousTLo}}{{{f(mc['exogenous_lo_t'], 2, True)}}}",
        rf"\newcommand{{\hFiveExogenousTHi}}{{{f(mc['exogenous_hi_t'], 2, True)}}}",
        rf"\newcommand{{\hFiveExogenousQ}}{{{f(mc['exogenous_q_bh'], 3)}}}",
        "",
        "% ── Counts that are qualitative prose and stay literal are NOT here ──────",
        "% (twelve sweep cells, forty institutional cells, five seeds). The counts",
        "% that do carry a claim are:",
        rf"\newcommand{{\nSweepCells}}{{{int(len(sensitivity_rows()))}}}",
        rf"\newcommand{{\nInstitutionalCells}}{{{int(mc['n_cells_tested'])}}}",
        rf"\newcommand{{\nRejectBh}}{{{int(mc['n_reject_bh_q05'])}}}",
        rf"\newcommand{{\nRejectHolm}}{{{int(mc['n_reject_holm_05'])}}}",
        rf"\newcommand{{\fwerUncorrected}}{{{f(mc['fwer_uncorrected_independent'], 3)}}}",
        "",
    ]

    # ── The fresh-seed replication (PREREGISTRATION E1) ──────────────────────
    # Both sets at 1 thread, one session. `fresh` is seeds 100-119; `exploratory`
    # is 42-61 regenerated rather than the committed rigor/base/, which predates
    # the current code and no longer reproduces (docs/REVIEW_RESPONSE.md, A3).
    fresh = twenty_seed("fresh")
    expl = twenty_seed("exploratory_1t")
    lines += [
        "% ── The fresh-seed replication (rigor/fresh, rigor/exploratory_1t) ──────",
        rf"\newcommand{{\gapFresh}}{{{f(fresh['gini_gap']['mean'], 4, True)}}}",
        rf"\newcommand{{\gapFreshLo}}{{{f(fresh['gini_gap']['lo'], 4, True)}}}",
        rf"\newcommand{{\gapFreshHi}}{{{f(fresh['gini_gap']['hi'], 4, True)}}}",
        rf"\newcommand{{\gapExploratory}}{{{f(expl['gini_gap']['mean'], 4, True)}}}",
        rf"\newcommand{{\gapExploratoryLo}}{{{f(expl['gini_gap']['lo'], 4, True)}}}",
        rf"\newcommand{{\gapExploratoryHi}}{{{f(expl['gini_gap']['hi'], 4, True)}}}",
        rf"\newcommand{{\giniNcfFresh}}{{{f(fresh['gini_ncf']['mean'], 4)}}}",
        rf"\newcommand{{\giniNcfExploratory}}{{{f(expl['gini_ncf']['mean'], 4)}}}",
        rf"\newcommand{{\giniOracleFresh}}{{{f(fresh['gini_oracle']['mean'], 4)}}}",
        rf"\newcommand{{\giniOracleExploratory}}{{{f(expl['gini_oracle']['mean'], 4)}}}",
        "% The spreads the fresh/exploratory comparison quotes, so the contrast",
        "% between them is a computed difference rather than a typed one.",
        rf"\newcommand{{\giniNcfSdFresh}}{{{f(fresh['gini_ncf']['sd'], 4)}}}",
        rf"\newcommand{{\giniNcfSdExploratory}}{{{f(expl['gini_ncf']['sd'], 4)}}}",
        rf"\newcommand{{\algLossSdFresh}}{{{f(fresh['alg_loss']['sd'], 4)}}}",
        rf"\newcommand{{\algLossSdExploratory}}{{{f(expl['alg_loss']['sd'], 4)}}}",
        "",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(lines)} lines, "
          f"{sum(1 for ln in lines if ln.startswith(chr(92) + 'newcommand'))} macros)")


if __name__ == "__main__":
    main()
