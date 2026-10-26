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


# The 300-epoch trajectory the paper quotes. It is `long_horizon_1t`, the series
# regenerated at 1 thread in this session, because the committed `long_horizon/`
# cannot be regenerated from this code at any thread count (docs/REVIEW_RESPONSE.md
# A3). The two agree at the tolerance fixed in docs/PREREGISTRATION.md E7 (tail
# difference 0.000144); where they differ, the paper quotes this one.
LONG_HORIZON = RIGOR / "long_horizon_1t"


def asymptote() -> dict[str, float]:
    rows = read_csv(LONG_HORIZON / "long_horizon_asymptote.csv")
    if not rows:
        raise ValueError("long_horizon_asymptote.csv is empty")
    # `window` is the window length in epochs and `epochs` is the range label
    # ("250-299"); only the former is a number, so the range is skipped here and
    # read separately where the prose needs it.
    skip = ("quantity", "regime", "window", "epochs", "method")
    return {k: float(v) for k, v in rows[0].items() if k not in skip}


def stationarity() -> dict[str, float]:
    rows = read_csv(LONG_HORIZON / "long_horizon_stationarity.csv")
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
    """The five-seed mean gap by epoch, from the regenerated 300-epoch series."""
    rows = read_csv(LONG_HORIZON / "long_horizon_gap_series.csv")
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


EQUIVALENCE_MARGIN = 0.002   # declared in docs/PREREGISTRATION.md E5 before the test
T_ONE_SIDED_4DF = 2.132      # t_{0.05}, 4 df: the one-sided critical value TOST uses


def _grid_cell(directory: str, cost: float, size: float) -> dict[str, float]:
    for row in read_csv(RIGOR / directory / "grid_summary.csv"):
        if abs(float(row["signal_cost"]) - cost) < 1e-9 and \
                abs(float(row["size_fraction"]) - size) < 1e-9:
            return {k: float(v) for k, v in row.items()}
    raise KeyError(f"{directory}: no cell c={cost} sf={size}")


def _grid_slice(directory: str, cost: float) -> list[dict[str, float]]:
    rows = [{k: float(v) for k, v in r.items()}
            for r in read_csv(RIGOR / directory / "grid_summary.csv")
            if abs(float(r["signal_cost"]) - cost) < 1e-9]
    return sorted(rows, key=lambda r: r["welfare_diff_residual_mean"], reverse=True)


def _per_seed_gap_series(directory: Path) -> dict[int, dict[int, float]]:
    out: dict[int, dict[int, float]] = {}
    for path in sorted(directory.glob("ncf_seed*.csv")):
        seed = int(path.stem.split("seed")[-1])
        ncf = {int(r["epoch"]): float(r["gini"]) for r in read_csv(path)}
        orc = {int(r["epoch"]): float(r["gini"])
               for r in read_csv(path.with_name(f"oracle_seed{seed}.csv"))}
        out[seed] = {e: ncf[e] - orc[e] for e in ncf}
    return out


def _first_stays_below(series: dict[int, float], margin: float) -> int:
    epochs = sorted(series)
    for i, e in enumerate(epochs):
        if all(abs(series[x]) < margin for x in epochs[i:]):
            return e
    return -1


def warm_start() -> dict[str, float]:
    """The warm-start arm of the mechanism intervention, against its own control.

    Reads `data/processed/rigor/interventions/interventions_seed<seed>.csv`, one
    row per (condition, epoch). Both arms share a population, a seed and a thread
    count, so the terminal comparison is a paired per-seed difference and not a
    difference of means over unrelated runs.

    The decay share is defined as (that arm's own peak - its final) for the warm
    arm over the same quantity for the control. Using each arm's own peak matters:
    the warm arm starts informed, so its peak is lower and earlier, and measuring
    both against a common peak would credit the warm start for a decay it never
    had.
    """
    directory = RIGOR / "interventions"
    frames: dict[str, dict[int, dict[int, float]]] = {"control": {}, "warm": {}}
    for path in sorted(directory.glob("interventions_seed*.csv")):
        seed = int(path.stem.split("seed")[-1])
        for row in read_csv(path):
            cond = row["condition"]
            if cond in frames:
                frames[cond].setdefault(seed, {})[int(row["epoch"])] = float(row["gini"])
    if not frames["control"] or not frames["warm"]:
        raise ValueError("interventions: missing control or warm arm")

    def mean_at(cond: str, epoch: int) -> float:
        vals = [s[epoch] for s in frames[cond].values() if epoch in s]
        return sum(vals) / len(vals)

    def decay(cond: str) -> float:
        series = {
            e: sum(s[e] for s in frames[cond].values() if e in s)
            / len([s for s in frames[cond].values() if e in s])
            for e in next(iter(frames[cond].values()))
        }
        peak_epoch = max(series, key=lambda e: series[e])
        return series[peak_epoch] - series[max(series)]

    terminal = [frames["warm"][s][99] - frames["control"][s][99]
                for s in frames["control"] if 99 in frames["control"][s]
                and 99 in frames["warm"].get(s, {})]
    n = len(terminal)
    mean_d = sum(terminal) / n
    sd_d = (sum((x - mean_d) ** 2 for x in terminal) / (n - 1)) ** 0.5
    return {
        "epoch0_control": mean_at("control", 0),
        "epoch0_warm": mean_at("warm", 0),
        "reach0_control": _mean_reach("control", 0),
        "reach0_warm": _mean_reach("warm", 0),
        "final_control": mean_at("control", 99),
        "final_warm": mean_at("warm", 99),
        "terminal_diff": mean_d,
        "terminal_sd": sd_d,
        "terminal_t": mean_d / (sd_d / n ** 0.5),
        "decay_share": decay("warm") / decay("control"),
    }


def _mean_reach(condition: str, epoch: int) -> float:
    directory = RIGOR / "interventions"
    vals = []
    for path in sorted(directory.glob("interventions_seed*.csv")):
        for row in read_csv(path):
            if row["condition"] == condition and int(row["epoch"]) == epoch:
                vals.append(float(row["reach"]))
    return sum(vals) / len(vals)


def revision_macros(f) -> list[str]:
    """Every result the revision of 2026-10-25 quotes, read from its own CSV.

    Kept in one function so the provenance of the session's numbers is in one
    place. Configuration values (gate phi, cost and size grids, sweep levels, the
    0.002 margin, the 0.05 ceiling) are settings rather than results and stay
    literal in the prose.
    """
    out = ["% ── Revision of 2026-10-25: results it added, each from its own CSV ───"]

    # The 300-epoch tail, regenerated series: equivalence test and time-to-threshold.
    asym = asymptote()
    se = asym["sd"] / asym["n_seeds"] ** 0.5
    m = asym["point"]
    firsts = [_first_stays_below(s, EQUIVALENCE_MARGIN)
              for _, s in sorted(_per_seed_gap_series(LONG_HORIZON).items())]
    mean_series = gap_series()
    out += [
        "% rigor/long_horizon_1t: tail statistics and the TOST (PREREGISTRATION E5)",
        rf"\newcommand{{\tailSe}}{{{f(se, 6)}}}",
        rf"\newcommand{{\tailSeedsPositive}}{{{int(asym['n_seeds_positive'])}}}",
        rf"\newcommand{{\tostTLower}}{{{f((m + EQUIVALENCE_MARGIN) / se, 1)}}}",
        rf"\newcommand{{\tostTUpper}}{{{f((EQUIVALENCE_MARGIN - m) / se, 1)}}}",
        rf"\newcommand{{\tostLo}}{{{f(m - T_ONE_SIDED_4DF * se, 6, True)}}}",
        rf"\newcommand{{\tostHi}}{{{f(m + T_ONE_SIDED_4DF * se, 6, True)}}}",
        rf"\newcommand{{\firstBelowMarginBySeed}}{{{', '.join(map(str, firsts[:-1]))} and {firsts[-1]}}}",
        rf"\newcommand{{\firstBelowMarginMin}}{{{min(firsts)}}}",
        rf"\newcommand{{\firstBelowMarginMax}}{{{max(firsts)}}}",
        rf"\newcommand{{\firstBelowMarginMean}}{{{_first_stays_below(mean_series, EQUIVALENCE_MARGIN)}}}",
        rf"\newcommand{{\tailRatioNinetyNine}}{{{f(mean_series[99] / m, 0)}}}",
        "% ratio of two imprecise quantities; quoted as an order of magnitude only",
        "% (paper/sections/results.tex, the cold-start trajectory paragraph)",
    ]
    fits = read_csv(LONG_HORIZON / "long_horizon_decay_fits.csv")
    peak_fit = [r for r in fits if r["fit_window"] == "from_peak_6"
                and r["model"] == "decay_to_zero"][0]
    out.append(rf"\newcommand{{\decayDeltaBic}}{{{f(peak_fit['delta_bic'], 1)}}}")

    # The width comparison of A1, as a computed ratio.
    live = regime_row("gini_gap_ncf_minus_oracle", "live100")
    base = regime_row("gini_gap_ncf_minus_oracle", "base20")
    ratio = (float(live["hi"]) - float(live["lo"])) / (float(base["hi"]) - float(base["lo"]))
    out.append(rf"\newcommand{{\widthRatioFiveTwenty}}{{{f(ratio, 1)}}}")

    # The thread-count spread and the between-seed sd it is compared against.
    spans: dict[str, list[float]] = {}
    for r in read_csv(RIGOR / "thread_sensitivity.csv"):
        spans.setdefault(r["seed"], []).append(float(r["gini"]))
    widths = [max(v) - min(v) for v in spans.values()]
    base_ginis = []
    for path in sorted((RIGOR / "base").glob("metrics_seed*.csv")):
        rows = read_csv(path)
        last = max(int(r["epoch"]) for r in rows)
        base_ginis += [float(r["gini"]) for r in rows
                       if int(r["epoch"]) == last and r["arm"] == "ncf"]
    bm = sum(base_ginis) / len(base_ginis)
    bsd = (sum((g - bm) ** 2 for g in base_ginis) / (len(base_ginis) - 1)) ** 0.5
    out += [
        "% rigor/thread_sensitivity.csv (seeds 42-44, 1/4/8 threads) and rigor/base/",
        rf"\newcommand{{\threadSpreadLo}}{{{f(min(widths), 4)}}}",
        rf"\newcommand{{\threadSpreadHi}}{{{f(max(widths), 4)}}}",
        rf"\newcommand{{\betweenSeedSdTwenty}}{{{f(bsd, 4)}}}",
    ]

    # A3: seed 42, epoch 99, NCF Gini in the two directories that disagree.
    def ep99(path: Path) -> float:
        return float([r for r in read_csv(path) if r["epoch"] == "99"][0]["gini"])
    out += [
        "% A3: ablation_h100 vs rigor/long_horizon, seed 42, NCF Gini at epoch 99",
        rf"\newcommand{{\giniSeedFortyTwoShort}}{{{f(ep99(PROCESSED / 'ablation_h100' / 'ncf_seed42.csv'), 4)}}}",
        rf"\newcommand{{\giniSeedFortyTwoLong}}{{{f(ep99(RIGOR / 'long_horizon' / 'ncf_seed42.csv'), 4)}}}",
    ]

    # The sweep: the cells the prose names, from rigor/sensitivity.csv.
    sens = {(r["axis"], r["level"]): r for r in sensitivity_rows()}
    low = sens[("shock_scale", "0.5")]
    high = sens[("shock_scale", "3.0")]
    deltas = [sens[("delta", lv)] for lv in ("0.0", "0.5", "1.0")]
    delta_widths = [float(r["gap_hi"]) - float(r["gap_lo"]) for r in deltas]
    delta_points = [float(r["gap_point"]) for r in deltas]
    out += [
        "% rigor/sensitivity.csv",
        rf"\newcommand{{\reachLowShock}}{{{f(low['reach_point'], 4)}}}",
        rf"\newcommand{{\reachLowShockCount}}{{{round(float(low['reach_point']) * 1000)}}}",
        rf"\newcommand{{\reachShippedCount}}{{{round(float(sens[('shock_scale', '1.5')]['reach_point']) * 1000)}}}",
        rf"\newcommand{{\gapLowShock}}{{{f(low['gap_point'], 4, True)}}}",
        rf"\newcommand{{\gapHighShockHi}}{{{f(high['gap_hi'], 4, True)}}}",
        rf"\newcommand{{\gapDeltaZero}}{{{f(delta_points[0], 4, True)}}}",
        rf"\newcommand{{\gapDeltaHalf}}{{{f(delta_points[1], 4, True)}}}",
        rf"\newcommand{{\gapDeltaOne}}{{{f(delta_points[2], 4, True)}}}",
        rf"\newcommand{{\gapDeltaSpread}}{{{f(max(delta_points) - min(delta_points), 4)}}}",
        rf"\newcommand{{\gapDeltaWidth}}{{{f(sum(delta_widths) / len(delta_widths), 3)}}}",
    ]
    def axis_spread(axis: str) -> float:
        pts = [float(r["gap_point"]) for (a, _), r in sens.items() if a == axis]
        return max(pts) - min(pts)
    delta_spread = axis_spread("delta")
    all_points = [float(r["gap_point"]) for r in sens.values()]
    out += [
        rf"\newcommand{{\gapRangeRatio}}{{{f(max(all_points) / min(all_points), 0)}}}",
        rf"\newcommand{{\betaAlphaVsDelta}}{{{f(axis_spread('beta_alpha') / delta_spread, 1)}}}",
        rf"\newcommand{{\shockVsDelta}}{{{f(axis_spread('shock_scale') / delta_spread, 0)}}}",
    ]

    # C4: the low-chemistry cell at 300 epochs.
    c4_dir = RIGOR / "c4_shock0.5"
    c4_gap = _per_seed_gap_series(c4_dir)
    c4_mean = {e: sum(s[e] for s in c4_gap.values()) / len(c4_gap)
               for e in next(iter(c4_gap.values()))}
    c4_peak = max(c4_mean, key=c4_mean.get)
    reach: dict[int, list[float]] = {}
    for path in sorted(c4_dir.glob("ncf_seed*.csv")):
        for r in read_csv(path):
            reach.setdefault(int(r["epoch"]), []).append(float(r["reach"]))
    def mean_reach(e: int) -> float:
        return sum(reach[e]) / len(reach[e])
    out += [
        "% rigor/c4_shock0.5 (300 epochs, shock_scale 0.5, 5 seeds, 1 thread)",
        rf"\newcommand{{\cfourReachZero}}{{{f(mean_reach(0), 3)}}}",
        rf"\newcommand{{\cfourReachFortyNine}}{{{f(mean_reach(49), 3)}}}",
        rf"\newcommand{{\cfourReachNinetyNine}}{{{f(mean_reach(99), 3)}}}",
        rf"\newcommand{{\cfourReachFinal}}{{{f(mean_reach(299), 3)}}}",
        rf"\newcommand{{\cfourReachFinalShort}}{{{f(mean_reach(299), 2)}}}",
        rf"\newcommand{{\cfourReachNinetyNineShort}}{{{f(mean_reach(99), 2)}}}",
        rf"\newcommand{{\cfourPeak}}{{{f(c4_mean[c4_peak], 4, True)}}}",
        rf"\newcommand{{\cfourPeakEpoch}}{{{c4_peak}}}",
        rf"\newcommand{{\cfourFinal}}{{{f(c4_mean[299], 4, True)}}}",
    ]

    # H4 at the shipped gate, and on fresh seeds at the inert gate.
    shipped = _grid_slice("h4_shipped_gate", 0.01)
    shipped_by_size = {r["size_fraction"]: r for r in shipped}
    rank = [r["size_fraction"] for r in shipped].index(0.65) + 1
    best = {c: _grid_slice("h4_shipped_gate", c)[0]["welfare_diff_residual_mean"]
            for c in (0.05, 0.10, 0.25)}
    out += [
        "% rigor/h4_shipped_gate (admissible_fraction 0.02, seeds 42-46)",
        rf"\newcommand{{\hFourShippedSmall}}{{{f(shipped_by_size[0.02]['welfare_diff_residual_mean'], 2, True)}}}",
        rf"\newcommand{{\hFourShippedFull}}{{{f(shipped_by_size[1.0]['welfare_diff_residual_mean'], 2, True)}}}",
        rf"\newcommand{{\hFourShippedOpt}}{{{f(shipped_by_size[0.65]['welfare_diff_residual_mean'], 2, True)}}}",
        rf"\newcommand{{\hFourShippedOptRank}}{{{rank}}}",
        rf"\newcommand{{\hFourShippedBestMid}}{{{f(best[0.05], 2, True)}}}",
        rf"\newcommand{{\hFourShippedBestHigh}}{{{f(best[0.10], 2, True)}}}",
        rf"\newcommand{{\hFourShippedBestTop}}{{{f(best[0.25], 2, True)}}}",
    ]
    fresh_cell = _grid_cell("c2_fresh_inert", 0.01, 0.65)
    fresh_values = [float(r["welfare_diff_residual"]) for r in read_csv(RIGOR / "c2_fresh_inert" / "grid.csv")
                    if abs(float(r["signal_cost"]) - 0.01) < 1e-9
                    and abs(float(r["size_fraction"]) - 0.65) < 1e-9]
    n = len(fresh_values)
    fm = sum(fresh_values) / n
    fsd = (sum((v - fm) ** 2 for v in fresh_values) / (n - 1)) ** 0.5
    fse = fsd / n ** 0.5
    t19 = 2.093
    fresh_rank = [r["size_fraction"] for r in _grid_slice("c2_fresh_inert", 0.01)].index(0.65) + 1
    original_sd = mc_values()["optimum_sd"]
    out += [
        "% rigor/c2_fresh_inert (seeds 100-119, admissible_fraction 0.2)",
        rf"\newcommand{{\hFourFresh}}{{{f(fm, 2, True)}}}",
        rf"\newcommand{{\hFourFreshSd}}{{{f(fsd, 2)}}}",
        rf"\newcommand{{\hFourFreshLo}}{{{f(fm - t19 * fse, 2, True)}}}",
        rf"\newcommand{{\hFourFreshHi}}{{{f(fm + t19 * fse, 2, True)}}}",
        rf"\newcommand{{\hFourFreshT}}{{{f(fm / fse, 2)}}}",
        rf"\newcommand{{\hFourFreshRank}}{{{fresh_rank}}}",
        rf"\newcommand{{\hFourFreshSdRatio}}{{{f(fsd / original_sd, 1)}}}",
    ]
    # The per-seed grid.csv and the summary must agree on the cell; a mismatch
    # would mean the mean above was computed from the wrong rows.
    if abs(fresh_cell["welfare_diff_residual_mean"] - fm) > 1e-9:
        raise ValueError("c2_fresh_inert: grid.csv and grid_summary.csv disagree at sf=0.65")
    opt = regime_row("welfare_diff_residual", "optimum_c0.01_sf0.65")
    out += [
        "% intervals.csv, the H4 optimum's uncorrected t interval",
        rf"\newcommand{{\optWelfareTLo}}{{{f(opt['lo_t'], 2, True)}}}",
        rf"\newcommand{{\optWelfareTHi}}{{{f(opt['hi_t'], 2, True)}}}",
        rf"\newcommand{{\optWelfareMag}}{{{f(mc_values()['optimum_mean'], 2)}}}",
        "",
    ]

    # The warm-start intervention (PREREGISTRATION E3ii). The frozen arm is
    # omitted deliberately: freezing the weights freezes the shortlists, so it
    # cannot test anything, and the paper says that rather than quoting a number
    # from it.
    warm = warm_start()
    out += [
        "% rigor/interventions: the warm-start arm against its own control",
        rf"\newcommand{{\warmEpochZeroControl}}{{{f(warm['epoch0_control'], 4, True)}}}",
        rf"\newcommand{{\warmEpochZeroTreated}}{{{f(warm['epoch0_warm'], 4, True)}}}",
        rf"\newcommand{{\warmReachZeroControl}}{{{f(warm['reach0_control'], 3)}}}",
        rf"\newcommand{{\warmReachZeroTreated}}{{{f(warm['reach0_warm'], 3)}}}",
        rf"\newcommand{{\warmFinalControl}}{{{f(warm['final_control'], 4, True)}}}",
        rf"\newcommand{{\warmFinalTreated}}{{{f(warm['final_warm'], 4, True)}}}",
        rf"\newcommand{{\warmTerminalDiff}}{{{f(warm['terminal_diff'], 5, True)}}}",
        rf"\newcommand{{\warmTerminalT}}{{{f(warm['terminal_t'], 2)}}}",
        rf"\newcommand{{\warmTerminalSd}}{{{f(warm['terminal_sd'], 5)}}}",
        rf"\newcommand{{\warmDecaySurviving}}{{{f(warm['decay_share'] * 100, 0)}}}",
        "",
    ]
    return out


def main() -> None:
    live = regime_row("gini_gap_ncf_minus_oracle", "live100")
    base = regime_row("gini_gap_ncf_minus_oracle", "base20")

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
        "% base20: the same quantity at 20 seeds (rigor/base, archival; see its README)",
        rf"\newcommand{{\gapBaseTwenty}}{{{f(base['point'], 4, True)}}}",
        rf"\newcommand{{\gapBaseTwentyLo}}{{{f(base['lo_t'], 4, True)}}}",
        rf"\newcommand{{\gapBaseTwentyHi}}{{{f(base['hi_t'], 4, True)}}}",
        "% base20 BCa bounds, for the full-width comparison of A1. The individual",
        "% endpoints are not quoted in the prose; the widths they produce are.",
        rf"\newcommand{{\gapBaseTwentyBcaWidth}}"
        f"{{{f(float(base['hi']) - float(base['lo']), 4)}}}",
        rf"\newcommand{{\gapLiveBcaWidth}}"
        f"{{{f(float(live['hi']) - float(live['lo']), 4)}}}",
        "% The 300-epoch tail is in the convergence block below and the revision",
        "% block at the end, both from rigor/long_horizon_1t.",
        "",
        "% ── The convergence tail (rigor/long_horizon_asymptote.csv) ─────────────",
        "% The tail mean, its t interval, and the t/Wilcoxon p-values",
        rf"\newcommand{{\tailMean}}{{{f(asym['point'], 6, True)}}}",
        rf"\newcommand{{\tailMeanLo}}{{{f(asym['lo'], 6, True)}}}",
        rf"\newcommand{{\tailMeanHi}}{{{f(asym['hi'], 6, True)}}}",
        rf"\newcommand{{\tailT}}{{{f(asym['t'], 2)}}}",
        rf"\newcommand{{\tailP}}{{{f(asym['p'], 3)}}}",
        rf"\newcommand{{\tailWilcoxonP}}{{{f(asym['wilcoxon_p'], 3)}}}",
        rf"\newcommand{{\finalEpochSd}}{{{f(asym['sd'], 6)}}}",
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
        "% The between-seed sd of the NCF arm's Gini at epoch 299: the noise scale the",
        "% 'ordered noise after epoch 124' caution compares the gap points against.",
        rf"\newcommand{{\finalEpochGiniSd}}{{{f(stat['seed_sd_final'], 4)}}}",
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
    # TeX control words are letters only: \gapEpoch6 parses as \gapEpoch followed
    # by the character 6, so the epoch is spelled out in the macro name.
    epoch_names = {0: "Zero", 6: "Six", 19: "Nineteen", 49: "FortyNine",
                   99: "NinetyNine", 124: "OneTwentyFour", 199: "OneNinetyNine"}
    for epoch, name in epoch_names.items():
        lines.append(
            rf"\newcommand{{\gapEpoch{name}}}"
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

    lines += revision_macros(f)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(lines)} lines, "
          f"{sum(1 for ln in lines if ln.startswith(chr(92) + 'newcommand'))} macros)")


if __name__ == "__main__":
    main()
