"""Write ``paper/tables/numbers.tex``: one LaTeX macro per number the prose quotes.

    python paper/gen_numbers.py

Every macro reads a CSV under ``data/processed/`` written by
``scripts/analyze_horizon.py``, ``scripts/analyze_extras.py`` or
``scripts/run_validation.py``. Nothing is typed. ``tests/python/test_paper_numbers.py``
fails if the committed ``numbers.tex`` differs from a fresh render, and recomputes
the headline numbers from the raw seed files independently.

TeX control words are letters only, so numbers in a name are spelled out
(``\\gapNinetyNine``). A macro holds a signed value when the quantity is a
difference and an unsigned one when it is a level, a share or a count.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pandas as pd
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parent.parent
H = ROOT / "data" / "processed" / "horizon"
OUT = pathlib.Path(__file__).resolve().parent / "tables" / "numbers.tex"


def A(exp: str, name: str) -> pd.DataFrame:
    return pd.read_csv(H / exp / "analysis" / name)


def long(exp: str, name: str) -> dict[str, float]:
    frame = A(exp, name)
    return dict(zip(frame["quantity"], frame["value"]))


def f(value: float, places: int = 4, signed: bool = False) -> str:
    value = float(value)
    if value == 0.0:
        value = 0.0  # no "-0.0000"
    out = f"{value:+.{places}f}" if signed else f"{value:.{places}f}"
    return out.replace("+-", "-")


def n(value: float) -> str:
    """A count or a round: an integer when it is one, else one decimal."""
    value = float(value)
    return str(int(round(value))) if abs(value - round(value)) < 1e-9 else f"{value:.1f}"


def thousands(value: float) -> str:
    return f"{int(round(float(value))):,}".replace(",", "{,}")


def sci(value: float, places: int = 1) -> str:
    """LaTeX scientific notation for a small positive number."""
    value = float(value)
    if value == 0.0:
        return "0"
    exp = int(math.floor(math.log10(abs(value))))
    mant = value / 10 ** exp
    return rf"{mant:.{places}f}\times 10^{{{exp}}}"


def censored_median(values: pd.Series, censored: pd.Series) -> float:
    """Median with censored values counted as larger than every observed one."""
    v = np.where(censored.to_numpy(), np.inf, values.to_numpy(dtype=float))
    med = float(np.median(np.sort(v)))
    if not math.isfinite(med):
        raise ValueError("more than half the seeds are censored; the median is not defined")
    return med


class Macros:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.names: set[str] = set()

    def comment(self, text: str) -> None:
        self.lines.append(f"% {text}")

    def add(self, name: str, value: str) -> None:
        if not name.isalpha():
            raise ValueError(f"TeX macro names are letters only: {name}")
        if name in self.names:
            raise ValueError(f"macro defined twice: {name}")
        self.names.add(name)
        self.lines.append(rf"\newcommand{{\{name}}}{{{value}}}")


MARGIN = 0.002


def sig(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}g}"


def transient(m: Macros) -> None:
    s = A("e09_canonical", "summary.csv").set_index("set")
    c, e = s.loc["confirmation"], s.loc["exploratory"]
    ser = A("e09_canonical", "series.csv")
    cs = ser[ser.set == "confirmation"].set_index("round")
    x = long("e09_canonical", "extras.csv")
    if int(c["gap_99_n_positive"]) != 20:
        raise ValueError("the prose says the round-99 gap is positive on all twenty seeds")
    pred = pd.read_csv(H / "predictions.csv")
    if not pred[pred.experiment == "R2-E9"]["held"].all():
        raise ValueError("the prose says every R2-E9 prediction held")
    off = A("e13c_offpolicy", "paired.csv")
    late = off[off["round"] == 299]
    if not ((late["lo_t95"] < 0) & (late["hi_t95"] > 0)).all():
        raise ValueError("the prose says the off-policy difference is gone by round 299")
    m.comment("R2-E9 and R2-E12: e09_canonical/analysis/{summary,series,extras}.csv")
    m.add("gapZero", f(cs.loc[0, "gap_mean"], 3, True))
    m.add("peakGap", f(c["peak_gap"], 3, True))
    m.add("peakRound", n(c["peak_round"]))
    m.add("gapNinetyNine", f(c["gap_99_mean"], 4, True))
    m.add("gapNinetyNineLo", f(c["gap_99_lo95"], 4, True))
    m.add("gapNinetyNineHi", f(c["gap_99_hi95"], 4, True))
    m.add("gapNinetyNineBcaLo", f(c["gap_99_bca_lo"], 4, True))
    m.add("gapNinetyNineBcaHi", f(c["gap_99_bca_hi"], 4, True))
    m.add("bcaAgree", f(x["bca_agree"], 4))
    m.add("peakGapExpl", f(e["peak_gap"], 3, True))
    m.add("gapNinetyNineExpl", f(e["gap_99_mean"], 4, True))
    m.add("gapNinetyNineExplLo", f(e["gap_99_lo95"], 4, True))
    m.add("gapNinetyNineExplHi", f(e["gap_99_hi95"], 4, True))
    for tag, row in (("Conf", c), ("Expl", e)):
        if int(row["n_censored"]) != 0:
            raise ValueError("the prose says no seed is censored")
        m.add(f"entryMedian{tag}", n(row["entry_median"]))
        m.add(f"entryMin{tag}", n(row["entry_min"]))
        m.add(f"entryMax{tag}", n(row["entry_max"]))
    m.add("entryMinAll", n(x["entry_min_all"]))
    m.add("entryMaxAll", n(x["entry_max_all"]))
    m.add("tailConf", f(c["tail_mean"], 5, True))
    m.add("tailConfLo", f(c["tail_lo95"], 5, True))
    m.add("tailConfHi", f(c["tail_hi95"], 5, True))
    m.add("tailConfPos", n(c["tail_n_positive"]))
    m.add("tostConfLower", f(c["tost_t_lower"], 1))
    m.add("tostConfUpper", f(c["tost_t_upper"], 1))
    m.add("tostConfLo", f(c["tost_lo90"], 5, True))
    m.add("tostConfHi", f(c["tost_hi90"], 5, True))
    m.add("tailExpl", f(e["tail_mean"], 5, True))
    m.add("tailExplLo", f(e["tail_lo95"], 5, True))
    m.add("tailExplHi", f(e["tail_hi95"], 5, True))
    m.add("giniOracleTail", f(c["gini_oracle_tail"], 3))
    m.add("oracleSettle", n(x["oracle_settle_round"]))
    m.add("giniRandomTail", f(x["gini_random_tail"], 3))
    m.add("algLossNcfFinal", f(x["alg_loss_ncf_299"], 3))
    m.add("algLossOracleFinal", f(x["alg_loss_oracle_299"], 3))
    m.add("algLossDiff", f(x["alg_loss_diff_299"], 3, True))
    m.add("algLossDiffLo", f(x["alg_loss_diff_299_lo"], 3, True))
    m.add("algLossDiffHi", f(x["alg_loss_diff_299_hi"], 3, True))
    m.add("jaccardZero", f(cs.loc[0, "jaccard_oracle"], 3))
    m.add("jaccardFinal", f(cs.loc[299, "jaccard_oracle"], 3))
    m.add("spearmanFinal", f(cs.loc[299, "spearman_mean"], 2))
    m.add("stepsPerPass", n(round(c["steps_per_pass"])))
    m.add("itemShareZeroMin", f(x["item_share_0_min_all"], 4))
    m.add("modalShareZeroMin", f(x["modal_share_0_min_all"], 3))
    m.add("giniZeroMaxDev", sci(x["gini_ncf_0_maxdev_all"]))
    m.add("identicalBelowRound", n(c["identical_below_01_by"]))
    m.add("itemBelowRound", n(c["item_below_05_by"]))
    m.add("corrMedian", f(x["corr_median_all"], 2, True))
    m.add("corrMin", f(x["corr_min"], 2, True))
    m.comment("Supplementary notes S1 and S2: e09_canonical/analysis/extras.csv")
    m.add("mcrZero", f(x["mcr_ncf_0"], 3))
    m.add("mcrFinal", f(x["mcr_ncf_299"], 3))
    for name in ("gap", "loss"):
        tag = name.capitalize()
        m.add(f"sdFive{tag}", f(x[f"sd_five_{name}"], 4))
        m.add(f"sdTwenty{tag}", f(x[f"sd_twenty_{name}"], 4))
        m.add(f"rankFive{tag}", n(x[f"rank_five_{name}"]))
    m.add("sdLossRatio", f(x["sd_twenty_loss"] / x["sd_five_loss"], 1))


def floor(m: Macros) -> None:
    fits = A("e10_long", "fits.csv").set_index("model")
    s = A("e10_long", "summary.csv").iloc[0]
    if s["winner"] != "exp" or bool(s["positive_floor"]):
        raise ValueError("the prose says the pure exponential wins and there is no floor")
    m.comment("R2-E10: e10_long/analysis/{fits,summary}.csv")
    if float(s["repro_max_abs_diff_rounds_0_299"]) != 0.0:
        raise ValueError("the prose says rounds 0-299 reproduce R2-E9 exactly")
    a_pow, b_pow = (float(v) for v in fits.loc["power", "params"].split())
    peak = int(fits.loc["power", "peak_round"])
    m.add("powerAtEnd", f(a_pow * (999 - peak + 1.0) ** -b_pow, 3))
    m.add("rmseExp", sig(fits.loc["exp", "oos_rmse"]))
    m.add("rmseExpConst", sig(fits.loc["exp_const", "oos_rmse"]))
    m.add("rmsePower", sig(fits.loc["power", "oos_rmse"]))
    m.add("halfLife", f(fits.loc["exp", "half_life"], 1))
    m.add("halfLifeLo", f(fits.loc["exp", "half_life_lo"], 1))
    m.add("halfLifeHi", f(fits.loc["exp", "half_life_hi"], 1))
    m.add("fitK", f(fits.loc["exp_const", "K"], 4, True))
    m.add("fitKLo", f(fits.loc["exp_const", "K_lo"], 4, True))
    m.add("fitKHi", f(fits.loc["exp_const", "K_hi"], 4, True))
    m.add("longTail", f(s["tail_mean"], 5, True))
    m.add("longTailLo", f(s["tail_lo95"], 5, True))
    m.add("longTailHi", f(s["tail_hi95"], 5, True))
    m.add("longTailPos", n(s["tail_n_positive"]))
    m.add("longTostLo", f(s["tost_lo90"], 5, True))
    m.add("longTostHi", f(s["tost_hi90"], 5, True))
    m.add("lateDiff", f(s["late_diff_mean"], 6, True))
    m.add("lateDiffLo", f(s["late_diff_lo95"], 6, True))
    m.add("lateDiffHi", f(s["late_diff_hi95"], 6, True))
    m.add("longTailMarginShare", f(100 * s["tail_mean"] / MARGIN, 0))


def where(m: Macros) -> None:
    cells = A("e11_sweep", "cells.csv").set_index("cell")
    per = A("e11_sweep", "per_seed.csv")
    lo_cell, hi_cell = cells["peak_mean"].idxmin(), cells["peak_mean"].idxmax()
    if (lo_cell, hi_cell) != ("shock_scale_0.5", "shock_scale_3.0"):
        raise ValueError(f"the prose names the extreme cells; they are {lo_cell}, {hi_cell}")
    if not cells.loc["shock_scale_0.5", "C5t"] or cells.loc["shock_scale_3.0", "C2t"]:
        raise ValueError("the prose describes C5t at chemistry 0.5 and C2t failing at 3.0")
    orc = per.groupby("cell")[["gini_oracle_299", "reach_oracle_299"]].mean()
    counts = {"C1t": 9, "C2t": 7, "C3t": 8, "C4t": 9}
    for claim, want in counts.items():
        if int(cells[claim].sum()) != want:
            raise ValueError(f"the prose says {claim} holds in {want} of nine cells")
    m.comment("R2-E11: e11_sweep/analysis/{cells,per_seed}.csv")
    m.add("tostShockHighHi", f(cells.loc["shock_scale_3.0", "tost_hi90"], 4, True))
    m.add("peakMinCell", f(cells.loc[lo_cell, "peak_mean"], 3, True))
    m.add("peakMaxCell", f(cells.loc[hi_cell, "peak_mean"], 3, True))
    m.add("oracleGiniShockLow", f(orc.loc["shock_scale_0.5", "gini_oracle_299"], 3))
    m.add("oracleGiniShockHigh", f(orc.loc["shock_scale_3.0", "gini_oracle_299"], 3))
    m.add("tailShockHigh", f(cells.loc["shock_scale_3.0", "tail_mean"], 4, True))
    m.add("censoredShockHigh", n(cells.loc["shock_scale_3.0", "n_censored"]))
    m.add("tailShockLow", f(cells.loc["shock_scale_0.5", "tail_mean"], 4, True))
    m.add("tailBetaHigh", f(cells.loc["beta_alpha_3.0", "tail_mean"], 4, True))
    m.add("reachShockLow", f(cells.loc["shock_scale_0.5", "reach_299"], 3))
    m.add("reachOracleShockLow", f(orc.loc["shock_scale_0.5", "reach_oracle_299"], 3))

    def spread(names: list[str]) -> tuple[float, float]:
        peaks = [cells.loc[c, "peak_mean"] for c in names]
        med = []
        for c in names:
            g = per[per.cell == c]
            med.append(censored_median(g["entry_round"].fillna(0), g["censored"]))
        return max(peaks) - min(peaks), max(med) - min(med)

    dp, de = spread(["delta_0.0", "shipped", "delta_1.0"])
    sp, se = spread(["shock_scale_0.5", "shipped", "shock_scale_3.0"])
    m.add("deltaPeakSpread", f(dp, 3))
    m.add("deltaEntrySpread", n(round(de)))
    m.add("shockPeakSpread", f(sp, 3))
    m.add("shockEntrySpread", n(round(se)))

    s17 = A("e17_size", "summary.csv").set_index("N")
    m.comment("R2-E17 and R2-E18: e17_size/analysis/summary.csv")
    for nn, tag in ((500, "NFive"), (1000, "NTen"), (2000, "NTwenty")):
        m.add(f"peak{tag}", f(s17.loc[nn, "peak"], 3, True))
        m.add(f"Tfive{tag}", n(round(s17.loc[nn, "T05"])))
        m.add(f"stepsTfive{tag}", thousands(s17.loc[nn, "steps_at_T05"]))
    m.add("censoredNTwenty", n(s17.loc[2000, "n_censored"]))
    m.add("rowsPerRound", n(round(s17["swipes_per_round"].mean())))


def what_ends(m: Macros) -> None:
    bd = A("e13a_dose", "by_dose.csv").set_index("dose")
    s = A("e13a_dose", "summary.csv").set_index("landmark")
    per = A("e13a_dose", "per_seed.csv")
    m.comment("R2-E13a: e13a_dose/analysis/{by_dose,summary,per_seed}.csv")
    for d, tag in ((0.5, "Half"), (1.0, "One"), (2.0, "Two"), (4.0, "Four")):
        m.add(f"TfiveDose{tag}", n(round(bd.loc[d, "T05"])))
    m.add("stepsTfiveMin", thousands(bd["steps_at_T05"].min()))
    m.add("stepsTfiveMax", thousands(bd["steps_at_T05"].max()))
    m.add("doseElasticity", f(s.loc["T05", "elasticity"], 2, True))
    m.add("doseElasticityLo", f(s.loc["T05", "lo"], 2, True))
    m.add("doseElasticityHi", f(s.loc["T05", "hi"], 2, True))
    four = A("e13a_dose", "series.csv")
    four = four[four.dose == 4.0].set_index("round")["mean"]
    if not four.loc[299] < -MARGIN:
        raise ValueError("the prose says the four-pass gap ends below the margin")
    m.add("doseFourTail", f(four.loc[299], 4, True))
    m.add("doseFourCensored", n(per[per.dose == 4.0]["entry"].isna().sum()))

    p = A("e13c_offpolicy", "paired.csv").set_index(["comparison", "round"])
    # The on-policy gap of the same ten seeds the shadow rankers ran on, so the share
    # and the slope have the comparison's own denominator.
    ser = A("e13c_offpolicy", "series.csv")
    cs = ser[ser.data_from == "oracle"].set_index("round")["on_policy"]
    m.comment("R2-E13c: e13c_offpolicy/analysis/{paired,series}.csv")
    for src, tag in (("oracle", "Oracle"), ("random", "Random")):
        row = p.loc[(f"offpolicy_{src}_minus_onpolicy", 49)]
        if not row["hi_t95"] < 0:
            raise ValueError("the prose says the off-policy gap is smaller at round 49")
        m.add(f"off{tag}FortyNine", f(-row["mean"], 3))
        m.add(f"off{tag}FortyNineLo", f(-row["hi_t95"], 3))
        m.add(f"off{tag}FortyNineHi", f(-row["lo_t95"], 3))
    mean_eff = -0.5 * (p.loc[("offpolicy_oracle_minus_onpolicy", 49), "mean"]
                       + p.loc[("offpolicy_random_minus_onpolicy", 49), "mean"])
    slope = (cs.loc[44] - cs.loc[54]) / 10.0
    m.add("offShare", n(round(100 * mean_eff / cs.loc[49])))
    m.add("offDelay", n(round(mean_eff / slope)))

    e = A("e13b_explore", "paired.csv").set_index(["comparison", "round"])
    m.comment("R2-E13b: e13b_explore/analysis/{paired,check}.csv")
    for share, tag in (("0.05", "One"), ("0.2", "Four")):
        for r, rtag in ((49, "FortyNine"), (99, "NinetyNine")):
            m.add(f"exp{tag}{rtag}",
                  f(e.loc[(f"exploit_minus_control_share{share}", r), "mean"], 3, True))
    m.add("expOneTwoNinetyNine", f(e.loc[("exploit_minus_control_share0.05", 299), "mean"],
                                   3, True))
    chk = A("e13b_explore", "check.csv").set_index(["share", "round"])
    row = chk.loc[(0.05, 299)]
    if not row["lo_t95"] < 0 < row["hi_t95"]:
        raise ValueError("the prose says the round-299 check interval includes zero")
    m.add("checkExploitTwoNinetyNine", f(row["mean"], 4, True))
    m.add("checkExploitTwoNinetyNineLo", f(row["lo_t95"], 4, True))
    m.add("checkExploitTwoNinetyNineHi", f(row["hi_t95"], 4, True))
    for share, tag in ((0.05, "One"), (0.2, "Four")):
        for r, rtag in ((49, "FortyNine"), (99, "NinetyNine")):
            row = chk.loc[(share, r)]
            if not row["hi_t95"] < 0:
                raise ValueError("the prose says every round-49 and round-99 check excludes zero")
            m.add(f"check{tag}{rtag}", f(row["mean"], 3, True))
    u = pd.read_csv(H / "units.csv").set_index(["experiment", "condition"])
    warm, ctrl = u.loc[("R2-E13d", "warm start, confirmation")], u.loc[("R2-E9", "shipped, confirmation")]
    if not (warm["T05_rounds"] < ctrl["T05_rounds"] and warm["T05_steps"] > ctrl["T05_steps"]):
        raise ValueError("the prose says the warm arm is sooner in rounds and later in steps")
    m.add("TfiveWarm", n(round(warm["T05_rounds"])))
    m.add("TfiveControl", n(round(ctrl["T05_rounds"])))
    m.add("stepsTfiveWarm", thousands(warm["T05_steps"]))
    m.add("stepsTfiveControl", thousands(ctrl["T05_steps"]))

    w = A("e13d_warm", "summary.csv").set_index(["set", "quantity"])
    wp = A("e13d_warm", "per_seed.csv")
    m.comment("R2-E13d: e13d_warm/analysis/{summary,per_seed}.csv")
    m.add("reachZeroControl", f(wp["reach_0_control"].mean(), 3))
    m.add("reachZeroWarm", f(wp["reach_0_warm"].mean(), 3))
    m.add("gapZeroControl", f(wp["gap_0_control"].mean(), 3, True))
    m.add("gapZeroWarm", f(wp["gap_0_warm"].mean(), 3, True))
    for st, tag in (("confirmation", "Conf"), ("exploratory", "Expl")):
        row = w.loc[(st, "gap_round0_F299")]
        if not 0.15 <= row["mean"] <= 0.25:
            raise ValueError("the prose says the warm start removes about four fifths")
        m.add(f"warmShare{tag}", f(100 * row["mean"], 1))
        if tag == "Conf":
            m.add("warmShareConfLo", f(100 * row["lo_t95"], 1))
            m.add("warmShareConfHi", f(100 * row["hi_t95"], 1))
    if int((wp["reach_0_warm"] <= 0.8).sum()) != 1:
        raise ValueError("the prose says round-0 reach under the warm start is above 0.8 "
                         "on every seed but one")
    m.add("reachWarmMin", f(wp["reach_0_warm"].min(), 3))
    m.add("warmTerminalTwoNinetyNine",
          f(w.loc[("confirmation", "terminal_diff_299"), "mean"], 4, True))


def arrivals(m: Macros) -> None:
    s = A("e14_arrivals", "summary.csv").set_index("rate")
    tr = A("e14_arrivals", "trend.csv").set_index("rate")
    m.comment("R2-E14: e14_arrivals/analysis/{summary,trend}.csv")
    for rate, tag in ((0.005, "Low"), (0.01, "Mid"), (0.02, "High")):
        row = s.loc[rate]
        if int(row["steady_n_positive"]) != int(row["n_seeds"]) or row["steady_lo95"] <= MARGIN:
            raise ValueError("the prose says every seed is positive and every rate is above "
                             "the margin")
        t = tr.loc[rate]
        if not (t["hi_t95"] < 0 if tag == "Low" else t["lo_t95"] > 0):
            raise ValueError("the prose says the gap falls at the lowest rate and rises at "
                             "the other two")
        m.add(f"openGap{tag}", f(row["steady_gap"], 3, True))
        m.add(f"openGap{tag}Lo", f(row["steady_lo95"], 3, True))
        m.add(f"openGap{tag}Hi", f(row["steady_hi95"], 3, True))
        m.add(f"statP{tag}", f(row["stationary_paired_p"], 2))
        m.add(f"stepRatio{tag}", f(row["stationary_max_step"] / row["stationary_seed_sd"], 2))
        m.add(f"trend{tag}", f(tr.loc[rate, "change_200"], 4, True))
        m.add(f"trend{tag}Lo", f(tr.loc[rate, "lo_t95"], 4, True))
        m.add(f"trend{tag}Hi", f(tr.loc[rate, "hi_t95"], 4, True))
        m.add(f"cohortNew{tag}", f(row["ratio_t000_009_ncf"], 3))
        m.add(f"cohortOld{tag}", f(row["ratio_t050_up_ncf"], 3))
        if tag != "Low":
            m.add(f"cohortNew{tag}Oracle", f(row["ratio_t000_009_oracle"], 3))


def institutional(m: Macros) -> None:
    c = A("e15_institutional", "cells.csv")
    m.comment("R2-E15: e15_institutional/analysis/{cells,inert_c001}.csv")
    for arm, tag in (("oracle", "Oracle"), ("ncf", "Ncf")):
        g = c[c.a_arm == arm]
        wins = g[g.b_beats_a]
        if not (wins.signal_cost == 0.01).all():
            raise ValueError("the prose says every winning cell is at c_B = 0.01")
        m.add(f"nBeats{tag}", n(len(wins)))
        m.add(f"beats{tag}Min", f(wins.size_fraction.min(), 2))
        m.add(f"beats{tag}Max", f(wins.size_fraction.max(), 2))
        best = g.loc[g["mean"].idxmax()]
        m.add(f"best{tag}Size", f(best.size_fraction, 2))
        m.add(f"best{tag}", f(best["mean"], 1, True))
        m.add(f"best{tag}Lo", f(best["lo_t95"], 1, True))
        m.add(f"best{tag}Hi", f(best["hi_t95"], 1, True))
        m.add(f"welfareA{tag}", f(g["welfare_a"].iloc[0], 1))
    high = c[c.signal_cost >= 0.05]
    if high["b_beats_a"].any():
        raise ValueError("the prose says no cell beats A after correction at c_B >= 0.05")
    inert = A("e15_institutional", "inert_c001.csv").set_index("size_fraction")
    row = inert.loc[0.65]
    if int(row["rank"]) != 1:
        raise ValueError("the prose says size 0.65 ranks first in the inert gate")
    half = float(stats.t.ppf(0.975, row["size"] - 1)) * row["std"] / math.sqrt(row["size"])
    m.add("inertBest", f(row["mean"], 1, True))
    m.add("inertBestLo", f(row["mean"] - half, 1, True))
    m.add("inertBestHi", f(row["mean"] + half, 1, True))
    cf = A("e15s_five", "cells.csv")
    if cf["b_beats_a"].any() or (cf["mean"] >= 0).any():
        raise ValueError("the prose says no five-round cell has B ahead of A")
    cell = cf[(cf.signal_cost == 0.01) & (cf.size_fraction == 0.65)].iloc[0]
    m.comment("R2-E15s: e15s_five/analysis/cells.csv")
    m.add("nABeatsFive", n(int(cf["a_beats_b"].sum())))
    m.add("fiveCell", f(cell["mean"], 1, True))
    m.add("fiveCellLo", f(cell["lo_t95"], 1, True))
    m.add("fiveCellHi", f(cell["hi_t95"], 1, True))


def sensitivity(m: Macros) -> None:
    s = A("e16_fp", "summary.csv").iloc[0]
    sp = A("e16_fp", "spread.csv").set_index("round")
    m.comment("R2-E16: e16_fp/analysis/{summary,spread}.csv")
    m.add("fpFirstRound", n(s["first_round_any_run_differs"]))
    m.add("fpRatioNinetyNine", f(sp.loc[99, "ratio"], 2))
    m.add("fpRatioTwoNinetyNine", f(sp.loc[299, "ratio"], 2))


def appendix(m: Macros) -> None:
    h1 = A("h1r", "curve.csv").set_index("signal_cost")
    h5 = A("h5r", "terms.csv").set_index("term")
    m.comment("R2-H1r and R2-H5r: h1r/analysis/curve.csv, h5r/analysis/terms.csv")
    m.add("hOneLow", f(h1.loc[0.0, "mean_tau"], 3))
    m.add("hOneHigh", f(h1.loc[2.0, "mean_tau"], 3))
    for term, tag in (("interaction", "Int"), ("algorithmic", "Alg"), ("exogenous", "Exo")):
        m.add(f"hFive{tag}", f(h5.loc[term, "mean"], 1, True))
        m.add(f"hFive{tag}Lo", f(h5.loc[term, "lo_t95"], 1, True))
        m.add(f"hFive{tag}Hi", f(h5.loc[term, "hi_t95"], 1, True))
    reg = pd.read_csv(H / "registrations.csv").set_index("key")
    m.comment("Registration commits: horizon/registrations.csv (from git history)")
    m.add("regCommit", reg.loc["main", "commit"])
    m.add("regCommitWarm", reg.loc["warm", "commit"])
    m.add("regCommitCheckB", reg.loc["check_b", "commit"])
    m.add("regCommitCheckC", reg.loc["check_c", "commit"])
    m.add("regCommitRepair", reg.loc["repair", "commit"])


def render() -> str:
    m = Macros()
    m.comment("Generated by paper/gen_numbers.py from data/processed/. Do not edit.")
    for section in (transient, floor, where, what_ends, arrivals, institutional,
                    sensitivity, appendix):
        section(m)
    return "\n".join(m.lines) + "\n"


def main() -> None:
    text = render()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({text.count(chr(92) + 'newcommand')} macros)")


if __name__ == "__main__":
    main()
