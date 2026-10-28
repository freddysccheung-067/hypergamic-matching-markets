"""Every figure of the paper, from the CSVs under data/processed/horizon/.

    python paper/figures.py

Writes paper/figures/<name>.pdf. Each function names the files it reads, and
each figure's caption in the paper names them again. Colors are the validated
categorical order (blue, orange, aqua, yellow) for identities and one blue ramp,
light to dark, for ordered magnitudes; every series also differs in line style or
marker, so a figure printed in grayscale keeps its encoding.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
H = ROOT / "data" / "processed" / "horizon"
OUT = pathlib.Path(__file__).resolve().parent / "figures"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
INK, INK2, GRID, SEEDLINE = "#0b0b0b", "#52514e", "#e4e3df", "#c9c8c2"
MARGIN = 0.002
FULL = 6.5  # text width, inches

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 1.4, "pdf.fonttype": 42,
    "axes.titlelocation": "left", "axes.titleweight": "bold",
})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote paper/figures/{name}.pdf")


def margin_band(ax) -> None:
    ax.axhspan(-MARGIN, MARGIN, color="#d9d8d3", alpha=0.6, lw=0, zorder=0)
    ax.axhline(0, color=INK2, lw=0.6, zorder=1)


def seeds_wide(directory: pathlib.Path, seeds) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(directory.glob("seed*.csv"))]
    d = pd.concat(frames)
    d = d[d.seed.isin(seeds)]
    p = d.pivot_table(index=["seed", "round"], columns="arm", values="gini")
    return (p["ncf"] - p["oracle"]).unstack(0)


def fig_transient() -> None:
    """R2-E9 and R2-E12: e09_canonical/seed*.csv, analysis/series.csv, per_seed.csv."""
    s = pd.read_csv(H / "e09_canonical" / "analysis" / "series.csv")
    per = pd.read_csv(H / "e09_canonical" / "analysis" / "per_seed.csv")
    conf, expl = s[s.set == "confirmation"], s[s.set == "exploratory"]
    wide = seeds_wide(H / "e09_canonical", range(100, 120))
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.5), gridspec_kw={"width_ratios": [1.35, 1]})
    margin_band(a)
    for col in wide.columns:
        a.plot(wide.index, wide[col], color=SEEDLINE, lw=0.4, zorder=2)
    a.fill_between(conf["round"], conf["gap_lo_t95"], conf["gap_hi_t95"], color=BLUE,
                   alpha=0.25, lw=0, zorder=3)
    a.plot(conf["round"], conf["gap_mean"], color=BLUE, zorder=4, label="confirmation, mean")
    a.plot(expl["round"], expl["gap_mean"], color=ORANGE, ls="--", lw=1.1, zorder=4,
           label="exploratory, mean")
    entries = per[per.set == "confirmation"]["entry_round"]
    a.plot(entries, np.full(len(entries), -0.012), "|", color=INK, ms=5, mew=0.7,
           label="seed enters margin")
    a.set(xlabel="round", ylabel="exposure Gini gap, NCF $-$ oracle", xlim=(0, 300),
          ylim=(-0.02, 0.34), title="(a) the gap, 20 + 20 seeds")
    a.legend(loc="upper right")
    b.plot(conf["round"], conf["reach_ncf"], color=BLUE, label="reach (NCF)")
    b.plot(conf["round"], conf["jaccard_oracle"], color=ORANGE, ls="--",
           label="Jaccard with oracle shortlist")
    b.plot(conf["round"], conf["spearman_mean"], color=AQUA, ls=":", lw=1.6,
           label="Spearman, scores vs utility")
    b.plot(conf["round"], conf["modal_share"], color=INK2, ls="-.", lw=1.0,
           label="share in modal list")
    b.set(xlabel="round", ylabel="share or correlation", xlim=(0, 300), ylim=(0, 1.02),
          title="(b) the ranker's lists, confirmation")
    # Below the axes: inside, any placement covers one of the four series.
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=6.5,
             columnspacing=1.0, handlelength=2.2)
    save(fig, "transient")


def fig_arms() -> None:
    """R2-E9: e09_canonical/analysis/series.csv, every arm's exposure Gini."""
    s = pd.read_csv(H / "e09_canonical" / "analysis" / "series.csv")
    c = s[s.set == "confirmation"]
    fig, ax = plt.subplots(figsize=(FULL * 0.62, 2.3))
    spec = [("popularity", YELLOW, "-", "popularity"), ("ncf", BLUE, "-", "NCF"),
            ("oracle", ORANGE, "--", "oracle"), ("random", AQUA, ":", "random")]
    # NCF and the oracle end at the same Gini, so their labels are offset apart.
    offsets = {"popularity": 0, "ncf": 6, "oracle": -6, "random": 0}
    for arm, color, ls, label in spec:
        ax.plot(c["round"], c[f"gini_{arm}"], color=color, ls=ls, label=label)
        ax.annotate(label, (300, c[f"gini_{arm}"].iloc[-1]), xytext=(4, offsets[arm]),
                    textcoords="offset points", va="center", fontsize=7, color=INK)
    ax.set(xlabel="round", ylabel="exposure Gini", xlim=(0, 300), ylim=(0.3, 1.0))
    save(fig, "arms")


def fig_decay() -> None:
    """R2-E10: e10_long/analysis/series.csv and fits.csv."""
    s = pd.read_csv(H / "e10_long" / "analysis" / "series.csv")
    fits = pd.read_csv(H / "e10_long" / "analysis" / "fits.csv").set_index("model")
    r = s["round"].to_numpy()
    peak = int(fits.loc["exp", "peak_round"])
    x = np.clip(r - peak + 1.0, 1.0, None)
    forms = {
        "exp": (lambda p: p[0] * np.exp(-p[1] * x), BLUE, "--", "exponential"),
        "exp_const": (lambda p: p[0] + p[1] * np.exp(-p[2] * x), ORANGE, "-.",
                      "exponential + constant"),
        "power": (lambda p: p[0] * np.power(x, -p[1]), AQUA, ":", "power law"),
    }
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.4), gridspec_kw={"wspace": 0.3})
    for ax, ylim, xlim in ((a, (-0.01, 0.34), (0, 1000)), (b, (-0.0045, 0.0045), (150, 1000))):
        # Here the shading marks the held-out rounds, so the margin is drawn as
        # dotted lines rather than as a band of the same kind.
        ax.axhline(0, color=INK2, lw=0.6, zorder=1)
        for y in (-MARGIN, MARGIN):
            ax.axhline(y, color=INK2, lw=0.8, ls=(0, (1, 2)), zorder=1)
        ax.axvspan(500.5, 1000, color="#f1f0ec", lw=0, zorder=0)
        ax.fill_between(r, s["lo_t95"], s["hi_t95"], color=INK2, alpha=0.25, lw=0, zorder=2)
        ax.plot(r, s["mean"], color=INK, lw=1.0, label="mean gap, 10 seeds", zorder=3)
        for model, (fn, color, ls, label) in forms.items():
            p = [float(v) for v in fits.loc[model, "params"].split()]
            y = fn(p)
            y[r < peak] = np.nan
            ax.plot(r, y, color=color, ls=ls, lw=1.3, label=label, zorder=4)
        ax.set(xlabel="round", ylabel="exposure Gini gap", xlim=xlim, ylim=ylim)
    a.set_title("(a) fit to round 500")
    b.set_title("(b) the tail, finer scale")
    a.legend(loc="upper right")
    save(fig, "decay")


def fig_dose() -> None:
    """R2-E13a: e13a_dose/analysis/series.csv."""
    s = pd.read_csv(H / "e13a_dose" / "analysis" / "series.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.4), sharey=True)
    styles = {0.5: ("-", "o"), 1.0: ("--", "s"), 2.0: ("-.", "^"), 4.0: (":", "D")}
    for color, dose in zip(RAMP, (0.5, 1.0, 2.0, 4.0)):
        d = s[s.dose == dose]
        ls, mk = styles[dose]
        label = f"{dose:g} pass{'es' if dose != 1 else ''} per round"
        for ax, x in ((a, d["round"]), (b, d["grad_steps"])):
            ax.plot(x, d["mean"], color=color, ls=ls, marker=mk, markevery=25, ms=3,
                    label=label)
    for ax in (a, b):
        margin_band(ax)
        ax.set_ylim(-0.01, 0.34)
    a.set(xlabel="round", ylabel="exposure Gini gap", xlim=(0, 300),
          title="(a) against rounds")
    b.set(xlabel="cumulative optimizer steps", xlim=(0, 8000),
          title="(b) against optimizer steps")
    a.legend(loc="upper right")
    save(fig, "dose")


def fig_feedback() -> None:
    """R2-E13c and R2-E13b: e13c_offpolicy/analysis/series.csv, e13b_explore/analysis/series.csv."""
    c = pd.read_csv(H / "e13c_offpolicy" / "analysis" / "series.csv")
    e = pd.read_csv(H / "e13b_explore" / "analysis" / "series.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.4))
    on = c[c.data_from == "oracle"]
    a.plot(on["round"], on["on_policy"], color=BLUE, label="on-policy (the market's own data)")
    for src, color, ls in (("oracle", ORANGE, "--"), ("random", AQUA, ":")):
        d = c[c.data_from == src]
        a.plot(d["round"], d["off_policy"], color=color, ls=ls, lw=1.5,
               label=f"off-policy, {src} market's data")
    ctrl = e[e.share == 0.05]
    b.plot(ctrl["round"], ctrl["control"], color=BLUE, label="no exploration")
    for share, color, ls in ((0.05, ORANGE, "--"), (0.2, AQUA, ":")):
        d = e[e.share == share]
        b.plot(d["round"], d["exploit"], color=color, ls=ls, lw=1.5,
               label=f"exploiting part, share {share:g}")
        b.plot(d["round"], d["whole"], color=color, ls="-", lw=0.7, alpha=0.8,
               label=f"whole list, share {share:g}")
    # The whole-list gap at share 0.2 goes well below zero, so (b) has its own
    # scale, set from the data so no curve leaves the axes.
    low = float(e[e["round"] <= 150][["exploit", "whole", "control"]].min().min())
    for ax, ylim in ((a, (-0.01, 0.34)), (b, (min(-0.01, low - 0.015), 0.34))):
        margin_band(ax)
        ax.set(xlabel="round", xlim=(0, 150), ylim=ylim)
    a.set(ylabel="exposure Gini gap", title="(a) whose data the ranker trains on")
    b.set_title("(b) random exploration slots")
    a.legend(loc="upper right")
    b.legend(loc="upper right")
    save(fig, "feedback")


CELL_LABELS = {
    "shipped": "shipped", "taste_sigma_0.2": "taste 0.2", "taste_sigma_1.0": "taste 1.0",
    "shock_scale_0.5": "chemistry 0.5", "shock_scale_3.0": "chemistry 3.0",
    "beta_alpha_1.0": "$\\beta/\\alpha$ 1", "beta_alpha_3.0": "$\\beta/\\alpha$ 3",
    "delta_0.0": "$\\delta$ 0", "delta_1.0": "$\\delta$ 1",
}


def fig_sweep() -> None:
    """R2-E11: e11_sweep/analysis/cells.csv."""
    c = pd.read_csv(H / "e11_sweep" / "analysis" / "cells.csv")
    c["label"] = c["cell"].map(CELL_LABELS)
    c = c.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(c))
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.6), sharey=True)
    a.errorbar(c["peak_mean"], y, xerr=[c["peak_mean"] - c["peak_lo"], c["peak_hi"] - c["peak_mean"]],
               fmt="o", color=BLUE, ms=4, capsize=2, lw=0.9)
    a.set(xlabel="peak gap (mean, 95% t interval)", yticks=y, yticklabels=c["label"],
          title="(a) peak gap")
    # Each seed's entry round; a seed still outside the margin at round 299 is drawn
    # as an open marker at the edge, so a censored seed is never hidden in a median.
    per = pd.read_csv(H / "e11_sweep" / "analysis" / "per_seed.csv")
    rng = np.random.default_rng(0)
    for yi, cell in zip(y, c["cell"]):
        g = per[per.cell == cell]
        jit = rng.uniform(-0.18, 0.18, len(g))
        ok = ~g["censored"].to_numpy()
        b.plot(g["entry_round"].to_numpy()[ok], yi + jit[ok], "o", color=ORANGE, ms=3,
               alpha=0.85)
        if (~ok).any():
            b.plot(np.full((~ok).sum(), 305.0), yi + jit[~ok], "o", mfc="none",
                   color=INK2, ms=3.5)
    b.axvline(299.5, color=INK2, lw=0.6, ls="--")
    b.set(xlabel="round of entering the margin, per seed", xlim=(150, 312),
          title="(b) when it ends")
    save(fig, "sweep")


def fig_arrivals() -> None:
    """R2-E14: e14_arrivals/analysis/series.csv and summary.csv; R2-E9 for no arrivals."""
    s = pd.read_csv(H / "e14_arrivals" / "analysis" / "series.csv")
    m = pd.read_csv(H / "e14_arrivals" / "analysis" / "summary.csv")
    base = pd.read_csv(H / "e09_canonical" / "analysis" / "series.csv")
    base = base[base.set == "confirmation"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.7),
                               gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.38})
    margin_band(a)
    a.plot(base["round"], base["gap_mean"], color=RAMP[0], ls=":", lw=1.5,
           label="no arrivals (to round 299)")
    for color, (rate, ls) in zip(RAMP[1:], ((0.005, "-."), (0.01, "--"), (0.02, "-"))):
        d = s[s.rate == rate]
        a.plot(d["round"], d["mean"], color=color, ls=ls, label=f"{100 * rate:g}% per round")
        a.fill_between(d["round"], d["lo_t95"], d["hi_t95"], color=color, alpha=0.2, lw=0)
    a.axvspan(500, 600, color="#f1f0ec", lw=0, zorder=0)
    a.set(xlabel="round", ylabel="exposure Gini gap", xlim=(0, 600), ylim=(-0.01, 0.34),
          title="(a) the gap with replacement")
    a.legend(loc="upper right")
    bands = [("t000_009", "tenure < 10"), ("t010_049", "10 to 49"), ("t050_up", "50 or more")]
    x = np.arange(len(m))
    width = 0.26
    for j, ((band, label), mk) in enumerate(zip(bands, ("o", "s", "^"))):
        off = (j - 1) * width
        b.plot(x + off, m[f"ratio_{band}_ncf"], mk, color=RAMP[3 - j], ms=5, label=label)
        b.plot(x + off, m[f"ratio_{band}_oracle"], mk, mfc="none", color=INK2, ms=5)
    b.axhline(1.0, color=INK2, lw=0.6)
    b.set(xticks=x, xticklabels=[f"{100 * r:g}%" for r in m["rate"]], xlabel="replacement rate",
          ylabel="exposure / population share", title="(b) by tenure, rounds 500 to 599",
          xlim=(-0.5, len(m) - 0.5))
    # Below the axes, where it covers no marker; filled is NCF, open is the oracle.
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=6.5,
             title="filled: NCF; open: oracle", title_fontsize=6.5, handletextpad=0.2,
             columnspacing=0.8)
    save(fig, "arrivals")


def fig_institutional() -> None:
    """R2-E15: e15_institutional/analysis/cells.csv."""
    c = pd.read_csv(H / "e15_institutional" / "analysis" / "cells.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.4), sharey=False)
    for arm, color, mk, dx in (("oracle", ORANGE, "o", -0.006), ("ncf", BLUE, "s", 0.006)):
        d = c[(c.a_arm == arm) & (c.signal_cost == 0.01)].sort_values("size_fraction")
        a.errorbar(d["size_fraction"] + dx, d["mean"], yerr=[d["mean"] - d["lo_t95"],
                   d["hi_t95"] - d["mean"]], fmt=mk + "-", color=color, ms=3.5, capsize=2,
                   lw=0.9, label=f"{'oracle' if arm == 'oracle' else 'NCF'} A")
    a.axhline(0, color=INK2, lw=0.6)
    a.set(xlabel="share of the population in B", ylabel="welfare, B $-$ A (rounds 250 to 299)",
          title="(a) $c_B = 0.01$")
    a.legend(loc="upper left")
    for cost, color, ls in ((0.05, RAMP[1], "--"), (0.10, RAMP[2], "-."), (0.25, RAMP[3], ":")):
        d = c[(c.a_arm == "ncf") & (c.signal_cost == cost)].sort_values("size_fraction")
        b.plot(d["size_fraction"], d["mean"], color=color, ls=ls, marker="o", ms=3,
               label=f"$c_B = {cost:g}$")
    b.axhline(0, color=INK2, lw=0.6)
    b.set(xlabel="share of the population in B", title="(b) higher costs, NCF A")
    b.legend(loc="lower left")
    save(fig, "institutional")


def fig_fp() -> None:
    """R2-E16: e16_fp/analysis/spread.csv."""
    s = pd.read_csv(H / "e16_fp" / "analysis" / "spread.csv")
    fig, ax = plt.subplots(figsize=(FULL * 0.62, 2.2))
    ax.plot(s["round"], s["between_sd"], color=BLUE, label="between seeds")
    ax.plot(s["round"], s["within_sd"], color=ORANGE, ls="--", label="within a seed")
    ax.set(xlabel="round", ylabel="sd of the gap", xlim=(0, 300))
    ax.legend(loc="upper right")
    save(fig, "fp")


def main() -> None:
    for fn in (fig_transient, fig_arms, fig_decay, fig_dose, fig_feedback, fig_sweep,
               fig_arrivals, fig_institutional):
        fn()
    if (H / "e16_fp" / "analysis" / "spread.csv").exists():
        fig_fp()


if __name__ == "__main__":
    main()
