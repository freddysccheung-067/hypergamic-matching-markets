"""The three headline figures (THESIS.md, "Structure of the work": Presentation).

Each function reads the committed CSVs under ``data/processed/``, draws one figure with
the control that makes it interpretable, saves a PNG to ``docs/figures/`` and returns the
``Figure``. The captions in ``CAPTIONS`` state what each figure does *not* show; they are
text rather than pixels so a README can render them under the image.

Figures are built on ``matplotlib.figure.Figure`` directly rather than through pyplot, so
nothing here touches a global backend or leaks open figures.

Spread everywhere is the standard deviation over the five seeds (ddof = 1), the
convention of THEORY.md §10.5 and §10.6.
"""

from __future__ import annotations

import pathlib

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
FIGURES = ROOT / "docs" / "figures"

# Okabe-Ito: distinguishable under the common colour-vision deficiencies. Colour is never
# the only channel; line style, marker and direct labels carry the same information.
BLUE, VERMILLION, GREEN, ORANGE, SKY, PURPLE = (
    "#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7",
)
INK, GREY, LIGHT = "#222222", "#6B6B6B", "#D9D9D9"

DPI = 150

_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#EAEAEA",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "font.size": 10,
    "axes.titlesize": 11,
    "legend.frameon": False,
    "mathtext.default": "regular",
}

CAPTIONS = {
    "separating_curve": (
        "**Figure 1. The separating curve (H1).** Points: mean intent of the agents who "
        "propose, E[τ | proposal], against the signalling cost c, with ±1 sd over seeds "
        "42-46 (N = 1000, epoch 0). Dashed line: the analytic cut τ*(c) = c / (V̄·p). "
        "Solid line: the mean intent a pure cut at τ* would leave, computed from the same "
        "population. The mean rises with c, so H1 is not killed. It sits below the pure-cut "
        "line, which is what a soft cut gives: the decision compares τ·V_ij·p with c pair "
        "by pair and V_ij varies, so a low-intent agent can still clear it against a "
        "high-value candidate. **What it does not show:** that real proposers face this cost "
        "or this utility scale (c is only meaningful against V̄ ≈ 4.7, a modelling choice), "
        "or that the accept rate the formula assumes is the one the market produces. "
        "The decision is handed p = 0.5, the epoch-0 prior, while the measured share of "
        "positive swipes that are reciprocated is 0.013-0.041. The curve is a statement "
        "about epoch 0 under that prior, not about the closed loop."
    ),
    "thin_market_curve": (
        "**Figure 2. The thin-market curve (H4).** Welfare against the share of the "
        "population drawn into Environment B, with Environment A (oracle) as the "
        "horizontal line and its seed spread shaded. Solid: welfare_common, residual "
        "accounting (§10.4), the number the thesis rests on. Dashed: welfare_selected, "
        "entrants only. The gap between the two is the selection artifact and it changes "
        "sign: selected sits far below common for small B, far above it in the middle "
        "(up to +312 at size 0.25), and below it again at full size, so the dashed curve "
        "is not a conservative or an optimistic bound. Bands are ±1 sd over seeds 42-46. "
        "Dotted and dashed verticals mark the critical mass N* of §10.5 (ρ = 0.5, "
        "acceptability bars q = 0.95 and 0.99), converted to a size fraction through the "
        "grid's own pool sizes; the θ bar's N* = 3.1 is left of the first grid point. "
        "**What it does not show:** that any real community is at the implied size; that "
        "the interior optimum survives a higher cost (right panel, c_B = 0.05, is the "
        "kill condition: B never beats A) or an NCF-mediated A (against which B wins at "
        "every size); or that both slopes are the thin-market penalty. The right-hand "
        "slope comes from top-k truncation, so N* locates the left slope only."
    ),
    "decomposition": (
        "**Figure 3. The decomposition (H5).** Left: welfare W as a waterfall from the "
        "baseline (no drift, oracle) through the exogenous status drift, the move to NCF "
        "mediation, and their interaction, to the both-on cell. Right: the three terms on "
        "their own scale, mean ± 1 sd over seeds with each seed as a dot, because the "
        "algorithmic term is large enough to hide the other two on the left. The "
        "interaction is the term that would support amplification; here it is "
        "indistinguishable from zero. **What it does not show:** that algorithms amplify "
        "anything. That claim needs a non-zero interaction and this one has t ≈ 0.9 on 4 df "
        "with the sign changing across seeds. The algorithmic bar is the NCF's estimation "
        "error against an oracle (H3), not a measurement of any deployed recommender."
    ),
}


# ── plumbing ─────────────────────────────────────────────────────────────────

def _new_figure(width: float, height: float, **kw) -> Figure:
    fig = Figure(figsize=(width, height), facecolor="white", **kw)
    return fig


def _save(fig: Figure, name: str, out_dir: pathlib.Path | str | None) -> pathlib.Path:
    out = pathlib.Path(out_dir) if out_dir is not None else FIGURES
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    return path


def _read(data_dir: pathlib.Path | str | None, *parts: str) -> pd.DataFrame:
    base = pathlib.Path(data_dir) if data_dir is not None else DATA
    path = base.joinpath(*parts)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing; regenerate it with the script named in its README"
        )
    return pd.read_csv(path)


def _by_seed_mean_sd(df: pd.DataFrame, key: str, col: str) -> pd.DataFrame:
    g = df.groupby(key)[col]
    return pd.DataFrame({"mean": g.mean(), "sd": g.std(ddof=1)}).reset_index()


# ── 1. the separating curve ──────────────────────────────────────────────────

def separating_curve(
    data_dir: pathlib.Path | str | None = None,
    out_dir: pathlib.Path | str | None = None,
    save: bool = True,
) -> Figure:
    """E[τ | proposal] against c, measured, with the analytic cut overlaid (H1)."""
    h1 = _read(data_dir, "separating", "h1.csv")
    meas = _by_seed_mean_sd(h1.dropna(subset=["mean_tau"]), "signal_cost", "mean_tau")
    trunc = _by_seed_mean_sd(h1.dropna(subset=["tau_trunc"]), "signal_cost", "tau_trunc")
    cut = h1.groupby("signal_cost")["tau_star"].mean().reset_index()
    pop_mean = float(h1[h1.signal_cost == 0.0]["mean_tau"].mean())

    with mpl.rc_context(_STYLE):
        fig = _new_figure(6.6, 4.4)
        ax = fig.add_subplot(111)

        ax.axhline(pop_mean, color=GREY, lw=1, ls=":", zorder=1)
        ax.text(
            cut.signal_cost.max(), pop_mean - 0.025, "pooling: E[τ | proposal] = E[τ]",
            ha="right", va="top", fontsize=8.5, color=GREY,
        )

        ax.plot(cut.signal_cost, cut.tau_star, color=VERMILLION, lw=2, ls="--",
                label=r"analytic cut $\tau^*(c)=c\,/\,(\bar{V}\,p)$, $p=0.5$", zorder=3)
        ax.plot(trunc.signal_cost, trunc["mean"], color=GREEN, lw=2, ls="-",
                label=r"mean intent a pure cut at $\tau^*$ would leave", zorder=3)
        ax.errorbar(
            meas.signal_cost, meas["mean"], yerr=meas["sd"], color=BLUE, marker="o",
            ms=5.5, lw=0, elinewidth=1.4, capsize=3, zorder=4,
            label=r"measured $E[\tau\,|\,\mathrm{proposal}]$, ±1 sd over 5 seeds",
        )

        p_lo, p_hi = float(h1.phat.min()), float(h1.phat.max())
        ax.text(
            0.02, 0.97,
            f"decision assumes p = 0.5\nmeasured reciprocity {p_lo:.3f}-{p_hi:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=INK,
            bbox={"boxstyle": "round,pad=0.35", "fc": "#F6F6F6", "ec": LIGHT},
        )

        ax.set_xlabel("signalling cost  c")
        ax.set_ylabel("intent  τ")
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(-0.05, float(h1.signal_cost.max()) + 0.05)
        ax.set_title("H1: a cost on proposing selects on intent", loc="left")
        ax.legend(loc="lower right", fontsize=8.5)
        fig.tight_layout()

    if save:
        _save(fig, "separating_curve", out_dir)
    return fig


# ── 2. the thin-market curve ─────────────────────────────────────────────────

def _critical_mass_as_fraction(
    summary: pd.DataFrame, n_star: float
) -> float:
    """N* (a pool size) as a size_fraction, from the grid's own eligible counts.

    Linear interpolation through the origin: |B| = 0 at size_fraction = 0.
    """
    x = np.r_[0.0, summary.size_fraction.to_numpy()]
    y = np.r_[0.0, summary.n_eligible_mean.to_numpy()]
    return float(np.interp(n_star, y, x))


def thin_market_curve(
    data_dir: pathlib.Path | str | None = None,
    out_dir: pathlib.Path | str | None = None,
    save: bool = True,
    costs: tuple[float, float] = (0.01, 0.05),
) -> Figure:
    """welfare_common and welfare_selected against size_fraction, A as a line (H4)."""
    grid = _read(data_dir, "institutional", "grid.csv")
    tm = {
        q: float(_read(data_dir, d, "curve_mean.csv").critical_mass.iloc[0])
        for q, d in (("θ", "thin_market"), ("0.95", "thin_market_q0.95"),
                     ("0.99", "thin_market_q0.99"))
    }

    with mpl.rc_context(_STYLE):
        fig = _new_figure(10.4, 4.5)
        axes = fig.subplots(1, len(costs), sharey=True)
        axes = np.atleast_1d(axes)

        for ax, cb in zip(axes, costs):
            sub = grid[np.isclose(grid.signal_cost, cb)]
            common = _by_seed_mean_sd(sub, "size_fraction", "welfare_common_residual_b")
            selected = _by_seed_mean_sd(sub, "size_fraction", "welfare_selected_b")
            a = sub.groupby("seed").welfare_common_a.first()
            a_mean, a_sd = float(a.mean()), float(a.std(ddof=1))
            summ = sub.groupby("size_fraction").n_eligible.mean().reset_index().rename(
                columns={"n_eligible": "n_eligible_mean"})

            ax.axhspan(a_mean - a_sd, a_mean + a_sd, color=GREY, alpha=0.12, lw=0)
            ax.axhline(a_mean, color=INK, lw=1.8, zorder=2,
                       label=f"Environment A (oracle), {a_mean:.0f}")

            ax.fill_between(selected.size_fraction, selected["mean"] - selected["sd"],
                            selected["mean"] + selected["sd"], color=ORANGE, alpha=0.15, lw=0)
            ax.plot(selected.size_fraction, selected["mean"], color=ORANGE, lw=2, ls="--",
                    marker="s", ms=4.5, zorder=3,
                    label="B, welfare_selected (entrants only)")
            ax.fill_between(common.size_fraction, common["mean"] - common["sd"],
                            common["mean"] + common["sd"], color=BLUE, alpha=0.18, lw=0)
            ax.plot(common.size_fraction, common["mean"], color=BLUE, lw=2.2, ls="-",
                    marker="o", ms=5, zorder=4,
                    label="B, welfare_common (residual accounting)")

            # N* of §10.5, converted from a pool size to a size fraction. The θ bar's
            # N* = 3.1 lies below the smallest grid point and is not drawn.
            for q, style in (("0.95", (0, (1, 2))), ("0.99", (0, (4, 2)))):
                xf = _critical_mass_as_fraction(summ, tm[q])
                ax.axvline(xf, color=PURPLE, lw=1.4, ls=style, zorder=1)
                ax.text(xf + 0.008, 0.98, f"N* = {tm[q]:.1f} (q = {q})",
                        transform=ax.get_xaxis_transform(), fontsize=8, color=PURPLE,
                        va="top", ha="left", rotation=90)

            # The largest paired gap over A, annotated only where it exceeds its own seed
            # spread. A "best" cell that is inside the noise is not a finding.
            gap = sub.groupby("size_fraction").welfare_diff_residual.agg(["mean", "std"])
            best_x = float(gap["mean"].idxmax())
            if gap.loc[best_x, "mean"] > gap.loc[best_x, "std"]:
                by = float(common.set_index("size_fraction").loc[best_x, "mean"])
                ax.annotate(
                    f"max gap over A: {gap.loc[best_x, 'mean']:+.1f} ± {gap.loc[best_x, 'std']:.1f}"
                    f"\nat size {best_x:.2f}",
                    xy=(best_x, by), xytext=(best_x, by + 330), fontsize=8.5, ha="center",
                    arrowprops={"arrowstyle": "-", "color": GREY, "lw": 0.9},
                )
            else:
                ax.text(0.97, 0.97, "no size beats A beyond seed noise",
                        transform=ax.transAxes, fontsize=8.5, ha="right", va="top", color=INK)
            ax.set_title(f"c_B = {cb:g}", loc="left")
            ax.set_xlabel("size fraction of population drawn into B")
            ax.set_xlim(0.0, 1.03)

        axes[0].set_ylabel("welfare W")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.5,
                   bbox_to_anchor=(0.5, -0.02))
        fig.suptitle("H4: is there an interior optimum?", x=0.01, ha="left", fontsize=11.5)
        fig.tight_layout(rect=(0, 0.06, 1, 1))

    if save:
        _save(fig, "thin_market_curve", out_dir)
    return fig


# ── 3. the decomposition ─────────────────────────────────────────────────────

def decomposition_terms(data_dir: pathlib.Path | str | None = None) -> dict:
    """The §10.6 numbers the figure draws, one place so the test can check them."""
    f = _read(data_dir, "institutional", "factorial.csv")
    n = len(f)
    out = {"n_seeds": n, "baseline": float(f.W_baseline.mean()), "both": float(f.W_both.mean())}
    for term in ("exogenous", "algorithmic", "interaction"):
        v = f[term].to_numpy(dtype=float)
        sd = float(v.std(ddof=1))
        out[term] = {
            "mean": float(v.mean()), "sd": sd, "se": sd / np.sqrt(n),
            "t": float(v.mean() / (sd / np.sqrt(n))) if sd > 0 else float("nan"),
            "per_seed": v,
        }
    return out


def decomposition(
    data_dir: pathlib.Path | str | None = None,
    out_dir: pathlib.Path | str | None = None,
    save: bool = True,
) -> Figure:
    """Waterfall: baseline, +exogenous, +algorithmic, +interaction (H5).

    Drawn so a zero-width bar is still visible: every step gets a marker line at its
    running total, an explicit signed label and a hatch keyed to the sign, and the
    right-hand panel puts the three terms on their own scale where a near-zero
    interaction is the point rather than an omission.
    """
    d = decomposition_terms(data_dir)
    steps = [("exogenous\ndrift", d["exogenous"]), ("algorithmic\n(NCF)", d["algorithmic"]),
             ("interaction", d["interaction"])]

    with mpl.rc_context(_STYLE):
        fig = _new_figure(10.6, 4.7)
        ax, az = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.35, 1]})

        # ── left: levels ──
        labels = ["baseline"] + [s[0] for s in steps] + ["both on"]
        run = d["baseline"]
        ax.bar(0, run, color=GREY, width=0.62, zorder=3)
        ax.text(0, run + 15, f"{run:.0f}", ha="center", va="bottom", fontsize=9)
        for i, (_, t) in enumerate(steps, start=1):
            new = run + t["mean"]
            lo, hi = sorted((run, new))
            up = t["mean"] >= 0
            ax.bar(i, max(hi - lo, 0.0), bottom=lo, width=0.62, zorder=3,
                   color=GREEN if up else VERMILLION, hatch="" if up else "//",
                   edgecolor="white" if up else "white", linewidth=0.6)
            # Running-total tick: a zero-height bar must still be seen.
            ax.plot([i - 0.31, i + 0.31], [new, new], color=INK, lw=1.4, zorder=4)
            ax.plot([i - 1 + 0.31, i - 0.31], [run, run], color=GREY, lw=0.9, ls=":", zorder=2)
            ax.text(i, hi + 15, f"{t['mean']:+.1f}", ha="center", va="bottom", fontsize=9)
            run = new
        ax.plot([len(steps) + 0.31 - 1, len(steps) + 0.69], [run, run],
                color=GREY, lw=0.9, ls=":", zorder=2)
        ax.bar(len(steps) + 1, d["both"], color=BLUE, width=0.62, zorder=3)
        ax.text(len(steps) + 1, d["both"] + 15, f"{d['both']:.0f}", ha="center",
                va="bottom", fontsize=9)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("welfare W (mean over seeds)")
        ax.set_ylim(0, max(d["baseline"], d["both"]) * 1.15)
        ax.set_title("levels", loc="left")
        ax.grid(axis="x", visible=False)

        # ── right: the three terms, own scale ──
        az.axhline(0.0, color=INK, lw=1.2, zorder=2)
        for i, (name, t) in enumerate(steps):
            rng = np.random.default_rng(0)
            jitter = (rng.random(len(t["per_seed"])) - 0.5) * 0.18
            az.scatter(np.full(len(t["per_seed"]), i) + jitter, t["per_seed"],
                       s=16, color=GREY, alpha=0.75, zorder=3, label="one seed" if i == 0 else None)
            az.errorbar(i, t["mean"], yerr=t["sd"], color=BLUE, marker="D", ms=7,
                        elinewidth=1.8, capsize=5, lw=0, zorder=4,
                        label="mean ± 1 sd" if i == 0 else None)
            az.text(i + 0.12, t["mean"], f"t = {t['t']:+.2f}", fontsize=8.5, va="center",
                    ha="left", color=INK)
        az.set_xticks(range(len(steps)))
        az.set_xticklabels([s[0] for s in steps], fontsize=9)
        az.set_xlim(-0.5, len(steps) - 0.2)
        az.set_ylabel("ΔW")
        az.set_title("the three terms, own scale", loc="left")
        az.legend(loc="lower right", fontsize=8.5)
        az.grid(axis="x", visible=False)

        fig.suptitle("H5: is the algorithmic effect separable from the demographic drift?",
                     x=0.01, ha="left", fontsize=11.5)
        fig.tight_layout()

    if save:
        _save(fig, "decomposition", out_dir)
    return fig


def make_all(out_dir: pathlib.Path | str | None = None) -> dict[str, Figure]:
    return {
        "separating_curve": separating_curve(out_dir=out_dir),
        "thin_market_curve": thin_market_curve(out_dir=out_dir),
        "decomposition": decomposition(out_dir=out_dir),
    }
