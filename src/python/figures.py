"""The headline figures (THESIS.md, "Structure of the work": Presentation).

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

import copy
import pathlib
import warnings

import matplotlib as mpl
import numpy as np
import pandas as pd
import yaml
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.python.metrics import gini as _gini

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
FIGURES = ROOT / "docs" / "figures"
CONFIG_PATH = ROOT / "experiments" / "base.yaml"

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
    "arm_comparison": (
        "**Figure 4. The ablation (A-side headline).** Final-epoch Gini of exposure for "
        "the four ranking rules on one population, seed and utility set, mean ± 1 sd "
        "over seeds 42-46, N = 1000, k = 20, 20 epochs: ncf 0.939 ± 0.006, oracle "
        "0.654 ± 0.011, random 0.378 ± 0.006, popularity 0.960 ± 0.000. Each bar is "
        "annotated with its reach, the share of agents who appear on at least one list. "
        "The dashed line is the popularity arm's closed form, Gini = 1 − 2k/N = 0.96 at "
        "these settings, and that bar sits on it to the third decimal with zero seed "
        "variance: the popularity arm is arithmetic, not a finding, and its hatch marks "
        "it as a derived quantity everywhere it appears. **Gini alone is not a measure "
        "of harm.** It measures how unequally exposure is distributed, and unequal "
        "exposure is not by itself a welfare loss — the oracle, which is the best "
        "available ranker on its own task, also concentrates. The random arm is the "
        "lowest of the four *by construction*: drawing k candidates uniformly for every "
        "agent gives a near-equal exposure profile, so its 0.378 is a property of "
        "uniform sampling rather than evidence that randomness serves anyone well, and "
        "its 99.4% reach is the other half of the same fact. **The comparison that "
        "carries the argument is ncf-minus-oracle, +0.285 ± 0.007, not ncf-minus-random. "
        "** Both arms are trying to solve the same ranking task, so their difference is "
        "what estimation error costs relative to a perfect ranker; ncf-minus-random "
        "(+0.561) mostly measures distance from uniform exposure and would be quoted by "
        "any arm that happened to concentrate more than chance. **What it does not "
        "show:** that any of these rules would produce these numbers on a real "
        "population. The bars are a ranking-rule comparison inside a specified model "
        "with no empirical calibration, at one admissible fraction (0.2, at which the θ "
        "gate never binds) and one horizon (20 epochs, at which the ncf arm is still "
        "moving — over 100 epochs it falls to 0.673 and the ncf-minus-oracle gap shrinks "
        "to +0.019 ± 0.006). The Gini values are about exposure, not about welfare or "
        "about anyone's experience of the market."
    ),
    "gini_evolution": (
        "**Figure 5. Concentration over epochs.** Gini of exposure against epoch, one "
        "line per arm, mean ± 1 sd over seeds 42-46, with dashed verticals at the "
        "generation boundaries read from the run's own `generation` column. The "
        "popularity arm jumps from the oracle's 0.654 at epoch 0 to its closed form 0.960 "
        "at epoch 1 — the first epoch has no in-degree history, so it falls back to the "
        "true-utility ranking — and then sits flat, because a shared list produces the "
        "same concentration every epoch. The oracle is flat throughout at 0.654, which is "
        "the control: with no learning and no feedback there is nothing to drift. The "
        "random arm *falls* from 0.501 to 0.378 as its admissible sets settle. The ncf "
        "arm falls slowly from 0.960 to 0.939 while its reach climbs from 4% to 18%: one "
        "epoch of implicit feedback is not enough for the user embeddings to separate, so "
        "the score is still dominated by the item side and every user receives nearly the "
        "same global ranking, and the arm spends twenty epochs leaving that state part of "
        "the way. **The ablation regime is `admissible_fraction` 0.2, at which the θ gate "
        "never binds** — the gate truncates only below 2k/N = 0.04 (THEORY.md §3), so no "
        "agent's admissible set is ever cut and the loop that would feed matching "
        "outcomes back into preferences is inert. **This is a static market, so an "
        "upward slope would be preference concentration, not feedback.** No arm slopes "
        "up here; the downward slopes are the arms relaxing *out* of their epoch-0 "
        "state, and the flat oracle is what a static market looks like when nothing is "
        "learned. **What it does not show:** that a live θ gate would leave these lines "
        "unchanged — it is inert at this setting and this figure cannot speak to what "
        "happens below it — or that the 20-epoch horizon is the market's steady state; "
        "the ncf arm is still moving when the run ends."
    ),
    "matthew_effect": (
        "**Figure 6. Lorenz curves of exposure.** Cumulative share of exposure against "
        "cumulative share of agents, least-exposed first, with the 45-degree equality "
        "diagonal, for the oracle rule (dashed) and the ncf rule (solid) on one "
        "population at the final epoch; the dotted curve is the ncf arm at epoch 0, so "
        "the figure shows where it started as well as where it ended. Annotated Gini "
        "values are computed from the same exposure vectors the curves are drawn from. "
        "The gap between the curves is the same ncf-minus-oracle difference as Figure 4, "
        "read as a shape rather than as a number: under the ncf rule a larger share of "
        "agents accounts for a smaller share of exposure. **What it does not show:** that "
        "users would behave differently if they were shown a different list. This is a "
        "statement about exposure concentration under two ranking rules on one "
        "population, not about behaviour: `generate_swipes` takes no shown-set argument, "
        "so what the recommender surfaced never shapes what anyone wants. The preferences "
        "are identical across the two runs by construction — only the lists differ, and "
        "only exposure responds. The horizon is the committed ablation's 20 epochs rather "
        "than the config's 100: at 100 epochs the ncf arm converges onto the oracle and "
        "the two curves nearly coincide, so these curves are a statement about a "
        "particular horizon, and a longer run would show a much smaller gap."
    ),
    "attribution": (
        "**Figure 7. Attribution.** Left: for each trait pair, the correlation across the "
        "population (grey; the generative null) against the correlation across "
        "selection-induced pooled shortlisted pairs (blue), mean ± 1 sd over seeds 42-48 "
        "at k = 20. The two bars are drawn together because the blue one is unreadable "
        "alone: the artifact can only be read against its null. In the population every "
        "pair sits within sampling error of zero, because the traits are drawn "
        "independently; among shortlisted pairs, status and attractiveness correlate "
        "−0.203 ± 0.107 against +0.010 ± 0.011 in the population, with the sign that "
        "selecting on two traits jointly produces — among candidates clearing a high bar, "
        "higher status comes with lower attractiveness on average. Right: leave-one-term-"
        "out overlap, the share of the top-k shortlist that survives zeroing one weight, "
        "sorted by effect with the baseline (term inert) marked at 1.0. Measured: "
        "attractiveness 0.607 and status 0.748 at k = 20, so **attractiveness drives "
        "selection more than status at beta = 2·alpha** — the attractiveness weight is "
        "twice the status weight and both traits are standardized, so the attractiveness "
        "term carries twice the linear influence by construction. **That ordering is true "
        "of this weight set, not of traits in general:** it is one weight set to zero at a "
        "time on one pair of trait distributions, and a different alpha/beta ratio or a "
        "Pareto rather than lognormal status tail moves it. A nonzero selected-set "
        "correlation is a property of the search, not of people. **What it does not "
        "show:** that any of these traits cause anything. In this model no causal path "
        "between them exists to estimate, which is exactly why the selected-set value is "
        "an artifact."
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


# ── 4. the four ablation arms ────────────────────────────────────────────────

# Colour and label per ablation arm, in one place so the three A-side figures cannot
# disagree about which arm is which line. The popularity arm is hatched wherever it
# is drawn: its numbers are a closed form, not a measurement, and the mark says so
# before the caption does.
ARM_STYLE = {
    "random": (PURPLE, "random", ""),
    "oracle": (GREEN, "oracle", ""),
    "ncf": (BLUE, "ncf", ""),
    "popularity": (ORANGE, "popularity", "//"),
}


def _config_dict(config: dict | pathlib.Path | str | None) -> dict:
    """``experiments/base.yaml`` as a dict, with an explicit override honoured."""
    if isinstance(config, dict):
        return copy.deepcopy(config)
    path = pathlib.Path(config) if config is not None else CONFIG_PATH
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _arm_summary(data_dir: pathlib.Path | str | None) -> pd.DataFrame:
    """summary.csv as one row per arm: ``gini``, ``gini_sd``, ``reach``, ``reach_sd``."""
    s = _read(data_dir, "ablation", "summary.csv")
    wide = s.pivot(index="arm", columns="metric", values=["mean", "sd"])
    out = pd.DataFrame({
        "gini": wide[("mean", "gini")],
        "gini_sd": wide[("sd", "gini")],
        "reach": wide[("mean", "reach")],
        "reach_sd": wide[("sd", "reach")],
    })
    # Ascending gini, so the closed-form reference line lands on the bar it describes
    # rather than crossing the panel to reach it.
    return out.sort_values("gini")


def _ablation_horizon(data_dir: pathlib.Path | str | None) -> int:
    """The number of epochs the committed ablation tables ran for.

    Read from the files rather than from the config: ``simulation.n_epochs`` says
    100, and these tables are a 20-epoch run. A figure that silently drew a
    different horizon from the tables beside it would be the same class of error
    as plotting the wrong column.
    """
    d = _read(data_dir, "ablation", "ncf_seed42.csv")
    return int(d["epoch"].max()) + 1


def plot_arm_comparison(
    data_dir: pathlib.Path | str | None = None,
    out_dir: pathlib.Path | str | None = None,
    save: bool = True,
    config: dict | pathlib.Path | str | None = None,
) -> Figure:
    """Final-epoch Gini per ablation arm, with the popularity closed form as a line."""
    summary = _arm_summary(data_dir)
    gaps = _read(data_dir, "ablation", "gaps.csv").set_index("gap")
    cfg = _config_dict(config)
    top_k = int(cfg["ncf"]["top_k"])
    n = int(cfg["population"]["n"])
    closed_form = 1.0 - 2.0 * top_k / n

    with mpl.rc_context(_STYLE):
        fig = _new_figure(7.4, 4.9)
        ax = fig.add_subplot(111)

        x = np.arange(len(summary))
        for xi, (arm, row) in zip(x, summary.iterrows()):
            colour, label, hatch = ARM_STYLE[arm]
            ax.bar(xi, row.gini, width=0.62, color=colour, hatch=hatch, zorder=3,
                   edgecolor="white", linewidth=0.8)
            ax.errorbar(xi, row.gini, yerr=row.gini_sd, color=INK, capsize=4,
                        elinewidth=1.3, lw=0, zorder=4)
            ax.text(xi, row.gini + row.gini_sd + 0.02,
                    f"{row.gini:.3f}\n± {row.gini_sd:.3f}",
                    ha="center", va="bottom", fontsize=8.5)

        ax.axhline(closed_form, color=GREY, lw=1.4, ls="--", zorder=2)
        # Two lines, in the band above the oracle bar and below the line: that region
        # is empty (the tallest bar there reaches 0.665) whereas a single full-width
        # line ran under the ncf bar and went unreadable against it.
        ax.text(
            -0.55, closed_form - 0.035,
            f"popularity closed form\n1 − 2k/N = {closed_form:.2f}  (k={top_k}, N={n})",
            ha="left", va="top", fontsize=8, color=GREY, zorder=6, linespacing=1.35,
        )

        # The comparison that carries the argument, drawn between the two bars it
        # relates: estimation error against a ranker that solves the same task.
        lo, hi = summary.index.get_loc("oracle"), summary.index.get_loc("ncf")
        gap = gaps.loc["gini_ncf_minus_oracle"]
        ax.annotate(
            "", xy=(hi, 1.10), xytext=(lo, 1.10),
            arrowprops={"arrowstyle": "<->", "color": VERMILLION, "lw": 1.4},
        )
        ax.text(
            (lo + hi) / 2, 1.125,
            f"ncf − oracle  {gap['mean']:+.3f} ± {gap['sd']:.3f}",
            ha="center", va="bottom", fontsize=9, color=VERMILLION,
        )

        # Reach goes under the arm name rather than in a legend: a legend here would
        # sit on the ncf bar's error bar, and an x tick is where a per-bar number
        # belongs anyway.
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{ARM_STYLE[a][1]}\nreach {summary.loc[a, 'reach']:.1%}"
             for a in summary.index], fontsize=9,
        )
        ax.set_ylabel("final-epoch Gini of exposure")
        ax.set_ylim(0.0, 1.24)
        ax.set_xlim(-0.62, len(summary) - 0.38)
        ax.set_title(
            f"A-side ablation: {_ablation_horizon(data_dir)} epochs, five seeds, N = {n}",
            loc="left",
        )
        fig.tight_layout()

    if save:
        _save(fig, "arm_comparison", out_dir)
    return fig


# ── 5. Gini over epochs ──────────────────────────────────────────────────────

def _arm_trajectories(data_dir: pathlib.Path | str | None) -> dict[str, pd.DataFrame]:
    """Per arm, the mean and sd of every metric at each epoch, over the seeds present."""
    out: dict[str, pd.DataFrame] = {}
    for arm in ARM_STYLE:
        frames = [
            pd.read_csv(p) for p in
            sorted(pathlib.Path(data_dir or DATA, "ablation").glob(f"{arm}_seed*.csv"))
        ]
        if not frames:
            raise FileNotFoundError(
                f"no {arm}_seed*.csv under data/processed/ablation; run "
                "scripts/run_ablation.py first"
            )
        d = pd.concat(frames, ignore_index=True)
        g = d.groupby("epoch")
        out[arm] = pd.DataFrame({
            "mean": g["gini"].mean(),
            "sd": g["gini"].std(ddof=1),
            "generation": g["generation"].first() if "generation" in d else np.nan,
        }).reset_index()
    return out


def plot_gini_evolution(
    data_dir: pathlib.Path | str | None = None,
    out_dir: pathlib.Path | str | None = None,
    save: bool = True,
) -> Figure:
    """Gini against epoch, one line per arm, mean ± sd over the seeds."""
    traj = _arm_trajectories(data_dir)

    # Generation boundaries, taken from the column rather than recomputed: the column
    # is what the run actually did, and n_epochs // n_generations is only what it
    # would have done under the config as it stands today.
    breaks: np.ndarray | None = None
    if not traj["ncf"]["generation"].isna().all():
        per_arm = [
            set(np.asarray(t.epoch)[np.flatnonzero(np.diff(np.asarray(t.generation)) != 0) + 1])
            for t in traj.values()
        ]
        shared = set.intersection(*per_arm) if per_arm else set()
        breaks = np.array(sorted(float(e) for e in shared))
    else:
        warnings.warn(
            "the ablation CSVs carry no `generation` column (they predate it); "
            "generation boundaries are not drawn. Re-run scripts/run_ablation.py "
            "to record it.",
            RuntimeWarning,
            stacklevel=2,
        )

    with mpl.rc_context(_STYLE):
        fig = _new_figure(7.6, 4.5)
        ax = fig.add_subplot(111)

        for arm in ("random", "oracle", "ncf", "popularity"):
            t = traj[arm]
            colour, label, hatch = ARM_STYLE[arm]
            ax.fill_between(t.epoch, t["mean"] - t["sd"], t["mean"] + t["sd"],
                            color=colour, alpha=0.16, lw=0)
            ax.plot(t.epoch, t["mean"], color=colour, lw=2.1, ls="-",
                    marker="s" if arm == "popularity" else "o", ms=3.6, zorder=3,
                    label=label)
            ax.annotate(
                f"{t['mean'].iloc[-1]:.3f}", xy=(t.epoch.iloc[-1], t["mean"].iloc[-1]),
                xytext=(4, 0), textcoords="offset points", va="center",
                fontsize=8.5, color=colour,
            )

        # Drawn once, not once per arm: the breaks are identical across arms (they are
        # a function of the epoch index alone), so drawing them four times over would
        # only darken one line.
        for e in (breaks if breaks is not None else []):
            ax.axvline(float(e) - 0.5, color=GREY, lw=1.0, ls="--", zorder=1)
        if breaks is not None and breaks.size:
            ax.text(float(breaks[0]) - 0.35, 0.02, "generation break",
                    rotation=90, transform=ax.get_xaxis_transform(),
                    fontsize=8, color=GREY, va="bottom", ha="right")

        ax.set_xlabel("epoch")
        ax.set_ylabel("Gini of exposure")
        ax.set_ylim(0.0, 1.05)
        ax.set_xlim(-0.6, traj["ncf"].epoch.max() + 1.6)
        ax.set_title(
            "A-side ablation: Gini over epochs, mean ± 1 sd over five seeds", loc="left"
        )
        ax.legend(loc="center right", fontsize=9, ncol=1)
        fig.tight_layout()

    if save:
        _save(fig, "gini_evolution", out_dir)
    return fig


# ── 6. Lorenz curves ─────────────────────────────────────────────────────────

def _lorenz(exposure: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative share of exposure against cumulative share of agents, both in [0, 1].

    Agents are sorted by exposure ascending, which is the order the curve is defined
    in: the worst-served agent comes first. The leading (0, 0) is prepended so the
    curve starts at the origin rather than at the first agent's share.
    """
    x = np.sort(np.asarray(exposure, dtype=np.float64).ravel())
    total = x.sum()
    if total <= 0.0:
        share = np.zeros(x.size + 1)
    else:
        share = np.r_[0.0, np.cumsum(x) / total]
    return np.linspace(0.0, 1.0, x.size + 1), share


def plot_lorenz(before: np.ndarray, after: np.ndarray, ax: Axes | None = None) -> Axes:
    """Two exposure distributions on one axes, against the equality diagonal.

    A primitive: it takes the two exposure arrays and nothing else. Two side-by-side
    axes would put each curve on its own scale and make the comparison the whole
    point of the figure, so both go on one.

    ``before`` is drawn as the reference and ``after`` as the treatment; the names
    are positional because the figure is used both for a ranking-rule contrast and
    for an epoch contrast.
    """
    if ax is None:
        ax = Figure(figsize=(5.4, 5.0), facecolor="white").add_subplot(111)

    ax.plot([0, 1], [0, 1], color=GREY, lw=1.2, ls="--", zorder=2)
    ax.text(0.62, 0.68, "equality", rotation=45, fontsize=8.5, color=GREY,
            ha="center", va="bottom", transform=ax.transAxes, rotation_mode="anchor")

    for values, colour, label, ls in (
        (before, GREEN, "reference", "--"),
        (after, BLUE, "treatment", "-"),
    ):
        fx, fy = _lorenz(values)
        g = _gini(values)
        ax.fill_between(fx, fy, fx, color=colour, alpha=0.12, lw=0)
        ax.plot(fx, fy, color=colour, lw=2.1, ls=ls, zorder=3,
                label=f"{label}: Gini {g:.3f}")

    ax.set_xlabel("cumulative share of agents, least exposed first")
    ax.set_ylabel("cumulative share of exposure")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper left", fontsize=9)
    return ax


def plot_matthew_effect(
    config: dict | pathlib.Path | str | None = None,
    epoch_a: int = 0,
    epoch_b: int | None = None,
    data_dir: pathlib.Path | str | None = None,
    out_dir: pathlib.Path | str | None = None,
    save: bool = True,
    n_epochs: int | None = None,
) -> Figure:
    """Lorenz curves of exposure under the oracle and the NCF ranking rules.

    Runs two ``Simulation``s on one population and reads ``exposure_series``, which is
    an instance attribute and is not persisted anywhere else.

    The run length defaults to the committed ablation's horizon rather than to
    ``simulation.n_epochs``, so this figure and the two drawn from those tables are
    about the same run. That matters here more than anywhere: at 100 epochs the NCF
    converges onto the oracle and the two curves nearly coincide, so the horizon is
    not a cosmetic choice.
    """
    cfg = _config_dict(config)
    horizon = int(n_epochs if n_epochs is not None else _ablation_horizon(data_dir))
    b = horizon - 1 if epoch_b is None else int(epoch_b)
    a = int(epoch_a)

    from src.python.orchestrator import Simulation  # noqa: PLC0415  (builds torch, slow)

    series: dict[str, np.ndarray] = {}
    for arm in ("oracle", "ncf"):
        sim = Simulation(config=cfg, recommender=arm)
        sim.run(n_epochs=horizon)
        series[arm] = sim.exposure_series

    with mpl.rc_context(_STYLE):
        fig = _new_figure(6.2, 5.2)
        ax = fig.add_subplot(111)
        plot_lorenz(series["oracle"][b], series["ncf"][b], ax)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, [f"oracle, epoch {b}" if "reference" in t else
                            f"ncf, epoch {b}" for t in labels],
                  loc="upper left", fontsize=9)

        # Where the treatment started, so the figure shows movement as well as level.
        if a != b:
            fx, fy = _lorenz(series["ncf"][a])
            ax.plot(fx, fy, color=BLUE, lw=1.4, ls=":", zorder=3)
            ax.text(0.03, 0.86,
                    f"dotted: ncf at epoch {a}, Gini {_gini(series['ncf'][a]):.3f}\n"
                    f"solid: ncf at epoch {b}, Gini {_gini(series['ncf'][b]):.3f}",
                    transform=ax.transAxes, fontsize=8.5, color=BLUE, ha="left",
                    va="top")

        ax.set_title(f"Exposure concentration, {horizon} epochs, one population",
                     loc="left")
        fig.tight_layout()

    if save:
        _save(fig, "matthew_effect", out_dir)
    return fig


# ── 7. attribution ───────────────────────────────────────────────────────────

def attribution_measurements(
    config: dict | pathlib.Path | str | None = None,
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46, 47, 48),
    k: int | None = None,
) -> dict:
    """Selection correlation and leave-one-term-out overlap, mean ± sd over seeds.

    Called once per figure and once per test, so the figure and the assertion are
    reading the same numbers rather than two separate invocations that could drift.
    The seed set matches ``scripts/run_attribution.py``, which is what makes these
    values comparable with ``docs/attribution.md``.
    """
    from src.python.attribution import (  # noqa: PLC0415
        leave_one_term_out_effect,
        selection_correlation,
    )
    from _hypergamic import sample_population  # noqa: PLC0415

    cfg = _config_dict(config)
    n = int(cfg["population"]["n"])
    k = int(k if k is not None else cfg["ncf"]["top_k"])

    pairs: dict[str, list[tuple[float, float]]] = {}
    terms: dict[str, list[float]] = {}
    for seed in seeds:
        states = sample_population(n, cfg, int(seed))
        for key, value in selection_correlation(states, cfg, k).items():
            pairs.setdefault(key, []).append((value["population"], value["selected"]))
        for term, value in leave_one_term_out_effect(states, cfg, k=k).items():
            terms.setdefault(term, []).append(value)

    def _ms(values) -> tuple[float, float]:
        v = np.asarray(values, dtype=np.float64)
        return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else float("nan")

    return {
        "k": k,
        "n_seeds": len(seeds),
        "pairs": {
            key: {"population": _ms([p for p, _ in v])[0],
                  "population_sd": _ms([p for p, _ in v])[1],
                  "selected": _ms([s for _, s in v])[0],
                  "selected_sd": _ms([s for _, s in v])[1]}
            for key, v in pairs.items()
        },
        "terms": {t: {"mean": _ms(v)[0], "sd": _ms(v)[1]} for t, v in terms.items()},
    }


def plot_attribution(
    config: dict | pathlib.Path | str | None = None,
    out_dir: pathlib.Path | str | None = None,
    save: bool = True,
    measurements: dict | None = None,
) -> Figure:
    """Population against selection-induced correlation, and leave-one-term-out."""
    d = measurements if measurements is not None else attribution_measurements(config)
    pairs = d["pairs"]
    terms = d["terms"]

    with mpl.rc_context(_STYLE):
        # Narrower and with no text below 8.5 pt: GitHub renders these at 880 px, and a
        # wider canvas scales every label down further. At 9.8 in the worst label lands
        # near 5 px tall after that scaling, the same as the other figures in this file.
        fig = _new_figure(10.2, 4.7)
        ax, az = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.2, 1]})

        # ── left: the artifact against its null ──
        keys = list(pairs)
        x = np.arange(len(keys))
        pop = np.array([pairs[ky]["population"] for ky in keys])
        pop_sd = np.array([pairs[ky]["population_sd"] for ky in keys])
        sel = np.array([pairs[ky]["selected"] for ky in keys])
        sel_sd = np.array([pairs[ky]["selected_sd"] for ky in keys])

        ax.axhline(0.0, color=INK, lw=1.1, zorder=2)
        ax.bar(x - 0.19, pop, width=0.36, color=GREY, zorder=3,
               label="population (the generative null)")
        ax.bar(x + 0.19, sel, width=0.36, color=BLUE, zorder=3,
               label="selection-induced")
        ax.errorbar(x - 0.19, pop, yerr=pop_sd, color=INK, lw=0, elinewidth=1.2,
                    capsize=3, zorder=4)
        ax.errorbar(x + 0.19, sel, yerr=sel_sd, color=INK, lw=0, elinewidth=1.2,
                    capsize=3, zorder=4)

        # The pair the section is named after: the one place the gap is large. Placed to
        # the right of its bar and inside the ylim, so tight-bbox cannot clip it.
        worst = max(keys, key=lambda ky: abs(pairs[ky]["selected"] - pairs[ky]["population"]))
        wi = keys.index(worst)
        ax.annotate(
            f"{sel[wi]:+.3f} selected\nvs {pop[wi]:+.3f} in the population",
            xy=(wi + 0.19, sel[wi]), xytext=(wi + 0.30, sel[wi] - 0.10),
            ha="left", va="top", fontsize=8.5, color=BLUE,
            arrowprops={"arrowstyle": "-", "color": BLUE, "lw": 0.9},
        )

        ax.set_xticks(x)
        ax.set_xticklabels([ky.replace("~", "\n") for ky in keys], fontsize=8.5,
                           rotation=30, ha="right", rotation_mode="anchor")
        ax.set_ylabel("Pearson correlation")
        ax.set_ylim(-0.44, 0.30)
        ax.set_title(f"(a) the artifact against its null, k = {d['k']}", loc="left",
                     pad=26)
        # The top band is the one region with no bars in it: every population bar is
        # within ±0.05 of zero and the largest selection bar is at −0.23. Below the
        # axes the legend collided with the title; at the lower left it landed on the
        # negative bar the panel is about.
        ax.legend(loc="upper right", fontsize=8.5, borderaxespad=0.4)
        ax.grid(axis="x", visible=False)

        # ── right: leave-one-term-out, most influential first ──
        order = sorted(terms, key=lambda t: terms[t]["mean"])
        y = np.arange(len(order))
        means = np.array([terms[t]["mean"] for t in order])
        sds = np.array([terms[t]["sd"] for t in order])

        az.axvline(1.0, color=INK, lw=1.4, ls="--", zorder=2)
        # Inside the axes, rotated along the line and placed left of it: the bars stop
        # at 0.93, and anything above the top bar or right of the line is clipped or
        # collides with the title at this width.
        az.text(0.988, 0.06, "baseline: term inert", fontsize=8, color=INK,
                va="bottom", ha="right", rotation=90,
                transform=az.get_yaxis_transform())
        az.barh(y, means, height=0.58, color=VERMILLION, zorder=3)
        az.errorbar(means, y, xerr=sds, color=INK, lw=0, elinewidth=1.2, capsize=3,
                    zorder=4)
        # Right-aligned inside the bar and clear of the error bar: outside the bar end
        # the label and the cap shared a few points of width.
        for yi, mean, sd in zip(y, means, sds):
            az.text(mean - sd - 0.015, yi, f"{mean:.3f}", fontsize=8.5, va="center",
                    ha="right", color="white", zorder=5)

        az.set_yticks(y)
        az.set_yticklabels(order, fontsize=9)
        az.set_xlim(0.0, 1.16)
        az.set_ylim(-0.6, len(order) - 0.25)
        az.set_xlabel("share of top-k shortlist surviving the zeroed term")
        az.set_title("(b) leave-one-term-out overlap", loc="left")
        az.grid(axis="y", visible=False)

        fig.suptitle("Attribution: what the search does, not what the traits are",
                     x=0.01, ha="left", fontsize=11.5)
        fig.tight_layout()

    if save:
        _save(fig, "attribution", out_dir)
    return fig


def make_all(out_dir: pathlib.Path | str | None = None) -> dict[str, Figure]:
    return {
        "separating_curve": separating_curve(out_dir=out_dir),
        "thin_market_curve": thin_market_curve(out_dir=out_dir),
        "decomposition": decomposition(out_dir=out_dir),
        "arm_comparison": plot_arm_comparison(out_dir=out_dir),
        "gini_evolution": plot_gini_evolution(out_dir=out_dir),
        "matthew_effect": plot_matthew_effect(out_dir=out_dir),
        "attribution": plot_attribution(out_dir=out_dir),
    }
