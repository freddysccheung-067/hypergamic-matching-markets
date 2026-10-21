"""The three headline figures: each returns a Figure, writes a PNG, and is not blank.

A plotting test passes vacuously when the axes are empty, so "the file exists" is not
the assertion. Three layers, in increasing strength:

  * the PNG has more than one distinct colour, and a count well above the two a blank
    white canvas with a frame would give;
  * the axes hold the data the figure claims to draw, read back from the artists and
    compared with the CSV rather than with the plotting code's own variables;
  * the waterfall works at the three interaction regimes the design must not assume
    away: positive, negative and exactly zero.
"""

import pathlib
import sys
import warnings
from contextlib import nullcontext as _nullcontext

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from PIL import Image

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.python import figures  # noqa: E402

DATA = PROJECT_ROOT / "data" / "processed"

# An empty axis is all greys: frame, ticks and anti-aliased text have no hue, but they do
# produce a couple of hundred distinct grey values, so a raw colour count cannot tell a
# blank axis from a drawn one. A drawn series is coloured. Chromatic = channel spread
# above CHROMA; a figure must have both more than one distinct chromatic colour and a
# real number of chromatic pixels.
CHROMA = 40
MIN_CHROMATIC_COLOURS = 2
MIN_CHROMATIC_PIXELS = 500

pytestmark = pytest.mark.skipif(
    not (DATA / "separating" / "h1.csv").is_file(),
    reason="data/processed/separating/h1.csv missing — run scripts/run_separating.py",
)


def _chromatic(path: pathlib.Path) -> tuple[int, int]:
    """(distinct chromatic colours, chromatic pixel count). Greys do not count."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB")).reshape(-1, 3).astype(int)
    chromatic = arr[arr.max(axis=1) - arr.min(axis=1) > CHROMA]
    return len(np.unique(chromatic, axis=0)), len(chromatic)


def _looks_drawn(path: pathlib.Path) -> bool:
    colours, pixels = _chromatic(path)
    return colours > MIN_CHROMATIC_COLOURS and pixels > MIN_CHROMATIC_PIXELS


def _is_light(path: pathlib.Path) -> bool:
    """Light-mode substrate: the four corners are white, so a README on white embeds."""
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        corners = [rgb.getpixel(p) for p in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1))]
    return all(c == (255, 255, 255) for c in corners)


@pytest.fixture(scope="module")
def drawn(tmp_path_factory):
    out = tmp_path_factory.mktemp("figs")
    return out, {
        "separating_curve": figures.separating_curve(out_dir=out),
        "thin_market_curve": figures.thin_market_curve(out_dir=out),
        "decomposition": figures.decomposition(out_dir=out),
    }


@pytest.mark.parametrize("name", ["separating_curve", "thin_market_curve", "decomposition"])
def test_returns_a_figure_and_writes_a_png(drawn, name):
    out, figs = drawn
    assert isinstance(figs[name], Figure)
    path = out / f"{name}.png"
    assert path.is_file() and path.stat().st_size > 10_000


@pytest.mark.parametrize("name", ["separating_curve", "thin_market_curve", "decomposition"])
def test_png_is_not_blank(drawn, name):
    out, _ = drawn
    assert _looks_drawn(out / f"{name}.png")


@pytest.mark.parametrize("name", ["separating_curve", "thin_market_curve", "decomposition"])
def test_png_has_a_light_substrate(drawn, name):
    out, _ = drawn
    assert _is_light(out / f"{name}.png")


def test_a_blank_canvas_would_fail_the_colour_check(tmp_path):
    """The guard guards: an empty axis saved the same way is rejected, and so is one with
    only a grey line, because the most common vacuous pass is a frame with nothing in it."""
    fig = Figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    ax.set_xlabel("size fraction")
    path = tmp_path / "blank.png"
    fig.savefig(path, dpi=figures.DPI, bbox_inches="tight", facecolor="white")
    assert not _looks_drawn(path)
    ax.plot([0, 1], [0, 1], color="#0072B2", lw=3)
    ax.plot([0, 1], [1, 0], color="#D55E00", lw=3)
    drawn_path = tmp_path / "drawn.png"
    fig.savefig(drawn_path, dpi=figures.DPI, bbox_inches="tight", facecolor="white")
    assert _looks_drawn(drawn_path)


def test_saved_at_the_specified_dpi(drawn):
    out, _ = drawn
    with Image.open(out / "separating_curve.png") as im:
        dpi = im.info.get("dpi")
    assert dpi is not None and round(dpi[0]) == 150


def test_save_false_writes_nothing(tmp_path):
    figures.separating_curve(out_dir=tmp_path, save=False)
    assert list(tmp_path.iterdir()) == []


def test_missing_data_names_the_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="separating"):
        figures.separating_curve(data_dir=tmp_path, save=False)


# ── 1. the separating curve draws the measured points ────────────────────────

def test_separating_curve_points_are_the_seed_means(drawn):
    _, figs = drawn
    h1 = pd.read_csv(DATA / "separating" / "h1.csv")
    expected = h1.groupby("signal_cost").mean_tau.mean().to_numpy()

    ax = figs["separating_curve"].axes[0]
    # errorbar draws its markers as a Line2D with no line style; the analytic and
    # truncation curves are the other two lines with the same x grid.
    marker_lines = [ln for ln in ax.lines if ln.get_marker() == "o"]
    assert len(marker_lines) == 1
    np.testing.assert_allclose(np.asarray(marker_lines[0].get_ydata(), dtype=float),
                               expected, rtol=1e-12)


def test_separating_curve_overlays_the_analytic_cut(drawn):
    _, figs = drawn
    h1 = pd.read_csv(DATA / "separating" / "h1.csv")
    ax = figs["separating_curve"].axes[0]
    dashed = [ln for ln in ax.lines if ln.get_linestyle() == "--"]
    assert len(dashed) == 1
    # Drawn as the seed mean of the per-seed cut; each seed has its own V-bar, so this
    # is not c / (mean V-bar * p), and the test says which one the figure means.
    expected = h1.groupby("signal_cost").tau_star.mean().to_numpy()
    np.testing.assert_allclose(np.asarray(dashed[0].get_ydata(), dtype=float),
                               expected, rtol=1e-12)
    # And it is the formula: every seed's tau* is c / (V-bar * 0.5).
    np.testing.assert_allclose(h1.tau_star, h1.signal_cost / (h1.v_bar * h1.p_decision),
                               rtol=1e-12)


def test_h1_measured_mean_rises_over_the_grid():
    """H1's own kill condition, read off the committed data the figure draws."""
    h1 = pd.read_csv(DATA / "separating" / "h1.csv")
    m = h1.groupby("signal_cost").mean_tau.mean().to_numpy()
    assert m[-1] > m[0] + 0.1
    assert np.all(np.diff(m) >= -1e-9)


# ── 2. the thin-market curve draws both accountings and A ────────────────────

def test_thin_market_draws_both_accountings_and_environment_a(drawn):
    _, figs = drawn
    grid = pd.read_csv(DATA / "institutional" / "grid.csv")
    sub = grid[np.isclose(grid.signal_cost, 0.01)]
    ax = figs["thin_market_curve"].axes[0]

    common = sub.groupby("size_fraction").welfare_common_residual_b.mean().to_numpy()
    selected = sub.groupby("size_fraction").welfare_selected_b.mean().to_numpy()

    solid = [ln for ln in ax.lines if ln.get_linestyle() == "-" and ln.get_marker() == "o"]
    dashed = [ln for ln in ax.lines if ln.get_linestyle() == "--" and ln.get_marker() == "s"]
    assert len(solid) == 1 and len(dashed) == 1
    np.testing.assert_allclose(solid[0].get_ydata(), common, rtol=1e-12)
    np.testing.assert_allclose(dashed[0].get_ydata(), selected, rtol=1e-12)

    # Environment A is a horizontal line at the mean over seeds.
    a_mean = sub.groupby("seed").welfare_common_a.first().mean()
    horizontals = [ln for ln in ax.lines
                   if len(set(np.asarray(ln.get_ydata()).tolist())) == 1 and
                   np.isclose(ln.get_ydata()[0], a_mean)]
    assert horizontals


def test_thin_market_shows_the_selection_artifact_gap(drawn):
    """The two curves must differ visibly; identical curves would hide the artifact."""
    _, figs = drawn
    ax = figs["thin_market_curve"].axes[0]
    solid = next(ln for ln in ax.lines if ln.get_linestyle() == "-" and ln.get_marker() == "o")
    dashed = next(ln for ln in ax.lines if ln.get_linestyle() == "--" and ln.get_marker() == "s")
    assert np.max(np.abs(np.asarray(solid.get_ydata()) - np.asarray(dashed.get_ydata()))) > 100


def test_thin_market_annotates_the_critical_mass(drawn):
    _, figs = drawn
    texts = [t.get_text() for t in figs["thin_market_curve"].axes[0].texts]
    assert any(t.startswith("N* = 13.3") for t in texts)
    assert any(t.startswith("N* = 65.6") for t in texts)


def test_critical_mass_conversion_is_monotone():
    summary = pd.DataFrame({"size_fraction": [0.1, 0.5, 1.0], "n_eligible_mean": [50, 250, 500]})
    fr = [figures._critical_mass_as_fraction(summary, n) for n in (10, 100, 400)]
    assert fr == sorted(fr) and fr[1] == pytest.approx(0.2)


# ── 3. the decomposition works whatever the sign of the interaction ──────────

def _factorial_csv(tmp_path, interaction):
    """A synthetic factorial with a chosen interaction, W_both consistent with it."""
    inst = tmp_path / "institutional"
    inst.mkdir(parents=True)
    base, exo, alg = 1000.0, -20.0, -700.0
    rows = []
    for i, noise in enumerate((-4.0, -2.0, 0.0, 2.0, 4.0)):
        inter = interaction + noise * (0 if interaction == 0 else 1)
        rows.append({
            "seed": 42 + i, "gamma": 0.02, "W_baseline": base,
            "W_drift": base + exo, "W_ncf": base + alg, "W_both": base + exo + alg + inter,
            "exogenous": exo + noise, "algorithmic": alg - noise, "interaction": inter,
        })
    pd.DataFrame(rows).to_csv(inst / "factorial.csv", index=False)
    return tmp_path


@pytest.mark.parametrize("interaction", [60.0, -60.0, 0.0])
def test_waterfall_renders_at_any_interaction_sign(tmp_path, interaction):
    data = _factorial_csv(tmp_path / "data", interaction)
    out = tmp_path / "out"
    fig = figures.decomposition(data_dir=data, out_dir=out)
    path = out / "decomposition.png"
    assert isinstance(fig, Figure)
    assert _looks_drawn(path)

    # Four step labels plus the levels: the interaction bar is labelled even at zero.
    labels = [t.get_text() for t in fig.axes[0].texts]
    assert any(t == f"{interaction:+.1f}" for t in labels), labels


def test_zero_interaction_bar_is_still_marked(tmp_path):
    """The "missing middle bar" chart: a running-total tick must exist at the step."""
    data = _factorial_csv(tmp_path / "data", 0.0)
    fig = figures.decomposition(data_dir=data, out_dir=tmp_path / "out")
    ax = fig.axes[0]
    ticks = [ln for ln in ax.lines
             if len(ln.get_xdata()) == 2 and ln.get_ydata()[0] == ln.get_ydata()[1]
             and ln.get_linewidth() >= 1.4]
    assert len(ticks) == 3  # exogenous, algorithmic, interaction


def test_decomposition_terms_match_the_csv_and_add_up():
    d = figures.decomposition_terms()
    f = pd.read_csv(DATA / "institutional" / "factorial.csv")
    assert d["interaction"]["mean"] == pytest.approx(f.interaction.mean())
    assert d["interaction"]["t"] == pytest.approx(0.93, abs=0.01)  # THEORY §10.6
    # Per seed, W_both - W_baseline is the sum of the three terms: the waterfall closes.
    resid = (f.W_both - f.W_baseline) - (f.exogenous + f.algorithmic + f.interaction)
    assert np.abs(resid).max() < 1e-6
    assert d["baseline"] + d["exogenous"]["mean"] + d["algorithmic"]["mean"] \
        + d["interaction"]["mean"] == pytest.approx(d["both"])


# ── 4. the ablation arms ─────────────────────────────────────────────────────
#
# These four come from two different places and the tests say which. `arm_comparison`
# and `gini_evolution` read the committed CSVs, so their assertions compare drawn
# artists against the file. `matthew_effect` and `attribution` run the model, so their
# assertions compare the figure against a second invocation of the function it calls,
# or against the published number.

try:
    import _hypergamic  # noqa: F401
    _HAVE_EXTENSION = True
except ImportError:  # pragma: no cover
    # The extension lands in build/Release, which is not on sys.path by default. Try
    # that before concluding it is missing, or every test that needs it would skip on
    # a checkout where it is built.
    sys.path.insert(0, str(PROJECT_ROOT / "build" / "Release"))
    try:
        import _hypergamic  # noqa: F401
        _HAVE_EXTENSION = True
    except ImportError:
        _HAVE_EXTENSION = False

needs_extension = pytest.mark.skipif(
    not _HAVE_EXTENSION,
    reason="C++ extension not built — run: cmake --build build --config Release",
)


# ── 4. arm comparison ────────────────────────────────────────────────────────

def test_plot_arm_comparison_returns_a_figure_and_writes_a_png(tmp_path):
    fig = figures.plot_arm_comparison(out_dir=tmp_path)
    assert isinstance(fig, Figure)
    path = tmp_path / "arm_comparison.png"
    assert path.is_file() and path.stat().st_size > 10_000
    assert _looks_drawn(path)
    assert _is_light(path)


def test_arm_comparison_bars_are_the_summary_gini_values(tmp_path):
    """A wrong column must not plot silently: read the bar heights back."""
    fig = figures.plot_arm_comparison(out_dir=tmp_path)
    summary = pd.read_csv(DATA / "ablation" / "summary.csv")
    summary = summary[summary.metric == "gini"].set_index("arm")

    heights = [p.get_height() for p in fig.axes[0].patches if p.get_width() > 0.3]
    assert len(heights) == 4

    # Sorted ascending by gini, the order _arm_summary imposes.
    expected = summary["mean"].sort_values().to_numpy()
    np.testing.assert_allclose(heights, expected, atol=0.01)

    # And the four published values, so a column swap has to get past the caption too.
    for arm, published in (("ncf", 0.939), ("oracle", 0.664), ("random", 0.378),
                           ("popularity", 0.960)):
        assert summary.loc[arm, "mean"] == pytest.approx(published, abs=0.01)


def test_arm_comparison_draws_the_popularity_closed_form():
    """1 - 2k/N at k=20, N=1000 is 0.96. The line must sit there rather than at a data
    mean: the popularity arm's Gini is arithmetic, and the line is what says so."""
    fig = figures.plot_arm_comparison(save=False)
    horizontals = [ln for ln in fig.axes[0].lines
                   if len(set(np.asarray(ln.get_ydata(), dtype=float).tolist())) == 1]
    assert any(np.isclose(ln.get_ydata()[0], 0.96) for ln in horizontals), \
        [ln.get_ydata()[0] for ln in horizontals]


def test_arm_comparison_annotates_reach_on_every_arm():
    fig = figures.plot_arm_comparison(save=False)
    labels = " ".join(t.get_text() for t in fig.axes[0].get_xticklabels())
    for pct in ("99.4%", "95.8%", "18.1%", "4.0%"):
        assert pct in labels, labels


# ── 5. gini evolution ────────────────────────────────────────────────────────

def _evolution_figure(**kw):
    """plot_gini_evolution, tolerating the missing-generation warning when it applies.

    The committed tables predate the `generation` column, so the function warns and
    omits the boundaries. That is a missing annotation on otherwise correct output, so
    the tests around it assert the absence explicitly and do not let the warning abort
    the rest.
    """
    has_generation = "generation" in pd.read_csv(DATA / "ablation" / "ncf_seed42.csv")
    with _nullcontext() if has_generation else pytest.warns(RuntimeWarning, match="generation"):
        return figures.plot_gini_evolution(**kw)


def test_plot_gini_evolution_returns_a_figure_and_writes_a_png(tmp_path):
    fig = _evolution_figure(out_dir=tmp_path)
    assert isinstance(fig, Figure)
    path = tmp_path / "gini_evolution.png"
    assert path.is_file() and path.stat().st_size > 10_000
    assert _looks_drawn(path)
    assert _is_light(path)


def test_gini_evolution_lines_are_the_seed_means(tmp_path):
    fig = _evolution_figure(out_dir=tmp_path)
    ax = fig.axes[0]
    assert len(ax.lines) == 4

    for arm, line in zip(("random", "oracle", "ncf", "popularity"), ax.lines):
        frames = [pd.read_csv(p) for p in
                  sorted((DATA / "ablation").glob(f"{arm}_seed*.csv"))]
        expected = pd.concat(frames).groupby("epoch").gini.mean().to_numpy()
        np.testing.assert_allclose(np.asarray(line.get_ydata(), dtype=float), expected,
                                   rtol=1e-12)


def test_gini_evolution_invents_no_boundary_without_the_column():
    """The committed tables predate `generation`, so this figure draws no boundaries and
    warns. The failure being guarded against is silently recomputing them from
    n_epochs // n_generations, which would draw a line the run never had."""
    if "generation" in pd.read_csv(DATA / "ablation" / "ncf_seed42.csv"):
        pytest.skip("committed ablation records `generation`; nothing to guard")

    with pytest.warns(RuntimeWarning, match="generation"):
        fig = figures.plot_gini_evolution(save=False)
    verticals = [ln for ln in fig.axes[0].lines
                 if len(set(np.asarray(ln.get_xdata(), dtype=float).tolist())) == 1]
    assert verticals == []


def test_gini_evolution_draws_the_generation_breaks_from_the_column(tmp_path):
    """The boundary path, exercised on synthetic tables that carry `generation`.

    The breaks are placed off the natural 20-epoch grid on purpose: a figure that
    recomputed them from n_epochs // n_generations would draw at 20 and 40 and this
    test would fail. What is asserted is the column, not the arithmetic.
    """
    ablation = tmp_path / "ablation"
    ablation.mkdir()
    breaks = (7, 15)  # deliberate: not multiples of anything in the config
    for arm in ("ncf", "oracle", "random", "popularity"):
        for seed in (42, 43, 44, 45, 46):
            epochs = np.arange(20)
            pd.DataFrame({
                "arm": arm, "seed": seed, "epoch": epochs,
                # Generation 0 up to epoch 6, 1 from 7 to 14, 2 from 15.
                "generation": (epochs >= 7).astype(int) + (epochs >= 15).astype(int),
                "gini": 0.5 + 0.01 * epochs + 0.001 * seed,
                "reach": 0.9, "top1_share": 0.1, "mcr": 0.5, "rsi": 0.4,
                "welfare_alg": 0.0, "noise_ratio": 1.0,
            }).to_csv(ablation / f"{arm}_seed{seed}.csv", index=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the missing-column warning must not fire here
        fig = figures.plot_gini_evolution(data_dir=tmp_path, save=False)

    verticals = [float(np.asarray(ln.get_xdata())[0]) for ln in fig.axes[0].lines
                 if len(set(np.asarray(ln.get_xdata(), dtype=float).tolist())) == 1]
    # Drawn once for all four arms, at every generation change: 7 and 15.
    assert verticals == [pytest.approx(b - 0.5) for b in breaks]


def test_gini_evolution_still_renders_and_is_not_blank(tmp_path):
    with pytest.warns(RuntimeWarning):
        fig = figures.plot_gini_evolution(out_dir=tmp_path)
    assert isinstance(fig, Figure)
    assert _looks_drawn(tmp_path / "gini_evolution.png")


# ── 6. lorenz / matthew ──────────────────────────────────────────────────────

def test_lorenz_curve_reproduces_the_exposure_gini():
    """plot_lorenz annotates a Gini; it must be the metric's Gini over the same array,
    and the curve must run from (0, 0) to (1, 1)."""
    rng = np.random.default_rng(0)
    before = rng.integers(0, 3, size=200)
    after = rng.integers(0, 12, size=200)

    ax = figures.plot_lorenz(before, after)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert len(labels) == 2
    for label, values in zip(labels, (before, after)):
        assert float(label.split()[-1]) == pytest.approx(figures._gini(values), abs=5e-4)

    fx, fy = figures._lorenz(after)
    assert fx[0] == 0.0 and fy[0] == 0.0
    assert fx[-1] == pytest.approx(1.0) and fy[-1] == pytest.approx(1.0)
    # Two arrays of different concentration must not give the same curve.
    assert not np.allclose(figures._lorenz(before)[1], figures._lorenz(after)[1])


@needs_extension
def test_plot_matthew_effect_returns_a_figure_and_writes_a_png(tmp_path):
    fig = figures.plot_matthew_effect(n_epochs=4, out_dir=tmp_path)
    assert isinstance(fig, Figure)
    path = tmp_path / "matthew_effect.png"
    assert path.is_file() and path.stat().st_size > 10_000
    assert _looks_drawn(path)
    assert _is_light(path)


@needs_extension
def test_matthew_effect_puts_the_higher_concentration_on_the_ncf_arm():
    """ncf concentrates more than oracle at this horizon: the solid curve must sit
    below the dashed one over the body of the plot. A figure that plotted the arms the
    other way round would look equally plausible."""
    fig = figures.plot_matthew_effect(n_epochs=6, save=False)
    ax = fig.axes[0]
    reference, treatment = ax.lines[1], ax.lines[2]
    assert np.asarray(treatment.get_ydata()).mean() < np.asarray(reference.get_ydata()).mean()


# ── 7. attribution ───────────────────────────────────────────────────────────

@needs_extension
def test_plot_attribution_returns_a_figure_and_writes_a_png(tmp_path):
    fig = figures.plot_attribution(out_dir=tmp_path)
    assert isinstance(fig, Figure)
    path = tmp_path / "attribution.png"
    assert path.is_file() and path.stat().st_size > 10_000
    assert _looks_drawn(path)
    assert _is_light(path)


@needs_extension
def test_attribution_values_are_the_published_ones():
    """The two numbers the caption quotes, from the function the figure draws from."""
    d = figures.attribution_measurements()
    assert d["terms"]["attract"]["mean"] == pytest.approx(0.607, abs=0.02)
    assert d["terms"]["status"]["mean"] == pytest.approx(0.748, abs=0.02)
    # Attractiveness ahead of status is the claimed ordering at beta = 2*alpha.
    assert d["terms"]["attract"]["mean"] < d["terms"]["status"]["mean"]

    pair = d["pairs"]["attractiveness~status"]
    assert pair["population"] == pytest.approx(0.0, abs=0.05)
    assert pair["selected"] < -0.1


def test_attribution_figure_draws_every_pair_and_every_term():
    """Both panels must carry the whole table, not a subset that happens to plot."""
    fig = figures.plot_attribution(save=False)
    ax, az = fig.axes
    assert len(ax.patches) == 12  # six trait pairs, two bars each
    assert len(az.patches) == 4   # four terms
    assert [t.get_text() for t in az.get_yticklabels()] == [
        "attract", "status", "indicator", "distance"]


# ── captions ─────────────────────────────────────────────────────────────────

A_SIDE = ("arm_comparison", "gini_evolution", "matthew_effect", "attribution")


@pytest.mark.parametrize("name", A_SIDE)
def test_a_side_captions_state_what_they_do_not_show(name):
    cap = figures.CAPTIONS[name]
    assert "What it does not show" in cap
    assert len(cap.split("What it does not show")[1]) > 150


def test_arm_caption_carries_the_three_required_limits():
    cap = figures.CAPTIONS["arm_comparison"]
    assert "Gini alone is not a measure of harm" in cap
    assert "by construction" in cap
    assert "ncf-minus-oracle" in cap and "not ncf-minus-random" in cap


def test_attribution_caption_qualifies_the_ordering():
    cap = figures.CAPTIONS["attribution"]
    assert "not of traits in general" in cap
    assert "property of the search, not of people" in cap


def test_gini_evolution_caption_names_the_regime_and_the_static_market():
    cap = figures.CAPTIONS["gini_evolution"]
    assert "admissible_fraction" in cap and "never binds" in cap
    assert "static market" in cap and "not feedback" in cap


def test_matthew_caption_names_the_missing_shown_set():
    assert "`generate_swipes` takes no shown-set argument" \
        in figures.CAPTIONS["matthew_effect"]
