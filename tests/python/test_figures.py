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


# ── captions ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["separating_curve", "thin_market_curve", "decomposition"])
def test_every_caption_states_what_it_does_not_show(name):
    cap = figures.CAPTIONS[name]
    assert "What it does not show" in cap
    assert len(cap.split("What it does not show")[1]) > 150


def test_captions_carry_the_three_required_limits():
    assert "real proposers" in figures.CAPTIONS["separating_curve"]
    assert "real community" in figures.CAPTIONS["thin_market_curve"]
    assert "algorithms amplify" in figures.CAPTIONS["decomposition"]
