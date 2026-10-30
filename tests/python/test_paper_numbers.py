"""The numbers the paper prints agree with the files they come from.

Two kinds of check. The first renders ``paper/tables/numbers.tex`` afresh and
compares it with the committed file, so no macro can go stale against its CSV.
The second recomputes the paper's headline numbers from the raw per-seed files
with code written here, independently of the analysis scripts, and compares them
with the printed macros. A ratio computed on the wrong quantity, or under the
wrong name, fails the second kind even when the first passes; that is how the
warm-start decay share once went wrong.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
H = ROOT / "data" / "processed" / "horizon"
NUMBERS = PAPER / "tables" / "numbers.tex"
CONF = range(100, 120)

pytestmark = pytest.mark.skipif(not (H / "e09_canonical").is_dir(),
                                reason="the horizon result files are not present")


def macros() -> dict[str, str]:
    text = NUMBERS.read_text(encoding="utf-8")
    return dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", text))


def value(name: str) -> float:
    return float(macros()[name].replace("{,}", ""))


def gap_wide(directory: pathlib.Path, seeds, ref: pathlib.Path | None = None) -> pd.DataFrame:
    """Per (round, seed): NCF Gini minus oracle Gini, from the seed CSVs only."""
    d = pd.concat([pd.read_csv(p) for p in sorted(directory.glob("seed*.csv"))])
    d = d[d.seed.isin(seeds)]
    o = d if ref is None else pd.concat([pd.read_csv(p) for p in sorted(ref.glob("seed*.csv"))])
    ncf = d[d.arm == "ncf"].pivot(index="round", columns="seed", values="gini")
    orc = o[o.arm == "oracle"].pivot(index="round", columns="seed", values="gini")
    return ncf - orc[ncf.columns]


def test_numbers_tex_is_current_with_its_csvs() -> None:
    spec = importlib.util.spec_from_file_location("gen_numbers", PAPER / "gen_numbers.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    committed = NUMBERS.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert committed == module.render(), "numbers.tex is stale: run python paper/gen_numbers.py"


def test_round_99_gap_and_peak() -> None:
    g = gap_wide(H / "e09_canonical", CONF)
    assert f"{g.loc[99].mean():+.4f}" == macros()["gapNinetyNine"]
    assert f"{g.mean(axis=1).max():+.3f}" == macros()["peakGap"]
    assert str(int(g.mean(axis=1).idxmax())) == macros()["peakRound"]


def entry(gap: np.ndarray) -> int | None:
    inside = np.abs(gap) < 0.002
    if not inside[-1]:
        return None
    out = np.flatnonzero(~inside)
    return int(out[-1] + 1) if out.size else 0


def test_entry_rounds_cover_all_forty_seeds() -> None:
    g = gap_wide(H / "e09_canonical", list(range(42, 62)) + list(CONF))
    rounds = [entry(g[s].to_numpy()) for s in g.columns]
    assert None not in rounds
    assert str(min(rounds)) == macros()["entryMinAll"]
    assert str(max(rounds)) == macros()["entryMaxAll"]


def test_warm_start_decay_share_is_the_gap_from_round_zero() -> None:
    """Phase 1's defect class: the share must be computed on gaps, from round 0."""
    warm = gap_wide(H / "e13d_warm", CONF, ref=H / "e09_canonical")
    ctrl = gap_wide(H / "e09_canonical", CONF)
    shares = [(warm[s].iloc[0] - warm[s].iloc[-1]) / (ctrl[s].iloc[0] - ctrl[s].iloc[-1])
              for s in warm.columns]
    assert f"{100 * np.mean(shares):.1f}" == macros()["warmShareConf"]


def test_open_market_gap_at_one_percent() -> None:
    g = gap_wide(H / "e14_arrivals" / "r0.01", range(100, 110))
    assert f"{g.loc[500:599].mean().mean():+.3f}" == macros()["openGapMid"]


def test_long_tail_residual() -> None:
    g = gap_wide(H / "e10_long", range(100, 110))
    assert f"{g.loc[950:999].mean().mean():+.5f}" == macros()["longTail"]


def test_every_macro_the_prose_uses_is_defined() -> None:
    defined = set(macros())
    used: set[str] = set()
    for path in [PAPER / "main.tex", *sorted((PAPER / "sections").glob("*.tex"))]:
        text = re.sub(r"(?<!\\)%.*", "", path.read_text(encoding="utf-8"))
        used |= set(re.findall(r"\\([A-Za-z]+)", text))
    # A name that numbers.tex could have defined but did not: anything not a
    # LaTeX or package command. The candidates are the macros the paper defines.
    known_latex = {"newcommand"}
    # Every macro this paper defines is camelCase, which no LaTeX command it uses
    # is (\delta, \item, \exp are all lowercase).
    missing = {u for u in used if u not in defined and u[0].islower()
               and any(ch.isupper() for ch in u) and any(
        u.startswith(p) for p in ("gap", "peak", "entry", "tail", "tost", "warm", "open",
                                  "half", "long", "fit", "rmse", "dose", "off", "exp",
                                  "check", "cohort", "trend", "stat", "step", "best",
                                  "beats", "inert", "hOne", "hFive", "reg", "sd", "rank",
                                  "mcr", "fp", "corr", "jaccard", "spearman", "item",
                                  "modal", "identical", "oracle", "reach", "censored",
                                  "rows", "shock", "delta", "alg", "gini", "late", "bca"))}
    assert not (missing - known_latex), sorted(missing - known_latex)
