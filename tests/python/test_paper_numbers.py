"""The numbers the paper prints agree with the CSVs they claim to come from.

Two kinds of check. The first regenerates ``paper/tables/numbers.tex`` in memory
and compares it with the committed file, so a macro cannot go stale against its
data. The second recomputes a derived number from the raw per-seed CSVs with code
written here, independently of the scripts that produced it, and compares it with
the value the paper prints. A ratio computed on the wrong quantity, or under the
wrong name, fails the second kind even when the first passes, which is how the
warm-start decay share went wrong once.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import pathlib
import re
import statistics

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
RIGOR = ROOT / "data" / "processed" / "rigor"
NUMBERS = PAPER / "tables" / "numbers.tex"
SEEDS = (42, 43, 44, 45, 46)


def _macros() -> dict[str, str]:
    text = NUMBERS.read_text(encoding="utf-8")
    return dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", text))


def _gini(path: pathlib.Path, column: str, value: str) -> dict[int, float]:
    with path.open(newline="", encoding="utf-8") as fh:
        return {int(r["epoch"]): float(r["gini"]) for r in csv.DictReader(fh)
                if r[column] == value}


def _load_gen_numbers():
    spec = importlib.util.spec_from_file_location("gen_numbers", PAPER / "gen_numbers.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_numbers_tex_is_current_with_its_csvs() -> None:
    expected = _load_gen_numbers().render()
    committed = NUMBERS.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert committed == expected, (
        "paper/tables/numbers.tex differs from a fresh render; "
        "run python paper/gen_numbers.py"
    )


def _warm_ratios() -> dict[str, list[float]]:
    """The four decay fractions per seed, recomputed from the raw CSVs."""
    out: dict[str, list[float]] = {"level0": [], "gap0": [], "levelpeak": [], "gappeak": []}
    for seed in SEEDS:
        path = RIGOR / "interventions" / f"interventions_seed{seed}.csv"
        control = _gini(path, "condition", "control")
        warm = _gini(path, "condition", "warm")
        oracle = _gini(RIGOR / "exploratory_1t" / f"metrics_seed{seed}.csv", "arm", "oracle")
        last = max(control)
        cg = {e: control[e] - oracle[e] for e in control}
        wg = {e: warm[e] - oracle[e] for e in warm}
        out["level0"].append((warm[0] - warm[last]) / (control[0] - control[last]))
        out["gap0"].append((wg[0] - wg[last]) / (cg[0] - cg[last]))
        out["levelpeak"].append((max(warm.values()) - warm[last])
                                / (max(control.values()) - control[last]))
        out["gappeak"].append((max(wg.values()) - wg[last]) / (max(cg.values()) - cg[last]))
    return out


@pytest.mark.parametrize("key, macro", [
    ("gap0", "warmDecayGapZero"),
    ("level0", "warmDecayLevelZero"),
    ("gappeak", "warmDecayGapPeak"),
    ("levelpeak", "warmDecayLevelPeak"),
])
def test_warm_decay_fraction_matches_the_printed_macro(key: str, macro: str) -> None:
    recomputed = 100 * statistics.fmean(_warm_ratios()[key])
    assert f"{recomputed:.1f}" == _macros()[macro]


def test_warm_decay_interval_matches_the_printed_bounds() -> None:
    values = [100 * v for v in _warm_ratios()["gap0"]]
    n = len(values)
    mean = statistics.fmean(values)
    half = 2.7764451051977987 * statistics.stdev(values) / math.sqrt(n)  # t(0.975, 4)
    macros = _macros()
    assert f"{mean - half:.1f}" == macros["warmDecayGapZeroLo"]
    assert f"{mean + half:.1f}" == macros["warmDecayGapZeroHi"]


def test_warm_levels_and_gaps_are_not_the_same_quantity() -> None:
    """The control's round-0 level is the popularity closed form; its gap is not.

    At round 0 every agent on a side sees nearly the same twenty candidates, so
    the NCF level is 1 - 2k/N = 0.96 at k = 20, N = 1000. A macro that holds that
    number must be named a level.
    """
    macros = _macros()
    assert macros["warmLevelZeroControl"] == "0.9600"
    assert float(macros["warmGapZeroControl"]) < 0.5
    assert "warmEpochZeroControl" not in macros
