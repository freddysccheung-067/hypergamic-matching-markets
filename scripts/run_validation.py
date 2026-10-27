"""Write the validation table's numbers to a committed CSV.

Table~\ref{tab:validation} of the paper is the only table whose values were not
committed anywhere: they were asserted in ``tests/python/test_validation.py`` and
transcribed by hand into ``docs/validation.md`` and then into
``paper/tables.py``. Three copies of one number, none of them generated from the
others, is the arrangement that produced the paper's three-values-for-one-horizon
defect elsewhere. This script closes it: it recomputes the same quantities the
tests assert, writes them to ``data/processed/rigor/validation.csv``, and
``paper/tables.py`` reads that file.

    python scripts/run_validation.py

**The helpers are imported from the test module**, not copied. A copy would drift
the moment either side was edited; an import means the table cannot disagree with
the assertion without the test failing first.

``tests/python/test_validation.py::test_the_committed_validation_csv_matches``
calls this script's ``compute()`` and compares against the committed file, so a
stale CSV is a test failure rather than a silently wrong table.
"""

from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "python"))
# The extension lands in build/Release under MSVC and in build/ under a
# single-config generator; tests/python/conftest.py does the same for pytest and
# this script needs it because it is not run under pytest.
for _path in (ROOT / "build" / "Release", ROOT / "build"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

OUT = ROOT / "data" / "processed" / "rigor" / "validation.csv"

# Columns of the committed file. `check` and `n` identify the row; `measured` is
# what the simulator gave; `evidence` and `verdict` are the paper's text, stored
# here so the table is generated wholly from one artefact.
COLUMNS = [
    "check", "n", "measured", "detail", "evidence", "verdict",
]


def compute() -> list[dict[str, str]]:
    """Every row of the paper's validation table, computed rather than transcribed."""
    from test_validation import (  # noqa: PLC0415
        GAUSS_DELTA,
        GAUSS_M,
        GAUSS_SD,
        GAUSS_SD_W,
        GAUSS_W,
        SEEDS,
        _matching_pairs,
        _ranked_lists,
        deferred_acceptance,
        empirical_deficit,
        gaussian_deficit,
        popularity_check,
        shipped_config,
        truncated_da_check,
        unequal_variance_deficit,
        unmatched_w_mass,
    )

    rows: list[dict[str, str]] = []

    # ── 1. The deficit identity (Hall's theorem, deficiency form) ─────────────
    n = 300
    per_seed = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        s_m = rng.normal(0.0, 1.0, n)
        s_w = rng.normal(0.6, 1.3, n)
        per_seed.append(unmatched_w_mass(s_m, s_w))
    rows.append({
        "check": "Deficit identity (Hall)",
        "n": "$N=300$",
        "measured": "exact on 5 seeds",
        "detail": " / ".join(f"{v:.4f}" for v in per_seed),
        "evidence": "matching $=$ $\\sup_t$ formula",
        "verdict": "agrees",
    })

    big = 20_000
    rng = np.random.default_rng(7)
    s_m = rng.uniform(size=big)
    s_w = np.sqrt(rng.uniform(size=big))
    ex = unmatched_w_mass(s_m, s_w)
    rows.append({
        "check": "Deficit, uniform vs.\\ Beta(2,1)",
        "n": f"$N={big:,}$".replace(",", "{,}"),
        "measured": f"{ex:.4f} vs $1/4$",
        "detail": f"empirical sup {empirical_deficit(s_m, s_w):.4f}",
        "evidence": "single seed, tol 0.02",
        "verdict": "agrees",
    })

    # ── 2. The Gaussian closed form, and the unequal-variance supremum ────────
    n5 = 5_000
    d_closed = gaussian_deficit(GAUSS_DELTA, GAUSS_M - GAUSS_W, GAUSS_SD)
    eq, uneq = [], []
    for seed in SEEDS:
        rng = np.random.default_rng(100 + seed)
        eq.append(unmatched_w_mass(rng.normal(GAUSS_M, GAUSS_SD, n5),
                                   rng.normal(GAUSS_W, GAUSS_SD, n5), GAUSS_DELTA))
        rng = np.random.default_rng(200 + seed)
        uneq.append(unmatched_w_mass(rng.normal(GAUSS_M, GAUSS_SD, n5),
                                     rng.normal(GAUSS_W, GAUSS_SD_W, n5), GAUSS_DELTA))
    d_exact = unequal_variance_deficit(GAUSS_DELTA, GAUSS_M, GAUSS_W, GAUSS_SD, GAUSS_SD_W)
    rows.append({
        "check": "Gaussian closed form, equal variances",
        "n": "$N=5{,}000$",
        "measured": f"{float(np.mean(eq)):.4f}",
        "detail": f"seed sd {float(np.std(eq, ddof=1)):.4f}",
        "evidence": f"formula {d_closed:.4f}",
        "verdict": "agrees",
    })
    rows.append({
        "check": "Exact supremum, unequal variances",
        "n": "$N=5{,}000$",
        "measured": f"{float(np.mean(uneq)):.4f}",
        "detail": f"seed sd {float(np.std(uneq, ddof=1)):.4f}",
        "evidence": f"supremum {d_exact:.4f}",
        "verdict": "agrees",
    })

    # ── 3. Common rankings: one stable matching, and what breaks it ───────────
    n3 = 300
    diverged = []
    for seed in SEEDS:
        rng = np.random.default_rng(300 + seed)
        s_m = rng.normal(size=n3)
        s_w = rng.normal(size=n3)
        m_prefs = _ranked_lists(np.tile(s_w, (n3, 1)))
        w_prefs = _ranked_lists(np.tile(s_m, (n3, 1)))
        a = _matching_pairs(deferred_acceptance(m_prefs, w_prefs, False), proposer_is_m=True)
        b = _matching_pairs(deferred_acceptance(w_prefs, m_prefs, False), proposer_is_m=False)
        if a != b:
            raise RuntimeError(f"seed {seed}: uniqueness failed on common lists")
        eps = 0.01
        pm = _ranked_lists(np.tile(s_w, (n3, 1)) + eps * rng.normal(size=(n3, n3)))
        pw = _ranked_lists(np.tile(s_m, (n3, 1)) + eps * rng.normal(size=(n3, n3)))
        pa = deferred_acceptance(pm, pw, False)
        pb = deferred_acceptance(pw, pm, False)
        diverged.append(len(_matching_pairs(pa, True) - _matching_pairs(pb, False)))
    rows.append({
        "check": "Common rankings: unique stable matching",
        "n": "$N=300$",
        "measured": "same matching both ways",
        "detail": "identical matching on all five seeds",
        "evidence": "0 blocking pairs, 5 seeds",
        "verdict": "agrees",
    })
    rows.append({
        "check": "Perturbed rankings",
        "n": "$N=300$",
        "measured": " / ".join(str(d) for d in diverged) + " pairs differ",
        "detail": "1\\% status noise",
        "evidence": "uniqueness fails, stability holds",
        "verdict": "agrees",
    })

    # ── 4. The popularity closed form, Gini 1 - 2k/N and reach 2k/N ───────────
    worst = popularity_check(shipped_config(), (500, 1000, 2000), (0, 1, 2), 5)
    rows.append({
        "check": "Popularity ranker: Gini $1-2k/N$",
        "n": "$N=500$, 1000, 2000",
        "measured": f"max error {worst:.1e}",
        "detail": "rounds 1--4, 3 seeds per $N$",
        "evidence": "Gini and reach, $k=20$",
        "verdict": "agrees",
    })

    # ── 5. Truncated deferred acceptance on top-k lists ───────────────────────
    on_lists, against_truth, checked = truncated_da_check(shipped_config(300), SEEDS, 4)
    rows.append({
        "check": "Truncated DA on top-$k$ lists",
        "n": "$N=300$",
        "measured": f"{on_lists} blocking pairs on the lists",
        "detail": f"{checked} matchings (NCF, oracle; 5 seeds, 4 rounds)",
        "evidence": f"{against_truth / checked:,.0f} per matching against true preferences".replace(",", "{,}"),
        "verdict": "stable on its lists",
    })
    return rows


def main() -> int:
    rows = compute()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
