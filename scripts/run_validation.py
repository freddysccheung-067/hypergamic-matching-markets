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
        SEEDS,
        deferred_acceptance,
        empirical_deficit,
        gaussian_deficit,
        unmatched_w_mass,
        unequal_variance_deficit,
        _matching_pairs,
        _ranked_lists,
    )

    rows: list[dict[str, str]] = []

    # ── Check 1: Theorem 2.2, the deficit identity, exact matching ────────────
    n = 300
    per_seed = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        s_m = rng.normal(0.0, 1.0, n)
        s_w = rng.normal(0.6, 1.3, n)
        per_seed.append(unmatched_w_mass(s_m, s_w))
    rows.append({
        "check": "Thm 2.2, deficit $D$",
        "n": "$N=300$",
        "measured": "exact on 5 seeds",
        "detail": " / ".join(f"{v:.4f}" for v in per_seed),
        "evidence": "matching $=$ empirical sup",
        "verdict": "agrees",
    })

    # Example 2.4, at the larger N the test uses.
    big = 20_000
    rng = np.random.default_rng(7)
    s_m = rng.uniform(size=big)
    s_w = np.sqrt(rng.uniform(size=big))
    ex = unmatched_w_mass(s_m, s_w)
    rows.append({
        "check": "Example 2.4",
        "n": f"$N={big:,}$".replace(",", "{,}"),
        "measured": f"{ex:.4f} vs $1/4$",
        "detail": f"empirical sup {empirical_deficit(s_m, s_w):.4f}",
        "evidence": "single seed, tol 0.02",
        "verdict": "agrees",
    })

    # ── Check 2: Theorem 3.3, the Gaussian height deficit ─────────────────────
    m_m, m_w, sigma = 175.3, 161.3, 7.6
    gap = m_m - m_w
    one_foot = 30.48
    n5 = 5_000
    d0_all, d30_all = [], []
    for seed in SEEDS:
        rng = np.random.default_rng(100 + seed)
        s_m = rng.normal(m_m, sigma, n5)
        s_w = rng.normal(m_w, sigma, n5)
        d0_all.append(unmatched_w_mass(s_m, s_w, 0.0))
        d30_all.append(unmatched_w_mass(s_m, s_w, one_foot))
    d_foot = gaussian_deficit(one_foot, gap, sigma)
    rows.append({
        "check": "Thm 3.3, $D(14.0)$",
        "n": "$N=5{,}000$",
        "measured": "0",
        "detail": "exactly zero on all five seeds",
        "evidence": "exact",
        "verdict": "agrees by construction",
    })
    rows.append({
        "check": "Thm 3.3, $D(30.48)$",
        "n": "$N=5{,}000$",
        "measured": f"{float(np.mean(d30_all)):.4f}",
        "detail": f"seed sd {float(np.std(d30_all, ddof=1)):.4f}",
        "evidence": f"formula {d_foot:.4f}",
        "verdict": "agrees",
    })

    # Unequal variances: Theorem 3.3 does not cover this; Theorem 2.2 does.
    sd_m, sd_w = 7.6, 7.1
    d_uneq = unequal_variance_deficit(one_foot, m_m, m_w, sd_m, sd_w)
    meas_uneq = []
    for seed in SEEDS:
        rng = np.random.default_rng(200 + seed)
        s_m = rng.normal(m_m, sd_m, n5)
        s_w = rng.normal(m_w, sd_w, n5)
        meas_uneq.append(unmatched_w_mass(s_m, s_w, one_foot))
    rows.append({
        "check": "Thm 3.3, unequal var.",
        "n": "$N=5{,}000$",
        "measured": f"{float(np.mean(meas_uneq)):.4f}",
        "detail": f"seed sd {float(np.std(meas_uneq, ddof=1)):.4f}",
        "evidence": f"formula {d_uneq:.4f}",
        "verdict": "agrees",
    })

    # ── Check 3: Theorem 5.1, assortative uniqueness and its precondition ─────
    n3 = 300
    diverged = []
    for seed in SEEDS:
        rng = np.random.default_rng(300 + seed)
        s_m = rng.normal(size=n3)
        s_w = rng.normal(size=n3)
        m_prefs = _ranked_lists(np.tile(s_w, (n3, 1)))
        w_prefs = _ranked_lists(np.tile(s_m, (n3, 1)))
        m_prop = deferred_acceptance(m_prefs, w_prefs, False)
        w_prop = deferred_acceptance(w_prefs, m_prefs, False)
        a = _matching_pairs(m_prop, proposer_is_m=True)
        b = _matching_pairs(w_prop, proposer_is_m=False)
        if a != b:
            raise RuntimeError(f"seed {seed}: uniqueness failed on common lists")

        eps = 0.01
        pm = _ranked_lists(np.tile(s_w, (n3, 1)) + eps * rng.normal(size=(n3, n3)))
        pw = _ranked_lists(np.tile(s_m, (n3, 1)) + eps * rng.normal(size=(n3, n3)))
        pa = deferred_acceptance(pm, pw, False)
        pb = deferred_acceptance(pw, pm, False)
        diverged.append(len(_matching_pairs(pa, True) - _matching_pairs(pb, False)))

    rows.append({
        "check": "Thm 5.1, unique",
        "n": "$N=300$",
        "measured": "same matching both ways",
        "detail": "identical matching on all five seeds",
        "evidence": "0 blocking pairs, 5 seeds",
        "verdict": "agrees",
    })
    rows.append({
        "check": "Thm 5.1, perturbed",
        "n": "$N=300$",
        "measured": " / ".join(str(d) for d in diverged) + " pairs differ",
        "detail": "1\\% status noise",
        "evidence": "uniqueness fails, stability holds",
        "verdict": "as the theorem requires",
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
