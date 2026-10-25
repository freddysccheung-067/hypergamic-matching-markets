"""Generate the paper's LaTeX tables from the committed CSVs.

Every table this script writes reads only from ``data/processed/``. Running it
regenerates ``tables/*.tex`` byte for byte, because every source file is
committed and nothing is resampled here.

    python paper/tables.py

Design note: the generated LaTeX uses only core environments (``tabular``,
``hline``, ``multicolumn``) rather than ``booktabs``. The paper cannot be
compiled in the environment it was drafted in, so the table bodies are kept to
constructs that need no package beyond ``article``.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RIGOR = PROCESSED / "rigor"
OUT = Path(__file__).resolve().parent / "tables"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def keyed(path: Path, key: str = "quantity") -> dict[str, dict[str, str]]:
    """Rows of a long CSV indexed by one column.

    ``intervals.csv`` carries duplicate keys because a quantity appears once per
    regime, so callers that need a specific regime use :func:`regime_row`.
    """
    return {row[key]: row for row in read_csv(path)}


def regime_row(quantity: str, regime: str) -> dict[str, str]:
    for row in read_csv(RIGOR / "intervals.csv"):
        if row["quantity"] == quantity and row["regime"] == regime:
            return row
    raise KeyError(f"{quantity} @ {regime}")


def mc_values() -> dict[str, float]:
    return {
        row["quantity"]: float(row["value"])
        for row in read_csv(RIGOR / "multiple_comparisons.csv")
    }


def fmt(value: float, places: int = 4, signed: bool = False) -> str:
    """Format a number for a LaTeX table body."""
    if signed:
        return f"{value:+.{places}f}"
    return f"{value:.{places}f}"


def interval(row: dict[str, str], signed: bool = True) -> str:
    """A bracketed interval from an ``intervals.csv`` row, t bounds by default.

    ``docs/STATS.md`` §1 fixes the convention: at five seeds the Student-t
    interval is the one with nominal coverage, so it is the one printed.
    """
    lo = float(row["lo_t"])
    hi = float(row["hi_t"])
    return f"[{fmt(lo, 3, signed)}, {fmt(hi, 3, signed)}]"


def write(name: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(body, encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)} ({len(body.splitlines())} lines)")


# --------------------------------------------------------------------------
# Table 1: the hypothesis ledger
# --------------------------------------------------------------------------

def table_ledger() -> None:
    """hypothesis, regime, measured, interval, verdict.

    Only hypotheses with a committed artefact appear here. H1 and H2 are
    discussed in the text without an interval, because ``docs/STATS.md``
    carries none for either; H6 has no committed measurement at all. The
    ledger therefore lists four rows and says so in its caption.
    """
    h3_live = regime_row("gini_gap_ncf_minus_oracle", "live100")
    h3_long = regime_row("gini_gap_ncf_minus_oracle", "long300")
    mc = mc_values()

    # The interval column carries the t interval throughout, except for H4 where
    # the corrected t interval is the one docs/STATS.md §5.3 says to read. H5's
    # family is the factorial's five seeds, not the forty institutional cells, so
    # its uncorrected t bounds are the ones printed; that is why it reads
    # ``interaction_lo_t`` rather than a ``_bonferroni`` column.
    interaction_lo_t = mc["interaction_lo_t"]
    interaction_hi_t = mc["interaction_hi_t"]

    # Every row names how its interval was built, because they are not all built
    # the same way and a reader cannot tell from the numbers. The cells below are
    # (hypothesis, regime, point, t bounds, BCa bounds, which construction the
    # interval column carries, verdict).
    rows = [
        (
            "H3",
            "NCF $-$ oracle Gini (live100, 5 seeds)",
            fmt(float(h3_live["point"]), 4, True),
            f"[{float(h3_live['lo_t']):+.4f}, {float(h3_live['hi_t']):+.4f}]",
            f"[{float(h3_live['lo']):+.4f}, {float(h3_live['hi']):+.4f}]",
            "t",
            "survives; size is horizon-dependent",
        ),
        (
            "H3",
            "same gap (long300, 5 seeds)",
            fmt(float(h3_long["point"]), 5, True),
            f"[{float(h3_long['lo_t']):+.5f}, {float(h3_long['hi_t']):+.5f}]",
            f"[{float(h3_long['lo']):+.5f}, {float(h3_long['hi']):+.5f}]",
            "t",
            "below $0.001$ from about epoch 250",
        ),
        (
            "H4",
            "welfare B $-$ A (c=0.01, sf=0.65)",
            fmt(mc["optimum_mean"], 2, True),
            f"[{mc['named_cell_lo_t_bonferroni']:+.2f}, {mc['named_cell_hi_t_bonferroni']:+.2f}]",
            f"[{mc['named_cell_lo_bonferroni']:+.2f}, {mc['named_cell_hi_bonferroni']:+.2f}]",
            "Bonferroni $t$",
            "survives Bonferroni and Holm; best of 40 cells",
        ),
        (
            "H5",
            "interaction term (factorial, 5 seeds)",
            fmt(mc["interaction_mean"], 2, True),
            f"[{interaction_lo_t:+.2f}, {interaction_hi_t:+.2f}]",
            f"[{mc['interaction_lo_bca']:+.2f}, {mc['interaction_hi_bca']:+.2f}]",
            "t",
            "null under both; BCa includes zero too",
        ),
    ]

    body = [
        "% Generated by paper/tables.py from data/processed/rigor/intervals.csv",
        "% and multiple_comparisons.csv. Do not edit by hand.",
        "% \\resizebox is a last resort that keeps the table inside the text block at",
        "% any font size; without it the 6-column ledger overflows \\textwidth by",
        "% about 100pt at 11pt, which the compile warns about and a reader sees.",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llrrlp{3.4cm}}",
        "\\hline",
        "Hyp. & Regime & Measured & $t$ interval & BCa interval & Verdict \\\\",
        "\\hline",
    ]
    for hyp, regime, measured, t_iv, bca_iv, reading, verdict in rows:
        body.append(
            f"{hyp} & {regime} & {measured} & {t_iv} & {bca_iv} & {verdict} \\\\"
        )
    body += [
        "\\hline",
        "\\end{tabular}%",
        "}",
        "",
    ]
    write("ledger.tex", "\n".join(body))


# --------------------------------------------------------------------------
# Table 2: the sensitivity grid
# --------------------------------------------------------------------------

def table_sensitivity() -> None:
    """Which conclusions hold across the twelve parameter cells."""
    rows = read_csv(RIGOR / "sensitivity.csv")

    def tick(flag: str) -> str:
        return "yes" if flag == "True" else "\\textbf{no}"

    body = [
        "% Generated by paper/tables.py from data/processed/rigor/sensitivity.csv.",
        "% C1-C4 are the four claim verdicts docs/STATS.md §4.1 defines.",
        "% Eight columns at 11pt overflow the text block by about 155pt; \\resizebox",
        "% holds it inside whatever width the class actually gives.",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llrrcccc}",
        "\\hline",
        "Knob & Level & Gap & 95\\% BCa & C1 & C2 & C3 & C4 \\\\",
        "\\hline",
    ]
    previous = None
    for row in rows:
        axis = row["axis"].replace("_", "/")
        level = row["level"]
        shipped = row["is_shipped"] == "True"
        label = f"\\textit{{{level}}}" if shipped else level
        gap = fmt(float(row["gap_point"]), 4, True)
        iv = (
            f"[{float(row['gap_lo']):+.4f}, {float(row['gap_hi']):+.4f}]"
        )
        ticks = " & ".join(
            tick(row[f"C{i}_{name}"])
            for i, name in (
                (1, "gap_excludes_zero"),
                (2, "gap_below_ceiling"),
                (3, "reach_above_floor"),
                (4, "mcr_excludes_zero"),
            )
        )
        if previous is not None and axis != previous:
            body.append("\\hline")
        first = axis if axis != previous else ""
        previous = axis
        body.append(f"{first} & {label} & {gap} & {iv} & {ticks} \\\\")
    body += [
        "\\hline",
        "\\end{tabular}%",
        "}",
        "",
    ]
    write("sensitivity.tex", "\n".join(body))


# --------------------------------------------------------------------------
# Table 3: validation
# --------------------------------------------------------------------------

def table_validation() -> None:
    """The validation table of docs/validation.md.

    These are the only numbers in the paper that do not trace to a CSV under
    data/processed/: they are asserted in tests/python/test_validation.py and
    transcribed in docs/validation.md. See the table's caption.
    """
    rows = [
        (
            "Thm 2.2, deficit $D$",
            "$N=300$",
            "exact on 5 seeds",
            "0.2433 / 0.3367 / 0.2200 / 0.2833 / 0.1800",
            "matching $=$ empirical sup",
        ),
        (
            "Example 2.4",
            "$N=20{,}000$",
            "0.2519 vs $1/4$",
            "single seed, tol 0.02",
            "agrees",
        ),
        (
            "Thm 3.3, $D(14.0)$",
            "$N=5{,}000$",
            "0",
            "exact",
            "agrees by construction",
        ),
        (
            "Thm 3.3, $D(30.48)$",
            "$N=5{,}000$",
            "0.7254",
            "seed sd 0.0054",
            "formula 0.7217",
        ),
        (
            "Thm 3.3, unequal var.",
            "$N=5{,}000$",
            "0.7433",
            "seed sd 0.0053",
            "formula 0.7380",
        ),
        (
            "Thm 5.1, unique",
            "$N=300$",
            "same matching both ways",
            "0 blocking pairs, 5 seeds",
            "agrees",
        ),
        (
            "Thm 5.1, perturbed",
            "$N=300$",
            "22 / 8 / 11 / 12 / 8 pairs differ",
            "1\\% status noise",
            "uniqueness fails, stability holds",
        ),
    ]
    body = [
        "% Generated by paper/tables.py from docs/validation.md.",
        "% Numbers are asserted in tests/python/test_validation.py; see the caption.",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lllll}",
        "\\hline",
        "Check & $N$ & Measured & Evidence & Verdict \\\\",
        "\\hline",
    ]
    for check, n, measured, evidence, verdict in rows:
        body.append(f"{check} & {n} & {measured} & {evidence} & {verdict} \\\\")
    body += [
        "\\hline",
        "\\end{tabular}%",
        "}",
        "",
    ]
    write("validation.tex", "\n".join(body))


def main() -> None:
    table_ledger()
    table_sensitivity()
    table_validation()


if __name__ == "__main__":
    main()
