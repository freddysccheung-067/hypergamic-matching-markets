"""Structural checks on the LaTeX sources, since no TeX toolchain is installed.

Checks, in the order the paper's build would fail on them:

1. every ``\\input{...}`` path exists (relative to ``paper/``);
2. every ``\\ref{...}`` has a matching ``\\label{...}``;
3. every ``\\cite{...}`` key has an entry in ``refs.bib``;
4. every ``\\label{...}`` is referenced (reports unused, does not fail);
5. every included figure file exists;
6. no duplicate labels;
7. brace balance per file (a cheap proxy for a parse failure).

    python paper/verify.py

Exit status is 0 when checks 1-3, 5 and 6 pass, 1 otherwise. Unused labels and
brace-balance notes are printed but do not fail the run.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
MAIN = PAPER / "main.tex"

INPUT_RE = re.compile(r"\\input\{([^}]+)\}")
REF_RE = re.compile(r"\\(?:ref|eqref|autoref)\{([^}]+)\}")
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
CITE_RE = re.compile(r"\\cite[a-zA-Z]*\{([^}]+)\}")
BIBKEY_RE = re.compile(r"@\w+\{([^,]+),")
GRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
# \newcommand{\gapShipped}{...} -> gapShipped
MACRO_DEF_RE = re.compile(r"\\newcommand\{\\([A-Za-z0-9]+)\}")
# A macro used as $...\name...$ in the prose. Restricted to the names numbers.py
# can emit, so ordinary formatting macros (\emph, \texttt, \S) are not flagged.
MACRO_USE_RE = re.compile(
    r"\\((?:gap|gini|mcr|rsi|reach|opt|hFive|tail|stat|nSweep|nInstitutional"
    r"|nReject|fwer|algLoss|finalEpoch)[A-Za-z0-9]*)"
)


def strip_comments(text: str) -> str:
    """Remove TeX comments so commented-out labels are not counted."""
    return "\n".join(
        line.split("%", 1)[0] if not line.lstrip().startswith("%") else ""
        for line in text.splitlines()
    )


def collect_files() -> list[Path]:
    """main.tex plus every file reachable through \\input, depth-first."""
    seen: list[Path] = []
    pending = [MAIN]
    while pending:
        path = pending.pop(0)
        if path in seen or not path.exists():
            continue
        seen.append(path)
        text = strip_comments(path.read_text(encoding="utf-8"))
        for match in INPUT_RE.finditer(text):
            target = (PAPER / match.group(1)).with_suffix(".tex")
            pending.append(target)
    return seen


def main() -> int:
    failures: list[str] = []
    notes: list[str] = []

    if not MAIN.exists():
        print(f"FAIL: {MAIN} does not exist")
        return 1

    files = collect_files()
    print(f"scanned {len(files)} .tex file(s):")
    for path in files:
        print(f"  {path.relative_to(PAPER)}")

    combined: list[tuple[Path, str]] = []
    for path in files:
        combined.append((path, strip_comments(path.read_text(encoding="utf-8"))))

    # 1. inputs resolve
    print("\n[1] input paths")
    for path in files:
        text = path.read_text(encoding="utf-8")
        for match in INPUT_RE.finditer(strip_comments(text)):
            target = (PAPER / match.group(1)).with_suffix(".tex")
            status = "ok" if target.exists() else "MISSING"
            if not target.exists():
                failures.append(f"{path.name}: \\input{{{match.group(1)}}} missing")
            print(f"  {status:8s} {match.group(1)}")

    # 2 and 4. refs and labels
    labels: dict[str, Path] = {}
    duplicates: list[str] = []
    for path, text in combined:
        for match in LABEL_RE.finditer(text):
            name = match.group(1)
            if name in labels:
                duplicates.append(f"{name} ({labels[name].name}, {path.name})")
            labels[name] = path

    refs: dict[str, list[Path]] = {}
    for path, text in combined:
        for match in REF_RE.finditer(text):
            for name in match.group(1).split(","):
                refs.setdefault(name.strip(), []).append(path)

    print("\n[2] ref -> label")
    for name in sorted(refs):
        ok = name in labels
        if not ok:
            failures.append(f"\\ref{{{name}}} has no \\label")
        print(f"  {'ok' if ok else 'MISSING':8s} {name}")

    print("\n[4] unused labels (informational)")
    unused = sorted(set(labels) - set(refs))
    for name in unused:
        print(f"  unused   {name}")
    if not unused:
        print("  none")

    print("\n[6] duplicate labels")
    if duplicates:
        for dup in duplicates:
            failures.append(f"duplicate label {dup}")
            print(f"  DUPLICATE {dup}")
    else:
        print("  none")

    # 3. citations against the bib
    bib_path = PAPER / "refs.bib"
    bib_keys = set(BIBKEY_RE.findall(bib_path.read_text(encoding="utf-8")))
    cited: dict[str, list[Path]] = {}
    for path, text in combined:
        for match in CITE_RE.finditer(text):
            for key in match.group(1).split(","):
                cited.setdefault(key.strip(), []).append(path)

    print(f"\n[3] cite -> bib ({len(cited)} cited of {len(bib_keys)} entries)")
    for key in sorted(cited):
        ok = key in bib_keys
        if not ok:
            failures.append(f"\\cite{{{key}}} has no bib entry")
        print(f"  {'ok' if ok else 'MISSING':8s} {key}")
    uncited = sorted(bib_keys - set(cited))
    if uncited:
        print(f"  ({len(uncited)} bib entries not cited: {', '.join(uncited)})")

    # 5. figures
    print("\n[5] figure files")
    for path, text in combined:
        for match in GRAPHICS_RE.finditer(text):
            target = PAPER / match.group(1)
            found = target.exists() or any(
                target.with_suffix(ext).exists() for ext in (".png", ".pdf", ".jpg")
            )
            if not found:
                failures.append(f"{path.name}: figure {match.group(1)} missing")
            print(f"  {'ok' if found else 'MISSING':8s} {match.group(1)}")

    # 7. brace balance (cheap parse-failure proxy)
    print("\n[7] brace balance")
    for path, text in combined:
        depth = 0
        low = 0
        for char in text:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                low = min(low, depth)
        status = "ok" if depth == 0 and low == 0 else "IMBALANCED"
        if status != "ok":
            notes.append(f"{path.name}: brace depth ends at {depth}, min {low}")
        print(f"  {status:10s} {path.name} (final depth {depth})")

    # 8. number macros: every \foo used in the prose must be defined in
    # tables/numbers.tex, and every macro defined there should be used. An
    # undefined one compiles to nothing or to an error depending on the engine;
    # either way the paper would silently lose a number.
    numbers_path = PAPER / "tables" / "numbers.tex"
    print("\n[8] number macros")
    if not numbers_path.exists():
        failures.append("tables/numbers.tex is missing; run python paper/numbers.py")
        print("  MISSING tables/numbers.tex")
    else:
        defined = set(MACRO_DEF_RE.findall(numbers_path.read_text(encoding="utf-8")))
        used: set[str] = set()
        for _path, text in combined:
            used |= set(MACRO_USE_RE.findall(text))
        # A TeX control word is letters only. \gapEpoch6 is \gapEpoch followed by
        # the character "6", so a digit in a defined name is a compile error that
        # every regex-level check above would wave through.
        for name in sorted(defined):
            if any(ch.isdigit() for ch in name):
                failures.append(
                    f"\\{name}: TeX macro names cannot contain digits; "
                    f"it parses as \\{name.rstrip('0123456789')} plus text"
                )
                print(f"  INVALID  \\{name}")
        undefined = sorted(used - defined)
        for name in undefined:
            failures.append(f"\\{name} is used but not defined in tables/numbers.tex")
            print(f"  MISSING  \\{name}")
        unused = sorted(defined - used)
        if unused:
            notes.append(f"{len(unused)} number macros defined but unused: "
                         + ", ".join(unused))
        print(f"  {len(defined)} defined, {len(used)} used, "
              f"{len(undefined)} undefined, {len(unused)} unused")

    print("\n[10] control characters")
    contro = 0
    for path, text in combined:
        for char in text:
            if ord(char) < 32 and char not in "\n\t":
                contro += 1
                failures.append(
                    f"{path.name}: control character U+{ord(char):04X} in source "
                    f"(a backslash was eaten by a text tool; the macro name that "
                    f"follows it is broken)"
                )
    print(f"  {contro} found")

    print("\n[11] backslash-stripped commands")
    # A text tool that eats "\" before a letter leaves the command name bare, or
    # leaves the letter as a control character. Check 10 catches the latter; this
    # catches "ef{...}" at the start of a line, where "\ref" lost its backslash
    # and the first letter with it.
    # Requires the opening brace: the point is a command whose name survived and
    # whose backslash did not. Without the brace, ordinary prose ("end of a
    # different one") would trip it, and a checker that cries wolf gets ignored.
    stripped = re.compile(
        r"(?m)^(?:ef|exttt|extbf|emph|site|ection|ubsection|label|eqref|itemize|"
        r"begin|end|input|includegraphics|frac|sqrt|times|quad|hline)\{"
    )
    bare = 0
    for path, text in combined:
        for match in stripped.finditer(text):
            bare += 1
            line = text.count("\n", 0, match.start()) + 1
            failures.append(
                f"{path.name}:{line}: '{match.group(0)}' at line start — a command "
                f"lost its backslash"
            )
    print(f"  {bare} found")

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED with {len(failures)} structural problem(s):")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("all structural checks passed")
    if notes:
        print("notes:")
        for note in notes:
            print(f"  - {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
