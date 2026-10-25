"""Audit: list every hardcoded decimal in the paper's prose.

A number that is not a macro is a number that can drift from its source, which is
the defect paper/numbers.py exists to prevent. This lists what is still literal so
a reader can decide which of them are qualitative (counts, section numbers, years)
and which should be macros.

    python /tmp/audit.py     (or anywhere; it has no dependencies)
"""

import io
import pathlib
import re

PREFIXED = re.compile(r"\\(?:texttt|ref|label|input|includegraphics|S|emph|textbf|citep|citet)\{[^}]*\}")
NUMBER = re.compile(r"[+-]?(?:\d+\.\d+|\d{3,})")

ROOT = pathlib.Path(r"D:\hypergamic-matching-markets")
FILES = [ROOT / "paper" / "main.tex"] + sorted(
    p for p in (ROOT / "paper" / "sections").glob("*.tex") if p.name != "appendix.tex"
)

for path in FILES:
    text = io.open(path, encoding="utf-8").read()
    text = PREFIXED.sub("", text)
    text = "\n".join(line.split("%")[0] for line in text.splitlines())
    found = sorted(set(NUMBER.findall(text)))
    print(f"{path.name:14s} {len(found):>2}: {' '.join(found)}")
