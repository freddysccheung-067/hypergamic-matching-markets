"""Audit: every decimal still typed as a literal in the paper's prose, with its line.

A number that is not a macro can drift from its source, which is the defect
paper/gen_numbers.py exists to prevent. This prints each remaining literal with file
and line so it can be traced to a CSV or classified as a configuration value,
a count, or a threshold (which stay literal: they are settings, not results).

    python scripts/_audit_numbers.py [section-name ...]
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PREFIXED = re.compile(r"\\(?:texttt|ref|label|input|includegraphics|citep|citet)\{[^}]*\}")
COMMENT = re.compile(r"(?<!\\)%.*")
NUMBER = re.compile(r"(?<![\w.])[+-]?\d+\.\d+(?:\\times10\^\{-?\d+\})?")

wanted = set(sys.argv[1:])
files = [ROOT / "paper" / "main.tex", *sorted((ROOT / "paper" / "sections").glob("*.tex"))]
for path in files:
    if wanted and path.stem not in wanted:
        continue
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = PREFIXED.sub("", COMMENT.sub("", line))
        found = NUMBER.findall(line)
        if found:
            print(f"{path.name}:{lineno}: {'  '.join(found)}")
