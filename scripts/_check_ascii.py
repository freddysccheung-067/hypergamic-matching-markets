"""Report any non-ASCII character in the TeX sources, with file and line.

The T1/lmodern font has no glyph for Unicode punctuation, so a stray em dash or
minus prints as nothing and the warning is one of a hundred. paper/verify.py
checks for control characters and bare commands; this checks the other way text
tools damage a source. Run before every compile.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
files = [ROOT / "paper" / "main.tex", *(ROOT / "paper" / "sections").glob("*.tex")]
bad = 0
for path in files:
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        # A "%" outside a comment starts one; skip comment text, which never prints.
        code = line
        idx = 0
        while idx < len(code):
            if code[idx] == "%" and (idx == 0 or code[idx - 1] != "\\"):
                code = code[:idx]
                break
            idx += 1
        for char in code:
            if ord(char) > 127:
                bad += 1
                print(f"{path.name}:{lineno}: U+{ord(char):04X} {char!r}")
print(f"{bad} non-ASCII character(s) in printed text")
sys.exit(1 if bad else 0)
