"""One-off: make the TeX sources ASCII-safe for the T1/lmodern font.

Unicode punctuation typed into the sources (em dash, en dash, minus, smart quotes)
has no glyph in ec-lmr10 and prints as nothing. Each is replaced by the TeX
ligature or command that produces it. Reports anything left over.
"""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
MAP = {
    "—": "---",   # em dash
    "–": "--",    # en dash
    "−": "$-$",   # minus sign in prose
    "‘": "`",
    "’": "'",
    "“": "``",
    "”": "''",
    "×": "$\\times$",
}

for path in [ROOT / "paper" / "main.tex", *(ROOT / "paper" / "sections").glob("*.tex")]:
    text = path.read_text(encoding="utf-8")
    new = text
    for src, dst in MAP.items():
        new = new.replace(src, dst)
    leftover = sorted({c for c in new if ord(c) > 127})
    if new != text:
        path.write_bytes(new.encode("utf-8"))
        print(f"{path.name}: fixed {sum(text.count(s) for s in MAP)} characters")
    if leftover:
        print(f"  {path.name}: still non-ASCII: {[hex(ord(c)) for c in leftover]}")
