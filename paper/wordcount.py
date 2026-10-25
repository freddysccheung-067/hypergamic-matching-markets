"""Word counts per section, for the drafting report.

Counts words in the rendered sense: LaTeX markup is stripped, float bodies and
captions are excluded, and only running prose plus inline math symbols counts.
Use it to check a section against its target before and after revision.

    python paper/wordcount.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
SECTIONS = ["intro", "related", "model", "design", "results", "limits", "conclusion"]

TARGETS = {
    "intro": 800,
    "related": 1000,
    "model": 1500,
    "design": 800,
    "results": 2000,
    "limits": 600,
    "conclusion": 300,
}


def strip_latex(text: str) -> str:
    text = "\n".join(
        line.split("%", 1)[0] if not line.lstrip().startswith("%") else ""
        for line in text.splitlines()
    )
    # drop float environments entirely (captions are not body prose)
    text = re.sub(r"\\begin\{table\}.*?\\end\{table\}", " ", text, flags=re.S)
    text = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", " ", text, flags=re.S)
    text = re.sub(r"\\begin\{center\}.*?\\end\{center\}", " ", text, flags=re.S)
    text = re.sub(r"\\input\{[^}]*\}", " ", text)
    text = re.sub(r"\\includegraphics(\[[^\]]*\])?\{[^}]*\}", " ", text)
    text = re.sub(r"\\(cite[a-zA-Z]*|ref|eqref|label)\{[^}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", text)
    return text.replace("{", " ").replace("}", " ")


def count_words(text: str) -> int:
    stripped = strip_latex(text)
    return len(re.findall(r"[A-Za-z][A-Za-z'\-]*", stripped))


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--abstract":
        abstract = re.search(
            r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
            (PAPER / "main.tex").read_text(encoding="utf-8"),
            flags=re.S,
        )
        if not abstract:
            print("no abstract found")
            return 1
        print(f"abstract  {count_words(abstract.group(1))} words")
        return 0

    total = 0
    print(f"{'section':12s} {'words':>6s} {'target':>7s} {'delta':>7s}")
    for name in SECTIONS:
        path = PAPER / "sections" / f"{name}.tex"
        words = count_words(path.read_text(encoding="utf-8"))
        total += words
        target = TARGETS[name]
        print(f"{name:12s} {words:6d} {target:7d} {words - target:+7d}")
    print(f"{'TOTAL':12s} {total:6d} {sum(TARGETS.values()):7d} "
          f"{total - sum(TARGETS.values()):+7d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
