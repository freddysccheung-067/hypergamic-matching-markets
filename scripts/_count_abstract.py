"""Count the abstract, ignoring LaTeX markup the way paper/wordcount.py does."""

import io
import pathlib
import re

ROOT = pathlib.Path(r"D:\hypergamic-matching-markets")
text = io.open(ROOT / "paper" / "main.tex", encoding="utf-8").read()
body = text.split(r"\begin{abstract}", 1)[1].split(r"\end{abstract}", 1)[0]

body = re.sub(r"\\[A-Za-z]+\{[^{}]*\}", " ", body)   # \macro{arg}
body = re.sub(r"\\[A-Za-z]+", " ", body)             # \macro
body = body.replace("$", " ").replace("\\", " ")
words = [w for w in body.split() if re.search(r"[A-Za-z0-9]", w)]
print(f"abstract: {len(words)} words")
