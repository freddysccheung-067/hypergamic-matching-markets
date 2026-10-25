"""Replace REVIEW_RESPONSE item 1.5 with the revised text and delete the temp file."""

import io
import pathlib

root = pathlib.Path(r"D:\hypergamic-matching-markets")
target = root / "docs" / "REVIEW_RESPONSE.md"
new_text = (root / "docs" / "_rr_15.md").read_text(encoding="utf-8")

text = io.open(target, encoding="utf-8").read()
start = text.index("## 1.5 The counterexample")
end = text.index("## 1.6 No stated rule")
io.open(target, "w", encoding="utf-8", newline="").write(text[:start] + new_text + text[end:])

(root / "docs" / "_rr_15.md").unlink()
print("replaced 1.5")
