"""One-off: rename digit-bearing epoch macros to letter-only TeX control words."""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
NAMES = {"6": "Six", "19": "Nineteen", "49": "FortyNine", "99": "NinetyNine",
         "124": "OneTwentyFour", "199": "OneNinetyNine"}
BS = chr(92)
pattern = re.compile(re.escape(BS) + r"gapEpoch(\d+)")

for path in [ROOT / "paper" / "main.tex", *(ROOT / "paper" / "sections").glob("*.tex")]:
    text = path.read_text(encoding="utf-8")
    new = pattern.sub(lambda m: BS + "gapEpoch" + NAMES[m.group(1)], text)
    if new != text:
        path.write_bytes(new.encode("utf-8"))
        print(f"{path.name}: {len(pattern.findall(text))} renamed")
