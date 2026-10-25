"""B1 end-to-end check: a one-cell CSV change must reach the compiled PDF.

Perturbs the live100 gap in intervals.csv by +0.0001, regenerates numbers.tex,
compiles with Tectonic into a scratch directory, extracts the PDF text, and
reverts the CSV and numbers.tex byte-for-byte whatever happens.
"""

import pathlib
import shutil
import subprocess
import sys

from pypdf import PdfReader

ROOT = pathlib.Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "processed" / "rigor" / "intervals.csv"
NUMBERS = ROOT / "paper" / "tables" / "numbers.tex"
TECTONIC = pathlib.Path(r"C:\Users\Freddy\.tectonic_scratch\tectonic.exe")
OUT = pathlib.Path(r"C:\Users\Freddy\.tectonic_scratch\b1")


def build() -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "paper" / "numbers.py")], check=True,
                   capture_output=True)
    subprocess.run([str(TECTONIC), "--outdir", str(OUT), "main.tex"],
                   cwd=ROOT / "paper", check=True, capture_output=True)
    return "\n".join(p.extract_text() or "" for p in PdfReader(OUT / "main.pdf").pages)


csv_backup = CSV.read_bytes()
numbers_backup = NUMBERS.read_bytes()
try:
    before = build()
    text = csv_backup.decode("utf-8")
    old = "0.025102107160399935"
    assert text.count(old) == 1, "expected the live100 gap point exactly once"
    CSV.write_bytes(text.replace(old, "0.025202107160399935").encode("utf-8"))
    after = build()
    print(f"'+0.0251' in PDF: before {before.count('+0.0251')}, after {after.count('+0.0251')}")
    print(f"'+0.0252' in PDF: before {before.count('+0.0252')}, after {after.count('+0.0252')}")
    moved = before.count("+0.0251") > after.count("+0.0251") and after.count("+0.0252") > before.count("+0.0252")
    print("B1 WIRING:", "OK - the PDF picked up the CSV change" if moved else "BROKEN")
finally:
    CSV.write_bytes(csv_backup)
    NUMBERS.write_bytes(numbers_backup)
    shutil.rmtree(OUT, ignore_errors=True)
    print("reverted CSV and numbers.tex")
