"""Build the paper from the committed result files, in one command.

    python paper/build.py              # numbers, tables, figures, then compile
    python paper/build.py --analysis   # first rerun every analysis from the seed files

The compile step uses Tectonic. It looks for ``tectonic`` on PATH, then at
``$TECTONIC``, then in two places Windows installs often leave it. The PDF is
written to ``paper/main.pdf``. The build fails if the log reports an undefined
reference or an overfull box.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

PAPER = pathlib.Path(__file__).resolve().parent
ROOT = PAPER.parent
CANDIDATES = [pathlib.Path.home() / ".tectonic_scratch" / "tectonic.exe",
              pathlib.Path(tempfile.gettempdir()) / "tect" / "tectonic.exe"]


def run(*args: str) -> None:
    print("$", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def tectonic() -> str:
    found = shutil.which("tectonic") or os.environ.get("TECTONIC")
    if found:
        return found
    for path in CANDIDATES:
        if path.exists():
            return str(path)
    raise SystemExit("tectonic not found: put it on PATH or set TECTONIC")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", action="store_true")
    args = parser.parse_args()
    if args.analysis:
        for exp in ("e09", "e10", "e11", "e13a", "e13b", "e13c", "e13d", "e14", "e15",
                    "e16", "e17", "appendix"):
            run("scripts/analyze_horizon.py", exp)
        run("scripts/analyze_extras.py")
        run("scripts/run_validation.py")
    run("paper/gen_numbers.py")
    run("paper/tables.py")
    run("paper/figures.py")
    out = pathlib.Path(tempfile.mkdtemp(prefix="paper_build_"))
    subprocess.run([tectonic(), "--keep-logs", "--outdir", str(out), "main.tex"],
                   cwd=PAPER, check=True)
    log = (out / "main.log").read_text(encoding="utf-8", errors="replace")
    problems = re.findall(r"(Overfull \\hbox.*|LaTeX Warning: (?:Reference|Citation).*undefined.*)",
                          log)
    shutil.copy(out / "main.pdf", PAPER / "main.pdf")
    if problems:
        print("\n".join(problems))
        raise SystemExit(f"{len(problems)} overfull box or undefined reference warnings")
    print("wrote paper/main.pdf; log clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
