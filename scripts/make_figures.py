"""Regenerate the three headline figures into docs/figures/ and print their captions.

    python scripts/make_figures.py

Reads the committed CSVs under data/processed/; run scripts/run_separating.py first if
data/processed/separating/h1.csv is absent.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


def main() -> None:
    from src.python import figures  # noqa: PLC0415

    figures.make_all()
    for name, caption in figures.CAPTIONS.items():
        print(f"docs/figures/{name}.png\n{caption}\n")


if __name__ == "__main__":
    main()
