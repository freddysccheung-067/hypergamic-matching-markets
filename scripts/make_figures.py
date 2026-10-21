"""Regenerate every figure into docs/figures/ and print their captions.

    python scripts/make_figures.py
    python scripts/make_figures.py --only arm_comparison,gini_evolution

Reads the committed CSVs under data/processed/; run scripts/run_separating.py first if
data/processed/separating/h1.csv is absent, and scripts/run_ablation.py if
data/processed/ablation/ is absent.

Two of the seven figures run rather than read: `matthew_effect` builds two
Simulations for exposure_series (about 25 s at the 20-epoch horizon), and
`attribution` samples seven populations for the correlation and leave-one-term-out
tables (a few seconds). Both need the C++ extension built and on sys.path.
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "build" / "Release"))

# The order the README presents them in. make_all draws this same set; naming it here
# as well lets --only select a subset without importing matplotlib first.
NAMES = (
    "separating_curve",
    "thin_market_curve",
    "decomposition",
    "arm_comparison",
    "gini_evolution",
    "matthew_effect",
    "attribution",
)


def main() -> None:
    from src.python import figures  # noqa: PLC0415

    # The captions carry typographic minus signs and em dashes; a Windows console
    # defaults to a codepage that cannot encode them and the print would abort after
    # the figures had already been written.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):  # pragma: no cover - non-reconfigurable stream
        pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None,
                        help=f"comma-separated subset of {', '.join(NAMES)}")
    args = parser.parse_args()

    wanted = NAMES if args.only is None else tuple(
        s.strip() for s in args.only.split(",") if s.strip()
    )
    unknown = set(wanted) - set(NAMES)
    if unknown:
        raise SystemExit(f"unknown figure(s) {sorted(unknown)}; expected {NAMES}")

    drawn = figures.make_all() if wanted == NAMES else {
        name: getattr(figures, {
            "separating_curve": "separating_curve",
            "thin_market_curve": "thin_market_curve",
            "decomposition": "decomposition",
            "arm_comparison": "plot_arm_comparison",
            "gini_evolution": "plot_gini_evolution",
            "matthew_effect": "plot_matthew_effect",
            "attribution": "plot_attribution",
        }[name])()
        for name in wanted
    }

    for name in wanted:
        print(f"docs/figures/{name}.png\n{figures.CAPTIONS[name]}\n")
    print(f"wrote {len(drawn)} figures to {figures.FIGURES}")


if __name__ == "__main__":
    main()
