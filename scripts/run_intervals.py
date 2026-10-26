"""Build data/processed/rigor/intervals.csv — one 95% interval per headline quantity.

Every row is a (quantity, regime) pair with a point estimate, a BCa interval, the
percentile interval beside it, a Student-t interval as the small-n floor, and the
seed count behind it. Read docs/STATS.md §1 for what each column licenses.

**The seven headline quantities, and where each comes from**

The list is fixed by the session and is the set THESIS.md quotes as mean ± sd.

1. `gini_ncf` by regime — Theme A's concentration result.
2. `gini_gap_ncf_minus_oracle` — H3. Paired by seed.
3. `mcr` (clearing rate), ncf arm — the market's own outcome.
4. `rsi` (rank satisfaction), ncf arm — M-proposing, so a lower bound (§5).
5. `algorithmic_loss` — (SW_full − SW_alg)/SW_full, ncf arm, epoch-final.
6. `welfare_common` by environment — A and B, from the institutional grid.
7. `welfare_diff_residual` — B−A under residual accounting, the conservative
   headline, once pooled and once at the shipped optimum cell.

**Regimes.** `inert` is φ = 0.2 at 20 epochs; `live20` is φ = 0.02 at 20; `live100`
is φ = 0.02 at 100 (the shipped horizon); `long300` is φ = 0.02 at 300 (§3).
Each is its own directory with its own README, and the seed set is the same five
in every case, so a difference between rows is a difference of regime and not of
population.

**Gaps are formed per seed before resampling** (`stats.paired_gap`). Two arms at
one seed share a population; averaging each arm first and subtracting would drop
the pairing and inflate the interval. This is the same convention as
`run_ablation.py`, carried into the interval.

Output is deterministic: every bootstrap is seeded from a fixed constant, so
re-running reproduces the file byte for byte.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.python.stats import Interval, bca_interval, paired_gap, t_interval  # noqa: E402

# Fixed bootstrap seed. Any constant works; what matters is that it does not
# change, so a committed table can be regenerated exactly.
BOOTSTRAP_SEED = 20261024
RESAMPLES = 10_000

# Where each regime's per-(arm, seed) frame lives. `epochs` is the final epoch
# index the run reached: the quantities here are all end-of-run, never pooled
# over epochs, because epochs inside a run are a trajectory rather than draws.
REGIMES = {
    "inert20": dict(
        dir="data/processed/ablation", epochs=20,
        note="phi=0.2, 20 epochs (inert gate)"),
    "live20": dict(
        dir="data/processed/ablation_live", epochs=20,
        note="phi=0.02, 20 epochs (live gate)"),
    "live100": dict(
        dir="data/processed/ablation_h100", epochs=100,
        note="phi=0.02, 100 epochs (shipped horizon)"),
    "long300": dict(
        dir="data/processed/rigor/long_horizon", epochs=300,
        note="phi=0.02, 300 epochs (convergence run, §3)"),
    "base20": dict(
        dir="data/processed/rigor/base", epochs=100,
        note="phi=0.02, 100 epochs, full metrics incl. algorithmic_loss"),
    # The 300-epoch run regenerated at 1 thread. long300 is kept because it is the
    # committed record, but its NCF arm cannot be regenerated from this code
    # (docs/REVIEW_RESPONSE.md A3); this is the series the paper quotes.
    "long300_1t": dict(
        dir="data/processed/rigor/long_horizon_1t", epochs=300,
        note="phi=0.02, 300 epochs, regenerated at 1 thread (PREREGISTRATION E7)"),
}


def _load_trimmed(directory: pathlib.Path) -> dict[str, pd.DataFrame]:
    """Read `<arm>_seed<seed>.csv` files written by run_ablation.py."""
    frames: dict[str, list[pd.DataFrame]] = {}
    for path in sorted(directory.glob("*_seed*.csv")):
        arm = path.name.split("_seed")[0]
        frame = pd.read_csv(path)
        frame["arm"] = arm
        frames.setdefault(arm, []).append(frame)
    return {arm: pd.concat(fs, ignore_index=True) for arm, fs in frames.items()}


def _load_metrics(directory: pathlib.Path) -> dict[str, pd.DataFrame]:
    """Read `metrics_seed<seed>.csv` files written by run_rigor_base.py."""
    parts = [pd.read_csv(p) for p in sorted(directory.glob("metrics_seed*.csv"))]
    if not parts:
        return {}
    table = pd.concat(parts, ignore_index=True)
    return {arm: grp for arm, grp in table.groupby("arm")}


def final_by_seed(frames: dict[str, pd.DataFrame], arm: str,
                  metric: str) -> pd.Series:
    """That arm's final-epoch value of `metric`, indexed by seed.

    The final epoch, not the mean over epochs: the run's own history has been
    fed back through θ by then, which is the state the thesis describes. Taking
    the mean over a non-stationary trajectory would average a transient into the
    steady state and understate the trend §3 tests for.
    """
    frame = frames[arm]
    last = frame.sort_values("epoch").groupby("seed").tail(1)
    return last.set_index("seed")[metric].astype(float).sort_index()


def _interval_rows(
    quantity: str,
    regime: str,
    iv: Interval,
    *,
    environment: str = "",
    note: str = "",
    n_units: int | None = None,
    unit: str = "seed",
    n_seeds: int | None = None,
) -> dict:
    """One CSV row: BCa as the headline, the percentile and t intervals beside it.

    `n_seeds` is the number of *independent populations* behind the row, while
    `n_units` is the number of resampling units, and the two differ whenever the
    row pools design cells: the institutional grid has 40 cells over 5 seeds, so a
    pooled row was resampled over its 40 cells while resting on 5 populations.
    `n_seeds` therefore defaults to the interval's own count but must be
    overridden to the grid's seed count wherever the row is pooled -- 40 is not a
    seed count and reporting it as one would misstate the replication.

    The t columns are filled by the caller, which has the raw series; they are
    present and NaN here so the column order is fixed even for a row that came
    from a gap (`paired_gap` returns only an Interval).
    """
    return {
        "quantity": quantity,
        "regime": regime,
        "environment": environment,
        "point": iv.point,
        "lo": iv.lo,
        "hi": iv.hi,
        "lo_percentile": iv.lo_percentile,
        "hi_percentile": iv.hi_percentile,
        "lo_t": float("nan"),
        "hi_t": float("nan"),
        "n_seeds": n_seeds if n_seeds is not None else iv.n_seeds,
        "n_units": n_units if n_units is not None else iv.n_seeds,
        "unit": unit,
        "n_resamples": iv.n_resamples,
        "method": iv.method,
        "note": note,
    }


def build(regimes: list[str]) -> pd.DataFrame:
    rows: list[dict] = []

    # ── the ablation-derived quantities, by regime ────────────────────────────
    for regime in regimes:
        spec = REGIMES[regime]
        directory = pathlib.Path(spec["dir"])
        if not directory.is_dir():
            print(f"  skip {regime}: {directory} missing", file=sys.stderr)
            continue

        has_full = any(directory.glob("metrics_seed*.csv"))
        frames = _load_metrics(directory) if has_full else _load_trimmed(directory)
        if not frames:
            print(f"  skip {regime}: no per-seed CSVs", file=sys.stderr)
            continue

        ncf_gini = final_by_seed(frames, "ncf", "gini")
        orc_gini = final_by_seed(frames, "oracle", "gini")

        def add(quantity, series, *, environment="", note=""):
            iv = bca_interval(series.to_numpy(), n_resamples=RESAMPLES,
                              rng=BOOTSTRAP_SEED)
            _, lo_t, hi_t = t_interval(series.to_numpy())
            r = _interval_rows(quantity, regime, iv, environment=environment, note=note)
            r["lo_t"], r["hi_t"] = lo_t, hi_t
            rows.append(r)

        add("gini_ncf", ncf_gini, note=spec["note"])
        add("gini_oracle", orc_gini, note=spec["note"])

        # H3's gap, paired by seed.
        common = ncf_gini.index.intersection(orc_gini.index)
        gap = paired_gap(ncf_gini.loc[common].to_numpy(),
                         orc_gini.loc[common].to_numpy(),
                         n_resamples=RESAMPLES, rng=BOOTSTRAP_SEED)
        _, lo_t, hi_t = t_interval((ncf_gini.loc[common] - orc_gini.loc[common]).to_numpy())
        r = _interval_rows("gini_gap_ncf_minus_oracle", regime, gap, note=spec["note"])
        r["lo_t"], r["hi_t"] = lo_t, hi_t
        rows.append(r)

        add("mcr_ncf", final_by_seed(frames, "ncf", "mcr"), note=spec["note"])
        add("rsi_ncf", final_by_seed(frames, "ncf", "rsi"), note=spec["note"])

        if has_full:
            add("algorithmic_loss_ncf",
                final_by_seed(frames, "ncf", "algorithmic_loss"),
                note=spec["note"])

        print(f"  {regime}: {len(common)} paired seeds", flush=True)

    # ── the institutional quantities ─────────────────────────────────────────
    grid_path = pathlib.Path("data/processed/institutional/grid.csv")
    if grid_path.is_file():
        grid = pd.read_csv(grid_path)

        # welfare_common by environment, pooled over every cell. The resampling
        # unit here is the CELL, not the row: the 200 rows are 40 cells times 5
        # seeds, and the five seeds inside a cell are five populations but the
        # cell is what the design varied. Resampling rows would treat the 5x
        # replication as 200 independent draws and shrink the interval by about
        # sqrt(5), which is the mistake this whole exercise exists to avoid.
        n_seeds = int(grid.seed.nunique())
        for env, col in (("A", "welfare_common_a"), ("B", "welfare_common_b")):
            cell_means = grid.groupby(["signal_cost", "size_fraction"])[col].mean()
            iv = bca_interval(cell_means.to_numpy(), n_resamples=RESAMPLES,
                              rng=BOOTSTRAP_SEED)
            _, lo_t, hi_t = t_interval(cell_means.to_numpy())
            # A's welfare is literally identical in all 40 cells: the oracle arm
            # does not read c_B or size_fraction, so every cell is the same run.
            # The zero-width interval is therefore correct, not degenerate, and
            # the note has to say so or a reader will read it as a failure.
            constant = cell_means.nunique() == 1
            note = (f"40 cells over {n_seeds} seeds; interval resampled over cells")
            if constant:
                note += ("; A is the oracle baseline and does not read c_B or "
                         "size_fraction, so this is one run repeated — zero width is structural")
            r = _interval_rows(
                "welfare_common", "grid_pooled", iv, environment=env,
                n_units=int(cell_means.size), unit="cell", n_seeds=n_seeds,
                note=note)
            r["lo_t"], r["hi_t"] = lo_t, hi_t
            rows.append(r)

        # The shipped optimum cell, where H4 lives. Five seeds, so the seed is
        # the unit and n_seeds == n_units.
        opt = grid[(grid.signal_cost == 0.01) & (grid.size_fraction == 0.65)]
        if not opt.empty:
            for col, label in (
                ("welfare_diff_residual", "welfare_diff_residual"),
                ("welfare_diff", "welfare_diff"),
            ):
                iv = bca_interval(opt[col].to_numpy(), n_resamples=RESAMPLES,
                                  rng=BOOTSTRAP_SEED)
                _, lo_t, hi_t = t_interval(opt[col].to_numpy())
                r = _interval_rows(label, "optimum_c0.01_sf0.65", iv,
                                   environment="B-A",
                                   note="the H4 optimum cell; §5 corrects this for 200 cells")
                r["lo_t"], r["hi_t"] = lo_t, hi_t
                rows.append(r)

        # Residual across the whole factorial grid, pooled by cell as above.
        cell_means = grid.groupby(["signal_cost", "size_fraction"])[
            "welfare_diff_residual"].mean()
        iv = bca_interval(cell_means.to_numpy(), n_resamples=RESAMPLES,
                          rng=BOOTSTRAP_SEED)
        _, lo_t, hi_t = t_interval(cell_means.to_numpy())
        r = _interval_rows("welfare_diff_residual", "grid_pooled", iv, environment="B-A",
                           n_units=int(cell_means.size), unit="cell", n_seeds=n_seeds,
                           note=f"40 cells over {n_seeds} seeds; the distribution §5 corrects within")
        r["lo_t"], r["hi_t"] = lo_t, hi_t
        rows.append(r)

        # The spread across seeds inside one cell, quoted so §2 can compare the
        # within-cell seed sd against the across-cell spread. This is the number
        # the power analysis needs and the pooled interval does not give.
        within = (grid.groupby(["signal_cost", "size_fraction"])[
            "welfare_diff_residual"].std(ddof=1))
        rows.append({
            "quantity": "welfare_diff_residual_within_cell_sd",
            "regime": "grid_pooled",
            "environment": "B-A",
            "point": float(within.mean()),
            "lo": float(within.min()),
            "hi": float(within.max()),
            "lo_percentile": float(within.min()),
            "hi_percentile": float(within.max()),
            "lo_t": float(within.min()),
            "hi_t": float(within.max()),
            "n_seeds": n_seeds,
            "n_units": int(within.size),
            "unit": "cell",
            "n_resamples": 0,
            "method": "descriptive",
            "note": "mean and range of the per-cell seed sd (ddof=1); lo/hi are the min/max cell",
        })

    # H5's interaction term, from the factorial decomposition.
    fac_path = pathlib.Path("data/processed/institutional/factorial.csv")
    if fac_path.is_file():
        fac = pd.read_csv(fac_path)
        for col in ("interaction", "exogenous", "algorithmic"):
            iv = bca_interval(fac[col].to_numpy(), n_resamples=RESAMPLES,
                              rng=BOOTSTRAP_SEED)
            _, lo_t, hi_t = t_interval(fac[col].to_numpy())
            r = _interval_rows(col, "factorial", iv,
                               note="the drift x mediation decomposition, 5 seeds")
            r["lo_t"], r["hi_t"] = lo_t, hi_t
            rows.append(r)

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regimes",
                        default="inert20,live20,live100,long300,base20,long300_1t")
    parser.add_argument("--output", default="data/processed/rigor/intervals.csv")
    args = parser.parse_args()

    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    table = build(regimes)

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    width = table.hi - table.lo
    print(f"\nwrote {len(table)} rows to {out}")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(table.assign(width=width)[
            ["quantity", "regime", "environment", "point", "lo", "hi", "width", "n_seeds"]
        ].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
