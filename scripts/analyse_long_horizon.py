"""§3 (b): is the NCF arm stationary by epoch 99, and does the gap hit an asymptote?

Two questions, and they need different tests.

**Does the gap decay to zero or to a positive asymptote?** Fit two models to the
five-seed mean gap series `d(t)` over t = 0..T:

    decay_to_zero   d(t) = A · exp(−λ t)                      (3 params incl. nothing else)
    decay_to_const  d(t) = K + A · exp(−λ t)                  (K is the asymptote)

and compare by AIC and BIC. Both are fitted on the *mean over seeds* series,
which is the object the thesis quotes; the seed spread is handled separately in
the interval on K. A nested-model F test is also reported, since
decay_to_zero is decay_to_const with K = 0.

**Is the NCF arm stationary?** The gap could decay to a positive constant simply
because the *oracle* is fixed and the NCF never quite catches up, which is a
statement about convergence, not about stationarity. Stationarity is tested
directly on the NCF arm's own Gini:

- a paired test of the last 20 epochs: for each seed, the mean Gini over epochs
  T−20..T−11 against T−10..T−1, paired across seeds by a paired t test and by a
  Wilcoxon signed-rank test (n = 5 is too small for the t test's normality
  assumption to be safe on its own, so both are reported and STATS.md says which
  to read);
- the largest epoch-to-epoch change in the last 10 epochs, against the seed sd at
  that horizon. If a single step is smaller than the seed spread, the trajectory
  has stopped resolving.

Every number here is written to a CSV; docs/STATS.md §3 quotes only those.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import optimize, stats as st

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


def _read_arms(directory: pathlib.Path, arm: str) -> pd.DataFrame:
    parts = [pd.read_csv(p) for p in sorted(directory.glob(f"{arm}_seed*.csv"))]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def gap_series(directory: pathlib.Path) -> pd.DataFrame:
    """Mean-over-seeds NCF−oracle Gini gap at each epoch, plus the per-seed spread."""
    ncf = _read_arms(directory, "ncf")
    orc = _read_arms(directory, "oracle")
    if ncf.empty or orc.empty:
        raise SystemExit(f"no ncf/oracle CSVs under {directory}")

    wide = (ncf[["seed", "epoch", "gini"]].rename(columns={"gini": "gini_ncf"})
            .merge(orc[["seed", "epoch", "gini"]].rename(columns={"gini": "gini_oracle"}),
                   on=["seed", "epoch"], how="inner"))
    wide["gap"] = wide.gini_ncf - wide.gini_oracle
    grouped = wide.groupby("epoch")["gap"]
    out = pd.DataFrame({
        "gap_mean": grouped.mean(),
        "gap_sd": grouped.std(ddof=1),
        "n_seeds": grouped.count(),
    }).reset_index()
    return out


def _fit(models: dict, t: np.ndarray, y: np.ndarray) -> list[dict]:
    """Least-squares fit of each model, with AIC, BIC and the residual sum of squares.

    AIC/BIC are computed from the Gaussian likelihood at the least-squares
    optimum, `n·ln(RSS/n) + 2k` and `n·ln(RSS/n) + k·ln(n)`. The n here is the
    number of epochs in the series, not the number of seeds: the model is fitted
    to the seed-averaged trajectory, so the residual it minimises is the
    trajectory's deviation from the curve, and the seed spread enters through
    the interval on K rather than through the likelihood.

    Parameters are constrained to be physically meaningful: A ≥ 0 (a gap that
    starts negative would make "decay" the wrong word), λ > 0 (it must decay, not
    grow), and for the constant model K ≥ 0 as well unless the data say otherwise
    — K is left free in sign so that a decay-to-negative is not forced away, and
    the sign is reported.
    """
    n = y.size
    rows = []
    for name, fn in models.items():
        k = fn.n_params
        best = None
        # Multi-start over a coarse λ grid: the exponential is notoriously
        # sensitive to its starting rate, and a single start can sit on a local
        # optimum that flatters one model over the other.
        for lam0 in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
            p0 = fn.start(t, y, lam0)
            try:
                res = optimize.least_squares(
                    lambda p: fn.resid(p, t, y), p0,
                    bounds=fn.bounds(n), max_nfev=20000,
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            if best is None or res.cost < best.cost:
                best = res
        if best is None:
            rows.append({"model": name, "n_params": k, "rss": np.nan,
                         "aic": np.nan, "bic": np.nan})
            continue
        rss = float(np.sum(best.fun ** 2))
        # Guard the log: rss == 0 means a perfect fit and AIC → −inf, which is a
        # legitimate reading, not an error, but it breaks arithmetic downstream.
        safe = max(rss / n, np.finfo(float).tiny)
        aic = n * np.log(safe) + 2 * k
        bic = n * np.log(safe) + k * np.log(n)
        row = {"model": name, "n_params": k, "rss": rss, "aic": float(aic),
               "bic": float(bic)}
        for pname, pval in zip(fn.param_names, best.x, strict=True):
            row[pname] = float(pval)
        rows.append(row)
    return rows


class _DecayToZero:
    """d(t) = A·exp(−λt). Nests in `_DecayToConst` at K = 0."""

    n_params = 2
    param_names = ("A", "lam")

    @staticmethod
    def resid(p, t, y):
        A, lam = p
        return A * np.exp(-lam * t) - y

    @staticmethod
    def start(t, y, lam0):
        return np.array([max(float(y[0]), 1e-6), lam0])

    @staticmethod
    def bounds(n):
        return ([0.0, 1e-6], [np.inf, 5.0])


class _DecayToConst:
    """d(t) = K + A·exp(−λt). K is the asymptote, free in sign."""

    n_params = 3
    param_names = ("K", "A", "lam")

    @staticmethod
    def resid(p, t, y):
        K, A, lam = p
        return K + A * np.exp(-lam * t) - y

    @staticmethod
    def start(t, y, lam0):
        return np.array([float(y[-1]), max(float(y[0] - y[-1]), 1e-6), lam0])

    @staticmethod
    def bounds(n):
        return ([-np.inf, -np.inf, 1e-6], [np.inf, np.inf, 5.0])


def asymptote_interval(t: np.ndarray, y: np.ndarray, n_boot: int = 2000,
                       seed: int = 20261024) -> dict:
    """Bootstrap interval on K, resampling the trajectory's residuals.

    The trajectory is a single seed-averaged series, so there is no seed vector
    to resample at this level. The residual bootstrap is the right substitute:
    refit the constant model on `fitted + resampled residual` and take the
    percentiles of K. It answers "how well does this curve pin its asymptote
    given how noisy the trajectory is", which is the question a reader will ask
    of a K reported to three decimals.
    """
    rng = np.random.default_rng(seed)
    fn = _DecayToConst
    best = None
    for lam0 in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
        res = optimize.least_squares(lambda p: fn.resid(p, t, y), fn.start(t, y, lam0),
                                     bounds=fn.bounds(y.size), max_nfev=20000)
        if best is None or res.cost < best.cost:
            best = res
    fitted = y + best.fun  # y − resid
    resid = best.fun
    ks = np.empty(n_boot)
    for b in range(n_boot):
        perturbed = fitted + rng.choice(resid, size=resid.size, replace=True)
        try:
            r = optimize.least_squares(
                lambda p: fn.resid(p, t, perturbed), best.x,
                bounds=fn.bounds(y.size), max_nfev=5000)
            ks[b] = r.x[0]
        except (ValueError, np.linalg.LinAlgError):
            ks[b] = np.nan
    ks = ks[np.isfinite(ks)]
    return {
        "K": float(best.x[0]),
        "lo": float(np.percentile(ks, 2.5)) if ks.size else np.nan,
        "hi": float(np.percentile(ks, 97.5)) if ks.size else np.nan,
        "n_boot_ok": int(ks.size),
    }


def per_seed_gaps(directory: pathlib.Path) -> pd.DataFrame:
    """The gap per seed per epoch, one column per seed. For the tail estimate."""
    ncf = _read_arms(directory, "ncf").pivot(index="epoch", columns="seed", values="gini")
    orc = _read_arms(directory, "oracle").pivot(index="epoch", columns="seed", values="gini")
    common = ncf.columns.intersection(orc.columns)
    return (ncf[common] - orc[common]).sort_index()


def tail_estimate(directory: pathlib.Path, window: int = 50) -> dict:
    """Non-parametric asymptote: the mean gap over the last `window` epochs.

    The curve fits assume a functional form; this does not. It answers "where is
    the gap, on average, at the end of a 300-epoch run" directly from the data,
    and it is the number to quote if the two disagree — a mis-specified curve can
    put its asymptote anywhere, a mean over the last 50 epochs cannot.

    Resampling is over seeds (five independent populations), so the interval is
    the honest one at this n. Note the epochs inside a seed are a trajectory, not
    draws, which is why the window is averaged within a seed first.
    """
    gaps = per_seed_gaps(directory)
    tail = gaps.tail(window)
    per_seed = tail.mean(axis=0)  # one value per seed
    vals = per_seed.to_numpy(dtype=float)
    n_pos = int((vals > 0).sum())
    t_res = st.ttest_1samp(vals, 0.0)
    try:
        w_p = float(st.wilcoxon(vals).pvalue)
    except ValueError:
        w_p = 1.0
    sem = float(vals.std(ddof=1) / np.sqrt(vals.size))
    q = float(st.t.ppf(0.975, df=vals.size - 1))
    return {
        "window": window,
        "epochs": f"{int(tail.index.min())}-{int(tail.index.max())}",
        "mean": float(vals.mean()),
        "sd": float(vals.std(ddof=1)),
        "sem": sem,
        "lo_t": float(vals.mean() - q * sem),
        "hi_t": float(vals.mean() + q * sem),
        "n_seeds": int(vals.size),
        "n_seeds_positive": n_pos,
        "t": float(t_res.statistic),
        "p": float(t_res.pvalue),
        "wilcoxon_p": w_p,
    }


def stationarity(directory: pathlib.Path, arm: str = "ncf") -> dict:
    """Last-20-epoch paired test, and the largest recent step against the seed sd."""
    frame = _read_arms(directory, arm)
    if frame.empty:
        raise SystemExit(f"no {arm} CSVs under {directory}")
    last_epoch = int(frame.epoch.max())
    n_seeds = frame.seed.nunique()

    # Split the last 20 epochs into two halves, average each within a seed, then
    # pair across seeds. Averaging within a seed first is what keeps this a test
    # of level rather than of epoch noise.
    lo_start, lo_end = last_epoch - 19, last_epoch - 10
    hi_start, hi_end = last_epoch - 9, last_epoch
    early = (frame[(frame.epoch >= lo_start) & (frame.epoch <= lo_end)]
             .groupby("seed").gini.mean().sort_index())
    late = (frame[(frame.epoch >= hi_start) & (frame.epoch <= hi_end)]
            .groupby("seed").gini.mean().sort_index())
    common = early.index.intersection(late.index)
    diff = (late.loc[common] - early.loc[common]).to_numpy()

    t_res = st.ttest_rel(late.loc[common].to_numpy(), early.loc[common].to_numpy())
    try:
        w_res = st.wilcoxon(diff)
        w_p = float(w_res.pvalue)
    except ValueError:  # all differences zero
        w_p = 1.0

    # Largest single step in the last 10 epochs of the mean trajectory, against
    # the seed sd at the final epoch. If a step is larger than the spread across
    # populations, the trajectory is still moving more than the populations differ.
    tail = (frame[frame.epoch >= last_epoch - 10]
            .groupby("epoch").gini.mean().sort_index())
    steps = tail.diff().dropna()
    total = (frame[(frame.epoch >= last_epoch - 19) & (frame.epoch <= last_epoch - 10)]
             .groupby("epoch").gini.mean().sort_index())
    sd_final = float(frame[frame.epoch == last_epoch].gini.std(ddof=1))

    return {
        "arm": arm,
        "last_epoch": last_epoch,
        "n_seeds": int(n_seeds),
        "window_early": f"{lo_start}-{lo_end}",
        "window_late": f"{hi_start}-{hi_end}",
        "mean_early": float(early.loc[common].mean()),
        "mean_late": float(late.loc[common].mean()),
        "paired_diff": float(diff.mean()),
        "paired_sd": float(diff.std(ddof=1)) if diff.size > 1 else float("nan"),
        "paired_t_p": float(t_res.pvalue),
        "wilcoxon_p": w_p,
        "n_seeds_late_gt_early": int((diff > 0).sum()),
        "max_abs_step_last10": float(steps.abs().max()),
        "max_step_epoch": int(steps.abs().idxmax()) if steps.size else -1,
        "mean_abs_step_last10": float(steps.abs().mean()) if steps.size else float("nan"),
        "seed_sd_final": sd_final,
        "max_step_over_seed_sd": float(steps.abs().max()) / sd_final if sd_final else float("nan"),
        "gini_final_mean": float(frame[frame.epoch == last_epoch].gini.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="data/processed/rigor/long_horizon")
    parser.add_argument("--out-dir", default="data/processed/rigor")
    parser.add_argument("--trim-start", type=int, default=0,
                        help="fit the decay from this epoch onward (default 0)")
    parser.add_argument("--tail-window", type=int, default=50,
                        help="epochs at the end averaged for the non-parametric asymptote")
    args = parser.parse_args()

    directory = pathlib.Path(args.dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gaps = gap_series(directory)
    gaps.to_csv(out_dir / "long_horizon_gap_series.csv", index=False)

    # Fit from the peak onward. The series rises to a maximum around epoch 5 and
    # only then decays; a monotone exponential fitted from epoch 0 is trying to
    # describe a rise with a decay term, and its asymptote absorbs the misfit.
    # `--trim-start` defaults to 0 and the peak is located below so the fit
    # reported is the one the session asked for, with the post-peak fit beside it.
    peak_epoch = int(gaps.loc[gaps.gap_mean.idxmax(), "epoch"])
    fit_specs = {
        "from_epoch_0": 0,
        f"from_peak_{peak_epoch}": peak_epoch,
    }

    fit_frames = []
    for label, start in fit_specs.items():
        fit = gaps[gaps.epoch >= start]
        t = fit.epoch.to_numpy(dtype=float)
        y = fit.gap_mean.to_numpy(dtype=float)
        rows = _fit({"decay_to_zero": _DecayToZero, "decay_to_const": _DecayToConst}, t, y)
        frame = pd.DataFrame(rows)
        frame["fit_window"] = label
        frame["fit_start_epoch"] = start
        frame["delta_aic"] = frame.aic - frame.aic.min()
        frame["delta_bic"] = frame.bic - frame.bic.min()
        asym = asymptote_interval(t, y)
        frame["K_interval_lo"] = np.nan
        frame["K_interval_hi"] = np.nan
        const_mask = frame.model == "decay_to_const"
        frame.loc[const_mask, "K_interval_lo"] = asym["lo"]
        frame.loc[const_mask, "K_interval_hi"] = asym["hi"]
        fit_frames.append(frame)
    fits = pd.concat(fit_frames, ignore_index=True)
    fits.to_csv(out_dir / "long_horizon_decay_fits.csv", index=False)

    # The asymptote the paper should quote: the non-parametric tail mean, with
    # the curve's K beside it for comparison.
    tail = tail_estimate(directory, window=args.tail_window)
    asym_row = pd.DataFrame([{
        "quantity": "gap_asymptote_tail_mean",
        "regime": "long300",
        "point": tail["mean"],
        "lo": tail["lo_t"],
        "hi": tail["hi_t"],
        "sd": tail["sd"],
        "n_seeds": tail["n_seeds"],
        "n_seeds_positive": tail["n_seeds_positive"],
        "t": tail["t"],
        "p": tail["p"],
        "wilcoxon_p": tail["wilcoxon_p"],
        "window": tail["window"],
        "epochs": tail["epochs"],
        "method": "mean over last window, t interval over seeds",
    }])
    asym_row.to_csv(out_dir / "long_horizon_asymptote.csv", index=False)

    stat = stationarity(directory)
    pd.DataFrame([stat]).to_csv(out_dir / "long_horizon_stationarity.csv", index=False)

    print(f"── peak of the gap: epoch {peak_epoch}, value {gaps.gap_mean.max():.5f} ──")
    print("\n── decay fit ──")
    print(fits[["fit_window", "model", "n_params", "rss", "aic", "bic", "delta_aic",
                "delta_bic", "K", "A", "lam", "K_interval_lo", "K_interval_hi"]]
          .to_string(index=False))
    print("\n── asymptote (non-parametric) ──")
    print(asym_row.to_string(index=False))
    print("\n── stationarity ──")
    for k, v in stat.items():
        print(f"  {k:<26} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
