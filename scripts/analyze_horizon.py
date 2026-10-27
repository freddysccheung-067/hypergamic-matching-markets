"""Analyses of the second revision's experiments, one subcommand each.

    python scripts/analyze_horizon.py e09

Each subcommand reads ``data/processed/horizon/<experiment>/seed*.csv`` and the
provenance records beside them, refuses to mix commits or config hashes, and
writes its results to ``data/processed/horizon/<experiment>/analysis/``. The
paper reads only those files (``paper/gen_numbers.py``, ``paper/figures.py``).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.python.provenance import summarize_directory  # noqa: E402
from src.python.revision_stats import (  # noqa: E402
    MARGIN,
    bca_interval,
    entry_round,
    first_stays_below,
    t_interval,
    t_test,
    tost,
)

HORIZON = ROOT / "data" / "processed" / "horizon"
SETS = {"confirmation": list(range(100, 120)), "exploratory": list(range(42, 62))}
DIAG = ["spearman_mean", "spearman_q10", "spearman_q50", "spearman_q90", "item_share",
        "modal_share", "identical_share", "jaccard_oracle"]


def load(directory: pathlib.Path, seeds=None, expect_one_design: bool = True) -> pd.DataFrame:
    """All seed CSVs of one experiment directory, after checking their provenance."""
    prov = summarize_directory(directory)
    if expect_one_design and (len(prov["git_commits"]) != 1 or len(prov["config_hashes"]) != 1
                              or prov["any_dirty"] or prov["num_threads"] != [1]):
        raise SystemExit(f"{directory}: not one design: {json.dumps(prov)}")
    paths = sorted(directory.glob("seed*.csv"))
    frames = [pd.read_csv(p) for p in paths]
    data = pd.concat(frames, ignore_index=True)
    if seeds is not None:
        data = data[data.seed.isin(seeds)]
    return data


def gaps(data: pd.DataFrame, arm: str = "ncf", ref: pd.DataFrame | None = None,
         column: str = "gini", ref_column: str = "gini") -> pd.DataFrame:
    """Per (seed, round): arm Gini minus oracle Gini on the same seed and round."""
    ref = data if ref is None else ref
    a = data[data.arm == arm].set_index(["seed", "round"])[column]
    o = ref[ref.arm == "oracle"].set_index(["seed", "round"])[ref_column]
    joined = pd.concat({"arm": a, "oracle": o}, axis=1, join="inner")
    joined["gap"] = joined["arm"] - joined["oracle"]
    return joined.reset_index()


def write(frame: pd.DataFrame, directory: pathlib.Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / name, index=False)
    print(f"wrote {(directory / name).relative_to(ROOT)} ({len(frame)} rows)")


def series_by_round(per: pd.DataFrame, value: str) -> pd.DataFrame:
    rows = []
    for r, grp in per.groupby("round"):
        ti = t_interval(grp[value])
        rows.append({"round": int(r), "n": ti["n"], "mean": ti["mean"], "sd": ti["sd"],
                     "lo_t95": ti["lo"], "hi_t95": ti["hi"]})
    return pd.DataFrame(rows)


def e09(args) -> None:
    """R2-E9 and R2-E12: the canonical long-horizon run and its diagnostics."""
    directory = HORIZON / "e09_canonical"
    out = directory / "analysis"
    data = load(directory)
    last = int(data["round"].max())
    per_gap = gaps(data)
    ncf = data[data.arm == "ncf"].set_index(["seed", "round"])

    seed_rows, series_rows, summary = [], [], []
    for set_name, seeds in SETS.items():
        present = sorted(set(seeds) & set(per_gap.seed))
        if not present:
            continue
        g = per_gap[per_gap.seed.isin(present)]
        for seed in present:
            s = g[g.seed == seed].sort_values("round")
            gap = s["gap"].to_numpy()
            n = ncf.loc[seed].sort_index()
            peak_round = int(np.argmax(gap))
            entry = entry_round(gap)
            late = (n.index >= 6)
            corr = float(np.corrcoef(gap[late], n["spearman_mean"].to_numpy()[late])[0, 1])
            seed_rows.append({
                "set": set_name, "seed": seed, "peak_gap": float(gap.max()),
                "peak_round": peak_round,
                "entry_round": np.nan if entry is None else entry,
                "censored": entry is None,
                "first_inside_round": int(np.flatnonzero(np.abs(gap) < MARGIN)[0])
                if (np.abs(gap) < MARGIN).any() else np.nan,
                "tail_mean": float(gap[250:300].mean()),
                "gap_99": float(gap[99]), "gap_299": float(gap[last]),
                "gini_ncf_0": float(n.loc[0, "gini"]), "reach_ncf_0": float(n.loc[0, "reach"]),
                "modal_share_0": float(n.loc[0, "modal_share"]),
                "item_share_0": float(n.loc[0, "item_share"]),
                "gini_ncf_99": float(n.loc[99, "gini"]),
                "gini_oracle_99": float(s.set_index("round").loc[99, "oracle"]),
                "gini_oracle_tail": float(s["oracle"].to_numpy()[250:300].mean()),
                "reach_ncf_99": float(n.loc[99, "reach"]),
                "reach_ncf_299": float(n.loc[last, "reach"]),
                "mcr_ncf_99": float(n.loc[99, "mcr"]),
                "alg_loss_ncf_99": float(n.loc[99, "algorithmic_loss"]),
                "alg_loss_ncf_299": float(n.loc[last, "algorithmic_loss"]),
                "corr_gap_spearman": corr,
                "grad_steps_per_round": float(n["grad_steps"].diff().iloc[1:].mean()),
                "swipes_per_agent_per_round": float(n["mean_swipes_per_agent"].mean()),
            })

        # Per-round series over the seeds of the set: the gap with its t interval,
        # every arm's Gini, reach, the diagnostics, and the data clock.
        gs = series_by_round(g, "gap").add_prefix("gap_").rename(columns={"gap_round": "round"})
        arms = data[data.seed.isin(present)].groupby(["arm", "round"])[["gini", "reach"]].mean()
        frame = gs.copy()
        for arm in ("ncf", "oracle", "random", "popularity"):
            frame[f"gini_{arm}"] = arms.loc[arm]["gini"].reindex(frame["round"]).to_numpy()
            frame[f"reach_{arm}"] = arms.loc[arm]["reach"].reindex(frame["round"]).to_numpy()
        nd = ncf.loc[present].groupby(level="round")
        for col in DIAG:
            frame[col] = nd[col].mean().reindex(frame["round"]).to_numpy()
        frame["grad_steps"] = nd["grad_steps"].mean().reindex(frame["round"]).to_numpy()
        frame["swipes_per_agent"] = nd["mean_swipes_per_agent"].mean().reindex(
            frame["round"]).cumsum().to_numpy()
        frame.insert(0, "set", set_name)
        series_rows.append(frame)

        ps = pd.DataFrame([r for r in seed_rows if r["set"] == set_name])
        mean_gap = frame["gap_mean"].to_numpy()
        tail = tost(ps["tail_mean"])
        tail_t = t_interval(ps["tail_mean"])
        tt = t_test(ps["tail_mean"])
        g99 = t_interval(ps["gap_99"])
        b99 = bca_interval(ps["gap_99"])
        entries = ps["entry_round"].dropna()
        mean_entry = entry_round(mean_gap)
        summary.append({
            "set": set_name, "n_seeds": len(present),
            "peak_gap": float(mean_gap.max()), "peak_round": int(np.argmax(mean_gap)),
            "mean_series_entry_round": np.nan if mean_entry is None else mean_entry,
            "entry_median": float(entries.median()) if len(entries) else np.nan,
            "entry_min": float(entries.min()) if len(entries) else np.nan,
            "entry_max": float(entries.max()) if len(entries) else np.nan,
            "n_censored": int(ps["censored"].sum()),
            "tail_mean": tail_t["mean"], "tail_sd": tail_t["sd"], "tail_lo95": tail_t["lo"],
            "tail_hi95": tail_t["hi"], "tail_t": tt["t"], "tail_p": tt["p"],
            "tail_n_positive": int((ps["tail_mean"] > 0).sum()),
            "tost_t_lower": tail["t_lower"], "tost_t_upper": tail["t_upper"],
            "tost_p": tail["p"], "tost_lo90": tail["lo90"], "tost_hi90": tail["hi90"],
            "tost_passes": tail["passes"],
            "gap_99_mean": g99["mean"], "gap_99_lo95": g99["lo"], "gap_99_hi95": g99["hi"],
            "gap_99_bca_lo": b99[0], "gap_99_bca_hi": b99[1],
            "gap_99_n_positive": int((ps["gap_99"] > 0).sum()),
            "gap_299_mean": float(ps["gap_299"].mean()),
            "gini_ncf_99": float(ps["gini_ncf_99"].mean()),
            "gini_oracle_99": float(ps["gini_oracle_99"].mean()),
            "gini_oracle_tail": float(ps["gini_oracle_tail"].mean()),
            "reach_ncf_99": float(ps["reach_ncf_99"].mean()),
            "reach_ncf_299": float(ps["reach_ncf_299"].mean()),
            "alg_loss_99": float(ps["alg_loss_ncf_99"].mean()),
            "alg_loss_299": float(ps["alg_loss_ncf_299"].mean()),
            "gini_ncf_0_maxdev": float((ps["gini_ncf_0"] - 0.96).abs().max()),
            "reach_ncf_0_maxdev": float((ps["reach_ncf_0"] - 0.04).abs().max()),
            "modal_share_0_min": float(ps["modal_share_0"].min()),
            "item_share_0_min": float(ps["item_share_0"].min()),
            "n_corr_below_08": int((ps["corr_gap_spearman"] < -0.8).sum()),
            "corr_median": float(ps["corr_gap_spearman"].median()),
            "identical_below_01_by": first_stays_below(frame["identical_share"].to_numpy(), 0.1),
            "item_below_05_by": first_stays_below(frame["item_share"].to_numpy(), 0.5),
            "steps_per_pass": float(ps["grad_steps_per_round"].mean()),
        })

    write(pd.DataFrame(seed_rows), out, "per_seed.csv")
    write(pd.concat(series_rows, ignore_index=True), out, "series.csv")
    write(pd.DataFrame(summary), out, "summary.csv")
    (out / "provenance.json").write_text(json.dumps(summarize_directory(directory), indent=2),
                                         encoding="utf-8")


def _e09() -> pd.DataFrame:
    return load(HORIZON / "e09_canonical")


def decay_fractions(c_gap, w_gap, c_lvl, w_lvl, final: int) -> dict[str, float]:
    """The four decay fractions of E3d for one seed, F = ``final``."""
    out = {}
    for name, c, w in (("gap", c_gap, w_gap), ("level", c_lvl, w_lvl)):
        c, w = np.asarray(c[: final + 1]), np.asarray(w[: final + 1])
        out[f"{name}_round0"] = (w[0] - w[final]) / (c[0] - c[final])
        out[f"{name}_peak"] = (w.max() - w[final]) / (c.max() - c[final])
    return out


def e13d(args) -> None:
    """R2-E13d: the warm start, levels and gaps, four constructions, F = 299 and 99."""
    warm = load(HORIZON / "e13d_warm")
    base = _e09()
    rows = []
    for set_name, seeds in SETS.items():
        for seed in seeds:
            w = warm[(warm.seed == seed) & (warm.arm == "ncf")].sort_values("round")
            c = base[(base.seed == seed) & (base.arm == "ncf")].sort_values("round")
            o = base[(base.seed == seed) & (base.arm == "oracle")].sort_values("round")
            if w.empty or c.empty:
                continue
            wl, cl, ol = w["gini"].to_numpy(), c["gini"].to_numpy(), o["gini"].to_numpy()
            row = {"set": set_name, "seed": seed,
                   "reach_0_control": float(c["reach"].iloc[0]),
                   "reach_0_warm": float(w["reach"].iloc[0]),
                   "level_0_control": cl[0], "level_0_warm": wl[0],
                   "gap_0_control": cl[0] - ol[0], "gap_0_warm": wl[0] - ol[0],
                   "gap_peak_round_warm": int(np.argmax(wl - ol)),
                   "gap_99_control": cl[99] - ol[99], "gap_99_warm": wl[99] - ol[99],
                   "gap_299_control": cl[-1] - ol[-1], "gap_299_warm": wl[-1] - ol[-1],
                   "terminal_diff_99": wl[99] - cl[99], "terminal_diff_299": wl[-1] - cl[-1],
                   "oracle_fall": ol[0] - ol[-1]}
            for final in (len(wl) - 1, 99):
                tag = "F299" if final != 99 else "F99"
                for k, v in decay_fractions(cl - ol, wl - ol, cl, wl, final).items():
                    row[f"{k}_{tag}"] = v
            rows.append(row)
    per = pd.DataFrame(rows)
    summary = []
    for set_name, grp in per.groupby("set"):
        for col in [c for c in per.columns if c not in ("set", "seed")]:
            ti = t_interval(grp[col])
            summary.append({"set": set_name, "quantity": col, "n": ti["n"], "mean": ti["mean"],
                            "sd": ti["sd"], "lo_t95": ti["lo"], "hi_t95": ti["hi"],
                            "min": float(grp[col].min()), "max": float(grp[col].max())})
    out = HORIZON / "e13d_warm" / "analysis"
    write(per, out, "per_seed.csv")
    write(pd.DataFrame(summary), out, "summary.csv")


def e11(args) -> None:
    """R2-E11: the sweep at 300 rounds as trajectory claims C1t to C5t."""
    base = _e09()
    cells = {"shipped": base[base.seed.isin(range(100, 110))]}
    for d in sorted((HORIZON / "e11_sweep").iterdir()):
        if d.is_dir() and list(d.glob("seed*.json")):
            cells[d.name] = load(d)
    rows, series = [], []
    for cell, data in cells.items():
        g = gaps(data)
        ncf = data[data.arm == "ncf"]
        per = []
        for seed, s in g.groupby("seed"):
            gap = s.sort_values("round")["gap"].to_numpy()
            n = ncf[ncf.seed == seed].sort_values("round")
            entry = entry_round(gap)
            per.append({"peak": gap.max(), "peak_round": int(np.argmax(gap)),
                        "entry": np.nan if entry is None else entry, "tail": gap[250:300].mean(),
                        "g299": gap[-1], "g99": gap[99], "reach299": n["reach"].iloc[-1],
                        "reach99": n["reach"].iloc[99], "mcr299": n["mcr"].iloc[-1]})
        p = pd.DataFrame(per)
        peak, tail, mcr = t_interval(p["peak"]), t_interval(p["tail"]), t_interval(p["mcr299"])
        g299, b299 = t_interval(p["g299"]), bca_interval(p["g299"])
        to = tost(p["tail"])
        rows.append({
            "cell": cell, "n_seeds": len(p), "peak_mean": peak["mean"], "peak_lo": peak["lo"],
            "peak_hi": peak["hi"], "peak_round_median": float(p["peak_round"].median()),
            "entry_median": float(p["entry"].median()), "entry_min": float(p["entry"].min()),
            "entry_max": float(p["entry"].max()), "n_censored": int(p["entry"].isna().sum()),
            "gap_99": float(p["g99"].mean()), "gap_299": g299["mean"], "gap_299_lo_t": g299["lo"],
            "gap_299_hi_t": g299["hi"], "gap_299_lo_bca": b299[0], "gap_299_hi_bca": b299[1],
            "tail_mean": tail["mean"], "tail_lo": tail["lo"], "tail_hi": tail["hi"],
            "tost_lo90": to["lo90"], "tost_hi90": to["hi90"],
            "reach_99": float(p["reach99"].mean()), "reach_299": float(p["reach299"].mean()),
            "mcr_299": mcr["mean"], "mcr_299_lo": mcr["lo"],
            "C1t": bool(peak["lo"] > 0), "C2t": bool(to["passes"]),
            "C3t": bool(p["reach299"].mean() >= 0.75), "C4t": bool(mcr["lo"] > 0),
            "C5t": bool(tail["hi"] < 0),
        })
        sr = series_by_round(g, "gap")
        sr.insert(0, "cell", cell)
        sr["reach"] = ncf.groupby("round")["reach"].mean().reindex(sr["round"]).to_numpy()
        series.append(sr)
    out = HORIZON / "e11_sweep" / "analysis"
    write(pd.DataFrame(rows), out, "cells.csv")
    write(pd.concat(series, ignore_index=True), out, "series.csv")

def _fit(model: str, s: np.ndarray, y: np.ndarray):
    """Least-squares fit of one decay form; returns (params, predict) or None."""
    from scipy.optimize import curve_fit  # noqa: PLC0415

    forms = {
        "exp_const": (lambda x, k, a, lam: k + a * np.exp(-lam * x), (0.0, y[0], 0.03),
                      ([-1, 0, 1e-6], [1, 10, 5])),
        "exp": (lambda x, a, lam: a * np.exp(-lam * x), (y[0], 0.03), ([0, 1e-6], [10, 5])),
        "power": (lambda x, a, b: a * np.power(x, -b), (y[0], 0.5), ([0, 1e-6], [10, 20])),
    }
    fn, p0, bounds = forms[model]
    try:
        params, _ = curve_fit(fn, s, y, p0=p0, bounds=bounds, maxfev=20000)
    except (RuntimeError, ValueError):
        return None
    return params, (lambda x, p=params: fn(x, *p))


def _half_life(model: str, params) -> float:
    if model in ("exp_const", "exp"):
        return float(np.log(2) / params[-1])
    return float(2.0 ** (1.0 / params[1]) - 1.0)


def e10(args) -> None:
    """R2-E10: floor or slow decay, 1000 rounds, three decay forms compared out of sample."""
    data = load(HORIZON / "e10_long")
    base = _e09()
    # Rounds 0-299 must reproduce R2-E9 exactly on the same seeds.
    check = data[data["round"] < 300].merge(
        base[base.seed.isin(data.seed.unique())], on=["arm", "seed", "round"], suffixes=("", "_e9"))
    repro = float((check["gini"] - check["gini_e9"]).abs().max())
    g = gaps(data)
    wide = g.pivot(index="round", columns="seed", values="gap").sort_index()
    seeds = list(wide.columns)
    rounds = wide.index.to_numpy()

    def fit_all(mean: np.ndarray):
        peak = int(np.argmax(mean[:50]))
        s_all = rounds - peak + 1.0
        fit_mask = (rounds >= peak) & (rounds <= 500)
        test_mask = rounds > 500
        res = {}
        for model in ("exp_const", "exp", "power"):
            out = _fit(model, s_all[fit_mask], mean[fit_mask])
            if out is None:
                res[model] = None
                continue
            params, pred = out
            resid = mean[fit_mask] - pred(s_all[fit_mask])
            n, k = int(fit_mask.sum()), len(params)
            rss = float((resid ** 2).sum())
            res[model] = {
                "params": params, "peak": peak, "rss": rss,
                "aic": n * np.log(rss / n) + 2 * k, "bic": n * np.log(rss / n) + k * np.log(n),
                "oos_rmse": float(np.sqrt(np.mean((mean[test_mask] - pred(s_all[test_mask])) ** 2))),
                "half_life": _half_life(model, params),
            }
        return res

    mean = wide.mean(axis=1).to_numpy()
    fits = fit_all(mean)
    winner = min((m for m in fits if fits[m]), key=lambda m: fits[m]["oos_rmse"])
    rng = np.random.default_rng(20261027)
    boot = {m: {"half_life": [], "K": []} for m in fits}
    failures = 0
    for _ in range(2000):
        pick = rng.choice(len(seeds), size=len(seeds), replace=True)
        res = fit_all(wide.iloc[:, pick].mean(axis=1).to_numpy())
        for m, r in res.items():
            if r is None:
                failures += 1
                continue
            boot[m]["half_life"].append(r["half_life"])
            if m == "exp_const":
                boot[m]["K"].append(r["params"][0])
    rows = []
    for m, r in fits.items():
        if r is None:
            rows.append({"model": m, "fit": "failed"})
            continue
        hl = np.asarray(boot[m]["half_life"])
        row = {"model": m, "fit": "ok", "peak_round": r["peak"], "rss": r["rss"],
               "aic": r["aic"], "bic": r["bic"], "oos_rmse": r["oos_rmse"],
               "half_life": r["half_life"], "half_life_lo": float(np.percentile(hl, 2.5)),
               "half_life_hi": float(np.percentile(hl, 97.5)), "winner": m == winner,
               "params": " ".join(f"{p:.6g}" for p in r["params"])}
        if m == "exp_const":
            ks = np.asarray(boot[m]["K"])
            row.update(K=float(r["params"][0]), K_lo=float(np.percentile(ks, 2.5)),
                       K_hi=float(np.percentile(ks, 97.5)))
        rows.append(row)
    fits_df = pd.DataFrame(rows)
    fits_df["delta_aic"] = fits_df["aic"] - fits_df["aic"].min()
    fits_df["delta_bic"] = fits_df["bic"] - fits_df["bic"].min()
    tail = wide.loc[950:999].mean()
    late = wide.loc[900:949].mean() - wide.loc[950:999].mean()
    ti, to, tl = t_interval(tail), tost(tail), t_interval(late)
    k_row = fits_df[fits_df.model == "exp_const"].iloc[0]
    floor = bool(ti["lo"] > MARGIN and winner == "exp_const" and k_row["K_lo"] > MARGIN)
    summary = pd.DataFrame([{
        "n_seeds": len(seeds), "repro_max_abs_diff_rounds_0_299": repro,
        "winner": winner, "bootstrap_fit_failures": failures,
        "tail_mean": ti["mean"], "tail_lo95": ti["lo"], "tail_hi95": ti["hi"],
        "tail_n_positive": int((tail > 0).sum()), "tost_lo90": to["lo90"], "tost_hi90": to["hi90"],
        "tost_passes": to["passes"], "late_diff_mean": tl["mean"], "late_diff_lo95": tl["lo"],
        "late_diff_hi95": tl["hi"], "positive_floor": floor,
        "gap_299": float(mean[299]), "gap_499": float(mean[499]), "gap_999": float(mean[999]),
    }])
    out = HORIZON / "e10_long" / "analysis"
    write(fits_df, out, "fits.csv")
    write(summary, out, "summary.csv")
    sr = series_by_round(g, "gap")
    write(sr, out, "series.csv")


def _threshold_round(gap: np.ndarray, threshold: float, horizon: int) -> tuple[float, bool]:
    r = first_stays_below(gap, threshold)
    return (float(horizon), True) if r is None else (float(r + 1), False)


def e13a(args) -> None:
    """R2-E13a: training dose; elasticity of the decay landmark in the dose."""
    base = _e09()
    oracle = base[base.arm == "oracle"]
    doses = {1.0: base[(base.arm == "ncf") & base.seed.isin(range(100, 110))]}
    for d in sorted((HORIZON / "e13a_dose").iterdir()):
        if d.is_dir() and list(d.glob("seed*.json")):
            doses[float(d.name[1:])] = load(d)
    rows, series = [], []
    for dose, data in doses.items():
        g = gaps(data, ref=oracle)
        for seed, s in g.groupby("seed"):
            s = s.sort_values("round")
            gap = s["gap"].to_numpy()
            steps = data[(data.seed == seed) & (data.arm == "ncf")].sort_values("round")["grad_steps"]
            t05, c05 = _threshold_round(gap, 0.05, len(gap))
            t15, c15 = _threshold_round(gap, 0.15, len(gap))
            entry = entry_round(gap)
            rows.append({"dose": dose, "seed": seed, "T05": t05, "T05_censored": c05,
                         "T15": t15, "T15_censored": c15, "peak": gap.max(),
                         "entry": np.nan if entry is None else entry,
                         "steps_at_T05": float(steps.iloc[min(int(t05) - 1, len(steps) - 1)]),
                         "steps_per_round": float(steps.diff().iloc[1:].mean())})
        sr = series_by_round(g, "gap")
        sr["grad_steps"] = data[data.arm == "ncf"].groupby("round")["grad_steps"].mean().reindex(
            sr["round"]).to_numpy()
        sr.insert(0, "dose", dose)
        series.append(sr)
    per = pd.DataFrame(rows)

    def elasticity(frame: pd.DataFrame, col: str) -> float:
        x, y = np.log(frame["dose"].to_numpy()), np.log(frame[col].to_numpy())
        return float(np.polyfit(x, y, 1)[0])

    rng = np.random.default_rng(20261027)
    seeds = sorted(per.seed.unique())
    summary = []
    for col in ("T05", "T15"):
        boot = []
        for _ in range(2000):
            pick = rng.choice(seeds, size=len(seeds), replace=True)
            boot.append(elasticity(pd.concat([per[per.seed == s] for s in pick]), col))
        e = elasticity(per, col)
        summary.append({"landmark": col, "elasticity": e, "lo": float(np.percentile(boot, 2.5)),
                        "hi": float(np.percentile(boot, 97.5)),
                        "n_censored": int(per[f"{col}_censored"].sum()),
                        "reading": "steps" if e <= -0.75 else ("rounds" if e >= -0.25 else "partial")})
    by_dose = per.groupby("dose").agg(T05=("T05", "mean"), T15=("T15", "mean"),
                                      peak=("peak", "mean"), entry=("entry", "median"),
                                      steps_at_T05=("steps_at_T05", "mean"),
                                      steps_per_round=("steps_per_round", "mean")).reset_index()
    out = HORIZON / "e13a_dose" / "analysis"
    write(per, out, "per_seed.csv")
    write(by_dose, out, "by_dose.csv")
    write(pd.DataFrame(summary), out, "summary.csv")
    write(pd.concat(series, ignore_index=True), out, "series.csv")

CHECK_ROUNDS = (20, 49, 99, 299)


def _paired_at(diff: pd.DataFrame, label: str) -> list[dict]:
    """t interval of a per-seed difference at each registered round."""
    out = []
    for r in CHECK_ROUNDS:
        ti = t_interval(diff[diff["round"] == r]["d"])
        band = 0.02 if r == 20 else 0.01
        out.append({"comparison": label, "round": r, "n": ti["n"], "mean": ti["mean"],
                    "lo_t95": ti["lo"], "hi_t95": ti["hi"], "band": band,
                    "inside_band": bool(ti["lo"] > -band and ti["hi"] < band)})
    return out


def e13b(args) -> None:
    """R2-E13b: exploration; the exploiting-part gap and the whole-list gap."""
    base = _e09()
    control = gaps(base)[["seed", "round", "gap"]].rename(columns={"gap": "control"})
    oracle = base[base.arm == "oracle"]
    rows, series = [], []
    for share, prefix in (("0.05", 19), ("0.2", 16)):
        ncf = load(HORIZON / "e13b_explore" / f"e{share}")
        orp = load(HORIZON / "e13b_explore" / f"oracle_p{prefix}")
        exploit = gaps(ncf, ref=orp, column="gini_prefix", ref_column="gini_prefix")
        whole = gaps(ncf, ref=oracle)
        m = exploit[["seed", "round", "gap"]].rename(columns={"gap": "exploit"}).merge(
            whole[["seed", "round", "gap"]].rename(columns={"gap": "whole"}), on=["seed", "round"]
        ).merge(control, on=["seed", "round"])
        m["d"] = m["exploit"] - m["control"]
        rows += _paired_at(m, f"exploit_minus_control_share{share}")
        m2 = m.assign(d=m["whole"] - m["control"])
        rows += _paired_at(m2, f"whole_minus_control_share{share}")
        sr = m.groupby("round")[["exploit", "whole", "control"]].mean().reset_index()
        sr.insert(0, "share", float(share))
        series.append(sr)
    out = HORIZON / "e13b_explore" / "analysis"
    write(pd.DataFrame(rows), out, "paired.csv")
    write(pd.concat(series, ignore_index=True), out, "series.csv")


def e13c(args) -> None:
    """R2-E13c: a shadow ranker trained off-policy on the oracle's or random arm's data."""
    base = _e09()
    shadow = load(HORIZON / "e13c_offpolicy")
    on = gaps(base)[["seed", "round", "gap"]].rename(columns={"gap": "on_policy"})
    o9 = base[base.arm == "oracle"][["seed", "round", "gini"]].rename(columns={"gini": "oracle"})
    # The oracle arm here is the oracle arm of R2-E9 run again; it must agree exactly.
    same = shadow[shadow.arm == "oracle"].merge(o9, on=["seed", "round"])
    oracle_repro = float((same["gini"] - same["oracle"]).abs().max())
    rows, series = [], []
    for arm in ("oracle", "random"):
        s = shadow[shadow.arm == arm][["seed", "round", "shadow_gini", "shadow_reach"]]
        m = s.merge(o9, on=["seed", "round"]).merge(on, on=["seed", "round"])
        m["off_policy"] = m["shadow_gini"] - m["oracle"]
        m["d"] = m["off_policy"] - m["on_policy"]
        rows += _paired_at(m, f"offpolicy_{arm}_minus_onpolicy")
        sr = m.groupby("round")[["off_policy", "on_policy", "shadow_reach"]].mean().reset_index()
        sr.insert(0, "data_from", arm)
        series.append(sr)
    out = HORIZON / "e13c_offpolicy" / "analysis"
    paired = pd.DataFrame(rows)
    paired["oracle_arm_repro_max_abs_diff"] = oracle_repro
    write(paired, out, "paired.csv")
    write(pd.concat(series, ignore_index=True), out, "series.csv")


def e14(args) -> None:
    """R2-E14: arrivals; steady-state gap, stationarity, tenure cohorts."""
    rows, series = [], []
    for d in sorted((HORIZON / "e14_arrivals").iterdir()):
        if not (d.is_dir() and list(d.glob("seed*.json"))):
            continue
        rate = float(d.name[1:])
        data = load(d)
        g = gaps(data)
        wide = g.pivot(index="round", columns="seed", values="gap").sort_index()
        steady = wide.loc[500:599].mean()
        ti, tt = t_interval(steady), t_test(steady)
        paired = wide.loc[590:599].mean() - wide.loc[580:589].mean()
        pt = t_test(paired)
        mean_series = wide.mean(axis=1)
        max_step = float(mean_series.loc[580:599].diff().abs().max())
        seed_sd = float(wide.loc[599].std(ddof=1))
        row = {"rate": rate, "n_seeds": wide.shape[1], "steady_gap": ti["mean"],
               "steady_lo95": ti["lo"], "steady_hi95": ti["hi"], "steady_p": tt["p"],
               "steady_n_positive": int((steady > 0).sum()),
               "stationary_paired_p": pt["p"], "stationary_max_step": max_step,
               "stationary_seed_sd": seed_sd,
               "stationary": bool(pt["p"] >= 0.05 and max_step < 0.1 * seed_sd),
               "gini_ncf": float(data[(data.arm == "ncf") & (data["round"] >= 500)]["gini"].mean()),
               "gini_oracle": float(data[(data.arm == "oracle") & (data["round"] >= 500)]["gini"].mean()),
               "reach_ncf": float(data[(data.arm == "ncf") & (data["round"] >= 500)]["reach"].mean())}
        late = data[data["round"] >= 500]
        for band in ("t000_009", "t010_049", "t050_up"):
            ratio = late[f"cohort_{band}_exposure"] / late[f"cohort_{band}_pop"]
            by_arm = late.assign(ratio=ratio).groupby(["arm", "seed"])["ratio"].mean()
            row[f"ratio_{band}_ncf"] = float(by_arm.loc["ncf"].mean())
            row[f"ratio_{band}_oracle"] = float(by_arm.loc["oracle"].mean())
            row[f"pop_{band}"] = float(late[f"cohort_{band}_pop"].mean())
        rows.append(row)
        sr = series_by_round(g, "gap")
        sr.insert(0, "rate", rate)
        series.append(sr)
    out = HORIZON / "e14_arrivals" / "analysis"
    write(pd.DataFrame(rows), out, "summary.csv")
    write(pd.concat(series, ignore_index=True), out, "series.csv")


def e15(args) -> None:
    """R2-E15: per-cell welfare difference B - A, both A arms, BH over 40 cells each."""
    from src.python.revision_stats import bh_qvalues, bh_reject  # noqa: PLC0415

    data = pd.concat([pd.read_csv(p) for p in sorted((HORIZON / "e15_institutional").glob("seed*.csv"))])
    rows = []
    for arm, grp in data.groupby("a_arm"):
        cells = []
        for (c, sf), cell in grp.groupby(["signal_cost", "size_fraction"]):
            ti, tt = t_interval(cell["welfare_diff_residual"]), t_test(cell["welfare_diff_residual"])
            cells.append({"a_arm": arm, "signal_cost": c, "size_fraction": sf, "n": ti["n"],
                          "mean": ti["mean"], "sd": ti["sd"], "lo_t95": ti["lo"], "hi_t95": ti["hi"],
                          "p": tt["p"], "welfare_a": float(cell["welfare_a"].mean()),
                          "clearance_b": float(cell["clearance_b"].mean())})
        frame = pd.DataFrame(cells)
        frame["q_bh"] = bh_qvalues(frame["p"])
        frame["bh_reject"] = bh_reject(frame["p"])
        frame["b_beats_a"] = frame["bh_reject"] & (frame["mean"] > 0)
        frame["a_beats_b"] = frame["bh_reject"] & (frame["mean"] < 0)
        rows.append(frame)
    out = HORIZON / "e15_institutional" / "analysis"
    cells = pd.concat(rows, ignore_index=True)
    write(cells, out, "cells.csv")
    summary = cells.groupby("a_arm").agg(n_cells=("mean", "size"), n_b_beats_a=("b_beats_a", "sum"),
                                         n_a_beats_b=("a_beats_b", "sum"), best=("mean", "max"),
                                         worst=("mean", "min"), max_hi=("hi_t95", "max")).reset_index()
    write(summary, out, "summary.csv")

    inert = pd.concat([pd.read_csv(p) for p in sorted((HORIZON / "e15i_inert").glob("seed*.csv"))])
    by = inert.groupby("size_fraction")["welfare_diff_residual"].agg(["mean", "std", "size"])
    by = by.sort_values("mean", ascending=False).reset_index()
    by["rank"] = np.arange(1, len(by) + 1)
    write(by, out, "inert_c001.csv")


def e16(args) -> None:
    """R2-E16: floating-point sensitivity; within-seed against between-seed spread."""
    base = _e09()
    oracle = base[base.arm == "oracle"]
    runs = []
    for d in sorted((HORIZON / "e16_fp").iterdir()):
        if d.is_dir() and list(d.glob("seed*.json")):
            g = gaps(load(d), ref=oracle)
            g["perturbation"] = d.name
            runs.append(g)
    allg = pd.concat(runs, ignore_index=True)
    within = allg.groupby(["seed", "round"])["gap"].std(ddof=1).groupby("round").mean()
    conf = gaps(base[base.seed.isin(range(100, 120))])
    between = conf.groupby("round")["gap"].std(ddof=1)
    frame = pd.DataFrame({"within_sd": within, "between_sd": between}).dropna().reset_index()
    frame["ratio"] = frame["within_sd"] / frame["between_sd"]
    unperturbed = gaps(base[base.seed.isin(range(100, 105))])
    diff_vs_base = allg.merge(unperturbed[["seed", "round", "gap"]], on=["seed", "round"],
                              suffixes=("", "_base"))
    window = frame[(frame["round"] >= 10) & (frame["round"] <= 150)]
    summary = pd.DataFrame([{
        "within_sd_round0": float(frame.loc[frame["round"] == 0, "within_sd"].iloc[0]),
        "max_ratio_10_150": float(window["ratio"].max()),
        "round_of_max_ratio": int(window.loc[window["ratio"].idxmax(), "round"]),
        "ratio_round_99": float(frame.loc[frame["round"] == 99, "ratio"].iloc[0]),
        "ratio_round_299": float(frame.loc[frame["round"] == 299, "ratio"].iloc[0]),
        "first_round_any_run_differs": int(diff_vs_base.loc[
            (diff_vs_base["gap"] - diff_vs_base["gap_base"]).abs() > 0, "round"].min()),
        "n_runs": int(allg.groupby(["seed", "perturbation"]).ngroups),
    }])
    out = HORIZON / "e16_fp" / "analysis"
    write(frame, out, "spread.csv")
    write(summary, out, "summary.csv")


def e17(args) -> None:
    """R2-E17 and R2-E18: population size, and every horizon in data units."""
    base = _e09()
    sizes = {1000: base[base.seed.isin(range(100, 110))]}
    for d in sorted((HORIZON / "e17_size").iterdir()):
        if d.is_dir() and list(d.glob("seed*.json")):
            sizes[int(d.name[1:])] = load(d)
    rows, series = [], []
    for n, data in sizes.items():
        g = gaps(data)
        ncf = data[data.arm == "ncf"]
        per = []
        for seed, s in g.groupby("seed"):
            gap = s.sort_values("round")["gap"].to_numpy()
            nn = ncf[ncf.seed == seed].sort_values("round")
            t05, c05 = _threshold_round(gap, 0.05, len(gap))
            entry = entry_round(gap)
            swipes = nn["mean_swipes_per_agent"].cumsum().to_numpy()
            steps = nn["grad_steps"].to_numpy()
            idx = min(int(t05) - 1, len(gap) - 1)
            per.append({"peak": gap.max(), "T05": t05, "c05": c05,
                        "entry": np.nan if entry is None else entry,
                        "gini0": nn["gini"].iloc[0], "swipes_at_T05": swipes[idx],
                        "steps_at_T05": steps[idx], "steps_per_round": float(np.diff(steps).mean()),
                        "swipes_per_round": float(nn["mean_swipes_per_agent"].mean())})
        p = pd.DataFrame(per)
        rows.append({"N": n, "n_seeds": len(p), "peak": float(p["peak"].mean()),
                     "T05": float(p["T05"].mean()), "T05_censored": int(p["c05"].sum()),
                     "entry_median": float(p["entry"].median()),
                     "n_censored": int(p["entry"].isna().sum()),
                     "gini0_maxdev": float((p["gini0"] - (1 - 40 / n)).abs().max()),
                     "swipes_at_T05": float(p["swipes_at_T05"].mean()),
                     "steps_at_T05": float(p["steps_at_T05"].mean()),
                     "steps_per_round": float(p["steps_per_round"].mean()),
                     "swipes_per_round": float(p["swipes_per_round"].mean())})
        sr = series_by_round(g, "gap")
        sr["swipes_per_agent"] = ncf.groupby("round")["mean_swipes_per_agent"].mean().reindex(
            sr["round"]).cumsum().to_numpy()
        sr["grad_steps"] = ncf.groupby("round")["grad_steps"].mean().reindex(sr["round"]).to_numpy()
        sr.insert(0, "N", n)
        series.append(sr)
    out = HORIZON / "e17_size" / "analysis"
    write(pd.DataFrame(rows), out, "summary.csv")
    write(pd.concat(series, ignore_index=True), out, "series.csv")


def appendix(args) -> None:
    """R2-H1r and R2-H5r."""
    h1 = pd.read_csv(HORIZON / "h1r" / "h1.csv")
    curve = h1.groupby("signal_cost").agg(mean_tau=("mean_tau", "mean"),
                                          n_proposals=("n_proposals", "mean"),
                                          tau_star=("tau_star", "mean"),
                                          tau_trunc=("tau_trunc", "mean")).reset_index()
    live = curve[curve["n_proposals"] > 0]["mean_tau"].to_numpy()
    curve["monotone_non_decreasing"] = bool(np.all(np.diff(live) >= 0))
    out = HORIZON / "h1r" / "analysis"
    write(curve, out, "curve.csv")
    h5 = pd.concat([pd.read_csv(p) for p in sorted((HORIZON / "h5r").glob("seed*.csv"))])
    rows = []
    for term in ("exogenous", "algorithmic", "interaction"):
        ti = t_interval(h5[term])
        rows.append({"term": term, "n": ti["n"], "mean": ti["mean"], "sd": ti["sd"],
                     "lo_t95": ti["lo"], "hi_t95": ti["hi"], "p": t_test(h5[term])["p"]})
    write(pd.DataFrame(rows), HORIZON / "h5r" / "analysis", "terms.csv")

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment")
    parser.add_argument("--partial", action="store_true",
                        help="allow a directory whose runs are not all finished")
    args = parser.parse_args()
    table = {"e09": e09, "e10": e10, "e11": e11, "e13a": e13a, "e13b": e13b, "e13c": e13c,
             "e13d": e13d, "e14": e14, "e15": e15, "e16": e16, "e17": e17,
             "appendix": appendix}
    table[args.experiment](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
