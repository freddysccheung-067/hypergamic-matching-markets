"""Quantities the paper quotes that the per-experiment analyses do not write.

    python scripts/analyze_extras.py

Reads the seed files and the analysis outputs of ``scripts/analyze_horizon.py``
and writes, under ``data/processed/horizon/``:

    e09_canonical/analysis/extras.csv      oracle loss, settle round, clearing,
                                            five-against-twenty seeds, BCa agreement
    e11_sweep/analysis/per_seed.csv        entry round, T(0.05), oracle Gini and reach
    e13b_explore/analysis/check.csv        R2-E13b-check, exploring head against a
                                            non-exploring head of the same length
    e14_arrivals/analysis/trend.csv        R2-E14-check, the trend over rounds 400-599
    e15s_five/analysis/cells.csv           R2-E15s, the five-round grid
    units.csv                              T(0.05) in rounds, rows per agent and steps
    run_index.csv                          one row per experiment directory
    predictions.csv                        every registered prediction, scored

Every scoring rule in ``predictions.csv`` is the threshold its registration in
``docs/PREREGISTRATION.md`` states, applied in code rather than by reading.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_horizon as ah  # noqa: E402
from src.python.revision_stats import (  # noqa: E402
    MARGIN,
    bh_qvalues,
    entry_round,
    first_stays_below,
    t_interval,
    t_test,
)

H = ah.HORIZON


def t05_units(gap: np.ndarray, ncf: pd.DataFrame) -> dict[str, float]:
    """T(0.05) as in R2-E13a, in rounds, rows of log per agent, and optimizer steps.

    ``ncf`` is one seed's NCF rows sorted by round. T is one plus the first round
    from which the gap stays below 0.05, censored at the horizon; the data and the
    steps are the cumulative totals through round T - 1.
    """
    r = first_stays_below(gap, 0.05)
    censored = r is None
    t = len(gap) if censored else r + 1
    idx = min(t, len(gap)) - 1
    rows = ncf["mean_swipes_per_agent"].to_numpy().cumsum()
    steps = ncf["grad_steps"].to_numpy()
    return {"T05": float(t), "T05_censored": censored, "rows_at_T05": float(rows[idx]),
            "steps_at_T05": float(steps[idx])}


def per_seed_t05(data: pd.DataFrame, ref: pd.DataFrame | None = None,
                 arm: str = "ncf") -> pd.DataFrame:
    g = ah.gaps(data, arm=arm, ref=ref)
    out = []
    for seed, s in g.groupby("seed"):
        gap = s.sort_values("round")["gap"].to_numpy()
        ncf = data[(data.seed == seed) & (data.arm == arm)].sort_values("round")
        out.append({"seed": seed, **t05_units(gap, ncf)})
    return pd.DataFrame(out)


def write_long(rows: dict[str, float], path: pathlib.Path) -> None:
    frame = pd.DataFrame([{"quantity": k, "value": v} for k, v in rows.items()])
    ah.write(frame, path.parent, path.name)


def e09_extras() -> dict[str, float]:
    """Oracle loss, settle round, clearing, five against twenty seeds, BCa agreement."""
    data = ah._e09()
    out: dict[str, float] = {}
    conf = data[data.seed.isin(ah.SETS["confirmation"])]
    at = conf.set_index(["arm", "seed", "round"])
    losses = {}
    for arm in ("ncf", "oracle"):
        loss = at.xs((arm, 299), level=("arm", "round"))["algorithmic_loss"]
        losses[arm] = loss
        ti = t_interval(loss)
        out[f"alg_loss_{arm}_299"], out[f"alg_loss_{arm}_299_lo"] = ti["mean"], ti["lo"]
        out[f"alg_loss_{arm}_299_hi"] = ti["hi"]
        for r in (0, 299):
            out[f"mcr_{arm}_{r}"] = float(at.xs((arm, r), level=("arm", "round"))["mcr"].mean())
    # Paired per seed: the ranker's algorithmic loss minus the oracle's, round 299.
    diff = t_interval((losses["ncf"] - losses["oracle"]).dropna())
    out["alg_loss_diff_299"], out["alg_loss_diff_299_lo"] = diff["mean"], diff["lo"]
    out["alg_loss_diff_299_hi"] = diff["hi"]
    rnd = conf[(conf.arm == "random") & (conf["round"] >= 250)]
    out["gini_random_tail"] = float(rnd["gini"].mean())
    oracle_mean = conf[conf.arm == "oracle"].groupby("round")["gini"].mean().sort_index()
    dev = (oracle_mean - oracle_mean.iloc[-1]).abs().to_numpy()
    out["oracle_settle_round"] = float(first_stays_below(dev, 1e-4))

    # Supplementary note S2: seeds 42-46 against the twenty exploratory seeds.
    expl = data[data.seed.isin(ah.SETS["exploratory"])]
    g = ah.gaps(expl)
    gap99 = g[g["round"] == 99].set_index("seed")["gap"].sort_index()
    loss99 = expl[(expl.arm == "ncf") & (expl["round"] == 99)].set_index("seed")[
        "algorithmic_loss"].sort_index()
    for name, s in (("gap", gap99), ("loss", loss99)):
        vals = s.to_numpy()
        blocks = [float(np.std(vals[i:i + 5], ddof=1)) for i in range(len(vals) - 4)]
        out[f"sd_five_{name}"] = blocks[0]
        out[f"sd_twenty_{name}"] = float(np.std(vals, ddof=1))
        out[f"rank_five_{name}"] = float(1 + sum(b < blocks[0] for b in blocks))
        out[f"n_blocks_{name}"] = float(len(blocks))

    s = pd.read_csv(H / "e09_canonical" / "analysis" / "summary.csv").set_index("set")
    c = s.loc["confirmation"]
    out["bca_agree"] = float(max(abs(c["gap_99_bca_lo"] - c["gap_99_lo95"]),
                                 abs(c["gap_99_bca_hi"] - c["gap_99_hi95"])))
    per = pd.read_csv(H / "e09_canonical" / "analysis" / "per_seed.csv")
    out["entry_min_all"] = float(per["entry_round"].min())
    out["entry_max_all"] = float(per["entry_round"].max())
    out["corr_min"] = float(per["corr_gap_spearman"].min())
    out["corr_median_all"] = float(per["corr_gap_spearman"].median())
    out["reach_ncf_0_maxdev_all"] = float((per["reach_ncf_0"] - 0.04).abs().max())
    out["gini_ncf_0_maxdev_all"] = float((per["gini_ncf_0"] - 0.96).abs().max())
    out["item_share_0_min_all"] = float(per["item_share_0"].min())
    out["modal_share_0_min_all"] = float(per["modal_share_0"].min())
    # R2-E9 (v): seeds 42-46 against the earlier one-thread 300-round run.
    archive = ah.ROOT / "data" / "processed" / "rigor" / "long_horizon_1t"
    worst = 0.0
    for seed in range(42, 47):
        new = data[data.seed == seed]
        for arm in ("ncf", "oracle", "random", "popularity"):
            old = pd.read_csv(archive / f"{arm}_seed{seed}.csv").set_index("epoch")
            a = new[new.arm == arm].set_index("round")
            for col in ("gini", "reach", "mcr", "rsi", "top1_share", "welfare_alg"):
                worst = max(worst, float((a[col] - old[col]).abs().max()))
    out["repro_42_46_max_abs_diff"] = worst
    # R2-E12 (iv): rank correlation of round and the seed-mean Jaccard, rounds 6-299.
    ser = pd.read_csv(H / "e09_canonical" / "analysis" / "series.csv")
    for set_name in ("confirmation", "exploratory"):
        x = ser[(ser.set == set_name) & (ser["round"] >= 6)]
        out[f"jaccard_rank_corr_{set_name}"] = float(
            x["round"].rank().corr(x["jaccard_oracle"].rank()))
    write_long(out, H / "e09_canonical" / "analysis" / "extras.csv")
    return out


def e11_per_seed() -> pd.DataFrame:
    """Entry round, T(0.05) in three units, oracle Gini and reach, per cell and seed."""
    base = ah._e09()
    cells = {"shipped": base[base.seed.isin(range(100, 110))]}
    for d in sorted((H / "e11_sweep").iterdir()):
        if d.is_dir() and list(d.glob("seed*.json")):
            cells[d.name] = ah.load(d)
    rows = []
    for cell, data in cells.items():
        g = ah.gaps(data)
        for seed, s in g.groupby("seed"):
            gap = s.sort_values("round")["gap"].to_numpy()
            ncf = data[(data.seed == seed) & (data.arm == "ncf")].sort_values("round")
            orc = data[(data.seed == seed) & (data.arm == "oracle")].sort_values("round")
            entry = entry_round(gap)
            rows.append({"cell": cell, "seed": seed,
                         "entry_round": np.nan if entry is None else entry,
                         "censored": entry is None, **t05_units(gap, ncf),
                         "gini_oracle_299": float(orc["gini"].iloc[-1]),
                         "reach_oracle_299": float(orc["reach"].iloc[-1])})
    per = pd.DataFrame(rows)
    ah.write(per, H / "e11_sweep" / "analysis", "per_seed.csv")
    return per


def e13b_check() -> pd.DataFrame:
    """R2-E13b-check: exploring head against a non-exploring head of the same length."""
    base = ah._e09()
    rows = []
    for share, prefix in (("0.05", 19), ("0.2", 16)):
        explore = ah.load(H / "e13b_explore" / f"e{share}")
        control = ah.load(H / "e13b_explore" / f"control_p{prefix}")
        e = explore[explore.arm == "ncf"][["seed", "round", "gini_prefix"]]
        c = control[control.arm == "ncf"][["seed", "round", "gini_prefix", "gini"]]
        m = e.merge(c, on=["seed", "round"], suffixes=("_explore", "_control"))
        # The non-exploring control is the shipped ranker; its whole-list Gini
        # must equal R2-E9's on the same seeds.
        ref = base[(base.arm == "ncf")][["seed", "round", "gini"]].rename(
            columns={"gini": "gini_e9"})
        repro = float((m.merge(ref, on=["seed", "round"])
                       .eval("gini - gini_e9").abs().max()))
        m["d"] = m["gini_prefix_explore"] - m["gini_prefix_control"]
        for r in ah.CHECK_ROUNDS:
            ti = t_interval(m[m["round"] == r]["d"])
            rows.append({"share": float(share), "prefix": prefix, "round": r, "n": ti["n"],
                         "mean": ti["mean"], "lo_t95": ti["lo"], "hi_t95": ti["hi"],
                         "control_repro_max_abs_diff": repro})
    out = pd.DataFrame(rows)
    ah.write(out, H / "e13b_explore" / "analysis", "check.csv")
    return out


def e14_trend() -> pd.DataFrame:
    """R2-E14-check: per seed, the least-squares slope over rounds 400-599, times 200."""
    rows = []
    for d in sorted((H / "e14_arrivals").iterdir()):
        if not (d.is_dir() and list(d.glob("seed*.json"))):
            continue
        g = ah.gaps(ah.load(d))
        wide = g.pivot(index="round", columns="seed", values="gap").sort_index().loc[400:599]
        x = wide.index.to_numpy(dtype=float)
        change = [200.0 * float(np.polyfit(x, wide[s].to_numpy(), 1)[0]) for s in wide.columns]
        ti, tt = t_interval(change), t_test(change)
        rows.append({"rate": float(d.name[1:]), "n": ti["n"], "change_200": ti["mean"],
                     "lo_t95": ti["lo"], "hi_t95": ti["hi"], "p": tt["p"],
                     "inside_margin": bool(ti["lo"] > -MARGIN and ti["hi"] < MARGIN)})
    out = pd.DataFrame(rows)
    ah.write(out, H / "e14_arrivals" / "analysis", "trend.csv")
    return out


def e15s_cells() -> pd.DataFrame | None:
    """R2-E15s: the five-round grid, same analysis as R2-E15."""
    d = H / "e15s_five"
    if not d.is_dir() or not list(d.glob("seed*.json")):
        print("e15s_five: no runs yet; skipped")
        return None
    data = pd.concat([pd.read_csv(p) for p in sorted(d.glob("seed*.csv"))])
    cells = []
    for (c, sf), cell in data.groupby(["signal_cost", "size_fraction"]):
        ti, tt = t_interval(cell["welfare_diff_residual"]), t_test(cell["welfare_diff_residual"])
        cells.append({"signal_cost": c, "size_fraction": sf, "n": ti["n"], "mean": ti["mean"],
                      "lo_t95": ti["lo"], "hi_t95": ti["hi"], "p": tt["p"]})
    frame = pd.DataFrame(cells)
    frame["q_bh"] = bh_qvalues(frame["p"])
    frame["b_beats_a"] = (frame["q_bh"] <= 0.05) & (frame["mean"] > 0)
    frame["a_beats_b"] = (frame["q_bh"] <= 0.05) & (frame["mean"] < 0)
    ah.write(frame, d / "analysis", "cells.csv")
    return frame


def _units_row(label: str, condition: str, per: pd.DataFrame) -> dict[str, float]:
    ok = per[~per["T05_censored"]]
    return {"experiment": label, "condition": condition, "n_seeds": len(per),
            "T05_rounds": float(ok["T05"].mean()) if len(ok) else np.nan,
            "T05_rows": float(ok["rows_at_T05"].mean()) if len(ok) else np.nan,
            "T05_steps": float(ok["steps_at_T05"].mean()) if len(ok) else np.nan,
            "n_censored": int(per["T05_censored"].sum())}


def units(e11: pd.DataFrame) -> pd.DataFrame:
    """R2-E18: T(0.05) in rounds, rows of log per agent and optimizer steps."""
    base = ah._e09()
    oracle = base[base.arm == "oracle"]
    rows = [_units_row("R2-E9", "shipped, confirmation",
                       per_seed_t05(base[base.seed.isin(ah.SETS["confirmation"])])),
            _units_row("R2-E9", "shipped, exploratory",
                       per_seed_t05(base[base.seed.isin(ah.SETS["exploratory"])]))]
    for cell, grp in e11.groupby("cell", sort=False):
        if cell != "shipped":
            rows.append(_units_row("R2-E11", cell, grp))
    for d in sorted((H / "e13a_dose").iterdir()):
        if d.is_dir() and list(d.glob("seed*.json")):
            rows.append(_units_row("R2-E13a", f"{float(d.name[1:]):g} passes per round",
                                   per_seed_t05(ah.load(d), ref=oracle)))
    shadow = ah.load(H / "e13c_offpolicy")
    o9 = oracle.set_index(["seed", "round"])["gini"]
    for src in ("oracle", "random"):
        s = shadow[shadow.arm == src]
        per = []
        for seed, grp in s.groupby("seed"):
            grp = grp.sort_values("round")
            gap = grp["shadow_gini"].to_numpy() - o9.loc[seed].reindex(grp["round"]).to_numpy()
            r = first_stays_below(gap, 0.05)
            t = len(gap) if r is None else r + 1
            per.append({"T05": float(t), "T05_censored": r is None, "rows_at_T05": np.nan,
                        "steps_at_T05": float(grp["shadow_grad_steps"].to_numpy()[t - 1])})
        rows.append(_units_row("R2-E13c", f"off-policy, {src} market's data", pd.DataFrame(per)))
    warm = ah.load(H / "e13d_warm")
    for set_name, seeds in ah.SETS.items():
        rows.append(_units_row("R2-E13d", f"warm start, {set_name}",
                               per_seed_t05(warm[warm.seed.isin(seeds)], ref=oracle)))
    sizes = [d for d in (H / "e17_size").iterdir() if d.is_dir() and list(d.glob("seed*.json"))]
    for d in sorted(sizes, key=lambda p: int(p.name[1:])):
        rows.append(_units_row("R2-E17", f"N = {int(d.name[1:])}", per_seed_t05(ah.load(d))))
    out = pd.DataFrame(rows)
    ah.write(out, H, "units.csv")
    return out


QUESTIONS = {
    "R2-E9": "the transient: four arms, 300 rounds",
    "R2-E10": "floor or slow decay: 1000 rounds",
    "R2-E11": "one-knob sweep, eight cells",
    "R2-E13a": "training dose: 0.5, 2, 4 passes per round",
    "R2-E13b": "exploration slots",
    "R2-E13b-check": "comparator check for exploration",
    "R2-E13c": "off-policy ranker",
    "R2-E13d": "warm start",
    "R2-E14": "open market: replacement",
    "R2-E15": "institutional grid, rounds 250-299",
    "R2-E15i": "institutional slice, inert gate",
    "R2-E15s": "institutional grid, rounds 0-4",
    "R2-E16": "floating-point sensitivity, ten runs per seed",
    "R2-E17": "population size 500 and 2000",
    "R2-H1r": "H1 at the shipped gate, round-0 log",
    "R2-H5r": "H5 at the shipped gate",
}
POST_HOC = {"R2-E13b-check", "R2-E15s"}


def run_index() -> pd.DataFrame:
    """One row per experiment, from the per-seed run records."""
    recs = []
    for p in H.rglob("*.json"):
        if p.name.startswith("seed") or p.name == "h1.json":
            r = json.loads(p.read_text(encoding="utf-8"))
            if "experiment" in r:
                recs.append(r)
    # One record can cover several seeds (the H1 curve is one job over ten seeds),
    # and a record with no horizon read only the round-0 log.
    df = pd.DataFrame([{
        "experiment": r["experiment"], "seeds": r.get("seeds") or [r["seed"]],
        "commit": r["git_commit"][:7], "dirty": r["git_dirty"],
        "config_hash": r["config_hash"][:10], "rounds": r.get("rounds", 1),
        "hours": r["wall_seconds"]["total"] / 3600.0,
    } for r in recs])
    rows = []
    for exp, grp in df.groupby("experiment"):
        seeds = sorted({s for lst in grp["seeds"] for s in lst})
        rows.append({"experiment": exp, "question": QUESTIONS.get(exp, ""),
                     "post_hoc": exp in POST_HOC, "n_records": len(grp),
                     "seeds": f"{seeds[0]}-{seeds[-1]}" if len(seeds) > 1 else str(seeds[0]),
                     "n_seeds": len(seeds),
                     "rounds": ",".join(str(int(x)) for x in sorted(set(grp["rounds"].dropna()))),
                     "commits": ",".join(sorted(set(grp["commit"]))),
                     "n_config_hashes": grp["config_hash"].nunique(),
                     "any_dirty": bool(grp["dirty"].any()),
                     "cpu_hours": float(grp["hours"].sum())})
    order = list(QUESTIONS)
    out = pd.DataFrame(rows)
    out["order"] = out["experiment"].map(lambda e: order.index(e) if e in order else 99)
    out = out.sort_values("order").drop(columns="order")
    ah.write(out, H, "run_index.csv")
    return out


READABLE = {"shock_scale_0.5": "chemistry 0.5", "shock_scale_3.0": "chemistry 3.0",
            "beta_alpha_1.0": "beta/alpha 1", "beta_alpha_3.0": "beta/alpha 3",
            "taste_sigma_0.2": "taste 0.2", "taste_sigma_1.0": "taste 1.0",
            "delta_0.0": "delta 0", "delta_1.0": "delta 1"}


def predictions(x9: dict[str, float]) -> pd.DataFrame:
    """Every registered prediction of R2-E9 to R2-E17, scored against its threshold."""
    A = lambda exp: H / exp / "analysis"  # noqa: E731
    s9 = pd.read_csv(A("e09_canonical") / "summary.csv").set_index("set")
    c11 = pd.read_csv(A("e11_sweep") / "cells.csv").set_index("cell")
    w = pd.read_csv(A("e13d_warm") / "summary.csv").set_index(["set", "quantity"])
    s10 = pd.read_csv(A("e10_long") / "summary.csv").iloc[0]
    s13a = pd.read_csv(A("e13a_dose") / "summary.csv").set_index("landmark")
    p13b = pd.read_csv(A("e13b_explore") / "paired.csv").set_index(["comparison", "round"])
    p13c = pd.read_csv(A("e13c_offpolicy") / "paired.csv")
    s15 = pd.read_csv(A("e15_institutional") / "summary.csv").set_index("a_arm")
    i15 = pd.read_csv(A("e15_institutional") / "inert_c001.csv").set_index("size_fraction")
    h1 = pd.read_csv(H / "h1r" / "analysis" / "curve.csv")
    h5 = pd.read_csv(H / "h5r" / "analysis" / "terms.csv").set_index("term")
    s14 = pd.read_csv(A("e14_arrivals") / "summary.csv").set_index("rate")
    s16 = pd.read_csv(A("e16_fp") / "summary.csv").iloc[0]
    sp16 = pd.read_csv(A("e16_fp") / "spread.csv").set_index("round")
    s17 = pd.read_csv(A("e17_size") / "summary.csv").set_index("N")
    sets = ("confirmation", "exploratory")
    rows = []

    def add(exp: str, text: str, measured: str, held: bool) -> None:
        rows.append({"experiment": exp, "prediction": text, "measured": measured,
                     "held": bool(held)})

    pk = [s9.loc[s, "peak_gap"] for s in sets]
    pr = [int(s9.loc[s, "peak_round"]) for s in sets]
    add("R2-E9", "peak gap in [0.27, 0.34] at a round in 3-10",
        f"{pk[0]:.3f} at {pr[0]}; {pk[1]:.3f} at {pr[1]}",
        all(0.27 <= p <= 0.34 for p in pk) and all(3 <= r <= 10 for r in pr))
    add("R2-E9", "tail equivalent to zero (TOST at 0.002), both sets",
        "passes both" if all(s9.loc[s, "tost_passes"] for s in sets) else "fails",
        all(s9.loc[s, "tost_passes"] for s in sets))
    ent = [20 - int(s9.loc[s, "n_censored"]) for s in sets]
    add("R2-E9", "at least 15 of 20 seeds enter the margin by round 299",
        f"{ent[0]} and {ent[1]} of 20", min(ent) >= 15)
    g99 = [s9.loc[s, "gap_99_mean"] for s in sets]
    add("R2-E9", "round-99 gap in [0.015, 0.035]", f"{g99[0]:.4f}; {g99[1]:.4f}",
        all(0.015 <= g <= 0.035 for g in g99))
    add("R2-E9", "seeds 42-46 reproduce the earlier one-thread run exactly",
        f"max difference {x9['repro_42_46_max_abs_diff']:.1g}",
        x9["repro_42_46_max_abs_diff"] == 0.0)
    add("R2-E12", "round-0 Gini within 0.002 of 0.96, all 40 seeds",
        f"max deviation {x9['gini_ncf_0_maxdev_all']:.1e}", x9["gini_ncf_0_maxdev_all"] <= 0.002)
    add("R2-E12", "round-0 reach within 0.002 of 0.04, all 40 seeds",
        f"max deviation {x9['reach_ncf_0_maxdev_all']:.3f}", x9["reach_ncf_0_maxdev_all"] <= 0.002)
    add("R2-E12", "round-0 modal share above 0.95 and item share above 0.9",
        f"min {x9['modal_share_0_min_all']:.3f} and {x9['item_share_0_min_all']:.4f}",
        x9["modal_share_0_min_all"] > 0.95 and x9["item_share_0_min_all"] > 0.9)
    n_neg = int(sum(s9.loc[s, "n_corr_below_08"] for s in sets))
    add("R2-E12", "gap-Spearman correlation below -0.8 on 36 of 40 seeds",
        f"{n_neg} of 40 (median {x9['corr_median_all']:+.2f})", n_neg >= 36)
    ib = [s9.loc[s, "identical_below_01_by"] for s in sets]
    add("R2-E12", "identical share below 0.1 by round 100", f"rounds {ib[0]:.0f}, {ib[1]:.0f}",
        max(ib) <= 100)
    it = [s9.loc[s, "item_below_05_by"] for s in sets]
    add("R2-E12", "item share below 0.5 by round 100", f"rounds {it[0]:.0f}, {it[1]:.0f}",
        max(it) <= 100)
    jr = [x9[f"jaccard_rank_corr_{s}"] for s in sets]
    add("R2-E12", "rank correlation of round and Jaccard above 0.9", f"{jr[0]:.3f}; {jr[1]:.3f}",
        min(jr) > 0.9)
    add("R2-E11", "C1t holds in all nine cells", f"{int(c11['C1t'].sum())} of 9",
        bool(c11["C1t"].all()))
    add("R2-E11", "C2t holds in at least seven of nine", f"{int(c11['C2t'].sum())} of 9",
        int(c11["C2t"].sum()) >= 7)
    fail3 = sorted(c11.index[~c11["C3t"]])
    add("R2-E11", "C3t fails at chemistry 0.5 and nowhere else",
        "fails at " + ", ".join(READABLE.get(c, c) for c in fail3), fail3 == ["shock_scale_0.5"])
    add("R2-E11", "C4t holds in all nine cells", f"{int(c11['C4t'].sum())} of 9",
        bool(c11["C4t"].all()))
    hit5 = sorted(c11.index[c11["C5t"]])
    add("R2-E11", "C5t (a reversal) at chemistry 0.5 and nowhere else",
        "at " + ", ".join(READABLE.get(c, c) for c in hit5), hit5 == ["shock_scale_0.5"])
    rmin = min(w.loc[(s, "reach_0_warm"), "min"] for s in sets)
    add("R2-E13d", "round-0 reach under the warm start above 0.8, every seed",
        f"minimum {rmin:.3f}", rmin > 0.8)
    ub = [w.loc[(s, "gap_round0_F299"), "hi_t95"] for s in sets]
    add("R2-E13d", "primary measure's upper bound below 0.5, both sets",
        f"{ub[0]:.3f}; {ub[1]:.3f}", max(ub) < 0.5)
    pb = [w.loc[(s, q), "mean"] for s in sets for q in ("gap_peak_F299", "level_peak_F299")]
    add("R2-E13d", "both peak-basis measures above 0.5, both sets", f"min {min(pb):.3f}",
        min(pb) > 0.5)
    add("R2-E10", "no positive floor", "none" if not s10["positive_floor"] else "floor",
        not bool(s10["positive_floor"]))
    add("R2-E10", "tail (rounds 950-999) equivalent to zero", "passes" if s10["tost_passes"]
        else "fails", bool(s10["tost_passes"]))
    el = s13a.loc["T05", "elasticity"]
    add("R2-E13a", "elasticity of T(0.05) in the dose at most -0.75", f"{el:.3f}", el <= -0.75)
    inside = [bool(p13b.loc[(f"exploit_minus_control_share{sh}", r), "inside_band"])
              for sh in ("0.05", "0.2") for r in ah.CHECK_ROUNDS]
    add("R2-E13b", "exploiting-part gap within the band of the control's",
        f"{sum(inside)} of {len(inside)} checks inside", all(inside))
    whole = p13b.loc[("whole_minus_control_share0.2", 20), "hi_t95"]
    add("R2-E13b", "whole-list gap below the control's at round 20, share 0.2",
        f"upper bound {whole:+.3f}", whole < 0)
    add("R2-E13c", "off-policy gap within the band of the on-policy gap",
        f"{int(p13c['inside_band'].sum())} of {len(p13c)} checks inside",
        bool(p13c["inside_band"].all()))
    no, nn = int(s15.loc["oracle", "n_b_beats_a"]), int(s15.loc["ncf", "n_b_beats_a"])
    add("R2-E15", "no cell has B beating the oracle A", f"{no} cells", no == 0)
    add("R2-E15", "at least as many cells beat NCF A as beat the oracle A", f"{nn} against {no}",
        nn >= no)
    rank = int(i15.loc[0.65, "rank"])
    add("R2-E15i", "size 0.65 ranks in the top three of ten, inert gate", f"rank {rank}",
        rank <= 3)
    mono = bool(h1["monotone_non_decreasing"].all())
    add("R2-H1r", "mean intent of proposers non-decreasing in the cost",
        "non-decreasing" if mono else "not", mono)
    lo, hi = h5.loc["interaction", "lo_t95"], h5.loc["interaction", "hi_t95"]
    add("R2-H5r", "interaction interval includes zero", f"[{lo:+.1f}, {hi:+.1f}]", lo < 0 < hi)
    gaps_ = [s14.loc[r, "steady_gap"] for r in (0.005, 0.01, 0.02)]
    add("R2-E14", "gap over rounds 500-599 increases with the rate",
        " < ".join(f"{g:.4f}" for g in gaps_), gaps_[0] < gaps_[1] < gaps_[2])
    add("R2-E14", "at rate 0.02 the interval excludes zero",
        f"lower bound {s14.loc[0.02, 'steady_lo95']:+.4f}", s14.loc[0.02, "steady_lo95"] > 0)
    newer = [s14.loc[r, "ratio_t000_009_ncf"] < s14.loc[r, "ratio_t000_009_oracle"]
             for r in (0.005, 0.01, 0.02)]
    add("R2-E14", "newest band under-exposed relative to the oracle at every rate",
        f"{sum(newer)} of 3 rates", all(newer))
    w0 = float(sp16.loc[0, "within_sd"])
    add("R2-E16", "within-seed spread at round 0 below 0.001", f"{w0:.1g}", w0 < 0.001)
    add("R2-E16", "within/between ratio reaches 0.3 in rounds 10-150",
        f"max {s16['max_ratio_10_150']:.2f}", s16["max_ratio_10_150"] >= 0.3)
    dev = float(s17["gini0_maxdev"].max())
    add("R2-E17", "round-0 Gini within 0.003 of 1 - 2k/N at each N", f"max {dev:.1e}",
        dev <= 0.003)
    pks = [s17.loc[n, "peak"] for n in (500, 1000, 2000)]
    add("R2-E17", "peak gap increases with N", " , ".join(f"{p:.3f}" for p in pks),
        pks[0] < pks[1] < pks[2])
    t5 = [s17.loc[n, "T05"] for n in (500, 2000)]
    add("R2-E17", "T(0.05) within a factor of two between N = 500 and 2000",
        f"{t5[0]:.0f} and {t5[1]:.0f} rounds", max(t5) / min(t5) <= 2)
    out = pd.DataFrame(rows)
    ah.write(out, H, "predictions.csv")
    return out


def parameters() -> pd.DataFrame:
    """Every model parameter: shipped value from base.yaml, swept values from the jobs.

    The swept values are those of the experiments' job lists (``scripts/make_jobs.py``)
    and registrations; they are repeated here as a list per parameter and checked
    against the job file's own sweep table where one exists.
    """
    import yaml  # noqa: PLC0415

    import make_jobs  # noqa: PLC0415

    with open(ROOT / "experiments" / "base.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    pop, hyp, ncf, sim = cfg["population"], cfg["hypergamy"], cfg["ncf"], cfg["simulation"]
    env, tr = cfg["environment_b"], pop["traits"]
    sweep = {k: v for k, v in make_jobs.SWEEP.items()}
    if set(sweep) != {"taste_sigma_0.2", "taste_sigma_1.0", "shock_scale_0.5", "shock_scale_3.0",
                      "beta_alpha_1.0", "beta_alpha_3.0", "delta_0.0", "delta_1.0"}:
        raise SystemExit("make_jobs.SWEEP no longer matches the parameter table")
    g = lambda d: f"{d['mean']:g}, {d['std']:g}"  # noqa: E731

    def num(v: float) -> str:
        """A power of ten as $10^{e}$, anything else in general format."""
        e = round(np.log10(v)) if v > 0 else 0
        return f"$10^{{{e}}}$" if v > 0 and abs(v - 10.0 ** e) < 1e-15 else f"{v:g}"
    rows = [
        ("market", "$N$", "agents", f"{pop['n']}", "500, 2000"),
        ("market", "$k$", "shortlist length", f"{ncf['top_k']}", "--"),
        ("market", r"$\phi$", "admissible share at round 0", f"{pop['admissible_fraction']:g}",
         "$20/N$ with $N$; 0.2 (inert gate)"),
        ("market", r"$\eta$", "threshold step", f"{sim['bayesian_eta']:g}", "--"),
        ("market", r"$\nu$", "replacement share per round", "0", "0.005, 0.01, 0.02"),
        ("utility", r"$\alpha$", "status weight", f"{hyp['alpha']:g}", "--"),
        ("utility", r"$\beta$", "attractiveness weight", f"{hyp['beta']:g}", "1, 3"),
        ("utility", r"$\gamma$", "distance penalty", f"{hyp['gamma']:g}", "--"),
        ("utility", r"$\lambda$", "status-surplus bonus", f"{hyp['lambda']:g}", "--"),
        ("utility", r"$\delta$", "status margin (sd)", f"{hyp['delta']:g}", "0, 1"),
        ("utility", r"$\sigma_L$", "chemistry scale", f"{hyp['shock_scale']:g}", "0.5, 3.0"),
        ("utility", r"$\sigma_\tau$", "taste spread (log-sd)", f"{pop['taste_sigma']:g}", "0.2, 1.0"),
        ("traits", "$s$", "status, lognormal (log-mean, log-sd)", g(tr["status"]), "--"),
        ("traits", "$a$", "attractiveness, normal, clipped", g(tr["attractiveness"]), "--"),
        ("traits", "$c$", "sociability, normal, clipped", g(tr["sociability"]), "--"),
        ("traits", "$t$", "age, normal", g(tr["age"]), "--"),
        ("traits", r"$\tau, g$", "intent, values: Beta",
         f"({pop['intent_beta_a']:g}, {pop['intent_beta_b']:g})", "--"),
        ("ranker", "--", "embedding dimension", f"{ncf['embed_dim']}", "--"),
        ("ranker", "--", "perceptron widths", ", ".join(map(str, ncf["mlp_layers"])), "--"),
        ("ranker", "--", "learning rate, weight decay", f"{num(ncf['lr'])}, {num(ncf['l2_reg'])}",
         "--"),
        ("ranker", "--", "positive weight, negatives per positive",
         f"{ncf['pos_weight']:g}, {ncf['n_negatives']}", "--"),
        ("ranker", r"$\rho$", "training passes per round", "1", "0.5, 2, 4"),
        ("ranker", "--", "exploration share", "0", "0.05, 0.2"),
        ("ranker", "--", "warm-start passes", "0", "50"),
        ("environment B", r"$\phi_B$", "subgraph share", f"{env['size_fraction']:g}",
         "0.02 to 1.00 (ten levels)"),
        ("environment B", "$c_B$", "proposal cost", f"{env['signal_cost']:g}",
         "0.01, 0.05, 0.10, 0.25"),
        ("environment B", r"$g_{\min}$", "values filter", f"{env['values_min']:g}", "--"),
        ("environment B", "$d$", "defection penalty", f"{env['defection_penalty']:g}", "--"),
        ("environment B", r"$\sigma_0, \sigma_\alpha, \sigma_\beta$", "survival",
         f"{env['sigma_0']:g}, {env['sigma_alpha']:g}, {env['sigma_beta']:g}", "--"),
        ("environment B", "$L$", "partner loss", f"{env['partner_loss']:g}", "--"),
    ]
    out = pd.DataFrame(rows, columns=["group", "symbol", "meaning", "shipped", "swept"])
    # "--" above is shorthand for "none"; the table prints words, not dashes.
    out["symbol"] = out["symbol"].replace("--", "")
    out["swept"] = out["swept"].replace("--", "not swept")
    ah.write(out, H, "parameters.csv")
    return out


REGISTRATIONS = {
    "main": "## R2-0: rules common to every R2 entry",
    "warm": "## E3d: the primary decay measure for the warm start",
    "check_b": "## R2-E13b-check: a post-hoc comparator check",
    "check_c": "## R2-E14-check: a trend test for the open market",
    "repair": "## R2-E16a: execution note",
}


def registrations() -> pd.DataFrame:
    """The commit at which each registration heading first entered the file.

    Read from git history with ``git log -S`` and written to a CSV, so a build
    from a shallow clone reads the hashes rather than needing the history.
    """
    import subprocess  # noqa: PLC0415

    rows = []
    for key, heading in REGISTRATIONS.items():
        res = subprocess.run(
            ["git", "log", "--format=%h %cs", "-S", heading, "--", "docs/PREREGISTRATION.md"],
            cwd=ah.ROOT, capture_output=True, text=True, check=True)
        lines = [ln for ln in res.stdout.splitlines() if ln.strip()]
        if not lines:
            raise SystemExit(f"registration {key!r} not found in git history: {heading}")
        commit, date = lines[-1].split()
        rows.append({"key": key, "heading": heading, "commit": commit, "date": date})
    out = pd.DataFrame(rows)
    ah.write(out, H, "registrations.csv")
    return out


DIRS = {"e09_canonical": ["R2-E9"], "e10_long": ["R2-E10"], "e11_sweep": ["R2-E11"],
        "e13a_dose": ["R2-E13a"], "e13b_explore": ["R2-E13b", "R2-E13b-check"],
        "e13c_offpolicy": ["R2-E13c"], "e13d_warm": ["R2-E13d"], "e14_arrivals": ["R2-E14"],
        "e15_institutional": ["R2-E15"], "e15i_inert": ["R2-E15i"], "e15s_five": ["R2-E15s"],
        "e16_fp": ["R2-E16"], "e17_size": ["R2-E17"], "h1r": ["R2-H1r"], "h5r": ["R2-H5r"]}


def readmes(index: pd.DataFrame) -> None:
    """A README per experiment directory, written from the run index."""
    idx = index.set_index("experiment")
    for d, labels in DIRS.items():
        path = H / d
        path.mkdir(parents=True, exist_ok=True)
        lines = [f"# horizon/{d}/", "",
                 "Written by `scripts/analyze_extras.py` from the run records in this "
                 "directory. The design of each experiment is its entry in "
                 "`docs/PREREGISTRATION.md`.", "",
                 "| registration | question | seeds | rounds | commits | CPU hours |",
                 "|---|---|---|---|---|---|"]
        for lab in labels:
            if lab in idx.index:
                r = idx.loc[lab]
                lines.append(f"| {lab}{' (post hoc)' if r['post_hoc'] else ''} | {r['question']} "
                             f"| {r['seeds']} ({r['n_seeds']}) | {r['rounds']} | {r['commits']} "
                             f"| {r['cpu_hours']:.1f} |")
            else:
                lines.append(f"| {lab} | {QUESTIONS.get(lab, '')} | not yet run | | | |")
        analysis = sorted(p.name for p in (path / "analysis").glob("*.csv")) \
            if (path / "analysis").is_dir() else []
        lines += ["", "Analysis outputs: " + (", ".join(f"`analysis/{a}`" for a in analysis)
                                             or "none yet") + ".", "",
                  "## Thread count", "",
                  "**1**: every run record here states `ncf.num_threads = 1`.", ""]
        (path / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {len(DIRS)} experiment READMEs")


def main() -> int:
    registrations()
    parameters()
    x9 = e09_extras()
    e11 = e11_per_seed()
    e13b_check()
    e14_trend()
    e15s_cells()
    units(e11)
    readmes(run_index())
    p = predictions(x9)
    print(f"{int(p['held'].sum())} of {len(p)} registered predictions held")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
