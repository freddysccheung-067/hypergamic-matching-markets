"""Measurements behind THEORY.md §2.3 and §7.2.

Oracle shortlists (top-k by true utility, gated by θ), so the numbers describe the
market rather than the NCF's approximation of it. Run with the extension built:

    python -I scripts/_heterogeneity_report.py
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import yaml

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "build" / "Release"))

from _hypergamic import (  # noqa: E402
    deferred_acceptance,
    generate_swipes,
    sample_population,
    utility_sd,
)


def base_config(**overrides):
    with open(_ROOT / "experiments" / "base.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for section, values in overrides.items():
        cfg[section].update(values)
    return cfg


def gini(counts: np.ndarray) -> float:
    x = np.sort(np.asarray(counts, dtype=np.float64))
    n = x.size
    total = x.sum()
    if total <= 0:
        return 0.0
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float(2.0 * np.dot(idx, x) / (n * total) - (n + 1.0) / n)


def concentration(cfg, n=1000, k=20, seed=42):
    """Reach and Gini of the oracle shortlists (positives of generate_swipes)."""
    states = sample_population(n, cfg, seed)
    log = generate_swipes(states, cfg, 0)
    pos = log[log[:, 2] == 1]
    exposure = np.bincount(pos[:, 1], minlength=n).astype(np.float64)

    types = states[:, 1]
    men, women = types == 0.0, types == 1.0
    return {
        "reach_all": float((exposure > 0).mean()),
        "reach_men": float((exposure[men] > 0).mean()),
        "reach_women": float((exposure[women] > 0).mean()),
        "gini_all": gini(exposure),
        "gini_men": gini(exposure[men]),
        "gini_women": gini(exposure[women]),
        "mean_exposure": float(exposure.mean()),
        "max_exposure": float(exposure.max()),
    }


def truncated_lists(states, k):
    """Oracle lists: top-k among θ-admissible candidates, per side (dense slots)."""
    log = generate_swipes(states, _CONFIG, 0)
    types = states[:, 1]
    prop_rows = np.flatnonzero(types == 0.0)
    acc_rows = np.flatnonzero(types == 1.0)
    prop_slot = {int(r): i for i, r in enumerate(prop_rows)}
    acc_slot = {int(r): i for i, r in enumerate(acc_rows)}

    prop_prefs = [[] for _ in prop_rows]
    acc_prefs = [[] for _ in acc_rows]
    # Positives arrive in descending utility per user, so truncating to k keeps the
    # best k — exactly the oracle shortlist.
    for user, item, label in log[log[:, 2] == 1]:
        if type(user) is np.int32:
            user, item = int(user), int(item)
        if user in prop_slot:
            prop_prefs[prop_slot[user]].append(acc_slot[item])
        else:
            acc_prefs[acc_slot[user]].append(prop_slot[item])
    return prop_prefs, acc_prefs


def mcr(cfg, n, k, seed=42):
    states = sample_population(n, cfg, seed)
    global _CONFIG
    _CONFIG = {**cfg, "ncf": {**cfg.get("ncf", {}), "top_k": k}}
    prop_prefs, acc_prefs = truncated_lists(states, k)
    da = deferred_acceptance(prop_prefs, acc_prefs, True)
    return 2.0 * da["matches"].shape[0] / n


_CONFIG: dict = {}


def main() -> None:
    print("== noise-to-signal ==")
    for sigma in (0.0, 0.5):
        cfg = base_config(population={"taste_sigma": sigma})
        states = sample_population(1000, cfg, 42)
        sd = utility_sd(states, cfg)
        print(f"  taste_sigma={sigma:.1f}  utility_sd={sd:.4f}  "
              f"ratio@1.5={1.5 * 1.8138 / sd:.3f}")

    print("== concentration, N=1000 k=20 ==")
    for label, over in (
        ("both off ", dict(population={"taste_sigma": 0.0}, hypergamy={"shock_scale": 0.0})),
        ("base     ", dict()),
    ):
        r = concentration(base_config(**over))
        print(f"  {label} reach={r['reach_all']:.3f} gini={r['gini_all']:.3f} "
              f"(men {r['reach_men']:.3f}/{r['gini_men']:.3f}, "
              f"women {r['reach_women']:.3f}/{r['gini_women']:.3f}) "
              f"mean={r['mean_exposure']:.1f} max={r['max_exposure']:.0f}")

    print("== MCR vs k, N=2000, base config ==")
    for k in (5, 10, 20, 40, 80):
        print(f"  k={k:3d}  mcr={mcr(base_config(), 2000, k):.3f}")

    print("== MCR vs N, k=20 ==")
    for n in (500, 1000, 2000, 4000):
        print(f"  N={n:5d}  mcr={mcr(base_config(), n, 20):.3f}")

    print("== MCR with k ~ 0.9*sqrt(N/2) ==")
    for n in (500, 1000, 2000, 4000):
        k = max(1, round(0.9 * (n / 2) ** 0.5))
        print(f"  N={n:5d}  k={k:3d}  mcr={mcr(base_config(), n, k):.3f}")

    print("== generate_swipes timing, N=2000 ==")
    for label, over in (
        ("both off", dict(population={"taste_sigma": 0.0}, hypergamy={"shock_scale": 0.0})),
        ("base    ", dict()),
    ):
        cfg = base_config(**over)
        states = sample_population(2000, cfg, 42)
        generate_swipes(states, cfg, 0)
        ts = []
        for e in range(5):
            t0 = time.perf_counter()
            generate_swipes(states, cfg, e)
            ts.append(time.perf_counter() - t0)
        print(f"  {label}: median {sorted(ts)[2] * 1000:.2f} ms")


if __name__ == "__main__":
    main()
