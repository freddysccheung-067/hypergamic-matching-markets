"""Per-round diagnostics of what the learned ranker knows (R2-E12).

Nothing here intervenes. Each function reads quantities the run already holds
(the ranker's score matrix, the true utility table, the shortlists the ranker
and the oracle would hand out) and returns numbers; none draws a random number
or touches a weight, so a run with diagnostics on is bit-identical to one with
them off, which ``tests/python/test_diagnostics.py`` checks.

The four quantities, each per side and pooled over both:

``spearman``        rank correlation between an agent's NCF scores and its true
                    utilities over every candidate on the other side. 1 is a
                    ranker that orders candidates exactly as the agent does.
``modal_share``     the share of an agent's shortlist that is the side-wide modal
                    list: the k candidates appearing on the most shortlists of
                    that side. ``identical_share`` is the fraction of agents whose
                    list is exactly that set. At round 0 the ranker's scores are
                    almost all a shared per-candidate term, so every agent on a
                    side sees nearly the same k candidates.
``item_share``      the share of within-agent score variance that an additive
                    per-candidate term explains: R^2 of s_ij = a_i + b_j + e_ij
                    on the variation that ranking acts on, which is the
                    variation within a row.
``jaccard_oracle``  overlap of an agent's NCF shortlist with the shortlist the
                    oracle would give it under the same θ, |A∩B| / |A∪B|.
"""

from __future__ import annotations

import numpy as np


def row_spearman(scores: np.ndarray, utility: np.ndarray) -> np.ndarray:
    """Spearman correlation of each row of ``scores`` with the same row of ``utility``.

    Ranks by a double argsort. Ties have measure zero for continuous scores and
    utilities, so average ranks would change nothing and are not computed.
    """
    if scores.shape != utility.shape:
        raise ValueError(f"shape mismatch {scores.shape} vs {utility.shape}")
    rs = np.argsort(np.argsort(scores, axis=1), axis=1).astype(np.float64)
    ru = np.argsort(np.argsort(utility, axis=1), axis=1).astype(np.float64)
    rs -= rs.mean(axis=1, keepdims=True)
    ru -= ru.mean(axis=1, keepdims=True)
    denom = np.sqrt((rs * rs).sum(axis=1) * (ru * ru).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rs * ru).sum(axis=1) / denom


def item_share(scores: np.ndarray) -> float:
    """R^2 of an additive per-column term on the within-row variation of ``scores``."""
    within = scores - scores.mean(axis=1, keepdims=True)
    col = within.mean(axis=0, keepdims=True)
    ss_within = float((within * within).sum())
    if ss_within <= 0.0:
        return float("nan")
    resid = within - col
    return 1.0 - float((resid * resid).sum()) / ss_within


def modal_list(lists: list[list[int]], n_candidates: int, k: int) -> np.ndarray:
    """The k candidates appearing on the most lists; ties to the lower index."""
    counts = np.zeros(n_candidates, dtype=np.int64)
    for lst in lists:
        if lst:
            counts[np.asarray(lst, dtype=np.int64)] += 1
    order = np.lexsort((np.arange(n_candidates), -counts))
    return np.sort(order[:k])


def modal_overlap(lists: list[list[int]], n_candidates: int, k: int) -> tuple[float, float]:
    """(mean share of each list inside the modal list, share of lists equal to it)."""
    modal = set(modal_list(lists, n_candidates, k).tolist())
    shares, identical = [], 0
    for lst in lists:
        s = set(lst)
        shares.append(len(s & modal) / k)
        identical += int(s == modal)
    return float(np.mean(shares)), identical / len(lists)


def mean_jaccard(a: list[list[int]], b: list[list[int]]) -> float:
    vals = []
    for x, y in zip(a, b):
        sx, sy = set(x), set(y)
        union = sx | sy
        vals.append(len(sx & sy) / len(union) if union else 1.0)
    return float(np.mean(vals))


def round_diagnostics(
    scores: tuple[np.ndarray, np.ndarray],
    utility: tuple[np.ndarray, np.ndarray],
    ncf_lists: tuple[list[list[int]], list[list[int]]],
    oracle_lists: tuple[list[list[int]], list[list[int]]],
    k: int,
) -> dict[str, float]:
    """Every diagnostic for one round, as flat named numbers.

    ``scores[0]`` is (proposers x acceptors), ``scores[1]`` (acceptors x
    proposers), both in dense slot order, and ``utility`` has the same layout.
    The lists are dense slots on the other side.
    """
    out: dict[str, float] = {}
    rho_all = []
    for side, s, u, nl, ol in (("m", scores[0], utility[0], ncf_lists[0], oracle_lists[0]),
                               ("w", scores[1], utility[1], ncf_lists[1], oracle_lists[1])):
        rho = row_spearman(s, u)
        rho_all.append(rho)
        share, identical = modal_overlap(nl, s.shape[1], k)
        out[f"spearman_mean_{side}"] = float(np.nanmean(rho))
        out[f"item_share_{side}"] = item_share(s)
        out[f"modal_share_{side}"] = share
        out[f"identical_share_{side}"] = identical
        out[f"jaccard_oracle_{side}"] = mean_jaccard(nl, ol)
    rho = np.concatenate(rho_all)
    out["spearman_mean"] = float(np.nanmean(rho))
    for q in (10, 50, 90):
        out[f"spearman_q{q}"] = float(np.nanpercentile(rho, q))
    for name in ("item_share", "modal_share", "identical_share", "jaccard_oracle"):
        out[name] = 0.5 * (out[f"{name}_m"] + out[f"{name}_w"])
    return out
