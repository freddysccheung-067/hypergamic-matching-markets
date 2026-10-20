"""Welfare accounting across the two environments (THEORY.md §10.4).

Every number here is built from one per-agent vector: the welfare each agent
contributes to the §10.4 sum in one epoch,

    payoff_i = tau_i * U_i(partner)            if matched
             - c * (proposals i made)
             - L * (1 - tau_defector)          if i's partner defected
             - d                               if i defected

and ``payoff.sum()`` is the epoch's ``welfare_total``. Working per agent rather
than per pair is what makes the comparability rule expressible at all: the rule
scores an *agent* at the outcome another market gave them, so the unit of account
has to be the agent.

The two accountings, for an Environment B run paired with an Environment A run on
the same population and seed:

``welfare_common``   every agent appears once. Agents matched in B are scored at
                     their B payoff; every other agent at their A payoff. An entrant
                     that proposed in B and failed has *also* spent c_B, and that
                     spend is charged on top of their A outcome (``sunk_signal_cost``).
                     §10.4 says only "scored at the outcome A would have given them";
                     dropping the spend would make a failed proposal in B free, and the
                     failed proposals are exactly the thin-market penalty this
                     comparison exists to price.
``welfare_selected`` entrants only, expressed in the same units as welfare_common:
                     the per-entrant mean scaled to N. A raw sum over entrants would be
                     smaller than welfare_common for the trivial reason that it counts
                     fewer agents; scaling to N makes it the number a reader gets by
                     treating B's entrants as representative of the market, which is
                     the mistake §10.4 warns against.
``welfare_selection_artifact``  selected - common.

**The artifact is not non-negative by construction.** It is N times the gap
between the per-entrant mean and the per-agent mean, and that gap is positive only
if B's entrants do better than the population does under the common rule. B charges
every proposer c_B per proposal, so at a high enough cost its entrants are *worse*
off than the agents it excluded, and the artifact goes negative. It is reported as
measured; ``tests/python/test_comparison.py`` records where its sign holds.

**The literal rule double-books.** An agent B did not match is scored at their
full-A outcome, and in full A their partner may be someone B *did* match. Both are
then counted: one at a B match, the other at an A match with a partner who is busy.
``n_double_booked`` counts these. ``welfare_common_residual`` is the feasible
version: A is re-run with B's matched agents struck from every list
(``residual_a_payoff``), and the agents B left behind are scored there. Both are
reported; they bracket the comparison, and the gap between them is the size of the
infeasibility in the literal rule.

Pairing is checked, not assumed. ``assert_comparable`` raises unless the two
outcomes come from the same seed and epoch and their state matrices agree in every
column except θ (column 6), which is the one column the feedback loop legitimately
moves differently in each environment.
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.python.config import EnvironmentBConfig
from src.python.environment_b import DefectionOutcome, resolve_defection

try:  # the extension lands in build/Release (README "Installation")
    from _hypergamic import utility_for_pairs
except ImportError:  # pragma: no cover - convenience for a fresh checkout
    _BUILD = pathlib.Path(__file__).resolve().parents[2] / "build" / "Release"
    if _BUILD.is_dir():
        sys.path.insert(0, str(_BUILD))
    from _hypergamic import utility_for_pairs  # noqa: F401

# Column indexes of the state matrix (docs/INTERFACES.md §1).
_THETA, _INTENT = 6, 11


@dataclass(frozen=True)
class EpochOutcome:
    """What one epoch of one environment gave each agent."""

    states: np.ndarray    # the matrix the epoch's matching saw (a copy)
    payoff: np.ndarray    # float64 (N,): per-agent §10.4 contribution
    matched: np.ndarray   # bool (N,)
    entrant: np.ndarray   # bool (N,): in the market at all; all True outside B
    seed: int
    epoch: int
    # What the matching was run on, kept so the A market can be re-run without the
    # agents B already matched (``residual_a_payoff``). Dense lists (INTERFACES.md
    # §2.3), the row-id proposal set, and the cost each proposal paid.
    prop_prefs: list[list[int]] | None = None
    acc_prefs: list[list[int]] | None = None
    proposals: np.ndarray | None = None
    cost: float = 0.0
    pairs: np.ndarray | None = None   # (P, 2) row ids of the matched pairs


@dataclass(frozen=True)
class WelfareAccounts:
    welfare_common: float
    welfare_selected: float
    welfare_selection_artifact: float
    sunk_signal_cost: float
    n_entrants: int
    # The feasible variant of welfare_common: agents B did not match are scored in
    # a re-run of A from which B's matched agents are removed. NaN when the outcome
    # carries no lists to re-run. See welfare_accounts.
    welfare_common_residual: float = float("nan")
    # Agents scored at an A outcome whose A partner was matched in B: the count of
    # A matches the literal rule books on one side while the other side is in B.
    n_double_booked: int = 0


def agent_payoffs(
    states: np.ndarray,
    config: dict,
    pairs: np.ndarray,
    proposals: np.ndarray,
    cost: float,
    env: EnvironmentBConfig,
    penalty: float,
    seed: int,
    epoch: int,
) -> tuple[np.ndarray, DefectionOutcome]:
    """Per-agent welfare for one epoch, and the defection draw it used.

    Args:
        pairs: (P, 2) *row* ids of matched pairs, [proposer, acceptor].
        proposals: (Q, 2) row ids, [proposer, acceptor]; each row costs its
            proposer ``cost``.
        penalty: d, charged to the defector of each dissolved pair. 0 in A.

    The value term reads ``utility_for_pairs``, the same ``utility_toward`` that
    ``compute_welfare`` sums, so the vector's sum and the C++ ``welfare_total``
    differ only by summation order.
    """
    n = states.shape[0]
    payoff = np.zeros(n, dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    tau = states[:, _INTENT]

    if pairs.shape[0]:
        fwd = np.asarray(utility_for_pairs(states, pairs.astype(np.int32), config))
        rev = np.asarray(utility_for_pairs(states, pairs[:, ::-1].astype(np.int32).copy(),
                                           config))
        # A matching is one partner per agent, so plain fancy-index assignment is
        # safe here; np.add.at would hide a duplicate rather than expose it.
        payoff[pairs[:, 0]] = tau[pairs[:, 0]] * fwd
        payoff[pairs[:, 1]] = tau[pairs[:, 1]] * rev

    proposals = np.asarray(proposals, dtype=np.int64).reshape(-1, 2)
    if cost > 0.0 and proposals.shape[0]:
        # One proposer makes many proposals, so this one does need add.at.
        np.add.at(payoff, proposals[:, 0], -float(cost))

    outcome = resolve_defection(states, pairs, env, penalty, seed, epoch)
    if outcome.n_dissolved:
        gone = pairs[~outcome.survived]
        defector = outcome.defector_rows
        partner = np.where(defector == gone[:, 0], gone[:, 1], gone[:, 0])
        payoff[partner] -= env.partner_loss * (1.0 - tau[defector])
        payoff[defector] -= float(penalty)
    return payoff, outcome


def assert_comparable(a: EpochOutcome, b: EpochOutcome) -> None:
    """Raise unless ``a`` and ``b`` describe the same agents at the same epoch.

    θ is excluded from the check and nothing else is: it is the one column each
    environment's feedback step moves on its own. A raise rather than an
    ``assert`` so the check survives ``python -O``: a comparison between two
    different populations is not a smaller error than a crash, it is a wrong
    number that looks right.
    """
    if a.seed != b.seed:
        raise ValueError(f"paired runs have different seeds ({a.seed} vs {b.seed})")
    if a.epoch != b.epoch:
        raise ValueError(f"paired outcomes are from different epochs ({a.epoch} vs {b.epoch})")
    if a.states.shape != b.states.shape:
        raise ValueError(
            f"paired runs have different populations (shapes {a.states.shape} vs "
            f"{b.states.shape})"
        )
    cols = np.arange(a.states.shape[1]) != _THETA
    if not np.array_equal(a.states[:, cols], b.states[:, cols]):
        raise ValueError(
            "paired runs do not share a population: the state matrices differ outside θ"
        )


def residual_a_payoff(
    a: EpochOutcome, removed: np.ndarray, config: dict, env: EnvironmentBConfig,
) -> np.ndarray:
    """A's per-agent payoff with the agents in ``removed`` taken out of A's market.

    A's own lists, with every removed agent struck from every list and given an
    empty list of their own, run through the same ``deferred_acceptance`` call. A
    proposal to a removed agent is never made, so it is never charged; a
    proposer's remaining proposals are. Defection is drawn from the same hash with
    A's d = 0, so a pair that also formed in the full A run gets the same draw.

    This is the feasible version of "the outcome A would have given them": the
    literal version reads the full A run, in which some of these agents are matched
    to partners who are in B at the same time.
    """
    from src.python.environment_b import deferred_acceptance  # noqa: PLC0415

    if a.prop_prefs is None or a.acc_prefs is None or a.proposals is None:
        raise ValueError("A outcome carries no lists; record it from a Simulation run")
    states = a.states
    types = states[:, 1]
    prop_rows = np.flatnonzero(types == 0.0)
    acc_rows = np.flatnonzero(types == 1.0)
    gone_p = removed[prop_rows]
    gone_a = removed[acc_rows]

    prop_prefs = [[] if gone_p[p] else [x for x in lst if not gone_a[x]]
                  for p, lst in enumerate(a.prop_prefs)]
    acc_prefs = [[] if gone_a[x] else [p for p in lst if not gone_p[p]]
                 for x, lst in enumerate(a.acc_prefs)]
    m = deferred_acceptance(prop_prefs, acc_prefs, True)["matches"]
    pairs = (np.stack([prop_rows[m[:, 0]], acc_rows[m[:, 1]]], axis=1)
             if m.shape[0] else np.zeros((0, 2), dtype=np.int64))
    props = a.proposals
    if props.shape[0]:
        props = props[~removed[props[:, 0]] & ~removed[props[:, 1]]]
    payoff, _ = agent_payoffs(states, config, pairs, props, a.cost, env, 0.0,
                              a.seed, a.epoch)
    return payoff


def welfare_accounts(
    b: EpochOutcome, a: EpochOutcome,
    config: dict | None = None, env: EnvironmentBConfig | None = None,
) -> WelfareAccounts:
    """The two accountings of THEORY.md §10.4 for one epoch (see module docstring).

    With ``config`` and ``env`` it also computes ``welfare_common_residual``, the
    same rule with A re-run on the agents B left behind.
    """
    assert_comparable(a, b)
    if np.any(b.payoff[~b.entrant] != 0.0) or np.any(b.matched & ~b.entrant):
        # An agent outside B cannot have a B outcome. If one does, the subgraph
        # mask and the matching disagree and both accountings would be wrong.
        raise ValueError("Environment B outcome assigns welfare to a non-entrant")

    n = b.payoff.size
    n_entrants = int(b.entrant.sum())
    unmatched = ~b.matched
    sunk = -float(b.payoff[b.entrant & unmatched].sum())
    # Σ_i b_i + Σ_{unmatched in B} a_i: matched agents at B, everyone else at A,
    # plus whatever an unmatched entrant spent in B (b_i for them is -spend).
    common = float(b.payoff.sum() + a.payoff[unmatched].sum())
    selected = (float(b.payoff[b.entrant].sum()) * n / n_entrants
                if n_entrants else float("nan"))

    residual = float("nan")
    n_double = 0
    if a.pairs is not None and a.pairs.shape[0]:
        partner = np.full(n, -1, dtype=np.int64)
        partner[a.pairs[:, 0]] = a.pairs[:, 1]
        partner[a.pairs[:, 1]] = a.pairs[:, 0]
        has = unmatched & (partner >= 0)
        n_double = int(np.count_nonzero(b.matched[partner[has]]))
    if config is not None and env is not None and a.prop_prefs is not None:
        res = residual_a_payoff(a, b.matched, config, env)
        residual = float(b.payoff.sum() + res[unmatched].sum())
    return WelfareAccounts(
        welfare_common=common,
        welfare_selected=selected,
        welfare_selection_artifact=selected - common,
        sunk_signal_cost=sunk,
        n_entrants=n_entrants,
        welfare_common_residual=residual,
        n_double_booked=n_double,
    )


def own_market_accounts(a: EpochOutcome) -> WelfareAccounts:
    """Accounts for an environment that excludes nobody.

    Every agent is an entrant scored in its own market, so the two accountings
    coincide and the artifact is exactly zero — by definition, not by measurement.
    """
    total = float(a.payoff.sum())
    return WelfareAccounts(total, total, 0.0, 0.0, int(a.payoff.size))


# ── Paired runs ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PairedRun:
    frame_a: pd.DataFrame
    frame_b: pd.DataFrame
    sim_a: Any
    sim_b: Any
    # Per epoch. Not frame columns: §6 is the frame every arm emits, and these exist
    # only for a pairing.
    welfare_common_residual: np.ndarray
    n_double_booked: np.ndarray


def paired_run(config: dict, n_epochs: int, arm_a: str = "oracle") -> PairedRun:
    """Run Environment A and Environment B on one population and fill the accounts.

    Both ``Simulation`` objects are built from the same config, so they sample the
    same population from the same seed; ``assert_comparable`` then checks that they
    did, at every epoch, rather than trusting the construction. A is ``arm_a``
    (the oracle by default: the comparison is between institutions, and the oracle
    is A's ranking with nothing estimated, so B's advantage is not inflated by the
    NCF's estimation error). B is the ``"complete"`` arm.

    On return, each frame's ``welfare_common`` / ``welfare_selected`` /
    ``welfare_selection_artifact`` columns are filled: B's from the pairing, A's
    from ``own_market_accounts`` (A excludes nobody).
    """
    from src.python.orchestrator import Simulation  # noqa: PLC0415 - import cycle

    sim_a = Simulation(config=config, recommender=arm_a)
    sim_b = Simulation(config=config, recommender="complete")
    if not np.array_equal(sim_a.states, sim_b.states):
        raise ValueError("A and B sampled different populations from one config")
    for sim in (sim_a, sim_b):
        sim.record_outcomes = True
    frame_a = sim_a.run(n_epochs)
    frame_b = sim_b.run(n_epochs)
    env = sim_b.config.require_environment_b()

    residual: list[float] = []
    double: list[int] = []
    for t, (oa, ob) in enumerate(zip(sim_a.outcomes, sim_b.outcomes)):
        acc_b = welfare_accounts(ob, oa, sim_a.config_dict, env)
        residual.append(acc_b.welfare_common_residual)
        double.append(acc_b.n_double_booked)
        acc_a = own_market_accounts(oa)
        for frame, acc in ((frame_a, acc_a), (frame_b, acc_b)):
            frame.loc[t, "welfare_common"] = acc.welfare_common
            frame.loc[t, "welfare_selected"] = acc.welfare_selected
            frame.loc[t, "welfare_selection_artifact"] = acc.welfare_selection_artifact
    return PairedRun(frame_a, frame_b, sim_a, sim_b,
                     welfare_common_residual=np.asarray(residual, dtype=np.float64),
                     n_double_booked=np.asarray(double, dtype=np.int64))
