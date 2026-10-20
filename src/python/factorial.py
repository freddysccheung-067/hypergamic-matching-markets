"""The factorial decomposition of THEORY.md §10.6.

Two switches, each a full run:

    drift       simulation.status_drift = 0 (off) or gamma (on)
    mediation   recommender = "oracle" (off) or "ncf" (on)

"Mediation off" is the oracle arm, not a market without lists: it ranks by true
utility with the same k, the same θ gate and the same deferred-acceptance call, so
the only thing the switch changes is whether the ranking is *estimated*. A
no-truncation market would change k as well and the algorithmic term would mix
the two.

    dW_exog        = W2 - W1
    dW_alg         = W3 - W1
    dW_interaction = W4 - W3 - W2 + W1

Each W is the §10.4 ``welfare_total`` averaged over the last ``window`` epochs of
a run, so a term is a level difference at the end of the drift, not a slope. The
terms are computed per seed and summarised over seeds, never formed from cell
means: the four cells at one seed share a population, and their difference at that
seed is one observation of the effect (the same pairing argument as
scripts/run_ablation.py).

Nothing here asserts a sign. The interaction is the one term that could support
"the algorithm amplifies the demographic shift", and THESIS.md H5 says in advance
that it may be zero.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

CELLS: tuple[tuple[str, bool, str], ...] = (
    ("baseline", False, "oracle"),
    ("drift", True, "oracle"),
    ("ncf", False, "ncf"),
    ("both", True, "ncf"),
)


@dataclass(frozen=True)
class FactorialResult:
    seed: int
    gamma: float
    cells: dict[str, float]           # cell name -> W over the window
    exogenous: float
    algorithmic: float
    interaction: float
    epoch0_states: dict[str, np.ndarray]


def run_factorial(config: dict[str, Any], gamma: float, n_epochs: int,
                  window: int = 10) -> FactorialResult:
    """Run the four cells on one population and decompose.

    Raises if the four cells did not start from the same population and seed.
    That is the only assertion: it is what makes the four numbers a factorial
    rather than four experiments, and it is checked on the matrices each cell's
    first epoch actually saw rather than inferred from the config.
    """
    from src.python.orchestrator import Simulation  # noqa: PLC0415 - import cycle

    if not 0 < window <= n_epochs:
        raise ValueError(f"window must be in [1, n_epochs]; got {window} of {n_epochs}")
    cells: dict[str, float] = {}
    first: dict[str, np.ndarray] = {}
    seeds: set[int] = set()
    for name, drift, arm in CELLS:
        cfg = copy.deepcopy(config)
        cfg.setdefault("simulation", {})["status_drift"] = float(gamma) if drift else 0.0
        sim = Simulation(config=cfg, recommender=arm)
        sim.record_states = True
        frame = sim.run(n_epochs)
        cells[name] = float(frame["welfare_total"].iloc[-window:].mean())
        first[name] = sim.states_seen[0]
        seeds.add(sim.rng_seed)

    if len(seeds) != 1:
        raise ValueError(f"factorial cells ran on different seeds: {sorted(seeds)}")
    ref = first["baseline"]
    for name, states in first.items():
        if not np.array_equal(ref, states):
            raise ValueError(f"factorial cell {name!r} did not start from the baseline population")

    w1, w2, w3, w4 = (cells[c[0]] for c in CELLS)
    return FactorialResult(
        seed=seeds.pop(), gamma=float(gamma), cells=cells,
        exogenous=w2 - w1, algorithmic=w3 - w1, interaction=w4 - w3 - w2 + w1,
        epoch0_states=first,
    )
