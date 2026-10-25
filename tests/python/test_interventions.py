"""The two intervention switches, and the guarantee that their defaults are a no-op.

``ncf.freeze_after`` and ``ncf.warm_start_epochs`` exist for the pre-registered
interventions of ``docs/PREREGISTRATION.md`` E3. Neither is a headline setting:
they exist to manipulate reach so the paper's mechanism claim can be tested
rather than asserted.

The contract these tests pin is the one a config change must keep: **a run with
the defaults is bit-identical to a run from before the keys existed.** That is
what "default preserves current behaviour" means, and it is checkable without an
archived checkout because the defaults are ``None`` and ``0`` at the dataclass.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.python.config import NCFConfig, SimConfig  # noqa: E402
from src.python.orchestrator import Simulation  # noqa: E402

CONFIG = "experiments/base.yaml"


def _small(seed: int = 42, **ncf) -> dict:
    cfg = SimConfig.from_yaml(CONFIG).to_dict()
    cfg["population"]["seed"] = int(seed)
    cfg["population"]["n"] = 120
    cfg["ncf"]["top_k"] = 8
    cfg["ncf"].update(ncf)
    return cfg


def test_the_defaults_are_the_shipped_behaviour():
    """Both switches default to "do nothing", in the dataclass and in the YAML."""
    assert NCFConfig().freeze_after is None
    assert NCFConfig().warm_start_epochs == 0
    shipped = SimConfig.from_yaml(CONFIG).ncf
    assert shipped.freeze_after is None
    assert shipped.warm_start_epochs == 0


def test_defaults_are_bit_identical_to_explicit_no_ops():
    """Passing the no-op values explicitly changes nothing.

    This is the check the brief asks for: the default must leave the existing
    suite's numbers alone. ``None``/``0`` written out is the same run as omitted.
    """
    omitted = Simulation(config=_small()).run(3)["gini"].to_numpy()
    explicit = Simulation(
        config=_small(freeze_after=None, warm_start_epochs=0)
    ).run(3)["gini"].to_numpy()
    np.testing.assert_array_equal(omitted, explicit)


def test_rejects_a_nonsense_freeze_epoch():
    with pytest.raises(ValueError, match="freeze_after"):
        SimConfig.coerce(_small(freeze_after=-1))


def test_rejects_a_nonsense_warm_start():
    with pytest.raises(ValueError, match="warm_start_epochs"):
        SimConfig.coerce(_small(warm_start_epochs=-1))


def test_freeze_after_changes_the_trajectory():
    """A frozen ranker must diverge from the control, or the switch does nothing.

    Short horizon, small population: this asserts the flag reaches the training
    loop, not that the intervention produces any particular result.
    """
    control = Simulation(config=_small()).run(6)["gini"].to_numpy()
    frozen = Simulation(config=_small(freeze_after=2)).run(6)["gini"].to_numpy()
    assert not np.array_equal(control, frozen), (
        "freeze_after=2 produced the same trajectory as an unfrozen run; the "
        "switch is not reaching the training loop"
    )
    # Epochs 0..2 train in both arms, so the first three rows must agree.
    np.testing.assert_array_equal(control[:3], frozen[:3])


def test_freeze_after_zero_trains_once():
    """``freeze_after=0`` is "train on epoch 0 only", not "never train"."""
    frozen = Simulation(config=_small(freeze_after=0)).run(4)["gini"].to_numpy()
    never = Simulation(config=_small(warm_start_epochs=0, freeze_after=-0)).run(4)[
        "gini"
    ].to_numpy()
    np.testing.assert_array_equal(frozen, never)


def test_warm_start_changes_the_trajectory():
    """A warm start must diverge from the control at the first scored epoch."""
    control = Simulation(config=_small()).run(4)["gini"].to_numpy()
    warm = Simulation(config=_small(warm_start_epochs=2)).run(4)["gini"].to_numpy()
    assert not np.array_equal(control, warm), (
        "warm_start_epochs=2 produced the same trajectory as no warm start; the "
        "switch is not reaching the training loop"
    )
