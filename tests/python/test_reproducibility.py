"""The reproducibility contract: one seed and one config, one run.

This file exists because of a defect found after the paper was drafted. The
contract in ``docs/INTERFACES.md`` §3 said results were "byte-identical within
one toolchain and across thread counts". The C++ half of that was true and
tested; the torch half was not, and nobody had checked.

Torch's intra-op reductions are not bitwise associative. The same seed, the same
data and the same weights at 1 and at 8 threads differ by roughly 2e-7 in
relative terms after twenty training steps. That is invisible in one step and
not invisible across a hundred epochs of a feedback loop, where the NCF's
shortlists feed the matching that feeds the next epoch's labels: measured on
this machine the exposure Gini at epoch 99 for seed 42 ranges from 0.6922 to
0.7076 across thread counts, a spread of 0.015 against a between-seed sd of
0.018.

So ``ncf.num_threads`` is part of the run's identity, and it defaults to 1. These
tests pin that:

1. The same seed at the default thread count reproduces exactly, twice.
2. The thread count is read from config and actually reaches torch, so a run
   cannot silently inherit the machine's core count.
3. Changing the thread count is *allowed* and still gives a usable result — the
   point is not that multithreading is forbidden, it is that it must be named.

Test 3 runs two short NCF simulations and would take minutes at the full
configuration, so it is marked ``slow`` and skipped by the default CI run. The
others are cheap because they use a small population.
"""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import pytest
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.python.config import SimConfig  # noqa: E402
from src.python.orchestrator import Simulation  # noqa: E402

CONFIG = "experiments/base.yaml"


def _small(seed: int = 42, **ncf) -> dict:
    """The shipped config at a population small enough for a fast test."""
    cfg = SimConfig.from_yaml(CONFIG).to_dict()
    cfg["population"]["seed"] = int(seed)
    cfg["population"]["n"] = 120
    # k must stay below the candidate side size, and the gate condition scales
    # with it; the tests below do not depend on the gate binding.
    cfg["ncf"]["top_k"] = 8
    cfg["ncf"].update(ncf)
    return cfg


def test_same_seed_reproduces_the_run_exactly():
    """Two runs of one seed and one config agree bit for bit.

    This is the whole contract in one assertion. If it fails, nothing else in
    the repository can be trusted to reproduce, because every committed number
    is one run of one seed.
    """
    cfg = _small()
    first = Simulation(config=cfg).run(3)
    second = Simulation(config=cfg).run(3)
    np.testing.assert_array_equal(
        first["gini"].to_numpy(),
        second["gini"].to_numpy(),
        err_msg="same seed and config produced different exposure Gini",
    )


def test_thread_count_reaches_torch_from_config():
    """``ncf.num_threads`` is applied, not merely accepted.

    A config field that parses and is then ignored would leave the contract
    unenforced while looking enforced, which is the failure this file exists to
    prevent.

    torch clamps the count to the machine's core count, so the request is kept
    inside what the runner can honour: asking for 3 on a 2-core box yields 2 and
    the assertion would fail for a reason that has nothing to do with the code
    under test.
    """
    cores = os.cpu_count() or 1
    requested = min(3, cores)
    if requested < 2:
        pytest.skip("single-core machine: no thread count above 1 to request")

    before = torch.get_num_threads()
    try:
        Simulation(config=_small(num_threads=requested))
        assert torch.get_num_threads() == requested
    finally:
        torch.set_num_threads(before)


def test_thread_count_defaults_to_one():
    """The shipped default is 1, so an unconfigured run is reproducible.

    Read from the dataclass default and from the YAML, because the two can drift
    and the YAML is what actually ships.
    """
    from src.python.config import NCFConfig  # noqa: PLC0415

    assert NCFConfig().num_threads == 1
    assert SimConfig.from_yaml(CONFIG).ncf.num_threads == 1


def test_rejects_a_nonsense_thread_count():
    with pytest.raises(ValueError, match="num_threads"):
        SimConfig.from_yaml(CONFIG)  # baseline parses
        cfg = SimConfig.from_yaml(CONFIG).to_dict()
        cfg["ncf"]["num_threads"] = 0
        SimConfig.coerce(cfg)


def test_the_pin_is_what_makes_a_run_reproducible():
    """Pin two thread counts and the run still reproduces; leave it loose and it may not.

    This replaces an earlier version that asserted torch's matmul *is* unstable
    across thread counts. That assertion was environment-dependent: on a runner
    whose BLAS happens to reduce in a stable order it fails while the code is
    fine, and the test would have been reporting on the machine rather than on
    the repository. The property that matters is the positive one — with
    ``num_threads`` pinned, repeated runs agree exactly — and that holds whether
    or not torch is unstable underneath.

    The instability itself is characterised where it is a fact about the data
    rather than about any one runner: ``data/processed/rigor/thread_sensitivity.csv``
    records the measured spread. If a future torch makes reduction order stable,
    nothing here breaks and the pin simply becomes unnecessary, which is the
    right failure mode for a guard.
    """
    for threads in (1, 2):
        if threads > (os.cpu_count() or 1):
            continue
        cfg = _small(num_threads=threads)
        first = Simulation(config=cfg).run(3)["gini"].to_numpy()
        second = Simulation(config=cfg).run(3)["gini"].to_numpy()
        np.testing.assert_array_equal(
            first,
            second,
            err_msg=f"num_threads={threads} did not reproduce its own run",
        )


@pytest.mark.slow
def test_unpinned_threads_can_change_the_numbers():
    """Characterisation, kept out of the default run because it is environment-shaped.

    Records that a full simulation at two thread counts can disagree. Marked slow
    and therefore excluded from CI: whether it holds depends on the runner's BLAS
    and core count, so as a gate it would fail for reasons the repository cannot
    control. It is useful locally, where the measurement is meaningful.
    """
    if (os.cpu_count() or 1) < 2:
        pytest.skip("single-core machine: nothing to compare")

    one = Simulation(config=_small(num_threads=1)).run(3)["gini"].to_numpy()
    two = Simulation(config=_small(num_threads=2)).run(3)["gini"].to_numpy()
    if np.array_equal(one, two):
        pytest.skip(
            "this machine's BLAS reduces in a stable order, so there is no "
            "instability to characterise here; the pin stands regardless"
        )
    assert one.shape == two.shape


# ── the archival promise ──────────────────────────────────────────────────────
#
# ``docs/INTERFACES.md`` §3.1(c) says the committed data under ``data/processed/``
# was generated before ``ncf.num_threads`` existed, and that each directory's
# README states the count it was generated at. That sentence is the only thing
# standing between a reader and a silent mismatch between the README and the
# files, and prose alone does not fail when it stops being true. These tests are
# what make it fail.
#
# The stronger check would be to read a thread count out of each committed CSV and
# compare it to the README. No CSV carries such a column (only
# ``rigor/thread_sensitivity.csv`` does, and that file *is* the measurement rather
# than a run), so the check is the structural one the contract can actually
# support.

PROCESSED = PROJECT_ROOT / "data" / "processed"

# Directories that hold committed simulation output. ``separating/`` and the three
# ``thin_market*`` directories were added to this list when their READMEs were
# written. Each has its own README and each states its own thread count, even
# where the answer is "not applicable", because a reader who lands on one of them
# should not have to find a sibling to learn that.
DATA_DIRS = [
    "ablation",
    "ablation_live",
    "ablation_h100",
    "flooding",
    "institutional",
    "rigor",
    "separating",
    "thin_market",
    "thin_market_q0.95",
    "thin_market_q0.99",
    # Subdirectories of rigor/. They were missed by an earlier version of the
    # coverage test, which only walked the top level and so never noticed that
    # rigor/base/ had no README at all.
    "rigor/base",
    "rigor/long_horizon",
    "rigor/sensitivity",
    "rigor/fresh",
    "rigor/exploratory_1t",
    "rigor/interventions",
    "rigor/latefreeze",
    "rigor/c4_shock0.5",
    "rigor/c2_fresh_inert",
    "rigor/c2_fresh_institutional",
    "rigor/h4_shipped_gate",
    "rigor/long_horizon_1t",
    # The second revision. Every run under horizon/ wrote a run record stating its
    # thread count, so the directory's own README covers every experiment below it,
    # and each experiment directory carries a README of its own as well.
    "horizon",
]
HORIZON_DIRS = ["e09_canonical", "e10_long", "e11_sweep", "e13a_dose", "e13b_explore",
                "e13c_offpolicy", "e13d_warm", "e14_arrivals", "e15_institutional",
                "e15i_inert", "e15s_five", "e16_fp", "e17_size", "h1r", "h5r"]
DATA_DIRS += [f"horizon/{d}" for d in HORIZON_DIRS]


def test_every_data_directory_documents_its_thread_count():
    """Each committed data directory states the thread count it was generated at.

    Adding a directory without a "Thread count" section fails here rather than
    leaving a reader to discover the gap by re-running something.

    Deliberately independent of this machine's core count: it reads files and
    compares strings, so it passes on a 4-core CI runner and on a 96-core one
    alike. That is the point of choosing the structural check over a re-run.
    """
    missing = []
    for name in DATA_DIRS:
        readme = PROCESSED / name / "README.md"
        if not readme.exists():
            missing.append(f"{name}/README.md does not exist")
            continue
        text = readme.read_text(encoding="utf-8")
        if "## Thread count" not in text:
            missing.append(f"{name}/README.md has no '## Thread count' section")
    assert not missing, (
        "committed data directories must document the thread count they were "
        "generated at (docs/INTERFACES.md §3.1c):\n  " + "\n  ".join(missing)
    )


def test_every_data_directory_on_disk_is_in_the_list():
    """The list above covers what is actually committed.

    Otherwise a new directory could be added and the test above would keep
    passing, because the test iterates the list rather than the filesystem.

    Walks one level into each top-level directory as well: ``rigor/`` holds ten
    subdirectories of committed output, and an earlier version of this test
    checked only ``rigor/README.md``, which is how ``rigor/base/`` came to have no
    README of its own.
    """
    on_disk: set[str] = set()
    for top in PROCESSED.iterdir():
        if not top.is_dir():
            continue
        if any(top.glob("*.csv")):
            on_disk.add(top.name)
        for sub in top.iterdir():
            if sub.is_dir() and (any(sub.glob("*.csv")) or any(sub.glob("*/*.csv"))):
                on_disk.add(f"{top.name}/{sub.name}")
    undocumented = on_disk - set(DATA_DIRS)
    assert not undocumented, (
        "these data directories exist but are not covered by DATA_DIRS, so their "
        f"README is not checked: {sorted(undocumented)}"
    )


def test_the_archival_directories_are_not_claimed_to_reproduce():
    """The READMEs point at §3.1 rather than promising a byte-for-byte re-run.

    This is the sentence that was wrong before: the repository used to say the
    runs regenerate identically "within one toolchain and across thread counts".
    The C++ half of that was true; the torch half was not. A grep-level assertion
    is crude, but the failure mode it guards against is exactly a sentence coming
    back, and a sentence is what it looks for.
    """
    offenders = []
    for path in [PROJECT_ROOT / "README.md", *(PROCESSED.glob("*/README.md"))]:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if "within one toolchain and across thread counts" in text:
            offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert not offenders, (
        "this sentence overstates the guarantee; point at docs/INTERFACES.md "
        f"§3.1 instead: {offenders}"
    )

