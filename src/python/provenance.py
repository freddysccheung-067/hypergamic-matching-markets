"""Provenance for every run of the second revision: what ran, from which code.

Each per-seed job writes its data and, beside it, a JSON record of the commit it
ran from, whether that tree was clean, the full config and its hash, the thread
counts, the library versions and the wall time. A directory of such records can
then be checked for being one experiment rather than several: one commit, one
config hash, one thread count.

The config hash leaves out the population seed, so every seed of one design
shares a hash and a different design cannot.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


def config_hash(config: dict[str, Any]) -> str:
    """sha256 of the canonical JSON of ``config`` with ``population.seed`` removed."""
    cfg = copy.deepcopy(config)
    cfg.get("population", {}).pop("seed", None)
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def git_state(path: pathlib.Path) -> dict[str, Any]:
    """HEAD and cleanliness of the tree the code was imported from."""
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=path, capture_output=True, text=True,
                              check=True).stdout.strip()
    try:
        head = run("rev-parse", "HEAD")
        dirty = bool(run("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": head, "dirty": dirty}


def environment() -> dict[str, Any]:
    import torch  # noqa: PLC0415

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }


def write_csv_atomic(frame: pd.DataFrame, path: pathlib.Path) -> None:
    """Write through a temporary name so a killed job never leaves half a file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def write_json_atomic(record: dict[str, Any], path: pathlib.Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)


class JobRecord:
    """Collects one job's provenance; ``finish`` writes it as the job's done-marker."""

    def __init__(self, code_root: pathlib.Path, experiment: str, seed: int,
                 config: dict[str, Any], argv: list[str], allow_dirty: bool = False) -> None:
        self.started = time.time()
        self.git = git_state(code_root)
        if self.git["dirty"] and not allow_dirty:
            raise SystemExit(
                f"refusing to run from a dirty tree at {code_root}: commit first, or pass "
                "--allow-dirty for a run that will not be quoted"
            )
        self.record: dict[str, Any] = {
            "experiment": experiment,
            "seed": int(seed),
            "git_commit": self.git["commit"],
            "git_dirty": self.git["dirty"],
            "config_hash": config_hash(config),
            "config": config,
            "argv": argv,
            "environment": environment(),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(self.started)),
            "wall_seconds": {},
        }

    def time(self, label: str, seconds: float) -> None:
        self.record["wall_seconds"][label] = round(float(seconds), 3)

    def finish(self, path: pathlib.Path, **extra: Any) -> None:
        self.record.update(extra)
        self.record["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.record["wall_seconds"]["total"] = round(time.time() - self.started, 3)
        write_json_atomic(self.record, path)


def summarize_directory(directory: pathlib.Path) -> dict[str, Any]:
    """One experiment's provenance, from the per-seed records in ``directory``.

    Returns the commits, config hashes and thread counts found, and whether each
    is unique. An analysis that reads the directory checks these before it
    computes anything.
    """
    records = [json.loads(p.read_text(encoding="utf-8"))
               for p in sorted(directory.glob("seed*.json"))]
    commits = sorted({r["git_commit"] for r in records})
    hashes = sorted({r["config_hash"] for r in records})
    threads = sorted({r["config"]["ncf"]["num_threads"] for r in records})
    return {
        "directory": directory.as_posix(),
        "n_seeds": len(records),
        "seeds": sorted(r["seed"] for r in records),
        "experiments": sorted({r["experiment"] for r in records}),
        "git_commits": commits,
        "any_dirty": any(r["git_dirty"] for r in records),
        "config_hashes": hashes,
        "num_threads": threads,
        "wall_seconds_total": round(sum(r["wall_seconds"]["total"] for r in records), 1),
    }
