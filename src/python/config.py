"""Typed configuration for the simulation (``experiments/base.yaml``).

The C++ extension takes a plain ``dict`` — pybind11's ``py::dict`` — so this
module's job is to be the *checked* side: parse the YAML once, refuse values that
the model cannot run with, and hand out a dict that is known-good. Anything that
would produce a silently meaningless run (an odd population, a top_k larger than
the candidate pool, a negative delta) fails here rather than N epochs into a
sweep.

``SimConfig.from_yaml`` is the only constructor that reads a file; the dataclasses
themselves are plain and can be built directly in tests.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field, fields
from typing import Any

import yaml

# Repository root, so the default config path works from any working directory.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]


@dataclass
class TraitDist:
    """One ``population.traits.<name>`` block.

    ``p1``/``p2`` are the two parameters of whichever distribution ``dist`` names,
    with the meanings documented on ``TraitDist`` in ``include/population.h``:
    (mean, std) for normal, (mu, sigma) for lognormal, (x_min, alpha) for pareto.
    Kept under neutral names here for the same reason the C++ keeps them: the same
    slot means different things per distribution.
    """

    dist: str
    mean: float
    std: float

    def __post_init__(self) -> None:
        if self.dist not in ("normal", "lognormal", "pareto"):
            raise ValueError(
                f"unsupported distribution {self.dist!r}; expected normal, lognormal or pareto"
            )
        if self.dist == "pareto" and (self.mean <= 0.0 or self.std <= 0.0):
            raise ValueError("pareto requires x_min > 0 and alpha > 0")

    def to_dict(self) -> dict[str, Any]:
        return {"dist": self.dist, "mean": self.mean, "std": self.std}


@dataclass
class PopulationConfig:
    n: int
    seed: int
    admissible_fraction: float
    traits: dict[str, TraitDist]
    # Absent by default: θ is then calibrated as a per-agent utility quantile.
    # A number here is an absolute utility cutoff and bypasses calibration, which
    # is why it is not given a plausible-looking default.
    init_threshold: float | None = None
    # Spread of the per-agent taste multipliers (THEORY.md §2.3). lognormal with
    # mean 1, so this is a coefficient of variation: 0 switches heterogeneity off
    # and every multiplier is exactly 1.0. No empirical anchor — a modeling knob.
    taste_sigma: float = 0.0

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError(f"population.n must be positive; got {self.n}")
        if self.n % 2 != 0:
            # The C++ sampler drops the last agent with a warning to stderr. That
            # is the right behavior for a library call and the wrong one for a
            # config: the run would be balanced but not the requested size, and
            # every N in the output would be off by one.
            raise ValueError(f"population.n must be even so the two sides are equal; got {self.n}")
        missing = {"status", "attractiveness", "sociability", "age"} - set(self.traits)
        if missing:
            raise ValueError(f"population.traits is missing {sorted(missing)}")
        if not 0.0 < self.admissible_fraction <= 1.0:
            raise ValueError(
                "population.admissible_fraction must be in (0, 1]; "
                f"got {self.admissible_fraction}"
            )
        if self.taste_sigma < 0.0:
            raise ValueError(f"population.taste_sigma must be >= 0; got {self.taste_sigma}")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n": self.n,
            "seed": self.seed,
            "admissible_fraction": self.admissible_fraction,
            "traits": {k: v.to_dict() for k, v in self.traits.items()},
            "taste_sigma": self.taste_sigma,
        }
        if self.init_threshold is not None:
            out["init_threshold"] = self.init_threshold
        return out


@dataclass
class HypergamyConfig:
    delta: float
    alpha: float
    beta: float
    gamma: float
    lam: float
    # Absolute utility units of persistent pairwise chemistry (THEORY.md §2.3).
    # Not a ratio: the orchestrator logs shock_scale * 1.8138 / utility_sd once at
    # construction so the implied noise-to-signal ratio stays visible across
    # retunes of alpha/beta. 0 switches the term off entirely.
    shock_scale: float = 0.0

    def __post_init__(self) -> None:
        if self.delta < 0.0:
            # δ is a required *surplus*; a negative value would make the indicator
            # fire for men below her own status and the asymmetry would mean the
            # opposite of what THEORY.md §2.1 defines.
            raise ValueError(f"hypergamy.delta must be >= 0; got {self.delta}")
        if self.gamma < 0.0:
            raise ValueError(f"hypergamy.gamma must be >= 0; got {self.gamma}")
        if self.shock_scale < 0.0:
            raise ValueError(f"hypergamy.shock_scale must be >= 0; got {self.shock_scale}")

    def to_dict(self) -> dict[str, Any]:
        # The C++ reads the key "lambda"; the Python attribute is `lam` because
        # `lambda` is a keyword. The mapping lives here and nowhere else.
        return {
            "delta": self.delta,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "lambda": self.lam,
            "shock_scale": self.shock_scale,
        }


@dataclass
class NCFConfig:
    embed_dim: int = 64
    mlp_layers: list[int] = field(default_factory=lambda: [128, 64, 32])
    lr: float = 1e-3
    l2_reg: float = 1e-5
    pos_weight: float = 4.0
    top_k: int = 20
    n_negatives: int = 4

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError(f"ncf.top_k must be >= 1; got {self.top_k}")
        if self.n_negatives < 0:
            raise ValueError(f"ncf.n_negatives must be >= 0; got {self.n_negatives}")
        if not self.mlp_layers:
            raise ValueError("ncf.mlp_layers must contain at least one width")

    def to_dict(self) -> dict[str, Any]:
        return {
            "embed_dim": self.embed_dim,
            "mlp_layers": list(self.mlp_layers),
            "lr": self.lr,
            "l2_reg": self.l2_reg,
            "pos_weight": self.pos_weight,
            "top_k": self.top_k,
            "n_negatives": self.n_negatives,
        }


@dataclass
class SimulationConfig:
    n_epochs: int = 100
    bayesian_eta: float = 0.1
    n_generations: int = 5
    turnover_fraction: float = 0.2

    def __post_init__(self) -> None:
        if self.delta_eta_out_of_range():
            raise ValueError(
                f"simulation.bayesian_eta must be in [0, 1]; got {self.bayesian_eta}"
            )
        if self.n_generations < 1:
            raise ValueError(f"simulation.n_generations must be >= 1; got {self.n_generations}")

    def delta_eta_out_of_range(self) -> bool:
        return not 0.0 <= self.bayesian_eta <= 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_epochs": self.n_epochs,
            "bayesian_eta": self.bayesian_eta,
            "n_generations": self.n_generations,
            "turnover_fraction": self.turnover_fraction,
        }


@dataclass
class OutputConfig:
    dir: str = "data/processed/"
    save_every: int = 10

    def to_dict(self) -> dict[str, Any]:
        return {"dir": self.dir, "save_every": self.save_every}


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ValueError(f"config is missing {where}.{key}")
    return mapping[key]


def _reject_unknown(block: dict[str, Any], known: set[str], where: str) -> None:
    """The same strict check _build applies, for a block parsed by hand.

    ``population`` is built field by field rather than through _build (its trait
    dict and optional init_threshold need their own handling), so it would
    otherwise be the one block where a misspelled key passes unnoticed — and
    ``taste_sigm`` silently sampling a homogeneous population is exactly the kind
    of failure this rejects.
    """
    for key in block:
        if key not in known:
            raise ValueError(f"unknown key {where}.{key}")


def _build(cls: type, block: dict[str, Any], where: str, renames: dict[str, str] | None = None):
    """Construct a dataclass from a config block, rejecting unknown keys.

    Unknown keys are an error rather than ignored: a misspelled ``topk`` would
    otherwise leave the default in place and the run would silently use a k
    nobody asked for. Renames map a YAML key to a Python attribute that cannot
    carry the same name (``lambda`` -> ``lam``).
    """
    renames = renames or {}
    known = {f.name for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in block.items():
        name = renames.get(key, key)
        if name not in known:
            raise ValueError(f"unknown key {where}.{key}")
        kwargs[name] = value
    return cls(**kwargs)


@dataclass
class SimConfig:
    population: PopulationConfig
    hypergamy: HypergamyConfig
    ncf: NCFConfig
    simulation: SimulationConfig
    output: OutputConfig

    def __post_init__(self) -> None:
        # top_k must be smaller than the pool an agent chooses from. The pool is
        # the opposite side, n/2 — not n. Beyond that, the NCF's top-k raises
        # (ncf.py rejects k > available candidates) and the run dies on epoch 1.
        pool = self.population.n // 2
        if self.ncf.top_k >= pool:
            raise ValueError(
                f"ncf.top_k ({self.ncf.top_k}) must be < n/2 ({pool}), the size of the "
                "candidate pool"
            )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SimConfig:
        """Build from a mapping in the YAML's shape, with the same validation."""
        if not isinstance(raw, dict):
            raise ValueError("config did not parse to a mapping")

        pop_raw = _require(raw, "population", "config")
        _reject_unknown(pop_raw, {"n", "seed", "admissible_fraction", "traits",
                                  "init_threshold", "taste_sigma"}, "population")
        traits_raw = _require(pop_raw, "traits", "population")
        traits = {
            name: _build(TraitDist, block, f"population.traits.{name}")
            for name, block in traits_raw.items()
        }
        population = PopulationConfig(
            n=_require(pop_raw, "n", "population"),
            seed=_require(pop_raw, "seed", "population"),
            admissible_fraction=pop_raw.get("admissible_fraction", 0.2),
            traits=traits,
            init_threshold=pop_raw.get("init_threshold"),
            taste_sigma=pop_raw.get("taste_sigma", 0.0),
        )

        # The strict unknown-key check lives in _build; these two blocks are built
        # through it, so a misspelled "shock_scal" fails here rather than silently
        # leaving the knob at its default.
        return cls(
            population=population,
            hypergamy=_build(
                HypergamyConfig,
                _require(raw, "hypergamy", "config"),
                "hypergamy",
                renames={"lambda": "lam"},
            ),
            ncf=_build(NCFConfig, raw.get("ncf", {}), "ncf"),
            simulation=_build(SimulationConfig, raw.get("simulation", {}), "simulation"),
            output=_build(OutputConfig, raw.get("output", {}), "output"),
        )

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path | None = None) -> SimConfig:
        """Load and validate ``experiments/base.yaml`` (UTF-8)."""
        resolved = (
            pathlib.Path(path) if path is not None
            else _PROJECT_ROOT / "experiments" / "base.yaml"
        )
        if not resolved.is_absolute():
            resolved = _PROJECT_ROOT / resolved
        # encoding="utf-8" is not optional on Windows: the file contains em
        # dashes, and the default codec there is gbk, which raises on them.
        with open(resolved, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        """The plain dict shape ``_hypergamic`` expects."""
        return {
            "population": self.population.to_dict(),
            "hypergamy": self.hypergamy.to_dict(),
            "ncf": self.ncf.to_dict(),
            "simulation": self.simulation.to_dict(),
            "output": self.output.to_dict(),
        }

    # ``Simulation`` accepts a raw dict as well as a SimConfig so a sweep can
    # mutate one block without rebuilding the whole object (scripts/run_sweep.py).
    @classmethod
    def coerce(cls, config: SimConfig | dict[str, Any] | None) -> SimConfig:
        if isinstance(config, SimConfig):
            return config
        if config is None:
            return cls.from_yaml()
        if isinstance(config, dict):
            # Through the same validation, so a dict built by hand gets the
            # checks a YAML file gets.
            return cls.from_dict(config)
        raise TypeError(f"config must be a SimConfig, a dict or None; got {type(config)}")
