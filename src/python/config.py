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

import math
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
    # ── Institutional state (THEORY.md §10.1) ────────────────────────────────
    # Beta shapes for the intent and values columns. intent is rescaled to (0, 1]
    # at sampling; values stays on [0, 1]. Symmetric (2, 2) is the shipped default:
    # a two-sided population, which is what makes a separating threshold visible at
    # all — with a = b -> inf every agent has the same intent and no cost separates
    # anything.
    intent_beta_a: float = 2.0
    intent_beta_b: float = 2.0
    values_beta_a: float = 2.0
    values_beta_b: float = 2.0
    # Share of agents eligible for the Environment B subgraph. A probability so the
    # subgraph scales with N; the real entry filter is g_min (THEORY.md §10.3).
    community_prob: float = 0.5

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
        for name in ("intent_beta_a", "intent_beta_b", "values_beta_a", "values_beta_b"):
            value = getattr(self, name)
            if value <= 0.0:
                raise ValueError(f"population.{name} must be > 0; got {value}")
        if not 0.0 <= self.community_prob <= 1.0:
            raise ValueError(
                f"population.community_prob must be in [0, 1]; got {self.community_prob}"
            )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n": self.n,
            "seed": self.seed,
            "admissible_fraction": self.admissible_fraction,
            "traits": {k: v.to_dict() for k, v in self.traits.items()},
            "taste_sigma": self.taste_sigma,
            "intent_beta_a": self.intent_beta_a,
            "intent_beta_b": self.intent_beta_b,
            "values_beta_a": self.values_beta_a,
            "values_beta_b": self.values_beta_b,
            "community_prob": self.community_prob,
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
    # Signalling cost c (THEORY.md §10.2): what a proposal costs the proposer,
    # spent rather than disliked. In utility units, so it is comparable to the
    # utilities it is weighed against — but it never enters utility_toward, which
    # is what keeps compute_welfare's numbers unchanged. 0.0 is the pooling regime
    # and reproduces every pre-session result bit for bit.
    signal_cost: float = 0.0

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
        if self.signal_cost < 0.0:
            # A negative cost would pay agents to propose, which is not this model
            # and would make the separating threshold negative — meaning every agent
            # proposes and the "cost" is a subsidy.
            raise ValueError(f"hypergamy.signal_cost must be >= 0; got {self.signal_cost}")

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
            "signal_cost": self.signal_cost,
        }


@dataclass
class EnvironmentBConfig:
    """The ``environment_b`` block: the constrained subgraph (THEORY.md §10.3).

    The first four fields are the parameter vector (N_sub, c_B, g_min, d) of §10.3,
    with N_sub expressed as a share of the population so the subgraph scales with N.
    The rest parameterise the defection survival
    ``sigma_ij = sigma_0 * (tau_i tau_j)^sigma_alpha * g_ij^sigma_beta`` and the loss
    it imposes. Both environments run the same formula; Environment A is the same
    parameters with ``defection_penalty = 0`` (``environment_b.environment_a``).
    """

    # Share of each side drawn into the candidate pool, before the g_min filter and
    # before side-balancing. |B| is therefore at most size_fraction * N and is
    # *measured*, never assumed: the filter and the balance step both remove agents.
    size_fraction: float
    # Entry filter: an agent with values < values_min is ineligible.
    values_min: float
    # c_B. Must exceed Environment A's cost (hypergamy.signal_cost); enforced in
    # SimConfig, which is the only place both numbers are visible.
    signal_cost: float
    # d: welfare charged to the defector of a dissolved match. Does not enter the
    # survival probability — it prices defection, it does not deter it.
    defection_penalty: float
    # Ceiling on survival. 1.0 adds no discount beyond the two alignment factors.
    sigma_0: float = 1.0
    sigma_alpha: float = 0.5
    sigma_beta: float = 0.5
    # L: scale of the loss a dissolved match imposes on the partner, L (1 - tau_defector).
    partner_loss: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.size_fraction <= 1.0:
            raise ValueError(
                f"environment_b.size_fraction must be in (0, 1]; got {self.size_fraction}"
            )
        if not 0.0 <= self.values_min <= 1.0:
            raise ValueError(
                f"environment_b.values_min must be in [0, 1]; got {self.values_min}"
            )
        if self.signal_cost < 0.0:
            raise ValueError(f"environment_b.signal_cost must be >= 0; got {self.signal_cost}")
        if self.defection_penalty < 0.0:
            raise ValueError(
                f"environment_b.defection_penalty must be >= 0; got {self.defection_penalty}"
            )
        if not 0.0 < self.sigma_0 <= 1.0:
            # sigma is a probability. Both alignment factors are in [0, 1], so a
            # sigma_0 above 1 would push sigma past 1 for aligned, high-intent pairs.
            raise ValueError(f"environment_b.sigma_0 must be in (0, 1]; got {self.sigma_0}")
        if self.sigma_alpha < 0.0 or self.sigma_beta < 0.0:
            raise ValueError(
                "environment_b.sigma_alpha and sigma_beta must be >= 0; "
                f"got {self.sigma_alpha}, {self.sigma_beta}"
            )
        if self.partner_loss < 0.0:
            raise ValueError(f"environment_b.partner_loss must be >= 0; got {self.partner_loss}")

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


_ENV_B_REQUIRED = ("size_fraction", "values_min", "signal_cost", "defection_penalty")


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
    # Exogenous drift of W's status (THEORY.md §10.6): s_w(t) = s_w(0) + gamma*t, with
    # gamma in units of the epoch-0 population status sd per epoch so the knob means
    # the same thing at any trait scale. 0.0 is the no-drift cell and leaves every
    # pre-existing number untouched.
    status_drift: float = 0.0

    def __post_init__(self) -> None:
        if self.delta_eta_out_of_range():
            raise ValueError(
                f"simulation.bayesian_eta must be in [0, 1]; got {self.bayesian_eta}"
            )
        if self.n_generations < 1:
            raise ValueError(f"simulation.n_generations must be >= 1; got {self.n_generations}")
        if not math.isfinite(self.status_drift):
            raise ValueError(f"simulation.status_drift must be finite; got {self.status_drift}")

    def delta_eta_out_of_range(self) -> bool:
        return not 0.0 <= self.bayesian_eta <= 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_epochs": self.n_epochs,
            "bayesian_eta": self.bayesian_eta,
            "n_generations": self.n_generations,
            "turnover_fraction": self.turnover_fraction,
            "status_drift": self.status_drift,
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
    # Absent means no Environment B is defined; the "complete" arm then refuses to
    # run rather than inventing a subgraph.
    environment_b: EnvironmentBConfig | None = None

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

    def require_environment_b(self) -> EnvironmentBConfig:
        """The B block, checked against A. Called wherever B is actually run.

        Deliberately not in ``__post_init__``: ``hypergamy.signal_cost`` is a sweep
        axis for Environment A (the H1 grid runs it to 1.0 and beyond), and a
        cross-check there would reject every A-side sweep value at or above B's cost
        merely because an ``environment_b`` block sits in the same file. The contrast
        only has to hold for a run that uses B.
        """
        env = self.environment_b
        if env is None:
            raise ValueError("config has no environment_b block")
        if env.signal_cost <= self.hypergamy.signal_cost:
            # B's defining contrast is a *costlier* signal. Equal or lower would make
            # the two environments differ in size and visibility only, which is a
            # different experiment from the one THEORY.md §10.3 specifies.
            raise ValueError(
                f"environment_b.signal_cost ({env.signal_cost}) must exceed "
                f"Environment A's hypergamy.signal_cost ({self.hypergamy.signal_cost})"
            )
        return env

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SimConfig:
        """Build from a mapping in the YAML's shape, with the same validation."""
        if not isinstance(raw, dict):
            raise ValueError("config did not parse to a mapping")

        pop_raw = _require(raw, "population", "config")
        _reject_unknown(pop_raw, {"n", "seed", "admissible_fraction", "traits",
                                  "init_threshold", "taste_sigma",
                                  "intent_beta_a", "intent_beta_b",
                                  "values_beta_a", "values_beta_b",
                                  "community_prob"}, "population")
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
            intent_beta_a=pop_raw.get("intent_beta_a", 2.0),
            intent_beta_b=pop_raw.get("intent_beta_b", 2.0),
            values_beta_a=pop_raw.get("values_beta_a", 2.0),
            values_beta_b=pop_raw.get("values_beta_b", 2.0),
            community_prob=pop_raw.get("community_prob", 0.5),
        )

        env_raw = raw.get("environment_b")
        environment_b = None
        if env_raw is not None:
            if not isinstance(env_raw, dict):
                raise ValueError("environment_b must be a mapping")
            for key in _ENV_B_REQUIRED:
                _require(env_raw, key, "environment_b")
            environment_b = _build(EnvironmentBConfig, env_raw, "environment_b")

        # The strict unknown-key check lives in _build; these two blocks are built
        # through it, so a misspelled "shock_scal" fails here rather than silently
        # leaving the knob at its default.
        return cls(
            environment_b=environment_b,
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
        out: dict[str, Any] = {
            "population": self.population.to_dict(),
            "hypergamy": self.hypergamy.to_dict(),
            "ncf": self.ncf.to_dict(),
            "simulation": self.simulation.to_dict(),
            "output": self.output.to_dict(),
        }
        if self.environment_b is not None:
            out["environment_b"] = self.environment_b.to_dict()
        return out

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
