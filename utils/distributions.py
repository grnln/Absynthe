"""Generic integer distribution sampling.

Reusable module for sampling integers from configurable distributions.
Uses only ``random.Random`` — no numpy dependency.

Usage::

    from utils.distributions import IntDistribution, sample_int
    from utils.rng import get_rng

    dist = IntDistribution(type="normal", params={"mean": 4, "std": 1})
    value = sample_int(dist, get_rng())          # integer >= 1
    value = sample_int(dist, get_rng(), min_value=0)  # integer >= 0

    # Optional bounds on the distribution itself (clipping)
    bounded = IntDistribution("normal", {"mean": 4, "std": 1}, min=2, max=6)
    value = sample_int(bounded, get_rng())       # integer in [2, 6]

    # Fields that accept either a fixed int or a distribution (``CountSpec``)
    resolve_count(3, get_rng())                  # -> 3
    resolve_count(dist, get_rng(), min_value=0)  # -> a fresh sample
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Union


@dataclass
class IntDistribution:
    """Configuration for sampling integer values from a distribution.

    Supported types and their params:

    * ``"uniform"``  — ``{"min": a, "max": b}``
    * ``"normal"``   — ``{"mean": mu, "std": sigma}``
    * ``"poisson"``  — ``{"lambda": lam}``

    ``min`` / ``max`` are optional bounds applied to every sample by
    clipping: a value outside the range is replaced by the nearest bound.
    When ``min`` is not set, the lower bound is the ``min_value`` of the
    context that samples (1 for motif and perturbation counts, 0 for extra
    vertices / edges). Samples are never negative.
    """

    type: str
    params: Dict[str, float] = field(default_factory=dict)
    min: Optional[int] = None
    max: Optional[int] = None

    def __post_init__(self):
        if self.min is not None and self.min < 0:
            raise ValueError("'min' must be >= 0")
        if self.max is not None and self.max < 0:
            raise ValueError("'max' must be >= 0")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("'min' must be <= 'max'")

    def to_dict(self) -> Dict:
        """JSON-ready form: ``{"type", "params"}`` plus only the bounds that are set."""
        out: Dict = {"type": self.type, "params": dict(self.params)}
        if self.min is not None:
            out["min"] = self.min
        if self.max is not None:
            out["max"] = self.max
        return out


# A count that is either fixed (int) or re-sampled from a distribution.
CountSpec = Union[int, IntDistribution]


def to_distribution(value) -> Optional[IntDistribution]:
    """Coerce an ``IntDistribution``, a ``{"type", "params"[, "min", "max"]}``
    mapping or an object with those attributes (e.g. a pydantic model) into
    an ``IntDistribution``. ``None`` stays ``None``."""
    if value is None or isinstance(value, IntDistribution):
        return value
    if isinstance(value, dict):
        return IntDistribution(
            type=value["type"],
            params=dict(value.get("params") or {}),
            min=value.get("min"),
            max=value.get("max"),
        )
    return IntDistribution(
        type=value.type,
        params=dict(value.params or {}),
        min=getattr(value, "min", None),
        max=getattr(value, "max", None),
    )


def resolve_count(
    spec: CountSpec,
    rng: Union[random.Random, "module"],
    min_value: int = 1,
) -> int:
    """Return *spec* itself when it is an int, otherwise sample it.

    Lets a configuration field hold either a fixed count or an
    :class:`IntDistribution` that is re-sampled on every call (per graph).
    """
    if isinstance(spec, IntDistribution):
        return sample_int(spec, rng, min_value=min_value)
    if isinstance(spec, dict):
        return sample_int(to_distribution(spec), rng, min_value=min_value)
    return max(min_value, int(spec))


def _sample_poisson(lam: float, rng: Union[random.Random, "module"]) -> int:
    """Knuth's algorithm for Poisson sampling using only ``rng.random()``."""
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k - 1


def sample_int(
    dist: IntDistribution,
    rng: Union[random.Random, "module"],
    min_value: int = 1,
) -> int:
    """Sample a single integer from *dist* and clip it to its bounds.

    Parameters
    ----------
    dist:
        Distribution configuration.
    rng:
        A ``random.Random`` instance or the ``random`` module itself.
    min_value:
        Default lower bound (``1``), used when ``dist.min`` is not set.
        ``dist.min`` / ``dist.max`` override it; the result is never
        negative.
    """
    t = dist.type

    if t == "uniform":
        lo = int(dist.params.get("min", 1))
        hi = int(dist.params.get("max", 5))
        value = rng.randint(lo, hi)

    elif t == "normal":
        mu = dist.params.get("mean", 3.0)
        sigma = dist.params.get("std", 1.0)
        value = round(rng.gauss(mu, sigma))

    elif t == "poisson":
        lam = dist.params.get("lambda", 3.0)
        value = _sample_poisson(lam, rng)

    else:
        raise ValueError(f"Unknown distribution type: {t!r}")

    lower = max(0, dist.min if dist.min is not None else min_value)
    value = max(lower, value)
    if dist.max is not None:
        value = min(dist.max, value)
    return value
