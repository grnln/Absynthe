"""Pydantic models for integer distributions.

Generic and reusable — not coupled to motifs, perturbations, or any
specific domain concept.  Import ``IntDistributionModel`` wherever a
configurable integer distribution is needed in an API request/response.
"""

from typing import Dict, Optional

from pydantic import BaseModel, Field, model_validator


class IntDistributionModel(BaseModel):
    """Integer distribution configuration for API requests.

    Supported types:

    * ``"uniform"``  — requires ``min`` and ``max``
    * ``"normal"``   — requires ``mean`` and ``std``
    * ``"poisson"``  — requires ``lambda``

    ``min`` / ``max`` (top level, not in ``params``) are optional bounds:
    every sample is clipped into ``[min, max]``. Without ``min`` the lower
    bound is the default of the field being sampled (1 for motif and
    perturbation counts, 0 for extra vertices / edges).
    """

    type: str = Field(..., description="Distribution type: uniform | normal | poisson")
    params: Dict[str, float] = Field(default_factory=dict)
    min: Optional[int] = Field(None, ge=0, description="Optional lower bound (clip); overrides the field default")
    max: Optional[int] = Field(None, ge=0, description="Optional upper bound (clip)")

    @model_validator(mode="after")
    def validate_params(self) -> "IntDistributionModel":
        t = self.type
        p = self.params
        if t == "uniform":
            if "min" not in p or "max" not in p:
                raise ValueError("uniform distribution requires 'min' and 'max' params")
            if p["min"] > p["max"]:
                raise ValueError("'min' must be <= 'max'")
        elif t == "normal":
            if "mean" not in p or "std" not in p:
                raise ValueError("normal distribution requires 'mean' and 'std' params")
            if p["std"] < 0:
                raise ValueError("'std' must be >= 0")
        elif t == "poisson":
            if "lambda" not in p:
                raise ValueError("poisson distribution requires 'lambda' param")
            if p["lambda"] <= 0:
                raise ValueError("'lambda' must be > 0")
        else:
            raise ValueError(f"Unknown distribution type: {t!r}")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("bound 'min' must be <= bound 'max'")
        return self
