from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ElementSpec:
    basis: list[str]
    op_type: str
    selector: int


@dataclass(frozen=True)
class Mesh1D:
    domain: tuple[float, float]
    element_bounds: np.ndarray
    element_specs: list[ElementSpec]

    def __post_init__(self) -> None:
        bounds = np.asarray(self.element_bounds, dtype=float)
        object.__setattr__(self, "element_bounds", bounds)

        if bounds.ndim != 1:
            raise ValueError("element_bounds must be one-dimensional")
        if bounds.size < 2:
            raise ValueError("element_bounds must contain at least two points")
        if np.any(np.diff(bounds) <= 0.0):
            raise ValueError("element_bounds must be strictly increasing")

        left, right = self.domain
        if not np.isclose(bounds[0], left) or not np.isclose(bounds[-1], right):
            raise ValueError("element_bounds must match domain endpoints")

        if len(self.element_specs) != bounds.size - 1:
            raise ValueError("Need one ElementSpec per element")

    @property
    def num_elements(self) -> int:
        return self.element_bounds.size - 1

    @classmethod
    def uniform(
        cls,
        domain: tuple[float, float],
        num_elements: int,
        element_spec: ElementSpec,
    ) -> "Mesh1D":
        if num_elements < 1:
            raise ValueError("num_elements must be >= 1")
        bounds = np.linspace(domain[0], domain[1], num_elements + 1)
        specs = [element_spec for _ in range(num_elements)]
        return cls(domain=domain, element_bounds=bounds, element_specs=specs)

    @classmethod
    def from_bounds(
        cls,
        domain: tuple[float, float],
        element_bounds: np.ndarray,
        element_specs: list[ElementSpec],
    ) -> "Mesh1D":
        return cls(domain=domain, element_bounds=element_bounds, element_specs=element_specs)
