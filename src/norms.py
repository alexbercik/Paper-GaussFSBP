from __future__ import annotations

import numpy as np

from .elements import Element1D
from .solve import split_global_vector


def _split(elements: list[Element1D], u: np.ndarray) -> list[np.ndarray]:
    sizes = [element.x.size for element in elements]
    return split_global_vector(u, sizes)


def global_H_norm(elements: list[Element1D], u: np.ndarray) -> float:
    local = _split(elements, u)
    val = 0.0
    for element, u_local in zip(elements, local):
        val += float(np.sum(element.H * u_local * u_local))
    return float(np.sqrt(val))


def global_weighted_energy(elements: list[Element1D], u: np.ndarray) -> float:
    local = _split(elements, u)
    val = 0.0
    for element, u_local in zip(elements, local):
        weights = (element.b / element.a) * element.H
        val += float(np.sum(weights * u_local * u_local))
    return float(val)


def global_L2_error(elements: list[Element1D], u: np.ndarray, exact) -> float:
    local = _split(elements, u)
    val = 0.0
    for element, u_local in zip(elements, local):
        err = u_local - np.asarray(exact(element.x), dtype=float)
        val += float(np.sum(element.H * err * err))
    return float(np.sqrt(val))


def global_H_error(elements: list[Element1D], u: np.ndarray, exact) -> float:
    return global_L2_error(elements, u, exact)


def convergence_rate(errors: np.ndarray, hs: np.ndarray) -> np.ndarray:
    errors = np.asarray(errors, dtype=float)
    hs = np.asarray(hs, dtype=float)
    if errors.size != hs.size:
        raise ValueError("errors and hs must have the same length")
    rates = np.full(errors.shape, np.nan, dtype=float)
    for i in range(1, errors.size):
        rates[i] = np.log(errors[i - 1] / errors[i]) / np.log(hs[i - 1] / hs[i])
    return rates
