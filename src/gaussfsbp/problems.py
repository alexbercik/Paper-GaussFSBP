from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


ArrayFun = Callable[[np.ndarray], np.ndarray]
ScalarFun = Callable[[float], float]


@dataclass(frozen=True)
class Problem:
    a_fun: ArrayFun
    b_fun: ArrayFun
    f_fun: ArrayFun
    exact_fun: ArrayFun
    left_bc_fun: ScalarFun | None = None
