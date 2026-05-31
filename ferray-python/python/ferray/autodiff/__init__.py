"""ferray.autodiff — forward-mode automatic differentiation.

Re-exports the Rust-backed `DualNumber` class and elementary functions
from `ferray._ferray.autodiff`, plus pure-Python helpers
(`derivative`, `gradient`, `jacobian`) that orchestrate the underlying
class for the common use cases.

```python
from ferray.autodiff import DualNumber, derivative, sin, exp

# 1. Direct dual-number arithmetic
x = DualNumber.variable(2.0)        # real=2, dual=1
y = sin(x) * exp(x)
y.dual                               # = cos(2)*e^2 + sin(2)*e^2

# 2. Helper API for scalar functions
def f(x):
    return sin(x) * exp(x)

derivative(f, 2.0)                   # same value as y.dual above
```
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as _np

from ferray._ferray.autodiff import (
    DualNumber,
    abs,
    acos,
    asin,
    atan,
    atan2,
    cos,
    cosh,
    exp,
    ln,
    log10,
    log2,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
)


def derivative(f: Callable[[DualNumber], DualNumber], x: float) -> float:
    """Compute df/dx at `x` using forward-mode autodiff.

    `f` must accept a `DualNumber` and return a `DualNumber`. The
    function's body should use `ferray.autodiff.{sin,cos,exp,...}`
    (or the dual-number arithmetic dunders) so derivatives propagate.
    """
    seed = DualNumber.variable(float(x))
    return float(f(seed).dual)


def gradient(
    f: Callable[[Sequence[DualNumber]], DualNumber],
    point: Sequence[float],
) -> _np.ndarray:
    """Compute the gradient of `f` at `point`.

    `f` takes a sequence of `DualNumber` (one per input dimension) and
    returns a single `DualNumber`. We seed each input as a variable in
    turn while holding the others as constants — `n` evaluations for
    an `n`-dimensional input.
    """
    point = list(point)
    n = len(point)
    grad = _np.empty(n, dtype=_np.float64)
    for i in range(n):
        args = [
            DualNumber(p, 1.0 if i == j else 0.0) for j, p in enumerate(point)
        ]
        grad[i] = f(args).dual
    return grad


def jacobian(
    f: Callable[[Sequence[DualNumber]], Sequence[DualNumber]],
    point: Sequence[float],
) -> _np.ndarray:
    """Compute the Jacobian of `f` (vector-valued) at `point`.

    Returns an `(m, n)` matrix where `m` is the output dimension and
    `n` is `len(point)`. `f` must return a sequence of `DualNumber`s.
    """
    point = list(point)
    n = len(point)
    # Probe with all-zero duals first to learn the output dimension.
    probe = f([DualNumber.constant(p) for p in point])
    m = len(probe)
    jac = _np.empty((m, n), dtype=_np.float64)
    for i in range(n):
        args = [
            DualNumber(p, 1.0 if i == j else 0.0) for j, p in enumerate(point)
        ]
        out = f(args)
        for k, dual in enumerate(out):
            jac[k, i] = dual.dual
    return jac


__all__ = [
    "DualNumber",
    "abs",
    "acos",
    "asin",
    "atan",
    "atan2",
    "cos",
    "cosh",
    "derivative",
    "exp",
    "gradient",
    "jacobian",
    "ln",
    "log10",
    "log2",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
]
