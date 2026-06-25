"""ferray.lib — NumPy `numpy.lib` parity layer.

Combines two sources:

1. **`stride_tricks`** comes from the compiled Rust extension
   (`ferray._ferray.lib.stride_tricks`) and exposes the strided-view
   utilities (`broadcast_arrays`, `broadcast_shapes`,
   `sliding_window_view`).
2. **Functional utilities** (`vectorize`, `apply_along_axis`,
   `apply_over_axes`, `piecewise`) are pure-Python wrappers that
   delegate to `numpy`. These functions in NumPy are *themselves*
   pure-Python wrappers — they exist to apply user-provided Python
   callables, and there's no Rust-side speedup available for that
   shape. We expose them so `from ferray.lib import vectorize`
   matches `from numpy import vectorize`.

If a user wants the Rust-fast equivalent of `apply_along_axis` for a
known kernel, they should reach for the typed function directly
(e.g. `ferray.sin(slice)`) instead of going through this layer.
"""

from __future__ import annotations

import numpy as _np
import sys as _sys

from ferray._ferray import lib as _rust_lib

# Re-export the Rust-side stride_tricks submodule.
stride_tricks = _rust_lib.stride_tricks


def as_strided(x, shape=None, strides=None, subok=False, writeable=True):
    """`numpy.lib.stride_tricks.as_strided(x, shape, strides)`.

    Delegates to NumPy's implementation. ferray's Rust `as_strided`
    uses element strides; NumPy uses byte strides. Rather than expose
    the mismatch, we wrap NumPy's already-correct version — the
    output is a numpy.ndarray view, which is what callers expect.
    """
    return _np.lib.stride_tricks.as_strided(
        x, shape=shape, strides=strides, subok=subok, writeable=writeable
    )


def vectorize(pyfunc, /, *args, **kwargs):
    """`numpy.vectorize(pyfunc, ...)` — return a vectorized function.

    Delegates directly to `numpy.vectorize`. The returned object is
    `numpy.vectorize`, which works on any ferray-produced ndarray
    because they ARE numpy arrays.
    """
    return _np.vectorize(pyfunc, *args, **kwargs)


def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """`numpy.apply_along_axis(func1d, axis, arr, *args, **kwargs)`.

    Calls `func1d` on each 1-D slice of `arr` along `axis`. The Python
    callable is dispatched in Python — there's no Rust speedup
    available for callbacks into user code.
    """
    return _np.apply_along_axis(func1d, axis, arr, *args, **kwargs)


def apply_over_axes(func, a, axes):
    """`numpy.apply_over_axes(func, a, axes)` — apply `func` repeatedly
    along each axis in `axes`."""
    return _np.apply_over_axes(func, a, axes)


def piecewise(x, condlist, funclist, *args, **kwargs):
    """`numpy.piecewise(x, condlist, funclist, *args, **kwargs)`.

    Evaluates a piecewise-defined function: `funclist[i]` is applied
    where `condlist[i]` is true.
    """
    return _np.piecewise(x, condlist, funclist, *args, **kwargs)


Arrayterator = _np.lib.Arrayterator
NumpyVersion = _np.lib.NumpyVersion
add_docstring = _np.lib.add_docstring
add_newdoc = _np.lib.add_newdoc
array_utils = _np.lib.array_utils
format = _np.lib.format
introspect = _np.lib.introspect
mixins = _np.lib.mixins
npyio = _np.lib.npyio
scimath = _np.lib.scimath
tracemalloc_domain = _np.lib.tracemalloc_domain

for _name in ("array_utils", "format", "introspect", "mixins", "npyio", "scimath"):
    _sys.modules[f"{__name__}.{_name}"] = getattr(_np.lib, _name)


__all__ = [
    "Arrayterator",
    "NumpyVersion",
    "add_docstring",
    "add_newdoc",
    "apply_along_axis",
    "apply_over_axes",
    "array_utils",
    "as_strided",
    "format",
    "introspect",
    "mixins",
    "npyio",
    "piecewise",
    "scimath",
    "stride_tricks",
    "tracemalloc_domain",
    "vectorize",
]

del _name, _sys
