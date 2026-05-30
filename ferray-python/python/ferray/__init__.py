"""ferray — a NumPy-equivalent library for Python, powered by Rust.

The goal of this package is **100% API parity with NumPy**, so that
`import ferray as np` is a true drop-in replacement once the rollout
is complete. The Rust core already mirrors the NumPy namespace
function-for-function; the work happening in this module is exposing
it through CPython.

```python
import ferray as np
a = np.zeros((3, 4))            # -> numpy.ndarray, shape (3, 4), dtype float64
b = np.arange(0, 10)            # -> numpy.ndarray, dtype int64
c = np.transpose(np.arange(6).reshape(2, 3))
d = np.sin(np.linspace(0, np.pi, 50))
m = np.mean(d)
```

Coverage is being filled in across phases — see the project README for
the live status of which NumPy functions are bound today.
"""

from ._ferray import (
    __version__,
    # creation
    arange,
    array,
    asanyarray,
    asarray,
    ascontiguousarray,
    asfortranarray,
    copy,
    empty,
    empty_like,
    eye,
    full,
    frombuffer,
    fromfile,
    fromfunction,
    fromiter,
    fromstring,
    full_like,
    geomspace,
    identity,
    linspace,
    logspace,
    meshgrid,
    mgrid,
    ogrid,
    ones,
    ones_like,
    tri,
    vander,
    zeros,
    zeros_like,
    # manipulation
    atleast_1d,
    atleast_2d,
    atleast_3d,
    broadcast_to,
    concatenate,
    diag,
    diagflat,
    dstack,
    expand_dims,
    flatten,
    flip,
    fliplr,
    flipud,
    hstack,
    append,
    array_split,
    block,
    c_,
    column_stack,
    delete,
    dsplit,
    hsplit,
    insert,
    moveaxis,
    pad,
    r_,
    ravel,
    repeat,
    reshape,
    resize,
    roll,
    row_stack,
    split,
    vsplit,
    rollaxis,
    rot90,
    squeeze,
    stack,
    swapaxes,
    tile,
    transpose,
    tril,
    trim_zeros,
    triu,
    vstack,
    # indexing
    argwhere,
    choose,
    compress,
    diag_indices,
    extract,
    flatnonzero,
    indices,
    mask_indices,
    nonzero,
    place,
    put,
    putmask,
    ravel_multi_index,
    select as select_indexing,
    take,
    take_along_axis,
    tril_indices,
    triu_indices,
    unravel_index,
    # ufunc — trig
    arccos,
    arccosh,
    arcsin,
    arcsinh,
    arctan,
    arctan2,
    arctanh,
    cos,
    cosh,
    deg2rad,
    degrees,
    hypot,
    rad2deg,
    radians,
    sin,
    sinh,
    tan,
    tanh,
    # ufunc — exp/log
    exp,
    exp2,
    expm1,
    heaviside,
    log,
    log10,
    log1p,
    log2,
    logaddexp,
    logaddexp2,
    # ufunc — arithmetic / roots / sign
    absolute,
    add,
    cbrt,
    divide,
    fabs,
    float_power,
    floor_divide,
    multiply,
    negative,
    positive,
    power,
    reciprocal,
    remainder,
    sign,
    sqrt,
    square,
    subtract,
    true_divide,
    # ufunc — float intrinsics
    ceil,
    clip,
    copysign,
    fix,
    fmax,
    fmin,
    floor,
    maximum,
    minimum,
    rint,
    round as _round_no_decimals,
    around,
    nan_to_num,
    unwrap,
    trunc,
    # ufunc — predicates
    isfinite,
    isinf,
    isnan,
    isneginf,
    isposinf,
    signbit,
    # ufunc — comparison / logical
    allclose,
    array_equal,
    array_equiv,
    bitwise_and,
    bitwise_count,
    bitwise_not,
    bitwise_or,
    bitwise_xor,
    equal,
    angle,
    conj,
    conjugate,
    imag,
    invert,
    iscomplex,
    iscomplexobj,
    isreal,
    isrealobj,
    left_shift,
    divmod as divmod_arr,
    exp_fast,
    fmod,
    frexp,
    gcd,
    lcm,
    ldexp,
    modf,
    nextafter,
    real,
    real_if_close,
    right_shift,
    convolve,
    correlate,
    ediff1d,
    gradient,
    i0,
    interp,
    sinc,
    spacing,
    trapezoid,
    greater,
    greater_equal,
    isclose,
    less,
    less_equal,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    not_equal,
    # stats — reductions
    all,
    any,
    average,
    argmax,
    argmin,
    cumprod,
    cumsum,
    nancumprod,
    nancumsum,
    nanpercentile,
    nanquantile,
    sort_complex,
    cross,
    trace,
    diff,
    max,
    mean,
    min,
    prod,
    ptp,
    std,
    sum,
    var,
    # stats — nan-aware
    nanargmax,
    nanargmin,
    nanmax,
    nanmean,
    nanmedian,
    nanmin,
    nanprod,
    nanstd,
    nansum,
    nanvar,
    # stats — sort / search / unique / set / where
    argpartition,
    argsort,
    corrcoef,
    count_nonzero,
    cov,
    median,
    percentile,
    quantile,
    bincount,
    digitize,
    einsum,
    histogram,
    histogram2d,
    histogram_bin_edges,
    histogramdd,
    diagonal,
    matrix_transpose,
    matvec,
    vecdot,
    vecmat,
    # top-level vector/matrix products (numpy exposes these at the root)
    dot,
    vdot,
    matmul,
    inner,
    outer,
    tensordot,
    kron,
    in1d,
    lexsort,
    partition,
    unique_extended,
    intersect1d,
    isin,
    searchsorted,
    setdiff1d,
    setxor1d,
    sort,
    union1d,
    unique,
    where_ as where,
    # top-level numpy alias + introspection surface (expansion batch 1)
    ndim,
    shape,
    size,
    isscalar,
    isfortran,
    astype,
    can_cast,
    promote_types,
    result_type,
    min_scalar_type,
    issubdtype,
    isdtype,
    common_type,
)

# numpy-2.0 array-API trig aliases + thin wrappers. numpy itself defines
# these as bare assignments in numpy/_core/__init__.py:138 (`acos =
# numeric.arccos`); ferray mirrors the exact same aliasing against its
# already-bound ufunc/manipulation/reduction surface.
acos = arccos
acosh = arccosh
asin = arcsin
asinh = arcsinh
atan = arctan
atanh = arctanh
atan2 = arctan2
bitwise_invert = invert
bitwise_left_shift = left_shift
bitwise_right_shift = right_shift
concat = concatenate
permute_dims = transpose
pow = power  # noqa: A001 - intentional NumPy compat shadow of builtin pow
amax = max
amin = min
cumulative_sum = cumsum
cumulative_prod = cumprod

# numpy.round / numpy.round_ are aliases of numpy.around (both take a
# `decimals` kwarg). The bare-decimals ufunc binding is kept available as
# `_round_no_decimals` but the public `round` must accept `decimals`, so it
# is the `around` binding. numpy/_core/fromnumeric.py:3754 `round = around`.
round = around  # noqa: A001 - intentional NumPy compat shadow of builtin round
round_ = around

# numpy.divmod ufunc — (floor_divide, mod) tuple, integer + float kernels.
from ._ferray import divmod as _divmod_ufunc

globals()["divmod"] = _divmod_ufunc  # noqa: A001 - NumPy compat shadow of builtin

# `mod` is a Python keyword and cannot be a bare `from ... import mod`, so it
# is bound by attribute. It is the numpy alias of `remainder` (umath.py).
from . import _ferray as _ferray_ext

globals()["mod"] = getattr(_ferray_ext, "mod")
from ._ferray import (
    # Top-level window aliases
    bartlett,
    blackman,
    broadcast_arrays,
    broadcast_shapes,
    hamming,
    hanning,
    kaiser,
)

# Top-level numpy poly1d family — classic 1-D polynomial functions on
# highest-degree-first coefficient arrays (numpy/lib/_polynomial_impl.py).
# Distinct from numpy.polynomial.polynomial.* (lowest-first).
from ._ferray import (
    poly,
    polyadd,
    polyder,
    polydiv,
    polyfit,
    polyint,
    polymul,
    polysub,
    polyval,
    roots,
)

# File I/O: .npy/.npz binary serialization + delimited text. Wired over the
# ferray-io crate (numpy/lib/_npyio_impl.py). `import ferray as np;
# np.save('x.npy', a); np.load('x.npy')` round-trips against numpy.
from ._ferray import (
    save,
    load,
    savez,
    savez_compressed,
    savetxt,
    loadtxt,
    genfromtxt,
)
from . import autodiff as _autodiff_module
from ._ferray import char as _char_module
from ._ferray import strings as _strings_module
import numpy as _np

# numpy.datetime64 / numpy.timedelta64 top-level surface (refs #831).
#
# isnat / datetime_as_string / is_busday / busday_count / busday_offset are
# bound in the Rust extension (src/datetime.rs). isnat and the busday calendar
# run the genuine ferray-ufunc / Rust algorithms; datetime arithmetic flows
# through ferray's own add / subtract, which dispatch datetime64 / timedelta64
# operands to the ferray-ufunc datetime kernels (sub_datetime_promoted /
# add_datetime_timedelta_promoted / add_timedelta_promoted).
from ._ferray import (
    isnat,
    datetime_as_string,
    is_busday,
    busday_count,
    busday_offset,
)


def datetime_data(dtype):
    """``numpy.datetime_data(dtype)`` -> ``(unit, count)``.

    Reads the unit metadata carried on a numpy datetime64 / timedelta64
    ``dtype`` object (e.g. ``datetime_data(dtype('datetime64[D]'))`` ->
    ``('D', 1)``). The metadata lives on the shared numpy dtype, so this
    is pure-Python delegation to ``numpy.datetime_data`` — there is no
    ferray dtype object to read it from
    (numpy/_core/src/multiarray/datetime.c ``datetime_data``).
    """
    return _np.datetime_data(dtype)


def add_datetime(a, b):
    """Add a timedelta to a datetime (or two timedeltas), via ferray's
    datetime-aware ``add`` (ferray-ufunc datetime kernels)."""
    return add(a, b)


def sub_datetime(a, b):
    """Subtract two datetimes (-> timedelta) or a timedelta from a datetime
    (-> datetime), via ferray's datetime-aware ``subtract``."""
    return subtract(a, b)
from ._ferray import dtype as _dtype_module
from ._ferray import emath as _emath_module
from ._ferray import fft as _fft_module
from . import lib as _lib_module
from ._ferray import linalg as _linalg_module
from ._ferray import ma as _ma_module
from ._ferray import polynomial as _polynomial_module
from ._ferray import random as _random_module
from ._ferray import window as _window_module

autodiff = _autodiff_module
char = _char_module
strings = _strings_module
dtype = _dtype_module
emath = _emath_module
# numpy aliases numpy.math to the emath module historically; numpy 2.x keeps
# only numpy.emath. Mirror just `emath` (the supported name).
fft = _fft_module
lib = _lib_module
linalg = _linalg_module
ma = _ma_module
polynomial = _polynomial_module
random = _random_module
window = _window_module

# Register the submodules under `ferray.X` paths in sys.modules so
# `from ferray.linalg import inv` (and the other submodules) works
# alongside attribute access (`ferray.linalg.inv`). Without this
# Python's import machinery only sees the `ferray._ferray.X` paths
# that the Rust extension registered.
import sys as _sys

_sys.modules["ferray.autodiff"] = _autodiff_module
_sys.modules["ferray.char"] = _char_module
_sys.modules["ferray.strings"] = _strings_module
_sys.modules["ferray.dtype"] = _dtype_module
_sys.modules["ferray.emath"] = _emath_module
_sys.modules["ferray.fft"] = _fft_module
_sys.modules["ferray.lib.stride_tricks"] = _lib_module.stride_tricks
_sys.modules["ferray.linalg"] = _linalg_module
_sys.modules["ferray.ma"] = _ma_module
_sys.modules["ferray.polynomial"] = _polynomial_module
_sys.modules["ferray.random"] = _random_module
_sys.modules["ferray.window"] = _window_module
del _sys

# `np.abs` is a long-standing alias for `np.absolute`. The Rust ufunc
# crate's `abs` is the complex-absolute, so we don't bind it from
# there — alias at the Python level instead.
abs = absolute  # noqa: A001 - intentional NumPy compat shadow

# ---------------------------------------------------------------------------
# Top-level array helpers that numpy itself implements in pure Python by
# composing primitives. We mirror that architecture here: each helper is a
# thin composition over already-bound ferray primitives (or numpy's own
# pure helper where the operation is a metadata-only query with no array
# math, e.g. `iterable`, `may_share_memory`). The return objects are
# numpy.ndarray, matching the boundary contract.
# ---------------------------------------------------------------------------


import builtins as _builtins

_builtins_all = _builtins.all


def copyto(dst, src, casting="same_kind", where=True):
    """`numpy.copyto(dst, src, casting='same_kind', where=True)` — copy
    values from `src` into `dst` in place (numpy/_core/multiarray copyto).
    `dst` must be a numpy.ndarray; ferray's array boundary IS numpy.ndarray,
    so the in-place mutation is observed by the caller."""
    return _np.copyto(dst, asarray(src), casting=casting, where=where)


def ix_(*args):
    """`numpy.ix_(*args)` — construct an open mesh from N 1-D sequences so
    that fancy-indexing with the result selects the cross-product
    (numpy/lib/_index_tricks_impl.py def ix_). Implemented by reshaping each
    1-D index array to broadcast along its own axis."""
    out = []
    nd = len(args)
    for k, seq in enumerate(args):
        new = asarray(seq)
        if new.ndim != 1:
            raise ValueError("Cross index must be 1 dimensional")
        if new.dtype == _np.bool_:
            (new,) = nonzero(new)
        shape = [1] * nd
        shape[k] = len(new)
        out.append(reshape(new, shape))
    return tuple(out)


def fill_diagonal(a, val, wrap=False):
    """`numpy.fill_diagonal(a, val, wrap=False)` — write `val` along the
    main diagonal of `a` IN PLACE (numpy/lib/_index_tricks_impl.py
    def fill_diagonal). `a` is a numpy.ndarray at the boundary, so the
    mutation is visible to the caller."""
    return _np.fill_diagonal(a, val, wrap=wrap)


def diag_indices_from(arr):
    """`numpy.diag_indices_from(arr)` — indices of the main diagonal of an
    n-D array with equal-length axes (numpy/lib/_index_tricks_impl.py
    def diag_indices_from). Composes the already-bound `diag_indices`."""
    arr = asarray(arr)
    if arr.ndim < 2:
        raise ValueError("input array must be at least 2-d")
    if not _builtins_all(arr.shape[0] == s for s in arr.shape[1:]):
        raise ValueError("All dimensions of input must be of equal length")
    return diag_indices(arr.shape[0], arr.ndim)


def tril_indices_from(arr, k=0):
    """`numpy.tril_indices_from(arr, k=0)` — indices for the lower triangle
    of a 2-D array (numpy/lib/_twodim_base_impl.py def tril_indices_from).
    Composes the already-bound `tril_indices`."""
    arr = asarray(arr)
    if arr.ndim != 2:
        raise ValueError("input array must be 2-d")
    return tril_indices(arr.shape[0], k, arr.shape[1])


def triu_indices_from(arr, k=0):
    """`numpy.triu_indices_from(arr, k=0)` — indices for the upper triangle
    of a 2-D array (numpy/lib/_twodim_base_impl.py def triu_indices_from).
    Composes the already-bound `triu_indices`."""
    arr = asarray(arr)
    if arr.ndim != 2:
        raise ValueError("input array must be 2-d")
    return triu_indices(arr.shape[0], k, arr.shape[1])


def put_along_axis(arr, indices, values, axis):
    """`numpy.put_along_axis(arr, indices, values, axis)` — write `values`
    into `arr` at the index array along `axis`, IN PLACE
    (numpy/lib/_shape_base_impl.py def put_along_axis)."""
    return _np.put_along_axis(arr, asarray(indices), values, axis)


def asarray_chkfinite(a, dtype=None, order=None):
    """`numpy.asarray_chkfinite(a, dtype=None, order=None)` — like
    `asarray` but raises `ValueError` if any element is NaN or Inf
    (numpy/lib/_function_base_impl.py def asarray_chkfinite)."""
    a = asarray(a, dtype=dtype) if dtype is not None else asarray(a)
    if a.dtype.char in _np.typecodes["AllFloat"] and not _np.isfinite(a).all():
        raise ValueError("array must not contain infs or NaNs")
    return a


def _memview(x):
    # Preserve object identity for ndarrays — coercing via asarray would
    # copy and destroy the shared-memory relationship the query is about.
    return x if isinstance(x, _np.ndarray) else asarray(x)


def may_share_memory(a, b, max_work=None):
    """`numpy.may_share_memory(a, b)` — conservative bool: could `a` and
    `b` share memory? (numpy/_core/multiarray may_share_memory)."""
    if max_work is None:
        return _np.may_share_memory(_memview(a), _memview(b))
    return _np.may_share_memory(_memview(a), _memview(b), max_work=max_work)


def shares_memory(a, b, max_work=None):
    """`numpy.shares_memory(a, b)` — exact bool: do `a` and `b` share any
    memory? (numpy/_core/multiarray shares_memory)."""
    if max_work is None:
        return _np.shares_memory(_memview(a), _memview(b))
    return _np.shares_memory(_memview(a), _memview(b), max_work=max_work)


def iterable(y):
    """`numpy.iterable(y)` — bool: is `y` iterable?
    (numpy/lib/_function_base_impl.py def iterable)."""
    try:
        iter(y)
    except TypeError:
        return False
    return True


import collections as _collections

_UniqueAllResult = _collections.namedtuple(
    "UniqueAllResult", ("values", "indices", "inverse_indices", "counts")
)
_UniqueCountsResult = _collections.namedtuple("UniqueCountsResult", ("values", "counts"))
_UniqueInverseResult = _collections.namedtuple(
    "UniqueInverseResult", ("values", "inverse_indices")
)


def unique_all(x):
    """`numpy.unique_all(x)` (NumPy 2.0 array-API) — values + indices +
    inverse_indices + counts (numpy/lib/_arraysetops_impl.py def
    unique_all). The inverse is reshaped to `x.shape`."""
    x = asarray(x)
    values, indices, inverse, counts = unique_extended(
        x, return_index=True, return_inverse=True, return_counts=True
    )
    return _UniqueAllResult(values, indices, reshape(inverse, x.shape), counts)


def unique_counts(x):
    """`numpy.unique_counts(x)` (NumPy 2.0 array-API) — values + counts
    (numpy/lib/_arraysetops_impl.py def unique_counts)."""
    values, counts = unique_extended(asarray(x), return_counts=True)
    return _UniqueCountsResult(values, counts)


def unique_inverse(x):
    """`numpy.unique_inverse(x)` (NumPy 2.0 array-API) — values +
    inverse_indices reshaped to `x.shape`
    (numpy/lib/_arraysetops_impl.py def unique_inverse)."""
    x = asarray(x)
    values, inverse = unique_extended(x, return_inverse=True)
    return _UniqueInverseResult(values, reshape(inverse, x.shape))


def unique_values(x):
    """`numpy.unique_values(x)` (NumPy 2.0 array-API) — the unique values
    of `x` (numpy/lib/_arraysetops_impl.py def unique_values). The
    array-API leaves the ordering unspecified; ferray returns them
    sorted."""
    return unique_extended(asarray(x))


# ---------------------------------------------------------------------------
# Top-level functional / dtype / meta surface (expansion batch, refs #818).
#
# numpy implements every function in this block in *pure Python* by composing
# array primitives (apply_along_axis, piecewise, vectorize, packbits, ...) or
# as scalar/string/metadata helpers with no array math (base_repr, typename,
# get_printoptions, errstate, ...). We mirror that exact architecture: array
# composables are built on already-bound ferray primitives; metadata helpers
# operate on the numpy dtype/scalar objects that ARE ferray's boundary
# contract. Every helper returns the numpy object numpy returns.
# ---------------------------------------------------------------------------


def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """`numpy.apply_along_axis(func1d, axis, arr, *args, **kwargs)` — apply
    `func1d` to 1-D slices of `arr` along `axis`
    (numpy/lib/_shape_base_impl.py def apply_along_axis). Mirrors numpy's
    documented contract: the `axis` dimension is replaced by the shape of
    `func1d`'s return value."""
    arr = asarray(arr)
    nd = arr.ndim
    if axis < 0:
        axis += nd
    if not (0 <= axis < nd):
        raise _np.exceptions.AxisError(axis, nd)
    # move the iteration axis to the end so each slice is arr_view[ind + (...,)]
    in_dims = list(range(nd))
    arr_view = transpose(arr, in_dims[:axis] + in_dims[axis + 1:] + [axis])
    inds = list(_np.ndindex(arr_view.shape[:-1]))
    if not inds:
        raise ValueError(
            "Cannot apply_along_axis when any iteration dimensions are 0"
        )
    results = [asarray(func1d(arr_view[ind + (slice(None),)], *args, **kwargs))
               for ind in inds]
    res0 = results[0]
    out_inner = res0.shape
    outer = arr_view.shape[:-1]
    buff = _np.empty(outer + out_inner, dtype=res0.dtype)
    for ind, r in zip(inds, results):
        buff[ind] = r
    # buff axes: (outer..., inner...) -> need (Ni..., Nj..., Nk...)
    # outer currently = Ni + Nk (axis removed, was at position `axis`).
    n_outer_pre = axis
    n_inner = res0.ndim
    n_outer = len(outer)
    perm = (list(range(n_outer_pre))
            + list(range(n_outer, n_outer + n_inner))
            + list(range(n_outer_pre, n_outer)))
    return transpose(buff, perm)


def apply_over_axes(func, a, axes):
    """`numpy.apply_over_axes(func, a, axes)` — repeatedly apply
    `func(a, axis)` over each axis in `axes`
    (numpy/lib/_shape_base_impl.py def apply_over_axes). `func` must keep or
    reduce that axis to length 1; the result is reshaped to keep dims when
    the reduction removed the axis."""
    val = asarray(a)
    N = val.ndim
    if _np.isscalar(axes):
        axes = (axes,)
    for axis in axes:
        if axis < 0:
            axis = N + axis
        res = func(val, axis)
        res = asarray(res)
        if res.ndim == val.ndim:
            val = res
        elif res.ndim == val.ndim - 1:
            val = expand_dims(res, axis)
        else:
            raise ValueError(
                "function is not returning an array of the correct shape"
            )
    return val


def piecewise(x, condlist, funclist, *args, **kw):
    """`numpy.piecewise(x, condlist, funclist, *args, **kw)` — evaluate a
    piecewise-defined function (numpy/lib/_function_base_impl.py def
    piecewise). Each entry of `funclist` is applied where the corresponding
    condition in `condlist` is True; a scalar entry is used as a fill value."""
    x = asarray(x)
    condlist = _np.atleast_1d(_np.asarray(condlist, dtype=_np.bool_))
    # A scalar/0-d x with a 1-D condlist is a single boolean mask per func.
    # numpy: if len(condlist) == len(funclist) - 1, an "otherwise" func is
    # implied from the negation of all conditions.
    n = len(condlist)
    n2 = len(funclist)
    if n == n2 - 1:
        condelse = ~_np.any(condlist, axis=0, keepdims=True)
        condlist = _np.concatenate([condlist, condelse], axis=0)
        n += 1
    elif n != n2:
        raise ValueError(
            "with {} condition(s), either {} or {} functions are expected".format(
                n, n, n + 1
            )
        )
    y = _np.zeros_like(x)
    for cond, func in zip(condlist, funclist):
        if not callable(func):
            y[cond] = func
        else:
            vals = x[cond]
            if vals.size > 0:
                y[cond] = func(vals, *args, **kw)
    return y


def select(condlist, choicelist, default=0):
    """`numpy.select(condlist, choicelist, default=0)` — return an array drawn
    from elements in `choicelist` depending on `condlist`
    (numpy/lib/_function_base_impl.py def select). Wires straight to ferray's
    already-bound `select` kernel."""
    return select_indexing(condlist, choicelist, default=default)


def unstack(x, /, *, axis=0):
    """`numpy.unstack(x, /, *, axis=0)` (NumPy 2.1 array-API) — split `x`
    along `axis` into a tuple of arrays with that axis removed
    (numpy/_core/shape_base.py def unstack). Composes ferray's `moveaxis`."""
    x = asarray(x)
    if x.ndim == 0:
        raise ValueError("Input array must be at least 1-d.")
    y = moveaxis(x, axis, 0)
    return tuple(y[i] for i in range(y.shape[0]))


def require(a, dtype=None, requirements=None, *, like=None):
    """`numpy.require(a, dtype=None, requirements=None)` — return an ndarray
    of `a` satisfying the layout/dtype `requirements`
    (numpy/_core/_asarray.py def require). Delegates to numpy's pure-Python
    requirement resolver over ferray's asarray boundary (which is
    numpy.ndarray)."""
    return _np.require(asarray(a) if not isinstance(a, _np.ndarray) else a,
                       dtype=dtype, requirements=requirements)


class vectorize(_np.vectorize):
    """`numpy.vectorize` — wrap a Python callable so it broadcasts over array
    inputs (numpy/lib/_function_base_impl.py class vectorize). numpy itself
    implements this entirely in Python; ferray inherits that pure-Python
    machinery so the elementwise-Python-callable contract is identical, then
    returns ferray-boundary numpy arrays."""

    __module__ = "ferray"


def frompyfunc(func, /, nin, nout, **kwargs):
    """`numpy.frompyfunc(func, nin, nout)` — turn a Python function into a
    ufunc-like object operating elementwise over object arrays
    (numpy/_core/umath frompyfunc). This is CPython object-array machinery
    with no ferray numeric kernel; we expose numpy's implementation so the
    elementwise-callable surface exists at `ferray.frompyfunc`."""
    return _np.frompyfunc(func, nin, nout, **kwargs)


def base_repr(number, base=2, padding=0):
    """`numpy.base_repr(number, base=2, padding=0)` — string representation of
    an integer in the given base (numpy/_core/numeric.py def base_repr).
    Pure-Python integer→string, exact."""
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if base > len(digits):
        raise ValueError("Bases greater than 36 not handled in base_repr.")
    elif base < 2:
        raise ValueError("Bases less than 2 not handled in base_repr.")
    num = abs(int(number))
    res = []
    while num:
        res.append(digits[num % base])
        num //= base
    if padding:
        res.append("0" * padding)
    if int(number) < 0:
        res.append("-")
    return "".join(reversed(res or "0"))


def binary_repr(num, width=None):
    """`numpy.binary_repr(num, width=None)` — binary string of an integer,
    two's-complement when `width` is given for negatives
    (numpy/_core/numeric.py def binary_repr)."""
    return _np.binary_repr(int(num), width=width)


# --- dtype / typecode helpers -------------------------------------------------

def mintypecode(typechars, typeset="GDFgdf", default="d"):
    """`numpy.mintypecode(typechars, typeset='GDFgdf', default='d')` — the
    smallest-precision typecode able to safely hold all of `typechars`
    (numpy/lib/_type_check_impl.py def mintypecode)."""
    return _np.mintypecode(typechars, typeset=typeset, default=default)


def typename(char):
    """`numpy.typename(char)` — the english description of a typecode char
    (numpy/_core/numerictypes.py def typename)."""
    return _np.typename(char)


def finfo(dtype):
    """`numpy.finfo(dtype)` — machine limits for floating-point types
    (numpy/_core/getlimits.py class finfo). ferray's dtype boundary IS
    numpy's dtype, so the limits object is shared."""
    return _np.finfo(dtype)


def iinfo(int_type):
    """`numpy.iinfo(int_type)` — machine limits for integer types
    (numpy/_core/getlimits.py class iinfo)."""
    return _np.iinfo(int_type)


# --- bit packing --------------------------------------------------------------

def packbits(a, /, axis=None, bitorder="big"):
    """`numpy.packbits(a, axis=None, bitorder='big')` — pack the elements of a
    binary-valued array into bits in a uint8 array
    (numpy/_core/multiarray packbits). Operates on the numpy.ndarray
    boundary."""
    return _np.packbits(asarray(a), axis=axis, bitorder=bitorder)


def unpackbits(a, /, axis=None, count=None, bitorder="big"):
    """`numpy.unpackbits(a, axis=None, count=None, bitorder='big')` — unpack
    the bits of a uint8 array into a binary-valued output
    (numpy/_core/multiarray unpackbits)."""
    return _np.unpackbits(_np.asarray(a, dtype=_np.uint8), axis=axis,
                          count=count, bitorder=bitorder)


# --- float formatting ---------------------------------------------------------

def format_float_positional(x, precision=None, unique=True, fractional=True,
                            trim="k", sign=False, pad_left=None, pad_right=None,
                            min_digits=None):
    """`numpy.format_float_positional(...)` — positional (non-scientific)
    string of a scalar float (numpy/_core/arrayprint.py def
    format_float_positional)."""
    return _np.format_float_positional(
        x, precision=precision, unique=unique, fractional=fractional, trim=trim,
        sign=sign, pad_left=pad_left, pad_right=pad_right, min_digits=min_digits)


def format_float_scientific(x, precision=None, unique=True, trim="k",
                            sign=False, pad_left=None, exp_digits=None,
                            min_digits=None):
    """`numpy.format_float_scientific(...)` — scientific-notation string of a
    scalar float (numpy/_core/arrayprint.py def format_float_scientific)."""
    return _np.format_float_scientific(
        x, precision=precision, unique=unique, trim=trim, sign=sign,
        pad_left=pad_left, exp_digits=exp_digits, min_digits=min_digits)


# --- array string representation ---------------------------------------------

def array2string(a, *args, **kwargs):
    """`numpy.array2string(a, ...)` — the formatted string of an array
    (numpy/_core/arrayprint.py def array2string). Operates on the
    numpy.ndarray boundary that ferray arrays already are."""
    return _np.array2string(asarray(a), *args, **kwargs)


def array_repr(arr, max_line_width=None, precision=None, suppress_small=None):
    """`numpy.array_repr(arr, ...)` — the `repr()` string of an array
    (numpy/_core/arrayprint.py def array_repr)."""
    return _np.array_repr(asarray(arr), max_line_width=max_line_width,
                          precision=precision, suppress_small=suppress_small)


def array_str(a, max_line_width=None, precision=None, suppress_small=None):
    """`numpy.array_str(a, ...)` — the `str()` string of an array
    (numpy/_core/arrayprint.py def array_str)."""
    return _np.array_str(asarray(a), max_line_width=max_line_width,
                         precision=precision, suppress_small=suppress_small)


# --- print options / floating-point error state (config infrastructure) ------
# These manage numpy's global formatting/FP-error state. ferray arrays ARE
# numpy arrays at the boundary, so sharing numpy's state machinery is the
# correct (and only coherent) behavior — a ferray-local copy would not affect
# how the boundary numpy.ndarray prints or how its FP errors are handled.

def get_printoptions():
    """`numpy.get_printoptions()` — current array-print options dict
    (numpy/_core/arrayprint.py def get_printoptions)."""
    return _np.get_printoptions()


def set_printoptions(*args, **kwargs):
    """`numpy.set_printoptions(...)` — set array-print options
    (numpy/_core/arrayprint.py def set_printoptions)."""
    return _np.set_printoptions(*args, **kwargs)


printoptions = _np.printoptions
errstate = _np.errstate


def geterr():
    """`numpy.geterr()` — current floating-point error-handling state
    (numpy/_core/_ufunc_config.py def geterr)."""
    return _np.geterr()


def seterr(**kwargs):
    """`numpy.seterr(...)` — set floating-point error-handling mode
    (numpy/_core/_ufunc_config.py def seterr)."""
    return _np.seterr(**kwargs)


def geterrcall():
    """`numpy.geterrcall()` — current FP-error callback
    (numpy/_core/_ufunc_config.py def geterrcall)."""
    return _np.geterrcall()


def seterrcall(func):
    """`numpy.seterrcall(func)` — set the FP-error callback
    (numpy/_core/_ufunc_config.py def seterrcall)."""
    return _np.seterrcall(func)


def getbufsize():
    """`numpy.getbufsize()` — ufunc buffer size
    (numpy/_core/_ufunc_config.py def getbufsize)."""
    return _np.getbufsize()


def setbufsize(size):
    """`numpy.setbufsize(size)` — set the ufunc buffer size
    (numpy/_core/_ufunc_config.py def setbufsize)."""
    return _np.setbufsize(size)


# --- meta / introspection -----------------------------------------------------

def get_include():
    """`numpy.get_include()` — directory of C header files. ferray exposes its
    numpy-compatible array boundary, so the include path that compiles against
    that ABI is numpy's own (numpy/lib/_utils_impl.py def get_include)."""
    return _np.get_include()


def show_config(mode="stdout"):
    """`numpy.show_config(mode='stdout')` — print/return build configuration.
    ferray's numeric core is the Rust workspace; the array boundary ABI is
    numpy's (numpy/_core/_utils def show_config)."""
    return _np.show_config(mode)


def show_runtime():
    """`numpy.show_runtime()` — print runtime/CPU-dispatch info
    (numpy/lib/_utils_impl.py def show_runtime)."""
    return _np.show_runtime()


def info(object=None, maxwidth=76, output=None, toplevel="ferray"):
    """`numpy.info(object, ...)` — print documentation for an object
    (numpy/lib/_utils_impl.py def info)."""
    return _np.info(object, maxwidth=maxwidth, output=output, toplevel=toplevel)


def test(*args, **kwargs):
    """`numpy.test(...)` — run the test suite. ferray's pytest suite lives at
    `ferray-python/tests`; this meta-hook exists for numpy-API compatibility
    and reports that the harness is external."""
    raise NotImplementedError(
        "ferray.test() — run the ferray test suite via "
        "`pytest ferray-python/tests`"
    )


def einsum_path(*operands, optimize="greedy", einsum_call=False):
    """`numpy.einsum_path(*operands, optimize='greedy')` — evaluate the
    optimal contraction order for an einsum expression
    (numpy/_core/einsumfunc.py def einsum_path). Pure-Python contraction
    planner that complements ferray's bound `einsum`."""
    return _np.einsum_path(*operands, optimize=optimize, einsum_call=einsum_call)


# ---------------------------------------------------------------------------
# NumPy dtype-scalar type objects + array-protocol iterator/info classes
# (expansion, refs #818).
#
# These names are the SHARED dtype/protocol vocabulary of the array boundary,
# NOT computation that ferray translates. ferray arrays ARE numpy.ndarray at
# the Python boundary (they marshal through the numpy crate), and ferray's
# creation/dtype functions already accept and produce numpy dtypes. For a
# true drop-in (`import ferray as np`), the dtype-scalar TYPE OBJECTS
# (np.float64, np.int8, np.complex128, np.bool_, ...) and the array-protocol
# classes (np.ndarray, np.nditer, np.ndindex, np.ndenumerate, np.broadcast,
# np.flatiter, np.poly1d) must exist on `ferray` and BE numpy's — exactly as
# ferray.dtype already re-exposes numpy's dtype constants. Binding each to the
# numpy object is the correct drop-in behavior: `fr.float64 is np.float64`,
# `isinstance(fr.zeros(3), fr.ndarray)`, `list(fr.ndindex(2,2))` all match
# numpy because the boundary object IS the numpy object.
# (numpy/_core/numerictypes.py — scalar type registration;
#  numpy/__init__.pyi — public type-object surface.)
# ---------------------------------------------------------------------------

# Concrete dtype-scalar types (sized + C-named aliases).
bool = _np.bool_  # noqa: A001 - NumPy compat: np.bool is the numpy bool scalar
bool_ = _np.bool_
byte = _np.byte
bytes_ = _np.bytes_
cdouble = _np.cdouble
character = _np.character
clongdouble = _np.clongdouble
complex64 = _np.complex64
complex128 = _np.complex128
complex256 = _np.complex256
csingle = _np.csingle
datetime64 = _np.datetime64
double = _np.double
float16 = _np.float16
float32 = _np.float32
float64 = _np.float64
float128 = _np.float128
half = _np.half
int8 = _np.int8
int16 = _np.int16
int32 = _np.int32
int64 = _np.int64
int_ = _np.int_
intc = _np.intc
intp = _np.intp
long = _np.long
longdouble = _np.longdouble
longlong = _np.longlong
object_ = _np.object_
short = _np.short
single = _np.single
str_ = _np.str_
timedelta64 = _np.timedelta64
ubyte = _np.ubyte
uint = _np.uint
uint8 = _np.uint8
uint16 = _np.uint16
uint32 = _np.uint32
uint64 = _np.uint64
uintc = _np.uintc
uintp = _np.uintp
ulong = _np.ulong
ulonglong = _np.ulonglong
ushort = _np.ushort
void = _np.void

# Abstract scalar-type bases (the numpy type hierarchy).
generic = _np.generic
number = _np.number
integer = _np.integer
signedinteger = _np.signedinteger
unsignedinteger = _np.unsignedinteger
inexact = _np.inexact
floating = _np.floating
complexfloating = _np.complexfloating
flexible = _np.flexible

# Array type + array-protocol iterator/info classes. ferray arrays ARE
# numpy.ndarray, so these operate on ferray arrays directly.
ndarray = _np.ndarray
broadcast = _np.broadcast
nditer = _np.nditer
ndindex = _np.ndindex
ndenumerate = _np.ndenumerate
flatiter = _np.flatiter

# numpy.poly1d — the classic 1-D polynomial CLASS (highest-degree-first
# coefficients), complementing ferray's already-bound poly* functions. numpy
# implements poly1d in pure Python over coefficient ndarrays
# (numpy/lib/_polynomial_impl.py class poly1d); ferray arrays are those
# numpy.ndarray coefficient arrays, so the class is exposed directly.
poly1d = _np.poly1d


__all__ = [
    "__version__",
    # submodules
    "autodiff", "char", "strings", "emath", "fft", "lib", "linalg", "ma", "polynomial", "random", "window",
    # top-level windows
    "bartlett", "blackman", "broadcast_arrays", "broadcast_shapes",
    "hamming", "hanning", "kaiser",
    # top-level poly1d family (highest-degree-first)
    "poly", "polyadd", "polyder", "polydiv", "polyfit", "polyint",
    "polymul", "polysub", "polyval", "roots",
    # file I/O (.npy/.npz binary + delimited text)
    "save", "load", "savez", "savez_compressed", "savetxt", "loadtxt",
    "genfromtxt",
    # creation
    "abs", "arange", "array", "asanyarray", "asarray", "ascontiguousarray",
    "asfortranarray", "copy", "dtype", "empty", "empty_like", "eye", "full", "full_like",
    "frombuffer", "fromfile", "fromfunction", "fromiter", "fromstring",
    "geomspace", "identity", "linspace", "logspace", "meshgrid", "mgrid",
    "ogrid", "ones", "ones_like", "tri", "vander", "zeros", "zeros_like",
    # manipulation
    "atleast_1d", "atleast_2d", "atleast_3d", "broadcast_to", "concatenate",
    "diag", "diagflat", "dstack", "expand_dims", "flatten", "flip", "fliplr",
    "flipud", "hstack", "moveaxis", "ravel", "reshape", "roll", "rollaxis",
    "rot90", "squeeze", "stack", "swapaxes", "transpose", "tril", "triu", "vstack",
    "append", "delete", "insert", "pad", "repeat", "resize", "tile", "trim_zeros",
    "block", "c_", "column_stack", "r_", "row_stack",
    "array_split", "dsplit", "hsplit", "split", "vsplit",
    # indexing
    "argwhere", "diag_indices", "flatnonzero", "indices", "mask_indices",
    "nonzero", "ravel_multi_index", "tril_indices", "triu_indices", "unravel_index",
    "choose", "compress", "extract", "place", "put", "putmask", "take", "take_along_axis",
    "select_indexing",
    # ufunc
    "arccos", "arccosh", "arcsin", "arcsinh", "arctan", "arctan2", "arctanh",
    "cos", "cosh", "deg2rad", "degrees", "hypot", "rad2deg", "radians", "sin",
    "sinh", "tan", "tanh", "exp", "exp2", "expm1", "heaviside", "log", "log10",
    "log1p", "log2", "logaddexp", "logaddexp2", "absolute", "add", "cbrt",
    "divide", "fabs", "multiply", "negative", "positive", "power", "reciprocal",
    "remainder", "mod", "true_divide", "floor_divide", "float_power",
    "sign", "sqrt", "square", "subtract", "ceil", "clip", "copysign", "fix",
    "fmax", "fmin", "floor", "maximum", "minimum", "rint", "round", "round_",
    "around", "nan_to_num", "unwrap", "trunc",
    "isfinite", "isinf", "isnan", "isneginf", "isposinf", "signbit", "equal",
    "greater", "greater_equal", "less", "less_equal", "logical_and",
    "logical_not", "logical_or", "logical_xor", "not_equal",
    "allclose", "array_equal", "array_equiv", "isclose",
    "bitwise_and", "bitwise_count", "bitwise_not", "bitwise_or", "bitwise_xor",
    "invert", "left_shift", "right_shift",
    "angle", "conj", "conjugate", "imag", "iscomplex", "iscomplexobj",
    "isreal", "isrealobj", "real", "real_if_close",
    "divmod_arr", "exp_fast", "fmod", "frexp", "gcd", "lcm", "ldexp",
    "modf", "nextafter", "spacing",
    "convolve", "correlate", "ediff1d", "gradient", "i0", "interp", "sinc", "trapezoid",
    "add_datetime", "isnat", "sub_datetime",
    "datetime_data", "datetime_as_string", "is_busday", "busday_count",
    "busday_offset",
    # stats
    "all", "any", "average", "nancumprod", "nancumsum",
    "nanpercentile", "nanquantile", "sort_complex", "cross", "trace",
    "argmax", "argmin", "cumprod", "cumsum", "diff", "max", "mean", "min",
    "prod", "ptp", "std", "sum", "var", "nanargmax", "nanargmin", "nanmax",
    "nanmean", "nanmedian", "nanmin", "nanprod", "nanstd", "nansum", "nanvar",
    "argsort", "count_nonzero", "in1d", "intersect1d", "isin", "searchsorted",
    "setdiff1d", "setxor1d", "sort", "union1d", "unique", "where",
    "argpartition", "lexsort", "partition", "unique_extended",
    "corrcoef", "cov", "median", "percentile", "quantile",
    "bincount", "digitize", "histogram", "histogram2d", "histogram_bin_edges", "histogramdd",
    "einsum", "matvec", "vecdot", "vecmat",
    "diagonal", "matrix_transpose",
    "dot", "vdot", "matmul", "inner", "outer", "tensordot", "kron",
    # top-level numpy aliases + introspection (expansion batch 1)
    "acos", "acosh", "asin", "asinh", "atan", "atanh", "atan2",
    "bitwise_invert", "bitwise_left_shift", "bitwise_right_shift",
    "concat", "permute_dims", "pow", "amax", "amin",
    "cumulative_sum", "cumulative_prod", "divmod",
    "ndim", "shape", "size", "isscalar", "isfortran",
    "astype", "can_cast", "promote_types", "result_type",
    "min_scalar_type", "issubdtype", "isdtype", "common_type",
    # top-level array helpers (expansion: linalg + array, refs #818)
    "copyto", "ix_", "fill_diagonal", "diag_indices_from", "tril_indices_from",
    "triu_indices_from", "put_along_axis", "asarray_chkfinite",
    "may_share_memory", "shares_memory", "iterable",
    "unique_all", "unique_counts", "unique_inverse", "unique_values",
    # top-level functional / dtype / meta surface (expansion, refs #818)
    "apply_along_axis", "apply_over_axes", "piecewise", "select", "unstack",
    "require", "vectorize", "frompyfunc", "base_repr", "binary_repr",
    "mintypecode", "typename", "finfo", "iinfo", "packbits", "unpackbits",
    "format_float_positional", "format_float_scientific",
    "array2string", "array_repr", "array_str",
    "get_printoptions", "set_printoptions", "printoptions", "errstate",
    "geterr", "seterr", "geterrcall", "seterrcall", "getbufsize", "setbufsize",
    "get_include", "show_config", "show_runtime", "info", "test", "einsum_path",
    # dtype-scalar type objects + abstract bases (expansion, refs #818)
    "bool", "bool_", "byte", "bytes_", "cdouble", "character", "clongdouble",
    "complex64", "complex128", "complex256", "csingle", "datetime64", "double",
    "float16", "float32", "float64", "float128", "half",
    "int8", "int16", "int32", "int64", "int_", "intc", "intp",
    "long", "longdouble", "longlong", "object_", "short", "single", "str_",
    "timedelta64", "ubyte", "uint", "uint8", "uint16", "uint32", "uint64",
    "uintc", "uintp", "ulong", "ulonglong", "ushort", "void",
    "generic", "number", "integer", "signedinteger", "unsignedinteger",
    "inexact", "floating", "complexfloating", "flexible",
    # array type + array-protocol classes (expansion, refs #818)
    "ndarray", "broadcast", "nditer", "ndindex", "ndenumerate", "flatiter",
    "poly1d",
]
