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
from . import autodiff as _autodiff_module
from ._ferray import char as _char_module
from ._ferray import strings as _strings_module
import numpy as _np

# numpy.datetime64 / numpy.timedelta64 arithmetic and predicates work
# natively on numpy ndarrays via operator overloading. Re-expose the
# canonical names at top-level so `ferray.isnat(arr)` works alongside
# `np.isnat(arr)`. For full datetime parity see numpy.datetime64 docs.
isnat = _np.isnat


def add_datetime(a, b):
    """Add a timedelta to a datetime (or vice versa). Routes through
    numpy's operator overloading on datetime64/timedelta64 arrays."""
    return _np.add(a, b)


def sub_datetime(a, b):
    """Subtract two datetimes (returns timedelta) or a timedelta from a
    datetime (returns datetime)."""
    return _np.subtract(a, b)
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
]
