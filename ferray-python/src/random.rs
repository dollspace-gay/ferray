//! Bindings for the `numpy.random` submodule (legacy convenience API).
//!
//! Each binding routes through a thread-local
//! `Generator<Xoshiro256StarStar>`. This matches NumPy's legacy
//! module-level functions (`numpy.random.random`, `numpy.random.normal`,
//! …) which are implicitly backed by a process-wide `RandomState`.
//!
//! The modern `Generator`-class API (`numpy.random.default_rng()`) is
//! exposed via [`PyGenerator`], a PyO3 `#[pyclass]` wrapper around
//! `ferray_random::Generator<B>` carrying the same distribution +
//! sampling method table as the module-level functions.
//!
//! `seed(s)` resets the thread-local generator to a fresh
//! `default_rng_seeded(s)`. Calling `seed` from one thread does not
//! affect any other thread — same semantics as Python's `random.seed`
//! when used with `threading`.

use std::cell::RefCell;

use ferray_core::array::aliases::{Array1, ArrayD};
use ferray_numpy_interop::{AsFerray, IntoNumPy};
use ferray_random::{Generator, SeedSequence, Xoshiro256StarStar, default_rng, default_rng_seeded};
use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::conv::{as_ndarray, dtype_name, ferr_to_pyerr, scalarize};
use crate::match_dtype_all;

thread_local! {
    /// Thread-local convenience generator backing the module-level
    /// functions. Initialised lazily from OS entropy.
    static RNG: RefCell<Generator<Xoshiro256StarStar>> = RefCell::new(default_rng());
}

fn with_rng<F, R>(f: F) -> R
where
    F: FnOnce(&mut Generator<Xoshiro256StarStar>) -> R,
{
    RNG.with(|cell| f(&mut cell.borrow_mut()))
}

// ---------------------------------------------------------------------------
// Shape coercion: int or sequence of ints
// ---------------------------------------------------------------------------

fn shape_from_pyany(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<usize>> {
    match obj {
        None => Ok(vec![]),
        Some(o) => {
            // Bind the size *signed* and validate non-negativity here so a
            // negative dim surfaces numpy's `ValueError("negative
            // dimensions are not allowed")` (R-DEV-2), not the PyO3
            // `OverflowError`/`TypeError` a `usize` extraction would raise
            // before ferray's own check runs. `random(-1)` ->
            // `ValueError`, `random((2, -1))` -> `ValueError`.
            let signed: Vec<isize> = if let Ok(n) = o.extract::<isize>() {
                vec![n]
            } else {
                o.extract::<Vec<isize>>().map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(
                        "size must be an int, a tuple of ints, or None",
                    )
                })?
            };
            let mut out = Vec::with_capacity(signed.len());
            for d in signed {
                if d < 0 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "negative dimensions are not allowed",
                    ));
                }
                out.push(d as usize);
            }
            Ok(out)
        }
    }
}

/// Construct a thread-local-equivalent `Generator` seeded from an
/// arbitrary numpy-compatible seed object: `None` (OS entropy), an `int`,
/// or an `array_like[int]` / sequence (folded via [`SeedSequence`]).
///
/// numpy's `default_rng(seed=...)` accepts
/// `None | int | array_like[int] | SeedSequence | BitGenerator`
/// (`numpy/random/_generator.pyi:758`). ferray maps a scalar int directly
/// and mixes a sequence's entropy through `SeedSequence` to derive a
/// single deterministic seed for the Xoshiro256** stream — the CONTRACT
/// here is "a list seed is accepted and reproducible", not bit-exact
/// agreement with PCG64 (different PRNG).
fn seed_generator_from(seed: Option<&Bound<'_, PyAny>>) -> PyResult<Generator<Xoshiro256StarStar>> {
    match seed {
        None => Ok(default_rng()),
        Some(o) => {
            if o.is_none() {
                return Ok(default_rng());
            }
            if let Ok(s) = o.extract::<u64>() {
                return Ok(default_rng_seeded(s));
            }
            // array_like / sequence seed: fold the entropy words through a
            // SeedSequence to produce a single deterministic u64.
            let words: Vec<u64> = o.extract::<Vec<u64>>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "seed must be None, an int, or a sequence of ints",
                )
            })?;
            let mut entropy: u64 = 0;
            for (i, w) in words.iter().enumerate() {
                // Mix each word with a position-dependent multiplier so
                // permutations of the same set seed differently.
                entropy ^= w
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .rotate_left((i as u32 % 64) + 1);
            }
            let ss = SeedSequence::new(entropy);
            Ok(default_rng_seeded(ss.generate_u64()))
        }
    }
}

// ---------------------------------------------------------------------------
// Seed control
// ---------------------------------------------------------------------------

/// `numpy.random.seed(s)` — re-seed the thread-local RNG.
#[pyfunction]
pub fn seed(s: u64) {
    RNG.with(|cell| {
        *cell.borrow_mut() = default_rng_seeded(s);
    });
}

// ---------------------------------------------------------------------------
// Continuous distributions
// ---------------------------------------------------------------------------

/// `numpy.random.random(size=None)` — uniform `[0, 1)` of given shape.
/// `size=None` returns a Python float (not an ndarray) — matching
/// numpy's legacy behaviour.
#[pyfunction]
#[pyo3(signature = (size = None))]
pub fn random<'py>(
    py: Python<'py>,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    if size.is_none() {
        let v: f64 = with_rng(|rng| {
            // Generator owns no `next_f64` shortcut on the public surface,
            // so use the 1-element `random` and unwrap.
            rng.random(1usize)
                .map(|a| a.iter().next().copied().unwrap_or(0.0))
        })
        .map_err(ferr_to_pyerr)?;
        return Ok(v.into_pyobject(py)?.into_any());
    }
    let shape = shape_from_pyany(size)?;
    let arr: ArrayD<f64> = with_rng(|rng| rng.random(shape.as_slice())).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// Alias `rand(*shape)` — varargs form of `random`.
#[pyfunction]
#[pyo3(signature = (*shape))]
pub fn rand<'py>(
    py: Python<'py>,
    shape: &Bound<'py, pyo3::types::PyTuple>,
) -> PyResult<Bound<'py, PyAny>> {
    if shape.is_empty() {
        let v: f64 = with_rng(|rng| {
            rng.random(1usize)
                .map(|a| a.iter().next().copied().unwrap_or(0.0))
        })
        .map_err(ferr_to_pyerr)?;
        return Ok(v.into_pyobject(py)?.into_any());
    }
    let dims = varargs_dims(shape)?;
    let arr: ArrayD<f64> = with_rng(|rng| rng.random(dims.as_slice())).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// Extract varargs `*shape` dims as a `usize` vector, mapping a negative
/// dimension to numpy's `ValueError("negative dimensions are not
/// allowed")` rather than the PyO3 `OverflowError` a direct `Vec<usize>`
/// extraction would raise (`np.random.rand(-1)` -> `ValueError`; R-DEV-2).
fn varargs_dims(shape: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Vec<usize>> {
    let signed: Vec<isize> = shape.extract()?;
    let mut out = Vec::with_capacity(signed.len());
    for d in signed {
        if d < 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "negative dimensions are not allowed",
            ));
        }
        out.push(d as usize);
    }
    Ok(out)
}

/// `numpy.random.random_sample(size=None)` — canonical legacy name that
/// `random` aliases. Listed in `numpy/random/__init__.py:159`'s `__all__`.
#[pyfunction]
#[pyo3(signature = (size = None))]
pub fn random_sample<'py>(
    py: Python<'py>,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    random(py, size)
}

/// `numpy.random.sample(size=None)` — legacy alias of `random_sample`
/// (`numpy/random/__init__.py:162`).
#[pyfunction]
#[pyo3(signature = (size = None))]
pub fn sample<'py>(
    py: Python<'py>,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    random(py, size)
}

/// `numpy.random.ranf(size=None)` — legacy alias of `random_sample`
/// (`numpy/random/__init__.py:160`).
#[pyfunction]
#[pyo3(signature = (size = None))]
pub fn ranf<'py>(py: Python<'py>, size: Option<&Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    random(py, size)
}

/// `numpy.random.random_integers(low, high=None, size=None)` — legacy,
/// returns integers in the **inclusive** range `[low, high]` (note: numpy
/// `integers`/`randint` use a half-open `[low, high)`).
///
/// numpy `mtrand.pyx:1312` `random_integers`: when `high is None` the
/// range is `[1, low]` (`randint(1, low + 1)`); otherwise it is
/// `[low, high]` (`randint(low, int(high) + 1)`). The bound is inclusive,
/// so we widen the half-open upper edge by one before delegating to the
/// shared `integers` machinery. Returns a scalar for `size=None`.
#[pyfunction]
#[pyo3(signature = (low, high = None, size = None))]
pub fn random_integers<'py>(
    py: Python<'py>,
    low: i64,
    high: Option<i64>,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    // Inclusive-high contract: random_integers(low, high) == integers(low,
    // high + 1); random_integers(low) == integers(1, low + 1).
    let (lo, hi_exclusive) = match high {
        Some(h) => (low, h.saturating_add(1)),
        None => (1, low.saturating_add(1)),
    };
    integers(py, lo, Some(hi_exclusive), size)
}

/// `numpy.random.bytes(length)` — return `length` random bytes as a
/// Python `bytes` object (`numpy/random/__init__.py:129`,
/// `mtrand.pyi:375` `def bytes(self, length: int) -> bytes`).
#[pyfunction]
pub fn bytes<'py>(py: Python<'py>, length: isize) -> PyResult<Bound<'py, PyAny>> {
    if length < 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "negative dimensions are not allowed",
        ));
    }
    let buf: Vec<u8> = with_rng(|rng| rng.bytes(length as usize));
    Ok(pyo3::types::PyBytes::new(py, &buf).into_any())
}

/// `numpy.random.standard_normal(size=None)`.
#[pyfunction]
#[pyo3(signature = (size = None))]
pub fn standard_normal<'py>(
    py: Python<'py>,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    if size.is_none() {
        let v: f64 = with_rng(|rng| {
            rng.standard_normal(1usize)
                .map(|a| a.iter().next().copied().unwrap_or(0.0))
        })
        .map_err(ferr_to_pyerr)?;
        return Ok(v.into_pyobject(py)?.into_any());
    }
    let shape = shape_from_pyany(size)?;
    let arr: ArrayD<f64> =
        with_rng(|rng| rng.standard_normal(shape.as_slice())).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// Alias `randn(*shape)` — varargs form of `standard_normal`.
#[pyfunction]
#[pyo3(signature = (*shape))]
pub fn randn<'py>(
    py: Python<'py>,
    shape: &Bound<'py, pyo3::types::PyTuple>,
) -> PyResult<Bound<'py, PyAny>> {
    if shape.is_empty() {
        let v: f64 = with_rng(|rng| {
            rng.standard_normal(1usize)
                .map(|a| a.iter().next().copied().unwrap_or(0.0))
        })
        .map_err(ferr_to_pyerr)?;
        return Ok(v.into_pyobject(py)?.into_any());
    }
    let dims = varargs_dims(shape)?;
    let arr: ArrayD<f64> =
        with_rng(|rng| rng.standard_normal(dims.as_slice())).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.random.normal(loc=0.0, scale=1.0, size=None)`.
#[pyfunction]
#[pyo3(signature = (loc = 0.0, scale = 1.0, size = None))]
pub fn normal<'py>(
    py: Python<'py>,
    loc: f64,
    scale: f64,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    if size.is_none() {
        let v: f64 = with_rng(|rng| {
            rng.normal(loc, scale, 1usize)
                .map(|a| a.iter().next().copied().unwrap_or(0.0))
        })
        .map_err(ferr_to_pyerr)?;
        return Ok(v.into_pyobject(py)?.into_any());
    }
    let shape = shape_from_pyany(size)?;
    let arr: ArrayD<f64> =
        with_rng(|rng| rng.normal(loc, scale, shape.as_slice())).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.random.uniform(low=0.0, high=1.0, size=None)`.
#[pyfunction]
#[pyo3(signature = (low = 0.0, high = 1.0, size = None))]
pub fn uniform<'py>(
    py: Python<'py>,
    low: f64,
    high: f64,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    if size.is_none() {
        let v: f64 = with_rng(|rng| {
            rng.uniform(low, high, 1usize)
                .map(|a| a.iter().next().copied().unwrap_or(0.0))
        })
        .map_err(ferr_to_pyerr)?;
        return Ok(v.into_pyobject(py)?.into_any());
    }
    let shape = shape_from_pyany(size)?;
    let arr: ArrayD<f64> =
        with_rng(|rng| rng.uniform(low, high, shape.as_slice())).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

// ---------------------------------------------------------------------------
// Discrete (integers)
// ---------------------------------------------------------------------------

/// `numpy.random.integers(low, high=None, size=None)` — uniformly
/// random integers in `[low, high)`.
///
/// NumPy's legacy `randint` is exposed by the same name (modern
/// `Generator.integers` is the canonical name; we follow the modern
/// spelling).
#[pyfunction]
#[pyo3(signature = (low, high = None, size = None))]
pub fn integers<'py>(
    py: Python<'py>,
    low: i64,
    high: Option<i64>,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let (lo, hi) = match high {
        Some(h) => (low, h),
        None => (0, low),
    };
    if size.is_none() {
        let v: i64 = with_rng(|rng| {
            rng.integers(lo, hi, 1usize)
                .map(|a| a.iter().next().copied().unwrap_or(0))
        })
        .map_err(ferr_to_pyerr)?;
        return Ok(v.into_pyobject(py)?.into_any());
    }
    let shape = shape_from_pyany(size)?;
    let arr: ArrayD<i64> =
        with_rng(|rng| rng.integers(lo, hi, shape.as_slice())).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.random.randint(low, high=None, size=None)` — legacy alias for
/// `integers`.
#[pyfunction]
#[pyo3(signature = (low, high = None, size = None))]
pub fn randint<'py>(
    py: Python<'py>,
    low: i64,
    high: Option<i64>,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    integers(py, low, high, size)
}

// ---------------------------------------------------------------------------
// Permutations / choice / shuffle
// ---------------------------------------------------------------------------

/// `numpy.random.permutation(x)` — if x is an int n, return a
/// permutation of `range(n)`; if x is an array, return a permuted
/// copy.
#[pyfunction]
pub fn permutation<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    // Integer argument: permute `range(n)`. numpy `permutation(-3)` ->
    // `array([], dtype=int64)` (it routes the int through `arange(n)`, and
    // `arange` of a negative count yields an empty array — *not* an error),
    // so bind the int signed and short-circuit a non-positive count to an
    // empty int64 array rather than letting a `usize` extraction fail and
    // fall through to the array path (which would TypeError).
    if x.is_instance_of::<pyo3::types::PyInt>() {
        let n: isize = x.extract()?;
        if n <= 0 {
            let empty: Array1<i64> =
                Array1::<i64>::from_vec(ferray_core::dimension::Ix1::new([0]), Vec::new())
                    .map_err(ferr_to_pyerr)?;
            return Ok(empty.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any());
        }
        let arr: Array1<i64> =
            with_rng(|rng| rng.permutation_range(n as usize)).map_err(ferr_to_pyerr)?;
        return Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any());
    }
    let arr = as_ndarray(py, x)?;
    let dt = dtype_name(&arr)?;
    let ndim: usize = arr.getattr("ndim")?.extract()?;
    // N-D input: numpy permutes along axis 0 and preserves shape. Make a
    // dynamic-dim copy and shuffle whole axis-0 hyperslices in place
    // (`Generator::shuffle_dyn`, axis 0), mirroring
    // `numpy.random.permutation` for `ndim >= 2`. 1-D keeps the contiguous
    // fast path.
    Ok(match_dtype_all!(dt.as_str(), T => {
        if ndim <= 1 {
            let view: PyReadonlyArray1<T> = arr.extract()?;
            let fa: Array1<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
            let r: Array1<T> = with_rng(|rng| rng.permutation(&fa))
                .map_err(ferr_to_pyerr)?;
            r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
        } else {
            let view: numpy::PyReadonlyArrayDyn<T> = arr.extract()?;
            let mut fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
            with_rng(|rng| rng.shuffle_dyn(&mut fa, 0)).map_err(ferr_to_pyerr)?;
            fa.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
        }
    }))
}

/// `numpy.random.shuffle(x)` — randomly permute `x` **in place** along
/// axis 0, returning `None`.
///
/// numpy `random/__init__.py:45` — "shuffle: Randomly permute a sequence
/// in place." The contract is: the *passed* array is mutated and the
/// function returns `None` (R-DEV-3). ferray writes the Fisher-Yates swaps
/// directly into the numpy buffer via a writeable view (`PyReadwriteArray1
/// ::as_slice_mut`), drawing bounded indices from the thread-local RNG —
/// so the caller's own array object is shuffled, matching numpy, rather
/// than a copy being returned. Only 1-D in-place shuffle is supported at
/// the boundary today; an N-D array writeback would need axis-0 hyperslice
/// swaps against the live buffer.
#[pyfunction]
pub fn shuffle<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<()> {
    let dt = dtype_name(x)?;
    match_dtype_all!(dt.as_str(), T => {
        let mut rw: PyReadwriteArray1<T> = x.extract().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "shuffle requires a writeable 1-D numpy array",
            )
        })?;
        let slice = rw.as_slice_mut().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be contiguous for in-place shuffle",
            )
        })?;
        let n = slice.len();
        if n > 1 {
            // Fisher-Yates over the live numpy buffer (#447 parity).
            with_rng(|rng| {
                for i in (1..n).rev() {
                    let j = rng.next_u64_bounded((i + 1) as u64) as usize;
                    slice.swap(i, j);
                }
            });
        }
    });
    let _ = py;
    Ok(())
}

// ---------------------------------------------------------------------------
// Additional distributions (#713)
// ---------------------------------------------------------------------------

/// Macro to bind a continuous distribution that takes a single
/// scale/shape parameter plus optional size.
macro_rules! bind_dist_1param {
    ($name:ident, $param:ident, $method:ident) => {
        #[pyfunction]
        #[pyo3(signature = ($param, size = None))]
        pub fn $name<'py>(
            py: Python<'py>,
            $param: f64,
            size: Option<&Bound<'py, PyAny>>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let shape = shape_from_pyany(size)?;
            let arr: ArrayD<f64> =
                with_rng(|rng| rng.$method($param, shape.as_slice())).map_err(ferr_to_pyerr)?;
            Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
        }
    };
}

bind_dist_1param!(exponential, scale, exponential);
bind_dist_1param!(rayleigh, scale, rayleigh);
bind_dist_1param!(weibull, a, weibull);
bind_dist_1param!(pareto, a, pareto);
bind_dist_1param!(chisquare, df, chisquare);

/// `numpy.random.geometric(p, size=None)` — returns int64.
#[pyfunction]
#[pyo3(signature = (p, size = None))]
pub fn geometric<'py>(
    py: Python<'py>,
    p: f64,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let shape = shape_from_pyany(size)?;
    let arr: ArrayD<i64> =
        with_rng(|rng| rng.geometric(p, shape.as_slice())).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.random.poisson(lam=1.0, size=None)` — returns int64.
#[pyfunction]
#[pyo3(signature = (lam = 1.0, size = None))]
pub fn poisson<'py>(
    py: Python<'py>,
    lam: f64,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let shape = shape_from_pyany(size)?;
    let arr: ArrayD<i64> =
        with_rng(|rng| rng.poisson(lam, shape.as_slice())).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.random.lognormal(mean=0, sigma=1, size=None)`.
#[pyfunction]
#[pyo3(signature = (mean = 0.0, sigma = 1.0, size = None))]
pub fn lognormal<'py>(
    py: Python<'py>,
    mean: f64,
    sigma: f64,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let shape = shape_from_pyany(size)?;
    let arr: ArrayD<f64> =
        with_rng(|rng| rng.lognormal(mean, sigma, shape.as_slice())).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.random.binomial(n, p, size=None)`.
///
/// numpy raises `ValueError("n < 0")` for a negative trial count
/// (`numpy/random/mtrand.pyx` `binomial`); binding `n` as `u64` would
/// instead surface a PyO3 `OverflowError` at extraction before any check
/// runs, so bind `n` signed and validate here to mirror numpy's exception
/// *type* (R-DEV-2).
#[pyfunction]
#[pyo3(signature = (n, p, size = None))]
pub fn binomial<'py>(
    py: Python<'py>,
    n: i64,
    p: f64,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    if n < 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("n < 0"));
    }
    let shape = shape_from_pyany(size)?;
    let arr: ArrayD<i64> =
        with_rng(|rng| rng.binomial(n as u64, p, shape.as_slice())).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.random.gamma(shape, scale=1.0, size=None)`.
#[pyfunction]
#[pyo3(signature = (shape, scale = 1.0, size = None))]
pub fn gamma<'py>(
    py: Python<'py>,
    shape: f64,
    scale: f64,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let shape_vec = shape_from_pyany(size)?;
    let arr: ArrayD<f64> =
        with_rng(|rng| rng.gamma(shape, scale, shape_vec.as_slice())).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.random.beta(a, b, size=None)`.
#[pyfunction]
#[pyo3(signature = (a, b, size = None))]
pub fn beta<'py>(
    py: Python<'py>,
    a: f64,
    b: f64,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let shape = shape_from_pyany(size)?;
    let arr: ArrayD<f64> =
        with_rng(|rng| rng.beta(a, b, shape.as_slice())).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.random.laplace(loc=0, scale=1, size=None)`.
#[pyfunction]
#[pyo3(signature = (loc = 0.0, scale = 1.0, size = None))]
pub fn laplace<'py>(
    py: Python<'py>,
    loc: f64,
    scale: f64,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let shape = shape_from_pyany(size)?;
    let arr: ArrayD<f64> =
        with_rng(|rng| rng.laplace(loc, scale, shape.as_slice())).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.random.gumbel(loc=0, scale=1, size=None)`.
#[pyfunction]
#[pyo3(signature = (loc = 0.0, scale = 1.0, size = None))]
pub fn gumbel<'py>(
    py: Python<'py>,
    loc: f64,
    scale: f64,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let shape = shape_from_pyany(size)?;
    let arr: ArrayD<f64> =
        with_rng(|rng| rng.gumbel(loc, scale, shape.as_slice())).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.random.triangular(left, mode, right, size=None)`.
#[pyfunction]
#[pyo3(signature = (left, mode, right, size = None))]
pub fn triangular<'py>(
    py: Python<'py>,
    left: f64,
    mode: f64,
    right: f64,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let shape = shape_from_pyany(size)?;
    let arr: ArrayD<f64> = with_rng(|rng| rng.triangular(left, mode, right, shape.as_slice()))
        .map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

// ---------------------------------------------------------------------------
// Distributions present in numpy/random/__init__.py:126-177's `__all__` and
// implemented by `ferray_random::Generator` but previously unbound at the
// module surface (#787 follow-up). Each delegates to the library method.
// ---------------------------------------------------------------------------

/// `numpy.random.logistic(loc=0.0, scale=1.0, size=None)`
/// (`numpy/random/__init__.py:142`).
#[pyfunction]
#[pyo3(signature = (loc = 0.0, scale = 1.0, size = None))]
pub fn logistic<'py>(
    py: Python<'py>,
    loc: f64,
    scale: f64,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let shape = shape_from_pyany(size)?;
    let arr: ArrayD<f64> =
        with_rng(|rng| rng.logistic(loc, scale, shape.as_slice())).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.random.power(a, size=None)` (`numpy/random/__init__.py:153`).
#[pyfunction]
#[pyo3(signature = (a, size = None))]
pub fn power<'py>(
    py: Python<'py>,
    a: f64,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let shape = shape_from_pyany(size)?;
    let arr: ArrayD<f64> = with_rng(|rng| rng.power(a, shape.as_slice())).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.random.standard_t(df, size=None)` (`numpy/random/__init__.py:169`).
#[pyfunction]
#[pyo3(signature = (df, size = None))]
pub fn standard_t<'py>(
    py: Python<'py>,
    df: f64,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let shape = shape_from_pyany(size)?;
    let arr: ArrayD<f64> =
        with_rng(|rng| rng.standard_t(df, shape.as_slice())).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.random.f(dfnum, dfden, size=None)` (`numpy/random/__init__.py:134`).
#[pyfunction]
#[pyo3(name = "f", signature = (dfnum, dfden, size = None))]
pub fn f_dist<'py>(
    py: Python<'py>,
    dfnum: f64,
    dfden: f64,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let shape = shape_from_pyany(size)?;
    let arr: ArrayD<f64> =
        with_rng(|rng| rng.f(dfnum, dfden, shape.as_slice())).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.random.vonmises(mu, kappa, size=None)`
/// (`numpy/random/__init__.py:173`).
#[pyfunction]
#[pyo3(signature = (mu, kappa, size = None))]
pub fn vonmises<'py>(
    py: Python<'py>,
    mu: f64,
    kappa: f64,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let shape = shape_from_pyany(size)?;
    let arr: ArrayD<f64> =
        with_rng(|rng| rng.vonmises(mu, kappa, shape.as_slice())).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.random.wald(mean, scale, size=None)`
/// (`numpy/random/__init__.py:174`).
#[pyfunction]
#[pyo3(signature = (mean, scale, size = None))]
pub fn wald<'py>(
    py: Python<'py>,
    mean: f64,
    scale: f64,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let shape = shape_from_pyany(size)?;
    let arr: ArrayD<f64> =
        with_rng(|rng| rng.wald(mean, scale, shape.as_slice())).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.random.standard_cauchy(size=None)`
/// (`numpy/random/__init__.py:166`).
#[pyfunction]
#[pyo3(signature = (size = None))]
pub fn standard_cauchy<'py>(
    py: Python<'py>,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let shape = shape_from_pyany(size)?;
    let arr: ArrayD<f64> =
        with_rng(|rng| rng.standard_cauchy(shape.as_slice())).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.random.standard_exponential(size=None)`
/// (`numpy/random/__init__.py:167`).
#[pyfunction]
#[pyo3(signature = (size = None))]
pub fn standard_exponential<'py>(
    py: Python<'py>,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let shape = shape_from_pyany(size)?;
    let arr: ArrayD<f64> =
        with_rng(|rng| rng.standard_exponential(shape.as_slice())).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.random.standard_gamma(shape, size=None)`
/// (`numpy/random/__init__.py:168`). The first positional is the gamma
/// *shape* parameter (`alpha`); `size` is the output shape.
#[pyfunction]
#[pyo3(name = "standard_gamma", signature = (shape, size = None))]
pub fn standard_gamma<'py>(
    py: Python<'py>,
    shape: f64,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let size_shape = shape_from_pyany(size)?;
    let arr: ArrayD<f64> =
        with_rng(|rng| rng.standard_gamma(shape, size_shape.as_slice())).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.random.negative_binomial(n, p, size=None)` — returns int64
/// (`numpy/random/__init__.py:146`).
#[pyfunction]
#[pyo3(signature = (n, p, size = None))]
pub fn negative_binomial<'py>(
    py: Python<'py>,
    n: f64,
    p: f64,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let shape = shape_from_pyany(size)?;
    let arr: ArrayD<i64> =
        with_rng(|rng| rng.negative_binomial(n, p, shape.as_slice())).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.random.hypergeometric(ngood, nbad, nsample, size=None)` — returns
/// int64 (`numpy/random/__init__.py:139`). numpy raises `ValueError` for a
/// negative count; binding the counts `u64` would surface `OverflowError`,
/// so bind them signed and validate (R-DEV-2).
#[pyfunction]
#[pyo3(signature = (ngood, nbad, nsample, size = None))]
pub fn hypergeometric<'py>(
    py: Python<'py>,
    ngood: i64,
    nbad: i64,
    nsample: i64,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    if ngood < 0 || nbad < 0 || nsample < 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "ngood, nbad, and nsample must be non-negative",
        ));
    }
    let shape = shape_from_pyany(size)?;
    let arr: ArrayD<i64> = with_rng(|rng| {
        rng.hypergeometric(ngood as u64, nbad as u64, nsample as u64, shape.as_slice())
    })
    .map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.random.choice(a, size=None, replace=True, p=None)`.
///
/// `size` is a shape: `None` returns a **scalar** (`mtrand.pyi:379`
/// `choice(a, size=None) -> int`/`Any`), an int returns a 1-D array, and a
/// tuple returns an array of that shape (`mtrand.pyi:387` `size:
/// _ShapeLike`). ferray draws `prod(shape)` samples and reshapes, then
/// down-converts a 0-d result to a Python scalar via [`scalarize`]
/// (R-DEV-3).
#[pyfunction]
#[pyo3(signature = (a, size = None, replace = true, p = None))]
pub fn choice<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    size: Option<&Bound<'py, PyAny>>,
    replace: bool,
    p: Option<Vec<f64>>,
) -> PyResult<Bound<'py, PyAny>> {
    let scalar = size.is_none();
    let shape = shape_from_pyany(size)?;
    let n: usize = shape
        .iter()
        .product::<usize>()
        .max(if scalar { 1 } else { 0 });
    // If the user passed an int, treat as range(int) (NumPy convention).
    if let Ok(k) = a.extract::<usize>() {
        let range_arr: Array1<i64> = (0..k as i64).collect::<Vec<_>>().pipe(|v| {
            Array1::<i64>::from_vec(ferray_core::dimension::Ix1::new([k]), v).map_err(ferr_to_pyerr)
        })?;
        let p_ref = p.as_deref();
        let r: Array1<i64> =
            with_rng(|rng| rng.choice(&range_arr, n, replace, p_ref)).map_err(ferr_to_pyerr)?;
        let out = r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
        return finish_choice(out, &shape, scalar);
    }
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    let p_ref = p.as_deref();
    let out = match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArray1<T> = arr.extract()?;
        let fa: Array1<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: Array1<T> = with_rng(|rng| rng.choice(&fa, n, replace, p_ref))
            .map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    });
    finish_choice(out, &shape, scalar)
}

/// Reshape the flat `choice` draw to the requested `shape` and, when
/// `size=None`, collapse the length-1 result to a 0-d scalar (numpy's
/// `choice(a, size=None)` returns a scalar, not a length-1 array).
fn finish_choice<'py>(
    flat: Bound<'py, PyAny>,
    shape: &[usize],
    scalar: bool,
) -> PyResult<Bound<'py, PyAny>> {
    if scalar {
        // length-1 1-D array -> 0-d -> Python scalar via scalarize.
        let zerod = flat.call_method1("reshape", (pyo3::types::PyTuple::empty(flat.py()),))?;
        return scalarize(zerod);
    }
    if shape.len() <= 1 {
        return Ok(flat);
    }
    let tuple = pyo3::types::PyTuple::new(flat.py(), shape)?;
    flat.call_method1("reshape", (tuple,))
}

// Tiny helper to keep the choice arm above readable — chains a closure
// onto a value, like Kotlin's `let`.
trait Pipe: Sized {
    fn pipe<F, R>(self, f: F) -> R
    where
        F: FnOnce(Self) -> R,
    {
        f(self)
    }
}
impl<T> Pipe for T {}

// ---------------------------------------------------------------------------
// Modern Generator class API (#712) — numpy.random.default_rng() returns
// a Generator instance with isolated state.
// ---------------------------------------------------------------------------

/// `numpy.random.Generator` — a stateful pseudo-random generator with
/// methods for each distribution. Constructed via
/// `ferray.random.default_rng(seed=None)`.
// Generator owns mutable RNG state — cloning would silently fork the
// stream, which is almost never what callers want. Without Clone we
// can't derive FromPyObject; the methods don't need it (each method
// takes &mut self by reference, not by-value).
#[pyclass(name = "Generator", module = "ferray.random", skip_from_py_object)]
pub struct PyGenerator {
    inner: Generator<Xoshiro256StarStar>,
}

#[pymethods]
impl PyGenerator {
    #[new]
    #[pyo3(signature = (seed = None))]
    fn py_new(seed: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        Ok(Self {
            inner: seed_generator_from(seed)?,
        })
    }

    fn __repr__(&self) -> String {
        "Generator(Xoshiro256**)".to_string()
    }

    /// `Generator.random(size=None)` — uniform [0, 1) floats.
    #[pyo3(signature = (size = None))]
    fn random<'py>(
        &mut self,
        py: Python<'py>,
        size: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if size.is_none() {
            let v = self
                .inner
                .random(1usize)
                .map_err(ferr_to_pyerr)?
                .iter()
                .next()
                .copied()
                .unwrap_or(0.0);
            return Ok(v.into_pyobject(py)?.into_any());
        }
        let shape = shape_from_pyany(size)?;
        let arr: ArrayD<f64> = self.inner.random(shape.as_slice()).map_err(ferr_to_pyerr)?;
        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    /// `Generator.standard_normal(size=None)`.
    #[pyo3(signature = (size = None))]
    fn standard_normal<'py>(
        &mut self,
        py: Python<'py>,
        size: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if size.is_none() {
            let v = self
                .inner
                .standard_normal(1usize)
                .map_err(ferr_to_pyerr)?
                .iter()
                .next()
                .copied()
                .unwrap_or(0.0);
            return Ok(v.into_pyobject(py)?.into_any());
        }
        let shape = shape_from_pyany(size)?;
        let arr: ArrayD<f64> = self
            .inner
            .standard_normal(shape.as_slice())
            .map_err(ferr_to_pyerr)?;
        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    /// `Generator.normal(loc=0, scale=1, size=None)`.
    #[pyo3(signature = (loc = 0.0, scale = 1.0, size = None))]
    fn normal<'py>(
        &mut self,
        py: Python<'py>,
        loc: f64,
        scale: f64,
        size: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let shape = shape_from_pyany(size)?;
        let arr: ArrayD<f64> = self
            .inner
            .normal(loc, scale, shape.as_slice())
            .map_err(ferr_to_pyerr)?;
        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    /// `Generator.uniform(low=0, high=1, size=None)`.
    #[pyo3(signature = (low = 0.0, high = 1.0, size = None))]
    fn uniform<'py>(
        &mut self,
        py: Python<'py>,
        low: f64,
        high: f64,
        size: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let shape = shape_from_pyany(size)?;
        let arr: ArrayD<f64> = self
            .inner
            .uniform(low, high, shape.as_slice())
            .map_err(ferr_to_pyerr)?;
        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    /// `Generator.integers(low, high=None, size=None)`.
    #[pyo3(signature = (low, high = None, size = None))]
    fn integers<'py>(
        &mut self,
        py: Python<'py>,
        low: i64,
        high: Option<i64>,
        size: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (lo, hi) = match high {
            Some(h) => (low, h),
            None => (0, low),
        };
        let shape = shape_from_pyany(size)?;
        let arr: ArrayD<i64> = self
            .inner
            .integers(lo, hi, shape.as_slice())
            .map_err(ferr_to_pyerr)?;
        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    /// `Generator.exponential(scale=1.0, size=None)`.
    #[pyo3(signature = (scale = 1.0, size = None))]
    fn exponential<'py>(
        &mut self,
        py: Python<'py>,
        scale: f64,
        size: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let shape = shape_from_pyany(size)?;
        let arr: ArrayD<f64> = self
            .inner
            .exponential(scale, shape.as_slice())
            .map_err(ferr_to_pyerr)?;
        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    /// `Generator.poisson(lam=1.0, size=None)`.
    #[pyo3(signature = (lam = 1.0, size = None))]
    fn poisson<'py>(
        &mut self,
        py: Python<'py>,
        lam: f64,
        size: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let shape = shape_from_pyany(size)?;
        let arr: ArrayD<i64> = self
            .inner
            .poisson(lam, shape.as_slice())
            .map_err(ferr_to_pyerr)?;
        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    /// `Generator.binomial(n, p, size=None)` — `n < 0` -> `ValueError`
    /// (R-DEV-2), bound signed to avoid a PyO3 `OverflowError`.
    #[pyo3(signature = (n, p, size = None))]
    fn binomial<'py>(
        &mut self,
        py: Python<'py>,
        n: i64,
        p: f64,
        size: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if n < 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("n < 0"));
        }
        let shape = shape_from_pyany(size)?;
        let arr: ArrayD<i64> = self
            .inner
            .binomial(n as u64, p, shape.as_slice())
            .map_err(ferr_to_pyerr)?;
        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    /// `Generator.gamma(shape, scale=1.0, size=None)`.
    #[pyo3(name = "gamma", signature = (shape, scale = 1.0, size = None))]
    fn gamma_<'py>(
        &mut self,
        py: Python<'py>,
        shape: f64,
        scale: f64,
        size: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let shape_vec = shape_from_pyany(size)?;
        let arr: ArrayD<f64> = self
            .inner
            .gamma(shape, scale, shape_vec.as_slice())
            .map_err(ferr_to_pyerr)?;
        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    /// `Generator.beta(a, b, size=None)`.
    #[pyo3(signature = (a, b, size = None))]
    fn beta<'py>(
        &mut self,
        py: Python<'py>,
        a: f64,
        b: f64,
        size: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let shape = shape_from_pyany(size)?;
        let arr: ArrayD<f64> = self
            .inner
            .beta(a, b, shape.as_slice())
            .map_err(ferr_to_pyerr)?;
        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    /// `Generator.laplace(loc=0, scale=1, size=None)`.
    #[pyo3(signature = (loc = 0.0, scale = 1.0, size = None))]
    fn laplace<'py>(
        &mut self,
        py: Python<'py>,
        loc: f64,
        scale: f64,
        size: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let shape = shape_from_pyany(size)?;
        let arr: ArrayD<f64> = self
            .inner
            .laplace(loc, scale, shape.as_slice())
            .map_err(ferr_to_pyerr)?;
        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    /// `Generator.gumbel(loc=0, scale=1, size=None)`.
    #[pyo3(signature = (loc = 0.0, scale = 1.0, size = None))]
    fn gumbel<'py>(
        &mut self,
        py: Python<'py>,
        loc: f64,
        scale: f64,
        size: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let shape = shape_from_pyany(size)?;
        let arr: ArrayD<f64> = self
            .inner
            .gumbel(loc, scale, shape.as_slice())
            .map_err(ferr_to_pyerr)?;
        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    /// `Generator.triangular(left, mode, right, size=None)`.
    #[pyo3(signature = (left, mode, right, size = None))]
    fn triangular<'py>(
        &mut self,
        py: Python<'py>,
        left: f64,
        mode: f64,
        right: f64,
        size: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let shape = shape_from_pyany(size)?;
        let arr: ArrayD<f64> = self
            .inner
            .triangular(left, mode, right, shape.as_slice())
            .map_err(ferr_to_pyerr)?;
        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    /// `Generator.rayleigh(scale=1.0, size=None)`.
    #[pyo3(signature = (scale = 1.0, size = None))]
    fn rayleigh<'py>(
        &mut self,
        py: Python<'py>,
        scale: f64,
        size: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let shape = shape_from_pyany(size)?;
        let arr: ArrayD<f64> = self
            .inner
            .rayleigh(scale, shape.as_slice())
            .map_err(ferr_to_pyerr)?;
        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    /// `Generator.weibull(a, size=None)`.
    #[pyo3(signature = (a, size = None))]
    fn weibull<'py>(
        &mut self,
        py: Python<'py>,
        a: f64,
        size: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let shape = shape_from_pyany(size)?;
        let arr: ArrayD<f64> = self
            .inner
            .weibull(a, shape.as_slice())
            .map_err(ferr_to_pyerr)?;
        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    /// `Generator.lognormal(mean=0, sigma=1, size=None)`.
    #[pyo3(signature = (mean = 0.0, sigma = 1.0, size = None))]
    fn lognormal<'py>(
        &mut self,
        py: Python<'py>,
        mean: f64,
        sigma: f64,
        size: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let shape = shape_from_pyany(size)?;
        let arr: ArrayD<f64> = self
            .inner
            .lognormal(mean, sigma, shape.as_slice())
            .map_err(ferr_to_pyerr)?;
        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    /// `Generator.geometric(p, size=None)` — returns int64.
    #[pyo3(signature = (p, size = None))]
    fn geometric<'py>(
        &mut self,
        py: Python<'py>,
        p: f64,
        size: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let shape = shape_from_pyany(size)?;
        let arr: ArrayD<i64> = self
            .inner
            .geometric(p, shape.as_slice())
            .map_err(ferr_to_pyerr)?;
        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    /// `Generator.choice(a, size=None, replace=True, p=None)` — mirrors the
    /// module-level `choice` (scalar for `size=None`, shape tuple
    /// supported) but draws from this Generator's isolated stream.
    #[pyo3(signature = (a, size = None, replace = true, p = None))]
    fn choice<'py>(
        &mut self,
        py: Python<'py>,
        a: &Bound<'py, PyAny>,
        size: Option<&Bound<'py, PyAny>>,
        replace: bool,
        p: Option<Vec<f64>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let scalar = size.is_none();
        let shape = shape_from_pyany(size)?;
        let n: usize = shape
            .iter()
            .product::<usize>()
            .max(if scalar { 1 } else { 0 });
        let p_ref = p.as_deref();
        if let Ok(k) = a.extract::<usize>() {
            let range_arr: Array1<i64> = (0..k as i64).collect::<Vec<_>>().pipe(|v| {
                Array1::<i64>::from_vec(ferray_core::dimension::Ix1::new([k]), v)
                    .map_err(ferr_to_pyerr)
            })?;
            let r: Array1<i64> = self
                .inner
                .choice(&range_arr, n, replace, p_ref)
                .map_err(ferr_to_pyerr)?;
            let out = r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
            return finish_choice(out, &shape, scalar);
        }
        let arr = as_ndarray(py, a)?;
        let dt = dtype_name(&arr)?;
        let out = match_dtype_all!(dt.as_str(), T => {
            let view: PyReadonlyArray1<T> = arr.extract()?;
            let fa: Array1<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
            let r: Array1<T> = self.inner.choice(&fa, n, replace, p_ref)
                .map_err(ferr_to_pyerr)?;
            r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
        });
        finish_choice(out, &shape, scalar)
    }

    /// `Generator.permutation(x)` — permute `range(n)` for an int, or an
    /// array along axis 0 (N-D preserves shape).
    fn permutation<'py>(
        &mut self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if x.is_instance_of::<pyo3::types::PyInt>() {
            let nn: isize = x.extract()?;
            if nn <= 0 {
                let empty: Array1<i64> =
                    Array1::<i64>::from_vec(ferray_core::dimension::Ix1::new([0]), Vec::new())
                        .map_err(ferr_to_pyerr)?;
                return Ok(empty.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any());
            }
            let arr: Array1<i64> = self
                .inner
                .permutation_range(nn as usize)
                .map_err(ferr_to_pyerr)?;
            return Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any());
        }
        let arr = as_ndarray(py, x)?;
        let dt = dtype_name(&arr)?;
        let ndim: usize = arr.getattr("ndim")?.extract()?;
        Ok(match_dtype_all!(dt.as_str(), T => {
            if ndim <= 1 {
                let view: PyReadonlyArray1<T> = arr.extract()?;
                let fa: Array1<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
                let r: Array1<T> = self.inner.permutation(&fa).map_err(ferr_to_pyerr)?;
                r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
            } else {
                let view: numpy::PyReadonlyArrayDyn<T> = arr.extract()?;
                let mut fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
                self.inner.shuffle_dyn(&mut fa, 0).map_err(ferr_to_pyerr)?;
                fa.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
            }
        }))
    }

    /// `Generator.shuffle(x)` — shuffle a 1-D array in place, returning
    /// `None` (mutates the live numpy buffer).
    fn shuffle(&mut self, x: &Bound<'_, PyAny>) -> PyResult<()> {
        let dt = dtype_name(x)?;
        match_dtype_all!(dt.as_str(), T => {
            let mut rw: PyReadwriteArray1<T> = x.extract().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "shuffle requires a writeable 1-D numpy array",
                )
            })?;
            let slice = rw.as_slice_mut().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(
                    "array must be contiguous for in-place shuffle",
                )
            })?;
            let n = slice.len();
            for i in (1..n).rev() {
                let j = self.inner.next_u64_bounded((i + 1) as u64) as usize;
                slice.swap(i, j);
            }
        });
        Ok(())
    }
}

/// `numpy.random.default_rng(seed=None)` — return a fresh `Generator`.
///
/// `seed` accepts `None`, an `int`, or an `array_like[int]` / sequence
/// (`numpy/random/_generator.pyi:758`); a sequence is folded through
/// `SeedSequence` (see [`seed_generator_from`]).
#[pyfunction]
#[pyo3(signature = (seed = None))]
pub fn default_rng_py(seed: Option<&Bound<'_, PyAny>>) -> PyResult<PyGenerator> {
    Ok(PyGenerator {
        inner: seed_generator_from(seed)?,
    })
}
