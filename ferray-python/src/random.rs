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

/// The thread-local global random state backing the module-level functions.
///
/// Beyond the `Generator<Xoshiro256**>` stream itself we retain the `seed`
/// that originated it. numpy's singleton `RandomState` owns a *bit generator*
/// object that `get_bit_generator()` hands out and `set_bit_generator()`
/// installs (`numpy/random/mtrand.pyx:4838,4864`). ferray reconstructs that
/// hand-out from `seed`: `get_bit_generator()` returns a `PyBitGenerator`
/// carrying this `seed`, and `set_bit_generator(bg)` re-seeds the global from
/// `bg.seed` — so `set_bit_generator(get_bit_generator())` reinstalls the same
/// reproducible stream (R-DEV-7: contract preserved over a different PRNG).
struct GlobalRng {
    rng: Generator<Xoshiro256StarStar>,
    seed: u64,
}

thread_local! {
    /// Thread-local convenience generator backing the module-level
    /// functions. Initialised from an OS-entropy seed that is *committed*
    /// (not discarded) so the global has a recoverable bit-generator origin.
    static RNG: RefCell<GlobalRng> = RefCell::new(GlobalRng::from_seed(os_entropy_u64()));
}

impl GlobalRng {
    fn from_seed(seed: u64) -> Self {
        Self {
            rng: default_rng_seeded(seed),
            seed,
        }
    }
}

fn with_rng<F, R>(f: F) -> R
where
    F: FnOnce(&mut Generator<Xoshiro256StarStar>) -> R,
{
    RNG.with(|cell| f(&mut cell.borrow_mut().rng))
}

/// Reset the thread-local global RNG to a fresh stream seeded from `seed`,
/// recording `seed` as its bit-generator origin.
fn reseed_global(seed: u64) {
    RNG.with(|cell| *cell.borrow_mut() = GlobalRng::from_seed(seed));
}

/// The resolved seed currently backing the thread-local global RNG — the
/// value `get_bit_generator()` wraps in the returned `PyBitGenerator`.
fn global_seed() -> u64 {
    RNG.with(|cell| cell.borrow().seed)
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
///
/// Returns both the constructed generator and the resolved `u64` Xoshiro seed.
/// The seed is what
/// [`get_bit_generator`]/`Generator.bit_generator` carry so that
/// `set_bit_generator(get_bit_generator())` re-seeds the same reproducible
/// stream. For an `int`/`BitGenerator`/`SeedSequence`/`array_like` seed the
/// resolution is deterministic; for `None` (OS entropy) we still draw and
/// commit a concrete `u64` so the resulting generator has a recoverable,
/// reproducible origin seed (the BitGenerator it later hands out re-seeds the
/// identical stream).
fn seed_generator_with_seed(
    seed: Option<&Bound<'_, PyAny>>,
) -> PyResult<(Generator<Xoshiro256StarStar>, u64)> {
    let resolved: u64 = match seed {
        None => os_entropy_u64(),
        Some(o) => {
            if o.is_none() {
                os_entropy_u64()
            } else if let Ok(bg) = o.extract::<PyRef<'_, PyBitGenerator>>() {
                // A BitGenerator instance (`PCG64(42)`, `MT19937(7)`, …) carries
                // the resolved Xoshiro seed it was built from. numpy's
                // `default_rng(bit_generator)` adopts that generator's stream;
                // ferray has a single PRNG (Xoshiro256**), so we re-seed from the
                // BitGenerator's stored seed — API-compatible, NOT bit-identical
                // to numpy's PCG64/MT19937/etc. streams (documented limitation).
                bg.seed
            } else if let Ok(s) = o.extract::<u64>() {
                s
            } else if let Ok(ss) = o.extract::<PyRef<'_, PySeedSequence>>() {
                // A SeedSequence instance: fold to a single deterministic u64.
                ss.inner.generate_u64()
            } else {
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
                SeedSequence::new(entropy).generate_u64()
            }
        }
    };
    Ok((default_rng_seeded(resolved), resolved))
}

// ---------------------------------------------------------------------------
// Seed control
// ---------------------------------------------------------------------------

/// `numpy.random.seed(seed=None)` — re-seed the thread-local RNG.
///
/// numpy's legacy `numpy.random.seed` accepts the tri-state
/// `None | int | array_like[int]` (`numpy/random/mtrand.pyx` `seed` ->
/// `RandomState.seed`, exposed at `numpy/random/__init__.py`): `seed()` and
/// `seed(None)` reseed the global generator from OS entropy (subsequent draws
/// are fresh / non-reproducible across two `seed()` calls), an `int` reseeds
/// deterministically, and an `array_like[int]` is folded via `SeedSequence`.
/// The previous `s: u64` signature made the argument mandatory, so `seed()` /
/// `seed(None)` raised `TypeError` instead of reseeding (#913).
///
/// We route every case through [`seed_generator_with_seed`], which already
/// resolves `None`/absent to fresh OS entropy, an `int` to itself, an
/// `array_like` through `SeedSequence`, and a `BitGenerator`/`SeedSequence`
/// instance to its stored seed — then install the resolved `u64` origin into
/// the global so the int-seed stream stays reproducible (#910 round-trip
/// preserved). Returns `None`, like numpy.
#[pyfunction]
#[pyo3(signature = (seed = None))]
pub fn seed(seed: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    let (_gen, resolved) = seed_generator_with_seed(seed)?;
    reseed_global(resolved);
    Ok(())
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

// ---------------------------------------------------------------------------
// Multivariate / extra distributions implemented in ferray-random but
// previously unbound at the Python surface (refs #834 #818). The library
// methods return a 2-D `[rows, k]` array (one draw per row); numpy's
// boundary contract is `size + (k,)` for an array/tuple `size` and a bare
// `(k,)` 1-D vector for `size=None`. `resolve_mv_size` maps a numpy `size`
// argument to `(rows, target_shape)`:
//   - None     -> (1,   None)            (drop leading dim -> 1-D `(k,)`)
//   - int n    -> (n,   Some([n]))       (natural `[n, k]`)
//   - tuple t  -> (prod(t), Some(t))     (reshape `[prod, k]` -> `t + (k,)`)
// ---------------------------------------------------------------------------

/// Resolve a numpy `size` argument to `(rows, outer_shape)` where `rows` is
/// the number of independent draws (rows the library must produce) and
/// `outer_shape` is the leading shape to prepend to the per-draw vector
/// length `k`. `None` -> `(1, None)` so the result collapses to 1-D `(k,)`.
fn resolve_mv_size(size: Option<&Bound<'_, PyAny>>) -> PyResult<(usize, Option<Vec<usize>>)> {
    match size {
        None => Ok((1, None)),
        Some(o) if o.is_none() => Ok((1, None)),
        Some(_) => {
            // An empty tuple `size=()` is a single draw; otherwise the
            // number of rows is the product of the requested dims.
            let dims = shape_from_pyany(size)?;
            let rows: usize = if dims.is_empty() {
                1
            } else {
                dims.iter().product()
            };
            Ok((rows, Some(dims)))
        }
    }
}

/// Reshape the flat `[rows, k]` boundary array to numpy's `size + (k,)`
/// contract. `outer_shape == None` (i.e. numpy `size=None`) collapses the
/// single row to a 1-D `(k,)` vector; an int `size` keeps `[n, k]`; a tuple
/// `size` reshapes to `outer + (k,)`.
fn reshape_mv<'py>(
    arr2d: Bound<'py, PyAny>,
    outer_shape: Option<Vec<usize>>,
    k: usize,
) -> PyResult<Bound<'py, PyAny>> {
    match outer_shape {
        None => {
            // size=None: single draw -> 1-D vector of length k.
            let tuple = pyo3::types::PyTuple::new(arr2d.py(), [k])?;
            arr2d.call_method1("reshape", (tuple,))
        }
        Some(dims) => {
            // int size -> already [n, k]; tuple size -> outer + (k,).
            if dims.len() <= 1 {
                return Ok(arr2d);
            }
            let mut full = dims;
            full.push(k);
            let tuple = pyo3::types::PyTuple::new(arr2d.py(), full)?;
            arr2d.call_method1("reshape", (tuple,))
        }
    }
}

/// `numpy.random.dirichlet(alpha, size=None)` — float64, each draw sums to
/// 1 along the last axis; shape `size + (len(alpha),)`
/// (`numpy/random/_generator.pyi` `dirichlet`).
#[pyfunction]
#[pyo3(signature = (alpha, size = None))]
pub fn dirichlet<'py>(
    py: Python<'py>,
    alpha: Vec<f64>,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let k = alpha.len();
    let (rows, outer) = resolve_mv_size(size)?;
    let arr = with_rng(|rng| rng.dirichlet(&alpha, rows)).map_err(ferr_to_pyerr)?;
    let out = arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    reshape_mv(out, outer, k)
}

/// `numpy.random.multinomial(n, pvals, size=None)` — int64, last axis
/// `len(pvals)`, each row sums to `n`
/// (`numpy/random/_generator.pyi` `multinomial`). `n < 0` -> `ValueError`
/// (bound signed to avoid a PyO3 `OverflowError`; R-DEV-2).
#[pyfunction]
#[pyo3(signature = (n, pvals, size = None))]
pub fn multinomial<'py>(
    py: Python<'py>,
    n: i64,
    pvals: Vec<f64>,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    if n < 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("n < 0"));
    }
    let k = pvals.len();
    let (rows, outer) = resolve_mv_size(size)?;
    let arr = with_rng(|rng| rng.multinomial(n as u64, &pvals, rows)).map_err(ferr_to_pyerr)?;
    let out = arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    reshape_mv(out, outer, k)
}

/// `numpy.random.multivariate_normal(mean, cov, size=None)` — float64,
/// shape `size + (len(mean),)`; `cov` is flattened row-major to a
/// `len(mean) x len(mean)` matrix (`numpy/random/_generator.pyi`
/// `multivariate_normal`).
#[pyfunction]
#[pyo3(signature = (mean, cov, size = None))]
pub fn multivariate_normal<'py>(
    py: Python<'py>,
    mean: Vec<f64>,
    cov: &Bound<'py, PyAny>,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let d = mean.len();
    // cov is array_like `[d, d]`; flatten row-major to `[d*d]` for the
    // library entry point, which validates `cov.len() == d * d`.
    let cov_arr = as_ndarray(py, cov)?;
    let cov_flat: Vec<f64> = cov_arr
        .call_method1("reshape", ((-1isize,),))?
        .call_method0("tolist")?
        .extract()?;
    let (rows, outer) = resolve_mv_size(size)?;
    let arr =
        with_rng(|rng| rng.multivariate_normal(&mean, &cov_flat, rows)).map_err(ferr_to_pyerr)?;
    let out = arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    reshape_mv(out, outer, d)
}

/// `numpy.random.multivariate_hypergeometric(colors, nsample, size=None)`
/// — int64, last axis `len(colors)`, each row sums to `nsample`
/// (`numpy/random/_generator.pyi` `multivariate_hypergeometric`). A
/// negative `colors`/`nsample` -> `ValueError` (bound signed to avoid a
/// PyO3 `OverflowError`; R-DEV-2).
#[pyfunction]
#[pyo3(signature = (colors, nsample, size = None))]
pub fn multivariate_hypergeometric<'py>(
    py: Python<'py>,
    colors: Vec<i64>,
    nsample: i64,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    if nsample < 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "nsample must be non-negative",
        ));
    }
    let mut colors_u = Vec::with_capacity(colors.len());
    for c in &colors {
        if *c < 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "colors must be non-negative",
            ));
        }
        colors_u.push(*c as u64);
    }
    let k = colors_u.len();
    let (rows, outer) = resolve_mv_size(size)?;
    let arr = with_rng(|rng| rng.multivariate_hypergeometric(&colors_u, nsample as u64, rows))
        .map_err(ferr_to_pyerr)?;
    let out = arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    reshape_mv(out, outer, k)
}

/// `numpy.random.logseries(p, size=None)` — int64 in `{1, 2, ...}`
/// (`numpy/random/_generator.pyi` `logseries`).
#[pyfunction]
#[pyo3(signature = (p, size = None))]
pub fn logseries<'py>(
    py: Python<'py>,
    p: f64,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let shape = shape_from_pyany(size)?;
    let arr: ArrayD<i64> =
        with_rng(|rng| rng.logseries(p, shape.as_slice())).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.random.noncentral_chisquare(df, nonc, size=None)` — float64
/// (`numpy/random/_generator.pyi` `noncentral_chisquare`).
#[pyfunction]
#[pyo3(signature = (df, nonc, size = None))]
pub fn noncentral_chisquare<'py>(
    py: Python<'py>,
    df: f64,
    nonc: f64,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let shape = shape_from_pyany(size)?;
    let arr: ArrayD<f64> = with_rng(|rng| rng.noncentral_chisquare(df, nonc, shape.as_slice()))
        .map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.random.noncentral_f(dfnum, dfden, nonc, size=None)` — float64
/// (`numpy/random/_generator.pyi` `noncentral_f`).
#[pyfunction]
#[pyo3(signature = (dfnum, dfden, nonc, size = None))]
pub fn noncentral_f<'py>(
    py: Python<'py>,
    dfnum: f64,
    dfden: f64,
    nonc: f64,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let shape = shape_from_pyany(size)?;
    let arr: ArrayD<f64> = with_rng(|rng| rng.noncentral_f(dfnum, dfden, nonc, shape.as_slice()))
        .map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.random.zipf(a, size=None)` — int64 in `{1, 2, ...}` with `a > 1`
/// (`numpy/random/_generator.pyi` `zipf`).
#[pyfunction]
#[pyo3(signature = (a, size = None))]
pub fn zipf<'py>(
    py: Python<'py>,
    a: f64,
    size: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let shape = shape_from_pyany(size)?;
    let arr: ArrayD<i64> = with_rng(|rng| rng.zipf(a, shape.as_slice())).map_err(ferr_to_pyerr)?;
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
    /// Resolved Xoshiro seed this Generator wraps — the value exposed via the
    /// `bit_generator` getter so the wrapped BitGenerator re-seeds the same
    /// stream (numpy's `Generator.bit_generator`, `_generator.pyi:50`).
    seed: u64,
}

#[pymethods]
impl PyGenerator {
    #[new]
    #[pyo3(signature = (seed = None))]
    fn py_new(seed: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let (inner, resolved) = seed_generator_with_seed(seed)?;
        Ok(Self {
            inner,
            seed: resolved,
        })
    }

    fn __repr__(&self) -> String {
        "Generator(Xoshiro256**)".to_string()
    }

    /// `Generator.bit_generator` — the BitGenerator object this Generator
    /// wraps (`numpy/random/_generator.pyi:50`; numpy's `default_rng(X)`
    /// wraps a `PCG64`). ferray returns a `PyBitGenerator` carrying this
    /// Generator's resolved Xoshiro seed, labelled `PCG64` to match numpy's
    /// default-rng bit generator class.
    #[getter]
    fn bit_generator(&self) -> PyBitGenerator {
        PyBitGenerator::from_seed(self.seed, "PCG64")
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

    // --- multivariate / extra distributions (refs #834 #818) -----------

    /// `Generator.dirichlet(alpha, size=None)` — float64, each draw sums to
    /// 1 along the last axis; shape `size + (len(alpha),)`.
    #[pyo3(signature = (alpha, size = None))]
    fn dirichlet<'py>(
        &mut self,
        py: Python<'py>,
        alpha: Vec<f64>,
        size: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let k = alpha.len();
        let (rows, outer) = resolve_mv_size(size)?;
        let arr = self.inner.dirichlet(&alpha, rows).map_err(ferr_to_pyerr)?;
        let out = arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
        reshape_mv(out, outer, k)
    }

    /// `Generator.multinomial(n, pvals, size=None)` — int64, each row sums
    /// to `n`; `n < 0` -> `ValueError` (R-DEV-2).
    #[pyo3(signature = (n, pvals, size = None))]
    fn multinomial<'py>(
        &mut self,
        py: Python<'py>,
        n: i64,
        pvals: Vec<f64>,
        size: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if n < 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("n < 0"));
        }
        let k = pvals.len();
        let (rows, outer) = resolve_mv_size(size)?;
        let arr = self
            .inner
            .multinomial(n as u64, &pvals, rows)
            .map_err(ferr_to_pyerr)?;
        let out = arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
        reshape_mv(out, outer, k)
    }

    /// `Generator.multivariate_normal(mean, cov, size=None)` — float64,
    /// shape `size + (len(mean),)`.
    #[pyo3(signature = (mean, cov, size = None))]
    fn multivariate_normal<'py>(
        &mut self,
        py: Python<'py>,
        mean: Vec<f64>,
        cov: &Bound<'py, PyAny>,
        size: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let d = mean.len();
        let cov_arr = as_ndarray(py, cov)?;
        let cov_flat: Vec<f64> = cov_arr
            .call_method1("reshape", ((-1isize,),))?
            .call_method0("tolist")?
            .extract()?;
        let (rows, outer) = resolve_mv_size(size)?;
        let arr = self
            .inner
            .multivariate_normal(&mean, &cov_flat, rows)
            .map_err(ferr_to_pyerr)?;
        let out = arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
        reshape_mv(out, outer, d)
    }

    /// `Generator.multivariate_hypergeometric(colors, nsample, size=None)`
    /// — int64, each row sums to `nsample`; negative inputs -> `ValueError`
    /// (R-DEV-2).
    #[pyo3(signature = (colors, nsample, size = None))]
    fn multivariate_hypergeometric<'py>(
        &mut self,
        py: Python<'py>,
        colors: Vec<i64>,
        nsample: i64,
        size: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if nsample < 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "nsample must be non-negative",
            ));
        }
        let mut colors_u = Vec::with_capacity(colors.len());
        for c in &colors {
            if *c < 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "colors must be non-negative",
                ));
            }
            colors_u.push(*c as u64);
        }
        let k = colors_u.len();
        let (rows, outer) = resolve_mv_size(size)?;
        let arr = self
            .inner
            .multivariate_hypergeometric(&colors_u, nsample as u64, rows)
            .map_err(ferr_to_pyerr)?;
        let out = arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
        reshape_mv(out, outer, k)
    }

    /// `Generator.logseries(p, size=None)` — int64 in `{1, 2, ...}`.
    #[pyo3(signature = (p, size = None))]
    fn logseries<'py>(
        &mut self,
        py: Python<'py>,
        p: f64,
        size: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let shape = shape_from_pyany(size)?;
        let arr: ArrayD<i64> = self
            .inner
            .logseries(p, shape.as_slice())
            .map_err(ferr_to_pyerr)?;
        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    /// `Generator.noncentral_chisquare(df, nonc, size=None)` — float64.
    #[pyo3(signature = (df, nonc, size = None))]
    fn noncentral_chisquare<'py>(
        &mut self,
        py: Python<'py>,
        df: f64,
        nonc: f64,
        size: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let shape = shape_from_pyany(size)?;
        let arr: ArrayD<f64> = self
            .inner
            .noncentral_chisquare(df, nonc, shape.as_slice())
            .map_err(ferr_to_pyerr)?;
        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    /// `Generator.noncentral_f(dfnum, dfden, nonc, size=None)` — float64.
    #[pyo3(signature = (dfnum, dfden, nonc, size = None))]
    fn noncentral_f<'py>(
        &mut self,
        py: Python<'py>,
        dfnum: f64,
        dfden: f64,
        nonc: f64,
        size: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let shape = shape_from_pyany(size)?;
        let arr: ArrayD<f64> = self
            .inner
            .noncentral_f(dfnum, dfden, nonc, shape.as_slice())
            .map_err(ferr_to_pyerr)?;
        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    /// `Generator.zipf(a, size=None)` — int64 in `{1, 2, ...}`, `a > 1`.
    #[pyo3(signature = (a, size = None))]
    fn zipf<'py>(
        &mut self,
        py: Python<'py>,
        a: f64,
        size: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let shape = shape_from_pyany(size)?;
        let arr: ArrayD<i64> = self
            .inner
            .zipf(a, shape.as_slice())
            .map_err(ferr_to_pyerr)?;
        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
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
    let (inner, resolved) = seed_generator_with_seed(seed)?;
    Ok(PyGenerator {
        inner,
        seed: resolved,
    })
}

// ---------------------------------------------------------------------------
// BitGenerator class family (refs #834 #818)
//
// numpy exposes a BitGenerator base plus the concrete streams MT19937 /
// PCG64 / PCG64DXSM / Philox / SFC64 (`numpy/random/__init__.py:181-191`).
// Each is a constructible object accepted as the `bit_generator` argument to
// `default_rng`. ferray's runtime PRNG is a single Xoshiro256** — there is
// no PCG64/MT19937 bit-stream behind these wrappers. They provide the numpy
// *API* (constructible with a seed, `.state` dict, `.seed_seq`, accepted by
// `default_rng`); the produced stream is ferray's Xoshiro seeded from the
// same seed, NOT numpy-bit-identical (R-DEV-7: the observable API contract is
// preserved; bit-stream identity across a different PRNG is out of scope and
// documented here).
// ---------------------------------------------------------------------------

/// Resolve a numpy-compatible BitGenerator `seed` argument
/// (`None | int | SeedSequence`) to a single deterministic `u64` seed for the
/// underlying Xoshiro256** stream. `None` draws from OS entropy via a fresh
/// `SeedSequence` so each unseeded BitGenerator is independent.
fn bitgen_seed_from(seed: Option<&Bound<'_, PyAny>>) -> PyResult<u64> {
    match seed {
        None => Ok(SeedSequence::new(os_entropy_u64()).generate_u64()),
        Some(o) => {
            if o.is_none() {
                return Ok(SeedSequence::new(os_entropy_u64()).generate_u64());
            }
            if let Ok(ss) = o.extract::<PyRef<'_, PySeedSequence>>() {
                return Ok(ss.inner.generate_u64());
            }
            if let Ok(s) = o.extract::<u64>() {
                return Ok(s);
            }
            // array_like / sequence seed -> fold through SeedSequence.
            let words: Vec<u64> = o.extract::<Vec<u64>>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "seed must be None, an int, a sequence of ints, or a SeedSequence",
                )
            })?;
            let mut entropy: u64 = 0;
            for (i, w) in words.iter().enumerate() {
                entropy ^= w
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .rotate_left((i as u32 % 64) + 1);
            }
            Ok(SeedSequence::new(entropy).generate_u64())
        }
    }
}

/// Draw a non-deterministic `u64` for an unseeded BitGenerator / SeedSequence.
/// Delegates to `ferray_random::default_rng()`, which seeds itself from OS
/// entropy (`generator.rs` `default_rng`), then pulls one `u64` — so each
/// unseeded object gets an independent stream without panicking at the
/// boundary (R-CODE-2).
fn os_entropy_u64() -> u64 {
    default_rng().next_u64()
}

/// `numpy.random.BitGenerator` family — a constructible PRNG-stream object
/// accepted by `default_rng`. The concrete numpy classes (`MT19937`,
/// `PCG64`, `PCG64DXSM`, `Philox`, `SFC64`) are registered as Python
/// subclasses of this base sharing one Rust `#[pyclass]`; `name` records
/// which numpy class was constructed so `.state["bit_generator"]` reports the
/// numpy-expected label.
#[pyclass(name = "BitGenerator", module = "ferray.random", subclass)]
pub struct PyBitGenerator {
    /// Resolved Xoshiro256** seed this BitGenerator wraps.
    seed: u64,
    /// numpy class label (`"PCG64"`, `"MT19937"`, …) for `.state`.
    name: String,
}

#[pymethods]
impl PyBitGenerator {
    #[new]
    #[pyo3(signature = (seed = None))]
    fn py_new(seed: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        Ok(Self {
            seed: bitgen_seed_from(seed)?,
            name: "BitGenerator".to_string(),
        })
    }

    fn __repr__(&self) -> String {
        format!("{}() <ferray Xoshiro256** stream>", self.name)
    }

    /// `BitGenerator.state` — numpy returns a dict describing the stream.
    /// ferray reports the numpy class label plus the resolved Xoshiro seed
    /// (the actual backing state), matching numpy's dict *shape*
    /// (`bit_generator` + `state` keys) without claiming bit-identity.
    #[getter]
    fn state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("bit_generator", &self.name)?;
        let inner = pyo3::types::PyDict::new(py);
        inner.set_item("seed", self.seed)?;
        d.set_item("state", inner)?;
        d.set_item("has_uint32", 0)?;
        d.set_item("uinteger", 0)?;
        Ok(d.into_any())
    }

    /// `BitGenerator.seed_seq` — the `SeedSequence` that initialised this
    /// stream (numpy exposes this on the modern BitGenerators).
    #[getter]
    fn seed_seq(&self) -> PySeedSequence {
        PySeedSequence {
            inner: SeedSequence::new(self.seed),
        }
    }
}

/// Macro defining a concrete numpy BitGenerator subclass over the shared
/// [`PyBitGenerator`] base. Each records its numpy class `name` for `.state`.
macro_rules! bitgen_subclass {
    ($cls:ident, $pyname:literal) => {
        #[pyclass(name = $pyname, module = "ferray.random", extends = PyBitGenerator)]
        pub struct $cls;

        #[pymethods]
        impl $cls {
            #[new]
            #[pyo3(signature = (seed = None))]
            fn py_new(seed: Option<&Bound<'_, PyAny>>) -> PyResult<(Self, PyBitGenerator)> {
                Ok((
                    $cls,
                    PyBitGenerator {
                        seed: bitgen_seed_from(seed)?,
                        name: $pyname.to_string(),
                    },
                ))
            }
        }
    };
}

bitgen_subclass!(PyMT19937, "MT19937");
bitgen_subclass!(PyPCG64, "PCG64");
bitgen_subclass!(PyPCG64DXSM, "PCG64DXSM");
bitgen_subclass!(PyPhilox, "Philox");
bitgen_subclass!(PySFC64, "SFC64");

impl PyBitGenerator {
    /// Construct a [`PyBitGenerator`] directly from a resolved Xoshiro seed,
    /// labelled with the numpy class `name`. Used to hand out the bit
    /// generator that backs the global state (`get_bit_generator`) and the
    /// one a `Generator`/`RandomState` wraps (`*.bit_generator`).
    fn from_seed(seed: u64, name: &str) -> Self {
        Self {
            seed,
            name: name.to_string(),
        }
    }

    /// Extract the resolved Xoshiro seed from any object numpy would accept as
    /// a bit generator: a ferray `PyBitGenerator` (or subclass) instance.
    /// `set_bit_generator(bitgen)` re-seeds the global from this value.
    fn seed_of(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
        obj.extract::<PyRef<'_, PyBitGenerator>>()
            .map(|bg| bg.seed)
            .map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "set_bit_generator requires a BitGenerator instance",
                )
            })
    }
}

// ---------------------------------------------------------------------------
// get_bit_generator / set_bit_generator — hot-swap the global bit generator
// (closes #910)
//
// numpy `numpy/random/mtrand.pyx:4838` `get_bit_generator()` returns the
// singleton RandomState's `_bit_generator` (numpy default: MT19937);
// `mtrand.pyx:4864` `set_bit_generator(bitgen)` installs `bitgen` on the
// singleton via `_initialize_bit_generator`. ferray's singleton is the
// thread-local Xoshiro256** `RNG`; we model its bit generator as a
// `PyBitGenerator` carrying the global's origin `seed` (labelled "MT19937" to
// match numpy's default singleton stream), and `set_bit_generator` re-seeds
// the global from the supplied bit generator's resolved seed. Round-trip
// `set_bit_generator(get_bit_generator())` reinstalls the identical
// reproducible stream (R-DEV-7: API/round-trip contract preserved; bit-stream
// identity with numpy's MT19937 is out of scope for a different PRNG).
// ---------------------------------------------------------------------------

/// `numpy.random.get_bit_generator()` — return the BitGenerator object
/// backing the global/legacy random state (`numpy/random/mtrand.pyx:4838`).
#[pyfunction]
pub fn get_bit_generator() -> PyBitGenerator {
    PyBitGenerator::from_seed(global_seed(), "MT19937")
}

/// `numpy.random.set_bit_generator(bitgen)` — install `bitgen` as the global
/// bit generator; subsequent top-level draws use it
/// (`numpy/random/mtrand.pyx:4864`). Returns `None`.
#[pyfunction]
pub fn set_bit_generator(bitgen: &Bound<'_, PyAny>) -> PyResult<()> {
    let seed = PyBitGenerator::seed_of(bitgen)?;
    reseed_global(seed);
    Ok(())
}

// ---------------------------------------------------------------------------
// SeedSequence class (refs #834 #818)
//
// numpy `numpy/random/__init__.py:186` `from .bit_generator import
// SeedSequence`. Wraps `ferray_random::SeedSequence` to expose `.entropy`,
// `.spawn(n)` and `.generate_state(n_words)`. The generated words are a
// deterministic function of `(entropy, spawn_key)` — reproducible, though not
// numpy-bit-identical.
// ---------------------------------------------------------------------------

/// `numpy.random.SeedSequence(entropy=None)` — entropy mixing + child spawn.
#[pyclass(name = "SeedSequence", module = "ferray.random")]
pub struct PySeedSequence {
    inner: SeedSequence,
}

#[pymethods]
impl PySeedSequence {
    #[new]
    #[pyo3(signature = (entropy = None))]
    fn py_new(entropy: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let e = match entropy {
            None => os_entropy_u64(),
            Some(o) if o.is_none() => os_entropy_u64(),
            Some(o) => {
                if let Ok(v) = o.extract::<u64>() {
                    v
                } else {
                    // sequence entropy: fold to a single u64.
                    let words: Vec<u64> = o.extract::<Vec<u64>>().map_err(|_| {
                        pyo3::exceptions::PyTypeError::new_err(
                            "entropy must be None, an int, or a sequence of ints",
                        )
                    })?;
                    let mut acc: u64 = 0;
                    for (i, w) in words.iter().enumerate() {
                        acc ^= w
                            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                            .rotate_left((i as u32 % 64) + 1);
                    }
                    acc
                }
            }
        };
        Ok(Self {
            inner: SeedSequence::new(e),
        })
    }

    fn __repr__(&self) -> String {
        format!("SeedSequence(entropy={})", self.inner.entropy())
    }

    /// `SeedSequence.entropy` — the root entropy value.
    #[getter]
    fn entropy(&self) -> u64 {
        self.inner.entropy()
    }

    /// `SeedSequence.spawn_key` — the spawn path (tuple of ints) identifying
    /// this child within the spawn tree.
    #[getter]
    fn spawn_key<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(pyo3::types::PyTuple::new(py, self.inner.spawn_key())?.into_any())
    }

    /// `SeedSequence.generate_state(n_words, dtype=uint32)` — `n_words`
    /// deterministic uint32 words as a numpy `uint32` array.
    #[pyo3(signature = (n_words, dtype = None))]
    fn generate_state<'py>(
        &self,
        py: Python<'py>,
        n_words: isize,
        dtype: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = dtype;
        if n_words < 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n_words must be non-negative",
            ));
        }
        let words: Vec<u32> = self.inner.generate_state(n_words as usize);
        let arr = Array1::<u32>::from_vec(ferray_core::dimension::Ix1::new([words.len()]), words)
            .map_err(ferr_to_pyerr)?;
        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    /// `SeedSequence.spawn(n)` — return `n` independent child SeedSequences.
    fn spawn(&mut self, n: usize) -> Vec<PySeedSequence> {
        self.inner
            .spawn(n)
            .into_iter()
            .map(|inner| PySeedSequence { inner })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// RandomState — the legacy numpy.random API (refs #834 #818)
//
// numpy `numpy/random/__init__.py:188` `from .mtrand import RandomState`.
// Wraps a ferray `Generator<Xoshiro256**>` and exposes the legacy method
// surface (`rand`, `randn`, `randint`, `random`/`random_sample`,
// `standard_normal`, `normal`, `uniform`, `choice`, `shuffle`,
// `permutation`, `seed`, `get_state`/`set_state`, plus common
// distributions). Most methods delegate to the same library calls the
// module-level functions and `PyGenerator` use. Legacy `RandomState` uses
// MT19937 in numpy; ferray backs it with Xoshiro256** — reproducible per
// seed but NOT numpy-bit-identical (documented limitation).
// ---------------------------------------------------------------------------

/// `numpy.random.RandomState(seed=None)` — the legacy stateful generator.
#[pyclass(name = "RandomState", module = "ferray.random")]
pub struct PyRandomState {
    inner: Generator<Xoshiro256StarStar>,
    /// Resolved Xoshiro seed this RandomState wraps — exposed via the
    /// `_bit_generator` getter (numpy `RandomState._bit_generator`,
    /// `mtrand.pyi:86`; numpy default: MT19937).
    seed: u64,
}

#[pymethods]
impl PyRandomState {
    #[new]
    #[pyo3(signature = (seed = None))]
    fn py_new(seed: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let (inner, resolved) = seed_generator_with_seed(seed)?;
        Ok(Self {
            inner,
            seed: resolved,
        })
    }

    fn __repr__(&self) -> String {
        "RandomState(Xoshiro256**)".to_string()
    }

    /// `RandomState._bit_generator` — the BitGenerator this legacy state wraps
    /// (numpy `mtrand.pyi:86` `_bit_generator: BitGenerator`; numpy's legacy
    /// singleton uses MT19937). `numpy.random.get_bit_generator()` reads the
    /// singleton's `_bit_generator`, so the label matches.
    #[getter]
    fn _bit_generator(&self) -> PyBitGenerator {
        PyBitGenerator::from_seed(self.seed, "MT19937")
    }

    /// `RandomState.seed(seed=None)` — re-seed in place.
    #[pyo3(signature = (seed = None))]
    fn seed(&mut self, seed: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let (inner, resolved) = seed_generator_with_seed(seed)?;
        self.inner = inner;
        self.seed = resolved;
        Ok(())
    }

    /// `RandomState.rand(*shape)` — uniform [0, 1); scalar for no args.
    #[pyo3(signature = (*shape))]
    fn rand<'py>(
        &mut self,
        py: Python<'py>,
        shape: &Bound<'py, pyo3::types::PyTuple>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if shape.is_empty() {
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
        let dims = varargs_dims(shape)?;
        let arr: ArrayD<f64> = self.inner.random(dims.as_slice()).map_err(ferr_to_pyerr)?;
        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    /// `RandomState.randn(*shape)` — standard normal; scalar for no args.
    #[pyo3(signature = (*shape))]
    fn randn<'py>(
        &mut self,
        py: Python<'py>,
        shape: &Bound<'py, pyo3::types::PyTuple>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if shape.is_empty() {
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
        let dims = varargs_dims(shape)?;
        let arr: ArrayD<f64> = self
            .inner
            .standard_normal(dims.as_slice())
            .map_err(ferr_to_pyerr)?;
        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    /// `RandomState.random_sample(size=None)` — uniform [0, 1); scalar for
    /// `size=None`.
    #[pyo3(signature = (size = None))]
    fn random_sample<'py>(
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
        let s = shape_from_pyany(size)?;
        let arr: ArrayD<f64> = self.inner.random(s.as_slice()).map_err(ferr_to_pyerr)?;
        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    /// `RandomState.random(size=None)` — alias of `random_sample`.
    #[pyo3(signature = (size = None))]
    fn random<'py>(
        &mut self,
        py: Python<'py>,
        size: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.random_sample(py, size)
    }

    /// `RandomState.standard_normal(size=None)`; scalar for `size=None`.
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
        let s = shape_from_pyany(size)?;
        let arr: ArrayD<f64> = self
            .inner
            .standard_normal(s.as_slice())
            .map_err(ferr_to_pyerr)?;
        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    /// `RandomState.normal(loc=0, scale=1, size=None)`.
    #[pyo3(signature = (loc = 0.0, scale = 1.0, size = None))]
    fn normal<'py>(
        &mut self,
        py: Python<'py>,
        loc: f64,
        scale: f64,
        size: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if size.is_none() {
            let v = self
                .inner
                .normal(loc, scale, 1usize)
                .map_err(ferr_to_pyerr)?
                .iter()
                .next()
                .copied()
                .unwrap_or(0.0);
            return Ok(v.into_pyobject(py)?.into_any());
        }
        let s = shape_from_pyany(size)?;
        let arr: ArrayD<f64> = self
            .inner
            .normal(loc, scale, s.as_slice())
            .map_err(ferr_to_pyerr)?;
        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    /// `RandomState.uniform(low=0, high=1, size=None)`.
    #[pyo3(signature = (low = 0.0, high = 1.0, size = None))]
    fn uniform<'py>(
        &mut self,
        py: Python<'py>,
        low: f64,
        high: f64,
        size: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if size.is_none() {
            let v = self
                .inner
                .uniform(low, high, 1usize)
                .map_err(ferr_to_pyerr)?
                .iter()
                .next()
                .copied()
                .unwrap_or(0.0);
            return Ok(v.into_pyobject(py)?.into_any());
        }
        let s = shape_from_pyany(size)?;
        let arr: ArrayD<f64> = self
            .inner
            .uniform(low, high, s.as_slice())
            .map_err(ferr_to_pyerr)?;
        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    /// `RandomState.randint(low, high=None, size=None)` — half-open
    /// `[low, high)`; scalar for `size=None`.
    #[pyo3(signature = (low, high = None, size = None))]
    fn randint<'py>(
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
        if size.is_none() {
            let v: i64 = self
                .inner
                .integers(lo, hi, 1usize)
                .map_err(ferr_to_pyerr)?
                .iter()
                .next()
                .copied()
                .unwrap_or(0);
            return Ok(v.into_pyobject(py)?.into_any());
        }
        let s = shape_from_pyany(size)?;
        let arr: ArrayD<i64> = self
            .inner
            .integers(lo, hi, s.as_slice())
            .map_err(ferr_to_pyerr)?;
        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    /// `RandomState.tomaxint`-free alias: `random_integers(low, high=None,
    /// size=None)` — inclusive `[low, high]` (or `[1, low]`).
    #[pyo3(signature = (low, high = None, size = None))]
    fn random_integers<'py>(
        &mut self,
        py: Python<'py>,
        low: i64,
        high: Option<i64>,
        size: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (lo, hi_excl) = match high {
            Some(h) => (low, h.saturating_add(1)),
            None => (1, low.saturating_add(1)),
        };
        self.randint(py, lo, Some(hi_excl), size)
    }

    /// `RandomState.exponential(scale=1.0, size=None)`.
    #[pyo3(signature = (scale = 1.0, size = None))]
    fn exponential<'py>(
        &mut self,
        py: Python<'py>,
        scale: f64,
        size: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let s = shape_from_pyany(size)?;
        let arr: ArrayD<f64> = self
            .inner
            .exponential(scale, s.as_slice())
            .map_err(ferr_to_pyerr)?;
        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    /// `RandomState.poisson(lam=1.0, size=None)` — int64.
    #[pyo3(signature = (lam = 1.0, size = None))]
    fn poisson<'py>(
        &mut self,
        py: Python<'py>,
        lam: f64,
        size: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let s = shape_from_pyany(size)?;
        let arr: ArrayD<i64> = self
            .inner
            .poisson(lam, s.as_slice())
            .map_err(ferr_to_pyerr)?;
        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    /// `RandomState.binomial(n, p, size=None)` — `n < 0` -> `ValueError`.
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
        let s = shape_from_pyany(size)?;
        let arr: ArrayD<i64> = self
            .inner
            .binomial(n as u64, p, s.as_slice())
            .map_err(ferr_to_pyerr)?;
        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    /// `RandomState.choice(a, size=None, replace=True, p=None)`.
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

    /// `RandomState.shuffle(x)` — shuffle a 1-D array in place, return `None`.
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
            let len = slice.len();
            for i in (1..len).rev() {
                let j = self.inner.next_u64_bounded((i + 1) as u64) as usize;
                slice.swap(i, j);
            }
        });
        Ok(())
    }

    /// `RandomState.permutation(x)` — permute `range(n)` for an int, or an
    /// array along axis 0.
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

    /// `RandomState.get_state()` — legacy state tuple. numpy returns
    /// `(str, ndarray, int, int, float)`; ferray reports the backing PRNG
    /// label plus its raw state bytes as a uint8 array — round-trippable via
    /// `set_state` (NOT numpy-bit-compatible; documented limitation).
    fn get_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let bytes = self.inner.state_bytes().map_err(ferr_to_pyerr)?;
        let arr = Array1::<u8>::from_vec(ferray_core::dimension::Ix1::new([bytes.len()]), bytes)
            .map_err(ferr_to_pyerr)?;
        let state_arr = arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
        let tuple = pyo3::types::PyTuple::new(
            py,
            [
                "Xoshiro256**".into_pyobject(py)?.into_any(),
                state_arr,
                0i64.into_pyobject(py)?.into_any(),
                0i64.into_pyobject(py)?.into_any(),
                0.0f64.into_pyobject(py)?.into_any(),
            ],
        )?;
        Ok(tuple.into_any())
    }

    /// `RandomState.set_state(state)` — restore from a `get_state` tuple.
    fn set_state(&mut self, state: &Bound<'_, PyAny>) -> PyResult<()> {
        let arr = state.get_item(1)?;
        let view: PyReadonlyArray1<u8> = arr.extract().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(
                "state[1] must be a 1-D uint8 array from get_state",
            )
        })?;
        let bytes = view.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("state array must be contiguous")
        })?;
        self.inner.set_state_bytes(bytes).map_err(ferr_to_pyerr)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Module-level legacy get_state / set_state over the thread-local RNG
// (`numpy/random/__init__.py:137,164`).
// ---------------------------------------------------------------------------

/// `numpy.random.get_state()` — legacy accessor over the process-wide RNG.
/// Returns the same tuple shape as `RandomState.get_state`.
#[pyfunction]
pub fn get_state<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    let bytes = with_rng(|rng| rng.state_bytes()).map_err(ferr_to_pyerr)?;
    let arr = Array1::<u8>::from_vec(ferray_core::dimension::Ix1::new([bytes.len()]), bytes)
        .map_err(ferr_to_pyerr)?;
    let state_arr = arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    let tuple = pyo3::types::PyTuple::new(
        py,
        [
            "Xoshiro256**".into_pyobject(py)?.into_any(),
            state_arr,
            0i64.into_pyobject(py)?.into_any(),
            0i64.into_pyobject(py)?.into_any(),
            0.0f64.into_pyobject(py)?.into_any(),
        ],
    )?;
    Ok(tuple.into_any())
}

/// `numpy.random.set_state(state)` — legacy setter over the process-wide RNG.
#[pyfunction]
pub fn set_state(state: &Bound<'_, PyAny>) -> PyResult<()> {
    let arr = state.get_item(1)?;
    let view: PyReadonlyArray1<u8> = arr.extract().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("state[1] must be a 1-D uint8 array from get_state")
    })?;
    let bytes = view
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("state array must be contiguous"))?;
    with_rng(|rng| rng.set_state_bytes(bytes)).map_err(ferr_to_pyerr)?;
    Ok(())
}
