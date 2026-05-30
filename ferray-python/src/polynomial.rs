//! Bindings for the `numpy.polynomial` submodule.
//!
//! NumPy's polynomial namespace has two layers:
//!
//! 1. **Class-based API** — `numpy.polynomial.Polynomial`,
//!    `numpy.polynomial.Chebyshev`, etc. Filed as a follow-up; needs
//!    `#[pyclass]` wrappers around ferray's `Polynomial` /
//!    `Chebyshev` / `Hermite` / `HermiteE` / `Laguerre` /
//!    `Legendre` types.
//! 2. **Function-style API** — `polyval2d`, `chebgauss`, `poly2cheb`,
//!    etc. on flat coefficient arrays. That's what we expose here.
//!
//! This module additionally hosts the **top-level numpy poly1d family**
//! (`numpy.polyval` / `poly` / `roots` / `polyadd` / `polysub` / `polymul`
//! / `polyder` / `polyint` / `polyfit` / `polydiv`, registered at the
//! `ferray` root, NOT under `ferray.polynomial`). These are the CLASSIC 1-D
//! polynomial functions on flat coefficient arrays in
//! `numpy/lib/_polynomial_impl.py` — and crucially use HIGHEST-degree-first
//! coefficient order, the opposite of `numpy.polynomial.polynomial.*`. They
//! marshal onto ferray-polynomial's lowest-first power-basis `Polynomial`
//! math by reversing coefficient arrays at the boundary; see the
//! `flip`/poly1d section below.
//!
//! Everything is `f64`-only at the ferray layer (the `extras` module
//! is single-precision in name only — internally f64). Callers
//! passing `float32` arrays get them coerced to `float64` for the
//! computation, matching NumPy's promotion behaviour for these
//! reductions.

use ferray_core::array::aliases::{Array1, Array2, ArrayD};
use ferray_core::dimension::{Ix1, Ix2, IxDyn};
use ferray_numpy_interop::IntoNumPy;
use ferray_polynomial as fp;
use ferray_polynomial::traits::{FromPowerBasis, Poly, ToPowerBasis};
use numpy::PyReadonlyArrayDyn;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyTuple};

use crate::conv::{as_ndarray, complex_roots_to_pyarray, ferr_to_pyerr};

// ---------------------------------------------------------------------------
// Helpers — f64 vector → numpy.ndarray
// ---------------------------------------------------------------------------

fn vec_to_pyarray1<'py>(py: Python<'py>, v: Vec<f64>) -> PyResult<Bound<'py, PyAny>> {
    let n = v.len();
    let arr = Array1::<f64>::from_vec(Ix1::new([n]), v).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

fn vec_to_pyarray_2d<'py>(
    py: Python<'py>,
    v: Vec<f64>,
    shape: (usize, usize),
) -> PyResult<Bound<'py, PyAny>> {
    let arr = Array2::<f64>::from_vec(Ix2::new([shape.0, shape.1]), v).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

fn vec_to_pyarray_dyn<'py>(
    py: Python<'py>,
    v: Vec<f64>,
    shape: Vec<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = ArrayD::<f64>::from_vec(IxDyn::new(&shape), v).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

// ---------------------------------------------------------------------------
// polyvalfromroots / polyvander
// ---------------------------------------------------------------------------

/// `numpy.polynomial.polynomial.polyvalfromroots(x, r)` for a single
/// scalar `x`.
#[pyfunction]
pub fn polyvalfromroots(x: f64, roots: Vec<f64>) -> f64 {
    fp::polyvalfromroots(x, &roots)
}

/// `numpy.polynomial.polynomial.polyval2d(x, y, c)` — evaluate a 2-D
/// polynomial at points `(x[i], y[i])`.
#[pyfunction]
pub fn polyval2d<'py>(
    py: Python<'py>,
    x: Vec<f64>,
    y: Vec<f64>,
    coeffs: Vec<f64>,
    nx: usize,
    ny: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let v = fp::polyval2d(&x, &y, &coeffs, nx, ny).map_err(ferr_to_pyerr)?;
    vec_to_pyarray1(py, v)
}

/// `numpy.polynomial.polynomial.polyval3d(x, y, z, c)`.
///
/// 8 args (3 coordinate arrays + flattened coeffs + 3 axis sizes + py)
/// is dictated by the NumPy API shape — flattening to fewer params
/// would just hide the size info that callers need to pass anyway.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn polyval3d<'py>(
    py: Python<'py>,
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    coeffs: Vec<f64>,
    nx: usize,
    ny: usize,
    nz: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let v = fp::polyval3d(&x, &y, &z, &coeffs, nx, ny, nz).map_err(ferr_to_pyerr)?;
    vec_to_pyarray1(py, v)
}

/// `numpy.polynomial.polynomial.polygrid2d(x, y, c)`.
#[pyfunction]
pub fn polygrid2d<'py>(
    py: Python<'py>,
    x: Vec<f64>,
    y: Vec<f64>,
    coeffs: Vec<f64>,
    nx: usize,
    ny: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let nx_pts = x.len();
    let ny_pts = y.len();
    let v = fp::polygrid2d(&x, &y, &coeffs, nx, ny).map_err(ferr_to_pyerr)?;
    vec_to_pyarray_2d(py, v, (nx_pts, ny_pts))
}

/// `numpy.polynomial.polynomial.polygrid3d(x, y, z, c)`.
///
/// See `polyval3d` for the rationale on the argument count.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn polygrid3d<'py>(
    py: Python<'py>,
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    coeffs: Vec<f64>,
    nx: usize,
    ny: usize,
    nz: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let nx_pts = x.len();
    let ny_pts = y.len();
    let nz_pts = z.len();
    let v = fp::polygrid3d(&x, &y, &z, &coeffs, nx, ny, nz).map_err(ferr_to_pyerr)?;
    vec_to_pyarray_dyn(py, v, vec![nx_pts, ny_pts, nz_pts])
}

/// `numpy.polynomial.polynomial.polyvander2d(x, y, deg)`.
#[pyfunction]
pub fn polyvander2d<'py>(
    py: Python<'py>,
    x: Vec<f64>,
    y: Vec<f64>,
    degx: usize,
    degy: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let n = x.len();
    let cols = (degx + 1) * (degy + 1);
    let v = fp::polyvander2d(&x, &y, degx, degy).map_err(ferr_to_pyerr)?;
    vec_to_pyarray_2d(py, v, (n, cols))
}

/// `numpy.polynomial.polynomial.polyvander3d(x, y, z, deg)`.
#[pyfunction]
pub fn polyvander3d<'py>(
    py: Python<'py>,
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    degx: usize,
    degy: usize,
    degz: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let n = x.len();
    let cols = (degx + 1) * (degy + 1) * (degz + 1);
    let v = fp::polyvander3d(&x, &y, &z, degx, degy, degz).map_err(ferr_to_pyerr)?;
    vec_to_pyarray_2d(py, v, (n, cols))
}

// ---------------------------------------------------------------------------
// Chebyshev sample points / weights
// ---------------------------------------------------------------------------

/// `numpy.polynomial.chebyshev.chebpts1(n)` — Chebyshev points of the
/// first kind.
///
/// ferray-Rust returns the points in the natural mathematical
/// (descending) order; NumPy returns them ascending. We reverse here
/// for Python-side parity.
#[pyfunction]
pub fn chebpts1<'py>(py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyAny>> {
    let mut v = fp::chebpts1(n).map_err(ferr_to_pyerr)?;
    v.reverse();
    vec_to_pyarray1(py, v)
}

/// `numpy.polynomial.chebyshev.chebpts2(n)` — Chebyshev points of the
/// second kind. Reversed to match NumPy's ascending convention.
#[pyfunction]
pub fn chebpts2<'py>(py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyAny>> {
    let mut v = fp::chebpts2(n).map_err(ferr_to_pyerr)?;
    v.reverse();
    vec_to_pyarray1(py, v)
}

/// `numpy.polynomial.chebyshev.chebweight(x)` — Chebyshev weight at x.
#[pyfunction]
pub fn chebweight(x: f64) -> f64 {
    fp::chebweight(x)
}

// ---------------------------------------------------------------------------
// Gauss quadrature
// ---------------------------------------------------------------------------

macro_rules! bind_gauss {
    ($name:ident, $ferr_path:path) => {
        #[pyfunction]
        pub fn $name<'py>(py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyAny>> {
            let (pts, weights) = $ferr_path(n).map_err(ferr_to_pyerr)?;
            let pts_py = vec_to_pyarray1(py, pts)?;
            let w_py = vec_to_pyarray1(py, weights)?;
            Ok(PyTuple::new(py, [pts_py, w_py])?.into_any())
        }
    };
}

bind_gauss!(chebgauss, fp::chebgauss);
bind_gauss!(leggauss, fp::leggauss);
bind_gauss!(hermgauss, fp::hermgauss);
bind_gauss!(hermegauss, fp::hermegauss);
bind_gauss!(laggauss, fp::laggauss);

// ---------------------------------------------------------------------------
// Basis conversions: power ↔ chebyshev / hermite / hermite-e / laguerre / legendre
// ---------------------------------------------------------------------------

macro_rules! bind_basis_conv {
    ($name:ident, $ferr_path:path) => {
        #[pyfunction]
        pub fn $name<'py>(py: Python<'py>, coeffs: Vec<f64>) -> PyResult<Bound<'py, PyAny>> {
            let v = $ferr_path(&coeffs).map_err(ferr_to_pyerr)?;
            vec_to_pyarray1(py, v)
        }
    };
}

bind_basis_conv!(poly2cheb, fp::poly2cheb);
bind_basis_conv!(cheb2poly, fp::cheb2poly);
bind_basis_conv!(poly2herm, fp::poly2herm);
bind_basis_conv!(herm2poly, fp::herm2poly);
bind_basis_conv!(poly2herme, fp::poly2herme);
bind_basis_conv!(herme2poly, fp::herme2poly);
bind_basis_conv!(poly2lag, fp::poly2lag);
bind_basis_conv!(lag2poly, fp::lag2poly);
bind_basis_conv!(poly2leg, fp::poly2leg);
bind_basis_conv!(leg2poly, fp::leg2poly);

// ---------------------------------------------------------------------------
// Polynomial class API (NumPy: numpy.polynomial.Polynomial / Chebyshev / …)
// ---------------------------------------------------------------------------
//
// Each class wraps the corresponding ferray-Rust polynomial type. The
// six classes share an identical Python-facing interface (constructor,
// `__call__`, `deriv`, `integ`, `roots`, `degree`, `coef`, arithmetic
// dunder methods, basis-conversion helpers), so a macro generates them
// uniformly.

/// A `degree` return value that is BOTH callable and integer-comparable.
///
/// `numpy.polynomial`'s `degree` is a method — `numpy/polynomial/_polybase.py:670`
/// `def degree(self):` returning `len(self) - 1` (no trailing-zero trim). So
/// `p.degree()` is a call. ferray's adversarial pins exercise `degree` two
/// ways: `divergence_polynomial.py::test_degree_is_callable_method` requires
/// `p.degree()` to be callable, while `::test_degree_counts_trailing_zeros`
/// reads `p.degree` as a value and compares it to `len(coef)-1`. A bare
/// `#[getter] -> usize` (the previous binding) satisfies neither contract
/// (`p.degree()` raises "int not callable"; the value was the *trimmed*
/// degree). This wrapper carries the untrimmed `len(coef)-1` integer and
/// exposes `__call__` (returns the same int), `__eq__` / `__int__` /
/// `__index__` (so it compares and indexes as that int), and `__repr__`.
#[pyclass(module = "ferray.polynomial", skip_from_py_object)]
#[derive(Clone, Copy)]
pub struct PolyDegree {
    value: usize,
}

#[pymethods]
impl PolyDegree {
    /// `p.degree()` — return the integer degree (numpy's method contract).
    fn __call__(&self) -> usize {
        self.value
    }

    fn __int__(&self) -> usize {
        self.value
    }

    fn __index__(&self) -> usize {
        self.value
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other.extract::<usize>().is_ok_and(|n| n == self.value)
    }

    fn __hash__(&self) -> u64 {
        self.value as u64
    }

    fn __repr__(&self) -> String {
        self.value.to_string()
    }
}

/// Render `coef` / `domain` / `window` exactly as numpy's
/// `BasePolynomial.__repr__` does.
///
/// `numpy/polynomial/_polybase.py:322-328` builds the repr from
/// `repr(self.coef)[6:-1]` (etc.) — i.e. numpy's array repr with the
/// leading `array(` and trailing `)` stripped — then assembles
/// `f"{name}({coef}, domain={domain}, window={window}, symbol='{symbol}')"`.
/// Delegating each component to `repr(np.asarray(...))` reproduces numpy's
/// exact float formatting (`[1., 2., 3.]`) rather than Rust `Debug`
/// (`[1.0, 2.0, 3.0]`).
fn numpy_array_repr_inner(py: Python<'_>, values: &[f64]) -> PyResult<String> {
    let arr = numpy::PyArray1::<f64>::from_slice(py, values);
    let full: String = arr.as_any().repr()?.extract()?;
    // Strip the `array(` prefix and the trailing `)`.
    Ok(full
        .strip_prefix("array(")
        .and_then(|s| s.strip_suffix(')'))
        .unwrap_or(&full)
        .to_string())
}

/// Coerce a `domain`/`window` kwarg to a `[f64; 2]`, raising the numpy
/// `ValueError` when it is not a two-element sequence.
///
/// `numpy/polynomial/_polybase.py:297-307` — `if len(domain) != 2: raise
/// ValueError("Domain has wrong number of elements.")` (and likewise for the
/// window). `which` is `"Domain"` or `"Window"` so the message matches numpy.
fn to_pair(v: &[f64], which: &str) -> PyResult<[f64; 2]> {
    if v.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{which} has wrong number of elements."
        )));
    }
    Ok([v[0], v[1]])
}

/// Coerce a Python value (scalar or array-like) into a flat `Vec<f64>`
/// for vectorised evaluation. Returns the original shape so the caller
/// can reshape the result back.
fn as_eval_input<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
) -> PyResult<(Vec<f64>, Option<Vec<usize>>)> {
    if let Ok(f) = x.extract::<f64>() {
        return Ok((vec![f], None));
    }
    let arr = as_ndarray(py, x)?;
    let cast = py
        .import("numpy")?
        .call_method1("asarray", (&arr, "float64"))?;
    let view: PyReadonlyArrayDyn<f64> = cast.extract()?;
    let nd = view.as_array();
    let shape: Vec<usize> = nd.shape().to_vec();
    let data: Vec<f64> = nd.iter().copied().collect();
    Ok((data, Some(shape)))
}

/// Generate a `#[pyclass]` wrapper around a ferray polynomial type.
///
/// The wrapped type must implement `Poly` (for arithmetic, evaluation,
/// derivative, integration, roots, etc.) and `ToPowerBasis` /
/// `FromPowerBasis` (for the cross-basis conversion helpers).
macro_rules! poly_class {
    ($PyName:ident, $RustType:ty, $module_str:literal) => {
        /// Numpy-compatible polynomial class wrapper. See `numpy.polynomial`
        /// for the API; ferray's implementation lives in `ferray-polynomial`.
        // `from_py_object` opts into the FromPyObject derive — needed
        // so `__add__(&self, other: &Self)` and related dunders can
        // accept another instance of the same class as a function arg.
        #[pyclass(module = $module_str, from_py_object)]
        #[derive(Clone)]
        pub struct $PyName {
            inner: $RustType,
            /// Symbol for the independent variable (numpy default `'x'`).
            /// Stored Python-side only — ferray's math is symbol-agnostic.
            /// `numpy/polynomial/_polybase.py:320` `self._symbol = symbol`.
            symbol: String,
        }

        #[pymethods]
        impl $PyName {
            // `numpy/polynomial/_polybase.py:292`
            // `def __init__(self, coef, domain=None, window=None, symbol='x')`.
            // `domain`/`window` override the basis mapping (threaded through
            // the inner polynomial's `with_mapping`); `symbol` is stored
            // Python-side. A `domain`/`window` of the wrong length raises
            // `ValueError` (`_polybase.py:297-307`).
            #[new]
            #[pyo3(signature = (coef = vec![0.0], domain = None, window = None, symbol = "x".to_string()))]
            fn py_new(
                coef: Vec<f64>,
                domain: Option<Vec<f64>>,
                window: Option<Vec<f64>>,
                symbol: String,
            ) -> PyResult<Self> {
                let mut inner = <$RustType>::new(&coef);
                if domain.is_some() || window.is_some() {
                    let d = match domain {
                        Some(ref v) => to_pair(v, "Domain")?,
                        None => inner.domain(),
                    };
                    let w = match window {
                        Some(ref v) => to_pair(v, "Window")?,
                        None => inner.window(),
                    };
                    inner = inner.with_mapping(d, w).map_err(ferr_to_pyerr)?;
                }
                Ok(Self { inner, symbol })
            }

            /// Polynomial coefficients in this basis (ascending order).
            #[getter]
            fn coef<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                let v = self.inner.coeffs().to_vec();
                let n = v.len();
                let arr = Array1::<f64>::from_vec(Ix1::new([n]), v).map_err(ferr_to_pyerr)?;
                Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
            }

            /// Polynomial degree — a callable returning `len(coef) - 1`.
            ///
            /// `numpy/polynomial/_polybase.py:670` `def degree(self): return
            /// len(self) - 1`. numpy's `degree` is a METHOD (so `p.degree()`)
            /// and does NOT trim trailing zeros, so `[1,2,0,0]` has degree 3.
            /// We return a [`PolyDegree`] so the value is both callable
            /// (`p.degree()`) and integer-comparable (`p.degree == 3`).
            #[getter]
            fn degree(&self) -> PolyDegree {
                PolyDegree {
                    value: self.inner.coeffs().len().saturating_sub(1),
                }
            }

            /// Symbol for the independent variable (numpy default `'x'`).
            /// `numpy/polynomial/_polybase.py:320` `self._symbol = symbol`.
            #[getter]
            fn symbol(&self) -> String {
                self.symbol.clone()
            }

            /// Two-element array `[low, high]` describing the polynomial's
            /// domain (NumPy convention).
            #[getter]
            fn domain<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                let d = self.inner.domain();
                let arr =
                    Array1::<f64>::from_vec(Ix1::new([2]), d.to_vec()).map_err(ferr_to_pyerr)?;
                Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
            }

            /// Two-element array `[low, high]` describing the window.
            #[getter]
            fn window<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                let w = self.inner.window();
                let arr =
                    Array1::<f64>::from_vec(Ix1::new([2]), w.to_vec()).map_err(ferr_to_pyerr)?;
                Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
            }

            /// Evaluate the polynomial. `x` may be a scalar or any
            /// array-like; the result follows the input's shape.
            fn __call__<'py>(
                &self,
                py: Python<'py>,
                x: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyAny>> {
                let (data, shape) = as_eval_input(py, x)?;
                let result = self.inner.eval_many(&data).map_err(ferr_to_pyerr)?;
                match shape {
                    None => {
                        // Scalar input — numpy returns a numpy SCALAR
                        // (np.float64), not a bare Python float
                        // (`numpy/polynomial/_polybase.py:510-512` `__call__`
                        // routes through `self._val`). Build a 0-d array and
                        // collapse it via `scalarize` to honour the
                        // numpy-scalar contract across the boundary (R-CODE-4).
                        let arr = ArrayD::<f64>::from_vec(IxDyn::new(&[]), vec![result[0]])
                            .map_err(ferr_to_pyerr)?;
                        let zerod = arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
                        crate::conv::scalarize(zerod)
                    }
                    Some(shape) => {
                        let arr = ArrayD::<f64>::from_vec(IxDyn::new(&shape), result)
                            .map_err(ferr_to_pyerr)?;
                        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
                    }
                }
            }

            /// Return a polynomial differentiated `m` times.
            #[pyo3(signature = (m = 1))]
            fn deriv(&self, m: usize) -> PyResult<Self> {
                Ok(self.rewrap(self.inner.deriv(m).map_err(ferr_to_pyerr)?))
            }

            /// Return a polynomial integrated `m` times with constants `k`.
            #[pyo3(signature = (m = 1, k = vec![]))]
            fn integ(&self, m: usize, k: Vec<f64>) -> PyResult<Self> {
                Ok(self.rewrap(self.inner.integ(m, &k).map_err(ferr_to_pyerr)?))
            }

            /// Roots of the polynomial. Returns a REAL (`float64`) ndarray
            /// when every root's imaginary part is zero, else `complex128`
            /// (`numpy/polynomial/polynomial.py:1606` `_to_real_if_imag_zero`).
            fn roots<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                let v = self.inner.roots().map_err(ferr_to_pyerr)?;
                complex_roots_to_pyarray(py, v)
            }

            /// Trim trailing coefficients smaller than `tol`.
            #[pyo3(signature = (tol = 0.0))]
            fn trim(&self, tol: f64) -> PyResult<Self> {
                Ok(self.rewrap(self.inner.trim(tol).map_err(ferr_to_pyerr)?))
            }

            /// Truncate to a series of length `size`.
            fn truncate(&self, size: usize) -> PyResult<Self> {
                Ok(self.rewrap(self.inner.truncate(size).map_err(ferr_to_pyerr)?))
            }

            /// Convert to power-basis coefficients.
            fn convert_to_power<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                let v = self.inner.to_power_basis().map_err(ferr_to_pyerr)?;
                vec_to_pyarray1(py, v)
            }

            // --- Arithmetic dunders ------------------------------------

            fn __add__(&self, other: &Self) -> PyResult<Self> {
                Ok(self.rewrap(self.inner.add(&other.inner).map_err(ferr_to_pyerr)?))
            }
            fn __sub__(&self, other: &Self) -> PyResult<Self> {
                Ok(self.rewrap(self.inner.sub(&other.inner).map_err(ferr_to_pyerr)?))
            }
            fn __mul__(&self, other: &Self) -> PyResult<Self> {
                Ok(self.rewrap(self.inner.mul(&other.inner).map_err(ferr_to_pyerr)?))
            }
            fn __pow__(&self, n: usize, _modulo: Option<usize>) -> PyResult<Self> {
                Ok(self.rewrap(self.inner.pow(n).map_err(ferr_to_pyerr)?))
            }

            /// `divmod(a, b)` — quotient and remainder of polynomial
            /// division. numpy implements the Python builtin via
            /// `numpy/polynomial/_polybase.py:577` `def __divmod__(self,
            /// other)`. Both forms (`divmod(a, b)` and `a.divmod(b)`) route
            /// here; the quotient/remainder carry `self`'s domain/window/symbol
            /// (`_polybase.py:582-583`).
            fn __divmod__(&self, other: &Self) -> PyResult<(Self, Self)> {
                let (q, r) = self.inner.divmod(&other.inner).map_err(ferr_to_pyerr)?;
                Ok((self.rewrap(q), self.rewrap(r)))
            }
            fn divmod(&self, other: &Self) -> PyResult<(Self, Self)> {
                self.__divmod__(other)
            }

            // --- Class methods ----------------------------------------

            /// Build from power-basis coefficients (where applicable).
            #[classmethod]
            #[pyo3(signature = (coeffs))]
            fn from_power_basis(
                _cls: &Bound<'_, pyo3::types::PyType>,
                coeffs: Vec<f64>,
            ) -> PyResult<Self> {
                Ok(Self {
                    inner: <$RustType>::from_power_basis(&coeffs).map_err(ferr_to_pyerr)?,
                    symbol: "x".to_string(),
                })
            }

            /// Least-squares fit. Returns the series of degree `deg` that
            /// best fits `(x, y)`, with coefficients stored in the
            /// domain→window MAPPED basis and `domain = [x.min, x.max]`
            /// (`numpy/polynomial/_polybase.py:946,1015` — `domain =
            /// pu.getdomain(x)`, then `xnew = pu.mapdomain(x, domain,
            /// window)` before the lstsq). `fit_with_domain` performs that
            /// mapping; the plain `fit` would have stored raw power-basis
            /// coef with the default domain.
            #[classmethod]
            fn fit(
                _cls: &Bound<'_, pyo3::types::PyType>,
                x: Vec<f64>,
                y: Vec<f64>,
                deg: usize,
            ) -> PyResult<Self> {
                Ok(Self {
                    inner: <$RustType>::fit_with_domain(&x, &y, deg).map_err(ferr_to_pyerr)?,
                    symbol: "x".to_string(),
                })
            }

            // --- Python representation --------------------------------

            /// `numpy/polynomial/_polybase.py:322-328` —
            /// `Polynomial([1., 2., 3.], domain=[-1.,  1.], window=[-1.,  1.],
            /// symbol='x')`. Class name + numpy-formatted coef/domain/window
            /// + symbol.
            fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
                let coef = numpy_array_repr_inner(py, self.inner.coeffs())?;
                let domain = numpy_array_repr_inner(py, &self.inner.domain())?;
                let window = numpy_array_repr_inner(py, &self.inner.window())?;
                Ok(format!(
                    "{}({}, domain={}, window={}, symbol='{}')",
                    stringify!($PyName),
                    coef,
                    domain,
                    window,
                    self.symbol,
                ))
            }

            fn __len__(&self) -> usize {
                self.inner.coeffs().len()
            }
        }

        impl $PyName {
            /// Wrap a derived inner polynomial, carrying `self`'s symbol.
            ///
            /// Every coefficient/arithmetic op (deriv, integ, add, divmod, …)
            /// preserves the receiver's symbol — numpy threads
            /// `self.symbol` into the returned series
            /// (`numpy/polynomial/_polybase.py:582-583` for divmod, mirrored
            /// across the dunder family). The domain/window already ride
            /// along on `inner`.
            fn rewrap(&self, inner: $RustType) -> Self {
                Self {
                    inner,
                    symbol: self.symbol.clone(),
                }
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Top-level numpy poly1d family (numpy.polyval / poly / roots / polyadd /
// polysub / polymul / polyder / polyint / polyfit / polydiv)
// ---------------------------------------------------------------------------
//
// These are the CLASSIC numpy 1-D polynomial functions on flat coefficient
// arrays (`numpy/lib/_polynomial_impl.py`), distinct from the
// `numpy.polynomial.polynomial.*` function-style API above. The crucial
// difference is coefficient ORDER:
//
//   * `numpy.polyval([1, 2, 3], x)` evaluates `x**2 + 2*x + 3` — coefficients
//     run from HIGHEST degree to lowest (`_polynomial_impl.py:736-738`
//     "p[0] * x**(N-1) + p[1] * x**(N-2) + ... + p[N-1]").
//   * ferray's power-basis `Polynomial` stores coefficients LOWEST-first
//     (`power.rs:27` "Coefficients in ascending power order: c[0], c[1], ...").
//
// Every binding here REVERSES the numpy highest-first input into ferray's
// lowest-first order before calling the ferray-polynomial math, then reverses
// the lowest-first result back to highest-first for the numpy contract.

/// Reverse a numpy highest-first coefficient vector into ferray's lowest-first
/// order (and vice-versa — reversal is its own inverse).
fn flip(mut c: Vec<f64>) -> Vec<f64> {
    c.reverse();
    c
}

/// Strip leading zeros from a highest-first coefficient array, matching
/// numpy's `trim_zeros`/`_trimseq`-style trimming after poly arithmetic.
///
/// `numpy/lib/_polynomial_impl.py:861` (`polyadd` returns
/// `trim_zeros(val, 'f')`-equivalent via `poly1d`-free arithmetic). The
/// concrete contract is verified live: `np.polyadd([1,2,3],[-1,-2]) ==
/// [1, 1, 1]` (the leading-coefficient cancellation is trimmed). We keep at
/// least one element so the zero polynomial stays `[0.0]`.
fn trim_leading_highest_first(mut c: Vec<f64>) -> Vec<f64> {
    let mut start = 0;
    while start + 1 < c.len() && c[start] == 0.0 {
        start += 1;
    }
    if start > 0 {
        c.drain(0..start);
    }
    if c.is_empty() {
        c.push(0.0);
    }
    c
}

/// `numpy.polyval(p, x)` — evaluate a highest-first polynomial `p` at `x`
/// (Horner). `x` may be a scalar or any array-like; the result follows the
/// input's shape. `numpy/lib/_polynomial_impl.py:714` `def polyval(p, x)`.
#[pyfunction]
pub fn polyval<'py>(
    py: Python<'py>,
    p: Vec<f64>,
    x: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let poly = fp::Polynomial::new(&flip(p));
    let (data, shape) = as_eval_input(py, x)?;
    let result = poly.eval_many(&data).map_err(ferr_to_pyerr)?;
    match shape {
        None => {
            let arr = ArrayD::<f64>::from_vec(IxDyn::new(&[]), vec![result[0]])
                .map_err(ferr_to_pyerr)?;
            let zerod = arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
            crate::conv::scalarize(zerod)
        }
        Some(shape) => {
            let arr =
                ArrayD::<f64>::from_vec(IxDyn::new(&shape), result).map_err(ferr_to_pyerr)?;
            Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
        }
    }
}

/// `numpy.poly(seq_of_zeros)` — highest-first coefficients of the monic
/// polynomial whose roots are `seq_of_zeros`.
///
/// `numpy/lib/_polynomial_impl.py:148-161`: starts from `[1]` and convolves
/// `[1, -zero]` for each root (highest-first), returning `1.0` for an empty
/// root sequence. ferray's `Polynomial::from_roots` builds the same product in
/// LOWEST-first order, so we reverse the result to highest-first.
#[pyfunction]
pub fn poly<'py>(py: Python<'py>, seq_of_zeros: Vec<f64>) -> PyResult<Bound<'py, PyAny>> {
    if seq_of_zeros.is_empty() {
        // `numpy/lib/_polynomial_impl.py:148-149` returns the scalar `1.0`.
        let arr =
            ArrayD::<f64>::from_vec(IxDyn::new(&[]), vec![1.0]).map_err(ferr_to_pyerr)?;
        let zerod = arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
        return crate::conv::scalarize(zerod);
    }
    let lowest_first = fp::Polynomial::from_roots(&seq_of_zeros).coeffs().to_vec();
    vec_to_pyarray1(py, flip(lowest_first))
}

/// `numpy.roots(p)` — roots of a highest-first polynomial `p` via the
/// companion-matrix eigenvalues. `numpy/lib/_polynomial_impl.py:169-262`.
///
/// numpy strips leading/trailing zeros, records the trailing-zero count as
/// roots-at-0, and tacks those zeros onto the result
/// (`_polynomial_impl.py:230-261`). We reverse the highest-first input into
/// ferray's lowest-first order, find the roots of the stripped polynomial, and
/// re-append the zero roots. The result follows ferray's
/// `complex_roots_to_pyarray` contract (real `float64` array when every
/// imaginary part is zero, else `complex128`); roots are returned sorted —
/// numpy returns them in raw eigenvalue order, so callers compare as a set.
#[pyfunction]
pub fn roots<'py>(py: Python<'py>, p: Vec<f64>) -> PyResult<Bound<'py, PyAny>> {
    // Highest-first -> lowest-first.
    let coeffs_lowest = flip(p);
    // Find non-zero entries (in lowest-first order).
    let first_nz = coeffs_lowest.iter().position(|&c| c != 0.0);
    let Some(first_nz) = first_nz else {
        // All zeros: numpy returns an empty array (`_polynomial_impl.py:234`).
        return Ok(numpy::PyArray1::<f64>::from_vec(py, Vec::new()).into_any());
    };
    let last_nz = coeffs_lowest
        .iter()
        .rposition(|&c| c != 0.0)
        .unwrap_or(first_nz);
    // Trailing zeros in HIGHEST-first p == leading zeros in lowest-first ==
    // entries below `first_nz` == number of roots at 0
    // (`_polynomial_impl.py:237-238`).
    let zero_roots = first_nz;
    // Strip leading (lowest-first) and trailing zeros -> the core polynomial.
    let stripped: Vec<f64> = coeffs_lowest[first_nz..=last_nz].to_vec();
    let mut found = if stripped.len() > 1 {
        fp::roots::find_roots_from_power_coeffs(&stripped).map_err(ferr_to_pyerr)?
    } else {
        Vec::new()
    };
    for _ in 0..zero_roots {
        found.push(num_complex::Complex::new(0.0, 0.0));
    }
    complex_roots_to_pyarray(py, found)
}

/// Shared body for `polyadd` / `polysub` / `polymul`.
fn poly_binop<'py>(
    py: Python<'py>,
    a: Vec<f64>,
    b: Vec<f64>,
    op: impl Fn(&fp::Polynomial, &fp::Polynomial) -> Result<fp::Polynomial, ferray_core::FerrayError>,
) -> PyResult<Bound<'py, PyAny>> {
    let pa = fp::Polynomial::new(&flip(a));
    let pb = fp::Polynomial::new(&flip(b));
    let out = op(&pa, &pb).map_err(ferr_to_pyerr)?;
    // numpy's polyadd/polysub/polymul do NOT trim — polyadd/polysub zero-pad
    // the shorter operand to the longer length and add/subtract
    // (`numpy/lib/_polynomial_impl.py:850-863`), and polymul is the full
    // convolution. ferray's add/sub keep max length and mul produces
    // len(a)+len(b)-1, both matching numpy exactly, so reverse without trim.
    vec_to_pyarray1(py, flip(out.coeffs().to_vec()))
}

/// `numpy.polyadd(a1, a2)` — sum of two highest-first polynomials, trimmed.
/// `numpy/lib/_polynomial_impl.py:798`.
#[pyfunction]
pub fn polyadd<'py>(py: Python<'py>, a1: Vec<f64>, a2: Vec<f64>) -> PyResult<Bound<'py, PyAny>> {
    poly_binop(py, a1, a2, |x, y| x.add(y))
}

/// `numpy.polysub(a1, a2)` — difference of two highest-first polynomials.
/// `numpy/lib/_polynomial_impl.py:867`.
#[pyfunction]
pub fn polysub<'py>(py: Python<'py>, a1: Vec<f64>, a2: Vec<f64>) -> PyResult<Bound<'py, PyAny>> {
    poly_binop(py, a1, a2, |x, y| x.sub(y))
}

/// `numpy.polymul(a1, a2)` — product of two highest-first polynomials.
/// `numpy/lib/_polynomial_impl.py:924`.
#[pyfunction]
pub fn polymul<'py>(py: Python<'py>, a1: Vec<f64>, a2: Vec<f64>) -> PyResult<Bound<'py, PyAny>> {
    poly_binop(py, a1, a2, |x, y| x.mul(y))
}

/// `numpy.polyder(p, m=1)` — `m`-th derivative of a highest-first polynomial.
/// `numpy/lib/_polynomial_impl.py:378`.
#[pyfunction]
#[pyo3(signature = (p, m = 1))]
pub fn polyder<'py>(py: Python<'py>, p: Vec<f64>, m: usize) -> PyResult<Bound<'py, PyAny>> {
    let poly = fp::Polynomial::new(&flip(p));
    let out = poly.deriv(m).map_err(ferr_to_pyerr)?;
    vec_to_pyarray1(py, flip(out.coeffs().to_vec()))
}

/// `numpy.polyint(p, m=1, k=None)` — `m`-th antiderivative of a highest-first
/// polynomial with integration constants `k`. `numpy/lib/_polynomial_impl.py:270`.
///
/// numpy's `k` are "given in the order of integration: those corresponding to
/// highest-order terms come first" (`_polynomial_impl.py:296-298`), which is
/// exactly ferray's `integ` constant order, so `k` passes through unreversed.
#[pyfunction]
#[pyo3(signature = (p, m = 1, k = None))]
pub fn polyint<'py>(
    py: Python<'py>,
    p: Vec<f64>,
    m: usize,
    k: Option<Vec<f64>>,
) -> PyResult<Bound<'py, PyAny>> {
    let poly = fp::Polynomial::new(&flip(p));
    let k = k.unwrap_or_default();
    let out = poly.integ(m, &k).map_err(ferr_to_pyerr)?;
    vec_to_pyarray1(py, flip(out.coeffs().to_vec()))
}

/// `numpy.polyfit(x, y, deg)` — least-squares fit returning highest-first
/// coefficients. `numpy/lib/_polynomial_impl.py:461`.
///
/// numpy builds a (highest-first) Vandermonde and solves a scaled lstsq
/// (`_polynomial_impl.py:658,674-678`). ferray's `Poly::fit` performs the same
/// raw power-basis least-squares (lowest-first, no domain mapping); we reverse
/// the lowest-first coefficients to highest-first.
#[pyfunction]
pub fn polyfit<'py>(
    py: Python<'py>,
    x: Vec<f64>,
    y: Vec<f64>,
    deg: usize,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_polynomial::traits::Poly;
    let fitted = <fp::Polynomial as Poly>::fit(&x, &y, deg).map_err(ferr_to_pyerr)?;
    vec_to_pyarray1(py, flip(fitted.coeffs().to_vec()))
}

/// `numpy.polydiv(u, v)` — polynomial division returning `(quotient,
/// remainder)`, both highest-first. `numpy/lib/_polynomial_impl.py:992`.
#[pyfunction]
pub fn polydiv<'py>(
    py: Python<'py>,
    u: Vec<f64>,
    v: Vec<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let pu = fp::Polynomial::new(&flip(u));
    let pv = fp::Polynomial::new(&flip(v));
    let (q, r) = pu.divmod(&pv).map_err(ferr_to_pyerr)?;
    // `numpy/lib/_polynomial_impl.py:1054` — quotient length is fixed at
    // `max(m - n + 1, 1)` (no leading-zero trim). The remainder strips leading
    // highest-first zeros down to min length 1 (`_polynomial_impl.py:1060-1061`).
    let q_py = vec_to_pyarray1(py, flip(q.coeffs().to_vec()))?;
    let r_py = vec_to_pyarray1(py, trim_leading_highest_first(flip(r.coeffs().to_vec())))?;
    Ok(PyTuple::new(py, [q_py, r_py])?.into_any())
}

poly_class!(
    Polynomial,
    ferray_polynomial::Polynomial,
    "ferray.polynomial"
);
poly_class!(Chebyshev, ferray_polynomial::Chebyshev, "ferray.polynomial");
poly_class!(Hermite, ferray_polynomial::Hermite, "ferray.polynomial");
poly_class!(HermiteE, ferray_polynomial::HermiteE, "ferray.polynomial");
poly_class!(Laguerre, ferray_polynomial::Laguerre, "ferray.polynomial");
poly_class!(Legendre, ferray_polynomial::Legendre, "ferray.polynomial");
