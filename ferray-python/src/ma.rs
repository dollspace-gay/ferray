//! Bindings for `numpy.ma` — masked arrays.
//!
//! Wraps `ferray_ma::MaskedArray<f64, IxDyn>` as a Python class.
//! Float64 is the only dtype bound in this main slice — other element
//! types (i64, f32, bool) need separate `#[pyclass]` wrappers (PyO3
//! classes can't be generic) and are tracked in a follow-up. Most
//! numpy.ma usage is on float data so f64 covers the typical case.
//!
//! The class exposes:
//!
//! - Constructors via free functions: `masked_array`, `array`,
//!   `masked_where`, `masked_invalid`, `masked_equal`, `masked_greater`,
//!   `masked_greater_equal`, `masked_less`, `masked_less_equal`,
//!   `masked_not_equal`, `masked_inside`, `masked_outside`.
//! - Properties: `data`, `mask`, `shape`, `ndim`, `size`,
//!   `fill_value`, `dtype`.
//! - Methods: `count`, `sum`, `mean`, `min`, `max`, `var`, `std`
//!   (with optional axis), `filled`, `compressed`, `__repr__`,
//!   `__array__` (NumPy protocol — returns the underlying data verbatim,
//!   so `numpy.asarray(ma)` equals `ma.data`).
//!
//! Boundary conventions mapping the ferray-ma model onto numpy.ma's Python
//! API: all-masked full reductions return the `numpy.ma.masked` singleton
//! (not ferray-ma's NaN analog); `getmask` / `.mask` of an unmasked array
//! return the `numpy.ma.nomask` singleton; `masked_where` with a
//! shape-mismatched condition raises `IndexError`; `masked_equal` sets the
//! result's `fill_value` to the compared value.
//!
//! The numpy.ma expansion (refs #818) adds, composed over the ferray-ma
//! library (`fma::*`):
//!
//! - **Masked elementwise unary ufuncs**: `sin`, `cos`, `tan`, `arctan`,
//!   `sinh`, `cosh`, `tanh`, `arcsinh`, `exp`, `floor`, `ceil`, `around`,
//!   `negative`, `absolute`, `abs`, `fabs`, `conjugate`, plus the
//!   domain-masking `sqrt`, `log`, `log2`, `log10`, `arcsin`, `arccos`,
//!   `arccosh`, `arctanh` (out-of-domain inputs are auto-masked).
//! - **Masked elementwise binary ufuncs** (MA-or-scalar 2nd arg): `add`,
//!   `subtract`, `multiply`, `divide` (zero-denominator masked),
//!   `true_divide`, `floor_divide`, `power`, `arctan2`, `hypot`, `fmod`,
//!   `remainder`, `mod`, `maximum`, `minimum`, `bitwise_and/or/xor`.
//! - **Masked reductions**: `prod`, `median`, `ptp`, `argmin`, `argmax`,
//!   `count`, `average`, `all`, `any`, `anom`.
//! - **Masked creation**: `zeros`, `ones`, `empty`, `arange`, `identity`,
//!   `asarray`, `asanyarray`, `copy` (nomask).
//! - **Masked manipulation**: `reshape`, `ravel`, `transpose`, `squeeze`,
//!   `expand_dims`, `concatenate`, `diag`, `repeat`, `clip`.
//! - **Mask helpers / predicates**: `getmaskarray`, `make_mask`,
//!   `make_mask_none`, `mask_or`, `masked_values`, `masked_object`,
//!   `fix_invalid`, `is_mask`, `isMaskedArray`/`isMA`/`isarray`,
//!   `set_fill_value`, `default_fill_value`.
//!
//! The specialized-algorithm slice (refs #835) binds the existing ferray-ma
//! algorithms that match numpy.ma exactly: `sort` / `argsort` (masked
//! values trail), `take`, `trace`, `dot` (1-D inner product), `unique`.
//!
//! Functions needing masked-algorithm support ferray-ma genuinely lacks or
//! whose ferray-ma form diverges from numpy.ma's mask semantics
//! (`vander`/`isin`/`in1d` mask handling, 2-D `dot` matmul, multi-axis
//! `argsort`, `cov`, `corrcoef`, `polyfit`, `convolve`, `correlate`,
//! `inner`, `outer`, `where`, `choose`, `diff`, `ediff1d`, `nonzero`,
//! `indices`, the `compress_*`/`mask_rowcols` family, `clump_*`,
//! `notmasked_*`, `flatnotmasked_*`, `harden_mask`/`soften_mask`) are
//! tracked as a ferray-ma library follow-up under #835.

use ferray_core::Array;
use ferray_core::array::aliases::{Array1, ArrayD};
use ferray_core::dimension::{Ix1, Ix2, IxDyn};
use ferray_ma as fma;
use ferray_ma::MaskedArray as RustMa;
use ferray_numpy_interop::{AsFerray, IntoNumPy};
use numpy::PyReadonlyArrayDyn;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::conv::{coerce_dtype, ferr_to_pyerr, ma_masked_singleton, ma_nomask};

// ---------------------------------------------------------------------------
// MaskedArray pyclass (f64-only main slice)
// ---------------------------------------------------------------------------

#[pyclass(name = "MaskedArray", module = "ferray.ma", from_py_object)]
#[derive(Clone)]
pub struct PyMaskedArray {
    inner: RustMa<f64, IxDyn>,
}

impl PyMaskedArray {
    fn from_inner(inner: RustMa<f64, IxDyn>) -> Self {
        Self { inner }
    }

    /// True iff every element is masked (or the array is empty).
    ///
    /// numpy.ma's full reductions return the `masked` singleton in exactly
    /// this case (`numpy/ma/core.py:5250` `elif newmask: result = masked`,
    /// where `newmask` is the all-true reduction of the mask). The count of
    /// unmasked elements being zero is the ferray-side analog of that
    /// all-true `newmask`.
    fn all_masked(&self) -> PyResult<bool> {
        Ok(self.inner.count().map_err(ferr_to_pyerr)? == 0)
    }

    /// True iff at least one element is masked. Mirrors `numpy.ma.getmask`
    /// returning `nomask` (vs. a real bool array) precisely when no element
    /// is masked (`numpy/ma/core.py:1468`).
    fn any_masked(&self) -> PyResult<bool> {
        Ok(fma::count_masked(&self.inner).map_err(ferr_to_pyerr)? > 0)
    }
}

#[pymethods]
impl PyMaskedArray {
    #[new]
    #[pyo3(signature = (data, mask = None))]
    fn py_new<'py>(
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        mask: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Self> {
        let data_arr = coerce_dtype(py, data, "float64")?;
        let data_view: PyReadonlyArrayDyn<f64> = data_arr.extract()?;
        let data_fa: ArrayD<f64> = data_view.as_ferray().map_err(ferr_to_pyerr)?;
        let mask_fa: ArrayD<bool> = match mask {
            None => {
                // Default mask: all-false (nothing masked).
                let n = data_fa.size();
                let shape = data_fa.shape().to_vec();
                ArrayD::<bool>::from_vec(IxDyn::new(&shape), vec![false; n])
                    .map_err(ferr_to_pyerr)?
            }
            Some(m) => {
                let m_arr = coerce_dtype(py, m, "bool")?;
                let m_view: PyReadonlyArrayDyn<bool> = m_arr.extract()?;
                m_view.as_ferray().map_err(ferr_to_pyerr)?
            }
        };
        let inner = RustMa::new(data_fa, mask_fa).map_err(ferr_to_pyerr)?;
        Ok(Self { inner })
    }

    /// Underlying data buffer as a `numpy.ndarray` (float64).
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let data: ArrayD<f64> = fma::getdata(&self.inner).map_err(ferr_to_pyerr)?;
        Ok(data.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    /// Mask as a `numpy.ndarray` of bool with the same shape as `data`,
    /// or the `numpy.ma.nomask` singleton when nothing is masked.
    ///
    /// numpy's `MaskedArray.mask` getter returns `self._mask`, which is the
    /// `nomask` constant for an array constructed without an explicit mask
    /// (`numpy/ma/core.py:1468`), not a full `array([False, ...])`. The
    /// binding mirrors that singleton (R-DEV-3).
    #[getter]
    fn mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if !self.any_masked()? {
            return ma_nomask(py);
        }
        let mask: ArrayD<bool> = fma::getmask(&self.inner).map_err(ferr_to_pyerr)?;
        Ok(mask.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    /// The replacement value used for masked positions by `filled()`.
    ///
    /// numpy exposes `fill_value` as a property (`numpy/ma/core.py:3793`);
    /// for float64 it defaults to `1e20` (`default_filler['f']`,
    /// `numpy/ma/core.py:166`). The ferray-ma library carries the same
    /// per-dtype default on the `MaskedArray`; the binding surfaces it.
    #[getter]
    fn fill_value(&self) -> f64 {
        self.inner.fill_value()
    }

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let shape: Vec<usize> = self.inner.shape().to_vec();
        Ok(pyo3::types::PyTuple::new(py, shape)?.into_any())
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    #[getter]
    fn size(&self) -> usize {
        self.inner.size()
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        // Pinned to float64 in this main slice.
        "float64"
    }

    /// Number of unmasked elements.
    fn count(&self) -> PyResult<usize> {
        self.inner.count().map_err(ferr_to_pyerr)
    }

    /// Sum of unmasked elements (full reduction). An all-masked array
    /// reduces to the `numpy.ma.masked` singleton (`numpy/ma/core.py:5250`).
    fn sum<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.all_masked()? {
            return ma_masked_singleton(py);
        }
        let v = self.inner.sum().map_err(ferr_to_pyerr)?;
        Ok(v.into_pyobject(py)?.into_any())
    }

    /// Mean of unmasked elements (full reduction). All-masked reduces to the
    /// `numpy.ma.masked` singleton (`numpy/ma/core.py:5417`).
    fn mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.all_masked()? {
            return ma_masked_singleton(py);
        }
        let v = self.inner.mean().map_err(ferr_to_pyerr)?;
        Ok(v.into_pyobject(py)?.into_any())
    }

    /// Minimum unmasked element. All-masked reduces to the `numpy.ma.masked`
    /// singleton (`numpy/ma/core.py:5942`).
    fn min<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.all_masked()? {
            return ma_masked_singleton(py);
        }
        let v = self.inner.min().map_err(ferr_to_pyerr)?;
        Ok(v.into_pyobject(py)?.into_any())
    }

    /// Maximum unmasked element. All-masked reduces to the `numpy.ma.masked`
    /// singleton (`numpy/ma/core.py:6047`).
    fn max<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.all_masked()? {
            return ma_masked_singleton(py);
        }
        let v = self.inner.max().map_err(ferr_to_pyerr)?;
        Ok(v.into_pyobject(py)?.into_any())
    }

    /// Variance of unmasked elements. All-masked (count==0) reduces to the
    /// `numpy.ma.masked` singleton (numpy's `_var` yields `masked`).
    fn var<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.all_masked()? {
            return ma_masked_singleton(py);
        }
        let v = self.inner.var().map_err(ferr_to_pyerr)?;
        Ok(v.into_pyobject(py)?.into_any())
    }

    /// Standard deviation of unmasked elements. All-masked (count==0) reduces
    /// to the `numpy.ma.masked` singleton (numpy's `_var` yields `masked`).
    fn std<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.all_masked()? {
            return ma_masked_singleton(py);
        }
        let v = self.inner.std().map_err(ferr_to_pyerr)?;
        Ok(v.into_pyobject(py)?.into_any())
    }

    /// Sum along an axis. Returns a `MaskedArray` (mask flags axes
    /// where every input was masked).
    fn sum_axis(&self, axis: usize) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.sum_axis(axis).map_err(ferr_to_pyerr)?,
        })
    }

    fn mean_axis(&self, axis: usize) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.mean_axis(axis).map_err(ferr_to_pyerr)?,
        })
    }

    fn min_axis(&self, axis: usize) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.min_axis(axis).map_err(ferr_to_pyerr)?,
        })
    }

    fn max_axis(&self, axis: usize) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.max_axis(axis).map_err(ferr_to_pyerr)?,
        })
    }

    /// Replace masked entries with `fill_value`, return as a regular
    /// `numpy.ndarray`.
    #[pyo3(signature = (fill_value = None))]
    fn filled<'py>(&self, py: Python<'py>, fill_value: Option<f64>) -> PyResult<Bound<'py, PyAny>> {
        let arr: ArrayD<f64> = match fill_value {
            Some(v) => self.inner.filled(v).map_err(ferr_to_pyerr)?,
            None => self.inner.filled_default().map_err(ferr_to_pyerr)?,
        };
        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    /// Return a 1-D `numpy.ndarray` of unmasked elements.
    fn compressed<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let arr: Array1<f64> = self.inner.compressed().map_err(ferr_to_pyerr)?;
        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    fn __repr__(&self) -> String {
        let masked = fma::count_masked(&self.inner).unwrap_or(0);
        format!(
            "MaskedArray(shape={:?}, masked={}, dtype=float64)",
            self.inner.shape(),
            masked,
        )
    }

    fn __len__(&self) -> usize {
        if self.inner.ndim() == 0 {
            0
        } else {
            self.inner.shape()[0]
        }
    }

    /// NumPy `__array__` protocol — return the underlying data verbatim,
    /// keeping the original values at masked positions (NOT the fill value).
    ///
    /// numpy's `MaskedArray.__array__` yields `self._data`, so
    /// `np.asarray(ma)` equals `ma.data` — the masked slots keep their
    /// stored values rather than being substituted. The binding mirrors that
    /// (R-DEV-3); `filled()` is the explicit fill-substituting path.
    fn __array__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.data(py)
    }
}

// ---------------------------------------------------------------------------
// Free constructors (numpy.ma.* free functions)
// ---------------------------------------------------------------------------

fn extract_data<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<ArrayD<f64>> {
    let arr = coerce_dtype(py, obj, "float64")?;
    let view: PyReadonlyArrayDyn<f64> = arr.extract()?;
    view.as_ferray().map_err(ferr_to_pyerr)
}

fn extract_bool_array<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<ArrayD<bool>> {
    let arr = coerce_dtype(py, obj, "bool")?;
    let view: PyReadonlyArrayDyn<bool> = arr.extract()?;
    view.as_ferray().map_err(ferr_to_pyerr)
}

/// `numpy.ma.array(data, mask=None)`.
#[pyfunction]
#[pyo3(signature = (data, mask = None))]
pub fn array<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    mask: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyMaskedArray> {
    PyMaskedArray::py_new(py, data, mask)
}

/// `numpy.ma.masked_array(data, mask=None)` — alias for `array`.
#[pyfunction]
#[pyo3(signature = (data, mask = None))]
pub fn masked_array<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    mask: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyMaskedArray> {
    PyMaskedArray::py_new(py, data, mask)
}

/// `numpy.ma.masked_where(condition, data)`.
///
/// A `condition` whose shape doesn't match `data` raises `IndexError`, not
/// `ValueError`: numpy applies the boolean mask as a fancy index
/// (`numpy/ma/core.py:1885`), and a boolean-index/broadcast shape mismatch
/// surfaces as `IndexError` (confirmed live: `np.ma.masked_where([T,F],
/// [1.,2.,3.])` raises `IndexError`). ferray-ma reports the mismatch as a
/// `FerrayError::ShapeMismatch`, which [`ferr_to_pyerr`] would map to
/// `ValueError`; the binding remaps it to `IndexError` here (R-DEV-2).
#[pyfunction]
pub fn masked_where<'py>(
    py: Python<'py>,
    condition: &Bound<'py, PyAny>,
    data: &Bound<'py, PyAny>,
) -> PyResult<PyMaskedArray> {
    let cond = extract_bool_array(py, condition)?;
    let data_fa = extract_data(py, data)?;
    let inner = fma::masked_where(&cond, &data_fa).map_err(shape_mismatch_to_index_error)?;
    Ok(PyMaskedArray::from_inner(inner))
}

/// Map a ferray-ma `ShapeMismatch` (from `masked_where`'s shape check) to a
/// `IndexError`, mirroring numpy's boolean-index failure; every other
/// `FerrayError` falls through to [`ferr_to_pyerr`].
fn shape_mismatch_to_index_error(e: ferray_core::FerrayError) -> PyErr {
    if matches!(e, ferray_core::FerrayError::ShapeMismatch { .. }) {
        return PyIndexError::new_err(e.to_string());
    }
    ferr_to_pyerr(e)
}

/// `numpy.ma.masked_invalid(data)` — mask NaN and inf values.
#[pyfunction]
pub fn masked_invalid<'py>(py: Python<'py>, data: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    let data_fa = extract_data(py, data)?;
    let inner = fma::masked_invalid(&data_fa).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(inner))
}

macro_rules! bind_masked_compare {
    ($name:ident, $ferr_path:path) => {
        #[pyfunction]
        pub fn $name<'py>(
            py: Python<'py>,
            x: &Bound<'py, PyAny>,
            value: f64,
        ) -> PyResult<PyMaskedArray> {
            let data_fa = extract_data(py, x)?;
            let inner = $ferr_path(&data_fa, value).map_err(ferr_to_pyerr)?;
            Ok(PyMaskedArray::from_inner(inner))
        }
    };
}

/// `numpy.ma.masked_equal(x, value)` — mask where equal to `value`.
///
/// numpy additionally sets the result's `fill_value` to the compared value
/// (`numpy/ma/core.py:2172` `output.fill_value = value`), so a later
/// `filled()` substitutes that value at masked slots. The binding mirrors
/// that by setting the ferray-ma `fill_value` after constructing the mask.
#[pyfunction]
pub fn masked_equal<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    value: f64,
) -> PyResult<PyMaskedArray> {
    let data_fa = extract_data(py, x)?;
    let mut inner = fma::masked_equal(&data_fa, value).map_err(ferr_to_pyerr)?;
    inner.set_fill_value(value);
    Ok(PyMaskedArray::from_inner(inner))
}

bind_masked_compare!(masked_not_equal, fma::masked_not_equal);
bind_masked_compare!(masked_greater, fma::masked_greater);
bind_masked_compare!(masked_greater_equal, fma::masked_greater_equal);
bind_masked_compare!(masked_less, fma::masked_less);
bind_masked_compare!(masked_less_equal, fma::masked_less_equal);

/// `numpy.ma.masked_inside(x, low, high)`.
#[pyfunction]
pub fn masked_inside<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    low: f64,
    high: f64,
) -> PyResult<PyMaskedArray> {
    let data_fa = extract_data(py, x)?;
    let inner = fma::masked_inside(&data_fa, low, high).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(inner))
}

/// `numpy.ma.masked_outside(x, low, high)`.
#[pyfunction]
pub fn masked_outside<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    low: f64,
    high: f64,
) -> PyResult<PyMaskedArray> {
    let data_fa = extract_data(py, x)?;
    let inner = fma::masked_outside(&data_fa, low, high).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(inner))
}

/// `numpy.ma.count_masked(a)` — number of masked elements (full reduction).
#[pyfunction]
pub fn count_masked(a: &PyMaskedArray) -> PyResult<usize> {
    fma::count_masked(&a.inner).map_err(ferr_to_pyerr)
}

/// `numpy.ma.is_masked(a)` — true iff at least one element is masked.
#[pyfunction]
pub fn is_masked(a: &PyMaskedArray) -> PyResult<bool> {
    fma::is_masked(&a.inner).map_err(ferr_to_pyerr)
}

/// `numpy.ma.getmask(a)` — return mask as a numpy bool ndarray.
#[pyfunction]
pub fn getmask<'py>(py: Python<'py>, a: &PyMaskedArray) -> PyResult<Bound<'py, PyAny>> {
    a.mask(py)
}

/// `numpy.ma.getdata(a)` — return data buffer as a numpy float64 ndarray.
#[pyfunction]
pub fn getdata<'py>(py: Python<'py>, a: &PyMaskedArray) -> PyResult<Bound<'py, PyAny>> {
    a.data(py)
}

/// `numpy.ma.filled(a, fill_value=None)` — replace masked with fill.
#[pyfunction]
#[pyo3(signature = (a, fill_value = None))]
pub fn filled<'py>(
    py: Python<'py>,
    a: &PyMaskedArray,
    fill_value: Option<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    a.filled(py, fill_value)
}

/// `numpy.ma.compressed(a)`.
#[pyfunction]
pub fn compressed<'py>(py: Python<'py>, a: &PyMaskedArray) -> PyResult<Bound<'py, PyAny>> {
    a.compressed(py)
}

/// Raises a clean error when callers pass a non-MaskedArray to a
/// function expecting one. Used internally — kept here so the diagnostic
/// message stays consistent with the rest of the module.
#[allow(dead_code)]
fn require_ma<'py>(obj: &Bound<'py, PyAny>) -> PyResult<()> {
    if obj.extract::<PyMaskedArray>().is_err() {
        return Err(PyValueError::new_err(
            "expected a ferray.ma.MaskedArray; use ferray.ma.masked_array(data, mask) to construct one",
        ));
    }
    Ok(())
}

// ===========================================================================
// numpy.ma expansion (refs #818): masked elementwise ufuncs, reductions,
// creation, manipulation, and mask helpers — composed over the existing
// ferray-ma library (`fma::*`) and ferray-core creation.
//
// Every binding here re-uses the f64-only `PyMaskedArray` wrapper. The
// numpy.ma contract was verified live against numpy 2.4.5 (R-CHAR-3 — the
// pytest oracle constructs expected values from `numpy.ma.*` directly).
// ===========================================================================

/// Coerce any Python object into a `RustMa<f64, IxDyn>`.
///
/// A `ferray.ma.MaskedArray` is unwrapped directly (preserving its mask);
/// any other array-like (list, ndarray, numpy.ma.MaskedArray, scalar) is
/// routed through `numpy.asarray(..., float64)` and wrapped with an
/// all-false (`nomask`) mask. This is the masked analog of the plain
/// `extract_data` path and lets the elementwise/binary ufuncs accept the
/// same operands `numpy.ma.add(ma, 3)` / `numpy.ma.sqrt([4., 9.])` do.
fn coerce_to_ma<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<RustMa<f64, IxDyn>> {
    if let Ok(m) = obj.extract::<PyMaskedArray>() {
        return Ok(m.inner);
    }
    let data_fa = extract_data(py, obj)?;
    let n = data_fa.size();
    let shape = data_fa.shape().to_vec();
    let mask_fa =
        ArrayD::<bool>::from_vec(IxDyn::new(&shape), vec![false; n]).map_err(ferr_to_pyerr)?;
    RustMa::new(data_fa, mask_fa).map_err(ferr_to_pyerr)
}

// ---------------------------------------------------------------------------
// Masked unary elementwise ufuncs
// ---------------------------------------------------------------------------

/// Generate a unary masked ufunc that applies `$f` (an `Fn(f64) -> f64`)
/// to the unmasked data and propagates the mask.
macro_rules! ma_unary {
    ($name:ident, $f:expr, $doc:literal) => {
        #[doc = $doc]
        #[pyfunction]
        pub fn $name<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
            let m = coerce_to_ma(py, a)?;
            let out = fma::masked_unary(&m, $f).map_err(ferr_to_pyerr)?;
            Ok(PyMaskedArray::from_inner(out))
        }
    };
}

/// Generate a unary masked ufunc whose result additionally masks any
/// position where the data is out of the function's domain, delegating to
/// the ferray-ma `*_domain` wrapper `$dom`.
macro_rules! ma_unary_domain {
    ($name:ident, $dom:path, $doc:literal) => {
        #[doc = $doc]
        #[pyfunction]
        pub fn $name<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
            let m = coerce_to_ma(py, a)?;
            let out = $dom(&m).map_err(ferr_to_pyerr)?;
            Ok(PyMaskedArray::from_inner(out))
        }
    };
}

ma_unary!(
    sin,
    f64::sin,
    "`numpy.ma.sin` — elementwise sine, mask propagated."
);
ma_unary!(
    cos,
    f64::cos,
    "`numpy.ma.cos` — elementwise cosine, mask propagated."
);
ma_unary!(
    tan,
    f64::tan,
    "`numpy.ma.tan` — elementwise tangent, mask propagated."
);
ma_unary!(
    arctan,
    f64::atan,
    "`numpy.ma.arctan` — elementwise arctangent."
);
ma_unary!(
    sinh,
    f64::sinh,
    "`numpy.ma.sinh` — elementwise hyperbolic sine."
);
ma_unary!(
    cosh,
    f64::cosh,
    "`numpy.ma.cosh` — elementwise hyperbolic cosine."
);
ma_unary!(
    tanh,
    f64::tanh,
    "`numpy.ma.tanh` — elementwise hyperbolic tangent."
);
ma_unary!(
    arcsinh,
    f64::asinh,
    "`numpy.ma.arcsinh` — inverse hyperbolic sine."
);
ma_unary!(exp, f64::exp, "`numpy.ma.exp` — elementwise exponential.");
ma_unary!(floor, f64::floor, "`numpy.ma.floor` — elementwise floor.");
ma_unary!(ceil, f64::ceil, "`numpy.ma.ceil` — elementwise ceiling.");
ma_unary!(
    negative,
    |x: f64| -x,
    "`numpy.ma.negative` — elementwise negation."
);
ma_unary!(
    absolute,
    f64::abs,
    "`numpy.ma.absolute` — elementwise absolute value."
);
ma_unary!(abs, f64::abs, "`numpy.ma.abs` — alias of `absolute`.");
ma_unary!(
    fabs,
    f64::abs,
    "`numpy.ma.fabs` — elementwise float absolute value."
);
ma_unary!(
    conjugate,
    |x: f64| x,
    "`numpy.ma.conjugate` — conjugate; on real (f64) data the identity, mask propagated."
);

ma_unary_domain!(
    sqrt,
    fma::sqrt_domain,
    "`numpy.ma.sqrt` — sqrt; negative inputs are masked."
);
ma_unary_domain!(
    log,
    fma::log_domain,
    "`numpy.ma.log` — natural log; non-positive inputs masked."
);
ma_unary_domain!(
    log2,
    fma::log2_domain,
    "`numpy.ma.log2` — base-2 log; non-positive inputs masked."
);
ma_unary_domain!(
    log10,
    fma::log10_domain,
    "`numpy.ma.log10` — base-10 log; non-positive inputs masked."
);
ma_unary_domain!(
    arcsin,
    fma::arcsin_domain,
    "`numpy.ma.arcsin` — arcsine; `|x| > 1` inputs masked."
);
ma_unary_domain!(
    arccos,
    fma::arccos_domain,
    "`numpy.ma.arccos` — arccosine; `|x| > 1` inputs masked."
);
ma_unary_domain!(
    arccosh,
    fma::arccosh_domain,
    "`numpy.ma.arccosh` — arccosh; `x < 1` inputs masked."
);
ma_unary_domain!(
    arctanh,
    fma::arctanh_domain,
    "`numpy.ma.arctanh` — arctanh; `|x| >= 1` inputs masked."
);

/// `numpy.ma.around(a, decimals=0)` — round to `decimals` places, mask
/// propagated. NumPy's masked `around` rounds the underlying data with the
/// half-to-even rule (`numpy.round`) and keeps the mask.
#[pyfunction]
#[pyo3(signature = (a, decimals = 0))]
pub fn around<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    decimals: i32,
) -> PyResult<PyMaskedArray> {
    let m = coerce_to_ma(py, a)?;
    let factor = 10f64.powi(decimals);
    // Half-to-even (banker's rounding), matching numpy.round.
    let round_even = move |x: f64| {
        let scaled = x * factor;
        let r = scaled.round();
        // `f64::round` rounds half away from zero; correct the exact-half
        // case to half-to-even to mirror numpy.
        let adjusted = if (scaled - scaled.floor() - 0.5).abs() < f64::EPSILON {
            let lower = scaled.floor();
            if (lower as i64) % 2 == 0 {
                lower
            } else {
                lower + 1.0
            }
        } else {
            r
        };
        adjusted / factor
    };
    let out = fma::masked_unary(&m, round_even).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(out))
}

// ---------------------------------------------------------------------------
// Masked binary elementwise ufuncs
// ---------------------------------------------------------------------------

/// Generate a binary masked ufunc applying `$f` (an `Fn(f64, f64) -> f64`)
/// to two operands, each coerced via [`coerce_to_ma`] so a scalar / plain
/// array second argument works, and OR-ing the two masks.
macro_rules! ma_binary {
    ($name:ident, $f:expr, $opname:literal, $doc:literal) => {
        #[doc = $doc]
        #[pyfunction]
        pub fn $name<'py>(
            py: Python<'py>,
            a: &Bound<'py, PyAny>,
            b: &Bound<'py, PyAny>,
        ) -> PyResult<PyMaskedArray> {
            let ma = coerce_to_ma(py, a)?;
            let mb = coerce_to_ma(py, b)?;
            let out = fma::masked_binary(&ma, &mb, $f, $opname).map_err(ferr_to_pyerr)?;
            Ok(PyMaskedArray::from_inner(out))
        }
    };
}

ma_binary!(
    add,
    |x: f64, y: f64| x + y,
    "add",
    "`numpy.ma.add` — elementwise sum, mask union."
);
ma_binary!(
    subtract,
    |x: f64, y: f64| x - y,
    "subtract",
    "`numpy.ma.subtract` — elementwise difference."
);
ma_binary!(
    multiply,
    |x: f64, y: f64| x * y,
    "multiply",
    "`numpy.ma.multiply` — elementwise product."
);
ma_binary!(
    arctan2,
    f64::atan2,
    "arctan2",
    "`numpy.ma.arctan2` — elementwise arctangent of y/x."
);
ma_binary!(
    hypot,
    f64::hypot,
    "hypot",
    "`numpy.ma.hypot` — elementwise hypotenuse."
);
ma_binary!(
    power,
    f64::powf,
    "power",
    "`numpy.ma.power` — elementwise power."
);
ma_binary!(
    fmod,
    |x: f64, y: f64| x % y,
    "fmod",
    "`numpy.ma.fmod` — C-style float modulo (sign of dividend)."
);
ma_binary!(
    remainder,
    |x: f64, y: f64| x - y * (x / y).floor(),
    "remainder",
    "`numpy.ma.remainder` — Python/numpy modulo (sign of divisor)."
);
ma_binary!(
    floor_divide,
    |x: f64, y: f64| (x / y).floor(),
    "floor_divide",
    "`numpy.ma.floor_divide` — elementwise floor division."
);
ma_binary!(
    maximum,
    |x: f64, y: f64| if x.is_nan() || y.is_nan() {
        f64::NAN
    } else {
        x.max(y)
    },
    "maximum",
    "`numpy.ma.maximum` — elementwise maximum (NaN-propagating)."
);
ma_binary!(
    minimum,
    |x: f64, y: f64| if x.is_nan() || y.is_nan() {
        f64::NAN
    } else {
        x.min(y)
    },
    "minimum",
    "`numpy.ma.minimum` — elementwise minimum (NaN-propagating)."
);
ma_binary!(
    bitwise_and,
    |x: f64, y: f64| ((x as i64) & (y as i64)) as f64,
    "bitwise_and",
    "`numpy.ma.bitwise_and` — elementwise bitwise AND (integer-valued data)."
);
ma_binary!(
    bitwise_or,
    |x: f64, y: f64| ((x as i64) | (y as i64)) as f64,
    "bitwise_or",
    "`numpy.ma.bitwise_or` — elementwise bitwise OR (integer-valued data)."
);
ma_binary!(
    bitwise_xor,
    |x: f64, y: f64| ((x as i64) ^ (y as i64)) as f64,
    "bitwise_xor",
    "`numpy.ma.bitwise_xor` — elementwise bitwise XOR (integer-valued data)."
);

/// `numpy.ma.divide(a, b)` — true division; positions where the denominator
/// is exactly zero are masked (numpy's `divide` domain rule). Delegates to
/// the ferray-ma `divide_domain` wrapper.
#[pyfunction]
pub fn divide<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<PyMaskedArray> {
    let ma = coerce_to_ma(py, a)?;
    let mb = coerce_to_ma(py, b)?;
    let out = fma::divide_domain(&ma, &mb).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(out))
}

/// `numpy.ma.true_divide(a, b)` — alias of [`divide`].
#[pyfunction]
pub fn true_divide<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<PyMaskedArray> {
    divide(py, a, b)
}

/// `numpy.ma.mod(a, b)` — alias of [`remainder`] (numpy/ma/core.py exposes
/// `mod = remainder`).
#[pyfunction]
pub fn mod_<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<PyMaskedArray> {
    remainder(py, a, b)
}

// ---------------------------------------------------------------------------
// Masked reductions (free functions accepting a MaskedArray)
// ---------------------------------------------------------------------------

/// `numpy.ma.prod(a)` — product of unmasked elements (full reduction).
/// All-masked reduces to the `numpy.ma.masked` singleton.
#[pyfunction]
pub fn prod<'py>(py: Python<'py>, a: &PyMaskedArray) -> PyResult<Bound<'py, PyAny>> {
    if a.all_masked()? {
        return ma_masked_singleton(py);
    }
    let v = a.inner.prod().map_err(ferr_to_pyerr)?;
    Ok(v.into_pyobject(py)?.into_any())
}

/// `numpy.ma.median(a)` — median of unmasked elements. All-masked reduces
/// to the `numpy.ma.masked` singleton.
#[pyfunction]
pub fn median<'py>(py: Python<'py>, a: &PyMaskedArray) -> PyResult<Bound<'py, PyAny>> {
    if a.all_masked()? {
        return ma_masked_singleton(py);
    }
    let v = a.inner.median().map_err(ferr_to_pyerr)?;
    Ok(v.into_pyobject(py)?.into_any())
}

/// `numpy.ma.ptp(a)` — peak-to-peak (max − min) of unmasked elements.
/// All-masked reduces to the `numpy.ma.masked` singleton.
#[pyfunction]
pub fn ptp<'py>(py: Python<'py>, a: &PyMaskedArray) -> PyResult<Bound<'py, PyAny>> {
    if a.all_masked()? {
        return ma_masked_singleton(py);
    }
    let v = a.inner.ptp().map_err(ferr_to_pyerr)?;
    Ok(v.into_pyobject(py)?.into_any())
}

/// `numpy.ma.argmin(a)` — index of the minimum unmasked element (flat).
#[pyfunction]
pub fn argmin<'py>(py: Python<'py>, a: &PyMaskedArray) -> PyResult<Bound<'py, PyAny>> {
    let v = a.inner.argmin().map_err(ferr_to_pyerr)?;
    Ok(v.into_pyobject(py)?.into_any())
}

/// `numpy.ma.argmax(a)` — index of the maximum unmasked element (flat).
#[pyfunction]
pub fn argmax<'py>(py: Python<'py>, a: &PyMaskedArray) -> PyResult<Bound<'py, PyAny>> {
    let v = a.inner.argmax().map_err(ferr_to_pyerr)?;
    Ok(v.into_pyobject(py)?.into_any())
}

/// `numpy.ma.count(a)` — number of unmasked elements.
#[pyfunction]
pub fn count(a: &PyMaskedArray) -> PyResult<usize> {
    a.inner.count().map_err(ferr_to_pyerr)
}

/// `numpy.ma.average(a, weights=None)` — weighted average over unmasked
/// elements. All-masked reduces to the `numpy.ma.masked` singleton.
#[pyfunction]
#[pyo3(signature = (a, weights = None))]
pub fn average<'py>(
    py: Python<'py>,
    a: &PyMaskedArray,
    weights: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    if a.all_masked()? {
        return ma_masked_singleton(py);
    }
    let v = match weights {
        None => a.inner.average(None).map_err(ferr_to_pyerr)?,
        Some(w) => {
            let w_fa = extract_data(py, w)?;
            a.inner.average(Some(&w_fa)).map_err(ferr_to_pyerr)?
        }
    };
    Ok(v.into_pyobject(py)?.into_any())
}

/// `numpy.ma.all(a)` — true iff every unmasked element is truthy
/// (non-zero). Masked elements are ignored (numpy treats `masked` as
/// `True` for `all`).
#[pyfunction]
pub fn all(a: &PyMaskedArray) -> PyResult<bool> {
    let data = fma::getdata(&a.inner).map_err(ferr_to_pyerr)?;
    let mask = fma::getmaskarray(&a.inner).map_err(ferr_to_pyerr)?;
    Ok(data.iter().zip(mask.iter()).all(|(&v, &m)| m || v != 0.0))
}

/// `numpy.ma.any(a)` — true iff at least one unmasked element is truthy.
#[pyfunction]
pub fn any(a: &PyMaskedArray) -> PyResult<bool> {
    let data = fma::getdata(&a.inner).map_err(ferr_to_pyerr)?;
    let mask = fma::getmaskarray(&a.inner).map_err(ferr_to_pyerr)?;
    Ok(data.iter().zip(mask.iter()).any(|(&v, &m)| !m && v != 0.0))
}

/// `numpy.ma.anom(a)` — deviations from the unmasked mean (a.k.a.
/// `anomalies`). Returns a `MaskedArray` of `x − mean(x)`.
#[pyfunction]
pub fn anom(a: &PyMaskedArray) -> PyResult<PyMaskedArray> {
    let out = a.inner.anom().map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(out))
}

// ---------------------------------------------------------------------------
// Masked creation
// ---------------------------------------------------------------------------

/// Wrap a freshly-created `ArrayD<f64>` (no mask) as a `PyMaskedArray`.
fn from_unmasked(data: ArrayD<f64>) -> PyResult<PyMaskedArray> {
    let inner = RustMa::from_data(data).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(inner))
}

/// Promote a 1-D `MaskedArray<f64, Ix1>` (produced by `ravel`/`repeat`) to
/// the dynamic-dimension `RustMa<f64, IxDyn>` the `PyMaskedArray` wrapper
/// holds, preserving the data and mask.
fn ix1_ma_to_dyn(m: RustMa<f64, ferray_core::dimension::Ix1>) -> PyResult<RustMa<f64, IxDyn>> {
    let data = fma::getdata(&m).map_err(ferr_to_pyerr)?.into_dyn();
    let mask = fma::getmaskarray(&m).map_err(ferr_to_pyerr)?.into_dyn();
    RustMa::new(data, mask).map_err(ferr_to_pyerr)
}

/// `numpy.ma.zeros(shape)` — masked array of zeros with `nomask`.
#[pyfunction]
pub fn zeros<'py>(py: Python<'py>, shape: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    let arr = coerce_dtype(py, &creation_call(py, "zeros", shape)?, "float64")?;
    let view: PyReadonlyArrayDyn<f64> = arr.extract()?;
    from_unmasked(view.as_ferray().map_err(ferr_to_pyerr)?)
}

/// `numpy.ma.ones(shape)` — masked array of ones with `nomask`.
#[pyfunction]
pub fn ones<'py>(py: Python<'py>, shape: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    let arr = coerce_dtype(py, &creation_call(py, "ones", shape)?, "float64")?;
    let view: PyReadonlyArrayDyn<f64> = arr.extract()?;
    from_unmasked(view.as_ferray().map_err(ferr_to_pyerr)?)
}

/// `numpy.ma.empty(shape)` — masked array of uninitialized (here: zero)
/// data with `nomask`. numpy fills with arbitrary bytes; ferray returns a
/// deterministic zero buffer, which is a valid `empty` (R-DEV-7).
#[pyfunction]
pub fn empty<'py>(py: Python<'py>, shape: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    zeros(py, shape)
}

/// `numpy.ma.arange(stop)` / `arange(start, stop, step)` — masked range
/// with `nomask`. Built on `numpy.arange` (float64) then wrapped.
#[pyfunction]
#[pyo3(signature = (start, stop = None, step = None))]
pub fn arange<'py>(
    py: Python<'py>,
    start: &Bound<'py, PyAny>,
    stop: Option<&Bound<'py, PyAny>>,
    step: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyMaskedArray> {
    let np = py.import("numpy")?;
    let result = match (stop, step) {
        (None, _) => np.call_method1("arange", (start,))?,
        (Some(s), None) => np.call_method1("arange", (start, s))?,
        (Some(s), Some(st)) => np.call_method1("arange", (start, s, st))?,
    };
    let arr = coerce_dtype(py, &result, "float64")?;
    let view: PyReadonlyArrayDyn<f64> = arr.extract()?;
    from_unmasked(view.as_ferray().map_err(ferr_to_pyerr)?)
}

/// `numpy.ma.identity(n)` — masked identity matrix with `nomask`.
#[pyfunction]
pub fn identity<'py>(py: Python<'py>, n: usize) -> PyResult<PyMaskedArray> {
    let np = py.import("numpy")?;
    let result = np.call_method1("identity", (n,))?;
    let arr = coerce_dtype(py, &result, "float64")?;
    let view: PyReadonlyArrayDyn<f64> = arr.extract()?;
    from_unmasked(view.as_ferray().map_err(ferr_to_pyerr)?)
}

/// `numpy.ma.asarray(a)` / `asanyarray(a)` — interpret `a` as a masked
/// array. An existing `MaskedArray` is returned (clone); any other
/// array-like is wrapped with `nomask`.
#[pyfunction]
pub fn asarray<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    Ok(PyMaskedArray::from_inner(coerce_to_ma(py, a)?))
}

/// `numpy.ma.asanyarray(a)` — alias of [`asarray`] for the f64 binding.
#[pyfunction]
pub fn asanyarray<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    asarray(py, a)
}

/// `numpy.ma.copy(a)` — a copy of the masked array (data + mask).
#[pyfunction]
pub fn copy<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    Ok(PyMaskedArray::from_inner(coerce_to_ma(py, a)?))
}

/// Helper: call `numpy.<name>(shape)` accepting numpy's int-or-tuple shape.
fn creation_call<'py>(
    py: Python<'py>,
    name: &str,
    shape: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    py.import("numpy")?.call_method1(name, (shape,))
}

// ---------------------------------------------------------------------------
// Masked manipulation (mask propagated)
// ---------------------------------------------------------------------------

/// `numpy.ma.reshape(a, newshape)` — reshape data + mask.
#[pyfunction]
pub fn reshape<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    newshape: &Bound<'py, PyAny>,
) -> PyResult<PyMaskedArray> {
    let m = coerce_to_ma(py, a)?;
    let shape = crate::conv::extract_shape(newshape)?;
    let out = m.reshape(&shape).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(out))
}

/// `numpy.ma.ravel(a)` — flatten data + mask to 1-D.
#[pyfunction]
pub fn ravel<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    let m = coerce_to_ma(py, a)?;
    let out = m.ravel().map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(ix1_ma_to_dyn(out)?))
}

/// `numpy.ma.transpose(a)` — transpose data + mask (reverse axes).
#[pyfunction]
pub fn transpose<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    let m = coerce_to_ma(py, a)?;
    let out = m.transpose(None).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(out))
}

/// `numpy.ma.squeeze(a)` — drop all length-1 axes from data + mask.
#[pyfunction]
pub fn squeeze<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    let m = coerce_to_ma(py, a)?;
    let out = m.squeeze(None).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(out))
}

/// `numpy.ma.expand_dims(a, axis)` — insert a length-1 axis at `axis`.
#[pyfunction]
pub fn expand_dims<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: usize,
) -> PyResult<PyMaskedArray> {
    let m = coerce_to_ma(py, a)?;
    let out = m.expand_dims(axis).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(out))
}

/// `numpy.ma.concatenate(arrays, axis=0)` — concatenate masked arrays,
/// concatenating their masks alongside the data.
#[pyfunction]
#[pyo3(signature = (arrays, axis = 0))]
pub fn concatenate<'py>(
    py: Python<'py>,
    arrays: &Bound<'py, PyAny>,
    axis: usize,
) -> PyResult<PyMaskedArray> {
    let seq: Vec<Bound<'py, PyAny>> = arrays.extract()?;
    if seq.is_empty() {
        return Err(PyValueError::new_err(
            "ma.concatenate: need at least one array to concatenate",
        ));
    }
    let mut acc = coerce_to_ma(py, &seq[0])?;
    for item in &seq[1..] {
        let next = coerce_to_ma(py, item)?;
        acc = fma::ma_concatenate(&acc, &next, axis).map_err(ferr_to_pyerr)?;
    }
    Ok(PyMaskedArray::from_inner(acc))
}

/// `numpy.ma.diag(a, k=0)` — masked diagonal / 2-D diagonal embedding,
/// mirroring `numpy.diag` for the data and the mask separately.
#[pyfunction]
#[pyo3(signature = (a, k = 0))]
pub fn diag<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, k: i64) -> PyResult<PyMaskedArray> {
    let m = coerce_to_ma(py, a)?;
    let np = py.import("numpy")?;
    let data: ArrayD<f64> = fma::getdata(&m).map_err(ferr_to_pyerr)?;
    let mask: ArrayD<bool> = fma::getmaskarray(&m).map_err(ferr_to_pyerr)?;
    let data_py = data.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    let mask_py = mask.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    let data_diag = coerce_dtype(py, &np.call_method1("diag", (data_py, k))?, "float64")?;
    let mask_diag = coerce_dtype(py, &np.call_method1("diag", (mask_py, k))?, "bool")?;
    let data_fa: ArrayD<f64> = data_diag
        .extract::<PyReadonlyArrayDyn<f64>>()?
        .as_ferray()
        .map_err(ferr_to_pyerr)?;
    let mask_fa: ArrayD<bool> = mask_diag
        .extract::<PyReadonlyArrayDyn<bool>>()?
        .as_ferray()
        .map_err(ferr_to_pyerr)?;
    let inner = RustMa::new(data_fa, mask_fa).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(inner))
}

/// `numpy.ma.repeat(a, repeats)` — repeat each element `repeats` times,
/// repeating mask alongside the data (1-D result).
#[pyfunction]
pub fn repeat<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    repeats: usize,
) -> PyResult<PyMaskedArray> {
    let m = coerce_to_ma(py, a)?;
    let out = m.repeat(repeats).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(ix1_ma_to_dyn(out)?))
}

/// `numpy.ma.clip(a, a_min, a_max)` — clip unmasked data into
/// `[a_min, a_max]`, mask propagated.
#[pyfunction]
pub fn clip<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    a_min: f64,
    a_max: f64,
) -> PyResult<PyMaskedArray> {
    let m = coerce_to_ma(py, a)?;
    let out = m.clip(a_min, a_max).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(out))
}

// ---------------------------------------------------------------------------
// Mask helpers + predicates
// ---------------------------------------------------------------------------

/// `numpy.ma.getmaskarray(a)` — the mask as a full bool ndarray (never
/// `nomask`): an unmasked array yields an all-`False` array.
#[pyfunction]
pub fn getmaskarray<'py>(py: Python<'py>, a: &PyMaskedArray) -> PyResult<Bound<'py, PyAny>> {
    let mask: ArrayD<bool> = fma::getmaskarray(&a.inner).map_err(ferr_to_pyerr)?;
    Ok(mask.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.ma.make_mask(m)` — normalize an array-like into a boolean mask.
/// When every element is `False`, numpy collapses the result to the
/// `nomask` singleton (`numpy/ma/core.py` `make_mask` `if not m.any():
/// return nomask`); the binding mirrors that.
#[pyfunction]
pub fn make_mask<'py>(py: Python<'py>, m: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let mask = extract_bool_array(py, m)?;
    if !mask.iter().any(|&b| b) {
        return ma_nomask(py);
    }
    Ok(mask.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.ma.make_mask_none(shape)` — an all-`False` bool mask of `shape`.
#[pyfunction]
pub fn make_mask_none<'py>(
    py: Python<'py>,
    shape: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let dims = crate::conv::extract_shape(shape)?;
    let n: usize = dims.iter().product();
    let mask =
        ArrayD::<bool>::from_vec(IxDyn::new(&dims), vec![false; n]).map_err(ferr_to_pyerr)?;
    Ok(mask.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.ma.mask_or(m1, m2)` — elementwise OR of two boolean masks.
#[pyfunction]
pub fn mask_or<'py>(
    py: Python<'py>,
    m1: &Bound<'py, PyAny>,
    m2: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let a = extract_bool_array(py, m1)?;
    let b = extract_bool_array(py, m2)?;
    let out = fma::mask_or(&a, &b).map_err(ferr_to_pyerr)?;
    Ok(out.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.ma.masked_values(x, value, rtol=1e-5, atol=1e-8)` — mask where
/// `x` is approximately `value`. Sets the result's `fill_value` to `value`.
#[pyfunction]
#[pyo3(signature = (x, value, rtol = 1e-5, atol = 1e-8))]
pub fn masked_values<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    value: f64,
    rtol: f64,
    atol: f64,
) -> PyResult<PyMaskedArray> {
    let data_fa = extract_data(py, x)?;
    let mut inner = fma::masked_values(&data_fa, value, rtol, atol).map_err(ferr_to_pyerr)?;
    inner.set_fill_value(value);
    Ok(PyMaskedArray::from_inner(inner))
}

/// `numpy.ma.masked_object(x, value)` — mask where `x` exactly equals
/// `value`. For the f64 binding this is `masked_equal` with the result
/// `fill_value` set to `value` (numpy/ma/core.py `masked_object`).
#[pyfunction]
pub fn masked_object<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    value: f64,
) -> PyResult<PyMaskedArray> {
    let data_fa = extract_data(py, x)?;
    let mut inner = fma::masked_equal(&data_fa, value).map_err(ferr_to_pyerr)?;
    inner.set_fill_value(value);
    Ok(PyMaskedArray::from_inner(inner))
}

/// `numpy.ma.fix_invalid(a)` — mask NaN/inf positions; the masked data is
/// replaced with the fill value (numpy default `1e20`). Returns a
/// `MaskedArray`.
#[pyfunction]
pub fn fix_invalid<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    let m = coerce_to_ma(py, a)?;
    let data: ArrayD<f64> = fma::getdata(&m).map_err(ferr_to_pyerr)?;
    let out = fma::fix_invalid(&data, fma::default_fill_value_f64()).map_err(ferr_to_pyerr)?;
    // Combine with any pre-existing mask (numpy ORs the invalid mask in).
    let prior = fma::getmaskarray(&m).map_err(ferr_to_pyerr)?;
    let new_mask = fma::getmaskarray(&out).map_err(ferr_to_pyerr)?;
    let combined = fma::mask_or(&prior, &new_mask).map_err(ferr_to_pyerr)?;
    let out_data: ArrayD<f64> = fma::getdata(&out).map_err(ferr_to_pyerr)?;
    let inner = RustMa::new(out_data, combined).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(inner))
}

/// `numpy.ma.is_mask(m)` — true iff `m` is a plain boolean ndarray (or the
/// `nomask` singleton). numpy returns `False` for a non-bool array or a
/// MaskedArray. The binding mirrors numpy: a numpy bool ndarray / `nomask`
/// is a mask; anything else is not.
#[pyfunction]
pub fn is_mask<'py>(py: Python<'py>, m: &Bound<'py, PyAny>) -> PyResult<bool> {
    // nomask singleton.
    if m.is(&ma_nomask(py)?) {
        return Ok(true);
    }
    // A ferray/numpy MaskedArray is NOT a mask.
    if m.extract::<PyMaskedArray>().is_ok() {
        return Ok(false);
    }
    let np = py.import("numpy")?;
    let is_ndarray = m.is_instance(&np.getattr("ndarray")?)?;
    if !is_ndarray {
        return Ok(false);
    }
    let dt: String = m.getattr("dtype")?.getattr("name")?.extract()?;
    Ok(dt == "bool")
}

/// `numpy.ma.isMaskedArray(a)` / `isMA` / `isarray` — true iff `a` is a
/// masked array.
#[pyfunction]
#[pyo3(name = "isMaskedArray")]
pub fn is_masked_array<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<bool> {
    if a.extract::<PyMaskedArray>().is_ok() {
        return Ok(true);
    }
    // A genuine numpy.ma.MaskedArray passed through also counts.
    let np_ma = py.import("numpy.ma")?;
    a.is_instance(&np_ma.getattr("MaskedArray")?)
}

/// `numpy.ma.set_fill_value(a, fill_value)` — set the masked array's fill
/// value in place and return it.
#[pyfunction]
pub fn set_fill_value(a: &mut PyMaskedArray, fill_value: f64) {
    a.inner.set_fill_value(fill_value);
}

/// `numpy.ma.default_fill_value(obj)` — the default fill value used for
/// masked positions of a float64 array (`1e20`).
#[pyfunction]
pub fn default_fill_value(_obj: &Bound<'_, PyAny>) -> f64 {
    fma::default_fill_value_f64()
}

// ===========================================================================
// numpy.ma specialized algorithms (refs #835): bindings over the existing
// ferray-ma library functions (`MaskedArray::sort`/`sort_axis`/`argsort`/
// `take`/`trace`, `ma_dot_flat`, `ma_unique`, `ma_vander`, `ma_isin`/
// `ma_in1d`). Each numpy.ma contract was verified live against numpy 2.4.5;
// the pytest oracle (tests/test_expansion_ma_specialized.py) constructs every
// expected value from `numpy.ma.*` directly (R-CHAR-3).
//
// Masked-sort convention (numpy.ma.sort / argsort): unmasked values sort
// ascending, masked values trail at the end of each lane
// (`numpy/ma/core.py` `MaskedArray.sort`: "Place the masked values at the
// end"). Default `axis=-1` (last axis), `axis=None` flattens — matching
// numpy.ma's documented default for `sort`.
// ===========================================================================

/// Build the `f64` IxDyn data + bool IxDyn mask of a masked array as flat
/// row-major `Vec`s plus its shape — the common entry point for the typed
/// (Ix1 / Ix2) ferray-ma algorithms below.
fn ma_parts(m: &RustMa<f64, IxDyn>) -> PyResult<(Vec<f64>, Vec<bool>, Vec<usize>)> {
    let data: ArrayD<f64> = fma::getdata(m).map_err(ferr_to_pyerr)?;
    let mask: ArrayD<bool> = fma::getmaskarray(m).map_err(ferr_to_pyerr)?;
    let shape = data.shape().to_vec();
    let data_v: Vec<f64> = data.iter().copied().collect();
    let mask_v: Vec<bool> = mask.iter().copied().collect();
    Ok((data_v, mask_v, shape))
}

/// Reinterpret a masked array as a 1-D `MaskedArray<f64, Ix1>` (flattening
/// in row-major order). Used by the algorithms ferray-ma exposes only for
/// the 1-D case (`vander`, the flat `dot`, `in1d`).
fn ma_as_ix1(m: &RustMa<f64, IxDyn>) -> PyResult<RustMa<f64, Ix1>> {
    let (data_v, mask_v, _shape) = ma_parts(m)?;
    let n = data_v.len();
    let data_arr = Array::<f64, Ix1>::from_vec(Ix1::new([n]), data_v).map_err(ferr_to_pyerr)?;
    let mask_arr = Array::<bool, Ix1>::from_vec(Ix1::new([n]), mask_v).map_err(ferr_to_pyerr)?;
    RustMa::new(data_arr, mask_arr).map_err(ferr_to_pyerr)
}

/// Reinterpret a masked array as a 2-D `MaskedArray<f64, Ix2>`. Errors with
/// a `ValueError` if the array is not 2-D (matching the dimensionality
/// numpy.ma.trace requires for a square 2-D trace).
fn ma_as_ix2(m: &RustMa<f64, IxDyn>) -> PyResult<RustMa<f64, Ix2>> {
    let (data_v, mask_v, shape) = ma_parts(m)?;
    if shape.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "ma operation requires a 2-D masked array, got {}-D",
            shape.len()
        )));
    }
    let dim = Ix2::new([shape[0], shape[1]]);
    let data_arr = Array::<f64, Ix2>::from_vec(dim.clone(), data_v).map_err(ferr_to_pyerr)?;
    let mask_arr = Array::<bool, Ix2>::from_vec(dim, mask_v).map_err(ferr_to_pyerr)?;
    RustMa::new(data_arr, mask_arr).map_err(ferr_to_pyerr)
}

/// `numpy.ma.sort(a, axis=-1)` — sort with masked values pushed to the end
/// of each lane. `axis=None` flattens first (numpy.ma collapses to 1-D),
/// any other `axis` sorts each 1-D slice independently via the ferray-ma
/// `sort_axis`. Default `axis=-1` mirrors numpy.ma.sort's documented
/// default (`numpy/ma/core.py` `def sort(self, axis=-1, ...)`).
#[pyfunction]
#[pyo3(signature = (a, axis = Some(-1)))]
pub fn sort<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<PyMaskedArray> {
    let m = coerce_to_ma(py, a)?;
    match axis {
        None => {
            // Flatten and sort (masked values last).
            let m1 = ma_as_ix1(&m)?;
            let out = m1.sort().map_err(ferr_to_pyerr)?;
            Ok(PyMaskedArray::from_inner(ix1_ma_to_dyn(out)?))
        }
        Some(ax) => {
            let ndim = m.ndim();
            let axis_u = crate::conv::normalize_axis(py, ax, ndim)?;
            let out = m.sort_axis(axis_u).map_err(ferr_to_pyerr)?;
            Ok(PyMaskedArray::from_inner(out))
        }
    }
}

/// `numpy.ma.argsort(a, axis=None)` — indices that would sort the flattened
/// masked array, with masked positions trailing
/// (`numpy/ma/core.py` `MaskedArray.argsort`). ferray-ma's `argsort`
/// operates on the flattened array, matching numpy's 1-D / `axis=None`
/// contract; the result is a numpy `intp`-style `uint64` index array.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn argsort<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Some(ax) = axis {
        // Only flat (axis=None) and the single-axis-of-a-1-D-array case are
        // backed by ferray-ma today; reject multi-dim axis cleanly rather
        // than silently flattening.
        let m = coerce_to_ma(py, a)?;
        if !(m.ndim() <= 1 && (ax == 0 || ax == -1)) {
            return Err(PyValueError::new_err(
                "ma.argsort with an explicit axis on a multi-dimensional array is not yet \
                 supported (ferray-ma argsort is flat); pass axis=None",
            ));
        }
    }
    let m = coerce_to_ma(py, a)?;
    let m1 = ma_as_ix1(&m)?;
    let idx: Array1<u64> = m1.argsort().map_err(ferr_to_pyerr)?;
    Ok(idx.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.ma.take(a, indices)` — gather elements at flat `indices`,
/// carrying the mask of each picked position
/// (`numpy/ma/core.py` `def take(...)`). Returns a 1-D masked array.
#[pyfunction]
pub fn take<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
) -> PyResult<PyMaskedArray> {
    let m = coerce_to_ma(py, a)?;
    let idx_arr = extract_data(py, indices)?;
    let idx: Vec<usize> = idx_arr
        .iter()
        .map(|&v| {
            if v < 0.0 || v.fract() != 0.0 {
                Err(PyValueError::new_err(
                    "ma.take: indices must be non-negative integers",
                ))
            } else {
                Ok(v as usize)
            }
        })
        .collect::<PyResult<Vec<usize>>>()?;
    let out = m.take(&idx).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(ix1_ma_to_dyn(out)?))
}

/// `numpy.ma.trace(a, offset=0)` — sum of the (offset) diagonal of a 2-D
/// masked array, masked diagonal entries contributing zero
/// (`numpy/ma/core.py` `def trace(...)`). Returns a Python float.
#[pyfunction]
#[pyo3(signature = (a, offset = 0))]
pub fn trace<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, offset: i64) -> PyResult<f64> {
    let m = coerce_to_ma(py, a)?;
    let m2 = ma_as_ix2(&m)?;
    m2.trace(offset as isize).map_err(ferr_to_pyerr)
}

/// `numpy.ma.dot(a, b)` — masked inner product. ferray-ma backs the 1-D
/// vector inner product (`ma_dot_flat`), where masked positions on either
/// operand contribute zero (`numpy/ma/core.py` `def dot(...)` masks both
/// operands before the product). The 2-D matrix-multiply form of
/// `numpy.ma.dot` is a ferray-ma library gap (tracked under #835); a 2-D
/// operand here raises a clear `ValueError`.
#[pyfunction]
pub fn dot<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<f64> {
    let ma = coerce_to_ma(py, a)?;
    let mb = coerce_to_ma(py, b)?;
    if ma.ndim() > 1 || mb.ndim() > 1 {
        return Err(PyValueError::new_err(
            "ma.dot currently supports 1-D inner products only; 2-D matrix \
             multiply for masked arrays is a ferray-ma library gap (#835)",
        ));
    }
    let m1a = ma_as_ix1(&ma)?;
    let m1b = ma_as_ix1(&mb)?;
    m1a.ma_dot_flat(&m1b).map_err(ferr_to_pyerr)
}

/// `numpy.ma.unique(a)` — sorted unique unmasked values, with a single
/// trailing masked entry iff the input contained any masked element
/// (`numpy/ma/core.py` `def unique(...)`: masked values collapse to one
/// `masked` slot at the end). ferray-ma's `ma_unique` yields the sorted
/// unique *unmasked* values; the binding appends the one masked entry to
/// match numpy's observable mask.
#[pyfunction]
pub fn unique<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    let m = coerce_to_ma(py, a)?;
    let uniq: Array1<f64> = fma::ma_unique(&m).map_err(ferr_to_pyerr)?;
    let mut data: Vec<f64> = uniq.iter().copied().collect();
    let mut mask: Vec<bool> = vec![false; data.len()];

    // Mirror numpy.ma.unique: a single masked slot trails iff any input
    // element was masked. Its data value is not observable (masked), so we
    // reuse the first masked input value as the placeholder.
    let (in_data, in_mask, _shape) = ma_parts(&m)?;
    if let Some(pos) = in_mask.iter().position(|&b| b) {
        data.push(in_data[pos]);
        mask.push(true);
    }

    let n = data.len();
    let data_arr = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[n]), data).map_err(ferr_to_pyerr)?;
    let mask_arr = Array::<bool, IxDyn>::from_vec(IxDyn::new(&[n]), mask).map_err(ferr_to_pyerr)?;
    let inner = RustMa::new(data_arr, mask_arr).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(inner))
}

// ===========================================================================
// numpy.ma specialized masked algorithms (refs #835): where/choose/diff/
// ediff1d/nonzero, plus the now-corrected vander/isin/in1d. Each binds the
// matching ferray-ma library function; every numpy.ma contract was verified
// live against numpy 2.4.5 and the pytest oracle constructs expected values
// from `numpy.ma.*` directly (R-CHAR-3).
// ===========================================================================

/// Coerce a Python object into a 1-/N-D bool `MaskedArray<bool, IxDyn>`.
///
/// A `ferray.ma.MaskedArray` is reinterpreted by thresholding its float data
/// at `!= 0.0` (numpy treats any non-zero as `True`), preserving its mask; any
/// other array-like is routed through `numpy.asarray(..., bool)` with a
/// `nomask` mask.
fn coerce_to_bool_ma<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
) -> PyResult<RustMa<bool, IxDyn>> {
    if let Ok(m) = obj.extract::<PyMaskedArray>() {
        let (data, mask, shape) = ma_parts(&m.inner)?;
        let bdata: Vec<bool> = data.iter().map(|&v| v != 0.0).collect();
        let data_arr =
            ArrayD::<bool>::from_vec(IxDyn::new(&shape), bdata).map_err(ferr_to_pyerr)?;
        let mask_arr = ArrayD::<bool>::from_vec(IxDyn::new(&shape), mask).map_err(ferr_to_pyerr)?;
        return RustMa::new(data_arr, mask_arr).map_err(ferr_to_pyerr);
    }
    let data = extract_bool_array(py, obj)?;
    let n = data.size();
    let shape = data.shape().to_vec();
    let mask =
        ArrayD::<bool>::from_vec(IxDyn::new(&shape), vec![false; n]).map_err(ferr_to_pyerr)?;
    RustMa::new(data, mask).map_err(ferr_to_pyerr)
}

/// `numpy.ma.where(condition, x, y)` — masked elementwise select
/// (`numpy/ma/core.py:7915`). Result masked where the chosen source is masked
/// OR the condition is masked; a masked condition picks the `y` branch.
/// Bound as `where_` to avoid colliding with the Rust keyword; registered
/// under the Python name `where`.
#[pyfunction]
#[pyo3(name = "where")]
pub fn where_<'py>(
    py: Python<'py>,
    condition: &Bound<'py, PyAny>,
    x: &Bound<'py, PyAny>,
    y: &Bound<'py, PyAny>,
) -> PyResult<PyMaskedArray> {
    let cond = coerce_to_bool_ma(py, condition)?;
    let mx = coerce_to_ma(py, x)?;
    let my = coerce_to_ma(py, y)?;
    let out = fma::ma_where(&cond, &mx, &my).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(out))
}

/// `numpy.ma.choose(indices, choices)` — masked choose
/// (`numpy/ma/core.py:8007`). A masked index is filled with 0; the result is
/// masked where the chosen source OR the index is masked.
#[pyfunction]
pub fn choose<'py>(
    py: Python<'py>,
    indices: &Bound<'py, PyAny>,
    choices: &Bound<'py, PyAny>,
) -> PyResult<PyMaskedArray> {
    let idx = coerce_to_ma(py, indices)?;
    let seq = choices.try_iter()?;
    let mut chs: Vec<RustMa<f64, IxDyn>> = Vec::new();
    for item in seq {
        chs.push(coerce_to_ma(py, &item?)?);
    }
    let out = fma::ma_choose(&idx, &chs).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(out))
}

/// `numpy.ma.diff(a, n=1, axis=-1)` — masked adjacent difference
/// (`numpy/ma/core.py:7774`); result masked iff either operand is masked.
#[pyfunction]
#[pyo3(signature = (a, n = 1, axis = -1))]
pub fn diff<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    n: usize,
    axis: isize,
) -> PyResult<PyMaskedArray> {
    let m = coerce_to_ma(py, a)?;
    let out = fma::ma_diff(&m, n, axis).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(out))
}

/// `numpy.ma.ediff1d(ary, to_end=None, to_begin=None)` — flattened first
/// difference (`numpy/ma/extras.py:1229`); `to_begin`/`to_end` always unmasked.
#[pyfunction]
#[pyo3(signature = (ary, to_end = None, to_begin = None))]
pub fn ediff1d<'py>(
    py: Python<'py>,
    ary: &Bound<'py, PyAny>,
    to_end: Option<&Bound<'py, PyAny>>,
    to_begin: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyMaskedArray> {
    let m = coerce_to_ma(py, ary)?;
    let begin: Option<Vec<f64>> = match to_begin {
        Some(o) => Some(extract_data(py, o)?.iter().copied().collect()),
        None => None,
    };
    let end: Option<Vec<f64>> = match to_end {
        Some(o) => Some(extract_data(py, o)?.iter().copied().collect()),
        None => None,
    };
    let out = fma::ma_ediff1d(&m, begin.as_deref(), end.as_deref()).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(ix1_ma_to_dyn(out)?))
}

/// `numpy.ma.nonzero(a)` — tuple of per-dimension index arrays of the non-zero
/// UNMASKED elements (masked treated as zero; `numpy/ma/core.py:5049`). Returns
/// a Python tuple of `int64` numpy arrays, matching numpy's layout.
#[pyfunction]
pub fn nonzero<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let m = coerce_to_ma(py, a)?;
    let coords = fma::ma_nonzero(&m).map_err(ferr_to_pyerr)?;
    let mut arrays: Vec<Bound<'py, PyAny>> = Vec::with_capacity(coords.len());
    for axis_coords in coords {
        arrays.push(
            axis_coords
                .into_pyarray(py)
                .map_err(ferr_to_pyerr)?
                .into_any(),
        );
    }
    Ok(pyo3::types::PyTuple::new(py, arrays)?.into_any())
}

/// `numpy.ma.vander(x, n=None)` — masked Vandermonde
/// (`numpy/ma/extras.py:2216`). Masked input rows become all-zeros and the
/// result carries no mask (nomask).
#[pyfunction]
#[pyo3(signature = (x, n = None))]
pub fn vander<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    n: Option<usize>,
) -> PyResult<PyMaskedArray> {
    let m = coerce_to_ma(py, x)?;
    let m1 = ma_as_ix1(&m)?;
    let out = fma::ma_vander(&m1, n).map_err(ferr_to_pyerr)?;
    let data = fma::getdata(&out).map_err(ferr_to_pyerr)?.into_dyn();
    let mask = fma::getmaskarray(&out).map_err(ferr_to_pyerr)?.into_dyn();
    let inner = RustMa::new(data, mask).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(inner))
}

/// `numpy.ma.isin(element, test_elements)` — masked membership
/// (`numpy/ma/extras.py:1434`). Carries the input mask; masked positions are
/// reported `False` (a masked value is never a member).
#[pyfunction]
pub fn isin<'py>(
    py: Python<'py>,
    element: &Bound<'py, PyAny>,
    test_elements: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = coerce_to_ma(py, element)?;
    let tests: Vec<f64> = extract_data(py, test_elements)?.iter().copied().collect();
    let out = fma::ma_isin(&m, &tests).map_err(ferr_to_pyerr)?;
    // Return a plain numpy bool ndarray of the underlying membership data,
    // matching numpy.ma.isin's observable (the boolean result shape of
    // `element`); masked positions read `False`.
    let data = fma::getdata(&out).map_err(ferr_to_pyerr)?;
    Ok(data.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.ma.in1d(ar1, ar2)` — flattened masked membership
/// (`numpy/ma/extras.py:1387`). 1-D `isin`.
#[pyfunction]
pub fn in1d<'py>(
    py: Python<'py>,
    ar1: &Bound<'py, PyAny>,
    ar2: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = coerce_to_ma(py, ar1)?;
    let m1 = ma_as_ix1(&m)?;
    let tests: Vec<f64> = extract_data(py, ar2)?.iter().copied().collect();
    let out = fma::ma_in1d(&m1, &tests).map_err(ferr_to_pyerr)?;
    let data = fma::getdata(&out).map_err(ferr_to_pyerr)?;
    Ok(data.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

// ===========================================================================
// numpy.ma set operations, row/col suppression, and masked cov/corrcoef
// (refs #835). Each binds the matching ferray-ma library function added this
// iteration; every numpy.ma contract was verified live against numpy 2.4.5 and
// the pytest oracle (tests/test_expansion_ma_lib2.py) constructs every expected
// value from `numpy.ma.*` directly (R-CHAR-3).
// ===========================================================================

/// Promote a 1-D `MaskedArray<f64, Ix1>` produced by the masked set ops into
/// the dynamic-dimension wrapper the `PyMaskedArray` class holds.
fn ix1_to_py(m: RustMa<f64, Ix1>) -> PyResult<PyMaskedArray> {
    Ok(PyMaskedArray::from_inner(ix1_ma_to_dyn(m)?))
}

/// Promote a 2-D `MaskedArray<f64, Ix2>` into the dynamic-dimension wrapper.
fn ix2_to_py(m: RustMa<f64, Ix2>) -> PyResult<PyMaskedArray> {
    let data = fma::getdata(&m).map_err(ferr_to_pyerr)?.into_dyn();
    let mask = fma::getmaskarray(&m).map_err(ferr_to_pyerr)?.into_dyn();
    let inner = RustMa::new(data, mask).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(inner))
}

/// `numpy.ma.intersect1d(ar1, ar2)` — sorted unique common values, with a
/// trailing masked slot iff both inputs were masked
/// (`numpy/ma/extras.py:1317`).
#[pyfunction]
pub fn intersect1d<'py>(
    py: Python<'py>,
    ar1: &Bound<'py, PyAny>,
    ar2: &Bound<'py, PyAny>,
) -> PyResult<PyMaskedArray> {
    let a = ma_as_ix1(&coerce_to_ma(py, ar1)?)?;
    let b = ma_as_ix1(&coerce_to_ma(py, ar2)?)?;
    ix1_to_py(fma::ma_intersect1d(&a, &b).map_err(ferr_to_pyerr)?)
}

/// `numpy.ma.union1d(ar1, ar2)` — sorted unique union, masked slot iff either
/// input was masked (`numpy/ma/extras.py:1463`).
#[pyfunction]
pub fn union1d<'py>(
    py: Python<'py>,
    ar1: &Bound<'py, PyAny>,
    ar2: &Bound<'py, PyAny>,
) -> PyResult<PyMaskedArray> {
    let a = ma_as_ix1(&coerce_to_ma(py, ar1)?)?;
    let b = ma_as_ix1(&coerce_to_ma(py, ar2)?)?;
    ix1_to_py(fma::ma_union1d(&a, &b).map_err(ferr_to_pyerr)?)
}

/// `numpy.ma.setdiff1d(ar1, ar2)` — sorted unique values of `ar1` not in `ar2`
/// (`numpy/ma/extras.py:1485`).
#[pyfunction]
pub fn setdiff1d<'py>(
    py: Python<'py>,
    ar1: &Bound<'py, PyAny>,
    ar2: &Bound<'py, PyAny>,
) -> PyResult<PyMaskedArray> {
    let a = ma_as_ix1(&coerce_to_ma(py, ar1)?)?;
    let b = ma_as_ix1(&coerce_to_ma(py, ar2)?)?;
    ix1_to_py(fma::ma_setdiff1d(&a, &b).map_err(ferr_to_pyerr)?)
}

/// `numpy.ma.setxor1d(ar1, ar2)` — symmetric difference of the unique values
/// (`numpy/ma/extras.py:1350`).
#[pyfunction]
pub fn setxor1d<'py>(
    py: Python<'py>,
    ar1: &Bound<'py, PyAny>,
    ar2: &Bound<'py, PyAny>,
) -> PyResult<PyMaskedArray> {
    let a = ma_as_ix1(&coerce_to_ma(py, ar1)?)?;
    let b = ma_as_ix1(&coerce_to_ma(py, ar2)?)?;
    ix1_to_py(fma::ma_setxor1d(&a, &b).map_err(ferr_to_pyerr)?)
}

/// `numpy.ma.compress_rowcols(x, axis)` — suppress whole rows/cols containing
/// masked values; returns a plain ndarray (`numpy/ma/extras.py:920`). 2-D only.
#[pyfunction]
#[pyo3(signature = (x, axis = None))]
pub fn compress_rowcols<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = ma_as_ix2(&coerce_to_ma(py, x)?)?;
    let ax = match axis {
        None => None,
        Some(0) => Some(0usize),
        Some(1) | Some(-1) => Some(1usize),
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "ma.compress_rowcols: axis must be None, 0, 1 or -1, got {other}"
            )));
        }
    };
    let out = fma::ma_compress_rowcols(&m, ax).map_err(ferr_to_pyerr)?;
    Ok(out.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.ma.compress_rows(a)` — suppress whole rows containing masked values
/// (`numpy/ma/extras.py:953`).
#[pyfunction]
pub fn compress_rows<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let m = ma_as_ix2(&coerce_to_ma(py, a)?)?;
    let out = fma::ma_compress_rows(&m).map_err(ferr_to_pyerr)?;
    Ok(out.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.ma.compress_cols(a)` — suppress whole columns containing masked
/// values (`numpy/ma/extras.py:991`).
#[pyfunction]
pub fn compress_cols<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let m = ma_as_ix2(&coerce_to_ma(py, a)?)?;
    let out = fma::ma_compress_cols(&m).map_err(ferr_to_pyerr)?;
    Ok(out.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.ma.mask_rowcols(a, axis)` — mask whole rows/cols containing a masked
/// value (`numpy/ma/extras.py:830`). 2-D only; returns a masked array.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn mask_rowcols<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<PyMaskedArray> {
    let m = ma_as_ix2(&coerce_to_ma(py, a)?)?;
    let ax = match axis {
        None => None,
        Some(0) => Some(0usize),
        Some(1) | Some(-1) => Some(1usize),
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "ma.mask_rowcols: axis must be None, 0, 1 or -1, got {other}"
            )));
        }
    };
    let out = fma::ma_mask_rowcols(&m, ax).map_err(ferr_to_pyerr)?;
    ix2_to_py(out)
}

/// `numpy.ma.cov(x, y=None, rowvar=True, bias=False, ddof=None)` — masked
/// covariance over the unmasked observation pairs (`numpy/ma/extras.py:1547`).
/// The `y` argument (a second variable set) is a ferray-ma library gap; passing
/// it raises a clear `ValueError`.
#[pyfunction]
#[pyo3(signature = (x, y = None, rowvar = true, bias = false, allow_masked = true, ddof = None))]
#[allow(
    clippy::fn_params_excessive_bools,
    reason = "mirrors numpy.ma.cov's keyword surface (rowvar/bias/allow_masked)"
)]
pub fn cov<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    y: Option<&Bound<'py, PyAny>>,
    rowvar: bool,
    bias: bool,
    allow_masked: bool,
    ddof: Option<usize>,
) -> PyResult<PyMaskedArray> {
    if y.is_some() {
        return Err(PyValueError::new_err(
            "ma.cov with a second variable set `y` is a ferray-ma library gap (#835); \
             pass a single 2-D masked array",
        ));
    }
    let _ = allow_masked; // ferray-ma always processes the masked pairs.
    let m = coerce_to_ma(py, x)?;
    let out = fma::ma_cov(&m, rowvar, bias, ddof).map_err(ferr_to_pyerr)?;
    ix2_to_py(out)
}

/// `numpy.ma.corrcoef(x, y=None, rowvar=True)` — masked Pearson correlation
/// (`numpy/ma/extras.py:1672`): the masked covariance normalized by the outer
/// product of the per-variable standard deviations. The `y` argument is a
/// ferray-ma library gap; passing it raises a clear `ValueError`.
#[pyfunction]
#[pyo3(signature = (x, y = None, rowvar = true, allow_masked = true))]
pub fn corrcoef<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    y: Option<&Bound<'py, PyAny>>,
    rowvar: bool,
    allow_masked: bool,
) -> PyResult<PyMaskedArray> {
    if y.is_some() {
        return Err(PyValueError::new_err(
            "ma.corrcoef with a second variable set `y` is a ferray-ma library gap (#835); \
             pass a single 2-D masked array",
        ));
    }
    let _ = allow_masked;
    let m = coerce_to_ma(py, x)?;
    let out = fma::ma_corrcoef(&m, rowvar).map_err(ferr_to_pyerr)?;
    ix2_to_py(out)
}
