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
//!   `__array__` (NumPy protocol — converts to a regular ndarray with
//!   masked entries replaced by the fill value).
//!
//! Other ferray-ma surface (arithmetic, sorting, manipulation,
//! linalg) is deferred to a sub-follow-up.

use ferray_core::array::aliases::{Array1, ArrayD};
use ferray_core::dimension::IxDyn;
use ferray_ma as fma;
use ferray_ma::MaskedArray as RustMa;
use ferray_numpy_interop::{AsFerray, IntoNumPy};
use numpy::PyReadonlyArrayDyn;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::conv::{coerce_dtype, ferr_to_pyerr};

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

    /// Mask as a `numpy.ndarray` of bool with the same shape as `data`.
    #[getter]
    fn mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mask: ArrayD<bool> = fma::getmask(&self.inner).map_err(ferr_to_pyerr)?;
        Ok(mask.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
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

    /// Sum of unmasked elements (full reduction).
    fn sum<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let v = self.inner.sum().map_err(ferr_to_pyerr)?;
        Ok(v.into_pyobject(py)?.into_any())
    }

    /// Mean of unmasked elements (full reduction).
    fn mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let v = self.inner.mean().map_err(ferr_to_pyerr)?;
        Ok(v.into_pyobject(py)?.into_any())
    }

    /// Minimum unmasked element.
    fn min<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let v = self.inner.min().map_err(ferr_to_pyerr)?;
        Ok(v.into_pyobject(py)?.into_any())
    }

    /// Maximum unmasked element.
    fn max<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let v = self.inner.max().map_err(ferr_to_pyerr)?;
        Ok(v.into_pyobject(py)?.into_any())
    }

    /// Variance of unmasked elements.
    fn var<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let v = self.inner.var().map_err(ferr_to_pyerr)?;
        Ok(v.into_pyobject(py)?.into_any())
    }

    /// Standard deviation of unmasked elements.
    fn std<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
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
    fn filled<'py>(
        &self,
        py: Python<'py>,
        fill_value: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
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

    /// NumPy `__array__` protocol — return the data with masked entries
    /// replaced by the default fill value. Lets `numpy.asarray(ma)`
    /// work on a ferray MaskedArray.
    fn __array__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.filled(py, None)
    }
}

// ---------------------------------------------------------------------------
// Free constructors (numpy.ma.* free functions)
// ---------------------------------------------------------------------------

fn extract_data<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
) -> PyResult<ArrayD<f64>> {
    let arr = coerce_dtype(py, obj, "float64")?;
    let view: PyReadonlyArrayDyn<f64> = arr.extract()?;
    view.as_ferray().map_err(ferr_to_pyerr)
}

fn extract_bool_array<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
) -> PyResult<ArrayD<bool>> {
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
#[pyfunction]
pub fn masked_where<'py>(
    py: Python<'py>,
    condition: &Bound<'py, PyAny>,
    data: &Bound<'py, PyAny>,
) -> PyResult<PyMaskedArray> {
    let cond = extract_bool_array(py, condition)?;
    let data_fa = extract_data(py, data)?;
    let inner = fma::masked_where(&cond, &data_fa).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(inner))
}

/// `numpy.ma.masked_invalid(data)` — mask NaN and inf values.
#[pyfunction]
pub fn masked_invalid<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
) -> PyResult<PyMaskedArray> {
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

bind_masked_compare!(masked_equal,         fma::masked_equal);
bind_masked_compare!(masked_not_equal,     fma::masked_not_equal);
bind_masked_compare!(masked_greater,       fma::masked_greater);
bind_masked_compare!(masked_greater_equal, fma::masked_greater_equal);
bind_masked_compare!(masked_less,          fma::masked_less);
bind_masked_compare!(masked_less_equal,    fma::masked_less_equal);

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
