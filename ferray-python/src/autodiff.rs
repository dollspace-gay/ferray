//! Bindings for `ferray.autodiff` — forward-mode automatic
//! differentiation via dual numbers.
//!
//! This is a ferray-specific surface (NumPy doesn't have first-class
//! autodiff), but it's exposed because dual numbers are extremely
//! useful for any Python user doing optimisation, root-finding, or
//! sensitivity analysis on top of ferray's array operations.
//!
//! Wrapped types and functions:
//!
//! - `DualNumber` pyclass over `f64` — pyo3 #[pyclass] cannot be
//!   generic, so we bind the f64 specialisation. f32/Complex variants
//!   would need separate classes (deferred).
//! - Elementary functions (`sin`, `cos`, `exp`, `ln`, `sqrt`, …)
//!   that take `DualNumber` and return `DualNumber`, exposed at the
//!   submodule level so users can write
//!   `from ferray.autodiff import sin, exp; sin(x) + exp(x)`.
//!
//! High-level helpers `derivative(f, x)` and `gradient(f, point)`
//! that take Python callables are implemented in pure Python on top
//! of the bound `DualNumber` class — see the autodiff Python module.

use ferray_autodiff::DualNumber as RustDual;
use ferray_autodiff::functions as fa;
use pyo3::prelude::*;
use pyo3::types::PyAny;

// ---------------------------------------------------------------------------
// DualNumber pyclass (f64-only)
// ---------------------------------------------------------------------------

#[pyclass(name = "DualNumber", module = "ferray.autodiff", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyDual {
    pub(crate) inner: RustDual<f64>,
}

impl PyDual {
    pub(crate) fn from_inner(inner: RustDual<f64>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyDual {
    #[new]
    #[pyo3(signature = (real, dual = 0.0))]
    fn py_new(real: f64, dual: f64) -> Self {
        Self {
            inner: RustDual::new(real, dual),
        }
    }

    /// Construct a "variable" dual number — real part `x`, dual part 1.
    /// Use this for the input you want to differentiate with respect to.
    #[staticmethod]
    fn variable(real: f64) -> Self {
        Self {
            inner: RustDual::variable(real),
        }
    }

    /// Construct a "constant" dual number — real part `x`, dual part 0.
    /// Use this for parameters held fixed during differentiation.
    #[staticmethod]
    fn constant(real: f64) -> Self {
        Self {
            inner: RustDual::constant(real),
        }
    }

    /// Real (primal) component.
    #[getter]
    fn real(&self) -> f64 {
        self.inner.real
    }

    /// Dual (derivative) component.
    #[getter]
    fn dual(&self) -> f64 {
        self.inner.dual
    }

    fn __repr__(&self) -> String {
        format!(
            "DualNumber(real={}, dual={})",
            self.inner.real, self.inner.dual
        )
    }

    // --- arithmetic dunders ----------------------------------------

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = coerce_dual(other)?;
        Ok(Self {
            inner: self.inner + rhs.inner,
        })
    }
    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__add__(other)
    }
    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = coerce_dual(other)?;
        Ok(Self {
            inner: self.inner - rhs.inner,
        })
    }
    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let lhs = coerce_dual(other)?;
        Ok(Self {
            inner: lhs.inner - self.inner,
        })
    }
    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = coerce_dual(other)?;
        Ok(Self {
            inner: self.inner * rhs.inner,
        })
    }
    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__mul__(other)
    }
    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = coerce_dual(other)?;
        Ok(Self {
            inner: self.inner / rhs.inner,
        })
    }
    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let lhs = coerce_dual(other)?;
        Ok(Self {
            inner: lhs.inner / self.inner,
        })
    }
    fn __neg__(&self) -> Self {
        Self { inner: -self.inner }
    }
    fn __pow__(&self, n: i32, _modulo: Option<i32>) -> Self {
        // Integer power via repeated multiplication preserves the
        // dual-arithmetic chain rule for free.
        let mut result = RustDual::constant(1.0);
        let base = self.inner;
        let abs_n = n.unsigned_abs() as usize;
        for _ in 0..abs_n {
            result *= base;
        }
        if n < 0 {
            result = RustDual::constant(1.0) / result;
        }
        Self { inner: result }
    }
}

/// Coerce a Python value (`int`, `float`, or `DualNumber`) into a
/// `PyDual`. Numbers become constants (real=value, dual=0).
fn coerce_dual(obj: &Bound<'_, PyAny>) -> PyResult<PyDual> {
    if let Ok(d) = obj.extract::<PyDual>() {
        return Ok(d);
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(PyDual {
            inner: RustDual::constant(f),
        });
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "expected DualNumber, int, or float",
    ))
}

// ---------------------------------------------------------------------------
// Elementary functions over DualNumber
// ---------------------------------------------------------------------------

macro_rules! bind_dual_fn {
    ($name:ident, $ferr_path:path) => {
        #[pyfunction]
        pub fn $name(x: &Bound<'_, PyAny>) -> PyResult<PyDual> {
            let d = coerce_dual(x)?;
            Ok(PyDual::from_inner($ferr_path(d.inner)))
        }
    };
}

bind_dual_fn!(sin, fa::sin);
bind_dual_fn!(cos, fa::cos);
bind_dual_fn!(tan, fa::tan);
bind_dual_fn!(asin, fa::asin);
bind_dual_fn!(acos, fa::acos);
bind_dual_fn!(atan, fa::atan);
bind_dual_fn!(sinh, fa::sinh);
bind_dual_fn!(cosh, fa::cosh);
bind_dual_fn!(tanh, fa::tanh);
bind_dual_fn!(exp, fa::exp);
bind_dual_fn!(ln, fa::ln);
bind_dual_fn!(log2, fa::log2);
bind_dual_fn!(log10, fa::log10);
bind_dual_fn!(sqrt, fa::sqrt);
bind_dual_fn!(abs, fa::abs);

/// `atan2(y, x)` — two-argument arc tangent.
#[pyfunction]
pub fn atan2(y: &Bound<'_, PyAny>, x: &Bound<'_, PyAny>) -> PyResult<PyDual> {
    let dy = coerce_dual(y)?;
    let dx = coerce_dual(x)?;
    Ok(PyDual::from_inner(fa::atan2(dy.inner, dx.inner)))
}
