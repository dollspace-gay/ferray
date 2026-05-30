//! # ferray-python — PyO3 extension module for ferray
//!
//! This crate is the Python-facing front door for ferray. The compiled
//! cdylib is loaded by CPython as `ferray._ferray`; the pure-Python
//! shim in `python/ferray/__init__.py` re-exports the symbols at the
//! top-level `ferray.*` namespace so users can write `import ferray as
//! np` as a true drop-in replacement for `numpy`.
//!
//! The target is **100% NumPy API parity**. The ferray Rust workspace
//! already mirrors the NumPy namespace function-for-function; the job
//! here is to wire each one through to CPython. Coverage is filled in
//! by phase (see the parent crosslink epic and the README), and every
//! bound function is verified against NumPy in `tests/`.
//!
//! ## Module layout
//!
//! - [`conv`] — shared helpers and the [`match_dtype_all`] /
//!   [`match_dtype_float`] dispatch macros.
//! - [`creation`] — `numpy.zeros`, `ones`, `arange`, `linspace`, …
//! - [`manipulation`] — `numpy.transpose`, `reshape`, `concatenate`, …
//! - [`indexing`] — `numpy.indices`, `nonzero`, `argwhere`, …
//!
//! All bindings live as free `#[pyfunction]` items inside those
//! modules and get registered on the top-level extension module
//! (`_ferray`) by [`_ferray`] at import time.

mod aliases;
mod autodiff;
mod char;
mod complex;
mod conv;
mod creation;
mod datetime;
mod emath;
mod fft;
mod indexing;
mod io;
mod linalg;
mod ma;
mod manipulation;
mod polynomial;
mod random;
mod stats;
mod stride_tricks;
mod strings;
mod ufunc;
mod window;

use pyo3::prelude::*;
use pyo3::types::PyModule;

// ---------------------------------------------------------------------------
// Submodule: ferray.dtype
// ---------------------------------------------------------------------------

/// Register the `dtype` submodule. Currently exposes the dtype
/// names as string constants so callers can write
/// `dtype=ferray.dtype.float64` for forward-compatibility — once we
/// have first-class dtype objects (phase 2+) the values will change to
/// those objects without breaking callers.
fn register_dtype_module<'py>(py: Python<'py>, parent: &Bound<'py, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "dtype")?;
    for name in [
        "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
        "float32", "float64",
    ] {
        m.add(name, name)?;
    }
    parent.add_submodule(&m)?;
    let sys_modules = py.import("sys")?.getattr("modules")?;
    sys_modules.set_item("ferray._ferray.dtype", &m)?;
    Ok(())
}

/// Register `ferray.linalg` with all phase-3 linalg bindings.
fn register_linalg_module<'py>(py: Python<'py>, parent: &Bound<'py, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "linalg")?;
    m.add_function(wrap_pyfunction!(linalg::norm, &m)?)?;
    // NumPy 2.0 array-API linalg norms / singular values / cross.
    m.add_function(wrap_pyfunction!(linalg::matrix_norm, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::vector_norm, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::svdvals, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::cross, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::det, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::slogdet, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::matrix_rank, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::trace, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::cholesky, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::qr, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::lu, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::svd, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::eigh, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::eigvalsh, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::eigvals, &m)?)?;
    // numpy.linalg.LinAlgError (subclass of ValueError) — expose the public
    // exception type so `except numpy.linalg.LinAlgError` works against
    // ferray (_linalg.py:115 `class LinAlgError(ValueError)`). It is the
    // same object numpy raises, imported from numpy.linalg.
    m.add(
        "LinAlgError",
        py.import("numpy.linalg")?.getattr("LinAlgError")?,
    )?;
    m.add_function(wrap_pyfunction!(linalg::solve, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::inv, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::pinv, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::matrix_power, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::matmul, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::dot, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::vdot, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::inner, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::outer, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::kron, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::tensordot, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::einsum, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::multi_dot, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::norm_axis, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::cond, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::matvec, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::vecmat, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::vecdot, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::lstsq, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::tensorinv, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::tensorsolve, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::matrix_transpose, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::diagonal, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::matmul_complex, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::inv_complex, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::det_complex, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::solve_complex, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::eig, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::cholesky_batched, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::qr_batched, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::svd_batched, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::eigh_batched, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::eigvalsh_batched, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::solve_batched, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::inv_batched, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::pinv_batched, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::det_batched, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::slogdet_batched, &m)?)?;
    m.add_function(wrap_pyfunction!(linalg::matrix_rank_batched, &m)?)?;
    parent.add_submodule(&m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("ferray._ferray.linalg", &m)?;
    Ok(())
}

/// Register `ferray.fft` with all phase-3 FFT bindings.
fn register_fft_module<'py>(py: Python<'py>, parent: &Bound<'py, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "fft")?;
    m.add_function(wrap_pyfunction!(fft::fft, &m)?)?;
    m.add_function(wrap_pyfunction!(fft::ifft, &m)?)?;
    m.add_function(wrap_pyfunction!(fft::fft2, &m)?)?;
    m.add_function(wrap_pyfunction!(fft::ifft2, &m)?)?;
    m.add_function(wrap_pyfunction!(fft::fftn, &m)?)?;
    m.add_function(wrap_pyfunction!(fft::ifftn, &m)?)?;
    m.add_function(wrap_pyfunction!(fft::rfft, &m)?)?;
    m.add_function(wrap_pyfunction!(fft::irfft, &m)?)?;
    m.add_function(wrap_pyfunction!(fft::rfft2, &m)?)?;
    m.add_function(wrap_pyfunction!(fft::irfft2, &m)?)?;
    m.add_function(wrap_pyfunction!(fft::rfftn, &m)?)?;
    m.add_function(wrap_pyfunction!(fft::irfftn, &m)?)?;
    m.add_function(wrap_pyfunction!(fft::hfft, &m)?)?;
    m.add_function(wrap_pyfunction!(fft::ihfft, &m)?)?;
    m.add_function(wrap_pyfunction!(fft::fftshift, &m)?)?;
    m.add_function(wrap_pyfunction!(fft::ifftshift, &m)?)?;
    m.add_function(wrap_pyfunction!(fft::fftfreq, &m)?)?;
    m.add_function(wrap_pyfunction!(fft::rfftfreq, &m)?)?;
    parent.add_submodule(&m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("ferray._ferray.fft", &m)?;
    Ok(())
}

/// Register `ferray.emath` (a.k.a. `numpy.lib.scimath`) — the complex-aware
/// sqrt/log/log2/log10/logn/power/arccos/arcsin/arctanh family. Each function
/// returns a COMPLEX result for inputs outside the real domain (`x < 0` for
/// sqrt/log/power-base; `|x| > 1` for arccos/arcsin/arctanh) and a REAL result
/// otherwise, mirroring `numpy/lib/_scimath_impl.py` exactly.
fn register_emath_module<'py>(py: Python<'py>, parent: &Bound<'py, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "emath")?;
    m.add_function(wrap_pyfunction!(emath::sqrt, &m)?)?;
    m.add_function(wrap_pyfunction!(emath::log, &m)?)?;
    m.add_function(wrap_pyfunction!(emath::log2, &m)?)?;
    m.add_function(wrap_pyfunction!(emath::log10, &m)?)?;
    m.add_function(wrap_pyfunction!(emath::logn, &m)?)?;
    m.add_function(wrap_pyfunction!(emath::power, &m)?)?;
    m.add_function(wrap_pyfunction!(emath::arccos, &m)?)?;
    m.add_function(wrap_pyfunction!(emath::arcsin, &m)?)?;
    m.add_function(wrap_pyfunction!(emath::arctanh, &m)?)?;
    parent.add_submodule(&m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("ferray._ferray.emath", &m)?;
    Ok(())
}

/// Register `ferray.window` with all phase-4 window function bindings.
fn register_window_module<'py>(py: Python<'py>, parent: &Bound<'py, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "window")?;
    m.add_function(wrap_pyfunction!(window::hanning, &m)?)?;
    m.add_function(wrap_pyfunction!(window::hamming, &m)?)?;
    m.add_function(wrap_pyfunction!(window::blackman, &m)?)?;
    m.add_function(wrap_pyfunction!(window::bartlett, &m)?)?;
    m.add_function(wrap_pyfunction!(window::kaiser, &m)?)?;
    m.add_function(wrap_pyfunction!(window::cosine, &m)?)?;
    m.add_function(wrap_pyfunction!(window::nuttall, &m)?)?;
    m.add_function(wrap_pyfunction!(window::parzen, &m)?)?;
    m.add_function(wrap_pyfunction!(window::gaussian, &m)?)?;
    m.add_function(wrap_pyfunction!(window::exponential, &m)?)?;
    m.add_function(wrap_pyfunction!(window::tukey, &m)?)?;
    m.add_function(wrap_pyfunction!(window::general_cosine, &m)?)?;
    m.add_function(wrap_pyfunction!(window::general_hamming, &m)?)?;
    m.add_function(wrap_pyfunction!(window::taylor, &m)?)?;
    parent.add_submodule(&m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("ferray._ferray.window", &m)?;
    Ok(())
}

/// Register `ferray.polynomial` with all phase-4 polynomial-function bindings.
fn register_polynomial_module<'py>(py: Python<'py>, parent: &Bound<'py, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "polynomial")?;
    m.add_function(wrap_pyfunction!(polynomial::polyvalfromroots, &m)?)?;
    m.add_function(wrap_pyfunction!(polynomial::polyval2d, &m)?)?;
    m.add_function(wrap_pyfunction!(polynomial::polyval3d, &m)?)?;
    m.add_function(wrap_pyfunction!(polynomial::polygrid2d, &m)?)?;
    m.add_function(wrap_pyfunction!(polynomial::polygrid3d, &m)?)?;
    m.add_function(wrap_pyfunction!(polynomial::polyvander2d, &m)?)?;
    m.add_function(wrap_pyfunction!(polynomial::polyvander3d, &m)?)?;
    m.add_function(wrap_pyfunction!(polynomial::chebpts1, &m)?)?;
    m.add_function(wrap_pyfunction!(polynomial::chebpts2, &m)?)?;
    m.add_function(wrap_pyfunction!(polynomial::chebweight, &m)?)?;
    m.add_function(wrap_pyfunction!(polynomial::chebgauss, &m)?)?;
    m.add_function(wrap_pyfunction!(polynomial::leggauss, &m)?)?;
    m.add_function(wrap_pyfunction!(polynomial::hermgauss, &m)?)?;
    m.add_function(wrap_pyfunction!(polynomial::hermegauss, &m)?)?;
    m.add_function(wrap_pyfunction!(polynomial::laggauss, &m)?)?;
    m.add_function(wrap_pyfunction!(polynomial::poly2cheb, &m)?)?;
    m.add_function(wrap_pyfunction!(polynomial::cheb2poly, &m)?)?;
    m.add_function(wrap_pyfunction!(polynomial::poly2herm, &m)?)?;
    m.add_function(wrap_pyfunction!(polynomial::herm2poly, &m)?)?;
    m.add_function(wrap_pyfunction!(polynomial::poly2herme, &m)?)?;
    m.add_function(wrap_pyfunction!(polynomial::herme2poly, &m)?)?;
    m.add_function(wrap_pyfunction!(polynomial::poly2lag, &m)?)?;
    m.add_function(wrap_pyfunction!(polynomial::lag2poly, &m)?)?;
    m.add_function(wrap_pyfunction!(polynomial::poly2leg, &m)?)?;
    m.add_function(wrap_pyfunction!(polynomial::leg2poly, &m)?)?;
    // Class API
    m.add_class::<polynomial::Polynomial>()?;
    m.add_class::<polynomial::Chebyshev>()?;
    m.add_class::<polynomial::Hermite>()?;
    m.add_class::<polynomial::HermiteE>()?;
    m.add_class::<polynomial::Laguerre>()?;
    m.add_class::<polynomial::Legendre>()?;
    parent.add_submodule(&m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("ferray._ferray.polynomial", &m)?;
    Ok(())
}

/// Register `ferray.autodiff` with the DualNumber class and elementary functions.
fn register_autodiff_module<'py>(py: Python<'py>, parent: &Bound<'py, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "autodiff")?;
    m.add_class::<autodiff::PyDual>()?;
    m.add_function(wrap_pyfunction!(autodiff::sin, &m)?)?;
    m.add_function(wrap_pyfunction!(autodiff::cos, &m)?)?;
    m.add_function(wrap_pyfunction!(autodiff::tan, &m)?)?;
    m.add_function(wrap_pyfunction!(autodiff::asin, &m)?)?;
    m.add_function(wrap_pyfunction!(autodiff::acos, &m)?)?;
    m.add_function(wrap_pyfunction!(autodiff::atan, &m)?)?;
    m.add_function(wrap_pyfunction!(autodiff::atan2, &m)?)?;
    m.add_function(wrap_pyfunction!(autodiff::sinh, &m)?)?;
    m.add_function(wrap_pyfunction!(autodiff::cosh, &m)?)?;
    m.add_function(wrap_pyfunction!(autodiff::tanh, &m)?)?;
    m.add_function(wrap_pyfunction!(autodiff::exp, &m)?)?;
    m.add_function(wrap_pyfunction!(autodiff::ln, &m)?)?;
    m.add_function(wrap_pyfunction!(autodiff::log2, &m)?)?;
    m.add_function(wrap_pyfunction!(autodiff::log10, &m)?)?;
    m.add_function(wrap_pyfunction!(autodiff::sqrt, &m)?)?;
    m.add_function(wrap_pyfunction!(autodiff::abs, &m)?)?;
    parent.add_submodule(&m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("ferray._ferray.autodiff", &m)?;
    Ok(())
}

/// Register `ferray.ma` with the masked-array class and constructors.
fn register_ma_module<'py>(py: Python<'py>, parent: &Bound<'py, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "ma")?;
    m.add_class::<ma::PyMaskedArray>()?;
    m.add_function(wrap_pyfunction!(ma::array, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::masked_array, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::masked_where, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::masked_invalid, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::masked_equal, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::masked_not_equal, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::masked_greater, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::masked_greater_equal, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::masked_less, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::masked_less_equal, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::masked_inside, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::masked_outside, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::count_masked, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::is_masked, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::getmask, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::getdata, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::filled, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::compressed, &m)?)?;
    // ----- numpy.ma expansion (refs #818) -----
    // unary elementwise ufuncs
    m.add_function(wrap_pyfunction!(ma::sin, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::cos, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::tan, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::arctan, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::sinh, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::cosh, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::tanh, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::arcsinh, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::exp, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::floor, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::ceil, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::negative, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::absolute, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::abs, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::fabs, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::conjugate, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::sqrt, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::log, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::log2, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::log10, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::arcsin, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::arccos, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::arccosh, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::arctanh, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::around, &m)?)?;
    // binary elementwise ufuncs
    m.add_function(wrap_pyfunction!(ma::add, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::subtract, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::multiply, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::divide, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::true_divide, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::floor_divide, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::power, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::arctan2, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::hypot, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::fmod, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::remainder, &m)?)?;
    m.add("mod", wrap_pyfunction!(ma::mod_, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::maximum, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::minimum, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::bitwise_and, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::bitwise_or, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::bitwise_xor, &m)?)?;
    // reductions
    m.add_function(wrap_pyfunction!(ma::prod, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::median, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::ptp, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::argmin, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::argmax, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::count, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::average, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::all, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::any, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::anom, &m)?)?;
    // creation
    m.add_function(wrap_pyfunction!(ma::zeros, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::ones, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::empty, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::arange, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::identity, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::asarray, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::asanyarray, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::copy, &m)?)?;
    // manipulation
    m.add_function(wrap_pyfunction!(ma::reshape, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::ravel, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::transpose, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::squeeze, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::expand_dims, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::concatenate, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::diag, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::repeat, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::clip, &m)?)?;
    // mask helpers + predicates
    m.add_function(wrap_pyfunction!(ma::getmaskarray, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::make_mask, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::make_mask_none, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::mask_or, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::masked_values, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::masked_object, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::fix_invalid, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::is_mask, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::is_masked_array, &m)?)?;
    // numpy aliases: isMA / isarray == isMaskedArray
    m.add("isMA", wrap_pyfunction!(ma::is_masked_array, &m)?)?;
    m.add("isarray", wrap_pyfunction!(ma::is_masked_array, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::set_fill_value, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::default_fill_value, &m)?)?;
    // ----- numpy.ma specialized algorithms (refs #835) -----
    m.add_function(wrap_pyfunction!(ma::sort, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::argsort, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::take, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::trace, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::dot, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::unique, &m)?)?;
    // ----- numpy.ma specialized algorithms (refs #835) -----
    m.add_function(wrap_pyfunction!(ma::where_, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::choose, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::diff, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::ediff1d, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::nonzero, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::vander, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::isin, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::in1d, &m)?)?;
    // numpy.ma mask-structure run-length analysis (#835).
    m.add_function(wrap_pyfunction!(ma::clump_masked, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::clump_unmasked, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::flatnotmasked_contiguous, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::flatnotmasked_edges, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::notmasked_contiguous, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::notmasked_edges, &m)?)?;
    // numpy.ma set ops + row/col suppression + masked cov/corrcoef (#835).
    m.add_function(wrap_pyfunction!(ma::intersect1d, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::union1d, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::setdiff1d, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::setxor1d, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::compress_rowcols, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::compress_rows, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::compress_cols, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::mask_rowcols, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::cov, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::corrcoef, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::polyfit, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::convolve, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::correlate, &m)?)?;
    // ----- numpy.ma composable manipulation/elementwise/aliases batch
    // (refs #835 #818) -----
    // mask-propagating manipulation
    m.add_function(wrap_pyfunction!(ma::atleast_1d, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::atleast_2d, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::atleast_3d, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::column_stack, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::dstack, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::vstack, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::hstack, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::stack, &m)?)?;
    m.add("row_stack", wrap_pyfunction!(ma::vstack, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::diagonal, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::diagflat, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::swapaxes, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::resize, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::compress, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::append, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::empty_like, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::ones_like, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::zeros_like, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::cumsum, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::cumprod, &m)?)?;
    // rounding + alias
    m.add_function(wrap_pyfunction!(ma::round, &m)?)?;
    m.add("round_", wrap_pyfunction!(ma::round, &m)?)?;
    // masked comparisons
    m.add_function(wrap_pyfunction!(ma::equal, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::not_equal, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::greater, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::greater_equal, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::less, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::less_equal, &m)?)?;
    // masked angle
    m.add_function(wrap_pyfunction!(ma::angle, &m)?)?;
    // reduction free-functions + numpy.ma aliases
    m.add_function(wrap_pyfunction!(ma::max, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::min, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::sum, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::mean, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::std, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::var, &m)?)?;
    m.add("amax", wrap_pyfunction!(ma::max, &m)?)?;
    m.add("amin", wrap_pyfunction!(ma::min, &m)?)?;
    m.add("product", wrap_pyfunction!(ma::prod, &m)?)?;
    m.add("alltrue", wrap_pyfunction!(ma::all, &m)?)?;
    m.add("sometrue", wrap_pyfunction!(ma::any, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::anomalies, &m)?)?;
    // predicates / fill-value helpers
    m.add_function(wrap_pyfunction!(ma::allequal, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::allclose, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::maximum_fill_value, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::minimum_fill_value, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::common_fill_value, &m)?)?;
    // numpy.ma shared exception/type vocabulary — re-export numpy's own
    // `MAError` / `MaskError` exception classes and the `MaskType` (== numpy
    // bool scalar), matching `numpy/ma/core.py` so `except numpy.ma.MAError`
    // catches ferray.ma errors of the same provenance.
    {
        let np_ma = py.import("numpy.ma")?;
        m.add("MAError", np_ma.getattr("MAError")?)?;
        m.add("MaskError", np_ma.getattr("MaskError")?)?;
        m.add("MaskType", np_ma.getattr("MaskType")?)?;
        // numpy.ma module-level singletons / constants: the `masked`
        // singleton, `nomask`, and `masked_singleton` (== masked) come from
        // numpy's own canonical objects so `x is ferray.ma.masked` matches
        // `numpy/ma/core.py`'s shared sentinels.
        m.add("masked", np_ma.getattr("masked")?)?;
        m.add("masked_singleton", np_ma.getattr("masked")?)?;
        m.add("nomask", np_ma.getattr("nomask")?)?;
    }
    // shape inspectors + masked logical / shift / product surface
    m.add_function(wrap_pyfunction!(ma::ndim, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::shape, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::size, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::logical_and, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::logical_or, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::logical_xor, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::logical_not, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::outer, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::inner, &m)?)?;
    m.add("outerproduct", wrap_pyfunction!(ma::outer, &m)?)?;
    m.add("innerproduct", wrap_pyfunction!(ma::inner, &m)?)?;
    // numpy.ma composable batch 3 (refs #835 #818): array-of-masked
    // constructors, masked iteration / split, whole-row/col masking, multi-axis
    // compress, the `bool_` dtype-scalar re-export, and `ids`.
    m.add_function(wrap_pyfunction!(ma::masked_all, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::masked_all_like, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::fromfunction, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::indices, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::apply_along_axis, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::apply_over_axes, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::ndenumerate, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::hsplit, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::mask_rows, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::mask_cols, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::compress_nd, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::ids, &m)?)?;
    // numpy.ma structured-mask + integer-bitwise batch 4 (refs #835 #818):
    // structured-dtype mask vocabulary, buffer constructor, and the
    // dtype-preserving integer masked bitwise shifts.
    m.add_function(wrap_pyfunction!(ma::make_mask_descr, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::flatten_mask, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::flatten_structured_array, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::fromflex, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::frombuffer, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::left_shift, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::right_shift, &m)?)?;
    // `bool_` — the numpy bool scalar type, shared vocabulary (== fr.bool_).
    // `mvoid` — the numpy.ma scalar type for a single masked structured/record
    // element, re-exported as shared scalar vocabulary (like np.float64).
    {
        let np = py.import("numpy")?;
        m.add("bool_", np.getattr("bool_")?)?;
        let np_ma = py.import("numpy.ma")?;
        m.add("mvoid", np_ma.getattr("mvoid")?)?;
    }
    // ----- stateful mask: harden/soften/put/putmask (#842 #843 #844 #845) -----
    m.add_function(wrap_pyfunction!(ma::harden_mask, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::soften_mask, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::put, &m)?)?;
    m.add_function(wrap_pyfunction!(ma::putmask, &m)?)?;
    parent.add_submodule(&m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("ferray._ferray.ma", &m)?;
    Ok(())
}

/// Register `ferray.char` (legacy `numpy.char`) with the full shared
/// string-op surface. `numpy.char` and `numpy.strings` are the same
/// operations; both modules register the identical function set via
/// [`strings::register_string_ops`].
fn register_char_module<'py>(py: Python<'py>, parent: &Bound<'py, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "char")?;
    strings::register_string_ops(py, &m)?;
    parent.add_submodule(&m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("ferray._ferray.char", &m)?;
    Ok(())
}

/// Register `ferray.strings` — the NumPy 2.0 canonical `numpy.strings`
/// namespace — over the same shared string-op surface as `ferray.char`.
fn register_strings_module<'py>(py: Python<'py>, parent: &Bound<'py, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "strings")?;
    strings::register_string_ops(py, &m)?;
    parent.add_submodule(&m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("ferray._ferray.strings", &m)?;
    Ok(())
}

/// Register `ferray.lib.stride_tricks` matching the NumPy submodule
/// path. The top-level `broadcast_arrays` / `broadcast_shapes` /
/// `sliding_window_view` aliases are added directly on the parent
/// module (matching `numpy.broadcast_arrays`).
fn register_stride_tricks_module<'py>(
    py: Python<'py>,
    parent: &Bound<'py, PyModule>,
) -> PyResult<()> {
    let lib_m = PyModule::new(py, "lib")?;
    let st_m = PyModule::new(py, "stride_tricks")?;
    st_m.add_function(wrap_pyfunction!(stride_tricks::broadcast_arrays, &st_m)?)?;
    st_m.add_function(wrap_pyfunction!(stride_tricks::broadcast_shapes, &st_m)?)?;
    st_m.add_function(wrap_pyfunction!(stride_tricks::sliding_window_view, &st_m)?)?;
    // numpy/lib/_stride_tricks_impl.py:39 — exposed on
    // `numpy.lib.stride_tricks.as_strided`.
    st_m.add_function(wrap_pyfunction!(stride_tricks::as_strided, &st_m)?)?;
    // broadcast_to lives on `numpy.lib.stride_tricks` too (it is defined in
    // _stride_tricks_impl.py); expose it here for namespace parity.
    st_m.add_function(wrap_pyfunction!(stride_tricks::broadcast_to, &st_m)?)?;
    lib_m.add_submodule(&st_m)?;
    parent.add_submodule(&lib_m)?;
    let sys_modules = py.import("sys")?.getattr("modules")?;
    sys_modules.set_item("ferray._ferray.lib", &lib_m)?;
    sys_modules.set_item("ferray._ferray.lib.stride_tricks", &st_m)?;
    Ok(())
}

/// Register `ferray.random` with all phase-3 random bindings.
fn register_random_module<'py>(py: Python<'py>, parent: &Bound<'py, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "random")?;
    // Bind default_rng as the canonical NumPy name (overriding the
    // py-prefixed Rust ident).
    m.add("default_rng", wrap_pyfunction!(random::default_rng_py, &m)?)?;
    m.add_function(wrap_pyfunction!(random::seed, &m)?)?;
    m.add_function(wrap_pyfunction!(random::random, &m)?)?;
    m.add_function(wrap_pyfunction!(random::random_sample, &m)?)?;
    m.add_function(wrap_pyfunction!(random::sample, &m)?)?;
    m.add_function(wrap_pyfunction!(random::ranf, &m)?)?;
    m.add_function(wrap_pyfunction!(random::random_integers, &m)?)?;
    m.add_function(wrap_pyfunction!(random::bytes, &m)?)?;
    m.add_function(wrap_pyfunction!(random::rand, &m)?)?;
    m.add_function(wrap_pyfunction!(random::standard_normal, &m)?)?;
    m.add_function(wrap_pyfunction!(random::randn, &m)?)?;
    m.add_function(wrap_pyfunction!(random::normal, &m)?)?;
    m.add_function(wrap_pyfunction!(random::uniform, &m)?)?;
    m.add_function(wrap_pyfunction!(random::integers, &m)?)?;
    m.add_function(wrap_pyfunction!(random::randint, &m)?)?;
    m.add_function(wrap_pyfunction!(random::permutation, &m)?)?;
    m.add_function(wrap_pyfunction!(random::shuffle, &m)?)?;
    m.add_function(wrap_pyfunction!(random::choice, &m)?)?;
    // distributions
    m.add_function(wrap_pyfunction!(random::exponential, &m)?)?;
    m.add_function(wrap_pyfunction!(random::rayleigh, &m)?)?;
    m.add_function(wrap_pyfunction!(random::weibull, &m)?)?;
    m.add_function(wrap_pyfunction!(random::pareto, &m)?)?;
    m.add_function(wrap_pyfunction!(random::chisquare, &m)?)?;
    m.add_function(wrap_pyfunction!(random::geometric, &m)?)?;
    m.add_function(wrap_pyfunction!(random::poisson, &m)?)?;
    m.add_function(wrap_pyfunction!(random::lognormal, &m)?)?;
    m.add_function(wrap_pyfunction!(random::binomial, &m)?)?;
    m.add_function(wrap_pyfunction!(random::gamma, &m)?)?;
    m.add_function(wrap_pyfunction!(random::beta, &m)?)?;
    m.add_function(wrap_pyfunction!(random::laplace, &m)?)?;
    m.add_function(wrap_pyfunction!(random::gumbel, &m)?)?;
    m.add_function(wrap_pyfunction!(random::triangular, &m)?)?;
    // distributions newly exposed at the module surface (#787 follow-up)
    m.add_function(wrap_pyfunction!(random::logistic, &m)?)?;
    m.add_function(wrap_pyfunction!(random::power, &m)?)?;
    m.add_function(wrap_pyfunction!(random::standard_t, &m)?)?;
    m.add_function(wrap_pyfunction!(random::f_dist, &m)?)?;
    m.add_function(wrap_pyfunction!(random::vonmises, &m)?)?;
    m.add_function(wrap_pyfunction!(random::wald, &m)?)?;
    m.add_function(wrap_pyfunction!(random::standard_cauchy, &m)?)?;
    m.add_function(wrap_pyfunction!(random::standard_exponential, &m)?)?;
    m.add_function(wrap_pyfunction!(random::standard_gamma, &m)?)?;
    m.add_function(wrap_pyfunction!(random::negative_binomial, &m)?)?;
    m.add_function(wrap_pyfunction!(random::hypergeometric, &m)?)?;
    // multivariate / extra distributions newly bound over ferray-random
    // (refs #834 #818)
    m.add_function(wrap_pyfunction!(random::dirichlet, &m)?)?;
    m.add_function(wrap_pyfunction!(random::multinomial, &m)?)?;
    m.add_function(wrap_pyfunction!(random::multivariate_normal, &m)?)?;
    m.add_function(wrap_pyfunction!(random::multivariate_hypergeometric, &m)?)?;
    m.add_function(wrap_pyfunction!(random::logseries, &m)?)?;
    m.add_function(wrap_pyfunction!(random::noncentral_chisquare, &m)?)?;
    m.add_function(wrap_pyfunction!(random::noncentral_f, &m)?)?;
    m.add_function(wrap_pyfunction!(random::zipf, &m)?)?;
    // Modern Generator class
    m.add_class::<random::PyGenerator>()?;
    m.add_function(wrap_pyfunction!(random::default_rng_py, &m)?)?;
    // BitGenerator family + SeedSequence + legacy RandomState (refs #834 #818)
    m.add_class::<random::PyBitGenerator>()?;
    m.add_class::<random::PyMT19937>()?;
    m.add_class::<random::PyPCG64>()?;
    m.add_class::<random::PyPCG64DXSM>()?;
    m.add_class::<random::PyPhilox>()?;
    m.add_class::<random::PySFC64>()?;
    m.add_class::<random::PySeedSequence>()?;
    m.add_class::<random::PyRandomState>()?;
    m.add_function(wrap_pyfunction!(random::get_state, &m)?)?;
    m.add_function(wrap_pyfunction!(random::set_state, &m)?)?;
    parent.add_submodule(&m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("ferray._ferray.random", &m)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Module entry point
// ---------------------------------------------------------------------------

/// PyO3 module entry point. The cdylib exports `PyInit__ferray` which
/// CPython calls on `import ferray._ferray`.
#[pymodule]
fn _ferray(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // ----- creation surface (phase 0 + phase 1) -----
    m.add_function(wrap_pyfunction!(creation::zeros, m)?)?;
    m.add_function(wrap_pyfunction!(creation::ones, m)?)?;
    m.add_function(wrap_pyfunction!(creation::empty, m)?)?;
    m.add_function(wrap_pyfunction!(creation::full, m)?)?;
    m.add_function(wrap_pyfunction!(creation::identity, m)?)?;
    m.add_function(wrap_pyfunction!(creation::eye, m)?)?;
    m.add_function(wrap_pyfunction!(creation::tri, m)?)?;
    m.add_function(wrap_pyfunction!(creation::arange, m)?)?;
    m.add_function(wrap_pyfunction!(creation::linspace, m)?)?;
    m.add_function(wrap_pyfunction!(creation::logspace, m)?)?;
    m.add_function(wrap_pyfunction!(creation::geomspace, m)?)?;
    m.add_function(wrap_pyfunction!(creation::zeros_like, m)?)?;
    m.add_function(wrap_pyfunction!(creation::ones_like, m)?)?;
    m.add_function(wrap_pyfunction!(creation::empty_like, m)?)?;
    m.add_function(wrap_pyfunction!(creation::full_like, m)?)?;
    m.add_function(wrap_pyfunction!(creation::copy, m)?)?;
    m.add_function(wrap_pyfunction!(creation::ascontiguousarray, m)?)?;
    m.add_function(wrap_pyfunction!(creation::asfortranarray, m)?)?;
    m.add_function(wrap_pyfunction!(creation::asanyarray, m)?)?;
    m.add_function(wrap_pyfunction!(creation::array, m)?)?;
    m.add_function(wrap_pyfunction!(creation::asarray, m)?)?;
    m.add_function(wrap_pyfunction!(creation::meshgrid, m)?)?;
    m.add_function(wrap_pyfunction!(creation::mgrid, m)?)?;
    m.add_function(wrap_pyfunction!(creation::ogrid, m)?)?;
    m.add_function(wrap_pyfunction!(creation::vander, m)?)?;
    m.add_function(wrap_pyfunction!(creation::frombuffer, m)?)?;
    m.add_function(wrap_pyfunction!(creation::fromiter, m)?)?;
    m.add_function(wrap_pyfunction!(creation::fromstring, m)?)?;
    m.add_function(wrap_pyfunction!(creation::fromfile, m)?)?;
    m.add_function(wrap_pyfunction!(creation::fromfunction, m)?)?;

    // ----- manipulation surface -----
    m.add_function(wrap_pyfunction!(manipulation::reshape, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::ravel, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::flatten, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::squeeze, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::expand_dims, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::broadcast_to, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::transpose, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::swapaxes, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::moveaxis, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::rollaxis, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::flip, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::fliplr, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::flipud, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::rot90, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::roll, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::atleast_1d, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::atleast_2d, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::atleast_3d, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::tril, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::triu, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::diag, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::diagflat, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::concatenate, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::stack, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::vstack, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::hstack, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::dstack, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::pad, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::tile, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::repeat, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::delete, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::insert, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::append, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::resize, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::trim_zeros, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::column_stack, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::row_stack, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::block, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::r_, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::c_, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::split, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::array_split, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::vsplit, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::hsplit, m)?)?;
    m.add_function(wrap_pyfunction!(manipulation::dsplit, m)?)?;

    // ----- ufunc surface (phase 2) -----
    // unary float
    m.add_function(wrap_pyfunction!(ufunc::sin, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::cos, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::tan, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::arcsin, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::arccos, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::arctan, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::sinh, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::cosh, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::tanh, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::arcsinh, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::arccosh, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::arctanh, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::degrees, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::radians, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::deg2rad, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::rad2deg, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::exp, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::exp2, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::expm1, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::log, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::log1p, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::log2, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::log10, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::cbrt, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::square, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::reciprocal, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::negative, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::positive, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::absolute, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::fabs, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::sign, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::floor, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::ceil, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::round, m)?)?;
    // numpy.around / numpy.round both accept a `decimals` kwarg; `around`
    // is the canonical fromnumeric.py name (round_ is its alias).
    m.add_function(wrap_pyfunction!(ufunc::around, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::nan_to_num, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::unwrap, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::trunc, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::rint, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::fix, m)?)?;
    // predicates
    m.add_function(wrap_pyfunction!(ufunc::isnan, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::isinf, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::isfinite, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::isneginf, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::isposinf, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::signbit, m)?)?;
    // binary numeric
    m.add_function(wrap_pyfunction!(ufunc::add, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::subtract, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::multiply, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::divide, m)?)?;
    // true_divide is an alias of divide; floor_divide / remainder / mod /
    // float_power are top-level numpy ufuncs (umath.py __all__,
    // generate_umath.py:404/:405/:490). `mod` is a Python keyword, so the
    // Rust ident `mod_` is registered under the name "mod".
    m.add_function(wrap_pyfunction!(ufunc::true_divide, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::floor_divide, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::remainder, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::float_power, m)?)?;
    m.add("mod", wrap_pyfunction!(ufunc::mod_, m)?)?;
    // binary float
    m.add_function(wrap_pyfunction!(ufunc::power, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::maximum, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::minimum, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::fmax, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::fmin, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::copysign, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::hypot, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::arctan2, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::logaddexp, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::logaddexp2, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::heaviside, m)?)?;
    // comparisons
    m.add_function(wrap_pyfunction!(ufunc::equal, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::not_equal, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::less, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::less_equal, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::greater, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::greater_equal, m)?)?;
    // logical
    m.add_function(wrap_pyfunction!(ufunc::logical_and, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::logical_or, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::logical_xor, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::logical_not, m)?)?;
    // misc
    m.add_function(wrap_pyfunction!(ufunc::clip, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::array_equal, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::array_equiv, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::allclose, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::isclose, m)?)?;
    // datetime64 / timedelta64 top-level surface (refs #831): isnat,
    // datetime_as_string, and the busday calendar (is_busday / busday_count /
    // busday_offset). datetime arithmetic flows through ufunc::add /
    // ufunc::subtract which dispatch to crate::datetime for time dtypes.
    m.add_function(wrap_pyfunction!(datetime::isnat, m)?)?;
    m.add_function(wrap_pyfunction!(datetime::datetime_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(datetime::is_busday, m)?)?;
    m.add_function(wrap_pyfunction!(datetime::busday_count, m)?)?;
    m.add_function(wrap_pyfunction!(datetime::busday_offset, m)?)?;
    // bitwise
    m.add_function(wrap_pyfunction!(ufunc::bitwise_and, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::bitwise_or, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::bitwise_xor, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::bitwise_not, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::invert, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::left_shift, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::right_shift, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::bitwise_count, m)?)?;
    // complex
    m.add_function(wrap_pyfunction!(complex::real, m)?)?;
    m.add_function(wrap_pyfunction!(complex::imag, m)?)?;
    m.add_function(wrap_pyfunction!(complex::conj, m)?)?;
    m.add_function(wrap_pyfunction!(complex::conjugate, m)?)?;
    m.add_function(wrap_pyfunction!(complex::angle, m)?)?;
    m.add_function(wrap_pyfunction!(complex::real_if_close, m)?)?;
    m.add_function(wrap_pyfunction!(complex::iscomplex, m)?)?;
    m.add_function(wrap_pyfunction!(complex::isreal, m)?)?;
    m.add_function(wrap_pyfunction!(complex::iscomplexobj, m)?)?;
    m.add_function(wrap_pyfunction!(complex::isrealobj, m)?)?;
    // float intrinsics
    m.add_function(wrap_pyfunction!(ufunc::exp_fast, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::spacing, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::gcd, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::lcm, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::fmod, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::nextafter, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::divmod, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::frexp, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::ldexp, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::modf, m)?)?;
    // signal / special
    m.add_function(wrap_pyfunction!(ufunc::sinc, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::i0, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::gradient, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::trapezoid, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::ediff1d, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::convolve, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::correlate, m)?)?;
    m.add_function(wrap_pyfunction!(ufunc::interp, m)?)?;

    // ----- stats surface (phase 2) -----
    m.add_function(wrap_pyfunction!(stats::sum, m)?)?;
    m.add_function(wrap_pyfunction!(stats::prod, m)?)?;
    m.add_function(wrap_pyfunction!(stats::min, m)?)?;
    m.add_function(wrap_pyfunction!(stats::max, m)?)?;
    m.add_function(wrap_pyfunction!(stats::ptp, m)?)?;
    m.add_function(wrap_pyfunction!(stats::mean, m)?)?;
    m.add_function(wrap_pyfunction!(stats::var, m)?)?;
    m.add_function(wrap_pyfunction!(stats::std, m)?)?;
    m.add_function(wrap_pyfunction!(stats::argmin, m)?)?;
    m.add_function(wrap_pyfunction!(stats::argmax, m)?)?;
    // boolean reductions
    m.add_function(wrap_pyfunction!(stats::all, m)?)?;
    m.add_function(wrap_pyfunction!(stats::any, m)?)?;
    m.add_function(wrap_pyfunction!(stats::average, m)?)?;
    // nan-aware
    m.add_function(wrap_pyfunction!(stats::nansum, m)?)?;
    m.add_function(wrap_pyfunction!(stats::nanprod, m)?)?;
    m.add_function(wrap_pyfunction!(stats::nanmean, m)?)?;
    m.add_function(wrap_pyfunction!(stats::nanmin, m)?)?;
    m.add_function(wrap_pyfunction!(stats::nanmax, m)?)?;
    m.add_function(wrap_pyfunction!(stats::nanvar, m)?)?;
    m.add_function(wrap_pyfunction!(stats::nanstd, m)?)?;
    m.add_function(wrap_pyfunction!(stats::nanmedian, m)?)?;
    m.add_function(wrap_pyfunction!(stats::nanargmin, m)?)?;
    m.add_function(wrap_pyfunction!(stats::nanargmax, m)?)?;
    m.add_function(wrap_pyfunction!(stats::nanpercentile, m)?)?;
    m.add_function(wrap_pyfunction!(stats::nanquantile, m)?)?;
    // cumulative
    m.add_function(wrap_pyfunction!(stats::cumsum, m)?)?;
    m.add_function(wrap_pyfunction!(stats::cumprod, m)?)?;
    m.add_function(wrap_pyfunction!(stats::nancumsum, m)?)?;
    m.add_function(wrap_pyfunction!(stats::nancumprod, m)?)?;
    m.add_function(wrap_pyfunction!(stats::diff, m)?)?;
    // sort/search
    m.add_function(wrap_pyfunction!(stats::sort, m)?)?;
    m.add_function(wrap_pyfunction!(stats::argsort, m)?)?;
    m.add_function(wrap_pyfunction!(stats::searchsorted, m)?)?;
    m.add_function(wrap_pyfunction!(stats::partition, m)?)?;
    m.add_function(wrap_pyfunction!(stats::argpartition, m)?)?;
    m.add_function(wrap_pyfunction!(stats::lexsort, m)?)?;
    m.add_function(wrap_pyfunction!(stats::sort_complex, m)?)?;
    m.add_function(wrap_pyfunction!(stats::cross, m)?)?;
    // numpy.trace lives at the package root (numpy/__init__.pyi); reuse the
    // existing linalg trace binding.
    m.add_function(wrap_pyfunction!(linalg::trace, m)?)?;
    // unique / count
    m.add_function(wrap_pyfunction!(stats::unique, m)?)?;
    m.add_function(wrap_pyfunction!(stats::unique_extended, m)?)?;
    // quantile / cov / corrcoef
    m.add_function(wrap_pyfunction!(stats::percentile, m)?)?;
    m.add_function(wrap_pyfunction!(stats::quantile, m)?)?;
    m.add_function(wrap_pyfunction!(stats::median, m)?)?;
    m.add_function(wrap_pyfunction!(stats::cov, m)?)?;
    m.add_function(wrap_pyfunction!(stats::corrcoef, m)?)?;
    // histogram
    m.add_function(wrap_pyfunction!(stats::histogram, m)?)?;
    m.add_function(wrap_pyfunction!(stats::histogram_bin_edges, m)?)?;
    m.add_function(wrap_pyfunction!(stats::histogram2d, m)?)?;
    m.add_function(wrap_pyfunction!(stats::histogramdd, m)?)?;
    m.add_function(wrap_pyfunction!(stats::bincount, m)?)?;
    m.add_function(wrap_pyfunction!(stats::digitize, m)?)?;
    // Top-level vector/matrix products. numpy exposes dot/vdot/matmul/inner/
    // outer at the package root (numpy/__init__.pyi), not just under
    // numpy.linalg, so a drop-in `import ferray as np` must mirror them.
    m.add_function(wrap_pyfunction!(linalg::dot, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::vdot, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::matmul, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::inner, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::outer, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::tensordot, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::kron, m)?)?;
    // Top-level vector products (numpy 2.0+)
    m.add_function(wrap_pyfunction!(linalg::matvec, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::vecmat, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::vecdot, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::einsum, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::matrix_transpose, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::diagonal, m)?)?;
    m.add_function(wrap_pyfunction!(stats::count_nonzero, m)?)?;
    // set
    m.add_function(wrap_pyfunction!(stats::union1d, m)?)?;
    m.add_function(wrap_pyfunction!(stats::intersect1d, m)?)?;
    m.add_function(wrap_pyfunction!(stats::setdiff1d, m)?)?;
    m.add_function(wrap_pyfunction!(stats::setxor1d, m)?)?;
    m.add_function(wrap_pyfunction!(stats::in1d, m)?)?;
    m.add_function(wrap_pyfunction!(stats::isin, m)?)?;
    // where
    m.add_function(wrap_pyfunction!(stats::where_fn, m)?)?;

    // ----- indexing surface -----
    m.add_function(wrap_pyfunction!(indexing::indices, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::diag_indices, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::tril_indices, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::triu_indices, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::mask_indices, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::ravel_multi_index, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::unravel_index, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::flatnonzero, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::nonzero, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::argwhere, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::take, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::take_along_axis, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::choose, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::compress, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::select, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::place, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::putmask, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::put, m)?)?;
    m.add_function(wrap_pyfunction!(indexing::extract, m)?)?;

    // ----- file-IO surface (.npy/.npz binary + delimited text) -----
    // numpy exposes save/load/savez/savez_compressed/savetxt/loadtxt/
    // genfromtxt at the package root (numpy/lib/_npyio_impl.py). The
    // bindings marshal numpy.ndarray <-> ferray_core::DynArray over the
    // already-shipped ferray-io crate.
    m.add_function(wrap_pyfunction!(io::save, m)?)?;
    m.add_function(wrap_pyfunction!(io::load, m)?)?;
    m.add_function(wrap_pyfunction!(io::savez, m)?)?;
    m.add_function(wrap_pyfunction!(io::savez_compressed, m)?)?;
    m.add_function(wrap_pyfunction!(io::savetxt, m)?)?;
    m.add_function(wrap_pyfunction!(io::loadtxt, m)?)?;
    m.add_function(wrap_pyfunction!(io::genfromtxt, m)?)?;

    // Top-level window aliases (numpy puts these at top level).
    m.add_function(wrap_pyfunction!(window::hanning, m)?)?;
    m.add_function(wrap_pyfunction!(window::hamming, m)?)?;
    m.add_function(wrap_pyfunction!(window::blackman, m)?)?;
    m.add_function(wrap_pyfunction!(window::bartlett, m)?)?;
    m.add_function(wrap_pyfunction!(window::kaiser, m)?)?;

    // Top-level stride_tricks aliases.
    m.add_function(wrap_pyfunction!(stride_tricks::broadcast_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(stride_tricks::broadcast_shapes, m)?)?;
    // Override the manipulation-module `broadcast_to` with the stride_tricks
    // binding so the top-level `numpy.broadcast_to` contract holds: a
    // read-only view (_stride_tricks_impl.py:517 `readonly=True`) plus the
    // `subok` kwarg (:475). Registered AFTER manipulation::broadcast_to so it
    // wins.
    m.add_function(wrap_pyfunction!(stride_tricks::broadcast_to, m)?)?;

    // ----- top-level numpy alias + introspection surface -----
    // numpy/_core/__init__.py + numpy/_core/fromnumeric.py +
    // numpy/_core/numerictypes.py — array & dtype introspection plus the
    // `divmod` ufunc tuple. The pure 1:1 trig/bitwise/manip/reduction
    // aliases are bare Python re-exports in python/ferray/__init__.py.
    m.add_function(wrap_pyfunction!(aliases::ndim, m)?)?;
    m.add_function(wrap_pyfunction!(aliases::shape, m)?)?;
    m.add_function(wrap_pyfunction!(aliases::size, m)?)?;
    m.add_function(wrap_pyfunction!(aliases::isscalar, m)?)?;
    m.add_function(wrap_pyfunction!(aliases::isfortran, m)?)?;
    m.add_function(wrap_pyfunction!(aliases::astype, m)?)?;
    m.add_function(wrap_pyfunction!(aliases::can_cast, m)?)?;
    m.add_function(wrap_pyfunction!(aliases::promote_types, m)?)?;
    m.add_function(wrap_pyfunction!(aliases::result_type, m)?)?;
    m.add_function(wrap_pyfunction!(aliases::min_scalar_type, m)?)?;
    m.add_function(wrap_pyfunction!(aliases::issubdtype, m)?)?;
    m.add_function(wrap_pyfunction!(aliases::isdtype, m)?)?;
    m.add_function(wrap_pyfunction!(aliases::common_type, m)?)?;
    m.add_function(wrap_pyfunction!(aliases::divmod, m)?)?;

    // ----- top-level numpy poly1d family (numpy/lib/_polynomial_impl.py) -----
    // The classic 1-D polynomial functions on highest-degree-first coefficient
    // arrays: np.polyval / poly / roots / polyadd / polysub / polymul /
    // polyder / polyint / polyfit / polydiv.
    m.add_function(wrap_pyfunction!(polynomial::polyval, m)?)?;
    m.add_function(wrap_pyfunction!(polynomial::poly, m)?)?;
    m.add_function(wrap_pyfunction!(polynomial::roots, m)?)?;
    m.add_function(wrap_pyfunction!(polynomial::polyadd, m)?)?;
    m.add_function(wrap_pyfunction!(polynomial::polysub, m)?)?;
    m.add_function(wrap_pyfunction!(polynomial::polymul, m)?)?;
    m.add_function(wrap_pyfunction!(polynomial::polyder, m)?)?;
    m.add_function(wrap_pyfunction!(polynomial::polyint, m)?)?;
    m.add_function(wrap_pyfunction!(polynomial::polyfit, m)?)?;
    m.add_function(wrap_pyfunction!(polynomial::polydiv, m)?)?;

    register_dtype_module(py, m)?;
    register_linalg_module(py, m)?;
    register_fft_module(py, m)?;
    register_emath_module(py, m)?;
    register_random_module(py, m)?;
    register_window_module(py, m)?;
    register_polynomial_module(py, m)?;
    register_stride_tricks_module(py, m)?;
    register_char_module(py, m)?;
    register_strings_module(py, m)?;
    register_ma_module(py, m)?;
    register_autodiff_module(py, m)?;

    Ok(())
}
