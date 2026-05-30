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

mod autodiff;
mod char;
mod complex;
mod conv;
mod creation;
mod fft;
mod indexing;
mod linalg;
mod ma;
mod manipulation;
mod polynomial;
mod random;
mod stats;
mod stride_tricks;
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
    parent.add_submodule(&m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("ferray._ferray.ma", &m)?;
    Ok(())
}

/// Register `ferray.char` with all phase-4-followup char-namespace bindings.
fn register_char_module<'py>(py: Python<'py>, parent: &Bound<'py, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "char")?;
    m.add_function(wrap_pyfunction!(char::lower, &m)?)?;
    m.add_function(wrap_pyfunction!(char::upper, &m)?)?;
    m.add_function(wrap_pyfunction!(char::capitalize, &m)?)?;
    m.add_function(wrap_pyfunction!(char::title, &m)?)?;
    m.add_function(wrap_pyfunction!(char::swapcase, &m)?)?;
    m.add_function(wrap_pyfunction!(char::strip, &m)?)?;
    m.add_function(wrap_pyfunction!(char::lstrip, &m)?)?;
    m.add_function(wrap_pyfunction!(char::rstrip, &m)?)?;
    m.add_function(wrap_pyfunction!(char::count, &m)?)?;
    m.add_function(wrap_pyfunction!(char::find, &m)?)?;
    m.add_function(wrap_pyfunction!(char::startswith, &m)?)?;
    m.add_function(wrap_pyfunction!(char::endswith, &m)?)?;
    m.add_function(wrap_pyfunction!(char::str_len, &m)?)?;
    m.add_function(wrap_pyfunction!(char::replace, &m)?)?;
    m.add_function(wrap_pyfunction!(char::add, &m)?)?;
    m.add_function(wrap_pyfunction!(char::multiply, &m)?)?;
    m.add_function(wrap_pyfunction!(char::equal, &m)?)?;
    m.add_function(wrap_pyfunction!(char::not_equal, &m)?)?;
    m.add_function(wrap_pyfunction!(char::less, &m)?)?;
    m.add_function(wrap_pyfunction!(char::less_equal, &m)?)?;
    m.add_function(wrap_pyfunction!(char::greater, &m)?)?;
    m.add_function(wrap_pyfunction!(char::greater_equal, &m)?)?;
    parent.add_submodule(&m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("ferray._ferray.char", &m)?;
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
    // Modern Generator class
    m.add_class::<random::PyGenerator>()?;
    m.add_function(wrap_pyfunction!(random::default_rng_py, &m)?)?;
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
    // cumulative
    m.add_function(wrap_pyfunction!(stats::cumsum, m)?)?;
    m.add_function(wrap_pyfunction!(stats::cumprod, m)?)?;
    m.add_function(wrap_pyfunction!(stats::diff, m)?)?;
    // sort/search
    m.add_function(wrap_pyfunction!(stats::sort, m)?)?;
    m.add_function(wrap_pyfunction!(stats::argsort, m)?)?;
    m.add_function(wrap_pyfunction!(stats::searchsorted, m)?)?;
    m.add_function(wrap_pyfunction!(stats::partition, m)?)?;
    m.add_function(wrap_pyfunction!(stats::argpartition, m)?)?;
    m.add_function(wrap_pyfunction!(stats::lexsort, m)?)?;
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
    m.add_function(wrap_pyfunction!(indexing::extract, m)?)?;

    // Top-level window aliases (numpy puts these at top level).
    m.add_function(wrap_pyfunction!(window::hanning, m)?)?;
    m.add_function(wrap_pyfunction!(window::hamming, m)?)?;
    m.add_function(wrap_pyfunction!(window::blackman, m)?)?;
    m.add_function(wrap_pyfunction!(window::bartlett, m)?)?;
    m.add_function(wrap_pyfunction!(window::kaiser, m)?)?;

    // Top-level stride_tricks aliases.
    m.add_function(wrap_pyfunction!(stride_tricks::broadcast_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(stride_tricks::broadcast_shapes, m)?)?;

    register_dtype_module(py, m)?;
    register_linalg_module(py, m)?;
    register_fft_module(py, m)?;
    register_random_module(py, m)?;
    register_window_module(py, m)?;
    register_polynomial_module(py, m)?;
    register_stride_tricks_module(py, m)?;
    register_char_module(py, m)?;
    register_ma_module(py, m)?;
    register_autodiff_module(py, m)?;

    Ok(())
}
