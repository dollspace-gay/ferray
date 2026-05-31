//! Bindings for the `numpy` window function surface.
//!
//! NumPy exposes `hanning`, `hamming`, `blackman`, `bartlett`, and
//! `kaiser` at the top level; ferray-Rust groups all of them in the
//! `ferray-window` crate. We expose both surfaces:
//!
//! - The five canonical names at top-level `ferray.*` (matching NumPy).
//! - All 14 (including SciPy / scipy.signal.windows extras like
//!   `gaussian`, `tukey`, `nuttall`, `parzen`, …) under
//!   `ferray.window.*` for callers who want the full set.
//!
//! Every window returns `Array1<f64>` regardless of length; the
//! signature is uniform so a single macro generates all bindings.
//!
//! ## REQ status
//!
//! Every window callable this module registers is SHIPPED, delegating to
//! `ferray-window` (`fw::*`); a genuinely-fractional `M` for a canonical numpy
//! window is delegated to numpy's float64 formula at the boundary (the `fw`
//! kernels take a `usize` sample count). (Evidence = the registered
//! `#[pyfunction]` + the `fw::*` fn it delegates to; pytest GREEN.)
//!
//! SHIPPED — canonical numpy windows (top-level `ferray.*`, with fractional-`M`
//! numpy delegation via `bind_window_n!`/`window_m_fractional`):
//!   - `hanning` → `fw::hanning`, `hamming` → `fw::hamming`,
//!     `blackman` → `fw::blackman`, `bartlett` → `fw::bartlett`,
//!     `kaiser(M, beta)` → `fw::kaiser`.
//!
//! SHIPPED — SciPy-extra windows (`ferray.window.*`, integer-count path):
//!   - `cosine` → `fw::cosine`, `nuttall` → `fw::nuttall`,
//!     `parzen` → `fw::parzen`, `gaussian` → `fw::gaussian`,
//!     `exponential` → `fw::exponential`, `tukey` → `fw::tukey`,
//!     `general_cosine` → `fw::general_cosine`,
//!     `general_hamming` → `fw::general_hamming`, `taylor` → `fw::taylor`.
//!
//! Shared boundary contract: `M < 1` returns numpy's `array([], dtype=float64)`
//! (`empty_window`), matching every numpy window's `if M < 1` guard.
//!
//! NOT-STARTED: none — the full registered window surface is shipped.

use ferray_core::array::aliases::Array1;
use ferray_core::dimension::Ix1;
use ferray_numpy_interop::IntoNumPy;
use ferray_window as fw;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::conv::{coerce_window_m, ferr_to_pyerr, window_m_fractional};

/// Delegate a genuinely-fractional `M` window call to numpy, which owns the
/// float64 window formula (`values = np.array([0.0, M]); n = arange(1 - M,
/// M, 2); ...`, numpy/lib/_function_base_impl.py:3334-3342 for hanning; the
/// other canonical windows share the structure). The ferray-window kernels
/// take a `usize` sample count and so cannot reproduce the fractional-`M`
/// length/values; `func` is the numpy attribute name (`"hanning"`, …) and
/// `args` the call tuple (`(M,)` or `(M, beta)`).
fn numpy_window<'py>(
    py: Python<'py>,
    func: &str,
    args: impl pyo3::call::PyCallArgs<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    py.import("numpy")?.getattr(func)?.call1(args)
}

/// Build numpy's `array([], dtype=float64)` — the result every window returns
/// for `M < 1` (numpy/lib/_function_base_impl.py:3349-3350 `hanning`, and the
/// equivalent guard in every other window). Returned as a Python `ndarray`.
fn empty_window<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    let r: Array1<f64> =
        Array1::<f64>::from_vec(Ix1::new([0]), Vec::new()).map_err(ferr_to_pyerr)?;
    Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

// ---------------------------------------------------------------------------
// Single-arg windows (M -> Array1<f64>)
// ---------------------------------------------------------------------------

macro_rules! bind_window_n {
    // `$np_name` (optional) is the numpy top-level function this window
    // mirrors; when present, a genuinely-fractional `M` is delegated to numpy
    // (which keeps `M` as float64 through the formula — see `window_m_fractional`).
    // SciPy-only windows (no numpy top-level entry) omit it and keep the native
    // integer-count path unchanged.
    ($name:ident, $ferr_path:path, $np_name:literal) => {
        #[pyfunction]
        #[pyo3(signature = (M))]
        #[allow(
            non_snake_case,
            reason = "numpy names this parameter `M` (def hanning(M))"
        )]
        pub fn $name<'py>(py: Python<'py>, M: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
            // A genuinely-fractional M is not representable in ferray-window's
            // usize sample count; numpy owns the float64 formula. Integral /
            // int / negative / M==1 inputs stay on the native path below.
            if window_m_fractional(M)?.is_some() {
                return numpy_window(py, $np_name, (M,));
            }
            // numpy casts M to float64 and guards `if M < 1: return array([])`.
            let m = coerce_window_m(M)?;
            if m < 1 {
                return empty_window(py);
            }
            let r: Array1<f64> = $ferr_path(m as usize).map_err(ferr_to_pyerr)?;
            Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
        }
    };
    // SciPy-only window: no numpy fractional-M delegation.
    ($name:ident, $ferr_path:path) => {
        #[pyfunction]
        #[pyo3(signature = (M))]
        #[allow(
            non_snake_case,
            reason = "numpy names this parameter `M` (def hanning(M))"
        )]
        pub fn $name<'py>(py: Python<'py>, M: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
            let m = coerce_window_m(M)?;
            if m < 1 {
                return empty_window(py);
            }
            let r: Array1<f64> = $ferr_path(m as usize).map_err(ferr_to_pyerr)?;
            Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
        }
    };
}

bind_window_n!(hanning, fw::hanning, "hanning");
bind_window_n!(hamming, fw::hamming, "hamming");
bind_window_n!(blackman, fw::blackman, "blackman");
bind_window_n!(bartlett, fw::bartlett, "bartlett");
bind_window_n!(cosine, fw::cosine);
bind_window_n!(nuttall, fw::nuttall);
bind_window_n!(parzen, fw::parzen);

// ---------------------------------------------------------------------------
// Parametric windows
// ---------------------------------------------------------------------------

/// `numpy.kaiser(M, beta)`.
///
/// numpy names the length parameter `M` and casts it to float64
/// (numpy/lib/_function_base_impl.py:3727 `values = np.array([0.0, M, beta])`);
/// a non-positive `M` yields an empty float64 array because `n = arange(0, M)`
/// is empty (:3733). The binding mirrors that: coerce `M`, short-circuit
/// `M < 1 -> empty`, otherwise call the library.
#[pyfunction]
#[pyo3(signature = (M, beta))]
#[allow(
    non_snake_case,
    reason = "numpy names this parameter `M` (def kaiser(M, beta))"
)]
pub fn kaiser<'py>(
    py: Python<'py>,
    M: &Bound<'py, PyAny>,
    beta: f64,
) -> PyResult<Bound<'py, PyAny>> {
    // Fractional M -> numpy's float64 kaiser formula (n = arange(0, M);
    // i0(beta*sqrt(1-((n-alpha)/alpha)**2))/i0(beta),
    // numpy/lib/_function_base_impl.py kaiser :3596); ferray-window's usize
    // kernel can't reproduce a fractional sample count.
    if window_m_fractional(M)?.is_some() {
        return numpy_window(py, "kaiser", (M, beta));
    }
    let m = coerce_window_m(M)?;
    if m < 1 {
        return empty_window(py);
    }
    let r: Array1<f64> = fw::kaiser(m as usize, beta).map_err(ferr_to_pyerr)?;
    Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `scipy.signal.windows.gaussian(M, std)` equivalent.
#[pyfunction]
pub fn gaussian<'py>(py: Python<'py>, m: usize, std: f64) -> PyResult<Bound<'py, PyAny>> {
    let r: Array1<f64> = fw::gaussian(m, std).map_err(ferr_to_pyerr)?;
    Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `scipy.signal.windows.exponential(M, center=None, tau=1.0)`.
#[pyfunction]
#[pyo3(signature = (m, center = None, tau = 1.0))]
pub fn exponential<'py>(
    py: Python<'py>,
    m: usize,
    center: Option<f64>,
    tau: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let r: Array1<f64> = fw::exponential(m, center, tau).map_err(ferr_to_pyerr)?;
    Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `scipy.signal.windows.tukey(M, alpha=0.5)`.
#[pyfunction]
#[pyo3(signature = (m, alpha = 0.5))]
pub fn tukey<'py>(py: Python<'py>, m: usize, alpha: f64) -> PyResult<Bound<'py, PyAny>> {
    let r: Array1<f64> = fw::tukey(m, alpha).map_err(ferr_to_pyerr)?;
    Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `scipy.signal.windows.general_cosine(M, coeffs)`.
#[pyfunction]
pub fn general_cosine<'py>(
    py: Python<'py>,
    m: usize,
    coeffs: Vec<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let r: Array1<f64> = fw::general_cosine(m, &coeffs).map_err(ferr_to_pyerr)?;
    Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `scipy.signal.windows.general_hamming(M, alpha)`.
#[pyfunction]
pub fn general_hamming<'py>(py: Python<'py>, m: usize, alpha: f64) -> PyResult<Bound<'py, PyAny>> {
    let r: Array1<f64> = fw::general_hamming(m, alpha).map_err(ferr_to_pyerr)?;
    Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `scipy.signal.windows.taylor(M, nbar, sll, norm)`.
#[pyfunction]
#[pyo3(signature = (m, nbar = 4, sll = 30.0, norm = true))]
pub fn taylor<'py>(
    py: Python<'py>,
    m: usize,
    nbar: usize,
    sll: f64,
    norm: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let r: Array1<f64> = fw::taylor(m, nbar, sll, norm).map_err(ferr_to_pyerr)?;
    Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}
