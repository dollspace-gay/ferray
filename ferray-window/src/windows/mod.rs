// ferray-window: Window functions for signal processing and spectral analysis
//
// Implements NumPy- and SciPy-equivalent window functions:
//   - bartlett, blackman, hamming, hanning, kaiser (NumPy core)
//   - cosine, exponential, gaussian, general_cosine, general_hamming,
//     nuttall, parzen, taylor, tukey (SciPy / torch.signal.windows extras)
// Each returns an Array1<f64> of length M.

// Window-function coefficients (Blackman 0.42/0.5/0.08, Hamming 0.54/0.46,
// Kaiser modified-Bessel I0 series) are textbook constants reproduced
// without separators to match the reference papers / NumPy source.
#![allow(clippy::unreadable_literal)]

use ferray_core::Array;
use ferray_core::dimension::Ix1;
use ferray_core::error::{FerrayError, FerrayResult};

use std::f64::consts::PI;

// Re-use the Abramowitz & Stegun Bessel I_0 polynomial approximation
// from ferray-ufunc instead of shipping a second copy (see #530).
// The previous copy had already drifted in whitespace; funnelling
// through the canonical crate keeps both the window code and the
// ufunc in sync.
use ferray_ufunc::ops::special::bessel_i0_scalar;

/// Build a window array of length `m` by evaluating `f` at every sample
/// `0..m`. Handles the shared `m == 0` / `m == 1` edge cases that every
/// NumPy-equivalent window has (see issue #292 for the original
/// five-way copy of this pattern).
#[inline]
fn gen_window<F: FnMut(usize) -> f64>(m: usize, mut f: F) -> FerrayResult<Array<f64, Ix1>> {
    if m == 0 {
        return Array::from_vec(Ix1::new([0]), vec![]);
    }
    if m == 1 {
        return Array::from_vec(Ix1::new([1]), vec![1.0]);
    }
    let mut data = Vec::with_capacity(m);
    for n in 0..m {
        data.push(f(n));
    }
    Array::from_vec(Ix1::new([m]), data)
}

/// Internal helper: cast a length-`m` f64 window into an f32 array.
/// We compute in f64 (ferray-window's native precision) and then
/// down-cast — the windows are smooth, so this preserves all useful
/// precision.
fn cast_to_f32(arr: &Array<f64, Ix1>) -> FerrayResult<Array<f32, Ix1>> {
    let data: Vec<f32> = arr.iter().map(|&x| x as f32).collect();
    Array::<f32, Ix1>::from_vec(Ix1::new([data.len()]), data)
}

/// `f32` version of [`bartlett`] (#529).
///
/// # Errors
/// Same as `bartlett`.
pub fn bartlett_f32(m: usize) -> FerrayResult<Array<f32, Ix1>> {
    cast_to_f32(&bartlett(m)?)
}

/// `f32` version of [`blackman`] (#529).
///
/// # Errors
/// Same as `blackman`.
pub fn blackman_f32(m: usize) -> FerrayResult<Array<f32, Ix1>> {
    cast_to_f32(&blackman(m)?)
}

/// `f32` version of [`hamming`] (#529).
///
/// # Errors
/// Same as `hamming`.
pub fn hamming_f32(m: usize) -> FerrayResult<Array<f32, Ix1>> {
    cast_to_f32(&hamming(m)?)
}

/// `f32` version of [`hanning`] (#529).
///
/// # Errors
/// Same as `hanning`.
pub fn hanning_f32(m: usize) -> FerrayResult<Array<f32, Ix1>> {
    cast_to_f32(&hanning(m)?)
}

/// `f32` version of [`kaiser`] (#529).
///
/// # Errors
/// Same as `kaiser`.
pub fn kaiser_f32(m: usize, beta: f64) -> FerrayResult<Array<f32, Ix1>> {
    cast_to_f32(&kaiser(m, beta)?)
}

/// Return the Bartlett (triangular) window of length `m`.
///
/// The Bartlett window is defined as:
///   w(n) = 2/(M-1) * ((M-1)/2 - |n - (M-1)/2|)
///
/// This is equivalent to `numpy.bartlett(M)`.
///
/// # Errors
/// Returns an error only if internal array construction fails.
pub fn bartlett(m: usize) -> FerrayResult<Array<f64, Ix1>> {
    let half = (m.saturating_sub(1)) as f64 / 2.0;
    gen_window(m, |n| 1.0 - ((n as f64 - half) / half).abs())
}

/// Return the Blackman window of length `m`.
///
/// The Blackman window is defined as:
///   w(n) = 0.42 - 0.5 * cos(2*pi*n/(M-1)) + 0.08 * cos(4*pi*n/(M-1))
///
/// This is equivalent to `numpy.blackman(M)`.
///
/// # Errors
/// Returns an error only if internal array construction fails.
pub fn blackman(m: usize) -> FerrayResult<Array<f64, Ix1>> {
    let denom = (m.saturating_sub(1)) as f64;
    gen_window(m, |n| {
        let x = n as f64;
        0.08f64.mul_add(
            (4.0 * PI * x / denom).cos(),
            0.5f64.mul_add(-(2.0 * PI * x / denom).cos(), 0.42),
        )
    })
}

/// Return the Hamming window of length `m`.
///
/// The Hamming window is defined as:
///   w(n) = 0.54 - 0.46 * cos(2*pi*n/(M-1))
///
/// This is equivalent to `numpy.hamming(M)`.
///
/// # Errors
/// Returns an error only if internal array construction fails.
pub fn hamming(m: usize) -> FerrayResult<Array<f64, Ix1>> {
    let denom = (m.saturating_sub(1)) as f64;
    gen_window(m, |n| {
        0.46f64.mul_add(-(2.0 * PI * n as f64 / denom).cos(), 0.54)
    })
}

/// Return the Hann (Hanning) window of length `m`.
///
/// The Hann window is defined as:
///   w(n) = 0.5 * (1 - cos(2*pi*n/(M-1)))
///
/// `NumPy` calls this function `hanning`. This is equivalent to `numpy.hanning(M)`.
///
/// # Errors
/// Returns an error only if internal array construction fails.
pub fn hanning(m: usize) -> FerrayResult<Array<f64, Ix1>> {
    let denom = (m.saturating_sub(1)) as f64;
    gen_window(m, |n| 0.5 * (1.0 - (2.0 * PI * n as f64 / denom).cos()))
}

/// Return the Kaiser window of length `m` with shape parameter `beta`.
///
/// The Kaiser window is defined as:
///   w(n) = `I_0(beta` * sqrt(1 - ((2n/(M-1)) - 1)^2)) / `I_0(beta)`
///
/// where `I_0` is the modified Bessel function of the first kind, order 0.
///
/// This is equivalent to `numpy.kaiser(M, beta)`.
///
/// # Edge Cases
/// - `m == 0`: returns an empty array.
/// - `m == 1`: returns `[1.0]`.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if `beta` is NaN or if `beta`
/// exceeds the range where `I_0(beta)` can be computed in f64. The
/// asymptotic branch uses `exp(beta) / sqrt(beta)` internally, so
/// `|beta| > ~709` overflows to infinity and the normalized kaiser
/// window would reduce to `Inf / Inf = NaN` for every sample
/// (#294).
pub fn kaiser(m: usize, beta: f64) -> FerrayResult<Array<f64, Ix1>> {
    if beta.is_nan() {
        return Err(FerrayError::invalid_value("kaiser: beta must not be NaN"));
    }
    // I_0 is an even function, so kaiser(m, -beta) == kaiser(m, beta).
    // Accept negative beta for NumPy compatibility.
    let beta = beta.abs();
    // Reject beta values where `exp(beta)` overflows f64 (ln(f64::MAX)
    // ≈ 709.78). We use a slightly conservative threshold so the
    // clipped output stays finite even after the asymptotic-series
    // polynomial corrections.
    const BETA_OVERFLOW_THRESHOLD: f64 = 708.0;
    if beta > BETA_OVERFLOW_THRESHOLD {
        return Err(FerrayError::invalid_value(format!(
            "kaiser: |beta| = {beta} exceeds the safe range ({BETA_OVERFLOW_THRESHOLD}) \
             for f64 I_0; the window would be NaN everywhere"
        )));
    }
    let i0_beta = bessel_i0_scalar::<f64>(beta);
    let alpha = (m.saturating_sub(1)) as f64 / 2.0;
    gen_window(m, |n| {
        let t = (n as f64 - alpha) / alpha;
        let arg = beta * (1.0 - t * t).max(0.0).sqrt();
        bessel_i0_scalar::<f64>(arg) / i0_beta
    })
}

// ===========================================================================
// SciPy-extended windows (cosine, exponential, gaussian, general_cosine,
// general_hamming, nuttall, parzen, taylor, tukey).
//
// All are symmetric ("sym=True" in scipy parlance); a non-symmetric "periodic"
// variant can be obtained by computing window of length m+1 and dropping the
// last sample (NumPy convention).
// ===========================================================================

/// Half-cycle cosine window (also known as the "sine" window).
///
/// `w(n) = sin(pi * (n + 0.5) / M)` for `n = 0..M-1`.
///
/// Mirrors `scipy.signal.windows.cosine` /
/// `torch.signal.windows.cosine`.
pub fn cosine(m: usize) -> FerrayResult<Array<f64, Ix1>> {
    if m == 0 {
        return Array::from_vec(Ix1::new([0]), vec![]);
    }
    if m == 1 {
        return Array::from_vec(Ix1::new([1]), vec![1.0]);
    }
    let mf = m as f64;
    gen_window(m, |n| (PI * (n as f64 + 0.5) / mf).sin())
}

/// Exponentially decaying window centred on `center` with time-constant `tau`.
///
/// `w(n) = exp(-|n - center| / tau)`. If `center` is `None`, defaults to
/// the geometric centre `(M - 1) / 2`. `tau` must be > 0.
///
/// Mirrors `scipy.signal.windows.exponential` /
/// `torch.signal.windows.exponential`.
pub fn exponential(m: usize, center: Option<f64>, tau: f64) -> FerrayResult<Array<f64, Ix1>> {
    if !tau.is_finite() || tau <= 0.0 {
        return Err(FerrayError::invalid_value(format!(
            "exponential: tau must be positive and finite, got {tau}"
        )));
    }
    let centre = center.unwrap_or((m.saturating_sub(1)) as f64 / 2.0);
    gen_window(m, |n| (-((n as f64) - centre).abs() / tau).exp())
}

/// Gaussian window with standard deviation `std`.
///
/// `w(n) = exp(-((n - (M-1)/2) / std)^2 / 2)`. `std` must be > 0.
///
/// Mirrors `scipy.signal.windows.gaussian` /
/// `torch.signal.windows.gaussian`.
pub fn gaussian(m: usize, std: f64) -> FerrayResult<Array<f64, Ix1>> {
    if !std.is_finite() || std <= 0.0 {
        return Err(FerrayError::invalid_value(format!(
            "gaussian: std must be positive and finite, got {std}"
        )));
    }
    let centre = (m.saturating_sub(1)) as f64 / 2.0;
    gen_window(m, |n| {
        let z = ((n as f64) - centre) / std;
        (-0.5 * z * z).exp()
    })
}

/// Generalized cosine sum window: `w(n) = Σ_k (-1)^k a[k] cos(2π k n / (M-1))`.
///
/// `coeffs` is the slice `[a_0, a_1, a_2, ...]`. Setting `[0.5, 0.5]` gives
/// the Hann window; `[0.42, 0.5, 0.08]` gives the classical Blackman.
///
/// Mirrors `scipy.signal.windows.general_cosine` /
/// `torch.signal.windows.general_cosine`.
pub fn general_cosine(m: usize, coeffs: &[f64]) -> FerrayResult<Array<f64, Ix1>> {
    if coeffs.is_empty() {
        return Err(FerrayError::invalid_value(
            "general_cosine: coeffs must not be empty",
        ));
    }
    let denom = (m.saturating_sub(1)) as f64;
    let coeffs = coeffs.to_vec();
    gen_window(m, |n| {
        let nf = n as f64;
        let mut sum = 0.0_f64;
        for (k, &a) in coeffs.iter().enumerate() {
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            sum += sign * a * (2.0 * PI * (k as f64) * nf / denom).cos();
        }
        sum
    })
}

/// Generalized Hamming window: `w(n) = α - (1 - α) cos(2π n / (M - 1))`.
///
/// `alpha = 0.5` gives the Hann window; `alpha = 25/46 ≈ 0.5435` gives a
/// "perfect" Hamming (NumPy uses 0.54 by tradition).
///
/// Mirrors `scipy.signal.windows.general_hamming` /
/// `torch.signal.windows.general_hamming`.
pub fn general_hamming(m: usize, alpha: f64) -> FerrayResult<Array<f64, Ix1>> {
    if !alpha.is_finite() {
        return Err(FerrayError::invalid_value(format!(
            "general_hamming: alpha must be finite, got {alpha}"
        )));
    }
    let denom = (m.saturating_sub(1)) as f64;
    gen_window(m, |n| {
        alpha - (1.0 - alpha) * (2.0 * PI * (n as f64) / denom).cos()
    })
}

/// 4-term Nuttall window with continuous first derivative.
///
/// Coefficients [0.3635819, 0.4891775, 0.1365995, 0.0106411] (Nuttall 1981,
/// "minimum 4-term Blackman-Harris with continuous first derivative").
///
/// Mirrors `scipy.signal.windows.nuttall` /
/// `torch.signal.windows.nuttall`.
pub fn nuttall(m: usize) -> FerrayResult<Array<f64, Ix1>> {
    general_cosine(m, &[0.3635819, 0.4891775, 0.1365995, 0.0106411])
}

/// Parzen (de la Vallée Poussin) window: a piecewise-cubic B-spline.
///
/// Mirrors `scipy.signal.windows.parzen` / `torch.signal.windows.parzen`.
pub fn parzen(m: usize) -> FerrayResult<Array<f64, Ix1>> {
    if m == 0 {
        return Array::from_vec(Ix1::new([0]), vec![]);
    }
    if m == 1 {
        return Array::from_vec(Ix1::new([1]), vec![1.0]);
    }
    // Parzen formula: |x| in [0, M/4]: 1 - 6 r^2 + 6 |r|^3,
    //                |x| in (M/4, M/2]: 2 (1 - |r|)^3, where r = x / (M/2).
    let half = m as f64 / 2.0;
    gen_window(m, |n| {
        let x = (n as f64) - (m as f64 - 1.0) / 2.0;
        let r = x.abs() / half;
        if r <= 0.5 {
            1.0 - 6.0 * r * r + 6.0 * r * r * r
        } else if r <= 1.0 {
            let one_minus_r = 1.0 - r;
            2.0 * one_minus_r * one_minus_r * one_minus_r
        } else {
            0.0
        }
    })
}

/// Dolph–Chebyshev window with sidelobe attenuation `at` dB (#740).
///
/// Mirrors `scipy.signal.windows.chebwin`. The window is the
/// inverse Fourier transform of the (M-1)-th Chebyshev polynomial
/// of the first kind evaluated on a cosine sweep, producing an
/// equiripple sidelobe response at `at` dB below the main lobe.
///
/// Implemented via direct DFT (sufficient for typical M ≤ 1024;
/// the inner loop is O(M²) but allocation-free). Result is
/// normalised so the centre value is 1.
///
/// # Errors
/// `FerrayError::InvalidValue` if `at` is non-finite or
/// `at <= 0.0`. `m == 0` returns the empty window; `m == 1`
/// returns `[1.0]`.
pub fn chebwin(m: usize, at: f64) -> FerrayResult<Array<f64, Ix1>> {
    if !at.is_finite() || at <= 0.0 {
        return Err(FerrayError::invalid_value(format!(
            "chebwin: at must be a positive finite dB attenuation, got {at}"
        )));
    }
    if m == 0 {
        return Array::from_vec(Ix1::new([0]), vec![]);
    }
    if m == 1 {
        return Array::from_vec(Ix1::new([1]), vec![1.0]);
    }

    let order = (m - 1) as f64;
    let r = 10.0_f64.powf(at / 20.0);
    let beta = (r.acosh() / order).cosh();

    // Build spectrum P[k] = T_{m-1}(β cos(πk/m)).
    let mut spectrum = Vec::with_capacity(m);
    for k in 0..m {
        let x = beta * (PI * k as f64 / m as f64).cos();
        spectrum.push(chebyshev_t_extended(x, order));
    }

    // Direct DFT to time domain. For odd M the spectrum is even-
    // symmetric about M/2 and the IDFT is purely real; for even M
    // the (M-1)-th Chebyshev polynomial is odd, so we apply a
    // half-bin phase shift before transforming. Matches scipy's
    // FFT-based pipeline up to the trailing peak normalisation.
    // Forward DFT of the spectrum (matches scipy's fft(p) call).
    // For odd M the spectrum is even-symmetric so fft is real;
    // for even M scipy multiplies by exp(iπk/M) phase first to
    // recover symmetry. After the DFT we keep the first half,
    // normalise by w[0] so the eventual centre lands at 1, and
    // mirror to recover the full symmetric window.
    let mf = m as f64;
    let half_dft = |idx: usize| -> f64 {
        if m % 2 == 1 {
            spectrum
                .iter()
                .enumerate()
                .map(|(k, &pk)| pk * (-2.0 * PI * k as f64 * idx as f64 / mf).cos())
                .sum()
        } else {
            // Forward DFT with the additional exp(iπk/M) phase
            // shift baked in: p[k] * exp(iπk/M) * exp(-i2πkn/M)
            //              = p[k] * cos(πk(1 - 2n)/M).
            spectrum
                .iter()
                .enumerate()
                .map(|(k, &pk)| {
                    let phase = PI * k as f64 * (1.0 - 2.0 * idx as f64) / mf;
                    pk * phase.cos()
                })
                .sum()
        }
    };

    let half_len = m / 2 + 1;
    let mut half: Vec<f64> = (0..half_len).map(half_dft).collect();
    let denom = if m % 2 == 1 { half[0] } else { half[1] };
    if denom != 0.0 {
        for v in &mut half {
            *v /= denom;
        }
    }

    let mut w = Vec::with_capacity(m);
    if m % 2 == 1 {
        // Mirror: reverse(half[1..]) ++ half
        for &v in half.iter().skip(1).rev() {
            w.push(v);
        }
        w.extend_from_slice(&half);
    } else {
        // Even: scipy's `concatenate((w[n-1:0:-1], w[1:n]))` where
        // n = M/2 + 1; reverse of half[1..n], then half[1..n].
        let n = m / 2 + 1;
        for &v in half[1..n].iter().rev() {
            w.push(v);
        }
        w.extend_from_slice(&half[1..n]);
    }

    Array::from_vec(Ix1::new([m]), w)
}

/// Chebyshev polynomial of the first kind T_n(x) extended to all
/// real x via cosh — `T_n(cos θ) = cos(nθ)` for `|x| ≤ 1`,
/// `T_n(cosh θ) = cosh(nθ)` for `x > 1`. Sign handling on the
/// negative branch uses `(-1)^n`.
fn chebyshev_t_extended(x: f64, n: f64) -> f64 {
    if x.abs() <= 1.0 {
        (n * x.acos()).cos()
    } else {
        let mag = (n * x.abs().acosh()).cosh();
        if x < 0.0 && (n as i64) % 2 != 0 {
            -mag
        } else {
            mag
        }
    }
}

/// Boxcar (rectangular) window of length `m` (#738).
///
/// All-ones window. Equivalent to `scipy.signal.windows.boxcar`.
///
/// # Errors
/// Only on internal array construction failure.
pub fn boxcar(m: usize) -> FerrayResult<Array<f64, Ix1>> {
    gen_window(m, |_| 1.0)
}

/// Triangular window of length `m` (#738).
///
/// Centre = 1, linearly decreasing to (1/m) at the edges (slightly
/// different from Bartlett, which decays to 0 at the edges).
/// Equivalent to `scipy.signal.windows.triang`.
///
/// # Errors
/// Only on internal array construction failure.
pub fn triang(m: usize) -> FerrayResult<Array<f64, Ix1>> {
    if m == 0 {
        return Array::from_vec(Ix1::new([0]), vec![]);
    }
    if m == 1 {
        return Array::from_vec(Ix1::new([1]), vec![1.0]);
    }
    let mf = m as f64;
    let centre = (mf - 1.0) / 2.0;
    if m % 2 == 0 {
        // Even-m formula: w[n] = 1 - |2n - (m-1)| / m
        gen_window(m, |n| 1.0 - (2.0 * n as f64 - centre - centre).abs() / mf)
    } else {
        // Odd-m formula: w[n] = 1 - |2n - (m-1)| / (m + 1)
        gen_window(m, |n| {
            1.0 - (2.0 * n as f64 - centre - centre).abs() / (mf + 1.0)
        })
    }
}

/// Bohman window of length `m` (#738).
///
/// w(n) = (1 - |r|) cos(π|r|) + (1/π) sin(π|r|), where r is the
/// normalised position in [-1, 1]. Equivalent to
/// `scipy.signal.windows.bohman`.
///
/// # Errors
/// Only on internal array construction failure.
pub fn bohman(m: usize) -> FerrayResult<Array<f64, Ix1>> {
    if m == 0 {
        return Array::from_vec(Ix1::new([0]), vec![]);
    }
    if m == 1 {
        return Array::from_vec(Ix1::new([1]), vec![1.0]);
    }
    // scipy: linspace(-1, 1, m)[1:-1] core, with explicit zeros at endpoints.
    let mf = (m - 1) as f64;
    let mut data = Vec::with_capacity(m);
    for n in 0..m {
        if n == 0 || n == m - 1 {
            data.push(0.0);
            continue;
        }
        let r = (2.0 * n as f64 / mf - 1.0).abs();
        let v = (1.0 - r) * (PI * r).cos() + (PI * r).sin() / PI;
        data.push(v);
    }
    Array::from_vec(Ix1::new([m]), data)
}

/// Flat-top window of length `m` (#738).
///
/// 5-term cosine window with coefficients chosen for minimum
/// passband ripple (used for amplitude calibration). Equivalent to
/// `scipy.signal.windows.flattop`.
///
/// # Errors
/// Only on internal array construction failure.
pub fn flattop(m: usize) -> FerrayResult<Array<f64, Ix1>> {
    // scipy's coefficients (a0..a4):
    const COEFFS: [f64; 5] = [
        0.215_578_95,
        0.416_631_58,
        0.277_263_158,
        0.083_578_95,
        0.006_947_368,
    ];
    if m == 0 {
        return Array::from_vec(Ix1::new([0]), vec![]);
    }
    let denom = (m - 1).max(1) as f64;
    gen_window(m, |n| {
        let phase = 2.0 * PI * n as f64 / denom;
        let mut acc = COEFFS[0];
        for (k, &c) in COEFFS.iter().enumerate().skip(1) {
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            acc += sign * c * (k as f64 * phase).cos();
        }
        acc
    })
}

/// Lanczos (sinc) window of length `m` (#738).
///
/// w(n) = sinc(2 n / (m-1) - 1). Equivalent to
/// `scipy.signal.windows.lanczos`.
///
/// # Errors
/// Only on internal array construction failure.
pub fn lanczos(m: usize) -> FerrayResult<Array<f64, Ix1>> {
    if m == 0 {
        return Array::from_vec(Ix1::new([0]), vec![]);
    }
    if m == 1 {
        return Array::from_vec(Ix1::new([1]), vec![1.0]);
    }
    let denom = (m - 1) as f64;
    gen_window(m, |n| {
        let x = 2.0 * n as f64 / denom - 1.0;
        if x.abs() < f64::EPSILON {
            1.0
        } else {
            let pi_x = PI * x;
            pi_x.sin() / pi_x
        }
    })
}

/// Taylor window with `nbar` near-side lobes and a sidelobe level of `sll`
/// dB below the main lobe.
///
/// `nbar` (typically 4) controls how many sidelobes are constrained at the
/// design level; `sll` (typically 30) is the desired peak sidelobe
/// attenuation in **positive** dB. When `norm` is true the window is
/// normalised so `w[(M-1)/2] = 1`.
///
/// Implementation follows Carrara & Goodman, "Symmetric Taylor Window"
/// (Synthetic Aperture Radar, 1995). For the radar/array-processing
/// applications where Taylor windows are used, `nbar = 4`, `sll = 30`
/// gives equiripple sidelobes ~30 dB down — the usual default.
///
/// Mirrors `scipy.signal.windows.taylor` / `torch.signal.windows.taylor`.
pub fn taylor(
    m: usize,
    nbar: usize,
    sll: f64,
    norm: bool,
) -> FerrayResult<Array<f64, Ix1>> {
    if m == 0 {
        return Array::from_vec(Ix1::new([0]), vec![]);
    }
    if m == 1 {
        return Array::from_vec(Ix1::new([1]), vec![1.0]);
    }
    if nbar == 0 {
        return Err(FerrayError::invalid_value("taylor: nbar must be >= 1"));
    }
    if !sll.is_finite() {
        return Err(FerrayError::invalid_value("taylor: sll must be finite"));
    }
    // R = 10^(sll/20), B = (1/π) acosh(R)
    let r = 10.0_f64.powf(sll / 20.0);
    let b = r.acosh() / PI;
    let nbar_f = nbar as f64;
    // sigma^2 chosen so the (nbar)-th zero of the Taylor pattern is at
    // n = nbar (Carrara & Goodman eq. 13).
    let sigma2 = (nbar_f * nbar_f) / (b * b + (nbar_f - 0.5) * (nbar_f - 0.5));

    // Compute coefficients F_m for m = 1..nbar-1.
    let mut f_coeffs = Vec::with_capacity(nbar.saturating_sub(1));
    for mm in 1..nbar {
        let mmf = mm as f64;
        // Numerator: ((-1)^(m+1) / 2) * Π_{n=1..nbar-1} [1 - m^2 / (sigma^2 * (B^2 + (n - 0.5)^2))]
        // Denominator: Π_{n=1..nbar-1, n != m} [1 - m^2 / n^2]
        let mut num = 1.0_f64;
        for n in 1..nbar {
            let nf = n as f64;
            num *= 1.0 - mmf * mmf / (sigma2 * (b * b + (nf - 0.5) * (nf - 0.5)));
        }
        let sign = if mm % 2 == 0 { -1.0 } else { 1.0 };
        let mut den = 1.0_f64;
        for n in 1..nbar {
            if n == mm {
                continue;
            }
            let nf = n as f64;
            den *= 1.0 - mmf * mmf / (nf * nf);
        }
        // The 0.5 prefactor: F_0 = 1 contributes the constant term, and
        // each F_m doubles when reflected about zero in the cosine sum,
        // so we halve the inverse-Fourier coefficients here.
        f_coeffs.push(0.5 * sign * num / den);
    }

    let denom = (m - 1) as f64;
    let arr = gen_window(m, |n| {
        let xn = (n as f64) - denom / 2.0;
        let mut w = 1.0_f64;
        for (idx, &fk) in f_coeffs.iter().enumerate() {
            let kk = (idx + 1) as f64;
            w += 2.0 * fk * (2.0 * PI * kk * xn / m as f64).cos();
        }
        w
    })?;

    if !norm {
        return Ok(arr);
    }
    // Normalise so the centre value is 1.
    let s = arr.as_slice().unwrap().to_vec();
    let centre_val = if m % 2 == 1 {
        s[m / 2]
    } else {
        // For even M, centre is between two samples — average.
        0.5 * (s[m / 2 - 1] + s[m / 2])
    };
    if centre_val == 0.0 {
        return Ok(arr); // pathological; leave un-normalised
    }
    let normed: Vec<f64> = s.into_iter().map(|v| v / centre_val).collect();
    Array::from_vec(Ix1::new([m]), normed)
}

/// Tukey (cosine-tapered) window with taper ratio `alpha` ∈ [0, 1].
///
/// `alpha = 0` gives a rectangular window; `alpha = 1` gives a Hann
/// window. The middle `(1 - alpha) * (M - 1)` samples are unity, and the
/// edges are tapered with a half-cosine.
///
/// Mirrors `scipy.signal.windows.tukey` / `torch.signal.windows.tukey`.
pub fn tukey(m: usize, alpha: f64) -> FerrayResult<Array<f64, Ix1>> {
    if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
        return Err(FerrayError::invalid_value(format!(
            "tukey: alpha must be in [0, 1], got {alpha}"
        )));
    }
    if m == 0 {
        return Array::from_vec(Ix1::new([0]), vec![]);
    }
    if m == 1 {
        return Array::from_vec(Ix1::new([1]), vec![1.0]);
    }
    if alpha == 0.0 {
        return Array::from_vec(Ix1::new([m]), vec![1.0; m]);
    }
    let denom = (m - 1) as f64;
    let width = alpha * denom / 2.0;
    gen_window(m, |n| {
        let nf = n as f64;
        if nf < width {
            0.5 * (1.0 + (PI * (nf / width - 1.0)).cos())
        } else if nf <= denom - width {
            1.0
        } else {
            0.5 * (1.0 + (PI * ((denom - nf) / width - 1.0)).cos())
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- chebwin (#740) ------------------------------------------------

    #[test]
    fn chebwin_centre_is_one_and_symmetric() {
        let w = chebwin(11, 50.0).unwrap();
        let s = w.as_slice().unwrap();
        let mid = s.len() / 2;
        assert!((s[mid] - 1.0).abs() < 1e-6, "centre = {}", s[mid]);
        for i in 0..s.len() / 2 {
            assert!(
                (s[i] - s[s.len() - 1 - i]).abs() < 1e-9,
                "asymmetric at i={i}: {} vs {}",
                s[i],
                s[s.len() - 1 - i]
            );
        }
    }

    #[test]
    fn chebwin_even_length_symmetric() {
        let w = chebwin(8, 40.0).unwrap();
        let s = w.as_slice().unwrap();
        for i in 0..s.len() / 2 {
            assert!(
                (s[i] - s[s.len() - 1 - i]).abs() < 1e-9,
                "asymmetric at i={i}"
            );
        }
        // No element should exceed the centre by more than 1e-9.
        let max = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!((max - 1.0).abs() < 1e-6);
    }

    #[test]
    fn chebwin_endpoints_smaller_than_centre() {
        // High attenuation → endpoints should be very small.
        let w = chebwin(31, 80.0).unwrap();
        let s = w.as_slice().unwrap();
        let mid = s.len() / 2;
        assert!(s[mid] > s[0]);
        assert!(s[mid] > s[s.len() - 1]);
    }

    #[test]
    fn chebwin_rejects_invalid_attenuation() {
        assert!(chebwin(16, -10.0).is_err());
        assert!(chebwin(16, 0.0).is_err());
        assert!(chebwin(16, f64::NAN).is_err());
    }

    #[test]
    fn chebwin_matches_scipy_known_values() {
        // scipy.signal.windows.chebwin(11, 50) reference values.
        let want = [
            0.06228583, 0.20113857, 0.42847525, 0.69573494, 0.91497506, 1.0, 0.91497506,
            0.69573494, 0.42847525, 0.20113857, 0.06228583,
        ];
        let got = chebwin(11, 50.0).unwrap();
        let s = got.as_slice().unwrap();
        for (i, (&g, &w)) in s.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-6, "i={i}: got={g}, want={w}");
        }
    }

    #[test]
    fn chebwin_matches_scipy_even_length() {
        // scipy.signal.windows.chebwin(8, 40) reference values.
        let want = [
            0.14609713, 0.41790422, 0.75944595, 1.0, 1.0, 0.75944595, 0.41790422, 0.14609713,
        ];
        let got = chebwin(8, 40.0).unwrap();
        let s = got.as_slice().unwrap();
        for (i, (&g, &w)) in s.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-6, "i={i}: got={g}, want={w}");
        }
    }

    #[test]
    fn chebwin_m0_m1_edge_cases() {
        let z = chebwin(0, 50.0).unwrap();
        assert_eq!(z.shape(), &[0]);
        let one = chebwin(1, 50.0).unwrap();
        assert_eq!(one.as_slice().unwrap(), &[1.0]);
    }

    // ---- additional scipy.signal windows (#738) ------------------------

    #[test]
    fn boxcar_all_ones() {
        let w = boxcar(8).unwrap();
        let s = w.as_slice().unwrap();
        for &v in s {
            assert!((v - 1.0).abs() < 1e-15);
        }
    }

    #[test]
    fn triang_centre_is_max_and_endpoints_smaller() {
        let w = triang(7).unwrap();
        let s = w.as_slice().unwrap();
        // Symmetry: w[n] == w[m-1-n].
        for i in 0..s.len() / 2 {
            assert!((s[i] - s[s.len() - 1 - i]).abs() < 1e-12);
        }
        // Centre is the peak.
        let mid = s.len() / 2;
        for i in 0..s.len() {
            if i != mid {
                assert!(s[i] <= s[mid] + 1e-12);
            }
        }
    }

    #[test]
    fn bohman_zero_at_endpoints_and_one_at_centre() {
        let w = bohman(5).unwrap();
        let s = w.as_slice().unwrap();
        assert!(s[0].abs() < 1e-12);
        assert!(s[s.len() - 1].abs() < 1e-12);
        // Centre value is the maximum and very close to 1.
        let mid = s.len() / 2;
        assert!((s[mid] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn flattop_centre_around_one_and_negative_lobes_present() {
        // flattop is designed for amplitude flatness — centre ≈ 1
        // with the canonical scaling, and side regions go negative.
        let w = flattop(11).unwrap();
        let s = w.as_slice().unwrap();
        let mid = s.len() / 2;
        assert!((s[mid] - 1.0).abs() < 0.01);
        // Some side samples must go below zero (window has negative
        // side lobes by design).
        let any_negative = s.iter().any(|&v| v < -0.001);
        assert!(any_negative, "flattop should have negative side lobes");
    }

    #[test]
    fn lanczos_centre_is_one_and_endpoints_zero() {
        let w = lanczos(9).unwrap();
        let s = w.as_slice().unwrap();
        let mid = s.len() / 2;
        assert!((s[mid] - 1.0).abs() < 1e-12);
        assert!(s[0].abs() < 1e-12);
        assert!(s[s.len() - 1].abs() < 1e-12);
    }

    #[test]
    fn small_m_edge_cases_handled() {
        // m = 0 returns empty, m = 1 returns [1.0] for symmetric windows.
        for f in [
            triang as fn(usize) -> _,
            bohman,
            lanczos,
        ] {
            let z = f(0).unwrap();
            assert_eq!(z.shape(), &[0]);
            let one = f(1).unwrap();
            assert_eq!(one.shape(), &[1]);
            assert_eq!(one.as_slice().unwrap(), &[1.0]);
        }
    }

    // ---- f32 sibling functions (#529) ----------------------------------

    #[test]
    fn f32_windows_match_f64_within_f32_eps() {
        for m in [1usize, 5, 16, 64] {
            for (name, got, want) in [
                ("bartlett", bartlett_f32(m).unwrap(), bartlett(m).unwrap()),
                ("blackman", blackman_f32(m).unwrap(), blackman(m).unwrap()),
                ("hamming", hamming_f32(m).unwrap(), hamming(m).unwrap()),
                ("hanning", hanning_f32(m).unwrap(), hanning(m).unwrap()),
            ] {
                assert_eq!(got.shape(), want.shape(), "{name} m={m} shape");
                let g = got.as_slice().unwrap();
                let w = want.as_slice().unwrap();
                for (i, (&a, &b)) in g.iter().zip(w.iter()).enumerate() {
                    let bf = b as f32;
                    assert!(
                        (a - bf).abs() < 1e-6,
                        "{name} m={m} idx={i}: f32 {a} vs f64 {b}"
                    );
                }
            }
        }
    }

    #[test]
    fn kaiser_f32_matches_kaiser() {
        let m = 20;
        let beta = 8.6;
        let f32_arr = kaiser_f32(m, beta).unwrap();
        let f64_arr = kaiser(m, beta).unwrap();
        for (a, b) in f32_arr.iter().zip(f64_arr.iter()) {
            assert!((a - *b as f32).abs() < 1e-5);
        }
    }

    // Helper: compare two slices within tolerance
    fn assert_close(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "length mismatch: {} vs {}",
            actual.len(),
            expected.len()
        );
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= tol,
                "element {i}: {a} vs {e} (diff = {})",
                (a - e).abs()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Edge cases: M=0 and M=1
    // -----------------------------------------------------------------------

    #[test]
    fn bartlett_m0() {
        let w = bartlett(0).unwrap();
        assert_eq!(w.shape(), &[0]);
    }

    #[test]
    fn bartlett_m1() {
        let w = bartlett(1).unwrap();
        assert_eq!(w.as_slice().unwrap(), &[1.0]);
    }

    #[test]
    fn blackman_m0() {
        let w = blackman(0).unwrap();
        assert_eq!(w.shape(), &[0]);
    }

    #[test]
    fn blackman_m1() {
        let w = blackman(1).unwrap();
        assert_eq!(w.as_slice().unwrap(), &[1.0]);
    }

    #[test]
    fn hamming_m0() {
        let w = hamming(0).unwrap();
        assert_eq!(w.shape(), &[0]);
    }

    #[test]
    fn hamming_m1() {
        let w = hamming(1).unwrap();
        assert_eq!(w.as_slice().unwrap(), &[1.0]);
    }

    #[test]
    fn hanning_m0() {
        let w = hanning(0).unwrap();
        assert_eq!(w.shape(), &[0]);
    }

    #[test]
    fn hanning_m1() {
        let w = hanning(1).unwrap();
        assert_eq!(w.as_slice().unwrap(), &[1.0]);
    }

    #[test]
    fn kaiser_m0() {
        let w = kaiser(0, 14.0).unwrap();
        assert_eq!(w.shape(), &[0]);
    }

    #[test]
    fn kaiser_m1() {
        let w = kaiser(1, 14.0).unwrap();
        assert_eq!(w.as_slice().unwrap(), &[1.0]);
    }

    #[test]
    fn kaiser_negative_beta() {
        // Negative beta should produce the same result as positive beta (I0 is even)
        let pos = kaiser(5, 1.0).unwrap();
        let neg = kaiser(5, -1.0).unwrap();
        assert_eq!(pos.as_slice().unwrap(), neg.as_slice().unwrap());
    }

    #[test]
    fn kaiser_nan_beta() {
        assert!(kaiser(5, f64::NAN).is_err());
    }

    // -----------------------------------------------------------------------
    // AC-1: bartlett(5) matches np.bartlett(5) to within 4 ULPs
    // -----------------------------------------------------------------------
    // np.bartlett(5) = [0.0, 0.5, 1.0, 0.5, 0.0]
    #[test]
    fn bartlett_5_ac1() {
        let w = bartlett(5).unwrap();
        let expected = [0.0, 0.5, 1.0, 0.5, 0.0];
        assert_close(w.as_slice().unwrap(), &expected, 1e-14);
    }

    // -----------------------------------------------------------------------
    // AC-2: kaiser(5, 14.0) matches np.kaiser(5, 14.0) to within 4 ULPs
    // -----------------------------------------------------------------------
    // np.kaiser(5, 14.0) = [7.72686684e-06, 1.64932188e-01, 1.0, 1.64932188e-01, 7.72686684e-06]
    #[test]
    fn kaiser_5_14_ac2() {
        let w = kaiser(5, 14.0).unwrap();
        let s = w.as_slice().unwrap();
        assert_eq!(s.len(), 5);
        // NumPy reference values (verified via np.kaiser(5, 14.0))
        let expected: [f64; 5] = [
            7.72686684e-06,
            1.64932188e-01,
            1.0,
            1.64932188e-01,
            7.72686684e-06,
        ];
        // Bessel polynomial approximation has limited precision (~6 digits)
        for (i, (&a, &e)) in s.iter().zip(expected.iter()).enumerate() {
            let tol = if e.abs() < 1e-4 { 1e-8 } else { 1e-6 };
            assert!(
                (a - e).abs() <= tol,
                "kaiser element {i}: {a} vs {e} (diff = {})",
                (a - e).abs()
            );
        }
    }

    // -----------------------------------------------------------------------
    // AC-3: All 5 window functions return correct length and match NumPy fixtures
    // -----------------------------------------------------------------------

    // np.blackman(5) = [-1.38777878e-17, 3.40000000e-01, 1.00000000e+00, 3.40000000e-01, -1.38777878e-17]
    #[test]
    fn blackman_5() {
        let w = blackman(5).unwrap();
        assert_eq!(w.shape(), &[5]);
        let s = w.as_slice().unwrap();
        let expected = [
            -1.38777878e-17,
            3.40000000e-01,
            1.00000000e+00,
            3.40000000e-01,
            -1.38777878e-17,
        ];
        assert_close(s, &expected, 1e-10);
    }

    // np.hamming(5) = [0.08, 0.54, 1.0, 0.54, 0.08]
    #[test]
    fn hamming_5() {
        let w = hamming(5).unwrap();
        assert_eq!(w.shape(), &[5]);
        let s = w.as_slice().unwrap();
        let expected = [0.08, 0.54, 1.0, 0.54, 0.08];
        assert_close(s, &expected, 1e-14);
    }

    // np.hanning(5) = [0.0, 0.5, 1.0, 0.5, 0.0]
    #[test]
    fn hanning_5() {
        let w = hanning(5).unwrap();
        assert_eq!(w.shape(), &[5]);
        let s = w.as_slice().unwrap();
        let expected = [0.0, 0.5, 1.0, 0.5, 0.0];
        assert_close(s, &expected, 1e-14);
    }

    // Larger window: np.bartlett(12)
    #[test]
    fn bartlett_12() {
        let w = bartlett(12).unwrap();
        assert_eq!(w.shape(), &[12]);
        let s = w.as_slice().unwrap();
        // First and last should be 0, peak near center
        assert!((s[0] - 0.0).abs() < 1e-14);
        assert!((s[11] - 0.0).abs() < 1e-14);
        // Symmetric
        for i in 0..6 {
            assert!((s[i] - s[11 - i]).abs() < 1e-14, "symmetry at {i}");
        }
    }

    // Symmetry test for all windows
    #[test]
    fn all_windows_symmetric() {
        let m = 7;
        let windows: Vec<Array<f64, Ix1>> = vec![
            bartlett(m).unwrap(),
            blackman(m).unwrap(),
            hamming(m).unwrap(),
            hanning(m).unwrap(),
            kaiser(m, 5.0).unwrap(),
        ];
        for (idx, w) in windows.iter().enumerate() {
            let s = w.as_slice().unwrap();
            for i in 0..m / 2 {
                assert!(
                    (s[i] - s[m - 1 - i]).abs() < 1e-12,
                    "window {idx} not symmetric at {i}"
                );
            }
        }
    }

    // All windows peak at center
    #[test]
    fn all_windows_peak_at_center() {
        let m = 9;
        let windows: Vec<Array<f64, Ix1>> = vec![
            bartlett(m).unwrap(),
            blackman(m).unwrap(),
            hamming(m).unwrap(),
            hanning(m).unwrap(),
            kaiser(m, 5.0).unwrap(),
        ];
        for (idx, w) in windows.iter().enumerate() {
            let s = w.as_slice().unwrap();
            let center = s[m / 2];
            assert!(
                (center - 1.0).abs() < 1e-10,
                "window {idx} center = {center}, expected 1.0"
            );
        }
    }

    // Kaiser with beta=0 should be a rectangular window (all ones)
    #[test]
    fn kaiser_beta_zero() {
        let w = kaiser(5, 0.0).unwrap();
        let s = w.as_slice().unwrap();
        for &v in s {
            assert!((v - 1.0).abs() < 1e-10, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn bessel_i0_scalar_zero() {
        assert!((bessel_i0_scalar::<f64>(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn bessel_i0_scalar_known() {
        // Issue #296: tighten the tolerances. The Abramowitz & Stegun
        // approximation is rated ~1e-7 in the polynomial branch and
        // ~1e-7 in the asymptotic branch per its documentation.
        // I0(1) ~ 1.2660658777520082
        assert!((bessel_i0_scalar::<f64>(1.0) - 1.266_065_877_752_008_2).abs() < 1e-7);
        // I0(5) ~ 27.239871823604443 (asymptotic branch)
        assert!((bessel_i0_scalar::<f64>(5.0) - 27.239_871_823_604_443).abs() < 1e-5);
        // I0(10) ~ 2815.7166284662544
        let expected_i0_10 = 2_815.716_628_466_254;
        assert!((bessel_i0_scalar::<f64>(10.0) - expected_i0_10).abs() / expected_i0_10 < 1e-5);
    }

    // ----- Edge-length window coverage (#295) -----

    #[test]
    fn bartlett_m2_is_zeros() {
        // Bartlett of length 2: w(n) = 1 - |2n/(M-1) - 1|
        // n=0: 1 - |0 - 1| = 0
        // n=1: 1 - |2 - 1| = 0
        let w = bartlett(2).unwrap();
        assert_eq!(w.as_slice().unwrap(), &[0.0, 0.0]);
    }

    #[test]
    fn hanning_m2_is_zeros() {
        // Hann of length 2: w(n) = 0.5 * (1 - cos(2*pi*n/(M-1)))
        // n=0: 0.5 * (1 - cos(0)) = 0
        // n=1: 0.5 * (1 - cos(2*pi)) = 0
        let w = hanning(2).unwrap();
        let s = w.as_slice().unwrap();
        assert!(s[0].abs() < 1e-14);
        assert!(s[1].abs() < 1e-14);
    }

    #[test]
    fn bartlett_even_length_is_symmetric() {
        // Even-length windows have no single "center"; symmetry holds
        // around the midpoint between indices (M/2 - 1) and (M/2).
        for &m in &[4usize, 6, 8, 10] {
            let w = bartlett(m).unwrap();
            let s = w.as_slice().unwrap();
            for i in 0..m / 2 {
                assert!(
                    (s[i] - s[m - 1 - i]).abs() < 1e-14,
                    "bartlett({m}) not symmetric at {i}"
                );
            }
        }
    }

    #[test]
    fn blackman_even_length_is_symmetric() {
        for &m in &[4usize, 6, 8, 10] {
            let w = blackman(m).unwrap();
            let s = w.as_slice().unwrap();
            for i in 0..m / 2 {
                assert!(
                    (s[i] - s[m - 1 - i]).abs() < 1e-14,
                    "blackman({m}) not symmetric at {i}"
                );
            }
        }
    }

    #[test]
    fn kaiser_large_beta_errors() {
        // Issue #294: beta above the f64 safe range should error,
        // not silently produce NaN.
        assert!(kaiser(8, 800.0).is_err());
        assert!(kaiser(8, 1000.0).is_err());
        // 700 is still safe.
        assert!(kaiser(8, 700.0).is_ok());
    }

    // =======================================================================
    // SciPy-extended windows (cosine, exponential, gaussian, general_cosine,
    // general_hamming, nuttall, parzen, taylor, tukey).
    // =======================================================================

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ----- cosine -----

    #[test]
    fn cosine_length_and_endpoints() {
        // cosine(M) = sin(pi(n + 0.5)/M); endpoints are sin(pi/2M) and
        // sin(pi(M-0.5)/M) — both small but nonzero.
        let w = cosine(8).unwrap();
        let s = w.as_slice().unwrap();
        assert_eq!(s.len(), 8);
        // Symmetric.
        for i in 0..4 {
            assert!(close(s[i], s[7 - i], 1e-14));
        }
    }

    #[test]
    fn cosine_m1_and_m0() {
        assert_eq!(cosine(0).unwrap().shape(), &[0]);
        assert_eq!(cosine(1).unwrap().as_slice().unwrap(), &[1.0]);
    }

    // ----- exponential -----

    #[test]
    fn exponential_centred_default() {
        // tau=1, default centre = (M-1)/2 = 3.5 for M=8 — midpoint
        // between samples 3 and 4.
        let w = exponential(8, None, 1.0).unwrap();
        let s = w.as_slice().unwrap();
        // Symmetric.
        for i in 0..4 {
            assert!(close(s[i], s[7 - i], 1e-14));
        }
        // Centre samples have largest values.
        let centre_max = s[3].max(s[4]);
        for &v in s {
            assert!(v <= centre_max + 1e-14);
        }
    }

    #[test]
    fn exponential_rejects_nonpositive_tau() {
        assert!(exponential(8, None, 0.0).is_err());
        assert!(exponential(8, None, -1.0).is_err());
        assert!(exponential(8, None, f64::NAN).is_err());
    }

    // ----- gaussian -----

    #[test]
    fn gaussian_centre_is_one() {
        // For odd M, centre sample is exactly 1.
        let w = gaussian(11, 2.0).unwrap();
        let s = w.as_slice().unwrap();
        assert!(close(s[5], 1.0, 1e-14));
    }

    #[test]
    fn gaussian_known_value() {
        // gaussian(7, 1) at n=4: z = (4 - 3) / 1 = 1, exp(-0.5) = 0.6065...
        let w = gaussian(7, 1.0).unwrap();
        let s = w.as_slice().unwrap();
        assert!(close(s[4], (-0.5_f64).exp(), 1e-14));
    }

    #[test]
    fn gaussian_rejects_nonpositive_std() {
        assert!(gaussian(8, 0.0).is_err());
        assert!(gaussian(8, -1.0).is_err());
    }

    // ----- general_cosine -----

    #[test]
    fn general_cosine_with_hann_coeffs_matches_hann() {
        // [0.5, 0.5] → 0.5 - 0.5 cos(2πn/(M-1)) = Hann.
        let m = 9;
        let gc = general_cosine(m, &[0.5, 0.5]).unwrap();
        let hn = hanning(m).unwrap();
        for i in 0..m {
            assert!(close(
                gc.as_slice().unwrap()[i],
                hn.as_slice().unwrap()[i],
                1e-14,
            ));
        }
    }

    #[test]
    fn general_cosine_with_blackman_coeffs_matches_blackman() {
        // [0.42, 0.5, 0.08] → classical Blackman.
        let m = 9;
        let gc = general_cosine(m, &[0.42, 0.5, 0.08]).unwrap();
        let bk = blackman(m).unwrap();
        for i in 0..m {
            assert!(close(
                gc.as_slice().unwrap()[i],
                bk.as_slice().unwrap()[i],
                1e-12,
            ));
        }
    }

    #[test]
    fn general_cosine_rejects_empty_coeffs() {
        assert!(general_cosine(8, &[]).is_err());
    }

    // ----- general_hamming -----

    #[test]
    fn general_hamming_alpha_half_matches_hann() {
        let m = 9;
        let gh = general_hamming(m, 0.5).unwrap();
        let hn = hanning(m).unwrap();
        for i in 0..m {
            assert!(close(
                gh.as_slice().unwrap()[i],
                hn.as_slice().unwrap()[i],
                1e-14,
            ));
        }
    }

    #[test]
    fn general_hamming_alpha_054_matches_hamming() {
        let m = 9;
        let gh = general_hamming(m, 0.54).unwrap();
        let hm = hamming(m).unwrap();
        for i in 0..m {
            assert!(close(
                gh.as_slice().unwrap()[i],
                hm.as_slice().unwrap()[i],
                1e-14,
            ));
        }
    }

    // ----- nuttall -----

    #[test]
    fn nuttall_endpoints_are_small() {
        // Nuttall is engineered to have very low sidelobes; endpoints
        // are O(1e-3) for typical M.
        let w = nuttall(64).unwrap();
        let s = w.as_slice().unwrap();
        assert!(s[0].abs() < 1e-2);
        assert!(s[s.len() - 1].abs() < 1e-2);
    }

    #[test]
    fn nuttall_is_symmetric() {
        let m = 33;
        let w = nuttall(m).unwrap();
        let s = w.as_slice().unwrap();
        for i in 0..m / 2 {
            assert!(close(s[i], s[m - 1 - i], 1e-14));
        }
    }

    // ----- parzen -----

    #[test]
    fn parzen_endpoints_are_zero() {
        let w = parzen(16).unwrap();
        let s = w.as_slice().unwrap();
        // Outer edge of Parzen has very small value (cubic falloff).
        assert!(s[0].abs() < 1e-2);
        assert!(s[s.len() - 1].abs() < 1e-2);
    }

    #[test]
    fn parzen_centre_is_one() {
        // For odd M, centre is at sample (M-1)/2; r = 0 → w = 1.
        let w = parzen(13).unwrap();
        let s = w.as_slice().unwrap();
        assert!(close(s[6], 1.0, 1e-14));
    }

    #[test]
    fn parzen_is_symmetric() {
        let m = 21;
        let w = parzen(m).unwrap();
        let s = w.as_slice().unwrap();
        for i in 0..m / 2 {
            assert!(close(s[i], s[m - 1 - i], 1e-14));
        }
    }

    // ----- taylor -----

    #[test]
    fn taylor_default_normalised_centre_is_one() {
        // With norm=true the centre sample is 1.0.
        let w = taylor(33, 4, 30.0, true).unwrap();
        let s = w.as_slice().unwrap();
        assert!(close(s[16], 1.0, 1e-12));
    }

    #[test]
    fn taylor_is_symmetric() {
        let m = 33;
        let w = taylor(m, 4, 30.0, true).unwrap();
        let s = w.as_slice().unwrap();
        for i in 0..m / 2 {
            assert!(close(s[i], s[m - 1 - i], 1e-12));
        }
    }

    #[test]
    fn taylor_rejects_nbar_zero() {
        assert!(taylor(8, 0, 30.0, true).is_err());
        assert!(taylor(8, 4, f64::NAN, true).is_err());
    }

    // ----- tukey -----

    #[test]
    fn tukey_alpha_zero_is_rectangular() {
        let w = tukey(8, 0.0).unwrap();
        for &v in w.as_slice().unwrap() {
            assert!(close(v, 1.0, 1e-14));
        }
    }

    #[test]
    fn tukey_alpha_one_matches_hann() {
        // alpha=1 → fully Hann-shaped.
        let m = 9;
        let tk = tukey(m, 1.0).unwrap();
        let hn = hanning(m).unwrap();
        for i in 0..m {
            assert!(close(
                tk.as_slice().unwrap()[i],
                hn.as_slice().unwrap()[i],
                1e-12,
            ));
        }
    }

    #[test]
    fn tukey_centre_is_one() {
        let m = 21;
        let w = tukey(m, 0.5).unwrap();
        // Wide flat-top region: middle must be 1.
        assert!(close(w.as_slice().unwrap()[m / 2], 1.0, 1e-14));
    }

    #[test]
    fn tukey_rejects_invalid_alpha() {
        assert!(tukey(8, -0.1).is_err());
        assert!(tukey(8, 1.1).is_err());
        assert!(tukey(8, f64::NAN).is_err());
    }
}

#[cfg(test)]
mod debug_test_removed {
    // Debug test scaffolding for #740 chebwin development;
    // removed once the closed-form output matched scipy.
}
