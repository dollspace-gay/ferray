// ferray-ufunc: Convolution
//
// convolve with modes: Full, Same, Valid

use ferray_core::Array;
use ferray_core::dimension::Ix1;
use ferray_core::dtype::Element;
use ferray_core::error::{FerrayError, FerrayResult};

/// Convolution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvolveMode {
    /// Full convolution output (length = N + M - 1).
    Full,
    /// Output has length max(N, M).
    Same,
    /// Output only where signals fully overlap (length = max(N, M) - min(N, M) + 1).
    Valid,
}

/// Discrete, linear convolution of two 1-D arrays.
///
/// Computes `convolve(a, v, mode)` following `NumPy` semantics — direct
/// O(n·m) algorithm, matching `numpy.convolve`. For large floating-point
/// inputs prefer [`fftconvolve`] (behind the `fft-convolve` feature),
/// which is O((n+m) · log(n+m)) via FFT.
///
/// The inner loop is restructured to iterate over output positions
/// rather than input pairs (#89). The previous form
/// `full[i+j] += a[i] * v[j]` had a strided write pattern that
/// confused auto-vectorisation; iterating over `k = i+j` and
/// accumulating a dot product `Σ a[i] * v[k-i]` gives the auto-
/// vectoriser a clean inner loop with linear writes.
pub fn convolve<T>(
    a: &Array<T, Ix1>,
    v: &Array<T, Ix1>,
    mode: ConvolveMode,
) -> FerrayResult<Array<T, Ix1>>
where
    T: Element + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Copy,
{
    let a_data: Vec<T> = a.iter().copied().collect();
    let v_data: Vec<T> = v.iter().copied().collect();
    let n = a_data.len();
    let m = v_data.len();

    if n == 0 || m == 0 {
        return Err(FerrayError::invalid_value(
            "convolve: input arrays must be non-empty",
        ));
    }

    // Full convolution — output position k = i + j, inner loop is a
    // dot product with linear indexing into both inputs.
    let full_len = n + m - 1;
    let mut full = vec![<T as Element>::zero(); full_len];

    for k in 0..full_len {
        // Range of valid i: max(0, k+1-m) <= i < min(n, k+1).
        let i_lo = (k + 1).saturating_sub(m);
        let i_hi = (k + 1).min(n);
        let mut acc = <T as Element>::zero();
        for i in i_lo..i_hi {
            // SAFETY-style bounds rationale: the i_lo/i_hi math
            // guarantees i < n and (k - i) < m. The compiler usually
            // proves this and eliminates bounds checks; if it can't,
            // the panics still produce correct (slightly slower) code.
            acc = acc + a_data[i] * v_data[k - i];
        }
        full[k] = acc;
    }

    match mode {
        ConvolveMode::Full => Array::from_vec(Ix1::new([full_len]), full),
        ConvolveMode::Same => {
            let out_len = n.max(m);
            let start = (full_len - out_len) / 2;
            let result = full[start..start + out_len].to_vec();
            Array::from_vec(Ix1::new([out_len]), result)
        }
        ConvolveMode::Valid => {
            let out_len = if n >= m { n - m + 1 } else { m - n + 1 };
            let start = m.min(n) - 1;
            let result = full[start..start + out_len].to_vec();
            Array::from_vec(Ix1::new([out_len]), result)
        }
    }
}

/// FFT-based convolution of two 1-D f64 arrays — O((n+m) · log(n+m))
/// via real-to-complex FFT.
///
/// Matches `scipy.signal.fftconvolve(a, v, mode)` for the supported
/// modes. Numerically equivalent to [`convolve`] up to float roundoff
/// (typically a few ULPs at the magnitude of the result; for inputs
/// that span many orders of magnitude the FFT path may diverge more
/// because FFT accumulates O(log N) ULPs of error per butterfly).
///
/// Available only with the `fft-convolve` feature enabled (pulls in
/// `ferray-fft` / `realfft`). For inputs where n·m < ~50_000 the
/// direct path in [`convolve`] is faster; the cross-over depends on
/// hardware but is typically around (n+m) > 1024.
#[cfg(feature = "fft-convolve")]
pub fn fftconvolve(
    a: &Array<f64, Ix1>,
    v: &Array<f64, Ix1>,
    mode: ConvolveMode,
) -> FerrayResult<Array<f64, Ix1>> {
    use ferray_fft::{FftNorm, irfft, rfft};

    let n = a.size();
    let m = v.size();
    if n == 0 || m == 0 {
        return Err(FerrayError::invalid_value(
            "fftconvolve: input arrays must be non-empty",
        ));
    }

    // Zero-pad both inputs to the full convolution length.
    let full_len = n + m - 1;
    let mut a_pad = vec![0.0f64; full_len];
    let mut v_pad = vec![0.0f64; full_len];
    for (dst, &src) in a_pad.iter_mut().zip(a.iter()) {
        *dst = src;
    }
    for (dst, &src) in v_pad.iter_mut().zip(v.iter()) {
        *dst = src;
    }
    let a_padded = Array::<f64, Ix1>::from_vec(Ix1::new([full_len]), a_pad)?;
    let v_padded = Array::<f64, Ix1>::from_vec(Ix1::new([full_len]), v_pad)?;

    // Forward real FFT on both, multiply elementwise, inverse real FFT.
    let a_fft = rfft(&a_padded, None, None, FftNorm::Backward)?;
    let v_fft = rfft(&v_padded, None, None, FftNorm::Backward)?;

    let a_spec: Vec<num_complex::Complex<f64>> = a_fft.iter().copied().collect();
    let v_spec: Vec<num_complex::Complex<f64>> = v_fft.iter().copied().collect();
    let prod: Vec<num_complex::Complex<f64>> = a_spec
        .iter()
        .zip(v_spec.iter())
        .map(|(a, b)| a * b)
        .collect();
    let prod_arr = Array::<num_complex::Complex<f64>, Ix1>::from_vec(Ix1::new([prod.len()]), prod)?;

    let inv = irfft(&prod_arr, Some(full_len), None, FftNorm::Backward)?;
    let inv_data: Vec<f64> = inv.iter().copied().collect();

    match mode {
        ConvolveMode::Full => Array::from_vec(Ix1::new([full_len]), inv_data),
        ConvolveMode::Same => {
            let out_len = n.max(m);
            let start = (full_len - out_len) / 2;
            let slice = inv_data[start..start + out_len].to_vec();
            Array::from_vec(Ix1::new([out_len]), slice)
        }
        ConvolveMode::Valid => {
            let out_len = if n >= m { n - m + 1 } else { m - n + 1 };
            let start = m.min(n) - 1;
            let slice = inv_data[start..start + out_len].to_vec();
            Array::from_vec(Ix1::new([out_len]), slice)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_util::arr1;

    #[test]
    fn test_convolve_full() {
        let a = arr1(vec![1.0, 2.0, 3.0]);
        let v = arr1(vec![0.0, 1.0, 0.5]);
        let r = convolve(&a, &v, ConvolveMode::Full).unwrap();
        let s = r.as_slice().unwrap();
        // [0*1, 1*1+0*2, 0.5*1+1*2+0*3, 0.5*2+1*3, 0.5*3]
        // = [0, 1, 2.5, 4, 1.5]
        assert_eq!(s.len(), 5);
        assert!((s[0] - 0.0).abs() < 1e-12);
        assert!((s[1] - 1.0).abs() < 1e-12);
        assert!((s[2] - 2.5).abs() < 1e-12);
        assert!((s[3] - 4.0).abs() < 1e-12);
        assert!((s[4] - 1.5).abs() < 1e-12);
    }

    #[test]
    fn test_convolve_same() {
        let a = arr1(vec![1.0, 2.0, 3.0]);
        let v = arr1(vec![0.0, 1.0, 0.5]);
        let r = convolve(&a, &v, ConvolveMode::Same).unwrap();
        assert_eq!(r.size(), 3);
        let s = r.as_slice().unwrap();
        // Full = [0, 1, 2.5, 4, 1.5], same takes middle 3 = [1, 2.5, 4]
        assert!((s[0] - 1.0).abs() < 1e-12);
        assert!((s[1] - 2.5).abs() < 1e-12);
        assert!((s[2] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_convolve_valid() {
        let a = arr1(vec![1.0, 2.0, 3.0]);
        let v = arr1(vec![0.0, 1.0, 0.5]);
        let r = convolve(&a, &v, ConvolveMode::Valid).unwrap();
        assert_eq!(r.size(), 1);
        let s = r.as_slice().unwrap();
        assert!((s[0] - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_convolve_simple() {
        let a = arr1(vec![1.0, 1.0, 1.0]);
        let v = arr1(vec![1.0, 1.0, 1.0]);
        let r = convolve(&a, &v, ConvolveMode::Full).unwrap();
        let s = r.as_slice().unwrap();
        assert_eq!(s.len(), 5);
        assert!((s[0] - 1.0).abs() < 1e-12);
        assert!((s[1] - 2.0).abs() < 1e-12);
        assert!((s[2] - 3.0).abs() < 1e-12);
        assert!((s[3] - 2.0).abs() < 1e-12);
        assert!((s[4] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_convolve_i32() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![1, 2, 3]).unwrap();
        let v = Array::<i32, Ix1>::from_vec(Ix1::new([2]), vec![1, 1]).unwrap();
        let r = convolve(&a, &v, ConvolveMode::Full).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[1, 3, 5, 3]);
    }
}
