// ferray-fft: Frequency generation — fftfreq, rfftfreq (REQ-9, REQ-10)

use ferray_core::Array;
use ferray_core::dimension::Ix1;
use ferray_core::error::{FerrayError, FerrayResult};

/// Return the Discrete Fourier Transform sample frequencies.
///
/// Analogous to `numpy.fft.fftfreq`. The returned array contains the
/// frequency bin centers in cycles per unit of the sample spacing.
///
/// For a window length `n` and sample spacing `d`:
/// ```text
/// f = [0, 1, ..., n/2-1, -n/2, ..., -1] / (d*n)   if n is even
/// f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd
/// ```
///
/// # Parameters
/// - `n`: Window length (number of samples).
/// - `d`: Sample spacing (inverse of the sampling rate). Default is 1.0.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if `n` is 0 or `d` is 0.
///
/// # Example
/// ```
/// use ferray_fft::fftfreq;
///
/// let freq = fftfreq(8, 1.0).unwrap();
/// // Returns [0.0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125]
/// ```
pub fn fftfreq(n: usize, d: f64) -> FerrayResult<Array<f64, Ix1>> {
    if n == 0 {
        return Err(FerrayError::invalid_value("fftfreq: n must be > 0"));
    }
    if d == 0.0 {
        return Err(FerrayError::invalid_value(
            "fftfreq: sample spacing d must be nonzero",
        ));
    }

    let nf = n as f64;
    let val = 1.0 / (nf * d);
    let mut result = Vec::with_capacity(n);

    // Positive frequencies: 0, 1, ..., (n-1)/2
    let positive_end = n.div_ceil(2);
    for i in 0..positive_end {
        result.push(i as f64 * val);
    }

    // Negative frequencies: -n/2, ..., -1
    let negative_start = if n % 2 == 0 {
        -(n as isize / 2)
    } else {
        -((n as isize - 1) / 2)
    };
    for i in negative_start..0 {
        result.push(i as f64 * val);
    }

    Array::from_vec(Ix1::new([n]), result)
}

/// Return the Discrete Fourier Transform sample frequencies for `rfft`.
///
/// Analogous to `numpy.fft.rfftfreq`. Since `rfft` only returns the
/// non-negative frequency terms, this returns `n/2 + 1` values.
///
/// ```text
/// f = [0, 1, ..., n/2] / (d*n)
/// ```
///
/// # Parameters
/// - `n`: Window length (number of samples in the original signal).
/// - `d`: Sample spacing (inverse of the sampling rate). Default is 1.0.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if `n` is 0 or `d` is 0.
///
/// # Example
/// ```
/// use ferray_fft::rfftfreq;
///
/// let freq = rfftfreq(8, 1.0).unwrap();
/// // Returns [0.0, 0.125, 0.25, 0.375, 0.5]
/// ```
pub fn rfftfreq(n: usize, d: f64) -> FerrayResult<Array<f64, Ix1>> {
    if n == 0 {
        return Err(FerrayError::invalid_value("rfftfreq: n must be > 0"));
    }
    if d == 0.0 {
        return Err(FerrayError::invalid_value(
            "rfftfreq: sample spacing d must be nonzero",
        ));
    }

    let nf = n as f64;
    let val = 1.0 / (nf * d);
    let out_len = n / 2 + 1;
    let mut result = Vec::with_capacity(out_len);

    for i in 0..out_len {
        result.push(i as f64 * val);
    }

    Array::from_vec(Ix1::new([out_len]), result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fftfreq_8() {
        // AC-4: fftfreq(8, 1.0) returns [0, 1, 2, 3, -4, -3, -2, -1] / 8
        let freq = fftfreq(8, 1.0).unwrap();
        let expected = [0.0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125];
        let data: Vec<f64> = freq.iter().copied().collect();
        assert_eq!(data.len(), 8);
        for (a, b) in data.iter().zip(expected.iter()) {
            assert!(
                (a - b).abs() < 1e-15,
                "fftfreq mismatch: got {a}, expected {b}"
            );
        }
    }

    #[test]
    fn fftfreq_odd() {
        // fftfreq(5, 1.0) = [0, 1, 2, -2, -1] / 5
        let freq = fftfreq(5, 1.0).unwrap();
        let expected = [0.0, 0.2, 0.4, -0.4, -0.2];
        let data: Vec<f64> = freq.iter().copied().collect();
        for (a, b) in data.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn fftfreq_with_spacing() {
        // fftfreq(4, 0.5) = [0, 1, -2, -1] / (4 * 0.5) = [0, 0.5, -1.0, -0.5]
        let freq = fftfreq(4, 0.5).unwrap();
        let expected = [0.0, 0.5, -1.0, -0.5];
        let data: Vec<f64> = freq.iter().copied().collect();
        for (a, b) in data.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn fftfreq_zero_n_errors() {
        assert!(fftfreq(0, 1.0).is_err());
    }

    #[test]
    fn fftfreq_zero_d_errors() {
        assert!(fftfreq(8, 0.0).is_err());
    }

    #[test]
    fn rfftfreq_8() {
        let freq = rfftfreq(8, 1.0).unwrap();
        let expected = [0.0, 0.125, 0.25, 0.375, 0.5];
        let data: Vec<f64> = freq.iter().copied().collect();
        assert_eq!(data.len(), 5);
        for (a, b) in data.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn rfftfreq_odd() {
        // rfftfreq(5, 1.0) = [0, 1, 2] / 5 = [0.0, 0.2, 0.4]
        let freq = rfftfreq(5, 1.0).unwrap();
        let expected = [0.0, 0.2, 0.4];
        let data: Vec<f64> = freq.iter().copied().collect();
        assert_eq!(data.len(), 3);
        for (a, b) in data.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn rfftfreq_zero_n_errors() {
        assert!(rfftfreq(0, 1.0).is_err());
    }

    #[test]
    fn rfftfreq_n1() {
        let freq = rfftfreq(1, 1.0).unwrap();
        assert_eq!(freq.shape(), &[1]);
        assert!((freq.iter().next().unwrap() - 0.0).abs() < 1e-15);
    }
}
