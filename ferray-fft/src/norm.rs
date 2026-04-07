// ferray-fft: FftNorm enum and scaling logic (REQ-14)

/// Normalization mode for FFT operations, matching NumPy's `norm` parameter.
///
/// - [`Backward`](FftNorm::Backward): No normalization on forward, `1/n` on inverse (default).
/// - [`Forward`](FftNorm::Forward): `1/n` on forward, no normalization on inverse.
/// - [`Ortho`](FftNorm::Ortho): `1/sqrt(n)` in both directions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FftNorm {
    /// No normalization on forward FFT, divide by `n` on inverse.
    /// This is the default, matching NumPy.
    #[default]
    Backward,
    /// Divide by `n` on forward FFT, no normalization on inverse.
    Forward,
    /// Divide by `sqrt(n)` in both directions (unitary transform).
    Ortho,
}

/// Direction of the FFT transform, used to determine the normalization factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftDirection {
    /// Forward transform (time → frequency).
    Forward,
    /// Inverse transform (frequency → time).
    Inverse,
}

impl FftNorm {
    /// Compute the normalization scale factor for a given FFT size and
    /// direction, as an `f64`.
    ///
    /// Returns `1.0` when no normalization is needed. The generic FFT
    /// code path uses [`crate::float::FftFloat::scale_factor`] which
    /// wraps this and casts to the correct precision.
    pub fn scale_factor_f64(self, n: usize, direction: FftDirection) -> f64 {
        let nf = n as f64;
        match (self, direction) {
            // Backward: no scaling on forward, 1/n on inverse
            (FftNorm::Backward, FftDirection::Forward) => 1.0,
            (FftNorm::Backward, FftDirection::Inverse) => 1.0 / nf,
            // Forward: 1/n on forward, no scaling on inverse
            (FftNorm::Forward, FftDirection::Forward) => 1.0 / nf,
            (FftNorm::Forward, FftDirection::Inverse) => 1.0,
            // Ortho: 1/sqrt(n) in both directions
            (FftNorm::Ortho, _) => 1.0 / nf.sqrt(),
        }
    }

    /// Legacy alias preserved for internal f64-only code paths.
    ///
    /// New code should prefer [`crate::float::FftFloat::scale_factor`]
    /// so it works for both f32 and f64 without an extra cast.
    #[inline]
    pub(crate) fn scale_factor(self, n: usize, direction: FftDirection) -> f64 {
        self.scale_factor_f64(n, direction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backward_norm_factors() {
        let n = 8;
        assert_eq!(
            FftNorm::Backward.scale_factor(n, FftDirection::Forward),
            1.0
        );
        assert!((FftNorm::Backward.scale_factor(n, FftDirection::Inverse) - 0.125).abs() < 1e-15);
    }

    #[test]
    fn forward_norm_factors() {
        let n = 8;
        assert!((FftNorm::Forward.scale_factor(n, FftDirection::Forward) - 0.125).abs() < 1e-15);
        assert_eq!(FftNorm::Forward.scale_factor(n, FftDirection::Inverse), 1.0);
    }

    #[test]
    fn ortho_norm_factors() {
        let n = 4;
        let expected = 1.0 / 2.0; // 1/sqrt(4)
        assert!((FftNorm::Ortho.scale_factor(n, FftDirection::Forward) - expected).abs() < 1e-15);
        assert!((FftNorm::Ortho.scale_factor(n, FftDirection::Inverse) - expected).abs() < 1e-15);
    }

    #[test]
    fn default_is_backward() {
        assert_eq!(FftNorm::default(), FftNorm::Backward);
    }
}
