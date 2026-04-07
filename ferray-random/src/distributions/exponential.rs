// ferray-random: Exponential distribution sampling — standard_exponential, exponential

use ferray_core::{Array, FerrayError, IxDyn};

use crate::bitgen::BitGenerator;
use crate::generator::{Generator, generate_vec, shape_size, vec_to_array_f64};
use crate::shape::IntoShape;

/// Generate a single standard exponential variate (rate=1) via inverse CDF.
pub(crate) fn standard_exponential_single<B: BitGenerator>(bg: &mut B) -> f64 {
    loop {
        let u = bg.next_f64();
        if u > f64::EPSILON {
            return -u.ln();
        }
    }
}

impl<B: BitGenerator> Generator<B> {
    /// Generate an array of standard exponential (rate=1, scale=1) variates.
    ///
    /// Uses the inverse CDF method: -ln(U) where U ~ Uniform(0,1).
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `shape` is invalid.
    pub fn standard_exponential(
        &mut self,
        size: impl IntoShape,
    ) -> Result<Array<f64, IxDyn>, FerrayError> {
        let shape = size.into_shape()?;
        let n = shape_size(&shape);
        let data = generate_vec(self, n, standard_exponential_single);
        vec_to_array_f64(data, &shape)
    }

    /// Generate an array of exponential variates with the given scale.
    ///
    /// The exponential distribution has PDF: f(x) = (1/scale) * exp(-x/scale).
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `scale <= 0` or `shape` is invalid.
    pub fn exponential(
        &mut self,
        scale: f64,
        size: impl IntoShape,
    ) -> Result<Array<f64, IxDyn>, FerrayError> {
        if scale <= 0.0 {
            return Err(FerrayError::invalid_value(format!(
                "scale must be positive, got {scale}"
            )));
        }
        let shape = size.into_shape()?;
        let n = shape_size(&shape);
        let data = generate_vec(self, n, |bg| scale * standard_exponential_single(bg));
        vec_to_array_f64(data, &shape)
    }
}

#[cfg(test)]
mod tests {
    use crate::default_rng_seeded;

    #[test]
    fn standard_exponential_positive() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.standard_exponential(10_000).unwrap();
        let slice = arr.as_slice().unwrap();
        for &v in slice {
            assert!(
                v > 0.0,
                "standard_exponential produced non-positive value: {v}"
            );
        }
    }

    #[test]
    fn standard_exponential_mean_variance() {
        let mut rng = default_rng_seeded(42);
        let n = 100_000;
        let arr = rng.standard_exponential(n).unwrap();
        let slice = arr.as_slice().unwrap();
        let mean: f64 = slice.iter().sum::<f64>() / n as f64;
        let var: f64 = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        // Exp(1): mean=1, var=1
        let se = (1.0 / n as f64).sqrt();
        assert!((mean - 1.0).abs() < 3.0 * se, "mean {mean} too far from 1");
        assert!((var - 1.0).abs() < 0.05, "variance {var} too far from 1");
    }

    #[test]
    fn exponential_mean() {
        let mut rng = default_rng_seeded(42);
        let n = 100_000;
        let scale = 3.0;
        let arr = rng.exponential(scale, n).unwrap();
        let slice = arr.as_slice().unwrap();
        let mean: f64 = slice.iter().sum::<f64>() / n as f64;
        let se = (scale * scale / n as f64).sqrt();
        assert!(
            (mean - scale).abs() < 3.0 * se,
            "mean {mean} too far from {scale}"
        );
    }

    #[test]
    fn exponential_bad_scale() {
        let mut rng = default_rng_seeded(42);
        assert!(rng.exponential(0.0, 100).is_err());
        assert!(rng.exponential(-1.0, 100).is_err());
    }

    #[test]
    fn exponential_deterministic() {
        let mut rng1 = default_rng_seeded(42);
        let mut rng2 = default_rng_seeded(42);
        let a = rng1.exponential(2.0, 100).unwrap();
        let b = rng2.exponential(2.0, 100).unwrap();
        assert_eq!(a.as_slice().unwrap(), b.as_slice().unwrap());
    }

    #[test]
    fn exponential_mean_and_variance() {
        let mut rng = default_rng_seeded(42);
        let n = 100_000;
        let scale = 3.0;
        let arr = rng.exponential(scale, n).unwrap();
        let s = arr.as_slice().unwrap();
        let mean: f64 = s.iter().sum::<f64>() / n as f64;
        let var: f64 = s.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        // Exponential(scale): mean=scale, var=scale^2
        assert!(
            (mean - scale).abs() < 0.1,
            "exponential mean {mean} too far from {scale}"
        );
        assert!(
            (var - scale * scale).abs() < 1.0,
            "exponential variance {var} too far from {}",
            scale * scale
        );
    }

    #[test]
    fn standard_exponential_mean() {
        let mut rng = default_rng_seeded(42);
        let n = 100_000;
        let arr = rng.standard_exponential(n).unwrap();
        let s = arr.as_slice().unwrap();
        let mean: f64 = s.iter().sum::<f64>() / n as f64;
        assert!(
            (mean - 1.0).abs() < 0.02,
            "standard_exponential mean {mean} too far from 1.0"
        );
        // All values should be non-negative
        assert!(s.iter().all(|&x| x >= 0.0), "negative exponential value");
    }
}
