// ferray-random: Exponential distribution sampling — standard_exponential, exponential

use ferray_core::{Array, FerrayError, Ix1};

use crate::bitgen::BitGenerator;
use crate::generator::{Generator, generate_vec, vec_to_array1};

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
    /// # Arguments
    /// * `size` - Number of values to generate.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `size` is zero.
    pub fn standard_exponential(&mut self, size: usize) -> Result<Array<f64, Ix1>, FerrayError> {
        if size == 0 {
            return Err(FerrayError::invalid_value("size must be > 0"));
        }
        let data = generate_vec(self, size, standard_exponential_single);
        vec_to_array1(data)
    }

    /// Generate an array of exponential variates with the given scale.
    ///
    /// The exponential distribution has PDF: f(x) = (1/scale) * exp(-x/scale).
    ///
    /// # Arguments
    /// * `scale` - Scale parameter (1/rate), must be positive.
    /// * `size` - Number of values to generate.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `scale <= 0` or `size` is zero.
    pub fn exponential(&mut self, scale: f64, size: usize) -> Result<Array<f64, Ix1>, FerrayError> {
        if size == 0 {
            return Err(FerrayError::invalid_value("size must be > 0"));
        }
        if scale <= 0.0 {
            return Err(FerrayError::invalid_value(format!(
                "scale must be positive, got {scale}"
            )));
        }
        let data = generate_vec(self, size, |bg| scale * standard_exponential_single(bg));
        vec_to_array1(data)
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
}
