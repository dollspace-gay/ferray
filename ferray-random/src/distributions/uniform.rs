// ferray-random: Uniform distribution sampling — random, uniform, integers

use ferray_core::{Array, FerrayError, Ix1};

use crate::bitgen::BitGenerator;
use crate::generator::{
    Generator, generate_vec, generate_vec_i64, vec_to_array1, vec_to_array1_i64,
};

impl<B: BitGenerator> Generator<B> {
    /// Generate an array of uniformly distributed `f64` values in [0, 1).
    ///
    /// Equivalent to NumPy's `Generator.random(size)`.
    ///
    /// # Arguments
    /// * `size` - Number of values to generate.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `size` is zero.
    ///
    /// # Example
    /// ```
    /// let mut rng = ferray_random::default_rng_seeded(42);
    /// let arr = rng.random(10).unwrap();
    /// assert_eq!(arr.shape(), &[10]);
    /// ```
    pub fn random(&mut self, size: usize) -> Result<Array<f64, Ix1>, FerrayError> {
        if size == 0 {
            return Err(FerrayError::invalid_value("size must be > 0"));
        }
        let data = generate_vec(self, size, |bg| bg.next_f64());
        vec_to_array1(data)
    }

    /// Generate an array of uniformly distributed `f64` values in [low, high).
    ///
    /// Equivalent to NumPy's `Generator.uniform(low, high, size)`.
    ///
    /// # Arguments
    /// * `low` - Lower bound (inclusive).
    /// * `high` - Upper bound (exclusive).
    /// * `size` - Number of values to generate.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `low >= high` or `size` is zero.
    pub fn uniform(
        &mut self,
        low: f64,
        high: f64,
        size: usize,
    ) -> Result<Array<f64, Ix1>, FerrayError> {
        if size == 0 {
            return Err(FerrayError::invalid_value("size must be > 0"));
        }
        if low >= high {
            return Err(FerrayError::invalid_value(format!(
                "low ({low}) must be less than high ({high})"
            )));
        }
        let range = high - low;
        let data = generate_vec(self, size, |bg| low + bg.next_f64() * range);
        vec_to_array1(data)
    }

    /// Generate an array of uniformly distributed random integers in [low, high).
    ///
    /// Equivalent to NumPy's `Generator.integers(low, high, size)`.
    ///
    /// # Arguments
    /// * `low` - Lower bound (inclusive).
    /// * `high` - Upper bound (exclusive).
    /// * `size` - Number of values to generate.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `low >= high` or `size` is zero.
    pub fn integers(
        &mut self,
        low: i64,
        high: i64,
        size: usize,
    ) -> Result<Array<i64, Ix1>, FerrayError> {
        if size == 0 {
            return Err(FerrayError::invalid_value("size must be > 0"));
        }
        if low >= high {
            return Err(FerrayError::invalid_value(format!(
                "low ({low}) must be less than high ({high})"
            )));
        }
        let range = (high - low) as u64;
        let data = generate_vec_i64(self, size, |bg| low + bg.next_u64_bounded(range) as i64);
        vec_to_array1_i64(data)
    }
}

#[cfg(test)]
mod tests {
    use crate::default_rng_seeded;

    #[test]
    fn random_in_range() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.random(10_000).unwrap();
        let slice = arr.as_slice().unwrap();
        for &v in slice {
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn random_deterministic() {
        let mut rng1 = default_rng_seeded(42);
        let mut rng2 = default_rng_seeded(42);
        let a = rng1.random(100).unwrap();
        let b = rng2.random(100).unwrap();
        assert_eq!(a.as_slice().unwrap(), b.as_slice().unwrap());
    }

    #[test]
    fn uniform_in_range() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.uniform(5.0, 10.0, 10_000).unwrap();
        let slice = arr.as_slice().unwrap();
        for &v in slice {
            assert!(v >= 5.0 && v < 10.0, "value {v} out of range");
        }
    }

    #[test]
    fn uniform_bad_range() {
        let mut rng = default_rng_seeded(42);
        assert!(rng.uniform(10.0, 5.0, 100).is_err());
        assert!(rng.uniform(5.0, 5.0, 100).is_err());
    }

    #[test]
    fn integers_in_range() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.integers(0, 10, 10_000).unwrap();
        let slice = arr.as_slice().unwrap();
        for &v in slice {
            assert!((0..10).contains(&v), "value {v} out of range");
        }
    }

    #[test]
    fn integers_negative_range() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.integers(-5, 5, 1000).unwrap();
        let slice = arr.as_slice().unwrap();
        for &v in slice {
            assert!((-5..5).contains(&v), "value {v} out of range");
        }
    }

    #[test]
    fn integers_bad_range() {
        let mut rng = default_rng_seeded(42);
        assert!(rng.integers(10, 5, 100).is_err());
    }

    #[test]
    fn uniform_mean_variance() {
        let mut rng = default_rng_seeded(42);
        let n = 100_000;
        let arr = rng.uniform(2.0, 8.0, n).unwrap();
        let slice = arr.as_slice().unwrap();
        let mean: f64 = slice.iter().sum::<f64>() / n as f64;
        let var: f64 = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        // Uniform(a,b): mean = (a+b)/2 = 5.0, var = (b-a)^2/12 = 3.0
        let expected_mean = 5.0;
        let expected_var = 3.0;
        let se_mean = (expected_var / n as f64).sqrt();
        assert!(
            (mean - expected_mean).abs() < 3.0 * se_mean,
            "mean {mean} too far from {expected_mean}"
        );
        // Variance check: use chi-squared-like tolerance
        assert!(
            (var - expected_var).abs() < 0.1,
            "variance {var} too far from {expected_var}"
        );
    }
}
