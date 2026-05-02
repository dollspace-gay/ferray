// ferray-random: Exponential distribution sampling — standard_exponential, exponential

use ferray_core::{Array, FerrayError, IxDyn};

use crate::bitgen::BitGenerator;
use crate::generator::{
    Generator, generate_vec, generate_vec_f32, shape_size, vec_to_array_f32, vec_to_array_f64,
};
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

/// Generate a single standard exponential f32 variate.
///
/// Performed in f64 then cast to avoid f32 log precision loss near 0.
pub(crate) fn standard_exponential_single_f32<B: BitGenerator>(bg: &mut B) -> f32 {
    standard_exponential_single(bg) as f32
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

    /// Fill a pre-allocated `out` buffer with standard exponential
    /// (rate=1) variates (#454).
    ///
    /// Equivalent to `numpy.random.Generator.standard_exponential(out=buffer)`.
    ///
    /// # Errors
    /// `FerrayError::InvalidValue` if `out` is non-contiguous.
    pub fn standard_exponential_into(
        &mut self,
        out: &mut Array<f64, IxDyn>,
    ) -> Result<(), FerrayError> {
        let slice = out.as_slice_mut().ok_or_else(|| {
            FerrayError::invalid_value("standard_exponential_into requires a contiguous out buffer")
        })?;
        for v in slice.iter_mut() {
            *v = standard_exponential_single(&mut self.bg);
        }
        Ok(())
    }

    /// Generate an array of exponential variates with a broadcast
    /// scale parameter (#449).
    ///
    /// `scale` is an array; each output element is sampled from
    /// `Exp(rate = 1/scale[i])`. The output shape equals
    /// `scale.shape()`.
    ///
    /// # Errors
    /// - `FerrayError::InvalidValue` if any `scale` element is `<= 0`.
    pub fn exponential_array(
        &mut self,
        scale: &Array<f64, IxDyn>,
    ) -> Result<Array<f64, IxDyn>, FerrayError> {
        let shape = scale.shape().to_vec();
        let total: usize = shape.iter().product();
        let mut out: Vec<f64> = Vec::with_capacity(total);
        for &s in scale.iter() {
            if s <= 0.0 {
                return Err(FerrayError::invalid_value(format!(
                    "scale must be positive, got {s}"
                )));
            }
            out.push(s * standard_exponential_single(&mut self.bg));
        }
        Array::<f64, IxDyn>::from_vec(IxDyn::new(&shape), out)
    }

    /// Generate an array of standard exponential `f32` variates.
    ///
    /// The f32 analogue of [`standard_exponential`](Self::standard_exponential).
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `shape` is invalid.
    pub fn standard_exponential_f32(
        &mut self,
        size: impl IntoShape,
    ) -> Result<Array<f32, IxDyn>, FerrayError> {
        let shape = size.into_shape()?;
        let n = shape_size(&shape);
        let data = generate_vec_f32(self, n, standard_exponential_single_f32);
        vec_to_array_f32(data, &shape)
    }

    /// Generate an array of `f32` exponential variates with the given scale.
    /// The f32 analogue of [`exponential`](Self::exponential).
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `scale <= 0` or `shape` is invalid.
    pub fn exponential_f32(
        &mut self,
        scale: f32,
        size: impl IntoShape,
    ) -> Result<Array<f32, IxDyn>, FerrayError> {
        if scale <= 0.0 {
            return Err(FerrayError::invalid_value(format!(
                "scale must be positive, got {scale}"
            )));
        }
        let shape = size.into_shape()?;
        let n = shape_size(&shape);
        let data = generate_vec_f32(self, n, |bg| scale * standard_exponential_single_f32(bg));
        vec_to_array_f32(data, &shape)
    }
}

#[cfg(test)]
mod tests {
    use crate::default_rng_seeded;

    // ---- standard_exponential_into (#454) ------------------------------

    #[test]
    fn standard_exponential_into_matches_allocating_version() {
        use ferray_core::{Array, IxDyn};
        let mut a = default_rng_seeded(42);
        let mut b = default_rng_seeded(42);
        let allocated = a.standard_exponential([7, 3]).unwrap();
        let mut buf = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[7, 3]), vec![0.0; 21]).unwrap();
        b.standard_exponential_into(&mut buf).unwrap();
        assert_eq!(allocated.as_slice().unwrap(), buf.as_slice().unwrap());
    }

    #[test]
    fn standard_exponential_into_values_positive() {
        use ferray_core::{Array, IxDyn};
        let mut rng = default_rng_seeded(123);
        let mut buf = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[1000]), vec![-1.0; 1000]).unwrap();
        rng.standard_exponential_into(&mut buf).unwrap();
        for &v in buf.as_slice().unwrap() {
            assert!(v > 0.0);
        }
    }

    // ---- exponential_array (#449) --------------------------------------

    #[test]
    fn exponential_array_shape_matches_scale() {
        use ferray_core::{Array, IxDyn};
        let mut rng = default_rng_seeded(42);
        let scale =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![1.0, 2.0, 0.5, 4.0, 1.5, 3.0])
                .unwrap();
        let out = rng.exponential_array(&scale).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        for &v in out.as_slice().unwrap() {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn exponential_array_per_element_mean() {
        use ferray_core::{Array, IxDyn};
        // Generate many trials with the same per-element scale and
        // confirm the per-element empirical mean ≈ scale[i].
        let mut rng = default_rng_seeded(7);
        let scales = [0.5_f64, 2.0, 5.0];
        let scale = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[3]), scales.to_vec()).unwrap();
        let n_trials = 20_000;
        let mut sums = [0.0_f64; 3];
        for _ in 0..n_trials {
            let out = rng.exponential_array(&scale).unwrap();
            let s = out.as_slice().unwrap();
            for j in 0..3 {
                sums[j] += s[j];
            }
        }
        for j in 0..3 {
            let m = sums[j] / n_trials as f64;
            // Exp(scale) has variance scale^2; SE for n_trials draws is scale / sqrt(n).
            let se = scales[j] / (n_trials as f64).sqrt();
            assert!(
                (m - scales[j]).abs() < 4.0 * se,
                "element {j}: mean {m} too far from {}",
                scales[j]
            );
        }
    }

    #[test]
    fn exponential_array_bad_scale_errors() {
        use ferray_core::{Array, IxDyn};
        let mut rng = default_rng_seeded(0);
        let scale = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2]), vec![1.0, -1.0]).unwrap();
        assert!(rng.exponential_array(&scale).is_err());
    }

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

    // ---------------------------------------------------------------
    // f32 variants (issue #441)
    // ---------------------------------------------------------------

    #[test]
    fn standard_exponential_f32_positive() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.standard_exponential_f32(10_000).unwrap();
        for &v in arr.as_slice().unwrap() {
            assert!(
                v > 0.0,
                "standard_exponential_f32 produced non-positive: {v}"
            );
        }
    }

    #[test]
    fn standard_exponential_f32_mean() {
        let mut rng = default_rng_seeded(42);
        let n = 100_000usize;
        let arr = rng.standard_exponential_f32(n).unwrap();
        let slice = arr.as_slice().unwrap();
        let mean: f64 = slice.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        assert!(
            (mean - 1.0).abs() < 0.02,
            "f32 exp mean {mean} too far from 1"
        );
    }

    #[test]
    fn exponential_f32_mean() {
        let mut rng = default_rng_seeded(42);
        let n = 100_000usize;
        let scale = 3.0f32;
        let arr = rng.exponential_f32(scale, n).unwrap();
        let slice = arr.as_slice().unwrap();
        let mean: f64 = slice.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        assert!(
            (mean - scale as f64).abs() < 0.1,
            "exponential_f32 mean {mean} too far from {scale}"
        );
    }

    #[test]
    fn exponential_f32_bad_scale() {
        let mut rng = default_rng_seeded(42);
        assert!(rng.exponential_f32(0.0, 100).is_err());
        assert!(rng.exponential_f32(-1.0, 100).is_err());
    }

    #[test]
    fn exponential_f32_deterministic() {
        let mut rng1 = default_rng_seeded(42);
        let mut rng2 = default_rng_seeded(42);
        let a = rng1.exponential_f32(2.0, 100).unwrap();
        let b = rng2.exponential_f32(2.0, 100).unwrap();
        assert_eq!(a.as_slice().unwrap(), b.as_slice().unwrap());
    }
}
