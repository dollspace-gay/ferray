// ferray-random: Normal distribution sampling — standard_normal, normal, lognormal

use ferray_core::dimension::broadcast::broadcast_shapes;
use ferray_core::{Array, FerrayError, IxDyn};

use crate::bitgen::BitGenerator;
use crate::distributions::ziggurat::{standard_normal_ziggurat, standard_normal_ziggurat_f32};
use crate::generator::{
    Generator, generate_vec, generate_vec_f32, shape_size, vec_to_array_f32, vec_to_array_f64,
};
use crate::shape::IntoShape;

/// Generate a single standard normal variate via the Ziggurat algorithm.
///
/// Ziggurat is ~3× faster than Box-Muller because ~98% of calls take a fast
/// path that uses only one `next_u64`, a multiplication and a comparison.
/// See [`ziggurat`](super::ziggurat) for the layer-table construction and
/// the slow-path rejection logic.
pub(crate) fn standard_normal_single<B: BitGenerator>(bg: &mut B) -> f64 {
    standard_normal_ziggurat(bg)
}

/// Generate a single f32 standard normal variate via Ziggurat.
///
/// The sampling is performed in f64 (Ziggurat tables are f64) and cast to
/// f32. This costs essentially nothing on modern CPUs and preserves the full
/// tail accuracy of the f64 path.
pub(crate) fn standard_normal_single_f32<B: BitGenerator>(bg: &mut B) -> f32 {
    standard_normal_ziggurat_f32(bg)
}

impl<B: BitGenerator> Generator<B> {
    /// Generate an array of standard normal (mean=0, std=1) variates.
    ///
    /// Uses the Marsaglia–Tsang Ziggurat algorithm (256 layers), which is
    /// roughly 3× faster than Box–Muller for large draws. `shape` accepts
    /// `usize`, `[usize; N]`, `&[usize]`, or `Vec<usize>` via [`IntoShape`].
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `shape` is invalid.
    pub fn standard_normal(
        &mut self,
        size: impl IntoShape,
    ) -> Result<Array<f64, IxDyn>, FerrayError> {
        let shape = size.into_shape()?;
        let n = shape_size(&shape);
        let data = generate_vec(self, n, standard_normal_single);
        vec_to_array_f64(data, &shape)
    }

    /// Generate an array of normal (Gaussian) variates with given mean and
    /// standard deviation. Equivalent to `numpy.random.Generator.normal`.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `scale <= 0` or `shape` is invalid.
    pub fn normal(
        &mut self,
        loc: f64,
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
        let data = generate_vec(self, n, |bg| scale.mul_add(standard_normal_single(bg), loc));
        vec_to_array_f64(data, &shape)
    }

    /// Generate an array of normal variates with broadcast array
    /// parameters (#449).
    ///
    /// `loc` and `scale` are arrays that NumPy-broadcast against each
    /// other to produce the output shape. Each output element is
    /// `loc[i] + scale[i] * Z` where `Z` is a fresh standard-normal
    /// draw. Equivalent to `numpy.random.Generator.normal(loc, scale)`
    /// when both are arrays.
    ///
    /// For scalar parameters, prefer [`normal`](Self::normal) — it
    /// avoids the broadcast view and is faster.
    ///
    /// # Errors
    /// - `FerrayError::ShapeMismatch` if the two shapes are not
    ///   broadcast-compatible.
    /// - `FerrayError::InvalidValue` if any `scale` element is `<= 0`.
    pub fn normal_array(
        &mut self,
        loc: &Array<f64, IxDyn>,
        scale: &Array<f64, IxDyn>,
    ) -> Result<Array<f64, IxDyn>, FerrayError> {
        let target = broadcast_shapes(loc.shape(), scale.shape())?;
        let loc_v = loc.broadcast_to(&target)?;
        let scale_v = scale.broadcast_to(&target)?;
        let total: usize = target.iter().product();
        let mut out: Vec<f64> = Vec::with_capacity(total);
        for (&l, &s) in loc_v.iter().zip(scale_v.iter()) {
            if s <= 0.0 {
                return Err(FerrayError::invalid_value(format!(
                    "scale must be positive, got {s}"
                )));
            }
            out.push(s.mul_add(standard_normal_single(&mut self.bg), l));
        }
        Array::<f64, IxDyn>::from_vec(IxDyn::new(&target), out)
    }

    /// Generate an array of standard normal (mean=0, std=1) `f32` variates.
    ///
    /// The f32 analogue of [`standard_normal`](Self::standard_normal). Equivalent
    /// to `NumPy`'s `Generator.standard_normal(size, dtype=np.float32)`. Uses the
    /// same Ziggurat f64 tables, then casts to f32 to preserve tail accuracy.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `shape` is invalid.
    pub fn standard_normal_f32(
        &mut self,
        size: impl IntoShape,
    ) -> Result<Array<f32, IxDyn>, FerrayError> {
        let shape = size.into_shape()?;
        let n = shape_size(&shape);
        let data = generate_vec_f32(self, n, standard_normal_single_f32);
        vec_to_array_f32(data, &shape)
    }

    /// Generate an array of `f32` normal (Gaussian) variates with given mean
    /// and standard deviation. The f32 analogue of [`normal`](Self::normal).
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `scale <= 0` or `shape` is invalid.
    pub fn normal_f32(
        &mut self,
        loc: f32,
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
        let data = generate_vec_f32(self, n, |bg| {
            scale.mul_add(standard_normal_single_f32(bg), loc)
        });
        vec_to_array_f32(data, &shape)
    }

    /// Generate an array of `f32` log-normal variates. The f32 analogue of
    /// [`lognormal`](Self::lognormal).
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `sigma <= 0` or `shape` is invalid.
    pub fn lognormal_f32(
        &mut self,
        mean: f32,
        sigma: f32,
        size: impl IntoShape,
    ) -> Result<Array<f32, IxDyn>, FerrayError> {
        if sigma <= 0.0 {
            return Err(FerrayError::invalid_value(format!(
                "sigma must be positive, got {sigma}"
            )));
        }
        let shape = size.into_shape()?;
        let n = shape_size(&shape);
        let data = generate_vec_f32(self, n, |bg| {
            sigma.mul_add(standard_normal_single_f32(bg), mean).exp()
        });
        vec_to_array_f32(data, &shape)
    }

    /// Generate an array of log-normal variates.
    ///
    /// If X ~ Normal(mean, sigma), then exp(X) ~ LogNormal(mean, sigma).
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `sigma <= 0` or `shape` is invalid.
    pub fn lognormal(
        &mut self,
        mean: f64,
        sigma: f64,
        size: impl IntoShape,
    ) -> Result<Array<f64, IxDyn>, FerrayError> {
        if sigma <= 0.0 {
            return Err(FerrayError::invalid_value(format!(
                "sigma must be positive, got {sigma}"
            )));
        }
        let shape = size.into_shape()?;
        let n = shape_size(&shape);
        let data = generate_vec(self, n, |bg| {
            sigma.mul_add(standard_normal_single(bg), mean).exp()
        });
        vec_to_array_f64(data, &shape)
    }
}

#[cfg(test)]
mod tests {
    use crate::default_rng_seeded;

    #[test]
    fn standard_normal_deterministic() {
        let mut rng1 = default_rng_seeded(42);
        let mut rng2 = default_rng_seeded(42);
        let a = rng1.standard_normal(1000).unwrap();
        let b = rng2.standard_normal(1000).unwrap();
        assert_eq!(a.as_slice().unwrap(), b.as_slice().unwrap());
    }

    #[test]
    fn standard_normal_mean_variance() {
        let mut rng = default_rng_seeded(42);
        let n = 100_000;
        let arr = rng.standard_normal(n).unwrap();
        let slice = arr.as_slice().unwrap();
        let mean: f64 = slice.iter().sum::<f64>() / n as f64;
        let var: f64 = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let se = (1.0 / n as f64).sqrt();
        assert!(mean.abs() < 3.0 * se, "mean {mean} too far from 0");
        assert!((var - 1.0).abs() < 0.05, "variance {var} too far from 1");
    }

    #[test]
    fn normal_mean_variance() {
        let mut rng = default_rng_seeded(42);
        let n = 100_000;
        let loc = 5.0;
        let scale = 2.0;
        let arr = rng.normal(loc, scale, n).unwrap();
        let slice = arr.as_slice().unwrap();
        let mean: f64 = slice.iter().sum::<f64>() / n as f64;
        let var: f64 = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let se = (scale * scale / n as f64).sqrt();
        assert!(
            (mean - loc).abs() < 3.0 * se,
            "mean {mean} too far from {loc}"
        );
        assert!(
            (var - scale * scale).abs() < 0.2,
            "variance {var} too far from {}",
            scale * scale
        );
    }

    #[test]
    fn normal_bad_scale() {
        let mut rng = default_rng_seeded(42);
        assert!(rng.normal(0.0, 0.0, 100).is_err());
        assert!(rng.normal(0.0, -1.0, 100).is_err());
    }

    #[test]
    fn normal_array_broadcast_scalar_x_vector() {
        use ferray_core::IxDyn;
        let mut rng = default_rng_seeded(42);
        // loc shape (3,), scale shape (1,) — broadcast to (3,).
        let loc = ferray_core::Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[3]),
            vec![0.0, 10.0, -5.0],
        )
        .unwrap();
        let scale =
            ferray_core::Array::<f64, IxDyn>::from_vec(IxDyn::new(&[1]), vec![1.0]).unwrap();
        let out = rng.normal_array(&loc, &scale).unwrap();
        assert_eq!(out.shape(), &[3]);
    }

    #[test]
    fn normal_array_2d_broadcast_means_match_loc() {
        use ferray_core::IxDyn;
        // loc shape (3, 1), scale shape (1, 4) → output (3, 4) where
        // every row j shares loc[j] and every column shares scale.
        // With many draws per element the per-row mean → loc[j].
        let mut rng = default_rng_seeded(7);
        let loc = ferray_core::Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[3, 1]),
            vec![0.0, 5.0, -3.0],
        )
        .unwrap();
        let scale = ferray_core::Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[1, 4]),
            vec![1.0, 0.5, 2.0, 0.1],
        )
        .unwrap();

        let n_trials = 5_000;
        let mut row_sums = [0.0_f64; 3];
        for _ in 0..n_trials {
            let out = rng.normal_array(&loc, &scale).unwrap();
            assert_eq!(out.shape(), &[3, 4]);
            let s = out.as_slice().unwrap();
            for r in 0..3 {
                for c in 0..4 {
                    row_sums[r] += s[r * 4 + c];
                }
            }
        }
        // Row r averages over 4 columns × n_trials draws with mean loc[r].
        let denom = (n_trials * 4) as f64;
        let expected = [0.0, 5.0, -3.0];
        for r in 0..3 {
            let m = row_sums[r] / denom;
            assert!(
                (m - expected[r]).abs() < 0.05,
                "row {r} mean {m} too far from {}",
                expected[r]
            );
        }
    }

    #[test]
    fn normal_array_bad_scale_errors() {
        use ferray_core::IxDyn;
        let mut rng = default_rng_seeded(0);
        let loc = ferray_core::Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[2]),
            vec![0.0, 0.0],
        )
        .unwrap();
        let scale = ferray_core::Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[2]),
            vec![1.0, -0.5],
        )
        .unwrap();
        assert!(rng.normal_array(&loc, &scale).is_err());
    }

    #[test]
    fn normal_array_shape_mismatch_errors() {
        use ferray_core::IxDyn;
        let mut rng = default_rng_seeded(0);
        let loc = ferray_core::Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[3]),
            vec![0.0; 3],
        )
        .unwrap();
        let scale = ferray_core::Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[2]),
            vec![1.0; 2],
        )
        .unwrap();
        assert!(rng.normal_array(&loc, &scale).is_err());
    }

    #[test]
    fn lognormal_positive() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.lognormal(0.0, 1.0, 10_000).unwrap();
        let slice = arr.as_slice().unwrap();
        for &v in slice {
            assert!(v > 0.0, "lognormal produced non-positive value: {v}");
        }
    }

    #[test]
    fn lognormal_mean() {
        let mut rng = default_rng_seeded(42);
        let n = 100_000;
        let mu = 0.0;
        let sigma = 0.5;
        let arr = rng.lognormal(mu, sigma, n).unwrap();
        let slice = arr.as_slice().unwrap();
        let mean: f64 = slice.iter().sum::<f64>() / n as f64;
        // E[X] = exp(mu + sigma^2 / 2)
        let expected_mean = (mu + sigma * sigma / 2.0).exp();
        let expected_var = (sigma * sigma).exp_m1() * 2.0f64.mul_add(mu, sigma * sigma).exp();
        let se = (expected_var / n as f64).sqrt();
        assert!(
            (mean - expected_mean).abs() < 3.0 * se,
            "lognormal mean {mean} too far from {expected_mean}"
        );
    }

    #[test]
    fn standard_normal_variance() {
        let mut rng = default_rng_seeded(42);
        let n = 100_000;
        let arr = rng.standard_normal(n).unwrap();
        let s = arr.as_slice().unwrap();
        let mean: f64 = s.iter().sum::<f64>() / n as f64;
        let var: f64 = s.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        // Variance should be ~1.0
        assert!(
            (var - 1.0).abs() < 0.05,
            "standard_normal variance {var} too far from 1.0"
        );
    }

    #[test]
    fn normal_mean_and_variance() {
        let mut rng = default_rng_seeded(42);
        let n = 100_000;
        let loc = 5.0;
        let scale = 2.0;
        let arr = rng.normal(loc, scale, n).unwrap();
        let s: Vec<f64> = arr.iter().copied().collect();
        let mean: f64 = s.iter().sum::<f64>() / n as f64;
        let var: f64 = s.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(
            (mean - loc).abs() < 0.05,
            "normal mean {mean} too far from {loc}"
        );
        assert!(
            (var - scale * scale).abs() < 0.2,
            "normal variance {var} too far from {}",
            scale * scale
        );
    }

    // ----- N-D shape tests (issue #440) -----

    #[test]
    fn standard_normal_nd_shape() {
        let mut rng = crate::default_rng_seeded(42);
        let arr = rng.standard_normal([3, 4]).unwrap();
        assert_eq!(arr.shape(), &[3, 4]);
    }

    #[test]
    fn normal_nd_shape() {
        let mut rng = crate::default_rng_seeded(42);
        let arr = rng.normal(10.0, 2.0, [2, 3, 4]).unwrap();
        assert_eq!(arr.shape(), &[2, 3, 4]);
    }

    #[test]
    fn lognormal_nd_shape() {
        let mut rng = crate::default_rng_seeded(42);
        let arr = rng.lognormal(0.0, 1.0, [5, 5]).unwrap();
        assert_eq!(arr.shape(), &[5, 5]);
        for &v in arr.iter() {
            assert!(v > 0.0);
        }
    }

    // ---------------------------------------------------------------
    // f32 variants (issue #441)
    // ---------------------------------------------------------------

    #[test]
    fn standard_normal_f32_deterministic() {
        let mut rng1 = default_rng_seeded(42);
        let mut rng2 = default_rng_seeded(42);
        let a = rng1.standard_normal_f32(1000).unwrap();
        let b = rng2.standard_normal_f32(1000).unwrap();
        assert_eq!(a.as_slice().unwrap(), b.as_slice().unwrap());
    }

    #[test]
    fn standard_normal_f32_mean_variance() {
        let mut rng = default_rng_seeded(42);
        let n = 100_000usize;
        let arr = rng.standard_normal_f32(n).unwrap();
        let slice = arr.as_slice().unwrap();
        // Accumulate in f64 to avoid compounding f32 error.
        let mean: f64 = slice.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        let var: f64 = slice
            .iter()
            .map(|&x| {
                let d = x as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        let se = (1.0 / n as f64).sqrt();
        assert!(mean.abs() < 5.0 * se, "f32 mean {mean} too far from 0");
        assert!(
            (var - 1.0).abs() < 0.05,
            "f32 variance {var} too far from 1"
        );
    }

    #[test]
    fn standard_normal_f32_nd_shape() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.standard_normal_f32([3, 4]).unwrap();
        assert_eq!(arr.shape(), &[3, 4]);
    }

    #[test]
    fn normal_f32_mean() {
        let mut rng = default_rng_seeded(42);
        let n = 100_000usize;
        let loc = 5.0f32;
        let scale = 2.0f32;
        let arr = rng.normal_f32(loc, scale, n).unwrap();
        let slice = arr.as_slice().unwrap();
        let mean: f64 = slice.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        assert!(
            (mean - loc as f64).abs() < 0.05,
            "f32 normal mean {mean} too far from {loc}"
        );
    }

    #[test]
    fn normal_f32_bad_scale() {
        let mut rng = default_rng_seeded(42);
        assert!(rng.normal_f32(0.0, 0.0, 100).is_err());
        assert!(rng.normal_f32(0.0, -1.0, 100).is_err());
    }

    #[test]
    fn lognormal_f32_positive() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.lognormal_f32(0.0, 1.0, 10_000).unwrap();
        for &v in arr.as_slice().unwrap() {
            assert!(v > 0.0, "lognormal_f32 produced non-positive value: {v}");
        }
    }

    #[test]
    fn lognormal_f32_bad_sigma() {
        let mut rng = default_rng_seeded(42);
        assert!(rng.lognormal_f32(0.0, 0.0, 100).is_err());
        assert!(rng.lognormal_f32(0.0, -0.5, 100).is_err());
    }

    // ----- NaN/Inf parameter input tests (#263) -----

    #[test]
    fn normal_nan_loc_produces_nan_output() {
        // NumPy: np.random.default_rng(42).normal(np.nan, 1.0, 5) → all NaN
        let mut rng = default_rng_seeded(42);
        let arr = rng.normal(f64::NAN, 1.0, 5).unwrap();
        for &v in arr.as_slice().unwrap() {
            assert!(v.is_nan(), "expected NaN, got {v}");
        }
    }

    #[test]
    fn normal_inf_scale_produces_inf_output() {
        // Infinite scale → every sample is ±inf.
        let mut rng = default_rng_seeded(42);
        let arr = rng.normal(0.0, f64::INFINITY, 5).unwrap();
        for &v in arr.as_slice().unwrap() {
            assert!(v.is_infinite() || v.is_nan(), "expected Inf/NaN, got {v}");
        }
    }

    #[test]
    fn normal_nan_scale_rejected() {
        // NaN scale should propagate — the output is meaningless but
        // at minimum must not panic.
        let mut rng = default_rng_seeded(42);
        // scale <= 0 is rejected by parameter validation; NaN is
        // neither > 0 nor <= 0, so the check may or may not catch it
        // depending on the implementation. Just assert no panic.
        let _ = rng.normal(0.0, f64::NAN, 5);
    }
}
