// ferray-random: Normal distribution sampling — standard_normal, normal, lognormal

use ferray_core::{Array, FerrayError, IxDyn};

use crate::bitgen::BitGenerator;
use crate::generator::{Generator, generate_vec, shape_size, vec_to_array_f64};
use crate::shape::IntoShape;

/// Generate a single standard normal variate using the Box-Muller transform.
///
/// Consumes two uniform [0,1) variates and produces two normal variates.
/// We use both and cache the second, but for simplicity here we just use
/// the Ziggurat-free approach generating one pair at a time.
pub(crate) fn standard_normal_pair<B: BitGenerator>(bg: &mut B) -> (f64, f64) {
    loop {
        let u1 = bg.next_f64();
        let u2 = bg.next_f64();
        // Avoid log(0)
        if u1 < f64::EPSILON {
            continue;
        }
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        return (r * theta.cos(), r * theta.sin());
    }
}

/// Generate a single standard normal variate.
pub(crate) fn standard_normal_single<B: BitGenerator>(bg: &mut B) -> f64 {
    standard_normal_pair(bg).0
}

impl<B: BitGenerator> Generator<B> {
    /// Generate an array of standard normal (mean=0, std=1) variates.
    ///
    /// Uses the Box-Muller transform. `shape` accepts `usize`, `[usize; N]`,
    /// `&[usize]`, or `Vec<usize>` via [`IntoShape`].
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `shape` is invalid.
    pub fn standard_normal(
        &mut self,
        size: impl IntoShape,
    ) -> Result<Array<f64, IxDyn>, FerrayError> {
        let shape = size.into_shape()?;
        let n = shape_size(&shape);
        let mut data = Vec::with_capacity(n);
        while data.len() < n {
            let (a, b) = standard_normal_pair(&mut self.bg);
            data.push(a);
            if data.len() < n {
                data.push(b);
            }
        }
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
        let data = generate_vec(self, n, |bg| loc + scale * standard_normal_single(bg));
        vec_to_array_f64(data, &shape)
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
            (mean + sigma * standard_normal_single(bg)).exp()
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
        let expected_var = ((sigma * sigma).exp() - 1.0) * (2.0 * mu + sigma * sigma).exp();
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
}
