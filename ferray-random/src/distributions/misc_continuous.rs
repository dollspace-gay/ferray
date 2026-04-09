// ferray-random: Miscellaneous continuous distributions
//
// laplace, logistic, rayleigh, weibull, pareto, gumbel, power, triangular,
// vonmises, wald, standard_cauchy

use ferray_core::{Array, FerrayError, IxDyn};

use crate::bitgen::BitGenerator;
use crate::distributions::exponential::standard_exponential_single;
use crate::distributions::normal::standard_normal_single;
use crate::generator::{Generator, generate_vec, shape_size, vec_to_array_f64};
use crate::shape::IntoShape;

impl<B: BitGenerator> Generator<B> {
    /// Generate an array of Laplace-distributed variates.
    ///
    /// PDF: f(x) = (1/(2*scale)) * exp(-|x - loc| / scale).
    ///
    /// # Arguments
    /// * `loc` - Location parameter.
    /// * `scale` - Scale parameter, must be positive.
    /// * `size` - Number of values to generate.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `scale <= 0` or `size` is zero.
    pub fn laplace(
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
        let shape_vec = size.into_shape()?;
        let n = shape_size(&shape_vec);
        let data = generate_vec(self, n, |bg| {
            let mut u = bg.next_f64() - 0.5;
            // Reject u.abs() >= 0.5 to avoid ln(0) = -inf
            while u.abs() >= 0.5 {
                u = bg.next_f64() - 0.5;
            }
            loc - scale * u.signum() * (1.0 - 2.0 * u.abs()).ln()
        });
        vec_to_array_f64(data, &shape_vec)
    }

    /// Generate an array of logistic-distributed variates.
    ///
    /// Uses inverse CDF: loc + scale * ln(u / (1 - u)).
    ///
    /// # Arguments
    /// * `loc` - Location parameter.
    /// * `scale` - Scale parameter, must be positive.
    /// * `size` - Number of values to generate.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `scale <= 0` or `size` is zero.
    pub fn logistic(
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
        let shape_vec = size.into_shape()?;
        let n = shape_size(&shape_vec);
        let data = generate_vec(self, n, |bg| {
            loop {
                let u = bg.next_f64();
                if u > f64::EPSILON && u < 1.0 - f64::EPSILON {
                    return loc + scale * (u / (1.0 - u)).ln();
                }
            }
        });
        vec_to_array_f64(data, &shape_vec)
    }

    /// Generate an array of Rayleigh-distributed variates.
    ///
    /// Uses inverse CDF: scale * sqrt(-2 * ln(1-u)).
    ///
    /// # Arguments
    /// * `scale` - Scale parameter, must be positive.
    /// * `size` - Number of values to generate.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `scale <= 0` or `size` is zero.
    pub fn rayleigh(
        &mut self,
        scale: f64,
        size: impl IntoShape,
    ) -> Result<Array<f64, IxDyn>, FerrayError> {
        if scale <= 0.0 {
            return Err(FerrayError::invalid_value(format!(
                "scale must be positive, got {scale}"
            )));
        }
        let shape_vec = size.into_shape()?;
        let n = shape_size(&shape_vec);
        let data = generate_vec(self, n, |bg| {
            scale * (2.0 * standard_exponential_single(bg)).sqrt()
        });
        vec_to_array_f64(data, &shape_vec)
    }

    /// Generate an array of Weibull-distributed variates.
    ///
    /// Uses inverse CDF: (-ln(1-u))^(1/a).
    ///
    /// # Arguments
    /// * `a` - Shape parameter, must be positive.
    /// * `size` - Number of values to generate.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `a <= 0` or `size` is zero.
    pub fn weibull(
        &mut self,
        a: f64,
        size: impl IntoShape,
    ) -> Result<Array<f64, IxDyn>, FerrayError> {
        if a <= 0.0 {
            return Err(FerrayError::invalid_value(format!(
                "a must be positive, got {a}"
            )));
        }
        let shape_vec = size.into_shape()?;
        let n = shape_size(&shape_vec);
        let data = generate_vec(self, n, |bg| standard_exponential_single(bg).powf(1.0 / a));
        vec_to_array_f64(data, &shape_vec)
    }

    /// Generate an array of Pareto (type II / Lomax) distributed variates.
    ///
    /// Uses inverse CDF: (1-u)^(-1/a) - 1 (then shifted by 1 to match NumPy).
    /// NumPy's Pareto: samples from Pareto(a) with x_m=1, so PDF = a / x^(a+1) for x >= 1.
    ///
    /// # Arguments
    /// * `a` - Shape parameter, must be positive.
    /// * `size` - Number of values to generate.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `a <= 0` or `size` is zero.
    pub fn pareto(
        &mut self,
        a: f64,
        size: impl IntoShape,
    ) -> Result<Array<f64, IxDyn>, FerrayError> {
        if a <= 0.0 {
            return Err(FerrayError::invalid_value(format!(
                "a must be positive, got {a}"
            )));
        }
        let shape_vec = size.into_shape()?;
        let n = shape_size(&shape_vec);
        let data = generate_vec(self, n, |bg| {
            let e = standard_exponential_single(bg);
            (e / a).exp() - 1.0
        });
        vec_to_array_f64(data, &shape_vec)
    }

    /// Generate an array of Gumbel-distributed variates.
    ///
    /// Uses inverse CDF: loc - scale * ln(-ln(u)).
    ///
    /// # Arguments
    /// * `loc` - Location parameter.
    /// * `scale` - Scale parameter, must be positive.
    /// * `size` - Number of values to generate.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `scale <= 0` or `size` is zero.
    pub fn gumbel(
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
        let shape_vec = size.into_shape()?;
        let n = shape_size(&shape_vec);
        let data = generate_vec(self, n, |bg| {
            loop {
                let u = bg.next_f64();
                if u > f64::EPSILON && u < 1.0 - f64::EPSILON {
                    return loc - scale * (-u.ln()).ln();
                }
            }
        });
        vec_to_array_f64(data, &shape_vec)
    }

    /// Generate an array of power-distributed variates.
    ///
    /// Power distribution with shape `a` on [0, 1]:
    /// PDF: a * x^(a-1), CDF: x^a. Inverse CDF: u^(1/a).
    ///
    /// # Arguments
    /// * `a` - Shape parameter, must be positive.
    /// * `size` - Number of values to generate.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `a <= 0` or `size` is zero.
    pub fn power(
        &mut self,
        a: f64,
        size: impl IntoShape,
    ) -> Result<Array<f64, IxDyn>, FerrayError> {
        if a <= 0.0 {
            return Err(FerrayError::invalid_value(format!(
                "a must be positive, got {a}"
            )));
        }
        let shape_vec = size.into_shape()?;
        let n = shape_size(&shape_vec);
        let data = generate_vec(self, n, |bg| {
            let e = standard_exponential_single(bg);
            // 1 - exp(-e) gives a U(0,1), then raise to 1/a
            // More precisely: u^(1/a) where u ~ U(0,1)
            // Using exponential: (1 - exp(-e))^(1/a)
            (1.0 - (-e).exp()).powf(1.0 / a)
        });
        vec_to_array_f64(data, &shape_vec)
    }

    /// Generate an array of triangular-distributed variates.
    ///
    /// # Arguments
    /// * `left` - Lower limit.
    /// * `mode` - Mode (peak), must be in `[left, right]`.
    /// * `right` - Upper limit, must be > `left`.
    /// * `size` - Number of values to generate.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if parameters are invalid or `size` is zero.
    pub fn triangular(
        &mut self,
        left: f64,
        mode: f64,
        right: f64,
        size: impl IntoShape,
    ) -> Result<Array<f64, IxDyn>, FerrayError> {
        if left >= right {
            return Err(FerrayError::invalid_value(format!(
                "left ({left}) must be less than right ({right})"
            )));
        }
        if mode < left || mode > right {
            return Err(FerrayError::invalid_value(format!(
                "mode ({mode}) must be in [{left}, {right}]"
            )));
        }
        let fc = (mode - left) / (right - left);
        let shape_vec = size.into_shape()?;
        let n = shape_size(&shape_vec);
        let data = generate_vec(self, n, |bg| {
            let u = bg.next_f64();
            if u < fc {
                left + ((right - left) * (mode - left) * u).sqrt()
            } else {
                right - ((right - left) * (right - mode) * (1.0 - u)).sqrt()
            }
        });
        vec_to_array_f64(data, &shape_vec)
    }

    /// Generate an array of von Mises distributed variates.
    ///
    /// The von Mises distribution is a continuous distribution on the circle.
    /// Uses Best & Fisher's algorithm.
    ///
    /// # Arguments
    /// * `mu` - Mean direction (in radians).
    /// * `kappa` - Concentration parameter, must be non-negative.
    /// * `size` - Number of values to generate.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `kappa < 0` or `size` is zero.
    pub fn vonmises(
        &mut self,
        mu: f64,
        kappa: f64,
        size: impl IntoShape,
    ) -> Result<Array<f64, IxDyn>, FerrayError> {
        if kappa < 0.0 {
            return Err(FerrayError::invalid_value(format!(
                "kappa must be non-negative, got {kappa}"
            )));
        }
        let shape_vec = size.into_shape()?;
        let n = shape_size(&shape_vec);

        if kappa < 1e-6 {
            // For very small kappa, the distribution is nearly uniform on [-pi, pi)
            let data = generate_vec(self, n, |bg| {
                mu + (bg.next_f64() * std::f64::consts::TAU - std::f64::consts::PI)
            });
            return vec_to_array_f64(data, &shape_vec);
        }

        // Best & Fisher algorithm
        let tau = 1.0 + (1.0 + 4.0 * kappa * kappa).sqrt();
        let rho = (tau - (2.0 * tau).sqrt()) / (2.0 * kappa);
        let r = (1.0 + rho * rho) / (2.0 * rho);

        let data = generate_vec(self, n, |bg| {
            loop {
                let u1 = bg.next_f64();
                let z = (std::f64::consts::TAU * u1 - std::f64::consts::PI).cos();
                let f_val = (1.0 + r * z) / (r + z);
                let c = kappa * (r - f_val);
                let u2 = bg.next_f64();
                if u2 < c * (2.0 - c) || u2 <= c * (-c).exp() {
                    let u3 = bg.next_f64();
                    let theta = if u3 > 0.5 {
                        mu + f_val.acos()
                    } else {
                        mu - f_val.acos()
                    };
                    return theta;
                }
            }
        });
        vec_to_array_f64(data, &shape_vec)
    }

    /// Generate an array of Wald (inverse Gaussian) distributed variates.
    ///
    /// # Arguments
    /// * `mean` - Mean of the distribution, must be positive.
    /// * `scale` - Scale parameter (lambda), must be positive.
    /// * `size` - Number of values to generate.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `mean <= 0`, `scale <= 0`, or `size` is zero.
    pub fn wald(
        &mut self,
        mean: f64,
        scale: f64,
        size: impl IntoShape,
    ) -> Result<Array<f64, IxDyn>, FerrayError> {
        if mean <= 0.0 {
            return Err(FerrayError::invalid_value(format!(
                "mean must be positive, got {mean}"
            )));
        }
        if scale <= 0.0 {
            return Err(FerrayError::invalid_value(format!(
                "scale must be positive, got {scale}"
            )));
        }
        // Michael, Schucany & Haas algorithm
        let shape_vec = size.into_shape()?;
        let n = shape_size(&shape_vec);
        let data = generate_vec(self, n, |bg| {
            let z = standard_normal_single(bg);
            let v = z * z;
            let mu = mean;
            let lam = scale;
            let x = mu + (mu * mu * v) / (2.0 * lam)
                - (mu / (2.0 * lam)) * (4.0 * mu * lam * v + mu * mu * v * v).sqrt();
            let u = bg.next_f64();
            if u <= mu / (mu + x) { x } else { mu * mu / x }
        });
        vec_to_array_f64(data, &shape_vec)
    }

    /// Generate an array of standard Cauchy distributed variates.
    ///
    /// Uses the inverse CDF: tan(pi * (u - 0.5)).
    ///
    /// # Arguments
    /// * `size` - Number of values to generate.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `size` is zero.
    pub fn standard_cauchy(
        &mut self,
        size: impl IntoShape,
    ) -> Result<Array<f64, IxDyn>, FerrayError> {
        let shape_vec = size.into_shape()?;
        let n = shape_size(&shape_vec);
        let data = generate_vec(self, n, |bg| {
            loop {
                let u = bg.next_f64();
                // Avoid u = 0.5 exactly (tan(0) = 0, which is fine, but avoid edge)
                if (u - 0.5).abs() > f64::EPSILON {
                    return (std::f64::consts::PI * (u - 0.5)).tan();
                }
            }
        });
        vec_to_array_f64(data, &shape_vec)
    }
}

#[cfg(test)]
mod tests {
    use crate::default_rng_seeded;

    #[test]
    fn laplace_mean_variance() {
        let mut rng = default_rng_seeded(42);
        let n = 100_000;
        let loc = 2.0;
        let scale = 3.0;
        let arr = rng.laplace(loc, scale, n).unwrap();
        let slice = arr.as_slice().unwrap();
        let mean: f64 = slice.iter().sum::<f64>() / n as f64;
        let var: f64 = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        // Laplace(loc, scale): mean=loc, var=2*scale^2
        let expected_var = 2.0 * scale * scale;
        let se = (expected_var / n as f64).sqrt();
        assert!(
            (mean - loc).abs() < 3.0 * se,
            "laplace mean {mean} too far from {loc}"
        );
        assert!(
            (var - expected_var).abs() / expected_var < 0.05,
            "laplace variance {var} too far from {expected_var}"
        );
    }

    #[test]
    fn logistic_mean() {
        let mut rng = default_rng_seeded(42);
        let n = 100_000;
        let loc = 1.0;
        let scale = 2.0;
        let arr = rng.logistic(loc, scale, n).unwrap();
        let slice = arr.as_slice().unwrap();
        let mean: f64 = slice.iter().sum::<f64>() / n as f64;
        let expected_var = (std::f64::consts::PI * scale).powi(2) / 3.0;
        let se = (expected_var / n as f64).sqrt();
        assert!(
            (mean - loc).abs() < 3.0 * se,
            "logistic mean {mean} too far from {loc}"
        );
    }

    #[test]
    fn rayleigh_positive() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.rayleigh(2.0, 10_000).unwrap();
        for &v in arr.as_slice().unwrap() {
            assert!(v > 0.0);
        }
    }

    #[test]
    fn rayleigh_mean() {
        let mut rng = default_rng_seeded(42);
        let n = 100_000;
        let scale = 2.0;
        let arr = rng.rayleigh(scale, n).unwrap();
        let slice = arr.as_slice().unwrap();
        let mean: f64 = slice.iter().sum::<f64>() / n as f64;
        // Rayleigh(sigma): mean = sigma * sqrt(pi/2)
        let expected_mean = scale * (std::f64::consts::FRAC_PI_2).sqrt();
        let expected_var = (4.0 - std::f64::consts::PI) / 2.0 * scale * scale;
        let se = (expected_var / n as f64).sqrt();
        assert!(
            (mean - expected_mean).abs() < 3.0 * se,
            "rayleigh mean {mean} too far from {expected_mean}"
        );
    }

    #[test]
    fn weibull_positive() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.weibull(1.5, 10_000).unwrap();
        for &v in arr.as_slice().unwrap() {
            assert!(v > 0.0);
        }
    }

    #[test]
    fn pareto_ge_zero() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.pareto(3.0, 10_000).unwrap();
        for &v in arr.as_slice().unwrap() {
            assert!(v >= 0.0, "pareto value {v} < 0");
        }
    }

    #[test]
    fn gumbel_mean() {
        let mut rng = default_rng_seeded(42);
        let n = 100_000;
        let loc = 1.0;
        let scale = 2.0;
        let arr = rng.gumbel(loc, scale, n).unwrap();
        let slice = arr.as_slice().unwrap();
        let mean: f64 = slice.iter().sum::<f64>() / n as f64;
        // Gumbel: mean = loc + scale * gamma_euler (0.5772...)
        let euler = 0.5772156649015329;
        let expected_mean = loc + scale * euler;
        let expected_var = std::f64::consts::PI * std::f64::consts::PI * scale * scale / 6.0;
        let se = (expected_var / n as f64).sqrt();
        assert!(
            (mean - expected_mean).abs() < 3.0 * se,
            "gumbel mean {mean} too far from {expected_mean}"
        );
    }

    #[test]
    fn power_in_range() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.power(2.0, 10_000).unwrap();
        for &v in arr.as_slice().unwrap() {
            assert!((0.0..=1.0).contains(&v), "power value {v} out of [0,1]");
        }
    }

    #[test]
    fn triangular_in_range() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.triangular(1.0, 3.0, 5.0, 10_000).unwrap();
        for &v in arr.as_slice().unwrap() {
            assert!(
                (1.0..=5.0).contains(&v),
                "triangular value {v} out of [1,5]"
            );
        }
    }

    #[test]
    fn triangular_mean() {
        let mut rng = default_rng_seeded(42);
        let n = 100_000;
        let (left, mode, right) = (1.0, 3.0, 5.0);
        let arr = rng.triangular(left, mode, right, n).unwrap();
        let slice = arr.as_slice().unwrap();
        let mean: f64 = slice.iter().sum::<f64>() / n as f64;
        let expected_mean = (left + mode + right) / 3.0;
        assert!(
            (mean - expected_mean).abs() < 0.05,
            "triangular mean {mean} too far from {expected_mean}"
        );
    }

    #[test]
    fn vonmises_basic() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.vonmises(0.0, 1.0, 10_000).unwrap();
        assert_eq!(arr.shape(), &[10_000]);
    }

    #[test]
    fn wald_positive() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.wald(1.0, 1.0, 10_000).unwrap();
        for &v in arr.as_slice().unwrap() {
            assert!(v > 0.0, "wald value {v} must be positive");
        }
    }

    #[test]
    fn standard_cauchy_basic() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.standard_cauchy(10_000).unwrap();
        assert_eq!(arr.shape(), &[10_000]);
        // Cauchy has no mean or variance, just check it runs
    }

    #[test]
    fn bad_params() {
        let mut rng = default_rng_seeded(42);
        assert!(rng.laplace(0.0, 0.0, 10).is_err());
        assert!(rng.logistic(0.0, -1.0, 10).is_err());
        assert!(rng.rayleigh(0.0, 10).is_err());
        assert!(rng.weibull(0.0, 10).is_err());
        assert!(rng.pareto(0.0, 10).is_err());
        assert!(rng.gumbel(0.0, 0.0, 10).is_err());
        assert!(rng.power(0.0, 10).is_err());
        assert!(rng.triangular(5.0, 3.0, 1.0, 10).is_err());
        assert!(rng.vonmises(0.0, -1.0, 10).is_err());
        assert!(rng.wald(0.0, 1.0, 10).is_err());
        assert!(rng.wald(1.0, 0.0, 10).is_err());
    }
}
