// ferray-random: Gamma family distributions — gamma, beta, chisquare, f, student_t, standard_gamma
//
// Gamma sampling uses Marsaglia & Tsang's method (shape >= 1) with
// Ahrens-Dieter transformation for shape < 1.

use ferray_core::{Array, FerrayError, Ix1};

use crate::bitgen::BitGenerator;
use crate::distributions::normal::standard_normal_single;
use crate::generator::{Generator, generate_vec, vec_to_array1};

/// Generate a single standard gamma variate with shape parameter `alpha`.
///
/// Uses Marsaglia & Tsang's method for alpha >= 1, and
/// the Ahrens-Dieter boost for alpha < 1.
pub(crate) fn standard_gamma_single<B: BitGenerator>(bg: &mut B, alpha: f64) -> f64 {
    if alpha < 1.0 {
        // Ahrens-Dieter: if X ~ Gamma(alpha+1), then X * U^(1/alpha) ~ Gamma(alpha)
        if alpha <= 0.0 {
            return 0.0;
        }
        loop {
            let u = bg.next_f64();
            if u > f64::EPSILON {
                let x = standard_gamma_ge1(bg, alpha + 1.0);
                return x * u.powf(1.0 / alpha);
            }
        }
    } else {
        standard_gamma_ge1(bg, alpha)
    }
}

/// Marsaglia & Tsang's method for Gamma(alpha) with alpha >= 1.
fn standard_gamma_ge1<B: BitGenerator>(bg: &mut B, alpha: f64) -> f64 {
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();

    loop {
        let x = standard_normal_single(bg);
        let v_base = 1.0 + c * x;
        if v_base <= 0.0 {
            continue;
        }
        let v = v_base * v_base * v_base;
        let u = bg.next_f64();
        // Squeeze test
        if u < 1.0 - 0.0331 * (x * x) * (x * x) {
            return d * v;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

impl<B: BitGenerator> Generator<B> {
    /// Generate an array of standard gamma variates with shape `shape`.
    ///
    /// # Arguments
    /// * `shape` - Shape parameter (alpha), must be positive.
    /// * `size` - Number of values to generate.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `shape <= 0` or `size` is zero.
    pub fn standard_gamma(
        &mut self,
        shape: f64,
        size: usize,
    ) -> Result<Array<f64, Ix1>, FerrayError> {
        if size == 0 {
            return Err(FerrayError::invalid_value("size must be > 0"));
        }
        if shape <= 0.0 {
            return Err(FerrayError::invalid_value(format!(
                "shape must be positive, got {shape}"
            )));
        }
        let data = generate_vec(self, size, |bg| standard_gamma_single(bg, shape));
        vec_to_array1(data)
    }

    /// Generate an array of gamma-distributed variates.
    ///
    /// The gamma distribution with shape `shape` and scale `scale` has
    /// PDF: f(x) = x^(shape-1) * exp(-x/scale) / (scale^shape * Gamma(shape)).
    ///
    /// # Arguments
    /// * `shape` - Shape parameter (alpha), must be positive.
    /// * `scale` - Scale parameter (beta), must be positive.
    /// * `size` - Number of values to generate.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `shape <= 0`, `scale <= 0`, or `size` is zero.
    pub fn gamma(
        &mut self,
        shape: f64,
        scale: f64,
        size: usize,
    ) -> Result<Array<f64, Ix1>, FerrayError> {
        if size == 0 {
            return Err(FerrayError::invalid_value("size must be > 0"));
        }
        if shape <= 0.0 {
            return Err(FerrayError::invalid_value(format!(
                "shape must be positive, got {shape}"
            )));
        }
        if scale <= 0.0 {
            return Err(FerrayError::invalid_value(format!(
                "scale must be positive, got {scale}"
            )));
        }
        let data = generate_vec(self, size, |bg| scale * standard_gamma_single(bg, shape));
        vec_to_array1(data)
    }

    /// Generate an array of beta-distributed variates in (0, 1).
    ///
    /// Uses the relationship: if X ~ Gamma(a), Y ~ Gamma(b), then X/(X+Y) ~ Beta(a,b).
    ///
    /// # Arguments
    /// * `a` - First shape parameter, must be positive.
    /// * `b` - Second shape parameter, must be positive.
    /// * `size` - Number of values to generate.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `a <= 0`, `b <= 0`, or `size` is zero.
    pub fn beta(&mut self, a: f64, b: f64, size: usize) -> Result<Array<f64, Ix1>, FerrayError> {
        if size == 0 {
            return Err(FerrayError::invalid_value("size must be > 0"));
        }
        if a <= 0.0 {
            return Err(FerrayError::invalid_value(format!(
                "a must be positive, got {a}"
            )));
        }
        if b <= 0.0 {
            return Err(FerrayError::invalid_value(format!(
                "b must be positive, got {b}"
            )));
        }
        let data = generate_vec(self, size, |bg| {
            let x = standard_gamma_single(bg, a);
            let y = standard_gamma_single(bg, b);
            if x + y == 0.0 {
                0.5 // Degenerate case
            } else {
                x / (x + y)
            }
        });
        vec_to_array1(data)
    }

    /// Generate an array of chi-squared distributed variates.
    ///
    /// Chi-squared(df) = Gamma(df/2, 2).
    ///
    /// # Arguments
    /// * `df` - Degrees of freedom, must be positive.
    /// * `size` - Number of values to generate.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `df <= 0` or `size` is zero.
    pub fn chisquare(&mut self, df: f64, size: usize) -> Result<Array<f64, Ix1>, FerrayError> {
        if size == 0 {
            return Err(FerrayError::invalid_value("size must be > 0"));
        }
        if df <= 0.0 {
            return Err(FerrayError::invalid_value(format!(
                "df must be positive, got {df}"
            )));
        }
        let data = generate_vec(self, size, |bg| 2.0 * standard_gamma_single(bg, df / 2.0));
        vec_to_array1(data)
    }

    /// Generate an array of F-distributed variates.
    ///
    /// F(d1, d2) = (Chi2(d1)/d1) / (Chi2(d2)/d2).
    ///
    /// # Arguments
    /// * `dfnum` - Numerator degrees of freedom, must be positive.
    /// * `dfden` - Denominator degrees of freedom, must be positive.
    /// * `size` - Number of values to generate.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if either df is non-positive or `size` is zero.
    pub fn f(
        &mut self,
        dfnum: f64,
        dfden: f64,
        size: usize,
    ) -> Result<Array<f64, Ix1>, FerrayError> {
        if size == 0 {
            return Err(FerrayError::invalid_value("size must be > 0"));
        }
        if dfnum <= 0.0 {
            return Err(FerrayError::invalid_value(format!(
                "dfnum must be positive, got {dfnum}"
            )));
        }
        if dfden <= 0.0 {
            return Err(FerrayError::invalid_value(format!(
                "dfden must be positive, got {dfden}"
            )));
        }
        let data = generate_vec(self, size, |bg| {
            let x1 = standard_gamma_single(bg, dfnum / 2.0);
            let x2 = standard_gamma_single(bg, dfden / 2.0);
            if x2 == 0.0 {
                f64::INFINITY
            } else {
                (x1 / dfnum) / (x2 / dfden)
            }
        });
        vec_to_array1(data)
    }

    /// Generate an array of Student's t-distributed variates.
    ///
    /// t(df) = Normal(0,1) / sqrt(Chi2(df)/df).
    ///
    /// # Arguments
    /// * `df` - Degrees of freedom, must be positive.
    /// * `size` - Number of values to generate.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `df <= 0` or `size` is zero.
    pub fn student_t(&mut self, df: f64, size: usize) -> Result<Array<f64, Ix1>, FerrayError> {
        if size == 0 {
            return Err(FerrayError::invalid_value("size must be > 0"));
        }
        if df <= 0.0 {
            return Err(FerrayError::invalid_value(format!(
                "df must be positive, got {df}"
            )));
        }
        let data = generate_vec(self, size, |bg| {
            let z = standard_normal_single(bg);
            let chi2 = 2.0 * standard_gamma_single(bg, df / 2.0);
            z / (chi2 / df).sqrt()
        });
        vec_to_array1(data)
    }
}

#[cfg(test)]
mod tests {
    use crate::default_rng_seeded;

    #[test]
    fn gamma_positive() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.gamma(2.0, 1.0, 10_000).unwrap();
        let slice = arr.as_slice().unwrap();
        for &v in slice {
            assert!(v > 0.0);
        }
    }

    #[test]
    fn gamma_mean_variance() {
        let mut rng = default_rng_seeded(42);
        let n = 100_000;
        let shape = 3.0;
        let scale = 2.0;
        let arr = rng.gamma(shape, scale, n).unwrap();
        let slice = arr.as_slice().unwrap();
        let mean: f64 = slice.iter().sum::<f64>() / n as f64;
        let var: f64 = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        // Gamma(k, theta): mean = k*theta, var = k*theta^2
        let expected_mean = shape * scale;
        let expected_var = shape * scale * scale;
        let se = (expected_var / n as f64).sqrt();
        assert!(
            (mean - expected_mean).abs() < 3.0 * se,
            "gamma mean {mean} too far from {expected_mean}"
        );
        assert!(
            (var - expected_var).abs() / expected_var < 0.05,
            "gamma variance {var} too far from {expected_var}"
        );
    }

    #[test]
    fn gamma_small_shape() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.gamma(0.5, 1.0, 10_000).unwrap();
        let slice = arr.as_slice().unwrap();
        for &v in slice {
            assert!(v > 0.0);
        }
    }

    #[test]
    fn beta_in_range() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.beta(2.0, 5.0, 10_000).unwrap();
        let slice = arr.as_slice().unwrap();
        for &v in slice {
            assert!(v > 0.0 && v < 1.0, "beta value {v} out of (0,1)");
        }
    }

    #[test]
    fn beta_mean() {
        let mut rng = default_rng_seeded(42);
        let n = 100_000;
        let a = 2.0;
        let b = 5.0;
        let arr = rng.beta(a, b, n).unwrap();
        let slice = arr.as_slice().unwrap();
        let mean: f64 = slice.iter().sum::<f64>() / n as f64;
        // Beta(a,b): mean = a/(a+b)
        let expected_mean = a / (a + b);
        let expected_var = (a * b) / ((a + b).powi(2) * (a + b + 1.0));
        let se = (expected_var / n as f64).sqrt();
        assert!(
            (mean - expected_mean).abs() < 3.0 * se,
            "beta mean {mean} too far from {expected_mean}"
        );
    }

    #[test]
    fn chisquare_positive() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.chisquare(5.0, 10_000).unwrap();
        let slice = arr.as_slice().unwrap();
        for &v in slice {
            assert!(v > 0.0);
        }
    }

    #[test]
    fn chisquare_mean() {
        let mut rng = default_rng_seeded(42);
        let n = 100_000;
        let df = 10.0;
        let arr = rng.chisquare(df, n).unwrap();
        let slice = arr.as_slice().unwrap();
        let mean: f64 = slice.iter().sum::<f64>() / n as f64;
        // Chi2(df): mean = df
        let expected_var = 2.0 * df;
        let se = (expected_var / n as f64).sqrt();
        assert!(
            (mean - df).abs() < 3.0 * se,
            "chisquare mean {mean} too far from {df}"
        );
    }

    #[test]
    fn f_positive() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.f(5.0, 10.0, 10_000).unwrap();
        let slice = arr.as_slice().unwrap();
        for &v in slice {
            assert!(v > 0.0);
        }
    }

    #[test]
    fn student_t_symmetric() {
        let mut rng = default_rng_seeded(42);
        let n = 100_000;
        let df = 10.0;
        let arr = rng.student_t(df, n).unwrap();
        let slice = arr.as_slice().unwrap();
        let mean: f64 = slice.iter().sum::<f64>() / n as f64;
        // t(df) with df > 1: mean = 0
        assert!(mean.abs() < 0.05, "student_t mean {mean} too far from 0");
    }

    #[test]
    fn standard_gamma_mean() {
        let mut rng = default_rng_seeded(42);
        let n = 100_000;
        let shape = 5.0;
        let arr = rng.standard_gamma(shape, n).unwrap();
        let slice = arr.as_slice().unwrap();
        let mean: f64 = slice.iter().sum::<f64>() / n as f64;
        let se = (shape / n as f64).sqrt();
        assert!(
            (mean - shape).abs() < 3.0 * se,
            "standard_gamma mean {mean} too far from {shape}"
        );
    }

    #[test]
    fn gamma_bad_params() {
        let mut rng = default_rng_seeded(42);
        assert!(rng.gamma(0.0, 1.0, 100).is_err());
        assert!(rng.gamma(1.0, 0.0, 100).is_err());
        assert!(rng.gamma(-1.0, 1.0, 100).is_err());
    }
}
