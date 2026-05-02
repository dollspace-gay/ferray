// ferray-random: Discrete distributions
//
// binomial, negative_binomial, poisson, geometric, hypergeometric, logseries

use ferray_core::{Array, FerrayError, IxDyn};

use crate::bitgen::BitGenerator;
use crate::distributions::gamma::standard_gamma_single;
use crate::generator::{Generator, generate_vec_i64, shape_size, vec_to_array_i64};
use crate::shape::IntoShape;

/// Generate a single Poisson variate using Knuth's algorithm for small lambda,
/// or the transformed rejection method (Hormann) for large lambda.
fn poisson_single<B: BitGenerator>(bg: &mut B, lam: f64) -> i64 {
    if lam < 30.0 {
        // Knuth's algorithm
        let l = (-lam).exp();
        let mut k: i64 = 0;
        let mut p = 1.0;
        loop {
            k += 1;
            p *= bg.next_f64();
            if p <= l {
                return k - 1;
            }
        }
    } else {
        // Transformed rejection method (PA algorithm, Ahrens & Dieter)
        let slam = lam.sqrt();
        let loglam = lam.ln();
        let b = 2.53f64.mul_add(slam, 0.931);
        let a = 0.02483f64.mul_add(b, -0.059);
        let inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
        let vr = 0.9277 - 3.6224 / (b - 2.0);

        loop {
            let u = bg.next_f64() - 0.5;
            let v = bg.next_f64();
            let us = 0.5 - u.abs();
            let k = ((2.0 * a / us + b).mul_add(u, lam) + 0.43).floor() as i64;
            if k < 0 {
                continue;
            }
            if us >= 0.07 && v <= vr {
                return k;
            }
            if k > 0
                && us >= 0.013
                && v <= (k as f64)
                    .ln()
                    .mul_add(
                        -0.5,
                        (k as f64).mul_add(loglam, -lam) - ln_factorial(k as u64),
                    )
                    .exp()
                    * inv_alpha
            {
                return k;
            }
            if us < 0.013 && v > us {
                continue;
            }
            // Full log test
            let kf = k as f64;
            let log_accept = -lam + kf * loglam - ln_factorial(k as u64);
            if v.ln() + inv_alpha.ln() - (a / (us * us) + b).ln() <= log_accept {
                return k;
            }
        }
    }
}

/// Approximate ln(n!) using Stirling's approximation with correction terms.
fn ln_factorial(n: u64) -> f64 {
    if n <= 20 {
        // Use exact values for small n
        let mut result = 0.0_f64;
        for i in 2..=n {
            result += (i as f64).ln();
        }
        result
    } else {
        // Stirling's approximation
        let nf = n as f64;
        0.5f64.mul_add((std::f64::consts::TAU).ln(), (nf + 0.5) * nf.ln()) - nf + 1.0 / (12.0 * nf)
            - 1.0 / (360.0 * nf * nf * nf)
    }
}

/// Generate a single binomial variate using the inverse transform for small n*p
/// or the BTPE algorithm for larger n*p.
fn binomial_single<B: BitGenerator>(bg: &mut B, n: u64, p: f64) -> i64 {
    if n == 0 || p == 0.0 {
        return 0;
    }
    if p == 1.0 {
        return n as i64;
    }

    // Use the smaller of p and 1-p for efficiency
    let (pp, flipped) = if p > 0.5 { (1.0 - p, true) } else { (p, false) };

    let np = n as f64 * pp;
    let q = 1.0 - pp;

    let result = if np < 30.0 {
        // Inverse transform (waiting time) method
        let s = pp / q;
        let a = (n as f64 + 1.0) * s;
        let mut r = q.powf(n as f64);
        let mut u = bg.next_f64();
        let mut x: i64 = 0;
        while u > r {
            u -= r;
            x += 1;
            r *= a / (x as f64) - s;
            if r < 0.0 {
                break;
            }
        }
        x.min(n as i64)
    } else {
        // BTPE algorithm (Hormann 1993) for large n*p.
        // Based on the transformed rejection method with decomposition
        // into triangular, parallelogram, and exponential regions.
        let fm = np + pp;
        let m = fm.floor() as i64;
        let mf = m as f64;
        let p1 = 2.195f64.mul_add((np * q).sqrt(), -(4.6 * q)).floor() + 0.5;
        let xm = mf + 0.5;
        let xl = xm - p1;
        let xr = xm + p1;
        let c = 0.134 + 20.5 / (15.3 + mf);
        let a = (fm - xl) / (fm - xl * pp);
        let lambda_l = a * 0.5f64.mul_add(a, 1.0);
        let a2 = (xr - fm) / (xr * q);
        let lambda_r = a2 * 0.5f64.mul_add(a2, 1.0);
        let p2 = p1 * 2.0f64.mul_add(c, 1.0);
        let p3 = p2 + c / lambda_l;
        let p4 = p3 + c / lambda_r;

        loop {
            let u = bg.next_f64() * p4;
            let v = bg.next_f64();
            let y: i64;

            if u <= p1 {
                // Triangular region
                y = (xm - p1 * v + u).floor() as i64;
            } else if u <= p2 {
                // Parallelogram region
                let x = xl + (u - p1) / c;
                // BTPE acceptance test: w = v + (x - xm)^2 / p1^2.
                // clippy::suspicious_operation_groupings would rewrite the
                // squared denominator to `x * p1`, which is mathematically
                // wrong here.
                #[allow(clippy::suspicious_operation_groupings)]
                let w = v + (x - xm) * (x - xm) / (p1 * p1);
                if w > 1.0 {
                    continue;
                }
                y = x.floor() as i64;
            } else if u <= p3 {
                // Left exponential tail
                y = (xl + v.ln() / lambda_l).floor() as i64;
                if y < 0 {
                    continue;
                }
            } else {
                // Right exponential tail
                y = (xr - v.ln() / lambda_r).floor() as i64;
                if y > n as i64 {
                    continue;
                }
            }

            // Squeeze acceptance
            let k = (y - m).abs();
            if k <= 20 || k as f64 >= (0.5 * np).mul_add(q, -1.0) {
                // Full acceptance/rejection via log-factorial comparison
                let kf = k as f64;
                let yf = y as f64;
                let rho =
                    (kf / (np * q)) * (kf.mul_add(kf / 3.0 + 0.625, 1.0 / 6.0) / (np * q) + 0.5);
                let t = -kf * kf / (2.0 * np * q);
                let log_a = t - rho;
                if v.ln() <= log_a {
                    break y;
                }
                // Full log-factorial test
                let log_v = v.ln();
                let log_accept = (yf - mf).mul_add(
                    (pp / q).ln(),
                    ln_factorial(m as u64) - ln_factorial(y as u64) - ln_factorial(n - y as u64)
                        + ln_factorial(n - m as u64),
                );
                if log_v <= log_accept {
                    break y;
                }
            } else {
                break y;
            }
        }
    };

    if flipped { n as i64 - result } else { result }
}

impl<B: BitGenerator> Generator<B> {
    /// Generate an array of binomial-distributed variates.
    ///
    /// Each value is the number of successes in `n` Bernoulli trials
    /// with success probability `p`.
    ///
    /// # Arguments
    /// * `n` - Number of trials.
    /// * `p` - Probability of success per trial, must be in [0, 1].
    /// * `size` - Number of values to generate.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` for invalid parameters.
    pub fn binomial(
        &mut self,
        n: u64,
        p: f64,
        size: impl IntoShape,
    ) -> Result<Array<i64, IxDyn>, FerrayError> {
        if !(0.0..=1.0).contains(&p) {
            return Err(FerrayError::invalid_value(format!(
                "p must be in [0, 1], got {p}"
            )));
        }
        let shape_vec = size.into_shape()?;
        let total = shape_size(&shape_vec);
        let data = generate_vec_i64(self, total, |bg| binomial_single(bg, n, p));
        vec_to_array_i64(data, &shape_vec)
    }

    /// Generate an array of negative binomial distributed variates.
    ///
    /// The number of failures before `n` successes with success probability `p`.
    /// Uses the gamma-Poisson mixture.
    ///
    /// # Arguments
    /// * `n` - Number of successes (positive).
    /// * `p` - Probability of success, must be in (0, 1].
    /// * `size` - Number of values to generate.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` for invalid parameters.
    pub fn negative_binomial(
        &mut self,
        n: f64,
        p: f64,
        size: impl IntoShape,
    ) -> Result<Array<i64, IxDyn>, FerrayError> {
        if n <= 0.0 {
            return Err(FerrayError::invalid_value(format!(
                "n must be positive, got {n}"
            )));
        }
        if p <= 0.0 || p > 1.0 {
            return Err(FerrayError::invalid_value(format!(
                "p must be in (0, 1], got {p}"
            )));
        }
        let shape_vec = size.into_shape()?;
        let total = shape_size(&shape_vec);
        let data = generate_vec_i64(self, total, |bg| {
            // Gamma-Poisson mixture:
            // Y ~ Gamma(n, (1-p)/p), then X ~ Poisson(Y)
            let y = standard_gamma_single(bg, n) * (1.0 - p) / p;
            poisson_single(bg, y)
        });
        vec_to_array_i64(data, &shape_vec)
    }

    /// Generate an array of Poisson-distributed variates.
    ///
    /// # Arguments
    /// * `lam` - Expected number of events (lambda), must be non-negative.
    /// * `size` - Number of values to generate.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `lam < 0` or `size` is zero.
    pub fn poisson(
        &mut self,
        lam: f64,
        size: impl IntoShape,
    ) -> Result<Array<i64, IxDyn>, FerrayError> {
        if lam < 0.0 {
            return Err(FerrayError::invalid_value(format!(
                "lam must be non-negative, got {lam}"
            )));
        }
        let shape_vec = size.into_shape()?;
        let total = shape_size(&shape_vec);
        if lam == 0.0 {
            let data = vec![0i64; total];
            return vec_to_array_i64(data, &shape_vec);
        }
        let data = generate_vec_i64(self, total, |bg| poisson_single(bg, lam));
        vec_to_array_i64(data, &shape_vec)
    }

    /// Generate an array of Poisson variates with a broadcast `lam`
    /// parameter (#449).
    ///
    /// `lam` is an array of expected counts; each output element is
    /// sampled from `Poisson(lam[i])`. Output shape equals
    /// `lam.shape()`.
    ///
    /// # Errors
    /// - `FerrayError::InvalidValue` if any `lam` element is negative.
    pub fn poisson_array(
        &mut self,
        lam: &Array<f64, IxDyn>,
    ) -> Result<Array<i64, IxDyn>, FerrayError> {
        let shape = lam.shape().to_vec();
        let total: usize = shape.iter().product();
        let mut out: Vec<i64> = Vec::with_capacity(total);
        for &l in lam.iter() {
            if l < 0.0 {
                return Err(FerrayError::invalid_value(format!(
                    "lam must be non-negative, got {l}"
                )));
            }
            if l == 0.0 {
                out.push(0);
            } else {
                out.push(poisson_single(&mut self.bg, l));
            }
        }
        Array::<i64, IxDyn>::from_vec(IxDyn::new(&shape), out)
    }

    /// Generate an array of geometric-distributed variates.
    ///
    /// The number of trials until the first success (1-based).
    ///
    /// # Arguments
    /// * `p` - Probability of success, must be in (0, 1].
    /// * `size` - Number of values to generate.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `p` not in (0, 1] or `size` is zero.
    pub fn geometric(
        &mut self,
        p: f64,
        size: impl IntoShape,
    ) -> Result<Array<i64, IxDyn>, FerrayError> {
        if p <= 0.0 || p > 1.0 {
            return Err(FerrayError::invalid_value(format!(
                "p must be in (0, 1], got {p}"
            )));
        }
        let shape_vec = size.into_shape()?;
        let total = shape_size(&shape_vec);
        if (p - 1.0).abs() < f64::EPSILON {
            let data = vec![1i64; total];
            return vec_to_array_i64(data, &shape_vec);
        }
        let log_q = (1.0 - p).ln();
        let data = generate_vec_i64(self, total, |bg| {
            loop {
                let u = bg.next_f64();
                if u > f64::EPSILON {
                    return (u.ln() / log_q).floor() as i64 + 1;
                }
            }
        });
        vec_to_array_i64(data, &shape_vec)
    }

    /// Generate an array of hypergeometric-distributed variates.
    ///
    /// Models drawing `nsample` items without replacement from a population
    /// containing `ngood` success states and `nbad` failure states.
    ///
    /// # Arguments
    /// * `ngood` - Number of success states in the population.
    /// * `nbad` - Number of failure states in the population.
    /// * `nsample` - Number of items drawn.
    /// * `size` - Number of values to generate.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `nsample > ngood + nbad` or `size` is zero.
    pub fn hypergeometric(
        &mut self,
        ngood: u64,
        nbad: u64,
        nsample: u64,
        size: impl IntoShape,
    ) -> Result<Array<i64, IxDyn>, FerrayError> {
        let total = ngood + nbad;
        if nsample > total {
            return Err(FerrayError::invalid_value(format!(
                "nsample ({nsample}) > ngood + nbad ({total})"
            )));
        }
        let shape_vec = size.into_shape()?;
        let total_n = shape_size(&shape_vec);
        let data = generate_vec_i64(self, total_n, |bg| {
            hypergeometric_single(bg, ngood, nbad, nsample)
        });
        vec_to_array_i64(data, &shape_vec)
    }

    /// Generate an array of logarithmic series distributed variates.
    ///
    /// # Arguments
    /// * `p` - Shape parameter, must be in (0, 1).
    /// * `size` - Number of values to generate.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `p` not in (0, 1) or `size` is zero.
    pub fn logseries(
        &mut self,
        p: f64,
        size: impl IntoShape,
    ) -> Result<Array<i64, IxDyn>, FerrayError> {
        if p <= 0.0 || p >= 1.0 {
            return Err(FerrayError::invalid_value(format!(
                "p must be in (0, 1), got {p}"
            )));
        }
        let r = (-(-p).ln_1p()).recip();
        let shape_vec = size.into_shape()?;
        let total = shape_size(&shape_vec);
        let data = generate_vec_i64(self, total, |bg| {
            // Kemp's "second" algorithm for the logarithmic distribution.
            // See Devroye, "Non-Uniform Random Variate Generation", p. 548.
            loop {
                let u = bg.next_f64();
                if u <= f64::EPSILON || u >= 1.0 - f64::EPSILON {
                    continue;
                }
                let v = bg.next_f64();
                let q = 1.0 - (-r.recip() * u.ln()).exp();
                if q <= 0.0 {
                    return 1;
                }
                if v < q * q {
                    let k = (1.0 + v.log(q)).floor() as i64;
                    return k.max(1);
                }
                if v < q {
                    return 2;
                }
                return 1;
            }
        });
        vec_to_array_i64(data, &shape_vec)
    }

    /// Generate an array of Zipf-distributed variates.
    ///
    /// Samples from the Zipf (zeta) distribution with shape parameter `a > 1`,
    /// using Devroye's rejection algorithm (Non-Uniform Random Variate
    /// Generation, p. 551). The PMF is `P(k) = k^(-a) / zeta(a)` for
    /// `k = 1, 2, ...`.
    ///
    /// Equivalent to `numpy.random.Generator.zipf`.
    ///
    /// # Errors
    /// - `FerrayError::InvalidValue` if `a <= 1` or `size` is invalid.
    pub fn zipf(&mut self, a: f64, size: impl IntoShape) -> Result<Array<i64, IxDyn>, FerrayError> {
        if a <= 1.0 {
            return Err(FerrayError::invalid_value(format!(
                "a must be > 1 for Zipf, got {a}"
            )));
        }
        let am1 = a - 1.0;
        let b = 2.0_f64.powf(am1);
        let shape_vec = size.into_shape()?;
        let total = shape_size(&shape_vec);
        let data = generate_vec_i64(self, total, |bg| {
            loop {
                let u = 1.0 - bg.next_f64();
                let v = bg.next_f64();
                let x = u.powf(-1.0 / am1).floor();
                // Guard against overflow / non-positive results.
                if !x.is_finite() || x < 1.0 {
                    continue;
                }
                let t = (1.0 + 1.0 / x).powf(am1);
                // Devroye's acceptance: v * x * (t - 1) / (b - 1) <= t / b
                if v * x * (t - 1.0) / (b - 1.0) <= t / b {
                    if x > i64::MAX as f64 {
                        continue;
                    }
                    return x as i64;
                }
            }
        });
        vec_to_array_i64(data, &shape_vec)
    }
}

/// Generate a single hypergeometric variate using the direct algorithm.
fn hypergeometric_single<B: BitGenerator>(bg: &mut B, ngood: u64, nbad: u64, nsample: u64) -> i64 {
    // Direct simulation: draw nsample items from population
    let mut good_remaining = ngood;
    let mut total_remaining = ngood + nbad;
    let mut successes: i64 = 0;

    for _ in 0..nsample {
        if total_remaining == 0 {
            break;
        }
        let u = bg.next_f64();
        if u < (good_remaining as f64) / (total_remaining as f64) {
            successes += 1;
            good_remaining -= 1;
        }
        total_remaining -= 1;
    }
    successes
}

#[cfg(test)]
mod tests {
    use crate::default_rng_seeded;

    // ---- poisson_array (#449) ------------------------------------------

    #[test]
    fn poisson_array_shape_matches_lam() {
        use ferray_core::{Array, IxDyn};
        use crate::default_rng_seeded;
        let mut rng = default_rng_seeded(42);
        let lam = Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[2, 2]),
            vec![1.0, 5.0, 50.0, 0.0],
        )
        .unwrap();
        let out = rng.poisson_array(&lam).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        // Last element is lam=0 → must be exactly 0.
        let s = out.as_slice().unwrap();
        assert_eq!(s[3], 0);
        for &v in s {
            assert!(v >= 0);
        }
    }

    #[test]
    fn poisson_array_per_element_mean() {
        use ferray_core::{Array, IxDyn};
        use crate::default_rng_seeded;
        let mut rng = default_rng_seeded(11);
        let lams = [3.0_f64, 50.0];
        let lam =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2]), lams.to_vec()).unwrap();
        let n_trials = 5_000;
        let mut sums = [0.0_f64; 2];
        for _ in 0..n_trials {
            let out = rng.poisson_array(&lam).unwrap();
            let s = out.as_slice().unwrap();
            for j in 0..2 {
                sums[j] += s[j] as f64;
            }
        }
        for j in 0..2 {
            let mean = sums[j] / n_trials as f64;
            // Poisson(λ) variance = λ; SE for n_trials draws = sqrt(λ/n).
            let se = (lams[j] / n_trials as f64).sqrt();
            assert!(
                (mean - lams[j]).abs() < 4.0 * se,
                "elt {j}: mean {mean} too far from {}",
                lams[j]
            );
        }
    }

    #[test]
    fn poisson_array_negative_lam_errors() {
        use ferray_core::{Array, IxDyn};
        use crate::default_rng_seeded;
        let mut rng = default_rng_seeded(0);
        let lam = Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[2]),
            vec![1.0, -2.0],
        )
        .unwrap();
        assert!(rng.poisson_array(&lam).is_err());
    }

    #[test]
    fn poisson_mean() {
        let mut rng = default_rng_seeded(42);
        let n = 100_000;
        let lam = 5.0;
        let arr = rng.poisson(lam, n).unwrap();
        let slice = arr.as_slice().unwrap();
        let mean: f64 = slice.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        // Poisson(lam): mean = lam, var = lam
        let se = (lam / n as f64).sqrt();
        assert!(
            (mean - lam).abs() < 3.0 * se,
            "poisson mean {mean} too far from {lam}"
        );
    }

    #[test]
    fn poisson_large_lambda() {
        let mut rng = default_rng_seeded(42);
        let n = 50_000;
        let lam = 100.0;
        let arr = rng.poisson(lam, n).unwrap();
        let slice = arr.as_slice().unwrap();
        let mean: f64 = slice.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        let se = (lam / n as f64).sqrt();
        assert!(
            (mean - lam).abs() < 3.0 * se,
            "poisson mean {mean} too far from {lam}"
        );
    }

    #[test]
    fn poisson_zero() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.poisson(0.0, 100).unwrap();
        for &v in arr.as_slice().unwrap() {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn binomial_mean() {
        let mut rng = default_rng_seeded(42);
        let size = 100_000;
        let n = 20u64;
        let p = 0.3;
        let arr = rng.binomial(n, p, size).unwrap();
        let slice = arr.as_slice().unwrap();
        let mean: f64 = slice.iter().map(|&x| x as f64).sum::<f64>() / size as f64;
        // Binomial(n, p): mean = n*p
        let expected_mean = n as f64 * p;
        let expected_var = n as f64 * p * (1.0 - p);
        let se = (expected_var / size as f64).sqrt();
        assert!(
            (mean - expected_mean).abs() < 3.0 * se,
            "binomial mean {mean} too far from {expected_mean}"
        );
        // Values must be in [0, n]
        for &v in slice {
            assert!(
                v >= 0 && v <= n as i64,
                "binomial value {v} out of [0, {n}]"
            );
        }
    }

    #[test]
    fn binomial_edge_cases() {
        let mut rng = default_rng_seeded(42);
        // p=0: always 0
        let arr = rng.binomial(10, 0.0, 100).unwrap();
        for &v in arr.as_slice().unwrap() {
            assert_eq!(v, 0);
        }
        // p=1: always n
        let arr = rng.binomial(10, 1.0, 100).unwrap();
        for &v in arr.as_slice().unwrap() {
            assert_eq!(v, 10);
        }
    }

    #[test]
    fn negative_binomial_positive() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.negative_binomial(5.0, 0.5, 10_000).unwrap();
        for &v in arr.as_slice().unwrap() {
            assert!(v >= 0, "negative_binomial value {v} must be >= 0");
        }
    }

    #[test]
    fn geometric_mean() {
        let mut rng = default_rng_seeded(42);
        let n = 100_000;
        let p = 0.3;
        let arr = rng.geometric(p, n).unwrap();
        let slice = arr.as_slice().unwrap();
        let mean: f64 = slice.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        // Geometric(p) (1-based): mean = 1/p
        let expected_mean = 1.0 / p;
        let expected_var = (1.0 - p) / (p * p);
        let se = (expected_var / n as f64).sqrt();
        assert!(
            (mean - expected_mean).abs() < 3.0 * se,
            "geometric mean {mean} too far from {expected_mean}"
        );
        for &v in slice {
            assert!(v >= 1, "geometric value {v} must be >= 1");
        }
    }

    #[test]
    fn hypergeometric_range() {
        let mut rng = default_rng_seeded(42);
        let ngood = 20u64;
        let nbad = 30u64;
        let nsample = 10u64;
        let arr = rng.hypergeometric(ngood, nbad, nsample, 10_000).unwrap();
        let slice = arr.as_slice().unwrap();
        for &v in slice {
            assert!(
                v >= 0 && v <= nsample.min(ngood) as i64,
                "hypergeometric value {v} out of range"
            );
        }
    }

    #[test]
    fn hypergeometric_mean() {
        let mut rng = default_rng_seeded(42);
        let n = 100_000;
        let ngood = 20u64;
        let nbad = 30u64;
        let nsample = 10u64;
        let arr = rng.hypergeometric(ngood, nbad, nsample, n).unwrap();
        let slice = arr.as_slice().unwrap();
        let mean: f64 = slice.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        // Hypergeometric: mean = nsample * ngood / (ngood + nbad)
        let total = (ngood + nbad) as f64;
        let expected_mean = nsample as f64 * ngood as f64 / total;
        let expected_var = nsample as f64
            * (ngood as f64 / total)
            * (nbad as f64 / total)
            * (total - nsample as f64)
            / (total - 1.0);
        let se = (expected_var / n as f64).sqrt();
        assert!(
            (mean - expected_mean).abs() < 3.0 * se,
            "hypergeometric mean {mean} too far from {expected_mean}"
        );
    }

    #[test]
    fn logseries_positive() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.logseries(0.5, 10_000).unwrap();
        for &v in arr.as_slice().unwrap() {
            assert!(v >= 1, "logseries value {v} must be >= 1");
        }
    }

    #[test]
    fn bad_params() {
        let mut rng = default_rng_seeded(42);
        assert!(rng.binomial(10, -0.1, 10).is_err());
        assert!(rng.binomial(10, 1.5, 10).is_err());
        assert!(rng.poisson(-1.0, 10).is_err());
        assert!(rng.geometric(0.0, 10).is_err());
        assert!(rng.geometric(1.5, 10).is_err());
        assert!(rng.hypergeometric(5, 5, 20, 10).is_err());
        assert!(rng.logseries(0.0, 10).is_err());
        assert!(rng.logseries(1.0, 10).is_err());
        assert!(rng.negative_binomial(0.0, 0.5, 10).is_err());
        assert!(rng.negative_binomial(5.0, 0.0, 10).is_err());
    }

    #[test]
    fn zipf_positive_integers() {
        use crate::default_rng_seeded;
        let mut rng = default_rng_seeded(42);
        let arr = rng.zipf(2.5, 1000).unwrap();
        for &v in arr.as_slice().unwrap() {
            assert!(v >= 1, "zipf output must be >= 1, got {v}");
        }
    }

    #[test]
    fn zipf_seed_reproducible() {
        use crate::default_rng_seeded;
        let mut a = default_rng_seeded(7);
        let mut b = default_rng_seeded(7);
        let xs = a.zipf(3.0, 200).unwrap();
        let ys = b.zipf(3.0, 200).unwrap();
        assert_eq!(xs.as_slice().unwrap(), ys.as_slice().unwrap());
    }

    #[test]
    fn zipf_bad_a_errs() {
        use crate::default_rng_seeded;
        let mut rng = default_rng_seeded(0);
        assert!(rng.zipf(1.0, 10).is_err());
        assert!(rng.zipf(0.5, 10).is_err());
        assert!(rng.zipf(-2.0, 10).is_err());
    }
}
