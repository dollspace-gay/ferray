// ferray-random: Multivariate distributions — multinomial, multivariate_normal, dirichlet

use ferray_core::{Array, FerrayError, Ix1, Ix2};

use crate::bitgen::BitGenerator;
use crate::distributions::gamma::standard_gamma_single;
use crate::distributions::normal::standard_normal_single;
use crate::generator::Generator;

impl<B: BitGenerator> Generator<B> {
    /// Generate multinomial samples.
    ///
    /// Each row of the output is one draw of `n` items distributed across
    /// `k` categories with probabilities `pvals`.
    ///
    /// # Arguments
    /// * `n` - Number of trials per sample.
    /// * `pvals` - Category probabilities (must sum to ~1.0, length k).
    /// * `size` - Number of multinomial draws (rows in output).
    ///
    /// # Returns
    /// An `Array<i64, Ix2>` with shape `[size, k]`.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` for invalid parameters.
    pub fn multinomial(
        &mut self,
        n: u64,
        pvals: &[f64],
        size: usize,
    ) -> Result<Array<i64, Ix2>, FerrayError> {
        if size == 0 {
            return Err(FerrayError::invalid_value("size must be > 0"));
        }
        if pvals.is_empty() {
            return Err(FerrayError::invalid_value(
                "pvals must have at least one element",
            ));
        }
        let psum: f64 = pvals.iter().sum();
        if (psum - 1.0).abs() > 1e-6 {
            return Err(FerrayError::invalid_value(format!(
                "pvals must sum to 1.0, got {psum}"
            )));
        }
        for (i, &p) in pvals.iter().enumerate() {
            if p < 0.0 {
                return Err(FerrayError::invalid_value(format!(
                    "pvals[{i}] = {p} is negative"
                )));
            }
        }

        let k = pvals.len();
        let mut data = Vec::with_capacity(size * k);

        for _ in 0..size {
            let mut remaining = n;
            let mut psum_remaining = 1.0;
            for (j, &pj) in pvals.iter().enumerate() {
                if j == k - 1 {
                    // Last category gets all remaining
                    data.push(remaining as i64);
                } else if psum_remaining <= 0.0 || remaining == 0 {
                    data.push(0);
                } else {
                    let p_cond = (pj / psum_remaining).clamp(0.0, 1.0);
                    let count = binomial_for_multinomial(&mut self.bg, remaining, p_cond);
                    data.push(count as i64);
                    remaining -= count;
                    psum_remaining -= pj;
                }
            }
        }

        Array::<i64, Ix2>::from_vec(Ix2::new([size, k]), data)
    }

    /// Generate multivariate normal samples.
    ///
    /// Uses the Cholesky decomposition of the covariance matrix.
    ///
    /// # Arguments
    /// * `mean` - Mean vector of length `d`.
    /// * `cov` - Covariance matrix, flattened in row-major order, shape `[d, d]`.
    /// * `size` - Number of samples (rows in output).
    ///
    /// # Returns
    /// An `Array<f64, Ix2>` with shape `[size, d]`.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` for invalid parameters or if
    /// the covariance matrix is not positive semi-definite.
    pub fn multivariate_normal(
        &mut self,
        mean: &[f64],
        cov: &[f64],
        size: usize,
    ) -> Result<Array<f64, Ix2>, FerrayError> {
        if size == 0 {
            return Err(FerrayError::invalid_value("size must be > 0"));
        }
        let d = mean.len();
        if d == 0 {
            return Err(FerrayError::invalid_value("mean must be non-empty"));
        }
        if cov.len() != d * d {
            return Err(FerrayError::invalid_value(format!(
                "cov must have {} elements for mean of length {d}, got {}",
                d * d,
                cov.len()
            )));
        }

        // Compute Cholesky decomposition L such that cov = L * L^T
        let l = cholesky_decompose(cov, d)?;

        let mut data = Vec::with_capacity(size * d);
        for _ in 0..size {
            // Generate d independent standard normals
            let mut z = Vec::with_capacity(d);
            for _ in 0..d {
                z.push(standard_normal_single(&mut self.bg));
            }

            // x = mean + L * z
            for i in 0..d {
                let mut val = mean[i];
                for j in 0..=i {
                    val += l[i * d + j] * z[j];
                }
                data.push(val);
            }
        }

        Array::<f64, Ix2>::from_vec(Ix2::new([size, d]), data)
    }

    /// Generate samples from the multivariate hypergeometric distribution.
    ///
    /// Models drawing `nsample` items without replacement from a population
    /// partitioned into `K` colors with `colors[k]` items of color `k`.
    /// Each row of the output is one such draw — a vector of `K` non-negative
    /// counts summing to `nsample`.
    ///
    /// Uses the marginals algorithm: a sequence of `K-1` univariate
    /// hypergeometric draws, each picking the count of one color from the
    /// remainder of the population. The final color is what's left. This
    /// matches `numpy.random.Generator.multivariate_hypergeometric` (#445).
    ///
    /// # Arguments
    /// * `colors` - Number of items of each color (length K, all non-negative).
    /// * `nsample` - Number of items drawn per sample (must be ≤ sum of `colors`).
    /// * `size` - Number of multivariate draws (rows in output).
    ///
    /// # Returns
    /// An `Array<i64, Ix2>` with shape `[size, K]`.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `colors` is empty, `size` is 0,
    /// or `nsample` exceeds the population total.
    pub fn multivariate_hypergeometric(
        &mut self,
        colors: &[u64],
        nsample: u64,
        size: usize,
    ) -> Result<Array<i64, Ix2>, FerrayError> {
        if size == 0 {
            return Err(FerrayError::invalid_value("size must be > 0"));
        }
        if colors.is_empty() {
            return Err(FerrayError::invalid_value(
                "colors must have at least one element",
            ));
        }
        let total: u64 = colors.iter().try_fold(0_u64, |acc, &c| {
            acc.checked_add(c).ok_or_else(|| {
                FerrayError::invalid_value("multivariate_hypergeometric: colors sum overflows u64")
            })
        })?;
        if nsample > total {
            return Err(FerrayError::invalid_value(format!(
                "nsample ({nsample}) > sum of colors ({total})"
            )));
        }

        let k = colors.len();
        let mut data = Vec::with_capacity(size * k);

        for _ in 0..size {
            // For each color j in 0..k-1, draw a univariate hypergeometric
            // from (ngood = colors[j], nbad = remaining_total - colors[j]).
            // Subtract the draw and proceed; the last color gets whatever
            // is left of `nsample`.
            let mut remaining_pop: u64 = total;
            let mut remaining_sample: u64 = nsample;

            for &ngood in &colors[..k - 1] {
                let nbad = remaining_pop - ngood;
                let draw = if remaining_sample == 0 || ngood == 0 {
                    0
                } else if remaining_sample >= remaining_pop {
                    // Take everything that's left of this color.
                    ngood as i64
                } else {
                    hypergeometric_for_multivariate(&mut self.bg, ngood, nbad, remaining_sample)
                };
                data.push(draw);
                remaining_pop -= ngood;
                remaining_sample -= draw as u64;
            }
            // Final color absorbs the remainder of the sample.
            data.push(remaining_sample as i64);
        }

        Array::<i64, Ix2>::from_vec(Ix2::new([size, k]), data)
    }

    /// Generate multivariate normal samples taking `Array` parameters.
    ///
    /// Ergonomic counterpart to [`multivariate_normal`] (#451): accepts
    /// the mean as `Array<f64, Ix1>` and the covariance as
    /// `Array<f64, Ix2>` directly, no manual flattening required.
    ///
    /// Cholesky decomposition is delegated to `ferray_linalg::cholesky`
    /// (#452) which is faer-backed and surfaces non-positive-definite
    /// inputs as `FerrayError::SingularMatrix` instead of the
    /// home-grown `cholesky_decompose` helper.
    ///
    /// # Errors
    /// - `FerrayError::ShapeMismatch` if `cov` is not square or its
    ///   side does not match `mean.len()`.
    /// - `FerrayError::SingularMatrix` if `cov` is not positive
    ///   definite (propagated from `ferray-linalg`).
    /// - `FerrayError::InvalidValue` for size = 0 or empty mean.
    pub fn multivariate_normal_array(
        &mut self,
        mean: &Array<f64, Ix1>,
        cov: &Array<f64, Ix2>,
        size: usize,
    ) -> Result<Array<f64, Ix2>, FerrayError> {
        if size == 0 {
            return Err(FerrayError::invalid_value("size must be > 0"));
        }
        let d = mean.shape()[0];
        if d == 0 {
            return Err(FerrayError::invalid_value("mean must be non-empty"));
        }
        let cov_shape = cov.shape();
        if cov_shape[0] != d || cov_shape[1] != d {
            return Err(FerrayError::shape_mismatch(format!(
                "cov shape {cov_shape:?} does not match mean of length {d}"
            )));
        }

        let l_arr = ferray_linalg::cholesky(cov)?;
        let l_slice = l_arr
            .as_slice()
            .ok_or_else(|| FerrayError::invalid_value("cholesky returned non-contiguous L"))?;
        let mean_slice = mean
            .as_slice()
            .ok_or_else(|| FerrayError::invalid_value("mean must be contiguous"))?;

        let mut data = Vec::with_capacity(size * d);
        let mut z = vec![0.0_f64; d];
        for _ in 0..size {
            for v in z.iter_mut() {
                *v = standard_normal_single(&mut self.bg);
            }
            for i in 0..d {
                let mut val = mean_slice[i];
                for j in 0..=i {
                    val += l_slice[i * d + j] * z[j];
                }
                data.push(val);
            }
        }
        Array::<f64, Ix2>::from_vec(Ix2::new([size, d]), data)
    }

    /// Generate Dirichlet-distributed samples.
    ///
    /// Each row is a sample from the Dirichlet distribution parameterized
    /// by `alpha`, producing vectors that sum to 1.
    ///
    /// # Arguments
    /// * `alpha` - Concentration parameters (all must be positive).
    /// * `size` - Number of samples (rows in output).
    ///
    /// # Returns
    /// An `Array<f64, Ix2>` with shape `[size, k]` where k = `alpha.len()`.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` for invalid parameters.
    pub fn dirichlet(
        &mut self,
        alpha: &[f64],
        size: usize,
    ) -> Result<Array<f64, Ix2>, FerrayError> {
        if size == 0 {
            return Err(FerrayError::invalid_value("size must be > 0"));
        }
        if alpha.is_empty() {
            return Err(FerrayError::invalid_value(
                "alpha must have at least one element",
            ));
        }
        for (i, &a) in alpha.iter().enumerate() {
            if a <= 0.0 {
                return Err(FerrayError::invalid_value(format!(
                    "alpha[{i}] = {a} must be positive"
                )));
            }
        }

        let k = alpha.len();
        let mut data = Vec::with_capacity(size * k);

        for _ in 0..size {
            let mut gammas = Vec::with_capacity(k);
            let mut sum = 0.0;
            for &a in alpha {
                let g = standard_gamma_single(&mut self.bg, a);
                gammas.push(g);
                sum += g;
            }
            // Normalize
            if sum > 0.0 {
                for g in &gammas {
                    data.push(g / sum);
                }
            } else {
                // Degenerate: uniform
                let val = 1.0 / k as f64;
                for _ in 0..k {
                    data.push(val);
                }
            }
        }

        Array::<f64, Ix2>::from_vec(Ix2::new([size, k]), data)
    }
}

/// Univariate hypergeometric draw used by `multivariate_hypergeometric`.
/// Direct simulation: draw `nsample` items from a population of
/// `ngood + nbad` and count successes. Equivalent to the helper in
/// `discrete.rs::hypergeometric_single` — kept private here to avoid
/// the cross-module visibility wiring that would otherwise pull in
/// generator internals.
fn hypergeometric_for_multivariate<B: BitGenerator>(
    bg: &mut B,
    ngood: u64,
    nbad: u64,
    nsample: u64,
) -> i64 {
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

/// Simple binomial sampling for multinomial (avoids circular dependency).
fn binomial_for_multinomial<B: BitGenerator>(bg: &mut B, n: u64, p: f64) -> u64 {
    if n == 0 || p <= 0.0 {
        return 0;
    }
    if p >= 1.0 {
        return n;
    }

    let (pp, flipped) = if p > 0.5 { (1.0 - p, true) } else { (p, false) };

    let result = if (n as f64) * pp < 30.0 {
        // Inverse transform
        let q = 1.0 - pp;
        let s = pp / q;
        let a = (n as f64 + 1.0) * s;
        let mut r = q.powf(n as f64);
        let mut u = bg.next_f64();
        let mut x: u64 = 0;
        while u > r {
            u -= r;
            x += 1;
            if x > n {
                x = n;
                break;
            }
            r *= a / (x as f64) - s;
            if r < 0.0 {
                break;
            }
        }
        x.min(n)
    } else {
        // Normal approximation for large n*p
        loop {
            let z = standard_normal_single(bg);
            let sigma = ((n as f64) * pp * (1.0 - pp)).sqrt();
            let x = ((n as f64).mul_add(pp, sigma * z) + 0.5).floor() as i64;
            if x >= 0 && x <= n as i64 {
                break x as u64;
            }
        }
    };

    if flipped { n - result } else { result }
}

/// Cholesky decomposition of a symmetric positive-definite matrix.
/// Input: flat row-major matrix `a` of size `n x n`.
/// Output: lower-triangular matrix L such that A = L * L^T.
fn cholesky_decompose(a: &[f64], n: usize) -> Result<Vec<f64>, FerrayError> {
    let mut l = vec![0.0; n * n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i * n + k] * l[j * n + k];
            }
            if i == j {
                let diag = a[i * n + i] - sum;
                if diag < -1e-10 {
                    return Err(FerrayError::invalid_value(
                        "covariance matrix is not positive semi-definite",
                    ));
                }
                l[i * n + j] = diag.max(0.0).sqrt();
            } else {
                let denom = l[j * n + j];
                if denom.abs() < 1e-15 {
                    l[i * n + j] = 0.0;
                } else {
                    l[i * n + j] = (a[i * n + j] - sum) / denom;
                }
            }
        }
    }

    Ok(l)
}

#[cfg(test)]
mod tests {
    use crate::default_rng_seeded;

    #[test]
    fn multinomial_shape() {
        let mut rng = default_rng_seeded(42);
        let pvals = [0.2, 0.3, 0.5];
        let arr = rng.multinomial(100, &pvals, 10).unwrap();
        assert_eq!(arr.shape(), &[10, 3]);
    }

    #[test]
    fn multinomial_row_sums() {
        let mut rng = default_rng_seeded(42);
        let pvals = [0.2, 0.3, 0.5];
        let n = 100u64;
        let arr = rng.multinomial(n, &pvals, 50).unwrap();
        let slice = arr.as_slice().unwrap();
        let k = pvals.len();
        for row in 0..50 {
            let row_sum: i64 = (0..k).map(|j| slice[row * k + j]).sum();
            assert_eq!(
                row_sum, n as i64,
                "row {row} sum is {row_sum}, expected {n}"
            );
        }
    }

    #[test]
    fn multinomial_nonnegative() {
        let mut rng = default_rng_seeded(42);
        let pvals = [0.1, 0.2, 0.3, 0.4];
        let arr = rng.multinomial(50, &pvals, 100).unwrap();
        for &v in arr.as_slice().unwrap() {
            assert!(v >= 0, "multinomial produced negative count: {v}");
        }
    }

    #[test]
    fn multinomial_bad_pvals() {
        let mut rng = default_rng_seeded(42);
        assert!(rng.multinomial(10, &[0.5, 0.6], 10).is_err()); // sum > 1
        assert!(rng.multinomial(10, &[-0.1, 1.1], 10).is_err()); // negative
        assert!(rng.multinomial(10, &[], 10).is_err()); // empty
    }

    #[test]
    fn multivariate_normal_shape() {
        let mut rng = default_rng_seeded(42);
        let mean = [1.0, 2.0, 3.0];
        let cov = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let arr = rng.multivariate_normal(&mean, &cov, 100).unwrap();
        assert_eq!(arr.shape(), &[100, 3]);
    }

    #[test]
    fn multivariate_normal_mean() {
        let mut rng = default_rng_seeded(42);
        let mean = [5.0, -3.0];
        let cov = [1.0, 0.0, 0.0, 1.0];
        let n = 100_000;
        let arr = rng.multivariate_normal(&mean, &cov, n).unwrap();
        let slice = arr.as_slice().unwrap();
        let d = mean.len();

        for j in 0..d {
            let col_mean: f64 = (0..n).map(|i| slice[i * d + j]).sum::<f64>() / n as f64;
            let se = (1.0 / n as f64).sqrt();
            assert!(
                (col_mean - mean[j]).abs() < 3.0 * se,
                "multivariate_normal mean[{j}] = {col_mean}, expected {}",
                mean[j]
            );
        }
    }

    #[test]
    fn multivariate_normal_bad_cov() {
        let mut rng = default_rng_seeded(42);
        let mean = [0.0, 0.0];
        // Wrong size cov
        assert!(
            rng.multivariate_normal(&mean, &[1.0, 0.0, 0.0], 10)
                .is_err()
        );
    }

    #[test]
    fn dirichlet_shape() {
        let mut rng = default_rng_seeded(42);
        let alpha = [1.0, 2.0, 3.0];
        let arr = rng.dirichlet(&alpha, 10).unwrap();
        assert_eq!(arr.shape(), &[10, 3]);
    }

    #[test]
    fn dirichlet_sums_to_one() {
        let mut rng = default_rng_seeded(42);
        let alpha = [0.5, 1.0, 2.0, 0.5];
        let arr = rng.dirichlet(&alpha, 100).unwrap();
        let slice = arr.as_slice().unwrap();
        let k = alpha.len();
        for row in 0..100 {
            let row_sum: f64 = (0..k).map(|j| slice[row * k + j]).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-10,
                "dirichlet row {row} sums to {row_sum}, expected 1.0"
            );
        }
    }

    #[test]
    fn dirichlet_nonnegative() {
        let mut rng = default_rng_seeded(42);
        let alpha = [0.5, 1.0, 2.0];
        let arr = rng.dirichlet(&alpha, 100).unwrap();
        for &v in arr.as_slice().unwrap() {
            assert!(v >= 0.0, "dirichlet produced negative value: {v}");
        }
    }

    #[test]
    fn dirichlet_bad_alpha() {
        let mut rng = default_rng_seeded(42);
        assert!(rng.dirichlet(&[], 10).is_err());
        assert!(rng.dirichlet(&[1.0, 0.0], 10).is_err());
        assert!(rng.dirichlet(&[1.0, -1.0], 10).is_err());
    }

    // ---- multivariate_normal_array (#451, #452) ------------------------

    #[test]
    fn mvn_array_shape() {
        use ferray_core::{Array, Ix1, Ix2};
        let mut rng = default_rng_seeded(42);
        let mean = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let cov = Array::<f64, Ix2>::from_vec(
            Ix2::new([3, 3]),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let arr = rng.multivariate_normal_array(&mean, &cov, 100).unwrap();
        assert_eq!(arr.shape(), &[100, 3]);
    }

    #[test]
    fn mvn_array_means_match() {
        use ferray_core::{Array, Ix1, Ix2};
        let mut rng = default_rng_seeded(42);
        let mean = Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![5.0, -3.0]).unwrap();
        let cov = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let n = 100_000;
        let arr = rng.multivariate_normal_array(&mean, &cov, n).unwrap();
        let slice = arr.as_slice().unwrap();
        for j in 0..2 {
            let m: f64 = (0..n).map(|i| slice[i * 2 + j]).sum::<f64>() / n as f64;
            let se = (1.0 / n as f64).sqrt();
            let want = mean.as_slice().unwrap()[j];
            assert!((m - want).abs() < 4.0 * se, "col {j} mean {m} ≠ {want}");
        }
    }

    #[test]
    fn mvn_array_rejects_non_square_cov() {
        use ferray_core::{Array, Ix1, Ix2};
        let mut rng = default_rng_seeded(0);
        let mean = Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![0.0, 0.0]).unwrap();
        // 2×3 cov — neither square nor matching mean length.
        let cov = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            .unwrap();
        assert!(rng.multivariate_normal_array(&mean, &cov, 5).is_err());
    }

    #[test]
    fn mvn_array_rejects_non_pd_cov() {
        use ferray_core::{Array, Ix1, Ix2};
        let mut rng = default_rng_seeded(0);
        let mean = Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![0.0, 0.0]).unwrap();
        // Indefinite (eigenvalues ±1).
        let cov = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![0.0, 1.0, 1.0, 0.0]).unwrap();
        let err = rng.multivariate_normal_array(&mean, &cov, 5).unwrap_err();
        assert!(matches!(
            err,
            ferray_core::FerrayError::SingularMatrix { .. }
        ));
    }

    #[test]
    fn mvn_array_correlated_recovers_cov() {
        use ferray_core::{Array, Ix1, Ix2};
        // Strongly correlated covariance — sample covariance should
        // approximate the input cov to within sampling error.
        let mut rng = default_rng_seeded(11);
        let mean = Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![0.0, 0.0]).unwrap();
        let cov = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![1.0, 0.7, 0.7, 1.0]).unwrap();
        let n = 50_000;
        let arr = rng.multivariate_normal_array(&mean, &cov, n).unwrap();
        let s = arr.as_slice().unwrap();
        let mean0: f64 = (0..n).map(|i| s[i * 2]).sum::<f64>() / n as f64;
        let mean1: f64 = (0..n).map(|i| s[i * 2 + 1]).sum::<f64>() / n as f64;
        let cov01: f64 = (0..n)
            .map(|i| (s[i * 2] - mean0) * (s[i * 2 + 1] - mean1))
            .sum::<f64>()
            / n as f64;
        assert!((cov01 - 0.7).abs() < 0.05, "sample cov01 {cov01} ≠ 0.7");
    }

    // ---- multivariate_hypergeometric (#445) ----------------------------

    #[test]
    fn mvhg_shape_and_row_sum() {
        let mut rng = default_rng_seeded(42);
        let colors = [10u64, 20, 30];
        let nsample = 15u64;
        let arr = rng
            .multivariate_hypergeometric(&colors, nsample, 50)
            .unwrap();
        assert_eq!(arr.shape(), &[50, 3]);
        let slice = arr.as_slice().unwrap();
        for row in 0..50 {
            let row_sum: i64 = (0..3).map(|j| slice[row * 3 + j]).sum();
            assert_eq!(row_sum, nsample as i64);
        }
    }

    #[test]
    fn mvhg_per_color_within_population() {
        let mut rng = default_rng_seeded(123);
        let colors = [5u64, 5, 5];
        let arr = rng.multivariate_hypergeometric(&colors, 10, 200).unwrap();
        let slice = arr.as_slice().unwrap();
        for row in 0..200 {
            for j in 0..3 {
                let v = slice[row * 3 + j];
                assert!(
                    v >= 0 && v <= colors[j] as i64,
                    "row {row} col {j}: count {v} out of [0, {}]",
                    colors[j]
                );
            }
        }
    }

    #[test]
    fn mvhg_marginal_means_match_theory() {
        // E[X_j] = nsample * colors[j] / sum(colors)
        let mut rng = default_rng_seeded(7);
        let colors = [10u64, 20, 30, 40];
        let total: f64 = colors.iter().sum::<u64>() as f64;
        let nsample = 25u64;
        let n_draws = 10_000;
        let arr = rng
            .multivariate_hypergeometric(&colors, nsample, n_draws)
            .unwrap();
        let slice = arr.as_slice().unwrap();
        let k = colors.len();
        for j in 0..k {
            let observed: f64 =
                (0..n_draws).map(|i| slice[i * k + j] as f64).sum::<f64>() / n_draws as f64;
            let expected = nsample as f64 * colors[j] as f64 / total;
            // Marginal variance: nsample * (Kj/N) * (N-Kj)/N * (N-nsample)/(N-1)
            let kj = colors[j] as f64;
            let var = nsample as f64
                * (kj / total)
                * ((total - kj) / total)
                * ((total - nsample as f64) / (total - 1.0));
            let se = (var / n_draws as f64).sqrt();
            assert!(
                (observed - expected).abs() < 4.0 * se,
                "color {j}: observed mean {observed}, expected {expected} ± {se}"
            );
        }
    }

    #[test]
    fn mvhg_take_all() {
        // nsample == total: result is exactly the colors vector.
        let mut rng = default_rng_seeded(0);
        let colors = [3u64, 7, 0, 5];
        let total: u64 = colors.iter().sum();
        let arr = rng.multivariate_hypergeometric(&colors, total, 5).unwrap();
        let slice = arr.as_slice().unwrap();
        for row in 0..5 {
            for j in 0..colors.len() {
                assert_eq!(slice[row * colors.len() + j], colors[j] as i64);
            }
        }
    }

    #[test]
    fn mvhg_seed_reproducible() {
        let mut a = default_rng_seeded(99);
        let mut b = default_rng_seeded(99);
        let xa = a.multivariate_hypergeometric(&[5, 10, 15], 8, 30).unwrap();
        let xb = b.multivariate_hypergeometric(&[5, 10, 15], 8, 30).unwrap();
        assert_eq!(xa.as_slice().unwrap(), xb.as_slice().unwrap());
    }

    #[test]
    fn mvhg_bad_params() {
        let mut rng = default_rng_seeded(0);
        // size = 0
        assert!(rng.multivariate_hypergeometric(&[1, 2], 1, 0).is_err());
        // empty colors
        assert!(rng.multivariate_hypergeometric(&[], 0, 5).is_err());
        // nsample > total
        assert!(rng.multivariate_hypergeometric(&[3, 4], 10, 5).is_err());
    }
}
