// ferray-random: Multivariate distributions — multinomial, multivariate_normal, dirichlet

use ferray_core::{Array, FerrayError, Ix2};

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
    /// An `Array<f64, Ix2>` with shape `[size, k]` where k = alpha.len().
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
            let x = ((n as f64) * pp + sigma * z + 0.5).floor() as i64;
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
}
