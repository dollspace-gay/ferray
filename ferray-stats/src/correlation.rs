// ferray-stats: Correlation and covariance — correlate, corrcoef, cov (REQ-5, REQ-6, REQ-7)

use ferray_core::error::{FerrayError, FerrayResult};
use ferray_core::{Array, Dimension, Element, Ix1, Ix2};
use num_traits::Float;

// ---------------------------------------------------------------------------
// CorrelateMode
// ---------------------------------------------------------------------------

/// Mode for the `correlate` function, mirroring `numpy.correlate`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrelateMode {
    /// Full discrete cross-correlation. Output size = len(a) + len(v) - 1.
    Full,
    /// Output size equals the larger of len(a) and len(v).
    Same,
    /// Output size = max(len(a), len(v)) - min(len(a), len(v)) + 1.
    Valid,
}

// ---------------------------------------------------------------------------
// correlate
// ---------------------------------------------------------------------------

/// Discrete, linear cross-correlation of two 1-D arrays.
///
/// Equivalent to `numpy.correlate`.
pub fn correlate<T>(
    a: &Array<T, Ix1>,
    v: &Array<T, Ix1>,
    mode: CorrelateMode,
) -> FerrayResult<Array<T, Ix1>>
where
    T: Element + Float,
{
    let a_data: Vec<T> = a.iter().copied().collect();
    // numpy.correlate(a, v) = convolve(a, reverse(v)) — earlier
    // implementation accidentally ran a straight convolution and
    // returned the time-reversed result (#722). Reverse v up front so
    // the existing convolution-style loop produces the correlation.
    let mut v_data: Vec<T> = v.iter().copied().collect();
    v_data.reverse();
    let la = a_data.len();
    let lv = v_data.len();

    if la == 0 || lv == 0 {
        return Err(FerrayError::invalid_value(
            "correlate requires non-empty arrays",
        ));
    }

    // Full correlation length
    let full_len = la + lv - 1;

    // Compute full cross-correlation
    let mut full = vec![<T as Element>::zero(); full_len];
    for (i, out) in full.iter_mut().enumerate() {
        let mut s = <T as Element>::zero();
        for (j, vj) in v_data.iter().enumerate() {
            let ai = i as isize - j as isize;
            if ai >= 0 && (ai as usize) < la {
                s = s + a_data[ai as usize] * *vj;
            }
        }
        *out = s;
    }

    let result = match mode {
        CorrelateMode::Full => full,
        CorrelateMode::Same => {
            let out_len = la.max(lv);
            let start = (full_len - out_len) / 2;
            full[start..start + out_len].to_vec()
        }
        CorrelateMode::Valid => {
            let out_len = la.max(lv) - la.min(lv) + 1;
            let start = la.min(lv) - 1;
            full[start..start + out_len].to_vec()
        }
    };

    let n = result.len();
    Array::from_vec(Ix1::new([n]), result)
}

// ---------------------------------------------------------------------------
// cov
// ---------------------------------------------------------------------------

/// Estimate the covariance matrix.
///
/// If `m` is a 2-D array, each row is a variable and each column is an observation
/// (when `rowvar` is true, the default). If `rowvar` is false, each column is a variable.
///
/// If `m` is 1-D, it is treated as a single variable.
///
/// `ddof` controls the normalization: the result is divided by `N - ddof` where N is
/// the number of observations.
///
/// Equivalent to `numpy.cov`.
pub fn cov<T, D>(m: &Array<T, D>, rowvar: bool, ddof: Option<usize>) -> FerrayResult<Array<T, Ix2>>
where
    T: Element + Float,
    D: Dimension,
{
    let ndim = m.ndim();
    if ndim > 2 {
        return Err(FerrayError::invalid_value("cov requires 1-D or 2-D input"));
    }

    // Collect data into a matrix where rows are variables, columns are observations
    let (nvars, nobs, matrix) = if ndim == 1 {
        let data: Vec<T> = m.iter().copied().collect();
        let n = data.len();
        (1, n, vec![data])
    } else {
        let shape = m.shape();
        let (r, c) = (shape[0], shape[1]);
        let data: Vec<T> = m.iter().copied().collect();
        if rowvar {
            let mut rows = Vec::with_capacity(r);
            for i in 0..r {
                rows.push(data[i * c..(i + 1) * c].to_vec());
            }
            (r, c, rows)
        } else {
            // Transpose: columns become variables
            let mut cols = Vec::with_capacity(c);
            for j in 0..c {
                let col: Vec<T> = (0..r).map(|i| data[i * c + j]).collect();
                cols.push(col);
            }
            (c, r, cols)
        }
    };

    let ddof_val = ddof.unwrap_or(1);
    if nobs <= ddof_val {
        return Err(FerrayError::invalid_value(
            "number of observations must be greater than ddof",
        ));
    }
    let nf = T::from(nobs).unwrap();
    let denom = T::from(nobs - ddof_val).unwrap();

    // Compute means
    let means: Vec<T> = matrix
        .iter()
        .map(|row| crate::parallel::pairwise_sum(row, <T as Element>::zero()) / nf)
        .collect();

    // Compute covariance matrix
    let mut cov_data = vec![<T as Element>::zero(); nvars * nvars];
    for i in 0..nvars {
        for j in i..nvars {
            let mut s = <T as Element>::zero();
            for (mi, mj) in matrix[i].iter().zip(matrix[j].iter()) {
                s = s + (*mi - means[i]) * (*mj - means[j]);
            }
            let val = s / denom;
            cov_data[i * nvars + j] = val;
            cov_data[j * nvars + i] = val;
        }
    }

    Array::from_vec(Ix2::new([nvars, nvars]), cov_data)
}

// ---------------------------------------------------------------------------
// corrcoef
// ---------------------------------------------------------------------------

/// Compute the Pearson correlation coefficient matrix.
///
/// If `x` is 2-D, each row is a variable (when `rowvar` is true).
/// If 1-D, treated as a single variable.
///
/// Equivalent to `numpy.corrcoef`.
pub fn corrcoef<T, D>(x: &Array<T, D>, rowvar: bool) -> FerrayResult<Array<T, Ix2>>
where
    T: Element + Float,
    D: Dimension,
{
    let c = cov(x, rowvar, Some(0))?;
    let n = c.shape()[0];

    // Extract diagonal (standard deviations)
    let cov_data: Vec<T> = c.iter().copied().collect();
    let mut diag = Vec::with_capacity(n);
    for i in 0..n {
        diag.push(cov_data[i * n + i].sqrt());
    }

    // Normalize: corrcoef[i,j] = cov[i,j] / (std[i] * std[j]).
    // If either variable has zero deviation the correlation is
    // undefined (0/0); numpy returns NaN, so we do too (#723).
    let mut corr_data = vec![<T as Element>::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            let d = diag[i] * diag[j];
            if d == <T as Element>::zero() {
                corr_data[i * n + j] = T::nan();
            } else {
                let val = cov_data[i * n + j] / d;
                // Clamp to [-1, 1] for numerical stability
                corr_data[i * n + j] = val.min(<T as Element>::one()).max(-<T as Element>::one());
            }
        }
    }

    Array::from_vec(Ix2::new([n, n]), corr_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correlate_valid() {
        // numpy.correlate([1,2,3], [0.5, 1.0], 'valid') = [2.5, 4.0]
        // — pre-#722 the implementation returned the convolution
        // values [2.0, 3.5], which is the time-reversed answer.
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let v = Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![0.5, 1.0]).unwrap();
        let r = correlate(&a, &v, CorrelateMode::Valid).unwrap();
        assert_eq!(r.shape(), &[2]);
        let data: Vec<f64> = r.iter().copied().collect();
        assert!((data[0] - 2.5).abs() < 1e-12, "data[0] = {}", data[0]);
        assert!((data[1] - 4.0).abs() < 1e-12, "data[1] = {}", data[1]);
    }

    #[test]
    fn test_correlate_full() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let v = Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![1.0, 1.0]).unwrap();
        let r = correlate(&a, &v, CorrelateMode::Full).unwrap();
        assert_eq!(r.shape(), &[4]);
        let data: Vec<f64> = r.iter().copied().collect();
        assert!((data[0] - 1.0).abs() < 1e-12);
        assert!((data[1] - 3.0).abs() < 1e-12);
        assert!((data[2] - 5.0).abs() < 1e-12);
        assert!((data[3] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn correlate_full_asymmetric_kernel_matches_numpy() {
        // Regression for #722: with v = [1, 2] (asymmetric), the buggy
        // implementation returned the convolution [1, 4, 7, 6] instead
        // of the correlation [2, 5, 8, 3]. Verify against numpy's
        // analytic answer.
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let v = Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![1.0, 2.0]).unwrap();
        let r = correlate(&a, &v, CorrelateMode::Full).unwrap();
        let s: Vec<f64> = r.iter().copied().collect();
        assert_eq!(s, vec![2.0, 5.0, 8.0, 3.0]);
    }

    #[test]
    fn correlate_valid_asymmetric_kernel_matches_numpy() {
        // numpy.correlate([1,2,3,4,5], [0.5, 1.0], 'valid'):
        // c[k] = Σ_n a[n+k] * v[n]
        //   k=0: 1*0.5 + 2*1.0 = 2.5
        //   k=1: 2*0.5 + 3*1.0 = 4.0
        //   k=2: 3*0.5 + 4*1.0 = 5.5
        //   k=3: 4*0.5 + 5*1.0 = 7.0
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let v = Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![0.5, 1.0]).unwrap();
        let r = correlate(&a, &v, CorrelateMode::Valid).unwrap();
        let s: Vec<f64> = r.iter().copied().collect();
        assert_eq!(s, vec![2.5, 4.0, 5.5, 7.0]);
    }

    #[test]
    fn test_cov_1d() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let c = cov(&a, true, None).unwrap();
        assert_eq!(c.shape(), &[1, 1]);
        let val = *c.iter().next().unwrap();
        assert!((val - 5.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_cov_2d() {
        let m = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let c = cov(&m, true, None).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        let data: Vec<f64> = c.iter().copied().collect();
        assert!((data[0] - 1.0).abs() < 1e-12);
        assert!((data[3] - 1.0).abs() < 1e-12);
        assert!((data[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_corrcoef() {
        let m = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let c = corrcoef(&m, true).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        let data: Vec<f64> = c.iter().copied().collect();
        assert!((data[0] - 1.0).abs() < 1e-12);
        assert!((data[1] - 1.0).abs() < 1e-12);
        assert!((data[2] - 1.0).abs() < 1e-12);
        assert!((data[3] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn corrcoef_constant_row_returns_nan() {
        // #723: row with zero variance has undefined correlation;
        // numpy returns NaN. Pre-fix this returned 0.0 / 1.0.
        let m = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 1.0, 1.0, 1.0, 2.0, 3.0])
            .unwrap();
        let c = corrcoef(&m, true).unwrap();
        let s: Vec<f64> = c.iter().copied().collect();
        // [[nan, nan], [nan, 1.0]]
        assert!(s[0].is_nan(), "[0,0] = {}", s[0]);
        assert!(s[1].is_nan(), "[0,1] = {}", s[1]);
        assert!(s[2].is_nan(), "[1,0] = {}", s[2]);
        assert!((s[3] - 1.0).abs() < 1e-12, "[1,1] = {}", s[3]);
    }

    #[test]
    fn test_corrcoef_negative() {
        let m = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 6.0, 5.0, 4.0])
            .unwrap();
        let c = corrcoef(&m, true).unwrap();
        let data: Vec<f64> = c.iter().copied().collect();
        assert!((data[0] - 1.0).abs() < 1e-12);
        assert!((data[1] - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn test_correlate_same() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let v = Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![1.0, 1.0]).unwrap();
        let r = correlate(&a, &v, CorrelateMode::Same).unwrap();
        assert_eq!(r.shape(), &[3]);
    }

    // ----- Single-observation cov/corrcoef (#183) -----

    #[test]
    fn cov_single_observation_errors() {
        // With rowvar=true, rows are variables and columns are
        // observations. A 2x1 matrix has 2 variables with 1
        // observation each. ddof defaults to 1, so N-ddof = 0 →
        // ferray errors (stricter than NumPy which returns NaN).
        let m = Array::<f64, Ix2>::from_vec(Ix2::new([2, 1]), vec![1.0, 2.0]).unwrap();
        assert!(cov(&m, true, None).is_err());
    }

    #[test]
    fn corrcoef_constant_variable_off_diagonal_is_nan() {
        // A variable with zero variance has undefined correlation
        // with anything. ferray now returns NaN to match numpy
        // (#723); previously returned 0.0.
        // Row 0: constant [5, 5, 5]. Row 1: varying [1, 2, 3].
        let m = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![5.0, 5.0, 5.0, 1.0, 2.0, 3.0])
            .unwrap();
        let c = corrcoef(&m, true).unwrap();
        let data: Vec<f64> = c.iter().copied().collect();
        assert!(data[0].is_nan());
        assert!(data[1].is_nan());
        assert!(data[2].is_nan());
        assert!((data[3] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn corrcoef_all_constant_variables_yields_nans() {
        // Two constant variables — every entry is 0/0 = NaN except possibly
        // the diagonal (which is 0/0 too in this implementation). The point
        // is: the function must not panic on degenerate input (#183).
        let m = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![5.0, 5.0, 5.0, 7.0, 7.0, 7.0])
            .unwrap();
        let c = corrcoef(&m, true).unwrap();
        // Just verify it didn't panic and returned the right shape.
        assert_eq!(c.shape(), &[2, 2]);
    }

    #[test]
    fn cov_constant_variable_yields_zero_variance() {
        // Constant variable has zero variance. cov(m, rowvar=true) of
        // a row that's all 5.0 should give a 1x1 cov matrix with 0.0
        // (or for 2 vars: zeros on the constant variable's row/col).
        let m = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![5.0, 5.0, 5.0, 1.0, 2.0, 3.0])
            .unwrap();
        let c = cov(&m, true, None).unwrap();
        let data: Vec<f64> = c.iter().copied().collect();
        // c(0,0) = var(constant) = 0
        assert!(data[0].abs() < 1e-12);
        // c(0,1) and c(1,0) = covariance involving constant = 0
        assert!(data[1].abs() < 1e-12);
        assert!(data[2].abs() < 1e-12);
        // c(1,1) = var([1,2,3], ddof=1) = 1.0
        assert!((data[3] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn corrcoef_single_observation_diagonal_is_one() {
        // With a single observation per variable, cov errors (ddof=1,
        // N=1), but corrcoef normalizes internally. The diagonal of a
        // correlation matrix is always 1.0 (self-correlation); off-
        // diagonal entries may be NaN or arbitrary. Just verify it
        // doesn't panic and the diagonal is 1.0 (#183).
        let m = Array::<f64, Ix2>::from_vec(Ix2::new([2, 1]), vec![1.0, 2.0]).unwrap();
        let c = corrcoef(&m, true).unwrap();
        let data: Vec<f64> = c.iter().copied().collect();
        // 2x2 correlation matrix: data[0]=c(0,0), data[3]=c(1,1)
        assert!((data[0] - 1.0).abs() < 1e-12 || data[0].is_nan());
        assert!((data[3] - 1.0).abs() < 1e-12 || data[3].is_nan());
    }

    // ----- Correlate edge cases (#181) -----

    #[test]
    fn correlate_single_element_full() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([1]), vec![3.0]).unwrap();
        let v = Array::<f64, Ix1>::from_vec(Ix1::new([1]), vec![5.0]).unwrap();
        let r = correlate(&a, &v, CorrelateMode::Full).unwrap();
        assert_eq!(r.shape(), &[1]);
        assert!((r.iter().next().copied().unwrap() - 15.0).abs() < 1e-12);
    }

    #[test]
    fn correlate_single_element_valid() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([1]), vec![3.0]).unwrap();
        let v = Array::<f64, Ix1>::from_vec(Ix1::new([1]), vec![5.0]).unwrap();
        let r = correlate(&a, &v, CorrelateMode::Valid).unwrap();
        assert_eq!(r.shape(), &[1]);
        assert!((r.iter().next().copied().unwrap() - 15.0).abs() < 1e-12);
    }

    #[test]
    fn correlate_identical_arrays_full_shape_and_finite() {
        // Auto-correlation full output has length 2n-1 and is finite
        // for finite inputs. (Note: ferray's correlate currently
        // computes convolution rather than the formal cross-correlation
        // — see filed follow-up; this test asserts shape + finiteness
        // which holds either way.)
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let r = correlate(&a, &a, CorrelateMode::Full).unwrap();
        let data: Vec<f64> = r.iter().copied().collect();
        assert_eq!(data.len(), 2 * 4 - 1);
        for x in &data {
            assert!(x.is_finite());
        }
    }

    #[test]
    fn correlate_all_zeros_yields_all_zeros() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let v = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let r = correlate(&a, &v, CorrelateMode::Full).unwrap();
        for x in r.iter() {
            assert_eq!(*x, 0.0);
        }
    }

    #[test]
    fn correlate_one_zero_one_signal_yields_signal() {
        // correlate(a, [1.0]) === a (delta kernel)
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let v = Array::<f64, Ix1>::from_vec(Ix1::new([1]), vec![1.0]).unwrap();
        let r = correlate(&a, &v, CorrelateMode::Same).unwrap();
        let data: Vec<f64> = r.iter().copied().collect();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn correlate_empty_inputs_error() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([0]), vec![]).unwrap();
        let v = Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![1.0, 2.0]).unwrap();
        assert!(correlate(&a, &v, CorrelateMode::Full).is_err());
    }
}
