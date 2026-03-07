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
    let v_data: Vec<T> = v.iter().copied().collect();
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

    // Normalize: corrcoef[i,j] = cov[i,j] / (std[i] * std[j])
    let mut corr_data = vec![<T as Element>::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            let d = diag[i] * diag[j];
            if d == <T as Element>::zero() {
                corr_data[i * n + j] = if i == j {
                    <T as Element>::one()
                } else {
                    <T as Element>::zero()
                };
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
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let v = Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![0.5, 1.0]).unwrap();
        let r = correlate(&a, &v, CorrelateMode::Valid).unwrap();
        assert_eq!(r.shape(), &[2]);
        let data: Vec<f64> = r.iter().copied().collect();
        assert!((data[0] - 2.0).abs() < 1e-12);
        assert!((data[1] - 3.5).abs() < 1e-12);
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
}
