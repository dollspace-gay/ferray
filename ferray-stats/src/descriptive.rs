// ferray-stats: Descriptive statistics — skew, kurtosis, zscore, mode,
// iqr, sem, gmean, hmean (#470).
//
// scipy.stats parity for the descriptive set the issue calls out as
// "at least these would be expected". Statistical *tests* (ttest,
// ks_2samp, chi2_contingency, pearsonr, spearmanr) need cumulative
// distribution functions and are a follow-up.

use ferray_core::error::{FerrayError, FerrayResult};
use ferray_core::{Array, Dimension, Element, IxDyn};
use num_traits::Float;

/// Sample skewness (third standardized moment).
///
/// Returns `E[((X - μ)/σ)^3]` computed with the population standard
/// deviation (denominator `n`, matching scipy's default
/// `bias=True` behavior).
///
/// # Errors
/// `FerrayError::InvalidValue` if the array is empty or constant
/// (variance is exactly zero).
pub fn skew<T, D>(a: &Array<T, D>) -> FerrayResult<f64>
where
    T: Element + Copy + Into<f64>,
    D: Dimension,
{
    let data: Vec<f64> = a.iter().copied().map(Into::into).collect();
    if data.is_empty() {
        return Err(FerrayError::invalid_value("skew on empty array"));
    }
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let m2 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let m3 = data.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n;
    if m2 == 0.0 {
        return Err(FerrayError::invalid_value(
            "skew is undefined for a constant array (zero variance)",
        ));
    }
    Ok(m3 / m2.powf(1.5))
}

/// Sample kurtosis.
///
/// If `fisher` is `true` (default in scipy), returns excess kurtosis
/// (subtracts 3 so a normal distribution has kurtosis 0). Otherwise
/// returns Pearson kurtosis (`E[((X - μ)/σ)^4]`, normal = 3).
///
/// Computed with the population standard deviation (denominator `n`,
/// `bias=True` in scipy).
///
/// # Errors
/// `FerrayError::InvalidValue` if the array is empty or constant.
pub fn kurtosis<T, D>(a: &Array<T, D>, fisher: bool) -> FerrayResult<f64>
where
    T: Element + Copy + Into<f64>,
    D: Dimension,
{
    let data: Vec<f64> = a.iter().copied().map(Into::into).collect();
    if data.is_empty() {
        return Err(FerrayError::invalid_value("kurtosis on empty array"));
    }
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let m2 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let m4 = data.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / n;
    if m2 == 0.0 {
        return Err(FerrayError::invalid_value(
            "kurtosis is undefined for a constant array (zero variance)",
        ));
    }
    let pearson = m4 / (m2 * m2);
    Ok(if fisher { pearson - 3.0 } else { pearson })
}

/// Per-element z-score: `(x - mean) / std`.
///
/// Uses the population standard deviation (denominator `n`, scipy's
/// `ddof=0` default). Returns an array with the same shape as the
/// input.
///
/// # Errors
/// `FerrayError::InvalidValue` if the array is empty or constant.
pub fn zscore<T, D>(a: &Array<T, D>) -> FerrayResult<Array<f64, IxDyn>>
where
    T: Element + Copy + Into<f64>,
    D: Dimension,
{
    let data: Vec<f64> = a.iter().copied().map(Into::into).collect();
    if data.is_empty() {
        return Err(FerrayError::invalid_value("zscore on empty array"));
    }
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    if var == 0.0 {
        return Err(FerrayError::invalid_value(
            "zscore is undefined for a constant array (zero variance)",
        ));
    }
    let std = var.sqrt();
    let out: Vec<f64> = data.iter().map(|x| (x - mean) / std).collect();
    let shape: Vec<usize> = a.shape().to_vec();
    Array::<f64, IxDyn>::from_vec(IxDyn::new(&shape), out)
}

/// Result of [`mode`]: the most-frequent value and its count.
#[derive(Debug, Clone, Copy)]
pub struct ModeResult<T> {
    /// The most frequent value (smallest one wins on ties).
    pub value: T,
    /// Number of occurrences in the input.
    pub count: u64,
}

/// Most-frequent value in an array (mode).
///
/// On ties, returns the smallest value (matches scipy's default).
/// Compares values via `partial_cmp`; types like `f64` work but NaN
/// elements are ignored.
///
/// # Errors
/// `FerrayError::InvalidValue` if the array is empty or contains
/// only NaN.
pub fn mode<T, D>(a: &Array<T, D>) -> FerrayResult<ModeResult<T>>
where
    T: Element + Copy + PartialOrd,
    D: Dimension,
{
    if a.is_empty() {
        return Err(FerrayError::invalid_value("mode on empty array"));
    }
    // Sort (excluding NaN-like values that fail partial_cmp). After
    // sorting, ties are run-length encoded: count consecutive equal
    // elements, track the longest run; tiebreak by the smaller value.
    let mut data: Vec<T> = a
        .iter()
        .copied()
        .filter(|v| v.partial_cmp(v).is_some())
        .collect();
    if data.is_empty() {
        return Err(FerrayError::invalid_value(
            "mode on array with no comparable values",
        ));
    }
    data.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));

    let mut best_val = data[0];
    let mut best_count: u64 = 1;
    let mut cur_val = data[0];
    let mut cur_count: u64 = 1;
    for &v in &data[1..] {
        if v.partial_cmp(&cur_val) == Some(std::cmp::Ordering::Equal) {
            cur_count += 1;
        } else {
            if cur_count > best_count {
                best_count = cur_count;
                best_val = cur_val;
            }
            cur_val = v;
            cur_count = 1;
        }
    }
    if cur_count > best_count {
        best_count = cur_count;
        best_val = cur_val;
    }
    Ok(ModeResult {
        value: best_val,
        count: best_count,
    })
}

/// Interquartile range: Q75 − Q25.
///
/// Equivalent to `scipy.stats.iqr` with the default linear
/// interpolation method.
///
/// # Errors
/// `FerrayError::InvalidValue` if the array is empty.
pub fn iqr<T, D>(a: &Array<T, D>) -> FerrayResult<T>
where
    T: Element + Float,
    D: Dimension,
{
    let q75 = crate::reductions::quantile::quantile(
        a,
        T::from(0.75).expect("0.75 must round-trip to T"),
        None,
    )?;
    let q25 = crate::reductions::quantile::quantile(
        a,
        T::from(0.25).expect("0.25 must round-trip to T"),
        None,
    )?;
    let q75v = *q75.as_slice().unwrap().first().unwrap();
    let q25v = *q25.as_slice().unwrap().first().unwrap();
    Ok(q75v - q25v)
}

/// Standard error of the mean: `std / sqrt(n)`.
///
/// Uses the sample standard deviation (denominator `n - 1`,
/// matching scipy's `ddof=1` default).
///
/// # Errors
/// `FerrayError::InvalidValue` if the array has fewer than 2
/// elements.
pub fn sem<T, D>(a: &Array<T, D>) -> FerrayResult<f64>
where
    T: Element + Copy + Into<f64>,
    D: Dimension,
{
    let data: Vec<f64> = a.iter().copied().map(Into::into).collect();
    let n = data.len();
    if n < 2 {
        return Err(FerrayError::invalid_value(
            "sem requires at least 2 elements",
        ));
    }
    let nf = n as f64;
    let mean = data.iter().sum::<f64>() / nf;
    let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (nf - 1.0);
    Ok((var / nf).sqrt())
}

/// Geometric mean: `(prod x_i) ^ (1/n)`.
///
/// Uses log-space to avoid overflow on large `n`. All elements must
/// be strictly positive.
///
/// # Errors
/// `FerrayError::InvalidValue` if the array is empty or contains a
/// non-positive value.
pub fn gmean<T, D>(a: &Array<T, D>) -> FerrayResult<f64>
where
    T: Element + Copy + Into<f64>,
    D: Dimension,
{
    let data: Vec<f64> = a.iter().copied().map(Into::into).collect();
    if data.is_empty() {
        return Err(FerrayError::invalid_value("gmean on empty array"));
    }
    let mut log_sum = 0.0_f64;
    for &x in &data {
        if x <= 0.0 {
            return Err(FerrayError::invalid_value(
                "gmean requires all elements > 0",
            ));
        }
        log_sum += x.ln();
    }
    Ok((log_sum / data.len() as f64).exp())
}

/// Harmonic mean: `n / sum(1/x_i)`.
///
/// All elements must be strictly positive.
///
/// # Errors
/// `FerrayError::InvalidValue` if the array is empty or contains a
/// non-positive value.
pub fn hmean<T, D>(a: &Array<T, D>) -> FerrayResult<f64>
where
    T: Element + Copy + Into<f64>,
    D: Dimension,
{
    let data: Vec<f64> = a.iter().copied().map(Into::into).collect();
    if data.is_empty() {
        return Err(FerrayError::invalid_value("hmean on empty array"));
    }
    let mut recip_sum = 0.0_f64;
    for &x in &data {
        if x <= 0.0 {
            return Err(FerrayError::invalid_value(
                "hmean requires all elements > 0",
            ));
        }
        recip_sum += 1.0 / x;
    }
    Ok(data.len() as f64 / recip_sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::{Ix1, Ix2};

    #[test]
    fn skew_symmetric_zero() {
        let a =
            Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![-2.0, -1.0, 0.0, 1.0, 2.0]).unwrap();
        assert!(skew(&a).unwrap().abs() < 1e-12);
    }

    #[test]
    fn skew_right_skewed_positive() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([6]), vec![1.0, 1.0, 1.0, 1.0, 1.0, 10.0])
            .unwrap();
        assert!(skew(&a).unwrap() > 0.0);
    }

    #[test]
    fn kurtosis_normal_fisher_near_zero() {
        // Symmetric narrow distribution: kurtosis (Fisher) close to 0
        // is too strong a guarantee for n=5, but Fisher version of an
        // arithmetic progression around 0 is a fixed value (-1.3) so
        // we check for the known answer.
        let a =
            Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![-2.0, -1.0, 0.0, 1.0, 2.0]).unwrap();
        let k = kurtosis(&a, true).unwrap();
        assert!((k - (-1.3)).abs() < 1e-12, "Fisher kurtosis = {k}");
    }

    #[test]
    fn kurtosis_pearson_is_fisher_plus_3() {
        let a =
            Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![-2.0, -1.0, 0.0, 1.0, 2.0]).unwrap();
        let f = kurtosis(&a, true).unwrap();
        let p = kurtosis(&a, false).unwrap();
        assert!((p - (f + 3.0)).abs() < 1e-12);
    }

    #[test]
    fn zscore_has_mean_zero_std_one() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let z = zscore(&a).unwrap();
        let s = z.as_slice().unwrap();
        let n = s.len() as f64;
        let mean: f64 = s.iter().sum::<f64>() / n;
        let std = (s.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();
        assert!(mean.abs() < 1e-12);
        assert!((std - 1.0).abs() < 1e-12);
    }

    #[test]
    fn zscore_constant_errors() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![3.0; 4]).unwrap();
        assert!(zscore(&a).is_err());
    }

    #[test]
    fn zscore_2d_preserves_shape() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let z = zscore(&a).unwrap();
        assert_eq!(z.shape(), &[2, 3]);
    }

    #[test]
    fn mode_picks_most_frequent() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([7]), vec![1, 2, 2, 3, 3, 3, 4]).unwrap();
        let m = mode(&a).unwrap();
        assert_eq!(m.value, 3);
        assert_eq!(m.count, 3);
    }

    #[test]
    fn mode_tiebreak_smallest_wins() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([6]), vec![5, 5, 7, 7, 1, 1]).unwrap();
        let m = mode(&a).unwrap();
        assert_eq!(m.value, 1);
        assert_eq!(m.count, 2);
    }

    #[test]
    fn iqr_50_percent_span() {
        // Data 1..9: Q25 = 2.75, Q75 = 6.25 → IQR = 3.5 (numpy default).
        let a = Array::<f64, Ix1>::from_vec(
            Ix1::new([8]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap();
        let v = iqr(&a).unwrap();
        assert!((v - 3.5).abs() < 1e-12, "IQR = {v}");
    }

    #[test]
    fn sem_constant_data_is_zero() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![3.0; 5]).unwrap();
        assert!(sem(&a).unwrap().abs() < 1e-12);
    }

    #[test]
    fn sem_known_value() {
        // [1, 2, 3, 4, 5]: mean=3, ddof=1 var = 10/4 = 2.5,
        // std = sqrt(2.5), sem = std / sqrt(5) ≈ 0.7071...
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let s = sem(&a).unwrap();
        let want = (2.5_f64 / 5.0).sqrt();
        assert!((s - want).abs() < 1e-12, "sem = {s}, want {want}");
    }

    #[test]
    fn gmean_known_value() {
        // gmean(1, 2, 4, 8) = (1*2*4*8)^(1/4) = 64^0.25 = 2*sqrt(2)
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 4.0, 8.0]).unwrap();
        let g = gmean(&a).unwrap();
        let want = 2.0_f64.sqrt() * 2.0;
        assert!((g - want).abs() < 1e-12, "gmean = {g}, want {want}");
    }

    #[test]
    fn gmean_rejects_nonpositive() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 0.0, 4.0]).unwrap();
        assert!(gmean(&a).is_err());
    }

    #[test]
    fn hmean_known_value() {
        // hmean(1, 2, 4) = 3 / (1 + 0.5 + 0.25) = 3 / 1.75 = 12/7
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 4.0]).unwrap();
        let h = hmean(&a).unwrap();
        assert!((h - 12.0 / 7.0).abs() < 1e-12, "hmean = {h}");
    }

    #[test]
    fn hmean_rejects_nonpositive() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, -1.0, 2.0]).unwrap();
        assert!(hmean(&a).is_err());
    }

    #[test]
    fn empty_array_errors_across_helpers() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([0]), Vec::new()).unwrap();
        assert!(skew(&a).is_err());
        assert!(kurtosis(&a, true).is_err());
        assert!(zscore(&a).is_err());
        assert!(mode(&a).is_err());
        assert!(gmean(&a).is_err());
        assert!(hmean(&a).is_err());
        assert!(sem(&a).is_err());
    }
}
