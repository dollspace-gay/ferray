// ferray-stats: scipy.stats statistical-test functions (#724).
//
// Provides pearsonr, spearmanr, ttest_1samp, ttest_ind, ks_2samp.
// p-values use the regularised incomplete beta function (Lentz's
// continued fraction) for the t-distribution CDF and the
// closed-form Kolmogorov-Smirnov series for ks_2samp. chi2 p-values
// (needing incomplete gamma) are tracked separately as #743.

use ferray_core::error::{FerrayError, FerrayResult};
use ferray_core::{Array, Dimension, Element};

// ---------------------------------------------------------------------------
// Regularised incomplete beta function I_x(a, b)
//
// Numerically stable evaluation via Lentz's continued fraction. Used
// for the t-distribution CDF when computing two-sided p-values.
// ---------------------------------------------------------------------------

/// `I_x(a, b)` = regularised incomplete beta function. Returns NaN
/// for inputs outside the valid domain.
fn betainc(a: f64, b: f64, x: f64) -> f64 {
    if !(0.0..=1.0).contains(&x) || a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return 1.0;
    }
    // Numerical-Recipes recipe: choose the side of the symmetry
    // identity that gives faster CF convergence.
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b)
        + a * x.ln()
        + b * (1.0 - x).ln())
    .exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

/// Stirling-approximation log-gamma. Accurate to ~1e-14 for x >= 1.
fn ln_gamma(x: f64) -> f64 {
    // Lanczos approximation with g = 5 (Numerical Recipes coefficients).
    const G: f64 = 5.0;
    const COEFFS: [f64; 7] = [
        1.000_000_000_190_015,
        76.180_091_729_471_46,
        -86.505_320_329_416_77,
        24.014_098_240_830_91,
        -1.231_739_572_450_155,
        0.001_208_650_973_866_179,
        -0.000_005_395_239_384_953,
    ];
    let mut sum = COEFFS[0];
    for (i, &c) in COEFFS.iter().enumerate().skip(1) {
        sum += c / (x + i as f64);
    }
    let half_log_2pi = 0.918_938_533_204_672_8;
    half_log_2pi + (x + 0.5) * (x + G + 0.5).ln() - (x + G + 0.5)
        + (sum / x).ln()
}

/// Continued-fraction tail used by `betainc`. Lentz's algorithm.
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAX_ITER: usize = 200;
    const EPS: f64 = 3e-16;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAX_ITER {
        let mf = m as f64;
        let m2 = 2.0 * mf;
        // Even step.
        let aa = mf * (b - mf) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        h *= d * c;
        // Odd step.
        let aa = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            return h;
        }
    }
    h
}

/// Two-sided p-value for a t-statistic with the given degrees of
/// freedom. Computed via the t-distribution CDF expressed in terms
/// of the regularised incomplete beta function.
fn t_two_sided_p(t: f64, df: f64) -> f64 {
    if !t.is_finite() || df <= 0.0 {
        return f64::NAN;
    }
    betainc(df / 2.0, 0.5, df / (df + t * t))
}

// ---------------------------------------------------------------------------
// Public statistical-test functions
// ---------------------------------------------------------------------------

/// Pearson product-moment correlation coefficient and two-sided
/// p-value (#724).
///
/// Equivalent to `scipy.stats.pearsonr(x, y)`. Returns `(r, p)`
/// where `r` is the sample correlation coefficient and `p` is the
/// two-sided p-value under the null hypothesis that the two
/// variables are uncorrelated. The p-value uses the t-distribution
/// with `n - 2` degrees of freedom, expressed in terms of the
/// regularised incomplete beta function.
///
/// # Errors
/// `FerrayError::InvalidValue` if either array has fewer than 2
/// elements or the lengths differ.
pub fn pearsonr<T, D>(x: &Array<T, D>, y: &Array<T, D>) -> FerrayResult<(f64, f64)>
where
    T: Element + Copy + Into<f64>,
    D: Dimension,
{
    let xs: Vec<f64> = x.iter().copied().map(Into::into).collect();
    let ys: Vec<f64> = y.iter().copied().map(Into::into).collect();
    if xs.len() != ys.len() {
        return Err(FerrayError::shape_mismatch(format!(
            "pearsonr: arrays have different lengths ({} vs {})",
            xs.len(),
            ys.len()
        )));
    }
    let n = xs.len();
    if n < 2 {
        return Err(FerrayError::invalid_value(
            "pearsonr: at least 2 observations required",
        ));
    }
    let nf = n as f64;
    let mx = xs.iter().sum::<f64>() / nf;
    let my = ys.iter().sum::<f64>() / nf;
    let mut sxx = 0.0_f64;
    let mut syy = 0.0_f64;
    let mut sxy = 0.0_f64;
    for (a, b) in xs.iter().zip(ys.iter()) {
        let dx = a - mx;
        let dy = b - my;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    let denom = (sxx * syy).sqrt();
    if denom == 0.0 {
        // Either x or y is constant — Pearson is undefined.
        return Ok((f64::NAN, f64::NAN));
    }
    let r = (sxy / denom).clamp(-1.0, 1.0);
    let df = nf - 2.0;
    let t = r * (df / (1.0 - r * r).max(f64::MIN_POSITIVE)).sqrt();
    let p = t_two_sided_p(t, df);
    Ok((r, p))
}

/// Spearman rank correlation coefficient and two-sided p-value (#724).
///
/// Equivalent to `scipy.stats.spearmanr(x, y)`. Computes Pearson
/// correlation on the ranks of the inputs (with average ranks for
/// ties). Returns `(rho, p)` using the same t-distribution p-value
/// approximation as Pearson with `df = n - 2`.
///
/// # Errors
/// Same as `pearsonr`.
pub fn spearmanr<T, D>(x: &Array<T, D>, y: &Array<T, D>) -> FerrayResult<(f64, f64)>
where
    T: Element + Copy + PartialOrd + Into<f64>,
    D: Dimension,
{
    let xs: Vec<f64> = x.iter().copied().map(Into::into).collect();
    let ys: Vec<f64> = y.iter().copied().map(Into::into).collect();
    if xs.len() != ys.len() {
        return Err(FerrayError::shape_mismatch(format!(
            "spearmanr: arrays have different lengths ({} vs {})",
            xs.len(),
            ys.len()
        )));
    }
    if xs.len() < 2 {
        return Err(FerrayError::invalid_value(
            "spearmanr: at least 2 observations required",
        ));
    }
    let rx = average_ranks(&xs);
    let ry = average_ranks(&ys);
    pearsonr_f64_slices(&rx, &ry)
}

/// Pearson correlation on already-prepared f64 slices (shared
/// helper for spearmanr).
fn pearsonr_f64_slices(xs: &[f64], ys: &[f64]) -> FerrayResult<(f64, f64)> {
    let n = xs.len();
    if n < 2 {
        return Err(FerrayError::invalid_value(
            "pearsonr: at least 2 observations required",
        ));
    }
    let nf = n as f64;
    let mx = xs.iter().sum::<f64>() / nf;
    let my = ys.iter().sum::<f64>() / nf;
    let mut sxx = 0.0_f64;
    let mut syy = 0.0_f64;
    let mut sxy = 0.0_f64;
    for (a, b) in xs.iter().zip(ys.iter()) {
        let dx = a - mx;
        let dy = b - my;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    let denom = (sxx * syy).sqrt();
    if denom == 0.0 {
        return Ok((f64::NAN, f64::NAN));
    }
    let r = (sxy / denom).clamp(-1.0, 1.0);
    let df = nf - 2.0;
    let t = r * (df / (1.0 - r * r).max(f64::MIN_POSITIVE)).sqrt();
    let p = t_two_sided_p(t, df);
    Ok((r, p))
}

/// Compute average ranks for ties.
fn average_ranks(xs: &[f64]) -> Vec<f64> {
    let n = xs.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        xs[a]
            .partial_cmp(&xs[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut ranks = vec![0.0_f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && xs[idx[j + 1]] == xs[idx[i]] {
            j += 1;
        }
        // Average rank for the tie group i..=j (1-based).
        let avg = (i as f64 + j as f64) / 2.0 + 1.0;
        for k in i..=j {
            ranks[idx[k]] = avg;
        }
        i = j + 1;
    }
    ranks
}

/// One-sample Student's t-test (#724).
///
/// Equivalent to `scipy.stats.ttest_1samp(a, popmean)`. Returns
/// `(t, p)` where `t = (mean(a) - popmean) / (s / sqrt(n))` with
/// `s` the sample standard deviation (ddof=1) and `p` the
/// two-sided p-value with `n - 1` degrees of freedom.
///
/// # Errors
/// `FerrayError::InvalidValue` if `a` has fewer than 2 elements.
pub fn ttest_1samp<T, D>(a: &Array<T, D>, popmean: f64) -> FerrayResult<(f64, f64)>
where
    T: Element + Copy + Into<f64>,
    D: Dimension,
{
    let xs: Vec<f64> = a.iter().copied().map(Into::into).collect();
    let n = xs.len();
    if n < 2 {
        return Err(FerrayError::invalid_value(
            "ttest_1samp: at least 2 observations required",
        ));
    }
    let nf = n as f64;
    let mean = xs.iter().sum::<f64>() / nf;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (nf - 1.0);
    if var == 0.0 {
        return Ok((f64::NAN, f64::NAN));
    }
    let std = var.sqrt();
    let se = std / nf.sqrt();
    let t = (mean - popmean) / se;
    let p = t_two_sided_p(t, nf - 1.0);
    Ok((t, p))
}

/// Independent (Welch's) two-sample t-test (#724).
///
/// Equivalent to `scipy.stats.ttest_ind(a, b, equal_var=False)`.
/// Returns `(t, p)`. The Welch–Satterthwaite degrees-of-freedom
/// approximation is used for unequal variances.
///
/// # Errors
/// `FerrayError::InvalidValue` if either sample has fewer than 2
/// elements.
pub fn ttest_ind<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<(f64, f64)>
where
    T: Element + Copy + Into<f64>,
    D: Dimension,
{
    let xs: Vec<f64> = a.iter().copied().map(Into::into).collect();
    let ys: Vec<f64> = b.iter().copied().map(Into::into).collect();
    if xs.len() < 2 || ys.len() < 2 {
        return Err(FerrayError::invalid_value(
            "ttest_ind: each sample needs at least 2 observations",
        ));
    }
    let n1 = xs.len() as f64;
    let n2 = ys.len() as f64;
    let m1 = xs.iter().sum::<f64>() / n1;
    let m2 = ys.iter().sum::<f64>() / n2;
    let v1 = xs.iter().map(|x| (x - m1).powi(2)).sum::<f64>() / (n1 - 1.0);
    let v2 = ys.iter().map(|x| (x - m2).powi(2)).sum::<f64>() / (n2 - 1.0);
    if v1 == 0.0 && v2 == 0.0 {
        return Ok((f64::NAN, f64::NAN));
    }
    let se = (v1 / n1 + v2 / n2).sqrt();
    let t = (m1 - m2) / se;
    let num = (v1 / n1 + v2 / n2).powi(2);
    let denom = (v1 / n1).powi(2) / (n1 - 1.0) + (v2 / n2).powi(2) / (n2 - 1.0);
    let df = num / denom;
    let p = t_two_sided_p(t, df);
    Ok((t, p))
}

/// Two-sample Kolmogorov–Smirnov test (#724).
///
/// Equivalent to `scipy.stats.ks_2samp(a, b)`. Returns
/// `(D, p)` where `D` is the maximum absolute difference between
/// the two empirical CDFs and `p` is the two-sided asymptotic
/// p-value via the closed-form Smirnov series.
///
/// # Errors
/// `FerrayError::InvalidValue` if either sample is empty.
pub fn ks_2samp<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<(f64, f64)>
where
    T: Element + Copy + PartialOrd + Into<f64>,
    D: Dimension,
{
    let mut xs: Vec<f64> = a.iter().copied().map(Into::into).collect();
    let mut ys: Vec<f64> = b.iter().copied().map(Into::into).collect();
    if xs.is_empty() || ys.is_empty() {
        return Err(FerrayError::invalid_value(
            "ks_2samp: both samples must be non-empty",
        ));
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // KS statistic: max over the union of sample points of
    // |F_n1(x) - F_n2(x)|. Computed by evaluating each empirical
    // CDF at every union point via partition_point (== numpy
    // searchsorted side='right'). Matches scipy.stats.ks_2samp.
    let n1 = xs.len() as f64;
    let n2 = ys.len() as f64;
    let mut union: Vec<f64> = Vec::with_capacity(xs.len() + ys.len());
    union.extend_from_slice(&xs);
    union.extend_from_slice(&ys);
    union.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut d = 0.0_f64;
    for &v in &union {
        let i = xs.partition_point(|x| *x <= v);
        let j = ys.partition_point(|y| *y <= v);
        let here = (i as f64 / n1 - j as f64 / n2).abs();
        if here > d {
            d = here;
        }
    }

    // Asymptotic Kolmogorov distribution for the p-value.
    let en = (n1 * n2 / (n1 + n2)).sqrt();
    let lambda = (en + 0.12 + 0.11 / en) * d;
    let p = ks_p_kolmogorov(lambda).clamp(0.0, 1.0);
    Ok((d, p))
}

/// Kolmogorov asymptotic survival series:
/// Q(λ) = 2 Σ_{k=1..∞} (-1)^(k-1) exp(-2 k² λ²).
fn ks_p_kolmogorov(lambda: f64) -> f64 {
    if lambda <= 0.0 {
        return 1.0;
    }
    let mut sum = 0.0_f64;
    let mut sign = 1.0_f64;
    let mut prev = 0.0_f64;
    for k in 1..=200 {
        let kf = k as f64;
        let term = sign * (-2.0 * kf * kf * lambda * lambda).exp();
        sum += term;
        if (sum - prev).abs() < 1e-15 {
            break;
        }
        prev = sum;
        sign = -sign;
    }
    2.0 * sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::Ix1;

    fn arr1(v: Vec<f64>) -> Array<f64, Ix1> {
        let n = v.len();
        Array::from_vec(Ix1::new([n]), v).unwrap()
    }

    #[test]
    fn pearsonr_perfect_positive() {
        let x = arr1(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = arr1(vec![2.0, 4.0, 6.0, 8.0, 10.0]);
        let (r, p) = pearsonr(&x, &y).unwrap();
        assert!((r - 1.0).abs() < 1e-12);
        // Perfect correlation → p effectively 0.
        assert!(p < 1e-6);
    }

    #[test]
    fn pearsonr_perfect_negative() {
        let x = arr1(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = arr1(vec![5.0, 4.0, 3.0, 2.0, 1.0]);
        let (r, p) = pearsonr(&x, &y).unwrap();
        assert!((r + 1.0).abs() < 1e-12);
        assert!(p < 1e-6);
    }

    #[test]
    fn pearsonr_uncorrelated_high_p() {
        // Random-ish numbers with no linear relationship → r small, p large.
        let x = arr1(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let y = arr1(vec![7.5, -1.0, 2.0, 5.5, -3.0, 6.0, 0.5, 1.5]);
        let (r, p) = pearsonr(&x, &y).unwrap();
        assert!(r.abs() < 1.0);
        assert!(p > 0.0 && p <= 1.0);
    }

    #[test]
    fn pearsonr_constant_returns_nan() {
        let x = arr1(vec![3.0, 3.0, 3.0, 3.0]);
        let y = arr1(vec![1.0, 2.0, 3.0, 4.0]);
        let (r, p) = pearsonr(&x, &y).unwrap();
        assert!(r.is_nan());
        assert!(p.is_nan());
    }

    #[test]
    fn pearsonr_length_mismatch_errors() {
        let x = arr1(vec![1.0, 2.0]);
        let y = arr1(vec![1.0, 2.0, 3.0]);
        assert!(pearsonr(&x, &y).is_err());
    }

    #[test]
    fn spearmanr_monotonic_nonlinear_is_one() {
        // y is a monotonic increasing nonlinear function of x.
        let x = arr1(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = arr1(vec![1.0, 4.0, 9.0, 16.0, 25.0]);
        let (rho, _) = spearmanr(&x, &y).unwrap();
        assert!((rho - 1.0).abs() < 1e-12);
    }

    #[test]
    fn ttest_1samp_known() {
        // Under H_0: μ = 5, sample [4, 5, 6] has mean 5, var 1, std 1.
        // SE = 1/√3, t = 0.
        let a = arr1(vec![4.0, 5.0, 6.0]);
        let (t, p) = ttest_1samp(&a, 5.0).unwrap();
        assert!(t.abs() < 1e-12);
        // Two-sided p for t = 0 is exactly 1.
        assert!((p - 1.0).abs() < 1e-12);
    }

    #[test]
    fn ttest_1samp_strong_signal() {
        // Sample very different from popmean — p should be tiny.
        let a = arr1(vec![10.0, 11.0, 9.0, 10.5, 9.5, 10.0]);
        let (_, p) = ttest_1samp(&a, 0.0).unwrap();
        assert!(p < 1e-6);
    }

    #[test]
    fn ttest_ind_identical_samples_t_zero() {
        let a = arr1(vec![1.0, 2.0, 3.0, 4.0]);
        let b = arr1(vec![1.0, 2.0, 3.0, 4.0]);
        let (t, p) = ttest_ind(&a, &b).unwrap();
        assert!(t.abs() < 1e-12);
        assert!((p - 1.0).abs() < 1e-12);
    }

    #[test]
    fn ttest_ind_strong_difference() {
        // Welch's t for [1..5] vs [10..14]: t = -9, df ≈ 8.
        // p ≈ 1.7e-5 (closed-form via the incomplete-beta tail).
        let a = arr1(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let b = arr1(vec![10.0, 11.0, 12.0, 13.0, 14.0]);
        let (_, p) = ttest_ind(&a, &b).unwrap();
        assert!(p < 1e-3, "p too large: {p}");
    }

    #[test]
    fn ks_2samp_identical_samples_d_zero() {
        let a = arr1(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let b = arr1(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let (d, p) = ks_2samp(&a, &b).unwrap();
        assert!(d < 1e-12);
        assert!(p > 0.99);
    }

    #[test]
    fn ks_2samp_clearly_different_distributions() {
        let a = arr1(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]);
        let b = arr1(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]);
        let (d, p) = ks_2samp(&a, &b).unwrap();
        // No overlap → D = 1.0.
        assert!((d - 1.0).abs() < 1e-12);
        assert!(p < 0.01);
    }

    #[test]
    fn ks_2samp_partial_overlap() {
        let a = arr1(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let b = arr1(vec![3.0, 4.0, 5.0, 6.0, 7.0]);
        let (d, _p) = ks_2samp(&a, &b).unwrap();
        // Visual inspection: at x=2, F(x)=2/5=0.4, G(x)=0/5=0.0 → diff 0.4.
        assert!(d > 0.0 && d < 1.0);
    }

    #[test]
    fn empty_inputs_error() {
        let empty = arr1(vec![]);
        let one = arr1(vec![1.0]);
        let two = arr1(vec![1.0, 2.0]);
        assert!(pearsonr(&one, &one).is_err());
        assert!(spearmanr(&one, &one).is_err());
        assert!(ttest_1samp(&one, 0.0).is_err());
        assert!(ttest_ind(&one, &two).is_err());
        assert!(ks_2samp(&empty, &two).is_err());
    }

    #[test]
    fn betainc_known_values() {
        // I_{0.5}(1, 1) = 0.5
        assert!((betainc(1.0, 1.0, 0.5) - 0.5).abs() < 1e-10);
        // I_0 = 0, I_1 = 1
        assert_eq!(betainc(2.0, 3.0, 0.0), 0.0);
        assert_eq!(betainc(2.0, 3.0, 1.0), 1.0);
    }
}
