// ferray-ma: Ufunc support wrappers (REQ-12)
//
// Wrapper functions that accept MaskedArray and call underlying ferray-ufunc
// operations on the data, then propagate masks. Masked elements are skipped;
// their output positions retain the masked array's `fill_value`.

use ferray_core::dimension::Dimension;
use ferray_core::dtype::Element;
use ferray_core::error::FerrayResult;
use num_traits::Float;

use ferray_core::Array;
use ferray_core::error::FerrayError;

use crate::MaskedArray;
use crate::arithmetic::{masked_binary_op, masked_unary_op};

// ---------------------------------------------------------------------------
// Domain-aware ufunc wrappers (#503)
//
// NumPy's masked ufuncs have domain checking: `ma.log(masked_array)`
// automatically masks elements that are out of the log domain
// (negative values). Similarly `ma.sqrt` masks negative inputs,
// `ma.divide` masks zero denominators, `ma.arcsin`/`ma.arccos` mask
// |x| > 1 inputs. The masked positions in the result combine the
// existing mask with a freshly-computed "domain mask" via OR.
//
// The plain `log`/`sqrt`/`arcsin`/… wrappers above call the raw Float
// method, so an in-domain masked element whose *underlying data* is
// out of domain silently produces NaN in the result data — annoying
// for pipelines that assume "masked = bad" but "unmasked = valid".
// The `_domain` variants fix that by computing a new mask = old_mask
// || !in_domain(x) and substituting the fill_value at positions
// newly marked as masked.
// ---------------------------------------------------------------------------

/// Apply a unary function to a masked array, additionally masking any
/// position where the underlying data is out of the function's domain.
///
/// `in_domain(x)` should return `true` when `x` is a valid input to
/// `f`. For any unmasked position where `in_domain(x)` returns
/// `false`, the result mask is set to `true` and the result data
/// carries the masked array's `fill_value`.
///
/// This is the generic helper backing [`log_domain`], [`sqrt_domain`],
/// [`arcsin_domain`], [`arccos_domain`] (#503).
pub fn masked_unary_domain<T, D, F, Dom>(
    ma: &MaskedArray<T, D>,
    f: F,
    in_domain: Dom,
) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Copy,
    D: Dimension,
    F: Fn(T) -> T,
    Dom: Fn(T) -> bool,
{
    let fill = ma.fill_value();
    let n = ma.size();
    let mut data_out: Vec<T> = Vec::with_capacity(n);
    let mut mask_out: Vec<bool> = Vec::with_capacity(n);
    for (&v, &m) in ma.data().iter().zip(ma.mask().iter()) {
        // Mask if already masked OR the domain predicate rejects v.
        let should_mask = m || !in_domain(v);
        if should_mask {
            data_out.push(fill);
            mask_out.push(true);
        } else {
            data_out.push(f(v));
            mask_out.push(false);
        }
    }
    let data_arr = Array::from_vec(ma.dim().clone(), data_out)?;
    let mask_arr = Array::from_vec(ma.dim().clone(), mask_out)?;
    let mut out = MaskedArray::new(data_arr, mask_arr)?;
    out.set_fill_value(fill);
    Ok(out)
}

/// Apply a binary function to two masked arrays, additionally masking
/// any position where `in_domain(a, b)` returns `false`.
///
/// The result's mask is `old_a_mask OR old_b_mask OR !in_domain(a, b)`.
/// Used by [`divide_domain`] to auto-mask zero denominators.
///
/// Requires same shape — broadcasting domain-aware ops is left as a
/// follow-up because the per-position domain check interacts awkwardly
/// with broadcast strides.
pub fn masked_binary_domain<T, D, F, Dom>(
    a: &MaskedArray<T, D>,
    b: &MaskedArray<T, D>,
    f: F,
    in_domain: Dom,
    op_name: &str,
) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Copy,
    D: Dimension,
    F: Fn(T, T) -> T,
    Dom: Fn(T, T) -> bool,
{
    if a.shape() != b.shape() {
        return Err(FerrayError::shape_mismatch(format!(
            "{op_name}: shapes {:?} and {:?} differ (broadcasting not supported for domain-aware ops)",
            a.shape(),
            b.shape()
        )));
    }
    let fill = a.fill_value();
    let n = a.size();
    let mut data_out: Vec<T> = Vec::with_capacity(n);
    let mut mask_out: Vec<bool> = Vec::with_capacity(n);
    for (((&x, &y), &ma_bit), &mb_bit) in a
        .data()
        .iter()
        .zip(b.data().iter())
        .zip(a.mask().iter())
        .zip(b.mask().iter())
    {
        // Mask the result if either input is already masked OR the
        // domain predicate rejects the pair.
        let should_mask = ma_bit || mb_bit || !in_domain(x, y);
        if should_mask {
            data_out.push(fill);
            mask_out.push(true);
        } else {
            data_out.push(f(x, y));
            mask_out.push(false);
        }
    }
    let data_arr = Array::from_vec(a.dim().clone(), data_out)?;
    let mask_arr = Array::from_vec(a.dim().clone(), mask_out)?;
    let mut out = MaskedArray::new(data_arr, mask_arr)?;
    out.set_fill_value(fill);
    Ok(out)
}

/// Natural log with auto-masking of non-positive inputs.
///
/// Equivalent to `numpy.ma.log`. Any unmasked element `x <= 0` is
/// added to the result mask and replaced with the fill value in the
/// result data.
pub fn log_domain<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    let zero = <T as Element>::zero();
    masked_unary_domain(ma, T::ln, move |x| x > zero)
}

/// Base-2 log with auto-masking of non-positive inputs.
pub fn log2_domain<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    let zero = <T as Element>::zero();
    masked_unary_domain(ma, T::log2, move |x| x > zero)
}

/// Base-10 log with auto-masking of non-positive inputs.
pub fn log10_domain<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    let zero = <T as Element>::zero();
    masked_unary_domain(ma, T::log10, move |x| x > zero)
}

/// Square root with auto-masking of negative inputs.
///
/// Equivalent to `numpy.ma.sqrt`. Any unmasked element `x < 0` is
/// added to the result mask.
pub fn sqrt_domain<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    let zero = <T as Element>::zero();
    masked_unary_domain(ma, T::sqrt, move |x| x >= zero)
}

/// Arc sine with auto-masking of out-of-domain (`|x| > 1`) inputs.
///
/// Equivalent to `numpy.ma.arcsin`.
pub fn arcsin_domain<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    let one = <T as Element>::one();
    masked_unary_domain(ma, T::asin, move |x| x.abs() <= one)
}

/// Arc cosine with auto-masking of out-of-domain (`|x| > 1`) inputs.
///
/// Equivalent to `numpy.ma.arccos`.
pub fn arccos_domain<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    let one = <T as Element>::one();
    masked_unary_domain(ma, T::acos, move |x| x.abs() <= one)
}

/// Inverse hyperbolic cosine with auto-masking of inputs `< 1`.
///
/// Equivalent to `numpy.ma.arccosh`.
pub fn arccosh_domain<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    let one = <T as Element>::one();
    masked_unary_domain(ma, T::acosh, move |x| x >= one)
}

/// Inverse hyperbolic tangent with auto-masking of inputs `|x| >= 1`.
///
/// Equivalent to `numpy.ma.arctanh`.
pub fn arctanh_domain<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    let one = <T as Element>::one();
    masked_unary_domain(ma, T::atanh, move |x| x.abs() < one)
}

/// Division with auto-masking of zero denominators.
///
/// Equivalent to `numpy.ma.divide`. Any position where the denominator
/// is exactly zero is added to the result mask.
pub fn divide_domain<T, D>(
    a: &MaskedArray<T, D>,
    b: &MaskedArray<T, D>,
) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    let zero = <T as Element>::zero();
    masked_binary_domain(a, b, |x, y| x / y, move |_x, y| y != zero, "divide_domain")
}

// ---------------------------------------------------------------------------
// Generic ufunc wrappers (#513)
//
// Instead of the hand-maintained list of ~20 per-function wrappers below,
// `masked_unary` and `masked_binary` let callers plug any closure and get
// mask propagation + fill-value handling for free. The named per-ufunc
// wrappers (sin, cos, log, sqrt, …) are kept as ergonomic shorthand for
// the hot cases.
// ---------------------------------------------------------------------------

/// Apply any unary function to a masked array, propagating the mask.
///
/// Equivalent to `numpy.ma.<unary>(a)` for any unary `T -> T` closure.
/// Masked elements are skipped; their output positions carry the
/// masked array's `fill_value` in the result data and remain masked.
///
/// This is the generic escape hatch for ufuncs that don't have a
/// dedicated wrapper in this module. Prefer a named wrapper
/// (`sin`, `log`, `sqrt`, …) when one exists — it's the same
/// implementation under the hood but documents intent better.
///
/// # Errors
/// Returns the underlying `from_vec` shape mismatch if the result
/// buffer is malformed (not possible in practice).
pub fn masked_unary<T, D, F>(
    ma: &MaskedArray<T, D>,
    f: F,
) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Copy,
    D: Dimension,
    F: Fn(T) -> T,
{
    masked_unary_op(ma, f)
}

/// Apply any binary function to two masked arrays, propagating the
/// union of their masks.
///
/// Equivalent to `numpy.ma.<binary>(a, b)` for any binary
/// `(T, T) -> T` closure. The result's mask is the elementwise OR of
/// the two inputs' masks; masked positions in the result data carry
/// the receiver's `fill_value`. Both inputs are broadcast to a common
/// shape via NumPy rules on the slow path — the same broadcast
/// machinery the named `add`/`multiply`/etc. wrappers use.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if the shapes are not
/// broadcast-compatible.
pub fn masked_binary<T, D, F>(
    a: &MaskedArray<T, D>,
    b: &MaskedArray<T, D>,
    f: F,
    op_name: &str,
) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Copy,
    D: Dimension,
    F: Fn(T, T) -> T,
{
    masked_binary_op(a, b, f, op_name)
}

// ---------------------------------------------------------------------------
// Trigonometric ufuncs
// ---------------------------------------------------------------------------

/// Elementwise sine on a masked array. Masked elements are skipped.
///
/// # Errors
/// Returns an error only for internal failures.
pub fn sin<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_unary_op(ma, T::sin)
}

/// Elementwise cosine on a masked array. Masked elements are skipped.
///
/// # Errors
/// Returns an error only for internal failures.
pub fn cos<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_unary_op(ma, T::cos)
}

/// Elementwise tangent on a masked array. Masked elements are skipped.
///
/// # Errors
/// Returns an error only for internal failures.
pub fn tan<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_unary_op(ma, T::tan)
}

/// Elementwise arc sine on a masked array. Masked elements are skipped.
///
/// # Errors
/// Returns an error only for internal failures.
pub fn arcsin<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_unary_op(ma, T::asin)
}

/// Elementwise arc cosine on a masked array. Masked elements are skipped.
///
/// # Errors
/// Returns an error only for internal failures.
pub fn arccos<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_unary_op(ma, T::acos)
}

/// Elementwise arc tangent on a masked array. Masked elements are skipped.
///
/// # Errors
/// Returns an error only for internal failures.
pub fn arctan<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_unary_op(ma, T::atan)
}

// ---------------------------------------------------------------------------
// Exponential / logarithmic
// ---------------------------------------------------------------------------

/// Elementwise exponential on a masked array. Masked elements are skipped.
///
/// # Errors
/// Returns an error only for internal failures.
pub fn exp<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_unary_op(ma, T::exp)
}

/// Elementwise base-2 exponential on a masked array. Masked elements are skipped.
///
/// # Errors
/// Returns an error only for internal failures.
pub fn exp2<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_unary_op(ma, T::exp2)
}

/// Elementwise natural logarithm on a masked array. Masked elements are skipped.
///
/// # Errors
/// Returns an error only for internal failures.
pub fn log<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_unary_op(ma, T::ln)
}

/// Elementwise base-2 logarithm on a masked array. Masked elements are skipped.
///
/// # Errors
/// Returns an error only for internal failures.
pub fn log2<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_unary_op(ma, T::log2)
}

/// Elementwise base-10 logarithm on a masked array. Masked elements are skipped.
///
/// # Errors
/// Returns an error only for internal failures.
pub fn log10<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_unary_op(ma, T::log10)
}

// ---------------------------------------------------------------------------
// Rounding
// ---------------------------------------------------------------------------

/// Elementwise floor on a masked array. Masked elements are skipped.
///
/// # Errors
/// Returns an error only for internal failures.
pub fn floor<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_unary_op(ma, T::floor)
}

/// Elementwise ceiling on a masked array. Masked elements are skipped.
///
/// # Errors
/// Returns an error only for internal failures.
pub fn ceil<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_unary_op(ma, T::ceil)
}

// ---------------------------------------------------------------------------
// Arithmetic ufuncs
// ---------------------------------------------------------------------------

/// Elementwise square root on a masked array. Masked elements are skipped.
///
/// # Errors
/// Returns an error only for internal failures.
pub fn sqrt<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_unary_op(ma, T::sqrt)
}

/// Elementwise absolute value on a masked array. Masked elements are skipped.
///
/// # Errors
/// Returns an error only for internal failures.
pub fn absolute<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_unary_op(ma, T::abs)
}

/// Elementwise negation on a masked array. Masked elements are skipped.
///
/// # Errors
/// Returns an error only for internal failures.
pub fn negative<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_unary_op(ma, T::neg)
}

/// Elementwise reciprocal on a masked array. Masked elements are skipped.
///
/// # Errors
/// Returns an error only for internal failures.
pub fn reciprocal<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_unary_op(ma, T::recip)
}

/// Elementwise square on a masked array. Masked elements are skipped.
///
/// # Errors
/// Returns an error only for internal failures.
pub fn square<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_unary_op(ma, |v| v * v)
}

/// Elementwise hyperbolic sine on a masked array.
pub fn sinh<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_unary_op(ma, T::sinh)
}

/// Elementwise hyperbolic cosine on a masked array.
pub fn cosh<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_unary_op(ma, T::cosh)
}

/// Elementwise hyperbolic tangent on a masked array.
pub fn tanh<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_unary_op(ma, T::tanh)
}

/// Elementwise inverse hyperbolic sine on a masked array.
pub fn arcsinh<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_unary_op(ma, T::asinh)
}

/// Elementwise inverse hyperbolic cosine on a masked array.
pub fn arccosh<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_unary_op(ma, T::acosh)
}

/// Elementwise inverse hyperbolic tangent on a masked array.
pub fn arctanh<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_unary_op(ma, T::atanh)
}

/// Elementwise `log(1 + x)` on a masked array.
pub fn log1p<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_unary_op(ma, T::ln_1p)
}

/// Elementwise `exp(x) - 1` on a masked array.
pub fn expm1<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_unary_op(ma, T::exp_m1)
}

/// Elementwise `trunc` (round toward zero) on a masked array.
pub fn trunc<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_unary_op(ma, T::trunc)
}

/// Elementwise `round` (round to nearest, halves to even) on a masked
/// array.
pub fn round<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_unary_op(ma, T::round)
}

/// Elementwise `sign` (-1, 0, or 1 per element) on a masked array.
///
/// Matches NumPy's `np.sign` rather than Rust's `f64::signum`: exact
/// zero (both +0.0 and -0.0) returns 0.0, not +1.0. NaN propagates
/// to NaN. For any other finite or infinite value the result is
/// `-1.0` if negative and `+1.0` if positive.
pub fn sign<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    let zero = <T as Element>::zero();
    let one = <T as Element>::one();
    masked_unary_op(ma, move |v| {
        if v.is_nan() {
            v
        } else if v == zero {
            zero
        } else if v < zero {
            -one
        } else {
            one
        }
    })
}

// ---------------------------------------------------------------------------
// Binary ufuncs on two MaskedArrays
// ---------------------------------------------------------------------------

/// Elementwise addition of two masked arrays with mask propagation.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if shapes differ.
pub fn add<T, D>(a: &MaskedArray<T, D>, b: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_binary_op(a, b, |x, y| x + y, "add")
}

/// Elementwise subtraction of two masked arrays with mask propagation.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if shapes differ.
pub fn subtract<T, D>(
    a: &MaskedArray<T, D>,
    b: &MaskedArray<T, D>,
) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_binary_op(a, b, |x, y| x - y, "subtract")
}

/// Elementwise multiplication of two masked arrays with mask propagation.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if shapes differ.
pub fn multiply<T, D>(
    a: &MaskedArray<T, D>,
    b: &MaskedArray<T, D>,
) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_binary_op(a, b, |x, y| x * y, "multiply")
}

/// Elementwise division of two masked arrays with mask propagation.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if shapes differ.
pub fn divide<T, D>(a: &MaskedArray<T, D>, b: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_binary_op(a, b, |x, y| x / y, "divide")
}

/// Elementwise power of two masked arrays with mask propagation.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if shapes differ.
pub fn power<T, D>(a: &MaskedArray<T, D>, b: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_binary_op(a, b, T::powf, "power")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::Array;
    use ferray_core::dimension::Ix1;

    fn make_ma(data: Vec<f64>, mask: Vec<bool>) -> MaskedArray<f64, Ix1> {
        let n = data.len();
        let d = Array::<f64, Ix1>::from_vec(Ix1::new([n]), data).unwrap();
        let m = Array::<bool, Ix1>::from_vec(Ix1::new([n]), mask).unwrap();
        MaskedArray::new(d, m).unwrap()
    }

    // ---- generic masked_unary / masked_binary (#513) ----

    #[test]
    fn masked_unary_applies_closure_to_unmasked_only() {
        // Closure that's not exposed as a named wrapper: `x * 10 + 1`.
        let ma = make_ma(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![false, true, false, false],
        );
        let r = masked_unary(&ma, |x| x * 10.0 + 1.0).unwrap();
        let d: Vec<f64> = r.data().iter().copied().collect();
        // Position 1 is masked → retains the fill_value (default: 0.0).
        assert_eq!(d, vec![11.0, 0.0, 31.0, 41.0]);
        // Mask is preserved identically.
        let m: Vec<bool> = r.mask().iter().copied().collect();
        assert_eq!(m, vec![false, true, false, false]);
    }

    #[test]
    fn masked_unary_preserves_fill_value() {
        let mut ma = make_ma(vec![1.0, 2.0, 3.0], vec![false, true, false]);
        ma.set_fill_value(-99.0);
        let r = masked_unary(&ma, f64::sqrt).unwrap();
        let d: Vec<f64> = r.data().iter().copied().collect();
        assert_eq!(d[1], -99.0);
        assert_eq!(r.fill_value(), -99.0);
    }

    #[test]
    fn masked_binary_mask_union_and_custom_closure() {
        // Custom binary op: `a * 2 + b`.
        let a = make_ma(vec![1.0, 2.0, 3.0, 4.0], vec![false, true, false, false]);
        let b = make_ma(vec![10.0, 20.0, 30.0, 40.0], vec![false, false, true, false]);
        let r = masked_binary(&a, &b, |x, y| x * 2.0 + y, "test_op").unwrap();
        let d: Vec<f64> = r.data().iter().copied().collect();
        // Position 0: 2*1 + 10 = 12; position 1: masked; position 2: masked;
        // position 3: 2*4 + 40 = 48.
        assert_eq!(d[0], 12.0);
        assert_eq!(d[3], 48.0);
        // Mask is the OR of both inputs.
        let m: Vec<bool> = r.mask().iter().copied().collect();
        assert_eq!(m, vec![false, true, true, false]);
    }

    #[test]
    fn masked_binary_broadcasts_cross_shape() {
        // masked_binary inherits broadcasting from masked_binary_op.
        use ferray_core::dimension::Ix2;
        let d1 = Array::<f64, Ix2>::from_vec(
            Ix2::new([2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
        let m1 = Array::<bool, Ix2>::from_vec(Ix2::new([2, 3]), vec![false; 6]).unwrap();
        let a = MaskedArray::new(d1, m1).unwrap();
        let d2 =
            Array::<f64, Ix2>::from_vec(Ix2::new([1, 3]), vec![10.0, 20.0, 30.0]).unwrap();
        let m2 = Array::<bool, Ix2>::from_vec(Ix2::new([1, 3]), vec![false; 3]).unwrap();
        let b = MaskedArray::new(d2, m2).unwrap();
        let r = masked_binary(&a, &b, |x, y| x + y, "add_broadcast").unwrap();
        let d: Vec<f64> = r.data().iter().copied().collect();
        assert_eq!(d, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    // ---- new named ufunc wrappers ----

    #[test]
    fn sinh_cosh_tanh_skip_masked() {
        let ma = make_ma(vec![0.0, 1.0, -1.0], vec![false, true, false]);
        let sh = sinh(&ma).unwrap();
        let ch = cosh(&ma).unwrap();
        let th = tanh(&ma).unwrap();
        let sd: Vec<f64> = sh.data().iter().copied().collect();
        let cd: Vec<f64> = ch.data().iter().copied().collect();
        let td: Vec<f64> = th.data().iter().copied().collect();
        assert!((sd[0] - 0.0).abs() < 1e-12);
        assert!((cd[0] - 1.0).abs() < 1e-12);
        assert!((td[0] - 0.0).abs() < 1e-12);
        // Position 1 masked → fill value (0.0).
        assert_eq!(sd[1], 0.0);
        assert_eq!(cd[1], 0.0);
        assert_eq!(td[1], 0.0);
        assert!((sd[2] - (-1.0_f64).sinh()).abs() < 1e-12);
    }

    #[test]
    fn log1p_expm1_are_precise_near_zero() {
        let ma = make_ma(vec![1e-15, 0.5, -0.5], vec![false, true, false]);
        let l = log1p(&ma).unwrap();
        let e = expm1(&ma).unwrap();
        let ld: Vec<f64> = l.data().iter().copied().collect();
        let ed: Vec<f64> = e.data().iter().copied().collect();
        // log1p(1e-15) ≈ 1e-15 (not lost to floating point)
        assert!((ld[0] - 1e-15_f64.ln_1p()).abs() < 1e-25);
        assert!((ed[2] - (-0.5_f64).exp_m1()).abs() < 1e-12);
    }

    #[test]
    fn trunc_round_sign_basic() {
        let ma = make_ma(
            vec![1.7, -2.5, 0.0, -3.2],
            vec![false, false, false, false],
        );
        let t = trunc(&ma).unwrap();
        let r = round(&ma).unwrap();
        let s = sign(&ma).unwrap();
        let td: Vec<f64> = t.data().iter().copied().collect();
        let rd: Vec<f64> = r.data().iter().copied().collect();
        let sd: Vec<f64> = s.data().iter().copied().collect();
        assert_eq!(td, vec![1.0, -2.0, 0.0, -3.0]);
        // f64::round on -2.5 rounds away from zero to -3 in Rust stdlib
        // (NOT round-half-to-even). Hand-check matches that behavior.
        assert_eq!(rd, vec![2.0, -3.0, 0.0, -3.0]);
        assert_eq!(sd, vec![1.0, -1.0, 0.0, -1.0]);
    }

    #[test]
    fn arcsinh_arccosh_arctanh_masked_positions_use_fill() {
        let ma = make_ma(
            vec![0.0, 2.0, 0.5, -1.0],
            vec![false, true, false, true],
        );
        let a = arcsinh(&ma).unwrap();
        let ad: Vec<f64> = a.data().iter().copied().collect();
        assert!((ad[0] - 0.0_f64.asinh()).abs() < 1e-12);
        // Masked → 0.0 fill
        assert_eq!(ad[1], 0.0);
        assert_eq!(ad[3], 0.0);

        // arccosh needs x >= 1, so use a different input for that test.
        let ma2 = make_ma(vec![1.0, 2.0, 5.0], vec![false, false, false]);
        let ac = arccosh(&ma2).unwrap();
        let acd: Vec<f64> = ac.data().iter().copied().collect();
        assert!((acd[0] - 0.0).abs() < 1e-12); // arccosh(1) = 0
        assert!((acd[1] - 2.0_f64.acosh()).abs() < 1e-12);

        // arctanh needs |x| < 1.
        let ma3 = make_ma(vec![0.0, 0.5, -0.5], vec![false, false, false]);
        let at = arctanh(&ma3).unwrap();
        let atd: Vec<f64> = at.data().iter().copied().collect();
        assert!((atd[1] - 0.5_f64.atanh()).abs() < 1e-12);
    }

    #[test]
    fn masked_unary_and_named_sin_agree() {
        // Generic path and named wrapper must produce identical results.
        let ma = make_ma(
            vec![0.0, 1.0, 2.0, 3.0],
            vec![false, true, false, false],
        );
        let via_named = sin(&ma).unwrap();
        let via_generic = masked_unary(&ma, f64::sin).unwrap();
        let vn: Vec<f64> = via_named.data().iter().copied().collect();
        let vg: Vec<f64> = via_generic.data().iter().copied().collect();
        assert_eq!(vn, vg);
        let mn: Vec<bool> = via_named.mask().iter().copied().collect();
        let mg: Vec<bool> = via_generic.mask().iter().copied().collect();
        assert_eq!(mn, mg);
    }

    #[test]
    fn masked_binary_and_named_add_agree() {
        let a = make_ma(vec![1.0, 2.0, 3.0], vec![false, true, false]);
        let b = make_ma(vec![10.0, 20.0, 30.0], vec![false, false, true]);
        let via_named = add(&a, &b).unwrap();
        let via_generic = masked_binary(&a, &b, |x, y| x + y, "add_generic").unwrap();
        let vn: Vec<f64> = via_named.data().iter().copied().collect();
        let vg: Vec<f64> = via_generic.data().iter().copied().collect();
        assert_eq!(vn, vg);
    }

    // ---- domain-aware ufuncs (#503) ----

    #[test]
    fn log_domain_masks_non_positive_inputs() {
        // [1.0, 2.0, -1.0, 0.0, 3.0] with no existing mask
        let ma = make_ma(
            vec![1.0, 2.0, -1.0, 0.0, 3.0],
            vec![false, false, false, false, false],
        );
        let r = log_domain(&ma).unwrap();
        let m: Vec<bool> = r.mask().iter().copied().collect();
        // Positions 2 (-1) and 3 (0) are out of domain → masked.
        assert_eq!(m, vec![false, false, true, true, false]);
        // Unmasked positions have correct log values.
        let d: Vec<f64> = r.data().iter().copied().collect();
        assert!((d[0] - 0.0).abs() < 1e-12);
        assert!((d[1] - 2.0_f64.ln()).abs() < 1e-12);
        assert!((d[4] - 3.0_f64.ln()).abs() < 1e-12);
        // Masked positions carry the fill value (0.0 default).
        assert_eq!(d[2], 0.0);
        assert_eq!(d[3], 0.0);
    }

    #[test]
    fn log_domain_preserves_existing_mask() {
        // Position 1 is already masked; position 3 goes to masked via domain.
        let ma = make_ma(
            vec![1.0, 2.0, 5.0, -1.0],
            vec![false, true, false, false],
        );
        let r = log_domain(&ma).unwrap();
        let m: Vec<bool> = r.mask().iter().copied().collect();
        assert_eq!(m, vec![false, true, false, true]);
    }

    #[test]
    fn log_domain_vs_plain_log_on_negative_input() {
        // Plain `log` produces NaN at the negative position; `log_domain`
        // masks it and substitutes the fill value.
        let ma = make_ma(vec![1.0, -2.0, 3.0], vec![false, false, false]);
        let plain = log(&ma).unwrap();
        let domain = log_domain(&ma).unwrap();
        let pd: Vec<f64> = plain.data().iter().copied().collect();
        let dd: Vec<f64> = domain.data().iter().copied().collect();
        // Plain: position 1 is NaN.
        assert!(pd[1].is_nan());
        // Domain: position 1 is the fill value, and the mask is set.
        assert_eq!(dd[1], 0.0);
        assert!(domain.mask().as_slice().unwrap()[1]);
    }

    #[test]
    fn sqrt_domain_masks_negative_inputs() {
        let ma = make_ma(
            vec![0.0, 1.0, 4.0, -9.0, -1e-10],
            vec![false, false, false, false, false],
        );
        let r = sqrt_domain(&ma).unwrap();
        let m: Vec<bool> = r.mask().iter().copied().collect();
        // 0.0 and positive values pass; negatives are masked.
        assert_eq!(m, vec![false, false, false, true, true]);
        let d: Vec<f64> = r.data().iter().copied().collect();
        assert!((d[0] - 0.0).abs() < 1e-12);
        assert!((d[1] - 1.0).abs() < 1e-12);
        assert!((d[2] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn arcsin_domain_masks_out_of_range() {
        let ma = make_ma(
            vec![-1.5, -0.5, 0.0, 0.5, 1.5],
            vec![false, false, false, false, false],
        );
        let r = arcsin_domain(&ma).unwrap();
        let m: Vec<bool> = r.mask().iter().copied().collect();
        // |x| > 1 → masked; |x| <= 1 passes.
        assert_eq!(m, vec![true, false, false, false, true]);
        let d: Vec<f64> = r.data().iter().copied().collect();
        assert!((d[1] - (-0.5_f64).asin()).abs() < 1e-12);
        assert!((d[2] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn arccos_domain_masks_out_of_range() {
        let ma = make_ma(vec![-1.5, 0.0, 1.0, 2.0], vec![false, false, false, false]);
        let r = arccos_domain(&ma).unwrap();
        let m: Vec<bool> = r.mask().iter().copied().collect();
        assert_eq!(m, vec![true, false, false, true]);
    }

    #[test]
    fn arccosh_domain_masks_below_one() {
        // arccosh domain is x >= 1.
        let ma = make_ma(vec![0.5, 1.0, 2.0, 10.0], vec![false, false, false, false]);
        let r = arccosh_domain(&ma).unwrap();
        let m: Vec<bool> = r.mask().iter().copied().collect();
        assert_eq!(m, vec![true, false, false, false]);
        let d: Vec<f64> = r.data().iter().copied().collect();
        assert!((d[1] - 0.0).abs() < 1e-12); // acosh(1) = 0
        assert!((d[2] - 2.0_f64.acosh()).abs() < 1e-12);
    }

    #[test]
    fn arctanh_domain_masks_boundary_and_beyond() {
        // arctanh domain is |x| < 1 strictly (not <=).
        let ma = make_ma(
            vec![-1.0, -0.5, 0.0, 0.5, 1.0],
            vec![false, false, false, false, false],
        );
        let r = arctanh_domain(&ma).unwrap();
        let m: Vec<bool> = r.mask().iter().copied().collect();
        assert_eq!(m, vec![true, false, false, false, true]);
    }

    #[test]
    fn divide_domain_masks_zero_denominators() {
        let num = make_ma(vec![1.0, 2.0, 3.0, 4.0], vec![false, false, false, false]);
        let den = make_ma(vec![2.0, 0.0, 1.0, 0.0], vec![false, false, false, false]);
        let r = divide_domain(&num, &den).unwrap();
        let m: Vec<bool> = r.mask().iter().copied().collect();
        // Positions 1 and 3 have zero denominators → masked.
        assert_eq!(m, vec![false, true, false, true]);
        let d: Vec<f64> = r.data().iter().copied().collect();
        assert!((d[0] - 0.5).abs() < 1e-12);
        assert!((d[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn divide_domain_preserves_numerator_and_denominator_masks() {
        let num = make_ma(vec![1.0, 2.0, 3.0, 4.0], vec![false, true, false, false]);
        let den = make_ma(vec![2.0, 5.0, 0.0, 4.0], vec![false, false, false, true]);
        let r = divide_domain(&num, &den).unwrap();
        let m: Vec<bool> = r.mask().iter().copied().collect();
        // pos 1: num masked; pos 2: zero denom; pos 3: denom masked.
        assert_eq!(m, vec![false, true, true, true]);
    }

    #[test]
    fn masked_unary_domain_generic_path() {
        // Custom domain: only positions where x is even allowed.
        let ma = make_ma(
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![false, false, false, false, false],
        );
        let r = masked_unary_domain(
            &ma,
            |x| x * 2.0,
            |x| (x as i32) % 2 == 0,
        )
        .unwrap();
        let m: Vec<bool> = r.mask().iter().copied().collect();
        assert_eq!(m, vec![true, false, true, false, true]);
        let d: Vec<f64> = r.data().iter().copied().collect();
        assert_eq!(d[1], 4.0);
        assert_eq!(d[3], 8.0);
    }

    #[test]
    fn masked_binary_domain_rejects_mismatched_shapes() {
        let a = make_ma(vec![1.0, 2.0], vec![false, false]);
        let b = make_ma(vec![1.0, 2.0, 3.0], vec![false, false, false]);
        assert!(divide_domain(&a, &b).is_err());
    }

    #[test]
    fn log_domain_with_custom_fill_value() {
        let mut ma = make_ma(vec![1.0, -1.0, 2.0], vec![false, false, false]);
        ma.set_fill_value(-999.0);
        let r = log_domain(&ma).unwrap();
        let d: Vec<f64> = r.data().iter().copied().collect();
        // Position 1 is auto-masked → carries fill value.
        assert_eq!(d[1], -999.0);
        assert_eq!(r.fill_value(), -999.0);
    }
}
