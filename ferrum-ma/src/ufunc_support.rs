// ferrum-ma: Ufunc support wrappers (REQ-12)
//
// Wrapper functions that accept MaskedArray and call underlying ferrum-ufunc
// operations on the data, then propagate masks. Masked elements are skipped
// (their output positions retain zero/default values).

use ferrum_core::dimension::Dimension;
use ferrum_core::dtype::Element;
use ferrum_core::error::FerrumResult;
use ferrum_core::Array;
use num_traits::Float;

use crate::MaskedArray;

// ---------------------------------------------------------------------------
// Internal helper: apply a unary ufunc only on unmasked elements
// ---------------------------------------------------------------------------

/// Apply a unary function only on unmasked elements, keeping masked positions
/// at `T::zero()`.
fn masked_unary_op<T, D>(
    ma: &MaskedArray<T, D>,
    f: impl Fn(T) -> T,
) -> FerrumResult<MaskedArray<T, D>>
where
    T: Element + Copy,
    D: Dimension,
{
    let data: Vec<T> = ma
        .data()
        .iter()
        .zip(ma.mask().iter())
        .map(|(v, m)| if *m { T::zero() } else { f(*v) })
        .collect();
    let result_data = Array::from_vec(ma.dim().clone(), data)?;
    MaskedArray::new(result_data, ma.mask().clone())
}

/// Apply a binary ufunc on two masked arrays, producing the mask union.
/// Masked positions get `T::zero()`.
fn masked_binary_op<T, D>(
    a: &MaskedArray<T, D>,
    b: &MaskedArray<T, D>,
    f: impl Fn(T, T) -> T,
) -> FerrumResult<MaskedArray<T, D>>
where
    T: Element + Copy,
    D: Dimension,
{
    let mask_data: Vec<bool> = a
        .mask()
        .iter()
        .zip(b.mask().iter())
        .map(|(ma, mb)| *ma || *mb)
        .collect();
    let result_mask = Array::from_vec(a.dim().clone(), mask_data)?;
    let data: Vec<T> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .zip(result_mask.iter())
        .map(|((x, y), m)| if *m { T::zero() } else { f(*x, *y) })
        .collect();
    let result_data = Array::from_vec(a.dim().clone(), data)?;
    MaskedArray::new(result_data, result_mask)
}

// ---------------------------------------------------------------------------
// Trigonometric ufuncs
// ---------------------------------------------------------------------------

/// Elementwise sine on a masked array. Masked elements are skipped.
///
/// # Errors
/// Returns an error only for internal failures.
pub fn sin<T, D>(ma: &MaskedArray<T, D>) -> FerrumResult<MaskedArray<T, D>>
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
pub fn cos<T, D>(ma: &MaskedArray<T, D>) -> FerrumResult<MaskedArray<T, D>>
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
pub fn tan<T, D>(ma: &MaskedArray<T, D>) -> FerrumResult<MaskedArray<T, D>>
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
pub fn arcsin<T, D>(ma: &MaskedArray<T, D>) -> FerrumResult<MaskedArray<T, D>>
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
pub fn arccos<T, D>(ma: &MaskedArray<T, D>) -> FerrumResult<MaskedArray<T, D>>
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
pub fn arctan<T, D>(ma: &MaskedArray<T, D>) -> FerrumResult<MaskedArray<T, D>>
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
pub fn exp<T, D>(ma: &MaskedArray<T, D>) -> FerrumResult<MaskedArray<T, D>>
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
pub fn exp2<T, D>(ma: &MaskedArray<T, D>) -> FerrumResult<MaskedArray<T, D>>
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
pub fn log<T, D>(ma: &MaskedArray<T, D>) -> FerrumResult<MaskedArray<T, D>>
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
pub fn log2<T, D>(ma: &MaskedArray<T, D>) -> FerrumResult<MaskedArray<T, D>>
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
pub fn log10<T, D>(ma: &MaskedArray<T, D>) -> FerrumResult<MaskedArray<T, D>>
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
pub fn floor<T, D>(ma: &MaskedArray<T, D>) -> FerrumResult<MaskedArray<T, D>>
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
pub fn ceil<T, D>(ma: &MaskedArray<T, D>) -> FerrumResult<MaskedArray<T, D>>
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
pub fn sqrt<T, D>(ma: &MaskedArray<T, D>) -> FerrumResult<MaskedArray<T, D>>
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
pub fn absolute<T, D>(ma: &MaskedArray<T, D>) -> FerrumResult<MaskedArray<T, D>>
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
pub fn negative<T, D>(ma: &MaskedArray<T, D>) -> FerrumResult<MaskedArray<T, D>>
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
pub fn reciprocal<T, D>(ma: &MaskedArray<T, D>) -> FerrumResult<MaskedArray<T, D>>
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
pub fn square<T, D>(ma: &MaskedArray<T, D>) -> FerrumResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_unary_op(ma, |v| v * v)
}

// ---------------------------------------------------------------------------
// Binary ufuncs on two MaskedArrays
// ---------------------------------------------------------------------------

/// Elementwise addition of two masked arrays with mask propagation.
///
/// # Errors
/// Returns `FerrumError::ShapeMismatch` if shapes differ.
pub fn add<T, D>(
    a: &MaskedArray<T, D>,
    b: &MaskedArray<T, D>,
) -> FerrumResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_binary_op(a, b, |x, y| x + y)
}

/// Elementwise subtraction of two masked arrays with mask propagation.
///
/// # Errors
/// Returns `FerrumError::ShapeMismatch` if shapes differ.
pub fn subtract<T, D>(
    a: &MaskedArray<T, D>,
    b: &MaskedArray<T, D>,
) -> FerrumResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_binary_op(a, b, |x, y| x - y)
}

/// Elementwise multiplication of two masked arrays with mask propagation.
///
/// # Errors
/// Returns `FerrumError::ShapeMismatch` if shapes differ.
pub fn multiply<T, D>(
    a: &MaskedArray<T, D>,
    b: &MaskedArray<T, D>,
) -> FerrumResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_binary_op(a, b, |x, y| x * y)
}

/// Elementwise division of two masked arrays with mask propagation.
///
/// # Errors
/// Returns `FerrumError::ShapeMismatch` if shapes differ.
pub fn divide<T, D>(
    a: &MaskedArray<T, D>,
    b: &MaskedArray<T, D>,
) -> FerrumResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_binary_op(a, b, |x, y| x / y)
}

/// Elementwise power of two masked arrays with mask propagation.
///
/// # Errors
/// Returns `FerrumError::ShapeMismatch` if shapes differ.
pub fn power<T, D>(
    a: &MaskedArray<T, D>,
    b: &MaskedArray<T, D>,
) -> FerrumResult<MaskedArray<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    masked_binary_op(a, b, T::powf)
}
