// ferray-ma: Ufunc support wrappers (REQ-12)
//
// Wrapper functions that accept MaskedArray and call underlying ferray-ufunc
// operations on the data, then propagate masks. Masked elements are skipped;
// their output positions retain the masked array's `fill_value`.

use ferray_core::dimension::Dimension;
use ferray_core::dtype::Element;
use ferray_core::error::FerrayResult;
use num_traits::Float;

use crate::MaskedArray;
use crate::arithmetic::{masked_binary_op, masked_unary_op};

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
