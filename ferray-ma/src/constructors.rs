// ferray-ma: Masking constructors (REQ-7, REQ-8, REQ-9)
//
// masked_where, masked_invalid, masked_equal, masked_greater, masked_less,
// masked_not_equal, masked_greater_equal, masked_less_equal,
// masked_inside, masked_outside

use ferray_core::Array;
use ferray_core::dimension::Dimension;
use ferray_core::dtype::Element;
use ferray_core::error::FerrayResult;
use num_traits::Float;

use crate::MaskedArray;

/// Create a `MaskedArray` by masking elements where the condition array is `true`.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if `condition` and `data` have different shapes.
pub fn masked_where<T: Element + Copy, D: Dimension>(
    condition: &Array<bool, D>,
    data: &Array<T, D>,
) -> FerrayResult<MaskedArray<T, D>> {
    MaskedArray::new(data.clone(), condition.clone())
}

/// Create a `MaskedArray` by masking NaN and Inf values.
///
/// # Errors
/// Returns an error only for internal failures.
pub fn masked_invalid<T: Element + Float, D: Dimension>(
    data: &Array<T, D>,
) -> FerrayResult<MaskedArray<T, D>> {
    let mask_data: Vec<bool> = data.iter().map(|v| v.is_nan() || v.is_infinite()).collect();
    let mask = Array::from_vec(data.dim().clone(), mask_data)?;
    MaskedArray::new(data.clone(), mask)
}

/// Create a `MaskedArray` by masking elements equal to `value`.
///
/// # Errors
/// Returns an error only for internal failures.
pub fn masked_equal<T: Element + PartialEq + Copy, D: Dimension>(
    data: &Array<T, D>,
    value: T,
) -> FerrayResult<MaskedArray<T, D>> {
    let mask_data: Vec<bool> = data.iter().map(|v| *v == value).collect();
    let mask = Array::from_vec(data.dim().clone(), mask_data)?;
    MaskedArray::new(data.clone(), mask)
}

/// Create a `MaskedArray` by masking elements not equal to `value`.
///
/// # Errors
/// Returns an error only for internal failures.
pub fn masked_not_equal<T: Element + PartialEq + Copy, D: Dimension>(
    data: &Array<T, D>,
    value: T,
) -> FerrayResult<MaskedArray<T, D>> {
    let mask_data: Vec<bool> = data.iter().map(|v| *v != value).collect();
    let mask = Array::from_vec(data.dim().clone(), mask_data)?;
    MaskedArray::new(data.clone(), mask)
}

/// Create a `MaskedArray` by masking elements greater than `value`.
///
/// # Errors
/// Returns an error only for internal failures.
pub fn masked_greater<T: Element + PartialOrd + Copy, D: Dimension>(
    data: &Array<T, D>,
    value: T,
) -> FerrayResult<MaskedArray<T, D>> {
    let mask_data: Vec<bool> = data.iter().map(|v| *v > value).collect();
    let mask = Array::from_vec(data.dim().clone(), mask_data)?;
    MaskedArray::new(data.clone(), mask)
}

/// Create a `MaskedArray` by masking elements less than `value`.
///
/// # Errors
/// Returns an error only for internal failures.
pub fn masked_less<T: Element + PartialOrd + Copy, D: Dimension>(
    data: &Array<T, D>,
    value: T,
) -> FerrayResult<MaskedArray<T, D>> {
    let mask_data: Vec<bool> = data.iter().map(|v| *v < value).collect();
    let mask = Array::from_vec(data.dim().clone(), mask_data)?;
    MaskedArray::new(data.clone(), mask)
}

/// Create a `MaskedArray` by masking elements greater than or equal to `value`.
///
/// # Errors
/// Returns an error only for internal failures.
pub fn masked_greater_equal<T: Element + PartialOrd + Copy, D: Dimension>(
    data: &Array<T, D>,
    value: T,
) -> FerrayResult<MaskedArray<T, D>> {
    let mask_data: Vec<bool> = data.iter().map(|v| *v >= value).collect();
    let mask = Array::from_vec(data.dim().clone(), mask_data)?;
    MaskedArray::new(data.clone(), mask)
}

/// Create a `MaskedArray` by masking elements less than or equal to `value`.
///
/// # Errors
/// Returns an error only for internal failures.
pub fn masked_less_equal<T: Element + PartialOrd + Copy, D: Dimension>(
    data: &Array<T, D>,
    value: T,
) -> FerrayResult<MaskedArray<T, D>> {
    let mask_data: Vec<bool> = data.iter().map(|v| *v <= value).collect();
    let mask = Array::from_vec(data.dim().clone(), mask_data)?;
    MaskedArray::new(data.clone(), mask)
}

/// Create a `MaskedArray` by masking elements inside the closed interval `[v1, v2]`.
///
/// # Errors
/// Returns an error only for internal failures.
pub fn masked_inside<T: Element + PartialOrd + Copy, D: Dimension>(
    data: &Array<T, D>,
    v1: T,
    v2: T,
) -> FerrayResult<MaskedArray<T, D>> {
    let mask_data: Vec<bool> = data.iter().map(|v| *v >= v1 && *v <= v2).collect();
    let mask = Array::from_vec(data.dim().clone(), mask_data)?;
    MaskedArray::new(data.clone(), mask)
}

/// Create a `MaskedArray` by masking elements outside the closed interval `[v1, v2]`.
///
/// # Errors
/// Returns an error only for internal failures.
pub fn masked_outside<T: Element + PartialOrd + Copy, D: Dimension>(
    data: &Array<T, D>,
    v1: T,
    v2: T,
) -> FerrayResult<MaskedArray<T, D>> {
    let mask_data: Vec<bool> = data.iter().map(|v| *v < v1 || *v > v2).collect();
    let mask = Array::from_vec(data.dim().clone(), mask_data)?;
    MaskedArray::new(data.clone(), mask)
}
