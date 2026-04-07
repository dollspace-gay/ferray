// ferray-ma: Masked binary ops with mask union (REQ-10, REQ-11)

use std::ops::{Add, Div, Mul, Sub};

use ferray_core::Array;
use ferray_core::dimension::Dimension;
use ferray_core::dtype::Element;
use ferray_core::error::{FerrayError, FerrayResult};

use crate::MaskedArray;

/// Helper: compute the union of two boolean mask arrays (element-wise OR).
fn mask_union<D: Dimension>(
    a: &Array<bool, D>,
    b: &Array<bool, D>,
) -> FerrayResult<Array<bool, D>> {
    if a.shape() != b.shape() {
        return Err(FerrayError::shape_mismatch(format!(
            "mask_union: shapes {:?} and {:?} differ",
            a.shape(),
            b.shape()
        )));
    }
    let data: Vec<bool> = a.iter().zip(b.iter()).map(|(x, y)| *x || *y).collect();
    Array::from_vec(a.dim().clone(), data)
}

/// Add two masked arrays elementwise, propagating the mask union.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if shapes differ.
pub fn masked_add<T, D>(
    a: &MaskedArray<T, D>,
    b: &MaskedArray<T, D>,
) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Add<Output = T> + Copy,
    D: Dimension,
{
    if a.shape() != b.shape() {
        return Err(FerrayError::shape_mismatch(format!(
            "masked_add: shapes {:?} and {:?} differ",
            a.shape(),
            b.shape()
        )));
    }
    let result_mask = mask_union(a.mask(), b.mask())?;
    let data: Vec<T> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .zip(result_mask.iter())
        .map(|((x, y), m)| if *m { T::zero() } else { *x + *y })
        .collect();
    let result_data = Array::from_vec(a.dim().clone(), data)?;
    MaskedArray::new(result_data, result_mask)
}

/// Subtract two masked arrays elementwise, propagating the mask union.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if shapes differ.
pub fn masked_sub<T, D>(
    a: &MaskedArray<T, D>,
    b: &MaskedArray<T, D>,
) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Sub<Output = T> + Copy,
    D: Dimension,
{
    if a.shape() != b.shape() {
        return Err(FerrayError::shape_mismatch(format!(
            "masked_sub: shapes {:?} and {:?} differ",
            a.shape(),
            b.shape()
        )));
    }
    let result_mask = mask_union(a.mask(), b.mask())?;
    let data: Vec<T> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .zip(result_mask.iter())
        .map(|((x, y), m)| if *m { T::zero() } else { *x - *y })
        .collect();
    let result_data = Array::from_vec(a.dim().clone(), data)?;
    MaskedArray::new(result_data, result_mask)
}

/// Multiply two masked arrays elementwise, propagating the mask union.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if shapes differ.
pub fn masked_mul<T, D>(
    a: &MaskedArray<T, D>,
    b: &MaskedArray<T, D>,
) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Mul<Output = T> + Copy,
    D: Dimension,
{
    if a.shape() != b.shape() {
        return Err(FerrayError::shape_mismatch(format!(
            "masked_mul: shapes {:?} and {:?} differ",
            a.shape(),
            b.shape()
        )));
    }
    let result_mask = mask_union(a.mask(), b.mask())?;
    let data: Vec<T> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .zip(result_mask.iter())
        .map(|((x, y), m)| if *m { T::zero() } else { *x * *y })
        .collect();
    let result_data = Array::from_vec(a.dim().clone(), data)?;
    MaskedArray::new(result_data, result_mask)
}

/// Divide two masked arrays elementwise, propagating the mask union.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if shapes differ.
pub fn masked_div<T, D>(
    a: &MaskedArray<T, D>,
    b: &MaskedArray<T, D>,
) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Div<Output = T> + Copy,
    D: Dimension,
{
    if a.shape() != b.shape() {
        return Err(FerrayError::shape_mismatch(format!(
            "masked_div: shapes {:?} and {:?} differ",
            a.shape(),
            b.shape()
        )));
    }
    let result_mask = mask_union(a.mask(), b.mask())?;
    let data: Vec<T> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .zip(result_mask.iter())
        .map(|((x, y), m)| if *m { T::zero() } else { *x / *y })
        .collect();
    let result_data = Array::from_vec(a.dim().clone(), data)?;
    MaskedArray::new(result_data, result_mask)
}

/// Add a masked array and a regular array, treating the regular array as unmasked.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if shapes differ.
pub fn masked_add_array<T, D>(
    ma: &MaskedArray<T, D>,
    arr: &Array<T, D>,
) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Add<Output = T> + Copy,
    D: Dimension,
{
    if ma.shape() != arr.shape() {
        return Err(FerrayError::shape_mismatch(format!(
            "masked_add_array: shapes {:?} and {:?} differ",
            ma.shape(),
            arr.shape()
        )));
    }
    let data: Vec<T> = ma
        .data()
        .iter()
        .zip(arr.iter())
        .zip(ma.mask().iter())
        .map(|((x, y), m)| if *m { T::zero() } else { *x + *y })
        .collect();
    let result_data = Array::from_vec(ma.dim().clone(), data)?;
    MaskedArray::new(result_data, ma.mask().clone())
}

/// Subtract a regular array from a masked array, treating the regular array as unmasked.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if shapes differ.
pub fn masked_sub_array<T, D>(
    ma: &MaskedArray<T, D>,
    arr: &Array<T, D>,
) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Sub<Output = T> + Copy,
    D: Dimension,
{
    if ma.shape() != arr.shape() {
        return Err(FerrayError::shape_mismatch(format!(
            "masked_sub_array: shapes {:?} and {:?} differ",
            ma.shape(),
            arr.shape()
        )));
    }
    let data: Vec<T> = ma
        .data()
        .iter()
        .zip(arr.iter())
        .zip(ma.mask().iter())
        .map(|((x, y), m)| if *m { T::zero() } else { *x - *y })
        .collect();
    let result_data = Array::from_vec(ma.dim().clone(), data)?;
    MaskedArray::new(result_data, ma.mask().clone())
}

/// Multiply a masked array and a regular array, treating the regular array as unmasked.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if shapes differ.
pub fn masked_mul_array<T, D>(
    ma: &MaskedArray<T, D>,
    arr: &Array<T, D>,
) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Mul<Output = T> + Copy,
    D: Dimension,
{
    if ma.shape() != arr.shape() {
        return Err(FerrayError::shape_mismatch(format!(
            "masked_mul_array: shapes {:?} and {:?} differ",
            ma.shape(),
            arr.shape()
        )));
    }
    let data: Vec<T> = ma
        .data()
        .iter()
        .zip(arr.iter())
        .zip(ma.mask().iter())
        .map(|((x, y), m)| if *m { T::zero() } else { *x * *y })
        .collect();
    let result_data = Array::from_vec(ma.dim().clone(), data)?;
    MaskedArray::new(result_data, ma.mask().clone())
}

/// Divide a masked array by a regular array, treating the regular array as unmasked.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if shapes differ.
pub fn masked_div_array<T, D>(
    ma: &MaskedArray<T, D>,
    arr: &Array<T, D>,
) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Div<Output = T> + Copy,
    D: Dimension,
{
    if ma.shape() != arr.shape() {
        return Err(FerrayError::shape_mismatch(format!(
            "masked_div_array: shapes {:?} and {:?} differ",
            ma.shape(),
            arr.shape()
        )));
    }
    let data: Vec<T> = ma
        .data()
        .iter()
        .zip(arr.iter())
        .zip(ma.mask().iter())
        .map(|((x, y), m)| if *m { T::zero() } else { *x / *y })
        .collect();
    let result_data = Array::from_vec(ma.dim().clone(), data)?;
    MaskedArray::new(result_data, ma.mask().clone())
}

// ---------------------------------------------------------------------------
// std::ops trait implementations for MaskedArray + MaskedArray
// ---------------------------------------------------------------------------

/// `&MaskedArray + &MaskedArray` — elementwise add with mask union.
impl<T, D> std::ops::Add<&MaskedArray<T, D>> for &MaskedArray<T, D>
where
    T: Element + Add<Output = T> + Copy,
    D: Dimension,
{
    type Output = FerrayResult<MaskedArray<T, D>>;

    fn add(self, rhs: &MaskedArray<T, D>) -> Self::Output {
        masked_add(self, rhs)
    }
}

/// `&MaskedArray - &MaskedArray` — elementwise subtract with mask union.
impl<T, D> std::ops::Sub<&MaskedArray<T, D>> for &MaskedArray<T, D>
where
    T: Element + Sub<Output = T> + Copy,
    D: Dimension,
{
    type Output = FerrayResult<MaskedArray<T, D>>;

    fn sub(self, rhs: &MaskedArray<T, D>) -> Self::Output {
        masked_sub(self, rhs)
    }
}

/// `&MaskedArray * &MaskedArray` — elementwise multiply with mask union.
impl<T, D> std::ops::Mul<&MaskedArray<T, D>> for &MaskedArray<T, D>
where
    T: Element + Mul<Output = T> + Copy,
    D: Dimension,
{
    type Output = FerrayResult<MaskedArray<T, D>>;

    fn mul(self, rhs: &MaskedArray<T, D>) -> Self::Output {
        masked_mul(self, rhs)
    }
}

/// `&MaskedArray / &MaskedArray` — elementwise divide with mask union.
impl<T, D> std::ops::Div<&MaskedArray<T, D>> for &MaskedArray<T, D>
where
    T: Element + Div<Output = T> + Copy,
    D: Dimension,
{
    type Output = FerrayResult<MaskedArray<T, D>>;

    fn div(self, rhs: &MaskedArray<T, D>) -> Self::Output {
        masked_div(self, rhs)
    }
}
