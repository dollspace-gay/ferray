// ferray-ma: Masked binary ops with mask union (REQ-10, REQ-11)
//
// Binary masked ops broadcast within the same dimension type per NumPy
// semantics. The fast path (identical shapes) avoids the broadcast overhead.
// Cross-rank broadcasting is unavailable through the std::ops operators
// since both sides must share `D`; users with mixed-rank arrays should call
// .broadcast_to() on the shorter array first.

use std::ops::{Add, Div, Mul, Sub};

use ferray_core::Array;
use ferray_core::dimension::Dimension;
use ferray_core::dimension::broadcast::{broadcast_shapes, broadcast_to};
use ferray_core::dtype::Element;
use ferray_core::error::{FerrayError, FerrayResult};

use crate::MaskedArray;

/// Result of broadcasting two masked arrays to a common shape.
///
/// Returns the broadcast (data, mask) pairs for both inputs as flat `Vec<T>`
/// / `Vec<bool>` along with the common dimension `D`. We materialize the
/// data so the caller can run a simple zipped fold without juggling views.
struct BroadcastedPair<T> {
    a_data: Vec<T>,
    a_mask: Vec<bool>,
    b_data: Vec<T>,
    b_mask: Vec<bool>,
}

/// Broadcast two masked arrays to a common shape using `NumPy` rules.
///
/// Both inputs must share dimension type `D`, so the broadcast result has
/// the same rank as `D` (or, for `IxDyn`, the maximum of the two ranks).
/// The result `Dimension` is reconstructed via [`Dimension::from_dim_slice`].
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if the shapes are not broadcast-compatible.
fn broadcast_masked_pair<T, D>(
    a: &MaskedArray<T, D>,
    b: &MaskedArray<T, D>,
    op_name: &str,
) -> FerrayResult<(BroadcastedPair<T>, D)>
where
    T: Element + Copy,
    D: Dimension,
{
    let target_shape = broadcast_shapes(a.shape(), b.shape()).map_err(|_| {
        FerrayError::shape_mismatch(format!(
            "{}: shapes {:?} and {:?} are not broadcast-compatible",
            op_name,
            a.shape(),
            b.shape()
        ))
    })?;

    let a_data_view = broadcast_to(a.data(), &target_shape)?;
    let a_mask_view = broadcast_to(a.mask(), &target_shape)?;
    let b_data_view = broadcast_to(b.data(), &target_shape)?;
    let b_mask_view = broadcast_to(b.mask(), &target_shape)?;

    let pair = BroadcastedPair {
        a_data: a_data_view.iter().copied().collect(),
        a_mask: a_mask_view.iter().copied().collect(),
        b_data: b_data_view.iter().copied().collect(),
        b_mask: b_mask_view.iter().copied().collect(),
    };

    let result_dim = D::from_dim_slice(&target_shape).ok_or_else(|| {
        FerrayError::shape_mismatch(format!(
            "{op_name}: cannot represent broadcast result shape {target_shape:?} as the input dimension type"
        ))
    })?;

    Ok((pair, result_dim))
}

/// Apply `f` only to unmasked elements of `ma`. Masked positions carry
/// the source array payload into the result. Crate-internal so
/// `ufunc_support::masked_unary_op`-style call sites can reuse it and
/// `ferray-ma` ships one definition instead of two (#150).
pub(crate) fn masked_unary_op<T, D, F>(
    ma: &MaskedArray<T, D>,
    f: F,
) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Copy,
    D: Dimension,
    F: Fn(T) -> T,
{
    let fill = ma.fill_value;

    // Fast path: nomask sentinel (#506) — no mask bits to consult, so
    // `f` is applied to every element unconditionally and the result
    // inherits the nomask state via `from_data`. Skips two full bool
    // allocations (iterating ma.mask() materializes the lazy mask, and
    // MaskedArray::new stores a cloned copy in the result).
    if !ma.has_real_mask() {
        let data: Vec<T> = ma.data().iter().map(|&v| f(v)).collect();
        let result_data = Array::from_vec(ma.dim().clone(), data)?;
        let mut out = MaskedArray::from_data(result_data)?;
        out.fill_value = fill;
        return Ok(out);
    }

    // Mask present: split the work into two passes so the hot path
    // (applying `f` to the data) can auto-vectorise without a mask
    // branch in the inner loop (#157). NumPy's `numpy.ma` wrappers keep
    // the source payload under existing masks while carrying the mask
    // separately, so masked positions get patched back to the original
    // payload below regardless of what `f` produced for them.
    let data_vec: Vec<T> = ma.data().iter().map(|&v| f(v)).collect();
    let data: Vec<T> = data_vec
        .into_iter()
        .zip(ma.data().iter())
        .zip(ma.mask().iter())
        .map(|((computed, original), m)| if *m { *original } else { computed })
        .collect();
    let result_data = Array::from_vec(ma.dim().clone(), data)?;
    let mut out = MaskedArray::new(result_data, ma.mask().clone())?;
    out.fill_value = fill;
    Ok(out)
}

/// Generic masked binary op: applies `op` to each (a, b) pair, propagates
/// the mask union, fills masked positions with the receiver's `fill_value`.
///
/// Fast path on identical shapes; broadcasts otherwise. Used by
/// [`masked_add`], [`masked_sub`], etc. Crate-internal so
/// `ufunc_support` can replace its own two-variant copy of the body
/// with this single source of truth (#150).
pub(crate) fn masked_binary_op<T, D, F>(
    a: &MaskedArray<T, D>,
    b: &MaskedArray<T, D>,
    op: F,
    op_name: &str,
) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Copy,
    D: Dimension,
    F: Fn(T, T) -> T,
{
    masked_binary_op_with_extra_mask(a, b, op, |_, _| false, op_name)
}

fn masked_binary_op_with_extra_mask<T, D, F, M>(
    a: &MaskedArray<T, D>,
    b: &MaskedArray<T, D>,
    op: F,
    extra_mask: M,
    op_name: &str,
) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Copy,
    D: Dimension,
    F: Fn(T, T) -> T,
    M: Fn(T, T) -> bool,
{
    // Fast path: identical shapes — no broadcasting needed.
    if a.shape() == b.shape() {
        let fill = a.fill_value;
        let mut any_masked = false;
        let mut result_mask = Vec::with_capacity(a.size());
        let mut data = Vec::with_capacity(a.size());
        for (((&x, &y), &am), &bm) in a
            .data()
            .iter()
            .zip(b.data().iter())
            .zip(a.mask().iter())
            .zip(b.mask().iter())
        {
            let masked = am || bm || extra_mask(x, y);
            any_masked |= masked;
            result_mask.push(masked);
            data.push(if masked { x } else { op(x, y) });
        }
        let result_data = Array::from_vec(a.dim().clone(), data)?;
        let mut result = if any_masked {
            let mask_arr = Array::from_vec(a.dim().clone(), result_mask)?;
            MaskedArray::new(result_data, mask_arr)?
        } else {
            MaskedArray::from_data(result_data)?
        };
        result.fill_value = fill;
        return Ok(result);
    }

    // Broadcasting path.
    let (pair, result_dim) = broadcast_masked_pair(a, b, op_name)?;
    let fill = a.fill_value;
    let n = pair.a_data.len();
    let mut result_data = Vec::with_capacity(n);
    let mut result_mask = Vec::with_capacity(n);
    let mut any_masked = false;
    for i in 0..n {
        let m = pair.a_mask[i] || pair.b_mask[i] || extra_mask(pair.a_data[i], pair.b_data[i]);
        any_masked |= m;
        result_mask.push(m);
        result_data.push(if m {
            pair.a_data[i]
        } else {
            op(pair.a_data[i], pair.b_data[i])
        });
    }
    let data_arr = Array::from_vec(result_dim.clone(), result_data)?;
    let mut out = if any_masked {
        let mask_arr = Array::from_vec(result_dim, result_mask)?;
        MaskedArray::new(data_arr, mask_arr)?
    } else {
        MaskedArray::from_data(data_arr)?
    };
    out.fill_value = fill;
    Ok(out)
}

/// Add two masked arrays elementwise with `NumPy` broadcasting.
///
/// Propagates the mask union and fills masked positions in the result data
/// with the receiver's [`MaskedArray::fill_value`].
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if shapes are not broadcast-compatible.
pub fn masked_add<T, D>(
    a: &MaskedArray<T, D>,
    b: &MaskedArray<T, D>,
) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Add<Output = T> + Copy,
    D: Dimension,
{
    masked_binary_op(a, b, |x, y| x + y, "masked_add")
}

/// Subtract two masked arrays elementwise with `NumPy` broadcasting.
pub fn masked_sub<T, D>(
    a: &MaskedArray<T, D>,
    b: &MaskedArray<T, D>,
) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Sub<Output = T> + Copy,
    D: Dimension,
{
    masked_binary_op(a, b, |x, y| x - y, "masked_sub")
}

/// Multiply two masked arrays elementwise with `NumPy` broadcasting.
pub fn masked_mul<T, D>(
    a: &MaskedArray<T, D>,
    b: &MaskedArray<T, D>,
) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Mul<Output = T> + Copy,
    D: Dimension,
{
    masked_binary_op(a, b, |x, y| x * y, "masked_mul")
}

/// Divide two masked arrays elementwise with `NumPy` broadcasting.
pub fn masked_div<T, D>(
    a: &MaskedArray<T, D>,
    b: &MaskedArray<T, D>,
) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Div<Output = T> + Copy,
    D: Dimension,
{
    masked_binary_op_with_extra_mask(a, b, |x, y| x / y, |_, y| y == T::zero(), "masked_div")
}

/// Generic masked-array-with-regular-array op with broadcasting.
///
/// Treats the regular array as unmasked. The result mask is the receiver's
/// mask broadcast to the common shape; result data uses `fill_value` at
/// masked positions.
fn masked_array_op<T, D, F>(
    ma: &MaskedArray<T, D>,
    arr: &Array<T, D>,
    op: F,
    op_name: &str,
) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Copy,
    D: Dimension,
    F: Fn(T, T) -> T,
{
    masked_array_op_with_extra_mask(ma, arr, op, |_, _| false, op_name)
}

fn masked_array_op_with_extra_mask<T, D, F, M>(
    ma: &MaskedArray<T, D>,
    arr: &Array<T, D>,
    op: F,
    extra_mask: M,
    op_name: &str,
) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Copy,
    D: Dimension,
    F: Fn(T, T) -> T,
    M: Fn(T, T) -> bool,
{
    let fill = ma.fill_value;

    // Fast path: identical shapes.
    if ma.shape() == arr.shape() {
        let mut any_masked = false;
        let mut data = Vec::with_capacity(ma.size());
        let mut mask = Vec::with_capacity(ma.size());
        for ((&x, &y), &m) in ma.data().iter().zip(arr.iter()).zip(ma.mask().iter()) {
            let masked = m || extra_mask(x, y);
            any_masked |= masked;
            mask.push(masked);
            data.push(if masked { x } else { op(x, y) });
        }
        let result_data = Array::from_vec(ma.dim().clone(), data)?;
        let mut out = if any_masked {
            let mask_arr = Array::from_vec(ma.dim().clone(), mask)?;
            MaskedArray::new(result_data, mask_arr)?
        } else {
            MaskedArray::from_data(result_data)?
        };
        out.fill_value = fill;
        return Ok(out);
    }

    // Broadcasting path.
    let target_shape = broadcast_shapes(ma.shape(), arr.shape()).map_err(|_| {
        FerrayError::shape_mismatch(format!(
            "{}: shapes {:?} and {:?} are not broadcast-compatible",
            op_name,
            ma.shape(),
            arr.shape()
        ))
    })?;
    let ma_data_view = broadcast_to(ma.data(), &target_shape)?;
    let ma_mask_view = broadcast_to(ma.mask(), &target_shape)?;
    let arr_view = broadcast_to(arr, &target_shape)?;

    let ma_data: Vec<T> = ma_data_view.iter().copied().collect();
    let ma_mask: Vec<bool> = ma_mask_view.iter().copied().collect();
    let arr_data: Vec<T> = arr_view.iter().copied().collect();

    let n = ma_data.len();
    let mut result_data = Vec::with_capacity(n);
    let mut result_mask = Vec::with_capacity(n);
    let mut any_masked = false;
    for i in 0..n {
        let m = ma_mask[i] || extra_mask(ma_data[i], arr_data[i]);
        any_masked |= m;
        result_mask.push(m);
        result_data.push(if m {
            ma_data[i]
        } else {
            op(ma_data[i], arr_data[i])
        });
    }
    let result_dim = D::from_dim_slice(&target_shape).ok_or_else(|| {
        FerrayError::shape_mismatch(format!(
            "{op_name}: cannot represent broadcast result shape {target_shape:?} as the input dimension type"
        ))
    })?;
    let data_arr = Array::from_vec(result_dim.clone(), result_data)?;
    let mut out = if any_masked {
        let mask_arr = Array::from_vec(result_dim, result_mask)?;
        MaskedArray::new(data_arr, mask_arr)?
    } else {
        MaskedArray::from_data(data_arr)?
    };
    out.fill_value = fill;
    Ok(out)
}

/// Add a masked array and a regular array, treating the regular array as
/// unmasked. Broadcasts within the same dimension type.
pub fn masked_add_array<T, D>(
    ma: &MaskedArray<T, D>,
    arr: &Array<T, D>,
) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Add<Output = T> + Copy,
    D: Dimension,
{
    masked_array_op(ma, arr, |x, y| x + y, "masked_add_array")
}

/// Subtract a regular array from a masked array, treating the regular array
/// as unmasked. Broadcasts within the same dimension type.
pub fn masked_sub_array<T, D>(
    ma: &MaskedArray<T, D>,
    arr: &Array<T, D>,
) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Sub<Output = T> + Copy,
    D: Dimension,
{
    masked_array_op(ma, arr, |x, y| x - y, "masked_sub_array")
}

/// Multiply a masked array and a regular array, treating the regular array
/// as unmasked. Broadcasts within the same dimension type.
pub fn masked_mul_array<T, D>(
    ma: &MaskedArray<T, D>,
    arr: &Array<T, D>,
) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Mul<Output = T> + Copy,
    D: Dimension,
{
    masked_array_op(ma, arr, |x, y| x * y, "masked_mul_array")
}

/// Divide a masked array by a regular array, treating the regular array as
/// unmasked. Broadcasts within the same dimension type.
pub fn masked_div_array<T, D>(
    ma: &MaskedArray<T, D>,
    arr: &Array<T, D>,
) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Div<Output = T> + Copy,
    D: Dimension,
{
    masked_array_op_with_extra_mask(
        ma,
        arr,
        |x, y| x / y,
        |_, y| y == T::zero(),
        "masked_div_array",
    )
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

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::Ix1;

    fn ma1d(data: Vec<f64>, mask: Vec<bool>) -> MaskedArray<f64, Ix1> {
        let n = data.len();
        let d = Array::<f64, Ix1>::from_vec(Ix1::new([n]), data).unwrap();
        let m = Array::<bool, Ix1>::from_vec(Ix1::new([n]), mask).unwrap();
        MaskedArray::new(d, m).unwrap()
    }

    // ---- #275: masked_div with division by zero ----
    //
    // NumPy's np.ma auto-masks division-by-zero results. ferray-ma follows
    // that behavior and keeps the left operand's data payload under masked
    // result positions.

    #[test]
    fn masked_div_positive_by_zero_yields_positive_infinity_unmasked() {
        let a = ma1d(vec![1.0, 2.0, 3.0], vec![false; 3]);
        let b = ma1d(vec![1.0, 0.0, 3.0], vec![false; 3]);
        let r = masked_div(&a, &b).unwrap();
        let rd: Vec<f64> = r.data().iter().copied().collect();
        let rm: Vec<bool> = r.mask().iter().copied().collect();
        assert_eq!(rd[0], 1.0);
        assert_eq!(rd[1], 2.0);
        assert_eq!(rd[2], 1.0);
        assert_eq!(rm, vec![false, true, false]);
    }

    #[test]
    fn masked_div_negative_by_zero_yields_negative_infinity_unmasked() {
        let a = ma1d(vec![-4.0], vec![false]);
        let b = ma1d(vec![0.0], vec![false]);
        let r = masked_div(&a, &b).unwrap();
        let v = r.data().iter().next().copied().unwrap();
        assert_eq!(v, -4.0);
        assert!(r.mask().iter().next().copied().unwrap());
    }

    #[test]
    fn masked_div_zero_by_zero_yields_nan_unmasked() {
        let a = ma1d(vec![0.0], vec![false]);
        let b = ma1d(vec![0.0], vec![false]);
        let r = masked_div(&a, &b).unwrap();
        let v = r.data().iter().next().copied().unwrap();
        assert_eq!(v, 0.0);
        assert!(r.mask().iter().next().copied().unwrap());
    }

    #[test]
    fn masked_div_skips_op_at_masked_divisor_positions() {
        // If the divisor is masked, the op is never evaluated — the masked
        // slot is filled with the receiver's fill_value, not inf/NaN.
        let a = ma1d(vec![1.0, 2.0, 3.0], vec![false; 3]).with_fill_value(-42.0);
        let b = ma1d(vec![2.0, 0.0, 4.0], vec![false, true, false]);
        let r = masked_div(&a, &b).unwrap();
        let rd: Vec<f64> = r.data().iter().copied().collect();
        let rm: Vec<bool> = r.mask().iter().copied().collect();
        assert_eq!(rd, vec![0.5, 2.0, 0.75]);
        assert_eq!(rm, vec![false, true, false]);
        // The masked position must not carry the div-by-zero IEEE value.
        assert!(!rd[1].is_infinite() && !rd[1].is_nan());
    }

    #[test]
    fn masked_div_array_by_zero_yields_infinity_unmasked() {
        // Same behavior holds for the masked-by-regular-array variant.
        let a = ma1d(vec![5.0, 6.0], vec![false; 2]);
        let divisor = Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![0.0, 2.0]).unwrap();
        let r = masked_div_array(&a, &divisor).unwrap();
        let rd: Vec<f64> = r.data().iter().copied().collect();
        assert_eq!(rd[0], 5.0);
        assert_eq!(rd[1], 3.0);
        let rm: Vec<bool> = r.mask().iter().copied().collect();
        assert_eq!(rm, vec![true, false]);
    }
}
