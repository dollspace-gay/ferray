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

/// Helper: compute the union of two boolean mask arrays (element-wise OR).
///
/// Requires identical shapes — broadcasting is handled at a higher level
/// via [`broadcast_masked_pair`].
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
/// the masked array's `fill_value` into the result. Crate-internal so
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
    // branch in the inner loop (#157). NumPy's masked ufuncs take
    // the same shape — apply elementwise across the full data, then
    // patch masked positions with fill_value. This wastes compute on
    // masked elements (typically a small fraction of the array) but
    // unlocks SIMD on the dominant unmasked bulk.
    //
    // Trade-off: `f` is applied to the underlying data even at
    // masked positions. For partial-domain ops (log, sqrt, …) that
    // would otherwise panic or NaN on masked-but-invalid data, the
    // *_domain helpers in ufunc_support.rs handle the validation
    // separately before this point, and masked positions get
    // overwritten with fill_value below regardless of what `f`
    // produced for them.
    let data_vec: Vec<T> = ma.data().iter().map(|&v| f(v)).collect();
    let data: Vec<T> = data_vec
        .into_iter()
        .zip(ma.mask().iter())
        .map(|(v, m)| if *m { fill } else { v })
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
    // Fast path: identical shapes — no broadcasting needed.
    if a.shape() == b.shape() {
        let fill = a.fill_value;

        // Double-fast path (#506): neither input has a real mask, so
        // there's no mask to union and no masked positions to skip.
        // `op` applies to every element unconditionally and the
        // result stays in the nomask-sentinel state.
        if !a.has_real_mask() && !b.has_real_mask() {
            let data: Vec<T> = a
                .data()
                .iter()
                .zip(b.data().iter())
                .map(|(&x, &y)| op(x, y))
                .collect();
            let result_data = Array::from_vec(a.dim().clone(), data)?;
            let mut result = MaskedArray::from_data(result_data)?;
            result.fill_value = fill;
            return Ok(result);
        }

        // Slow path: at least one input has a real mask; compute the
        // union and apply `op` only at unmasked positions.
        let result_mask = mask_union(a.mask(), b.mask())?;
        let data: Vec<T> = a
            .data()
            .iter()
            .zip(b.data().iter())
            .zip(result_mask.iter())
            .map(|((x, y), m)| if *m { fill } else { op(*x, *y) })
            .collect();
        let result_data = Array::from_vec(a.dim().clone(), data)?;
        let mut result = MaskedArray::new(result_data, result_mask)?;
        result.fill_value = fill;
        return Ok(result);
    }

    // Broadcasting path.
    let (pair, result_dim) = broadcast_masked_pair(a, b, op_name)?;
    let fill = a.fill_value;
    let n = pair.a_data.len();
    let mut result_data = Vec::with_capacity(n);
    let mut result_mask = Vec::with_capacity(n);
    for i in 0..n {
        let m = pair.a_mask[i] || pair.b_mask[i];
        result_mask.push(m);
        result_data.push(if m {
            fill
        } else {
            op(pair.a_data[i], pair.b_data[i])
        });
    }
    let data_arr = Array::from_vec(result_dim.clone(), result_data)?;
    let mask_arr = Array::from_vec(result_dim, result_mask)?;
    let mut out = MaskedArray::new(data_arr, mask_arr)?;
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
    masked_binary_op(a, b, |x, y| x / y, "masked_div")
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
    let fill = ma.fill_value;

    // Fast path: identical shapes.
    if ma.shape() == arr.shape() {
        let data: Vec<T> = ma
            .data()
            .iter()
            .zip(arr.iter())
            .zip(ma.mask().iter())
            .map(|((x, y), m)| if *m { fill } else { op(*x, *y) })
            .collect();
        let result_data = Array::from_vec(ma.dim().clone(), data)?;
        let mut out = MaskedArray::new(result_data, ma.mask().clone())?;
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
    for i in 0..n {
        let m = ma_mask[i];
        result_mask.push(m);
        result_data.push(if m { fill } else { op(ma_data[i], arr_data[i]) });
    }
    let result_dim = D::from_dim_slice(&target_shape).ok_or_else(|| {
        FerrayError::shape_mismatch(format!(
            "{op_name}: cannot represent broadcast result shape {target_shape:?} as the input dimension type"
        ))
    })?;
    let data_arr = Array::from_vec(result_dim.clone(), result_data)?;
    let mask_arr = Array::from_vec(result_dim, result_mask)?;
    let mut out = MaskedArray::new(data_arr, mask_arr)?;
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
    masked_array_op(ma, arr, |x, y| x / y, "masked_div_array")
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
    // NumPy's np.ma auto-masks division-by-zero results. ferray-ma does NOT
    // auto-mask: division is a plain IEEE 754 f64 divide, so the result
    // carries ±inf / NaN at the offending positions and the output mask
    // equals the union of the input masks. These tests pin that current
    // behavior so any future NumPy-alignment work is an intentional change.

    #[test]
    fn masked_div_positive_by_zero_yields_positive_infinity_unmasked() {
        let a = ma1d(vec![1.0, 2.0, 3.0], vec![false; 3]);
        let b = ma1d(vec![1.0, 0.0, 3.0], vec![false; 3]);
        let r = masked_div(&a, &b).unwrap();
        let rd: Vec<f64> = r.data().iter().copied().collect();
        let rm: Vec<bool> = r.mask().iter().copied().collect();
        assert_eq!(rd[0], 1.0);
        assert!(rd[1].is_infinite() && rd[1].is_sign_positive());
        assert_eq!(rd[2], 1.0);
        // Current behavior: div-by-zero is NOT auto-masked.
        assert_eq!(rm, vec![false, false, false]);
    }

    #[test]
    fn masked_div_negative_by_zero_yields_negative_infinity_unmasked() {
        let a = ma1d(vec![-4.0], vec![false]);
        let b = ma1d(vec![0.0], vec![false]);
        let r = masked_div(&a, &b).unwrap();
        let v = r.data().iter().next().copied().unwrap();
        assert!(v.is_infinite() && v.is_sign_negative());
        assert!(!r.mask().iter().next().copied().unwrap());
    }

    #[test]
    fn masked_div_zero_by_zero_yields_nan_unmasked() {
        let a = ma1d(vec![0.0], vec![false]);
        let b = ma1d(vec![0.0], vec![false]);
        let r = masked_div(&a, &b).unwrap();
        let v = r.data().iter().next().copied().unwrap();
        assert!(v.is_nan());
        assert!(!r.mask().iter().next().copied().unwrap());
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
        assert_eq!(rd, vec![0.5, -42.0, 0.75]);
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
        assert!(rd[0].is_infinite() && rd[0].is_sign_positive());
        assert_eq!(rd[1], 3.0);
        let rm: Vec<bool> = r.mask().iter().copied().collect();
        assert_eq!(rm, vec![false, false]);
    }
}
