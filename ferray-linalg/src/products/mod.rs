// ferray-linalg: Matrix products (REQ-1 through REQ-7b)
//
// dot, vdot, inner, outer, matmul, kron, multi_dot, vecdot, tensordot, einsum

/// Einstein summation notation.
pub mod einsum;
/// Tensor dot product along specified axes.
pub mod tensordot;

use ferray_core::array::owned::Array;
use ferray_core::dimension::{Ix2, IxDyn};
use ferray_core::error::{FerrayError, FerrayResult};

use crate::scalar::LinalgFloat;

pub use einsum::einsum;
pub use tensordot::{TensordotAxes, tensordot};

/// Borrow contiguous data directly or copy if strided, avoiding allocation
/// for the common case of C-contiguous arrays.
enum DataRef<'a, T> {
    Borrowed(&'a [T]),
    Owned(Vec<T>),
}

impl<T> std::ops::Deref for DataRef<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        match self {
            DataRef::Borrowed(s) => s,
            DataRef::Owned(v) => v,
        }
    }
}

fn borrow_data<T: LinalgFloat>(a: &Array<T, IxDyn>) -> DataRef<'_, T> {
    if let Some(s) = a.as_slice() {
        DataRef::Borrowed(s)
    } else {
        DataRef::Owned(a.iter().copied().collect())
    }
}

/// Generalized dot product matching `np.dot` semantics.
///
/// - 1D x 1D: inner product (scalar returned as 1-element array)
/// - 2D x 2D: matrix multiplication
/// - ND x 1D: sum over last axis of a
/// - ND x MD: sum over last axis of a and second-to-last of b
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if dimensions are incompatible.
pub fn dot<T: LinalgFloat>(
    a: &Array<T, IxDyn>,
    b: &Array<T, IxDyn>,
) -> FerrayResult<Array<T, IxDyn>> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let zero = T::from_f64_const(0.0);

    match (a_shape.len(), b_shape.len()) {
        (1, 1) => {
            // Inner product
            if a_shape[0] != b_shape[0] {
                return Err(FerrayError::shape_mismatch(format!(
                    "dot: vectors have different lengths {} and {}",
                    a_shape[0], b_shape[0]
                )));
            }
            let sum: T = a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| x * y)
                .fold(zero, |acc, v| acc + v);
            Array::from_vec(IxDyn::new(&[]), vec![sum])
                .or_else(|_| Array::from_vec(IxDyn::new(&[1]), vec![sum]))
        }
        (2, 2) => {
            // Matrix multiplication
            let result = matmul_2d(a, b)?;
            let data: Vec<T> = result.iter().copied().collect();
            Array::from_vec(IxDyn::new(result.shape()), data)
        }
        (_, 1) => {
            // Sum product over last axis of a with b
            let k = a_shape[a_shape.len() - 1];
            if k != b_shape[0] {
                return Err(FerrayError::shape_mismatch(format!(
                    "dot: last axis of a ({}) != length of b ({})",
                    k, b_shape[0]
                )));
            }
            let out_shape: Vec<usize> = a_shape[..a_shape.len() - 1].to_vec();
            let out_size: usize = out_shape.iter().product::<usize>().max(1);

            let a_data = borrow_data(a);
            let b_data = borrow_data(b);

            let mut result = Vec::with_capacity(out_size);
            for i in 0..out_size {
                let mut sum = zero;
                for j in 0..k {
                    sum = sum + a_data[i * k + j] * b_data[j];
                }
                result.push(sum);
            }

            if out_shape.is_empty() {
                Array::from_vec(IxDyn::new(&[1]), result)
            } else {
                Array::from_vec(IxDyn::new(&out_shape), result)
            }
        }
        _ => {
            // General case: use tensordot
            tensordot(a, b, TensordotAxes::Scalar(1))
        }
    }
}

/// Flattened dot product with complex conjugation.
///
/// Flattens both arrays and computes the dot product.
/// For real arrays, this is the same as `dot` on flattened inputs.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if total element counts differ.
pub fn vdot<T: LinalgFloat>(a: &Array<T, IxDyn>, b: &Array<T, IxDyn>) -> FerrayResult<T> {
    if a.size() != b.size() {
        return Err(FerrayError::shape_mismatch(format!(
            "vdot: arrays have different sizes {} and {}",
            a.size(),
            b.size()
        )));
    }
    let a_data = borrow_data(a);
    let b_data = borrow_data(b);
    let zero = T::from_f64_const(0.0);
    Ok(a_data
        .iter()
        .zip(b_data.iter())
        .map(|(&x, &y)| x * y)
        .fold(zero, |acc, v| acc + v))
}

/// Inner product of two arrays.
///
/// For 1D arrays, this is the same as `dot`.
/// For higher dimensions, it sums over the last axis of both arrays.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if the last dimensions don't match.
pub fn inner<T: LinalgFloat>(
    a: &Array<T, IxDyn>,
    b: &Array<T, IxDyn>,
) -> FerrayResult<Array<T, IxDyn>> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape.len() == 1 && b_shape.len() == 1 {
        return dot(a, b);
    }

    // For ND arrays, contract last axis of a with last axis of b
    let last_a = a_shape.last().copied().unwrap_or(1);
    let last_b = b_shape.last().copied().unwrap_or(1);
    if last_a != last_b {
        return Err(FerrayError::shape_mismatch(format!(
            "inner: last dimensions must match ({} != {})",
            last_a, last_b
        )));
    }

    let axes_a = vec![a_shape.len() - 1];
    let axes_b = vec![b_shape.len() - 1];
    tensordot(a, b, TensordotAxes::Pairs(axes_a, axes_b))
}

/// Outer product of two arrays.
///
/// Computes the outer product of two 1D arrays, producing a 2D array.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if inputs are not 1D.
pub fn outer<T: LinalgFloat>(
    a: &Array<T, IxDyn>,
    b: &Array<T, IxDyn>,
) -> FerrayResult<Array<T, IxDyn>> {
    let a_data = borrow_data(a);
    let b_data = borrow_data(b);
    let m = a_data.len();
    let n = b_data.len();

    let mut result = Vec::with_capacity(m * n);
    for &ai in &*a_data {
        for &bj in &*b_data {
            result.push(ai * bj);
        }
    }

    Array::from_vec(IxDyn::new(&[m, n]), result)
}

/// Matrix multiplication matching `np.matmul` / `@` semantics.
///
/// Supports:
/// - 2D x 2D: standard matrix multiplication
/// - 1D x 2D: vector-matrix (prepend 1 to first)
/// - 2D x 1D: matrix-vector (append 1 to second)
/// - ND x ND: batched matmul over leading dimensions
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if inner dimensions don't match.
pub fn matmul<T: LinalgFloat>(
    a: &Array<T, IxDyn>,
    b: &Array<T, IxDyn>,
) -> FerrayResult<Array<T, IxDyn>> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let zero = T::from_f64_const(0.0);

    match (a_shape.len(), b_shape.len()) {
        (1, 1) => {
            return Err(FerrayError::shape_mismatch(
                "matmul: cannot multiply two 1D arrays (use dot instead)",
            ));
        }
        (1, 2) => {
            // Vector-matrix: treat a as (1, k)
            let k = a_shape[0];
            let n = b_shape[1];
            if k != b_shape[0] {
                return Err(FerrayError::shape_mismatch(format!(
                    "matmul: shapes ({},) and ({},{}) not aligned",
                    k, b_shape[0], n
                )));
            }
            let a_data: Vec<T> = a.iter().copied().collect();
            let b_data: Vec<T> = b.iter().copied().collect();
            let mut result = vec![zero; n];
            for j in 0..n {
                for p in 0..k {
                    result[j] = result[j] + a_data[p] * b_data[p * n + j];
                }
            }
            Array::from_vec(IxDyn::new(&[n]), result)
        }
        (2, 1) => {
            // Matrix-vector: treat b as (k, 1)
            let (m, k) = (a_shape[0], a_shape[1]);
            if k != b_shape[0] {
                return Err(FerrayError::shape_mismatch(format!(
                    "matmul: shapes ({},{}) and ({},) not aligned",
                    m, k, b_shape[0]
                )));
            }
            let a_data: Vec<T> = a.iter().copied().collect();
            let b_data: Vec<T> = b.iter().copied().collect();
            let mut result = vec![zero; m];
            for i in 0..m {
                for p in 0..k {
                    result[i] = result[i] + a_data[i * k + p] * b_data[p];
                }
            }
            Array::from_vec(IxDyn::new(&[m]), result)
        }
        (2, 2) => {
            let result = matmul_2d(a, b)?;
            let data: Vec<T> = result.iter().copied().collect();
            Array::from_vec(IxDyn::new(result.shape()), data)
        }
        _ => {
            // Batched matmul: use tensordot-like approach
            matmul_batched(a, b)
        }
    }
}

/// Below this threshold, use the naive ikj loop (avoids faer setup overhead).
const FAER_MATMUL_THRESHOLD: usize = 64;

/// Above this threshold, use faer with Rayon parallelism.
const FAER_PARALLEL_THRESHOLD: usize = 256;

/// Compute C = A @ B where A is m×k (row-major), B is k×n (row-major),
/// and returns C as a row-major flat Vec of length m×n.
///
/// Dispatches to a naive ikj loop for small problems (avoiding faer setup
/// overhead) and to faer's optimized matmul otherwise, with Rayon
/// parallelism enabled for larger sizes. Used by every 2-D matmul site in
/// this crate (`matmul_2d`, `einsum` matmul shortcut, `matrix_power`) so
/// they all share the same small/large dispatch policy.
pub(crate) fn matmul_raw<T: LinalgFloat>(
    a: &[T],
    b: &[T],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<T> {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);

    let zero = T::from_f64_const(0.0);
    let max_dim = m.max(n).max(k);

    // Degenerate shapes: nothing to do.
    if m == 0 || n == 0 {
        return vec![zero; m * n];
    }
    if k == 0 {
        return vec![zero; m * n];
    }

    // Small matrices: naive ikj loop is faster than paying faer's setup cost.
    if max_dim <= FAER_MATMUL_THRESHOLD {
        let mut result = vec![zero; m * n];
        for i in 0..m {
            for p in 0..k {
                let a_ip = a[i * k + p];
                for j in 0..n {
                    result[i * n + j] = result[i * n + j] + a_ip * b[p * n + j];
                }
            }
        }
        return result;
    }

    // Faer's column-major Mat fed directly from the row-major slices.
    let a_faer = faer::Mat::<T>::from_fn(m, k, |i, j| a[i * k + j]);
    let b_faer = faer::Mat::<T>::from_fn(k, n, |i, j| b[i * n + j]);
    let mut c_faer = faer::Mat::<T>::zeros(m, n);

    let par = if max_dim >= FAER_PARALLEL_THRESHOLD {
        faer::Par::Rayon(
            std::num::NonZeroUsize::new(0).unwrap_or(std::num::NonZeroUsize::new(1).unwrap()),
        )
    } else {
        faer::Par::Seq
    };

    faer::linalg::matmul::matmul(
        c_faer.as_mut(),
        faer::Accum::Replace,
        a_faer.as_ref(),
        b_faer.as_ref(),
        T::from_f64_const(1.0),
        par,
    );

    // Faer is column-major; transcribe back to row-major.
    let mut result = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            result.push(c_faer[(i, j)]);
        }
    }
    result
}

fn matmul_2d<T: LinalgFloat>(
    a: &Array<T, IxDyn>,
    b: &Array<T, IxDyn>,
) -> FerrayResult<Array<T, Ix2>> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let (m, k1) = (a_shape[0], a_shape[1]);
    let (k2, n) = (b_shape[0], b_shape[1]);
    if k1 != k2 {
        return Err(FerrayError::shape_mismatch(format!(
            "matmul: inner dimensions don't match ({}x{} @ {}x{})",
            m, k1, k2, n
        )));
    }

    let k = k1;
    let a_data = borrow_data(a);
    let b_data = borrow_data(b);
    let result = matmul_raw::<T>(&a_data, &b_data, m, k, n);
    Array::from_vec(Ix2::new([m, n]), result)
}

fn matmul_batched<T: LinalgFloat>(
    a: &Array<T, IxDyn>,
    b: &Array<T, IxDyn>,
) -> FerrayResult<Array<T, IxDyn>> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let zero = T::from_f64_const(0.0);

    if a_shape.len() < 2 || b_shape.len() < 2 {
        return Err(FerrayError::shape_mismatch(
            "matmul: need at least 2D arrays for batched matmul",
        ));
    }

    let a_m = a_shape[a_shape.len() - 2];
    let a_k = a_shape[a_shape.len() - 1];
    let b_k = b_shape[b_shape.len() - 2];
    let b_n = b_shape[b_shape.len() - 1];

    if a_k != b_k {
        return Err(FerrayError::shape_mismatch(format!(
            "matmul: inner dimensions don't match ({} != {})",
            a_k, b_k
        )));
    }

    // Broadcast batch dimensions
    let a_batch = &a_shape[..a_shape.len() - 2];
    let b_batch = &b_shape[..b_shape.len() - 2];
    let batch_shape = broadcast_shapes(a_batch, b_batch)?;

    let batch_size: usize = batch_shape.iter().product::<usize>().max(1);

    let a_data: Vec<T> = a.iter().copied().collect();
    let b_data: Vec<T> = b.iter().copied().collect();
    let a_mat_size = a_m * a_k;
    let b_mat_size = b_k * b_n;
    let out_mat_size = a_m * b_n;

    let a_batch_size: usize = a_batch.iter().product::<usize>().max(1);
    let b_batch_size: usize = b_batch.iter().product::<usize>().max(1);

    let mut result = vec![zero; batch_size * out_mat_size];

    for batch_idx in 0..batch_size {
        let a_idx = batch_idx % a_batch_size;
        let b_idx = batch_idx % b_batch_size;
        let a_offset = a_idx * a_mat_size;
        let b_offset = b_idx * b_mat_size;
        let out_offset = batch_idx * out_mat_size;

        let a_slice = &a_data[a_offset..a_offset + a_mat_size];
        let b_slice = &b_data[b_offset..b_offset + b_mat_size];
        let c = matmul_raw::<T>(a_slice, b_slice, a_m, a_k, b_n);
        result[out_offset..out_offset + out_mat_size].copy_from_slice(&c);
    }

    let mut out_shape: Vec<usize> = batch_shape;
    out_shape.push(a_m);
    out_shape.push(b_n);
    Array::from_vec(IxDyn::new(&out_shape), result)
}

fn broadcast_shapes(a: &[usize], b: &[usize]) -> FerrayResult<Vec<usize>> {
    let max_len = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_len);
    for i in 0..max_len {
        let da = if i < max_len - a.len() {
            1
        } else {
            a[i - (max_len - a.len())]
        };
        let db = if i < max_len - b.len() {
            1
        } else {
            b[i - (max_len - b.len())]
        };
        if da == db {
            result.push(da);
        } else if da == 1 {
            result.push(db);
        } else if db == 1 {
            result.push(da);
        } else {
            return Err(FerrayError::broadcast_failure(a, b));
        }
    }
    Ok(result)
}

/// Kronecker product of two 2D arrays.
///
/// The Kronecker product of A (m x n) and B (p x q) is an (m*p x n*q) matrix.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if inputs are not 2D.
pub fn kron<T: LinalgFloat>(
    a: &Array<T, IxDyn>,
    b: &Array<T, IxDyn>,
) -> FerrayResult<Array<T, IxDyn>> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(FerrayError::shape_mismatch("kron: both arrays must be 2D"));
    }

    let (m, n) = (a_shape[0], a_shape[1]);
    let (p, q) = (b_shape[0], b_shape[1]);
    let a_data: Vec<T> = a.iter().copied().collect();
    let b_data: Vec<T> = b.iter().copied().collect();

    let out_rows = m * p;
    let out_cols = n * q;
    let zero = T::from_f64_const(0.0);
    let mut result = vec![zero; out_rows * out_cols];

    for i in 0..m {
        for j in 0..n {
            let a_ij = a_data[i * n + j];
            for k in 0..p {
                for l in 0..q {
                    result[(i * p + k) * out_cols + (j * q + l)] = a_ij * b_data[k * q + l];
                }
            }
        }
    }

    Array::from_vec(IxDyn::new(&[out_rows, out_cols]), result)
}

/// Optimized chain matrix multiplication using dynamic programming.
///
/// `multi_dot` computes the product of a chain of matrices, choosing the
/// optimal parenthesization to minimize total floating point operations.
/// For long chains, this can be 10-100x faster than naive left-to-right chaining.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if adjacent matrix dimensions are incompatible.
/// - `FerrayError::InvalidValue` if fewer than 2 matrices are provided.
pub fn multi_dot<T: LinalgFloat>(arrays: &[&Array<T, IxDyn>]) -> FerrayResult<Array<T, IxDyn>> {
    if arrays.len() < 2 {
        return Err(FerrayError::invalid_value(
            "multi_dot: need at least 2 matrices",
        ));
    }

    if arrays.len() == 2 {
        return matmul(arrays[0], arrays[1]);
    }

    // Extract dimensions for chain matrix multiplication
    let n = arrays.len();
    let mut dims = Vec::with_capacity(n + 1);
    for (i, arr) in arrays.iter().enumerate() {
        let shape = arr.shape();
        if shape.len() != 2 {
            // First and last can be 1D
            if i == 0 && shape.len() == 1 {
                dims.push(1);
                dims.push(shape[0]);
                continue;
            } else if i == n - 1 && shape.len() == 1 {
                dims.push(shape[0]);
                if i == n - 1 {
                    dims.push(1);
                }
                continue;
            }
            return Err(FerrayError::shape_mismatch(format!(
                "multi_dot: matrix {} has {} dimensions (expected 2)",
                i,
                shape.len()
            )));
        }
        if i == 0 {
            dims.push(shape[0]);
        }
        dims.push(shape[1]);
    }

    // Verify compatible dimensions
    for i in 1..arrays.len() {
        let prev_shape = arrays[i - 1].shape();
        let curr_shape = arrays[i].shape();
        let prev_cols = prev_shape.last().copied().unwrap_or(0);
        let curr_rows = if curr_shape.len() == 1 {
            curr_shape[0]
        } else {
            curr_shape[0]
        };
        if prev_cols != curr_rows {
            return Err(FerrayError::shape_mismatch(format!(
                "multi_dot: shapes of matrices {} and {} not aligned ({} != {})",
                i - 1,
                i,
                prev_cols,
                curr_rows
            )));
        }
    }

    // Dynamic programming for optimal parenthesization
    let mut cost = vec![vec![0u64; n]; n];
    let mut split = vec![vec![0usize; n]; n];

    for chain_len in 2..=n {
        for i in 0..=n - chain_len {
            let j = i + chain_len - 1;
            cost[i][j] = u64::MAX;
            for k in i..j {
                let q = cost[i][k]
                    + cost[k + 1][j]
                    + dims[i] as u64 * dims[k + 1] as u64 * dims[j + 1] as u64;
                if q < cost[i][j] {
                    cost[i][j] = q;
                    split[i][j] = k;
                }
            }
        }
    }

    // Recursively multiply according to optimal split
    fn multiply_chain<T: LinalgFloat>(
        arrays: &[&Array<T, IxDyn>],
        split: &[Vec<usize>],
        i: usize,
        j: usize,
    ) -> FerrayResult<Array<T, IxDyn>> {
        if i == j {
            return Ok(arrays[i].clone());
        }
        let k = split[i][j];
        let left = multiply_chain(arrays, split, i, k)?;
        let right = multiply_chain(arrays, split, k + 1, j)?;
        matmul(&left, &right)
    }

    multiply_chain(arrays, &split, 0, n - 1)
}

/// Vector dot product along a specified axis.
///
/// Computes the dot product of corresponding vectors along `axis`.
/// This is equivalent to `numpy.vecdot` (new in NumPy 2.0).
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if arrays have incompatible shapes.
/// - `FerrayError::AxisOutOfBounds` if axis is out of range.
pub fn vecdot<T: LinalgFloat>(
    a: &Array<T, IxDyn>,
    b: &Array<T, IxDyn>,
    axis: Option<isize>,
) -> FerrayResult<Array<T, IxDyn>> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let zero = T::from_f64_const(0.0);

    if a_shape != b_shape {
        return Err(FerrayError::shape_mismatch(format!(
            "vecdot: shapes {:?} and {:?} must match",
            a_shape, b_shape
        )));
    }

    let ndim = a_shape.len();
    let ax = match axis {
        None => {
            if ndim == 0 {
                return Err(FerrayError::shape_mismatch(
                    "vecdot: 0D arrays not supported",
                ));
            }
            ndim - 1
        }
        Some(ax) => {
            let ax = if ax < 0 {
                (ndim as isize + ax) as usize
            } else {
                ax as usize
            };
            if ax >= ndim {
                return Err(FerrayError::axis_out_of_bounds(ax, ndim));
            }
            ax
        }
    };

    let axis_len = a_shape[ax];
    let mut out_shape: Vec<usize> = Vec::with_capacity(ndim - 1);
    for (i, &s) in a_shape.iter().enumerate() {
        if i != ax {
            out_shape.push(s);
        }
    }

    let out_size: usize = out_shape.iter().product::<usize>().max(1);

    // Compute strides
    let mut a_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        a_strides[i] = a_strides[i + 1] * a_shape[i + 1];
    }

    let a_data: Vec<T> = a.iter().copied().collect();
    let b_data: Vec<T> = b.iter().copied().collect();
    let mut result = vec![zero; out_size];

    // For each output element, sum over the axis
    let mut out_idx = vec![0usize; ndim - 1];
    for flat in 0..out_size {
        // Decode flat index to out_idx
        let mut rem = flat;
        for d in (0..out_shape.len()).rev() {
            if out_shape[d] > 0 {
                out_idx[d] = rem % out_shape[d];
                rem /= out_shape[d];
            }
        }

        // Map out_idx back to full index (inserting axis dim)
        let mut sum = zero;
        for k in 0..axis_len {
            let mut full_flat = 0;
            let mut od = 0;
            for d in 0..ndim {
                let idx = if d == ax {
                    k
                } else {
                    let v = out_idx[od];
                    od += 1;
                    v
                };
                full_flat += idx * a_strides[d];
            }
            sum = sum + a_data[full_flat] * b_data[full_flat];
        }
        result[flat] = sum;
    }

    if out_shape.is_empty() {
        Array::from_vec(IxDyn::new(&[1]), result)
    } else {
        Array::from_vec(IxDyn::new(&out_shape), result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_1d() {
        let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[3]), vec![4.0, 5.0, 6.0]).unwrap();
        let c = dot(&a, &b).unwrap();
        let d: Vec<f64> = c.iter().copied().collect();
        assert!((d[0] - 32.0).abs() < 1e-10);
    }

    #[test]
    fn dot_2d() {
        let a =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let b = Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[3, 2]),
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        )
        .unwrap();
        let c = dot(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
    }

    #[test]
    fn dot_f32() {
        let a = Array::<f32, IxDyn>::from_vec(IxDyn::new(&[3]), vec![1.0f32, 2.0, 3.0]).unwrap();
        let b = Array::<f32, IxDyn>::from_vec(IxDyn::new(&[3]), vec![4.0f32, 5.0, 6.0]).unwrap();
        let c = dot(&a, &b).unwrap();
        let d: Vec<f32> = c.iter().copied().collect();
        assert!((d[0] - 32.0f32).abs() < 1e-5);
    }

    #[test]
    fn vdot_test() {
        let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[3]), vec![4.0, 5.0, 6.0]).unwrap();
        let result = vdot(&a, &b).unwrap();
        assert!((result - 32.0).abs() < 1e-10);
    }

    #[test]
    fn outer_test() {
        let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2]), vec![1.0, 2.0]).unwrap();
        let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[3]), vec![3.0, 4.0, 5.0]).unwrap();
        let c = outer(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
        let data: Vec<f64> = c.iter().copied().collect();
        assert_eq!(data, vec![3.0, 4.0, 5.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn matmul_2d_test() {
        let a =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let b = Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[3, 2]),
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        )
        .unwrap();
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        let data: Vec<f64> = c.iter().copied().collect();
        assert!((data[0] - 58.0).abs() < 1e-10);
        assert!((data[1] - 64.0).abs() < 1e-10);
        assert!((data[2] - 139.0).abs() < 1e-10);
        assert!((data[3] - 154.0).abs() < 1e-10);
    }

    #[test]
    fn matmul_vec_mat() {
        let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        let b =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[3, 2]), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
                .unwrap();
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2]);
    }

    #[test]
    fn matmul_mat_vec() {
        let a =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[3]), vec![1.0, 1.0, 1.0]).unwrap();
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2]);
        let data: Vec<f64> = c.iter().copied().collect();
        assert!((data[0] - 6.0).abs() < 1e-10);
        assert!((data[1] - 15.0).abs() < 1e-10);
    }

    #[test]
    fn kron_test() {
        let a =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2]), vec![0.0, 5.0, 6.0, 7.0]).unwrap();
        let c = kron(&a, &b).unwrap();
        assert_eq!(c.shape(), &[4, 4]);
    }

    #[test]
    fn multi_dot_test() {
        let a =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[3, 4]), vec![1.0; 12]).unwrap();
        let c = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[4, 2]), vec![1.0; 8]).unwrap();

        let result = multi_dot(&[&a, &b, &c]).unwrap();
        // Compare with naive left-to-right
        let ab = matmul(&a, &b).unwrap();
        let abc = matmul(&ab, &c).unwrap();

        let r1: Vec<f64> = result.iter().copied().collect();
        let r2: Vec<f64> = abc.iter().copied().collect();
        assert_eq!(result.shape(), abc.shape());
        for i in 0..r1.len() {
            assert!(
                (r1[i] - r2[i]).abs() < 1e-10,
                "multi_dot[{}] = {} != naive[{}] = {}",
                i,
                r1[i],
                i,
                r2[i]
            );
        }
    }

    #[test]
    fn vecdot_test() {
        let a =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let b =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
                .unwrap();
        let c = vecdot(&a, &b, None).unwrap();
        assert_eq!(c.shape(), &[2]);
        let data: Vec<f64> = c.iter().copied().collect();
        assert!((data[0] - 6.0).abs() < 1e-10);
        assert!((data[1] - 15.0).abs() < 1e-10);
    }

    #[test]
    fn matmul_100x100() {
        // AC-1: matmul of two (100,100) f64 matrices
        let n = 100;
        let a_data: Vec<f64> = (0..n * n).map(|i| (i as f64) * 0.01).collect();
        let b_data: Vec<f64> = (0..n * n).map(|i| ((n * n - i) as f64) * 0.01).collect();

        let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[n, n]), a_data.clone()).unwrap();
        let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[n, n]), b_data.clone()).unwrap();
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[n, n]);

        // Verify a few elements against naive computation
        let c_data: Vec<f64> = c.iter().copied().collect();
        for check_i in [0, 50, 99] {
            for check_j in [0, 50, 99] {
                let mut expected = 0.0;
                for k in 0..n {
                    expected += a_data[check_i * n + k] * b_data[k * n + check_j];
                }
                let diff = (c_data[check_i * n + check_j] - expected).abs();
                let ulps = diff / (expected.abs() * f64::EPSILON).max(f64::MIN_POSITIVE);
                assert!(
                    ulps < 4.0 || diff < 1e-10,
                    "matmul[{},{}]: diff={}, ulps={}",
                    check_i,
                    check_j,
                    diff,
                    ulps
                );
            }
        }
    }

    #[test]
    fn matmul_batched_3d() {
        // Batch of 2 matrices: (2, 2, 3) @ (2, 3, 2) -> (2, 2, 2)
        let a = Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[2, 2, 3]),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let b = Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[2, 3, 2]),
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        )
        .unwrap();
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2, 2]);

        let data: Vec<f64> = c.iter().copied().collect();
        // batch 0: [[1,2,3],[4,5,6]] @ [[1,0],[0,1],[1,1]] = [[4,5],[10,11]]
        assert!((data[0] - 4.0).abs() < 1e-10);
        assert!((data[1] - 5.0).abs() < 1e-10);
        assert!((data[2] - 10.0).abs() < 1e-10);
        assert!((data[3] - 11.0).abs() < 1e-10);
        // batch 1: [[7,8,9],[10,11,12]] @ [[1,0],[0,1],[0,0]] = [[7,8],[10,11]]
        assert!((data[4] - 7.0).abs() < 1e-10);
        assert!((data[5] - 8.0).abs() < 1e-10);
        assert!((data[6] - 10.0).abs() < 1e-10);
        assert!((data[7] - 11.0).abs() < 1e-10);
    }
}
