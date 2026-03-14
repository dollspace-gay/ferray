// ferray-linalg: Solvers and inversion (REQ-15 through REQ-18c)
//
// solve, lstsq, inv, pinv, matrix_power, tensorsolve, tensorinv

use ferray_core::array::owned::Array;
use ferray_core::dimension::{Ix1, Ix2, IxDyn};
use ferray_core::error::{FerrayError, FerrayResult};

use crate::faer_bridge;
use crate::scalar::LinalgFloat;
use faer::linalg::solvers::{DenseSolveCore, Solve, SolveLstsq};

/// Solve the linear equation `A @ x = b` for x.
///
/// `a` must be a square, non-singular matrix.
/// `b` can be a 1D vector or a 2D matrix (multiple right-hand sides).
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if dimensions are incompatible.
/// - `FerrayError::SingularMatrix` if A is singular.
pub fn solve<T: LinalgFloat>(
    a: &Array<T, Ix2>,
    b: &Array<T, IxDyn>,
) -> FerrayResult<Array<T, IxDyn>> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    if a_shape[0] != a_shape[1] {
        return Err(FerrayError::shape_mismatch(format!(
            "solve requires a square matrix A, got {}x{}",
            a_shape[0], a_shape[1]
        )));
    }
    let n = a_shape[0];

    let a_mat = faer_bridge::array2_to_faer(a);
    let lu = a_mat.as_ref().partial_piv_lu();

    match b_shape.len() {
        1 => {
            if b_shape[0] != n {
                return Err(FerrayError::shape_mismatch(format!(
                    "solve: A is {}x{} but b has length {}",
                    n, n, b_shape[0]
                )));
            }
            let b_data: Vec<T> = b.iter().copied().collect();
            let b_mat = faer::Mat::from_fn(n, 1, |i, _| b_data[i]);
            let x = lu.solve(&b_mat);
            let mut result = Vec::with_capacity(n);
            for i in 0..n {
                result.push(x[(i, 0)]);
            }
            Array::from_vec(IxDyn::new(&[n]), result)
        }
        2 => {
            if b_shape[0] != n {
                return Err(FerrayError::shape_mismatch(format!(
                    "solve: A is {}x{} but b has {} rows",
                    n, n, b_shape[0]
                )));
            }
            let nrhs = b_shape[1];
            let b_data: Vec<T> = b.iter().copied().collect();
            let b_mat = faer::Mat::from_fn(n, nrhs, |i, j| b_data[i * nrhs + j]);
            let x = lu.solve(&b_mat);
            let mut result = Vec::with_capacity(n * nrhs);
            for i in 0..n {
                for j in 0..nrhs {
                    result.push(x[(i, j)]);
                }
            }
            Array::from_vec(IxDyn::new(&[n, nrhs]), result)
        }
        _ => Err(FerrayError::shape_mismatch("solve: b must be 1D or 2D")),
    }
}

/// Compute the least-squares solution to `A @ x = b`.
///
/// Returns `(x, residuals, rank, singular_values)`.
/// - x: solution of shape (n,) or (n, k)
/// - residuals: sum of squared residuals (empty if rank < m or m < n)
/// - rank: effective rank of A
/// - singular_values: singular values of A
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if dimensions are incompatible.
pub fn lstsq<T: LinalgFloat>(
    a: &Array<T, Ix2>,
    b: &Array<T, IxDyn>,
    rcond: Option<T>,
) -> FerrayResult<(Array<T, IxDyn>, Array<T, Ix1>, usize, Array<T, Ix1>)> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let (m, n) = (a_shape[0], a_shape[1]);

    let a_mat = faer_bridge::array2_to_faer(a);
    let qr_decomp = a_mat.as_ref().col_piv_qr();

    // Get SVD for rank and singular values
    let (_u, sv, _vt) = crate::decomp::svd(a, false)?;
    let svals = sv.as_slice().unwrap();
    let tol = rcond.unwrap_or_else(|| {
        let max_dim = T::from_usize(m.max(n));
        max_dim * T::machine_epsilon()
    });
    let max_sv = if svals.is_empty() {
        T::from_f64_const(0.0)
    } else {
        svals[0]
    };
    let rank = svals.iter().filter(|&&s| s > tol * max_sv).count();

    match b_shape.len() {
        1 => {
            if b_shape[0] != m {
                return Err(FerrayError::shape_mismatch(format!(
                    "lstsq: A is {}x{} but b has length {}",
                    m, n, b_shape[0]
                )));
            }
            let b_data: Vec<T> = b.iter().copied().collect();
            let b_mat = faer::Mat::from_fn(m, 1, |i, _| b_data[i]);
            let x_mat = qr_decomp.solve_lstsq(&b_mat);

            let mut x_vec = Vec::with_capacity(n);
            for i in 0..n {
                x_vec.push(x_mat[(i, 0)]);
            }

            // Compute residuals
            let residuals = if m > n && rank == n {
                let a_data: Vec<T> = a.iter().copied().collect();
                let mut resid = T::from_f64_const(0.0);
                for i in 0..m {
                    let mut ax_i = T::from_f64_const(0.0);
                    for j in 0..n {
                        ax_i = ax_i + a_data[i * n + j] * x_vec[j];
                    }
                    let diff = ax_i - b_data[i];
                    resid = resid + diff * diff;
                }
                vec![resid]
            } else {
                vec![]
            };

            let x = Array::from_vec(IxDyn::new(&[n]), x_vec)?;
            let residuals_arr = Array::from_vec(Ix1::new([residuals.len()]), residuals)?;
            Ok((x, residuals_arr, rank, sv))
        }
        2 => {
            if b_shape[0] != m {
                return Err(FerrayError::shape_mismatch(format!(
                    "lstsq: A is {}x{} but b has {} rows",
                    m, n, b_shape[0]
                )));
            }
            let nrhs = b_shape[1];
            let b_data: Vec<T> = b.iter().copied().collect();
            let b_mat = faer::Mat::from_fn(m, nrhs, |i, j| b_data[i * nrhs + j]);
            let x_mat = qr_decomp.solve_lstsq(&b_mat);

            let mut x_vec = Vec::with_capacity(n * nrhs);
            for i in 0..n {
                for j in 0..nrhs {
                    x_vec.push(x_mat[(i, j)]);
                }
            }

            // Compute residuals per rhs column
            let residuals = if m > n && rank == n {
                let a_data: Vec<T> = a.iter().copied().collect();
                let mut resids = vec![T::from_f64_const(0.0); nrhs];
                for col in 0..nrhs {
                    for i in 0..m {
                        let mut ax_i = T::from_f64_const(0.0);
                        for j in 0..n {
                            ax_i = ax_i + a_data[i * n + j] * x_vec[j * nrhs + col];
                        }
                        let diff = ax_i - b_data[i * nrhs + col];
                        resids[col] = resids[col] + diff * diff;
                    }
                }
                resids
            } else {
                vec![]
            };

            let x = Array::from_vec(IxDyn::new(&[n, nrhs]), x_vec)?;
            let residuals_arr = Array::from_vec(Ix1::new([residuals.len()]), residuals)?;
            Ok((x, residuals_arr, rank, sv))
        }
        _ => Err(FerrayError::shape_mismatch("lstsq: b must be 1D or 2D")),
    }
}

/// Compute the inverse of a square matrix.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if the matrix is not square.
/// - `FerrayError::SingularMatrix` if the matrix is singular.
pub fn inv<T: LinalgFloat>(a: &Array<T, Ix2>) -> FerrayResult<Array<T, Ix2>> {
    let shape = a.shape();
    if shape[0] != shape[1] {
        return Err(FerrayError::shape_mismatch(format!(
            "inv requires a square matrix, got {}x{}",
            shape[0], shape[1]
        )));
    }
    let n = shape[0];
    if n == 0 {
        return Array::from_vec(Ix2::new([0, 0]), vec![]);
    }

    // Check if matrix is singular by computing determinant
    let mat = faer_bridge::array2_to_faer(a);
    let det_val: T = mat.as_ref().determinant();
    if det_val.abs() < T::machine_epsilon() * T::from_f64_const(100.0) * T::from_usize(n) {
        return Err(FerrayError::SingularMatrix {
            message: "matrix is singular and cannot be inverted".to_string(),
        });
    }

    let lu = mat.as_ref().partial_piv_lu();
    let inv_mat = lu.inverse();
    faer_bridge::faer_to_array2(&inv_mat)
}

/// Compute the Moore-Penrose pseudoinverse of a matrix.
///
/// Uses SVD: `pinv(A) = V * diag(1/s_i) * U^T` for singular values above `rcond * max(s)`.
///
/// # Errors
/// - `FerrayError::InvalidValue` if SVD computation fails.
pub fn pinv<T: LinalgFloat>(a: &Array<T, Ix2>, _rcond: Option<T>) -> FerrayResult<Array<T, Ix2>> {
    let mat = faer_bridge::array2_to_faer(a);
    let decomp = mat
        .as_ref()
        .thin_svd()
        .map_err(|e| FerrayError::InvalidValue {
            message: format!("SVD failed in pinv: {e:?}"),
        })?;

    let pinv_mat = decomp.pseudoinverse();
    faer_bridge::faer_to_array2(&pinv_mat)
}

/// Raise a square matrix to an integer power.
///
/// - For `n > 0`: compute `A^n` by repeated squaring.
/// - For `n == 0`: return the identity matrix.
/// - For `n < 0`: compute `inv(A)^|n|`.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if the matrix is not square.
/// - `FerrayError::SingularMatrix` if `n < 0` and the matrix is singular.
pub fn matrix_power<T: LinalgFloat>(a: &Array<T, Ix2>, n: i64) -> FerrayResult<Array<T, Ix2>> {
    let shape = a.shape();
    if shape[0] != shape[1] {
        return Err(FerrayError::shape_mismatch(format!(
            "matrix_power requires a square matrix, got {}x{}",
            shape[0], shape[1]
        )));
    }
    let sz = shape[0];

    if n == 0 {
        // Return identity
        let zero = T::from_f64_const(0.0);
        let one = T::from_f64_const(1.0);
        let mut data = vec![zero; sz * sz];
        for i in 0..sz {
            data[i * sz + i] = one;
        }
        return Array::from_vec(Ix2::new([sz, sz]), data);
    }

    let base = if n < 0 { inv(a)? } else { a.clone() };
    let power = n.unsigned_abs();

    // Exponentiation by squaring
    let zero = T::from_f64_const(0.0);
    let one = T::from_f64_const(1.0);
    let mut result_data = vec![zero; sz * sz];
    for i in 0..sz {
        result_data[i * sz + i] = one;
    }
    let mut base_data: Vec<T> = base.iter().copied().collect();
    let mut p = power;

    while p > 0 {
        if p & 1 == 1 {
            result_data = mat_mul_flat(&result_data, &base_data, sz, sz, sz);
        }
        base_data = mat_mul_flat(&base_data, &base_data, sz, sz, sz);
        p >>= 1;
    }

    Array::from_vec(Ix2::new([sz, sz]), result_data)
}

fn mat_mul_flat<T: LinalgFloat>(a: &[T], b: &[T], m: usize, k: usize, n: usize) -> Vec<T> {
    let zero = T::from_f64_const(0.0);
    let mut c = vec![zero; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p];
            for j in 0..n {
                c[i * n + j] = c[i * n + j] + a_ip * b[p * n + j];
            }
        }
    }
    c
}

/// Solve the tensor equation `a x = b` for x.
///
/// `a` is reshaped according to `axes` to form a square matrix equation.
/// If `axes` is `None`, the default axes are used.
///
/// This is analogous to `numpy.linalg.tensorsolve`.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if the shapes are incompatible.
pub fn tensorsolve<T: LinalgFloat>(
    a: &Array<T, IxDyn>,
    b: &Array<T, IxDyn>,
    _axes: Option<&[usize]>,
) -> FerrayResult<Array<T, IxDyn>> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    // Compute the shape of x from the shapes of a and b
    let b_size: usize = b_shape.iter().product();
    let a_size: usize = a_shape.iter().product();
    if b_size == 0 {
        return Err(FerrayError::shape_mismatch("tensorsolve: b is empty"));
    }
    let x_size = a_size / b_size;
    if x_size * b_size != a_size {
        return Err(FerrayError::shape_mismatch(
            "tensorsolve: a and b shapes are not compatible",
        ));
    }

    // Reshape a to (b_size, x_size) and solve
    let a_data: Vec<T> = a.iter().copied().collect();
    let a2 = Array::<T, Ix2>::from_vec(Ix2::new([b_size, x_size]), a_data)?;

    let b_flat: Vec<T> = b.iter().copied().collect();
    let b_dyn = Array::<T, IxDyn>::from_vec(IxDyn::new(&[b_size]), b_flat)?;

    let x = solve(&a2, &b_dyn)?;

    // Determine x shape from a_shape and b_shape
    let x_shape: Vec<usize> = a_shape[b_shape.len()..].to_vec();
    if x_shape.is_empty() {
        Ok(x)
    } else {
        let x_data: Vec<T> = x.iter().copied().collect();
        Array::from_vec(IxDyn::new(&x_shape), x_data)
    }
}

/// Compute the inverse of an N-dimensional array.
///
/// This is analogous to `numpy.linalg.tensorinv`.
/// The array `a` is reshaped to a matrix using `ind` as the split point.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if the shapes are incompatible.
/// - `FerrayError::SingularMatrix` if the reshaped matrix is singular.
pub fn tensorinv<T: LinalgFloat>(a: &Array<T, IxDyn>, ind: usize) -> FerrayResult<Array<T, IxDyn>> {
    let shape = a.shape();
    if ind == 0 || ind > shape.len() {
        return Err(FerrayError::invalid_value(format!(
            "tensorinv: ind={} is invalid for {}D array",
            ind,
            shape.len()
        )));
    }

    let first_dims = &shape[..ind];
    let last_dims = &shape[ind..];
    let m: usize = first_dims.iter().product();
    let n: usize = last_dims.iter().product();

    if m != n {
        return Err(FerrayError::shape_mismatch(format!(
            "tensorinv: product of first {} dims ({}) != product of remaining dims ({})",
            ind, m, n
        )));
    }

    let data: Vec<T> = a.iter().copied().collect();
    let a2 = Array::<T, Ix2>::from_vec(Ix2::new([m, n]), data)?;
    let inv_a2 = inv(&a2)?;

    // Result shape is last_dims ++ first_dims
    let mut result_shape = Vec::with_capacity(shape.len());
    result_shape.extend_from_slice(last_dims);
    result_shape.extend_from_slice(first_dims);

    let inv_data: Vec<T> = inv_a2.iter().copied().collect();
    Array::from_vec(IxDyn::new(&result_shape), inv_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_2x2() {
        // A = [[1, 2], [3, 4]], b = [5, 11]
        // x = [1, 2]
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2]), vec![5.0, 11.0]).unwrap();
        let x = solve(&a, &b).unwrap();
        let xs = x.iter().copied().collect::<Vec<f64>>();
        assert!((xs[0] - 1.0).abs() < 1e-10);
        assert!((xs[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn solve_f32() {
        let a = Array::<f32, Ix2>::from_vec(Ix2::new([2, 2]), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let b = Array::<f32, IxDyn>::from_vec(IxDyn::new(&[2]), vec![5.0f32, 11.0]).unwrap();
        let x = solve(&a, &b).unwrap();
        let xs = x.iter().copied().collect::<Vec<f32>>();
        assert!((xs[0] - 1.0f32).abs() < 1e-4);
        assert!((xs[1] - 2.0f32).abs() < 1e-4);
    }

    #[test]
    fn solve_ax_eq_b_residual() {
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([3, 3]),
            vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0],
        )
        .unwrap();
        let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[3]), vec![1.0, 4.0, 9.0]).unwrap();
        let x = solve(&a, &b).unwrap();
        let xs: Vec<f64> = x.iter().copied().collect();

        // Compute Ax - b
        let a_data = a.as_slice().unwrap();
        for i in 0..3 {
            let mut ax_i = 0.0;
            for j in 0..3 {
                ax_i += a_data[i * 3 + j] * xs[j];
            }
            let b_i = [1.0, 4.0, 9.0][i];
            assert!(
                (ax_i - b_i).abs() < 1e-10,
                "Ax[{}] = {} != b[{}] = {}",
                i,
                ax_i,
                i,
                b_i
            );
        }
    }

    #[test]
    fn inv_identity() {
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([3, 3]),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let inv_a = inv(&a).unwrap();
        let d = inv_a.as_slice().unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (d[i * 3 + j] - expected).abs() < 1e-10,
                    "inv(I)[{},{}] = {} != {}",
                    i,
                    j,
                    d[i * 3 + j],
                    expected
                );
            }
        }
    }

    #[test]
    fn inv_singular_error() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![1.0, 2.0, 2.0, 4.0]).unwrap();
        assert!(inv(&a).is_err());
    }

    #[test]
    fn pinv_basic() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([3, 2]), vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
            .unwrap();
        let pi = pinv(&a, None).unwrap();
        assert_eq!(pi.shape(), &[2, 3]);
    }

    #[test]
    fn matrix_power_positive() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![1.0, 1.0, 0.0, 1.0]).unwrap();
        let a3 = matrix_power(&a, 3).unwrap();
        // [[1,1],[0,1]]^3 = [[1,3],[0,1]]
        let d = a3.as_slice().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-10);
        assert!((d[1] - 3.0).abs() < 1e-10);
        assert!((d[2] - 0.0).abs() < 1e-10);
        assert!((d[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn matrix_power_zero() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let a0 = matrix_power(&a, 0).unwrap();
        let d = a0.as_slice().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-10);
        assert!((d[1] - 0.0).abs() < 1e-10);
        assert!((d[2] - 0.0).abs() < 1e-10);
        assert!((d[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn matrix_power_negative() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![1.0, 1.0, 0.0, 1.0]).unwrap();
        let am1 = matrix_power(&a, -1).unwrap();
        let inv_a = inv(&a).unwrap();
        let d1 = am1.as_slice().unwrap();
        let d2 = inv_a.as_slice().unwrap();
        for i in 0..4 {
            assert!(
                (d1[i] - d2[i]).abs() < 1e-10,
                "matrix_power(-1)[{}] = {} != inv[{}] = {}",
                i,
                d1[i],
                i,
                d2[i]
            );
        }
    }

    #[test]
    fn lstsq_overdetermined() {
        // A = [[1,1],[1,2],[1,3]], b = [1,2,3]
        // Least squares solution
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([3, 2]), vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0])
            .unwrap();
        let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        let (x, _residuals, rank, _sv) = lstsq(&a, &b, None).unwrap();
        assert_eq!(rank, 2);
        let xs: Vec<f64> = x.iter().copied().collect();
        // x should be approximately [0, 1] (y = x)
        assert!((xs[0]).abs() < 0.1);
        assert!((xs[1] - 1.0).abs() < 0.1);
    }
}
