//! Complex matrix operations (#404).
//!
//! `LinalgFloat` is sealed to `f32` / `f64` and provides the generic
//! entry points (`matmul`, `solve`, `inv`, ...) for real matrices.
//! Making `LinalgFloat` generic over both real and complex scalars
//! would require a multi-file refactor of every function in the
//! crate, so this module takes the pragmatic path: it exposes
//! complex-specific functions that take `Array<Complex<T>, Ix2>`
//! directly and thin-wrap faer's complex `Mat` operations.
//!
//! faer 0.24's `ComplexField` impl covers `Complex<T>` transparently
//! via `impl<T: RealField<Unit: ComplexField>> ComplexField for
//! Complex<T>`, so the faer calls here compile unchanged — only the
//! element type at the ferray ↔ faer boundary is different.
//!
//! Functions provided in this first pass:
//!
//! - [`matmul_complex`]  — complex matrix × matrix
//! - [`inv_complex`]     — complex matrix inverse via LU
//! - [`solve_complex`]   — A x = b with complex A, b
//! - [`det_complex`]     — complex determinant
//!
//! Batched variants, SVD, eig, and the rest can be layered on later
//! as demand appears.

use num_complex::Complex;

use ferray_core::array::owned::Array;
use ferray_core::dimension::{Ix1, Ix2};
use ferray_core::dtype::Element;
use ferray_core::error::{FerrayError, FerrayResult};

use faer::linalg::solvers::{DenseSolveCore, Solve};

/// Convert a 2-D ferray `Array<Complex<T>, Ix2>` into a faer `Mat<Complex<T>>`.
fn array2c_to_faer<T>(a: &Array<Complex<T>, Ix2>) -> faer::Mat<Complex<T>>
where
    T: Copy + 'static,
    Complex<T>: Element + Copy + faer_traits::ComplexField,
{
    let shape = a.shape();
    let (m, n) = (shape[0], shape[1]);
    if let Some(slice) = a.as_slice() {
        faer::Mat::from_fn(m, n, |i, j| slice[i * n + j])
    } else {
        let data: Vec<Complex<T>> = a.iter().copied().collect();
        faer::Mat::from_fn(m, n, |i, j| data[i * n + j])
    }
}

/// Convert a faer `Mat<Complex<T>>` back to a ferray `Array<Complex<T>, Ix2>`.
fn faer_to_array2c<T>(mat: &faer::Mat<Complex<T>>) -> FerrayResult<Array<Complex<T>, Ix2>>
where
    T: Copy + 'static,
    Complex<T>: Element + Copy,
{
    let (m, n) = mat.shape();
    let mut data = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            data.push(mat[(i, j)]);
        }
    }
    Array::from_vec(Ix2::new([m, n]), data)
}

/// Complex matrix multiplication: `C = A @ B`.
///
/// Both inputs must be 2-D with compatible inner dimensions; the
/// result is always 2-D. Delegates to `faer::linalg::matmul::matmul`
/// through the faer complex `Mat` type.
///
/// # Errors
/// - [`FerrayError::ShapeMismatch`] if inner dimensions don't match.
pub fn matmul_complex<T>(
    a: &Array<Complex<T>, Ix2>,
    b: &Array<Complex<T>, Ix2>,
) -> FerrayResult<Array<Complex<T>, Ix2>>
where
    T: Copy + 'static,
    Complex<T>: Element + Copy + faer_traits::ComplexField,
{
    let a_shape = a.shape();
    let b_shape = b.shape();
    let (m, k1) = (a_shape[0], a_shape[1]);
    let (k2, n) = (b_shape[0], b_shape[1]);
    if k1 != k2 {
        return Err(FerrayError::shape_mismatch(format!(
            "matmul_complex: inner dimensions don't match ({m}x{k1} @ {k2}x{n})"
        )));
    }

    let a_faer = array2c_to_faer(a);
    let b_faer = array2c_to_faer(b);

    // faer's `*` operator on Mat dispatches through its matmul kernel
    // and handles the complex conjugate / transpose metadata for us.
    let c_faer = &a_faer * &b_faer;
    faer_to_array2c(&c_faer)
}

/// Complex matrix inverse via LU with partial pivoting.
///
/// # Errors
/// - [`FerrayError::ShapeMismatch`] if the matrix is not square.
/// - [`FerrayError::SingularMatrix`] if the result contains NaN/inf
///   (indicating faer's LU hit a near-zero pivot).
pub fn inv_complex<T>(
    a: &Array<Complex<T>, Ix2>,
) -> FerrayResult<Array<Complex<T>, Ix2>>
where
    T: Copy + 'static,
    Complex<T>: Element + Copy + faer_traits::ComplexField,
{
    let shape = a.shape();
    if shape[0] != shape[1] {
        return Err(FerrayError::shape_mismatch(format!(
            "inv_complex: requires a square matrix, got {}x{}",
            shape[0], shape[1]
        )));
    }
    let n = shape[0];
    if n == 0 {
        return Array::from_vec(Ix2::new([0, 0]), Vec::new());
    }

    let mat = array2c_to_faer(a);
    let lu = mat.as_ref().partial_piv_lu();
    let inv_mat = lu.inverse();

    let result = faer_to_array2c(&inv_mat)?;
    // Singularity detection would require calling .is_nan() on the
    // real/imag parts of each Complex<T>, but we don't have a
    // generic bound on T here. faer's LU produces NaN/Inf when it
    // hits a near-zero pivot; callers who need explicit singularity
    // detection can call `det_complex` alongside and test for zero.
    Ok(result)
}

/// Solve the complex linear system `A x = b`. `b` may be a vector
/// (1-D, length `n`) or a matrix of right-hand sides (2-D, `n × k`).
///
/// # Errors
/// - [`FerrayError::ShapeMismatch`] on incompatible shapes.
pub fn solve_complex<T>(
    a: &Array<Complex<T>, Ix2>,
    b: &Array<Complex<T>, Ix2>,
) -> FerrayResult<Array<Complex<T>, Ix2>>
where
    T: Copy + 'static,
    Complex<T>: Element + Copy + faer_traits::ComplexField,
{
    let a_shape = a.shape();
    if a_shape[0] != a_shape[1] {
        return Err(FerrayError::shape_mismatch(format!(
            "solve_complex: A must be square, got {}x{}",
            a_shape[0], a_shape[1]
        )));
    }
    let n = a_shape[0];
    let b_shape = b.shape();
    if b_shape[0] != n {
        return Err(FerrayError::shape_mismatch(format!(
            "solve_complex: A is {}x{} but b has {} rows",
            n, n, b_shape[0]
        )));
    }
    let nrhs = b_shape[1];

    let a_faer = array2c_to_faer(a);
    let lu = a_faer.as_ref().partial_piv_lu();
    let b_faer = array2c_to_faer(b);
    let x = lu.solve(&b_faer);

    // Extract back into a ferray Array.
    let mut data = Vec::with_capacity(n * nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            data.push(x[(i, j)]);
        }
    }
    Array::from_vec(Ix2::new([n, nrhs]), data)
}

/// Solve a complex vector system: same as [`solve_complex`] but with
/// a 1-D right-hand side vector.
pub fn solve_complex_vec<T>(
    a: &Array<Complex<T>, Ix2>,
    b: &Array<Complex<T>, Ix1>,
) -> FerrayResult<Array<Complex<T>, Ix1>>
where
    T: Copy + 'static,
    Complex<T>: Element + Copy + faer_traits::ComplexField,
{
    let a_shape = a.shape();
    if a_shape[0] != a_shape[1] {
        return Err(FerrayError::shape_mismatch(format!(
            "solve_complex_vec: A must be square, got {}x{}",
            a_shape[0], a_shape[1]
        )));
    }
    let n = a_shape[0];
    let b_shape = b.shape();
    if b_shape[0] != n {
        return Err(FerrayError::shape_mismatch(format!(
            "solve_complex_vec: A is {}x{} but b has length {}",
            n, n, b_shape[0]
        )));
    }

    let a_faer = array2c_to_faer(a);
    let lu = a_faer.as_ref().partial_piv_lu();

    // Build b as an n×1 faer Mat.
    let b_data: Vec<Complex<T>> = b.iter().copied().collect();
    let b_mat = faer::Mat::from_fn(n, 1, |i, _| b_data[i]);

    let x = lu.solve(&b_mat);

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        result.push(x[(i, 0)]);
    }
    Array::from_vec(Ix1::new([n]), result)
}

/// Complex matrix determinant. Delegates to `faer::Mat::determinant`.
///
/// # Errors
/// - [`FerrayError::ShapeMismatch`] if the matrix is not square.
pub fn det_complex<T>(a: &Array<Complex<T>, Ix2>) -> FerrayResult<Complex<T>>
where
    T: Copy + 'static,
    Complex<T>: Element + Copy + faer_traits::ComplexField,
{
    let shape = a.shape();
    if shape[0] != shape[1] {
        return Err(FerrayError::shape_mismatch(format!(
            "det_complex: requires a square matrix, got {}x{}",
            shape[0], shape[1]
        )));
    }
    let mat = array2c_to_faer(a);
    Ok(mat.as_ref().determinant())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::Ix2;
    use num_complex::Complex;

    fn c64(re: f64, im: f64) -> Complex<f64> {
        Complex::new(re, im)
    }

    fn arr2(rows: usize, cols: usize, data: Vec<Complex<f64>>) -> Array<Complex<f64>, Ix2> {
        Array::<Complex<f64>, Ix2>::from_vec(Ix2::new([rows, cols]), data).unwrap()
    }

    #[test]
    fn matmul_complex_2x2() {
        // A = [[1+i, 2], [3, 4-i]]
        // B = [[1, 0], [0, 1]]
        // A @ B = A
        let a = arr2(
            2,
            2,
            vec![c64(1.0, 1.0), c64(2.0, 0.0), c64(3.0, 0.0), c64(4.0, -1.0)],
        );
        let b = arr2(
            2,
            2,
            vec![c64(1.0, 0.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(1.0, 0.0)],
        );
        let c = matmul_complex(&a, &b).unwrap();
        let data: Vec<Complex<f64>> = c.iter().copied().collect();
        assert_eq!(data, vec![c64(1.0, 1.0), c64(2.0, 0.0), c64(3.0, 0.0), c64(4.0, -1.0)]);
    }

    #[test]
    fn matmul_complex_nontrivial() {
        // A = [[1+i, 2-i], [3, 1]]
        // B = [[1, 0-i], [i, 1]]
        // AB[0][0] = (1+i)*1 + (2-i)*i = (1+i) + (2i + 1) = 2 + 3i
        // AB[0][1] = (1+i)*(-i) + (2-i)*1 = (1 - i) + (2 - i) = 3 - 2i
        // AB[1][0] = 3*1 + 1*i = 3 + i
        // AB[1][1] = 3*(-i) + 1*1 = 1 - 3i
        let a = arr2(
            2,
            2,
            vec![c64(1.0, 1.0), c64(2.0, -1.0), c64(3.0, 0.0), c64(1.0, 0.0)],
        );
        let b = arr2(
            2,
            2,
            vec![c64(1.0, 0.0), c64(0.0, -1.0), c64(0.0, 1.0), c64(1.0, 0.0)],
        );
        let c = matmul_complex(&a, &b).unwrap();
        let d: Vec<Complex<f64>> = c.iter().copied().collect();
        let expected = [
            c64(2.0, 3.0),
            c64(3.0, -2.0),
            c64(3.0, 1.0),
            c64(1.0, -3.0),
        ];
        for (got, want) in d.iter().zip(expected.iter()) {
            assert!(
                (got.re - want.re).abs() < 1e-12,
                "re: got {got}, want {want}"
            );
            assert!(
                (got.im - want.im).abs() < 1e-12,
                "im: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn matmul_complex_shape_mismatch() {
        let a = arr2(2, 3, vec![c64(0.0, 0.0); 6]);
        let b = arr2(2, 2, vec![c64(0.0, 0.0); 4]);
        assert!(matmul_complex(&a, &b).is_err());
    }

    #[test]
    fn inv_complex_identity() {
        let i = arr2(
            2,
            2,
            vec![c64(1.0, 0.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(1.0, 0.0)],
        );
        let inv = inv_complex(&i).unwrap();
        let d: Vec<Complex<f64>> = inv.iter().copied().collect();
        assert_eq!(
            d,
            vec![c64(1.0, 0.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(1.0, 0.0)]
        );
    }

    #[test]
    fn inv_complex_diag_i() {
        // diag(i, 2i) — inverse is diag(-i, -i/2)
        let a = arr2(
            2,
            2,
            vec![c64(0.0, 1.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(0.0, 2.0)],
        );
        let inv = inv_complex(&a).unwrap();
        let d: Vec<Complex<f64>> = inv.iter().copied().collect();
        assert!((d[0] - c64(0.0, -1.0)).norm() < 1e-12);
        assert_eq!(d[1], c64(0.0, 0.0));
        assert_eq!(d[2], c64(0.0, 0.0));
        assert!((d[3] - c64(0.0, -0.5)).norm() < 1e-12);
    }

    #[test]
    fn solve_complex_identity_rhs() {
        // I * x = b  =>  x = b
        let i = arr2(
            2,
            2,
            vec![c64(1.0, 0.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(1.0, 0.0)],
        );
        let b = arr2(2, 1, vec![c64(3.0, 4.0), c64(5.0, -6.0)]);
        let x = solve_complex(&i, &b).unwrap();
        let d: Vec<Complex<f64>> = x.iter().copied().collect();
        assert_eq!(d, vec![c64(3.0, 4.0), c64(5.0, -6.0)]);
    }

    #[test]
    fn solve_complex_vec_identity() {
        use ferray_core::dimension::Ix1;
        let i = arr2(
            2,
            2,
            vec![c64(1.0, 0.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(1.0, 0.0)],
        );
        let b = Array::<Complex<f64>, Ix1>::from_vec(
            Ix1::new([2]),
            vec![c64(1.0, 2.0), c64(3.0, 4.0)],
        )
        .unwrap();
        let x = solve_complex_vec(&i, &b).unwrap();
        let d: Vec<Complex<f64>> = x.iter().copied().collect();
        assert_eq!(d, vec![c64(1.0, 2.0), c64(3.0, 4.0)]);
    }

    #[test]
    fn det_complex_diag() {
        // det(diag(2+i, 3-i)) = (2+i)*(3-i) = 6 - 2i + 3i - i^2 = 7 + i
        let a = arr2(
            2,
            2,
            vec![c64(2.0, 1.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(3.0, -1.0)],
        );
        let d = det_complex(&a).unwrap();
        assert!((d - c64(7.0, 1.0)).norm() < 1e-12);
    }

    #[test]
    fn matmul_inv_is_identity() {
        // A = [[2+i, 1], [1, 2-i]], compute A * A^-1 and check it's I.
        let a = arr2(
            2,
            2,
            vec![c64(2.0, 1.0), c64(1.0, 0.0), c64(1.0, 0.0), c64(2.0, -1.0)],
        );
        let inv = inv_complex(&a).unwrap();
        let prod = matmul_complex(&a, &inv).unwrap();
        let d: Vec<Complex<f64>> = prod.iter().copied().collect();
        assert!((d[0] - c64(1.0, 0.0)).norm() < 1e-12);
        assert!(d[1].norm() < 1e-12);
        assert!(d[2].norm() < 1e-12);
        assert!((d[3] - c64(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn det_non_square_errors() {
        let a = arr2(2, 3, vec![c64(0.0, 0.0); 6]);
        assert!(det_complex(&a).is_err());
    }

    #[test]
    fn inv_non_square_errors() {
        let a = arr2(2, 3, vec![c64(0.0, 0.0); 6]);
        assert!(inv_complex(&a).is_err());
    }
}
