//! Convenience entry points for forward-mode AD: `derivative` for
//! univariate functions, `gradient` for scalar-valued functions of a
//! vector, and `jacobian` for vector-valued functions.
//!
//! These take plain Rust slices/scalars. For ferray-array-aware
//! versions (`derivative_elementwise`, `gradient_vector`,
//! `jacobian_array`) see `array_ops.rs`.

use num_traits::Float;

use crate::dual::DualNumber;

/// Compute the derivative of a univariate function at a point.
///
/// Seeds the input as a variable (dual part = 1), evaluates the function,
/// and extracts the dual part of the result, which is the derivative.
///
/// # Examples
///
/// ```
/// use ferray_autodiff::derivative;
///
/// // d/dx sin(x) at x=0 is cos(0) = 1
/// let d = derivative(|x| x.sin(), 0.0_f64);
/// assert!((d - 1.0).abs() < 1e-10);
///
/// // d/dx x^3 at x=2 is 3*4 = 12
/// let d = derivative(|x| x.powi(3), 2.0_f64);
/// assert!((d - 12.0).abs() < 1e-10);
/// ```
pub fn derivative<T: Float>(f: impl Fn(DualNumber<T>) -> DualNumber<T>, x: T) -> T {
    f(DualNumber::variable(x)).dual
}

/// Compute the gradient of a multivariate function at a point.
///
/// For each component of the input, seeds that component as a variable
/// (dual = 1) while keeping all others as constants (dual = 0), then
/// evaluates the function and collects the partial derivatives.
///
/// # Examples
///
/// ```
/// use ferray_autodiff::{gradient, DualNumber};
///
/// // f(x, y) = x^2 + y^2
/// // grad f = (2x, 2y)
/// let g = gradient(
///     |v| v[0] * v[0] + v[1] * v[1],
///     &[3.0_f64, 4.0],
/// );
/// assert!((g[0] - 6.0).abs() < 1e-10);
/// assert!((g[1] - 8.0).abs() < 1e-10);
/// ```
pub fn gradient<T: Float>(f: impl Fn(&[DualNumber<T>]) -> DualNumber<T>, point: &[T]) -> Vec<T> {
    let n = point.len();
    let mut grad = Vec::with_capacity(n);

    // Pre-allocate a single buffer of constants; toggle one variable at a time.
    let mut args: Vec<DualNumber<T>> = point.iter().map(|&val| DualNumber::constant(val)).collect();

    for i in 0..n {
        // Seed the i-th component as the variable
        args[i].dual = T::one();
        grad.push(f(&args).dual);
        // Reset back to constant
        args[i].dual = T::zero();
    }
    grad
}

/// Compute the Jacobian matrix of a vector-valued function at a point.
///
/// For `f: R^n -> R^m`, returns an `m x n` matrix (as a flat `Vec<T>` in
/// row-major order) where `J[i*n + j] = df_i/dx_j`.
///
/// Uses forward-mode AD: each evaluation with the j-th input seeded
/// produces column j of the Jacobian (all m partial derivatives with
/// respect to x_j).
///
/// # Examples
///
/// ```
/// use ferray_autodiff::{jacobian, DualNumber};
///
/// // f(x, y) = (x*y, x + y)
/// // J = [[y, x], [1, 1]]
/// let (jac, m) = jacobian(
///     |v| vec![v[0] * v[1], v[0] + v[1]],
///     &[3.0_f64, 4.0],
/// );
/// assert_eq!(m, 2); // 2 outputs
/// assert!((jac[0] - 4.0).abs() < 1e-10); // df0/dx = y = 4
/// assert!((jac[1] - 3.0).abs() < 1e-10); // df0/dy = x = 3
/// assert!((jac[2] - 1.0).abs() < 1e-10); // df1/dx = 1
/// assert!((jac[3] - 1.0).abs() < 1e-10); // df1/dy = 1
/// ```
pub fn jacobian<T: Float>(
    f: impl Fn(&[DualNumber<T>]) -> Vec<DualNumber<T>>,
    point: &[T],
) -> (Vec<T>, usize) {
    let n = point.len();
    let mut args: Vec<DualNumber<T>> = point.iter().map(|&val| DualNumber::constant(val)).collect();

    // First evaluation to determine output dimension m
    let first_out = f(&args);
    let m = first_out.len();

    // Jacobian stored row-major: J[i*n + j] = df_i/dx_j
    let mut jac = vec![T::zero(); m * n];

    for j in 0..n {
        args[j].dual = T::one();
        let out = f(&args);
        for i in 0..m {
            jac[i * n + j] = out[i].dual;
        }
        args[j].dual = T::zero();
    }

    (jac, m)
}
