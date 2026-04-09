//! Array-aware forward-mode AD (#534).
//!
//! `DualNumber<f64>` can't be stored in `ferray_core::Array<T, D>` directly
//! because `Element` is sealed inside `ferray-core` and a downstream impl
//! would require a dependency cycle (ferray-core → ferray-autodiff →
//! ferray-core). Instead we expose *Array-aware* derivative/gradient/
//! jacobian functions that take plain `Array<f64, D>` (or `f32`) inputs,
//! internally seed each element as a `DualNumber`, run forward-mode AD on
//! the closure, and reproject the dual parts back into a ferray array.
//!
//! This is the option-2 path from the issue discussion: keep the public
//! ferray array ecosystem working with real scalars, but provide the
//! differentiation primitives that ML workloads actually reach for
//! (elementwise derivative, full gradient of a scalar-valued function,
//! and the Jacobian of a vector-valued function).

use ferray_core::array::owned::Array;
use ferray_core::dimension::{Dimension, Ix1, Ix2};
use ferray_core::dtype::Element;
use ferray_core::error::{FerrayError, FerrayResult};
use num_traits::Float;

use crate::DualNumber;

// ---------------------------------------------------------------------------
// Elementwise derivative
// ---------------------------------------------------------------------------

/// Elementwise derivative of a scalar-valued function `f` applied to an
/// array. For each element `x` in the input, returns `f'(x)`.
///
/// Equivalent to seeding each element as a `DualNumber::variable`,
/// evaluating `f`, and reading back the dual part. The output shape
/// matches the input. Works for any contiguous-layout ferray array.
///
/// # Examples
///
/// ```
/// use ferray_autodiff::array_ops::derivative_elementwise;
/// use ferray_core::{Array, Ix1};
///
/// let x = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![0.0, 1.0, 2.0]).unwrap();
/// let dx = derivative_elementwise(|v| v * v, &x).unwrap();
/// // d/dx (x^2) = 2x
/// assert_eq!(dx.as_slice().unwrap(), &[0.0, 2.0, 4.0]);
/// ```
///
/// # Errors
/// Returns [`FerrayError::InvalidValue`] if the input array is not
/// C-contiguous (since we iterate by flat index into the output buffer).
pub fn derivative_elementwise<T, D, F>(f: F, x: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
    F: Fn(DualNumber<T>) -> DualNumber<T>,
{
    let x_slice = x.as_slice().ok_or_else(|| {
        FerrayError::invalid_value(
            "derivative_elementwise: input must be C-contiguous; call as_standard_layout first",
        )
    })?;
    let out: Vec<T> = x_slice
        .iter()
        .map(|&xi| f(DualNumber::variable(xi)).dual)
        .collect();
    Array::from_vec(x.dim().clone(), out)
}

/// Elementwise function value and derivative in one pass.
///
/// Returns `(values, derivatives)` as two arrays matching the input shape.
/// Cheaper than calling `derivative_elementwise` and the function
/// separately because the dual evaluation computes both halves at once.
///
/// # Errors
/// Same as [`derivative_elementwise`].
pub fn value_and_derivative_elementwise<T, D, F>(
    f: F,
    x: &Array<T, D>,
) -> FerrayResult<(Array<T, D>, Array<T, D>)>
where
    T: Element + Float,
    D: Dimension,
    F: Fn(DualNumber<T>) -> DualNumber<T>,
{
    let x_slice = x.as_slice().ok_or_else(|| {
        FerrayError::invalid_value("value_and_derivative_elementwise: input must be C-contiguous")
    })?;
    let n = x_slice.len();
    let mut values = Vec::with_capacity(n);
    let mut derivs = Vec::with_capacity(n);
    for &xi in x_slice {
        let fx = f(DualNumber::variable(xi));
        values.push(fx.real);
        derivs.push(fx.dual);
    }
    Ok((
        Array::from_vec(x.dim().clone(), values)?,
        Array::from_vec(x.dim().clone(), derivs)?,
    ))
}

// ---------------------------------------------------------------------------
// Gradient (full gradient of a scalar function of a vector)
// ---------------------------------------------------------------------------

/// Full gradient of a scalar-valued function `f: R^n → R` evaluated at
/// the vector `x`.
///
/// Forward-mode AD computes one partial derivative per pass by seeding
/// exactly one element as a `DualNumber::variable` and the rest as
/// `DualNumber::constant`. For an `n`-element input this means `n`
/// forward evaluations — `O(n)` work, which is efficient when `n` is
/// small. For large `n` prefer reverse-mode (not implemented yet; use
/// ferrotorch once it ships).
///
/// # Errors
/// Returns [`FerrayError::InvalidValue`] if `x` is not contiguous.
///
/// # Examples
///
/// ```
/// use ferray_autodiff::array_ops::gradient_vector;
/// use ferray_autodiff::DualNumber;
/// use ferray_core::{Array, Ix1};
///
/// // f(x) = x[0]^2 + 3*x[1] + x[2]^3
/// // ∇f(x) = [2*x[0], 3, 3*x[2]^2]
/// // At x = [1, 2, 4]: ∇f = [2, 3, 48]
/// let x = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 4.0]).unwrap();
/// let grad = gradient_vector(
///     |xs: &[DualNumber<f64>]| xs[0] * xs[0] + xs[1] * 3.0 + xs[2] * xs[2] * xs[2],
///     &x,
/// )
/// .unwrap();
/// let g = grad.as_slice().unwrap();
/// assert!((g[0] - 2.0).abs() < 1e-10);
/// assert!((g[1] - 3.0).abs() < 1e-10);
/// assert!((g[2] - 48.0).abs() < 1e-10);
/// ```
pub fn gradient_vector<T, F>(f: F, x: &Array<T, Ix1>) -> FerrayResult<Array<T, Ix1>>
where
    T: Element + Float,
    F: Fn(&[DualNumber<T>]) -> DualNumber<T>,
{
    let x_slice = x
        .as_slice()
        .ok_or_else(|| FerrayError::invalid_value("gradient_vector: input must be C-contiguous"))?;
    let n = x_slice.len();
    let mut grad = Vec::with_capacity(n);

    // Scratch buffer for the seed vector; reused across iterations to
    // avoid per-pass allocation.
    let mut seed: Vec<DualNumber<T>> = x_slice.iter().map(|&v| DualNumber::constant(v)).collect();

    for i in 0..n {
        // Seed only element i as the active variable.
        seed[i] = DualNumber::variable(x_slice[i]);
        let fx = f(&seed);
        grad.push(fx.dual);
        // Reset for the next iteration.
        seed[i] = DualNumber::constant(x_slice[i]);
    }

    Array::from_vec(Ix1::new([n]), grad)
}

/// Scalar value and gradient of `f: R^n → R` in one call. The value is
/// only computed during the first partial-derivative pass (it's
/// identical across all passes since only the dual parts differ).
pub fn value_and_gradient<T, F>(f: F, x: &Array<T, Ix1>) -> FerrayResult<(T, Array<T, Ix1>)>
where
    T: Element + Float,
    F: Fn(&[DualNumber<T>]) -> DualNumber<T>,
{
    let x_slice = x.as_slice().ok_or_else(|| {
        FerrayError::invalid_value("value_and_gradient: input must be C-contiguous")
    })?;
    let n = x_slice.len();
    let mut grad = Vec::with_capacity(n);
    let mut seed: Vec<DualNumber<T>> = x_slice.iter().map(|&v| DualNumber::constant(v)).collect();

    let mut value = <T as num_traits::Zero>::zero();
    for i in 0..n {
        seed[i] = DualNumber::variable(x_slice[i]);
        let fx = f(&seed);
        if i == 0 {
            value = fx.real;
        }
        grad.push(fx.dual);
        seed[i] = DualNumber::constant(x_slice[i]);
    }

    let g = Array::from_vec(Ix1::new([n]), grad)?;
    Ok((value, g))
}

// ---------------------------------------------------------------------------
// Jacobian (of a vector-valued function R^n → R^m)
// ---------------------------------------------------------------------------

/// Jacobian of `f: R^n → R^m` evaluated at `x`. Returns the m×n matrix
/// `J[i][j] = ∂f_i/∂x_j`.
///
/// Forward-mode AD computes one column of the Jacobian per pass, so
/// total work is `n` evaluations — efficient when `n ≤ m`. For tall
/// Jacobians (`m ≪ n`) prefer reverse-mode once it's available.
///
/// `f` takes a slice of `DualNumber<T>` inputs and returns a `Vec` of
/// `DualNumber<T>` outputs (one per output dimension).
///
/// # Errors
/// Returns [`FerrayError::InvalidValue`] if the input is not
/// contiguous, or [`FerrayError::ShapeMismatch`] if `f` returns a
/// different number of outputs across calls.
pub fn jacobian_array<T, F>(f: F, x: &Array<T, Ix1>) -> FerrayResult<Array<T, Ix2>>
where
    T: Element + Float,
    F: Fn(&[DualNumber<T>]) -> Vec<DualNumber<T>>,
{
    let x_slice = x
        .as_slice()
        .ok_or_else(|| FerrayError::invalid_value("jacobian: input must be C-contiguous"))?;
    let n = x_slice.len();

    let mut seed: Vec<DualNumber<T>> = x_slice.iter().map(|&v| DualNumber::constant(v)).collect();

    // Pass 0: seed element 0, evaluate. That tells us `m` and gives us
    // the first column of the Jacobian.
    seed[0] = DualNumber::variable(x_slice[0]);
    let first = f(&seed);
    let m = first.len();
    seed[0] = DualNumber::constant(x_slice[0]);

    // Jacobian stored row-major as an (m, n) matrix.
    let mut jac = vec![<T as num_traits::Zero>::zero(); m * n];
    for (i, d) in first.iter().enumerate() {
        jac[i * n] = d.dual;
    }

    // Remaining n-1 columns.
    for j in 1..n {
        seed[j] = DualNumber::variable(x_slice[j]);
        let col = f(&seed);
        if col.len() != m {
            return Err(FerrayError::shape_mismatch(format!(
                "jacobian: f returned {} outputs at j={j} but {} at j=0",
                col.len(),
                m
            )));
        }
        for (i, d) in col.iter().enumerate() {
            jac[i * n + j] = d.dual;
        }
        seed[j] = DualNumber::constant(x_slice[j]);
    }

    Array::from_vec(Ix2::new([m, n]), jac)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::Ix2;

    // ---- derivative_elementwise ----

    #[test]
    fn elementwise_square_derivative() {
        let x = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        // d/dx (x^2) = 2x
        let dx = derivative_elementwise(|v| v * v, &x).unwrap();
        assert_eq!(dx.as_slice().unwrap(), &[0.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn elementwise_sin_derivative() {
        use std::f64::consts::PI;
        let x = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![0.0, PI / 2.0, PI]).unwrap();
        // d/dx sin(x) = cos(x)  →  [1, 0, -1]
        let dx = derivative_elementwise(|v| v.sin(), &x).unwrap();
        let slice = dx.as_slice().unwrap();
        assert!((slice[0] - 1.0).abs() < 1e-10);
        assert!(slice[1].abs() < 1e-10);
        assert!((slice[2] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn elementwise_2d_shape_preserved() {
        let x = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        // d/dx (x^3) = 3*x^2
        let dx = derivative_elementwise(|v| v * v * v, &x).unwrap();
        assert_eq!(dx.shape(), &[2, 3]);
        let expected = [3.0, 12.0, 27.0, 48.0, 75.0, 108.0];
        for (a, e) in dx.as_slice().unwrap().iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-10);
        }
    }

    #[test]
    fn value_and_derivative_both() {
        let x = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        // f(x) = x^2; values = [1,4,9], derivatives = [2,4,6]
        let (values, derivs) = value_and_derivative_elementwise(|v| v * v, &x).unwrap();
        assert_eq!(values.as_slice().unwrap(), &[1.0, 4.0, 9.0]);
        assert_eq!(derivs.as_slice().unwrap(), &[2.0, 4.0, 6.0]);
    }

    // ---- gradient_vector ----

    #[test]
    fn gradient_quadratic_form() {
        // f(x) = x[0]^2 + 2*x[1]^2 + 3*x[2]^2
        // ∇f(x) = [2*x[0], 4*x[1], 6*x[2]]
        let x = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let grad = gradient_vector(
            |xs: &[DualNumber<f64>]| xs[0] * xs[0] + xs[1] * xs[1] * 2.0 + xs[2] * xs[2] * 3.0,
            &x,
        )
        .unwrap();
        let g = grad.as_slice().unwrap();
        assert!((g[0] - 2.0).abs() < 1e-10);
        assert!((g[1] - 8.0).abs() < 1e-10);
        assert!((g[2] - 18.0).abs() < 1e-10);
    }

    #[test]
    fn value_and_gradient_both() {
        // f(x) = x[0] + x[1]*x[2]
        // f(1, 2, 3) = 1 + 2*3 = 7
        // ∇f = [1, x[2], x[1]] = [1, 3, 2]
        let x = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let (value, grad) =
            value_and_gradient(|xs: &[DualNumber<f64>]| xs[0] + xs[1] * xs[2], &x).unwrap();
        assert!((value - 7.0).abs() < 1e-10);
        let g = grad.as_slice().unwrap();
        assert!((g[0] - 1.0).abs() < 1e-10);
        assert!((g[1] - 3.0).abs() < 1e-10);
        assert!((g[2] - 2.0).abs() < 1e-10);
    }

    // ---- jacobian ----

    #[test]
    fn jacobian_linear_map() {
        // f(x) = [x[0]+x[1], x[0]-x[1]]
        // J = [[1, 1], [1, -1]]
        let x = Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![3.0, 4.0]).unwrap();
        let j = jacobian_array(
            |xs: &[DualNumber<f64>]| vec![xs[0] + xs[1], xs[0] - xs[1]],
            &x,
        )
        .unwrap();
        assert_eq!(j.shape(), &[2, 2]);
        let j_slice = j.as_slice().unwrap();
        assert_eq!(j_slice, &[1.0, 1.0, 1.0, -1.0]);
    }

    #[test]
    fn jacobian_polar_to_cartesian() {
        // f(r, theta) = [r*cos(theta), r*sin(theta)]
        // J = [[cos(theta), -r*sin(theta)], [sin(theta), r*cos(theta)]]
        use std::f64::consts::FRAC_PI_4;
        let r = 2.0;
        let theta = FRAC_PI_4;
        let x = Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![r, theta]).unwrap();
        let j = jacobian_array(
            |xs: &[DualNumber<f64>]| {
                let r = xs[0];
                let t = xs[1];
                vec![r * t.cos(), r * t.sin()]
            },
            &x,
        )
        .unwrap();
        assert_eq!(j.shape(), &[2, 2]);
        let j_slice = j.as_slice().unwrap();
        let ct = theta.cos();
        let st = theta.sin();
        assert!((j_slice[0] - ct).abs() < 1e-10); // ∂(r cos θ)/∂r = cos θ
        assert!((j_slice[1] - (-r * st)).abs() < 1e-10); // ∂(r cos θ)/∂θ = -r sin θ
        assert!((j_slice[2] - st).abs() < 1e-10); // ∂(r sin θ)/∂r = sin θ
        assert!((j_slice[3] - (r * ct)).abs() < 1e-10); // ∂(r sin θ)/∂θ = r cos θ
    }

    #[test]
    fn jacobian_rejects_inconsistent_output_dim() {
        // f returns different-length vectors for different inputs —
        // should be rejected on the second pass.
        let x = Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![1.0, 2.0]).unwrap();
        let result = jacobian_array(
            |xs: &[DualNumber<f64>]| {
                if xs[0].dual > 0.0 {
                    vec![xs[0], xs[1]]
                } else {
                    vec![xs[0] + xs[1]]
                }
            },
            &x,
        );
        assert!(result.is_err());
    }

    // ---- f32 coverage ----

    #[test]
    fn elementwise_f32() {
        let x = Array::<f32, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let dx = derivative_elementwise(|v| v * v, &x).unwrap();
        assert_eq!(dx.as_slice().unwrap(), &[2.0, 4.0, 6.0]);
    }
}
