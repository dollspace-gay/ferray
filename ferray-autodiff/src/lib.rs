//! Forward-mode automatic differentiation for ferray.
//!
//! This crate provides a [`DualNumber`] type that tracks derivatives through
//! computations using the dual number algebra: `(a + b*eps)` where `eps^2 = 0`.
//!
//! # Quick Start
//!
//! ```
//! use ferray_autodiff::{DualNumber, derivative};
//!
//! // Compute d/dx sin(x) at x = 0
//! let d = derivative(|x| x.sin(), 0.0_f64);
//! assert!((d - 1.0).abs() < 1e-10);
//! ```
//!
//! # Crate layout
//!
//! The implementation is split across several modules (previously all
//! in a single 2280-line lib.rs — see issue #535):
//!
//! - [`dual`]: the [`DualNumber`] struct, constructors, `Display`,
//!   arithmetic operators (dual-dual and dual-scalar), and `PartialOrd`.
//! - [`functions`]: differentiable methods on `DualNumber` (`sin`,
//!   `exp`, `powf`, …) and flat free-function mirrors.
//! - [`num_impls`]: `num_traits` impls (`Zero`, `One`, `ToPrimitive`,
//!   `FromPrimitive`, `NumCast`, `Num`, and the big `Float` trait).
//! - [`api`]: [`derivative`], [`gradient`], and the slice-based
//!   [`jacobian`] convenience API.
//! - [`array_ops`]: Array-aware forward-mode AD — derivative /
//!   gradient / jacobian over `ferray_core::Array` inputs (issue #534).
//!
//! The public API is re-exported at the crate root for ergonomic use.

pub mod api;
pub mod array_ops;
pub mod dual;
pub mod functions;
pub mod num_impls;

pub use api::{derivative, gradient, jacobian};
pub use array_ops::{
    derivative_elementwise, gradient_vector, jacobian_array, value_and_derivative_elementwise,
    value_and_gradient,
};
pub use dual::DualNumber;
pub use functions::{
    abs, acos, asin, atan, atan2, cos, cosh, exp, ln, log10, log2, sin, sinh, sqrt, tan, tanh,
};

// Existing test suite (unit, finite-difference, property) lives in a
// dedicated file so that lib.rs stays a pure module-assembly point.
#[cfg(test)]
mod tests;
