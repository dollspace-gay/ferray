// ferray-polynomial: Complete numpy.polynomial implementation
//
// Provides polynomial operations in multiple bases:
// - Power (monomial) basis
// - Chebyshev (first kind)
// - Legendre
// - Laguerre
// - Hermite (physicist's)
// - HermiteE (probabilist's)
//
// All polynomial types implement the `Poly` trait for evaluation,
// differentiation, integration, root-finding, arithmetic, and fitting.
// Basis conversion uses power basis as a canonical pivot.

//! # ferray-polynomial
//!
//! Complete `numpy.polynomial` implementation for the ferray ecosystem.
//!
//! Provides polynomial operations in six bases: power, Chebyshev, Legendre,
//! Laguerre, Hermite (physicist's), and `HermiteE` (probabilist's).
//!
//! All polynomial types implement the [`Poly`] trait for evaluation,
//! differentiation, integration, root-finding, arithmetic, and least-squares fitting.
//!
//! Basis conversion uses power basis as a canonical pivot via the
//! [`ConvertBasis`] trait.

#![deny(unsafe_code)]
// Polynomial evaluation, fitting, and basis conversion span integer
// degrees/indices and floating-point coefficients; the underlying linear
// algebra (companion-matrix eig, Vandermonde solve) further requires
// `usize -> f64` lifts. Domain/window equality and trim-tolerance checks
// rely on exact float comparison by design.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::float_cmp,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::items_after_statements,
    clippy::option_if_let_else,
    clippy::too_long_first_doc_paragraph,
    clippy::needless_pass_by_value,
    clippy::match_same_arms,
    clippy::suboptimal_flops,
    clippy::while_float
)]

pub mod chebyshev;
pub mod companion;
pub mod fitting;
pub mod hermite;
pub mod hermite_e;
pub mod laguerre;
pub mod legendre;
pub mod mapping;
pub mod power;
pub mod roots;
pub mod traits;

// Re-export key types at crate root for ergonomics
pub use chebyshev::Chebyshev;
pub use hermite::Hermite;
pub use hermite_e::HermiteE;
pub use laguerre::Laguerre;
pub use legendre::Legendre;
pub use power::Polynomial;
pub use traits::{ConvertBasis, FromPowerBasis, Poly, ToPowerBasis};
