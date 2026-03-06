//! # ferrum-numpy-interop
//!
//! A companion crate providing zero-copy (where possible) conversions between
//! ferrum arrays and external array ecosystems:
//!
//! - **NumPy** (via PyO3) — feature `"python"`
//! - **Apache Arrow** — feature `"arrow"`
//! - **Polars** — feature `"polars"`
//!
//! All three backends are feature-gated and disabled by default. Enable them
//! in your `Cargo.toml`:
//!
//! ```toml
//! [dependencies.ferrum-numpy-interop]
//! version = "0.1"
//! features = ["arrow"]  # or "python", "polars"
//! ```
//!
//! ## Design principles
//!
//! 1. **Safety first** — every conversion validates dtypes and memory layout
//!    before returning. No silent reinterpretation of memory.
//! 2. **Zero-copy when possible** — C-contiguous arrays are shared without
//!    copying where the target format supports it.
//! 3. **Explicit errors** — dtype mismatches, null values, and unsupported
//!    types produce clear [`FerrumError`](ferrum_core::FerrumError) messages.

pub mod dtype_map;

#[cfg(feature = "python")]
pub mod numpy_conv;

#[cfg(feature = "arrow")]
pub mod arrow_conv;

#[cfg(feature = "polars")]
pub mod polars_conv;

// Re-export the main conversion traits at crate root for ergonomics.

#[cfg(feature = "arrow")]
pub use arrow_conv::{FromArrow, FromArrowBool, ToArrow, ToArrowBool};

#[cfg(feature = "polars")]
pub use polars_conv::{FromPolars, FromPolarsBool, ToPolars, ToPolarsBool};

#[cfg(feature = "python")]
pub use numpy_conv::{AsFerrum, IntoNumPy};
