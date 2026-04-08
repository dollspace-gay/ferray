//! # ferray-numpy-interop
//!
//! A companion crate providing owning conversions between ferray arrays and
//! external array ecosystems:
//!
//! - **NumPy** (via PyO3) — feature `"python"`
//! - **Apache Arrow** — feature `"arrow"`
//! - **Polars** — feature `"polars"`
//!
//! All three backends are feature-gated and disabled by default. Enable them
//! in your `Cargo.toml`:
//!
//! ```toml
//! [dependencies.ferray-numpy-interop]
//! version = "0.1"
//! features = ["arrow"]  # or "python", "polars"
//! ```
//!
//! ## Memory semantics
//!
//! Every conversion in this crate currently **copies** the data buffer.
//! The previous documentation claimed "zero-copy where possible", but in
//! practice all six conversion paths (NumPy / Arrow / Polars × both
//! directions) allocate a new buffer and memcpy the elements:
//!
//! | Path                     | Reason                                       |
//! |--------------------------|----------------------------------------------|
//! | `NumPy → ferray`         | `PyReadonlyArray::iter().cloned().collect()` |
//! | `ferray → NumPy`         | `Array::to_vec_flat()` then `from_vec`       |
//! | `Arrow ↔ ferray`         | `PrimitiveArray::values()` cloned into `Vec` |
//! | `Polars ↔ ferray`        | `ChunkedArray` → `Vec<T>` via per-chunk copy |
//!
//! True zero-copy ferray↔NumPy would require ferray arrays to share the
//! raw buffer with a Python-owned `PyArray` (refcount handshake plus
//! pinning), which is a significant design change. Zero-copy to Arrow
//! would require ferray arrays to expose their backing buffer as an
//! `arrow::buffer::Buffer` with a compatible `Drop` hook. Both are
//! tracked as potential follow-ups; for now the crate provides a
//! correct, allocation-aware API that clearly acknowledges the copy.
//!
//! The copies are usually still *cheap enough* for interop boundaries —
//! they are a single `memcpy` per conversion, not per element — but
//! callers on hot paths should prefer to stay inside one ecosystem.
//!
//! ## Design principles
//!
//! 1. **Safety first** — every conversion validates dtypes and memory
//!    layout before returning. No silent reinterpretation of memory.
//! 2. **Honest about allocation** — see the table above. The docstrings
//!    on individual functions say "copy" explicitly.
//! 3. **Explicit errors** — dtype mismatches, null values, and
//!    unsupported types produce clear
//!    [`FerrayError`](ferray_core::FerrayError) messages.

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
pub use numpy_conv::{AsFerray, IntoNumPy};
