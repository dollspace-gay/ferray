//! # ferray-numpy-interop
//!
//! A companion crate providing owning conversions between ferray arrays and
//! external array ecosystems:
//!
//! - **`NumPy`** (via `PyO3`) — feature `"python"`
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
//! practice all six conversion paths (`NumPy` / Arrow / Polars × both
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

// Interop kernels marshal byte buffers across `numpy`/`arrow`/`polars`
// type systems and routinely cross integer-width and signed/unsigned
// boundaries that those external types contractually represent.
// Workspace convention is to document FerrayError variants on the type.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::items_after_statements,
    clippy::option_if_let_else,
    clippy::too_long_first_doc_paragraph,
    clippy::needless_pass_by_value,
    clippy::match_same_arms
)]

pub mod dtype_map;

#[cfg(feature = "python")]
pub mod numpy_conv;

#[cfg(feature = "arrow")]
pub mod arrow_conv;

#[cfg(feature = "polars")]
pub mod polars_conv;

// Re-export the main conversion traits at crate root for ergonomics.

#[cfg(feature = "arrow")]
pub use arrow_conv::{
    FromArrow, FromArrowBool, ToArrow, ToArrowBool, array2_from_arrow_columns,
    array2_to_arrow_columns, arrayd_from_arrow_flat, arrayd_to_arrow_flat,
};

#[cfg(feature = "polars")]
pub use polars_conv::{
    FromPolars, FromPolarsBool, ToPolars, ToPolarsBool, array2_from_polars_dataframe,
    array2_to_polars_dataframe,
};

#[cfg(feature = "python")]
pub use numpy_conv::{AsFerray, IntoNumPy};
