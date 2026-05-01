// ferray-core: N-dimensional array type and foundational primitives
//
// This is the root crate for the ferray workspace. It provides NdArray<T, D>,
// the full ownership model (Array, ArrayView, ArrayViewMut, ArcArray, CowArray),
// the Element trait, DType system, FerrayError, and array introspection/iteration.
//
// ndarray is used internally but is NOT part of the public API.
//
// When the `no_std` feature is enabled, only the subset of functionality that
// does not depend on `std` or `ndarray` is compiled: DType, Element, FerrayError
// (simplified), dimension types (without ndarray conversions), constants, and
// layout enums. The full Array type and related features require `std`.

#![cfg_attr(not(feature = "std"), no_std)]
// Core numerical kernels lift integer shape/index data into floating-point
// (`linspace`, `arange`, `geomspace`), narrow `f64` boundaries back to
// `usize` indices (`pad`, `tile`, broadcasting), and expose `finfo`/`iinfo`
// limit constants that necessarily round-trip through cast-equivalent
// types. Constants comparison and oracle tests assert exact float equality
// against canonical values. These are part of the NumPy-compatible contract.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::float_cmp,
    // Every public function in the workspace returns FerrayResult<T>. The
    // FerrayError variants are documented once on the type, not on every
    // returning function — that is a workspace-wide doc convention.
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    // Numerical / array code uses i, j, k, n, m as canonical loop and
    // shape names — same as NumPy's reference docs and any linear algebra
    // text. Renaming for clippy's taste hurts readability.
    clippy::many_single_char_names,
    clippy::similar_names,
    // Items-after-statements covers local helper fns inside functions —
    // a deliberate ferray pattern for keeping helper closures named.
    clippy::items_after_statements,
    // The `if let Some(x) = .. else ..` pattern is more readable than
    // `Option::map_or_else` for non-trivial blocks; this is a
    // nursery-level stylistic preference.
    clippy::option_if_let_else,
    // Pedantic readability preferences that don't apply to this codebase.
    clippy::too_long_first_doc_paragraph,
    clippy::needless_pass_by_value,
    clippy::match_same_arms
)]

#[cfg(not(feature = "std"))]
extern crate alloc;

// Modules that work in both std and no_std modes
pub mod constants;
pub mod dimension;
pub mod dtype;
pub mod error;
pub mod layout;
pub mod record;

// Modules that require std (depend on ndarray or std-only features)
#[cfg(feature = "std")]
pub mod array;
#[cfg(feature = "std")]
pub mod buffer;
#[cfg(feature = "std")]
pub mod creation;
#[cfg(feature = "std")]
pub mod dynarray;
#[cfg(feature = "std")]
pub mod indexing;
#[cfg(feature = "std")]
pub mod manipulation;
#[cfg(feature = "std")]
pub mod nditer;
#[cfg(feature = "std")]
pub mod ops;
#[cfg(feature = "std")]
pub mod prelude;
#[cfg(feature = "std")]
pub mod writeback;

// Re-export key types at crate root for ergonomics (std only)
#[cfg(feature = "std")]
pub use array::ArrayFlags;
#[cfg(feature = "std")]
pub use array::aliases;
#[cfg(feature = "std")]
pub use array::arc::ArcArray;
#[cfg(feature = "std")]
pub use array::cow::CowArray;
#[cfg(feature = "std")]
pub use array::display::{get_print_options, set_print_options};
#[cfg(feature = "std")]
pub use array::owned::Array;
#[cfg(feature = "std")]
pub use array::view::ArrayView;
#[cfg(feature = "std")]
pub use array::view_mut::ArrayViewMut;

pub use dimension::{Axis, Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};

#[cfg(all(feature = "const_shapes", feature = "std"))]
pub use dimension::static_shape::{
    Assert, DefaultNdarrayDim, IsTrue, Shape1, Shape2, Shape3, Shape4, Shape5, Shape6,
    StaticBroadcast, StaticMatMul, StaticSize, static_reshape_array,
};

pub use dtype::{DType, Element, SliceInfoElem};

pub use error::{FerrayError, FerrayResult};

pub use layout::MemoryLayout;

#[cfg(feature = "std")]
pub use buffer::AsRawBuffer;

#[cfg(feature = "std")]
pub use dynarray::DynArray;

#[cfg(feature = "std")]
pub use nditer::NdIter;

pub use record::FieldDescriptor;

// Re-export proc macros from ferray-core-macros.
// The derive macro FerrayRecord shares its name with the trait in record::FerrayRecord.
// Both are re-exported: the derive macro lives in macro namespace, the trait in type namespace.
pub use ferray_core_macros::{FerrayRecord, promoted_type, s};
pub use record::FerrayRecord;

// Kani formal verification harnesses (only compiled during `cargo kani`)
#[cfg(kani)]
mod verification_kani;
