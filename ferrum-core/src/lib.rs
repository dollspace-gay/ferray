// ferrum-core: N-dimensional array type and foundational primitives
//
// This is the root crate for the ferrum workspace. It provides NdArray<T, D>,
// the full ownership model (Array, ArrayView, ArrayViewMut, ArcArray, CowArray),
// the Element trait, DType system, FerrumError, and array introspection/iteration.
//
// ndarray is used internally but is NOT part of the public API.

pub mod array;
pub mod buffer;
pub mod constants;
pub mod creation;
pub mod dimension;
pub mod dtype;
pub mod dynarray;
pub mod error;
pub mod indexing;
pub mod layout;
pub mod manipulation;
pub mod prelude;
pub mod record;

// Re-export key types at crate root for ergonomics
pub use array::owned::Array;
pub use array::view::ArrayView;
pub use array::view_mut::ArrayViewMut;
pub use array::arc::ArcArray;
pub use array::cow::CowArray;
pub use array::ArrayFlags;
pub use array::aliases;
pub use array::display::{set_print_options, get_print_options};

pub use dimension::{Axis, Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};

pub use dtype::{DType, Element, SliceInfoElem};

pub use error::{FerrumError, FerrumResult};

pub use layout::MemoryLayout;

pub use buffer::AsRawBuffer;

pub use dynarray::DynArray;

pub use record::FieldDescriptor;

// Re-export proc macros from ferrum-core-macros.
// The derive macro FerrumRecord shares its name with the trait in record::FerrumRecord.
// Both are re-exported: the derive macro lives in macro namespace, the trait in type namespace.
pub use ferrum_core_macros::{s, promoted_type, FerrumRecord};
pub use record::FerrumRecord;
