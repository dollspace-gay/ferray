// ferray-core: Array module — NdArray<T, D> and ownership variants (REQ-1, REQ-3)
//
// NdArray wraps ndarray::ArrayBase internally. ndarray is NEVER part of the
// public API surface.

pub mod aliases;
pub mod arc;
pub mod cow;
pub mod display;
pub mod introspect;
pub mod iter;
pub mod methods;
mod index_impl;
pub mod owned;
pub mod reductions;
#[cfg(feature = "serde")]
mod serde_impl;
pub mod view;
pub mod view_mut;

/// Flags describing the memory properties of an array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArrayFlags {
    /// Whether the data is C-contiguous (row-major).
    pub c_contiguous: bool,
    /// Whether the data is Fortran-contiguous (column-major).
    pub f_contiguous: bool,
    /// Whether the array owns its data.
    pub owndata: bool,
    /// Whether the array is writeable.
    pub writeable: bool,
}

// Re-export the main types at this module level for convenience.
pub use self::arc::ArcArray;
pub use self::cow::CowArray;
pub use self::owned::Array;
pub use self::view::ArrayView;
pub use self::view_mut::ArrayViewMut;
