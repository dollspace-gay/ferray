// ferray-core: Array module — NdArray<T, D> and ownership variants (REQ-1, REQ-3)
//
// NdArray wraps ndarray::ArrayBase internally. ndarray is NEVER part of the
// public API surface.

pub mod aliases;
pub mod arc;
pub mod cow;
pub mod display;
mod index_impl;
pub mod introspect;
pub mod iter;
pub mod methods;
pub mod owned;
pub mod reductions;
#[cfg(feature = "serde")]
mod serde_impl;
pub mod sort;
pub mod view;
pub mod view_mut;

/// Flags describing the memory properties of an array.
///
/// Mirrors `NumPy`'s `arr.flags` field: a flat collection of independent
/// boolean memory-layout properties. The names map 1:1 to `NumPy` and are
/// accessed individually by callers, so a bitfield-style enum would
/// hurt ergonomics rather than help.
#[allow(clippy::struct_excessive_bools)]
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
    /// Whether the underlying data pointer is properly aligned for the
    /// element type (#345). Mirrors numpy's NPY_ARRAY_ALIGNED flag.
    ///
    /// Owned arrays produced through ferray's safe constructors are
    /// always aligned because they go through `Vec<T>` which the
    /// allocator guarantees to align for `T`. Views constructed via
    /// `from_shape_ptr` from a raw pointer can be misaligned and
    /// will report `aligned = false` here so callers can detect the
    /// degenerate case before passing into a SIMD path.
    pub aligned: bool,
}

// Re-export the main types at this module level for convenience.
pub use self::arc::ArcArray;
pub use self::cow::CowArray;
pub use self::owned::Array;
pub use self::view::ArrayView;
pub use self::view_mut::ArrayViewMut;
