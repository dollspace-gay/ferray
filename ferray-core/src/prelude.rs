// ferray-core: Prelude — convenient glob import
//
// Usage: `use ferray_core::prelude::*;`

// Core array types
pub use crate::array::ArrayFlags;
pub use crate::array::arc::ArcArray;
pub use crate::array::cow::CowArray;
pub use crate::array::owned::Array;
pub use crate::array::view::ArrayView;
pub use crate::array::view_mut::ArrayViewMut;

// Common aliases
pub use crate::array::aliases::{
    Array1, Array2, Array3, Array4, Array5, Array6, ArrayD, ArrayView1, ArrayView2, ArrayView3,
    ArrayViewD, ArrayViewMut1, ArrayViewMut2, ArrayViewMut3, ArrayViewMutD, F32Array1, F32Array2,
    F64Array1, F64Array2,
};

// Dimension types
pub use crate::dimension::{Axis, Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};

// Dtype system
pub use crate::dtype::{DType, Element, SliceInfoElem};

// Type casting and promotion
pub use crate::dtype::casting::{AsType, CastKind};
pub use crate::dtype::promotion::{PromoteTo, Promoted};

// Proc macros
pub use ferray_core_macros::{FerrayRecord, promoted_type, s};

// Error handling
pub use crate::error::{FerrayError, FerrayResult};

// Memory layout
pub use crate::layout::MemoryLayout;

// Buffer interop
pub use crate::buffer::AsRawBuffer;

// Runtime-typed array
pub use crate::dynarray::DynArray;

// Record types
pub use crate::record::FerrayRecord;

// Display configuration
pub use crate::array::display::{get_print_options, set_print_options};
