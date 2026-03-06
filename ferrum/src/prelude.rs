// ferrum: Prelude — `use ferrum::prelude::*` covers 95% of use cases (REQ-3)

// Core array types
pub use ferrum_core::Array;
pub use ferrum_core::ArrayView;
pub use ferrum_core::ArrayViewMut;
pub use ferrum_core::ArcArray;
pub use ferrum_core::CowArray;
pub use ferrum_core::ArrayFlags;

// Common aliases
pub use ferrum_core::aliases::{
    Array1, Array2, Array3, Array4, Array5, Array6, ArrayD,
    ArrayView1, ArrayView2, ArrayView3, ArrayViewD,
    ArrayViewMut1, ArrayViewMut2, ArrayViewMut3, ArrayViewMutD,
    F32Array1, F32Array2, F64Array1, F64Array2,
};

// Dimension types
pub use ferrum_core::dimension::{Axis, Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};

// Dtype system
pub use ferrum_core::{DType, Element, SliceInfoElem};

// Error handling
pub use ferrum_core::{FerrumError, FerrumResult};

// Macros
pub use ferrum_core::{s, promoted_type};

// Memory layout
pub use ferrum_core::MemoryLayout;

// Runtime-typed array
pub use ferrum_core::DynArray;

// Array creation functions
pub use ferrum_core::creation::{
    array, asarray, zeros, ones, full,
    zeros_like, ones_like, full_like,
    arange, linspace, logspace, geomspace,
    eye, identity, fromiter,
};

// Common math functions (ufuncs)
pub use ferrum_ufunc::{
    // Trig
    sin, cos, tan, arcsin, arccos, arctan, arctan2,
    sinh, cosh, tanh, arcsinh, arccosh, arctanh,
    deg2rad, rad2deg, degrees, radians, hypot,
    // Exp/log
    exp, exp2, expm1, log, log2, log10, log1p,
    // Rounding
    round, floor, ceil, trunc, fix, rint, around,
    // Arithmetic
    add, subtract, multiply, divide, power, sqrt, square,
    absolute, negative, sign, remainder, mod_, fmod,
    cumsum, cumprod, diff, gradient,
    // Float
    isnan, isinf, isfinite, clip, nan_to_num,
    maximum, minimum,
    // Comparison
    equal, not_equal, less, greater, less_equal, greater_equal,
    allclose, isclose, array_equal,
    // Logical
    logical_and, logical_or, logical_xor, logical_not,
    all, any,
    // Special
    sinc,
};

// Stats (reductions)
pub use ferrum_stats::{
    sum, prod, min, max, argmin, argmax,
    mean, var, std_, median,
    nansum, nanmean, nanmin, nanmax,
};
