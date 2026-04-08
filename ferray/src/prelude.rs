// ferray: Prelude — `use ferray::prelude::*` covers 95% of use cases (REQ-3)

// Core array types
pub use ferray_core::ArcArray;
pub use ferray_core::Array;
pub use ferray_core::ArrayFlags;
pub use ferray_core::ArrayView;
pub use ferray_core::ArrayViewMut;
pub use ferray_core::CowArray;

// Common aliases
pub use ferray_core::aliases::{
    Array1, Array2, Array3, Array4, Array5, Array6, ArrayD, ArrayView1, ArrayView2, ArrayView3,
    ArrayViewD, ArrayViewMut1, ArrayViewMut2, ArrayViewMut3, ArrayViewMutD, F32Array1, F32Array2,
    F64Array1, F64Array2,
};

// Dimension types
pub use ferray_core::dimension::{Axis, Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};

// Dtype system
pub use ferray_core::{DType, Element, SliceInfoElem};

// Error handling
pub use ferray_core::{FerrayError, FerrayResult};

// Macros
pub use ferray_core::{promoted_type, s};

// Memory layout
pub use ferray_core::MemoryLayout;

// Runtime-typed array
pub use ferray_core::DynArray;

// Array creation functions
pub use ferray_core::creation::{
    arange, array, asarray, eye, fromiter, full, full_like, geomspace, identity, linspace,
    logspace, ones, ones_like, zeros, zeros_like,
};

// Common math functions (ufuncs)
pub use ferray_ufunc::{
    absolute,
    // Arithmetic
    add,
    all,
    allclose,
    any,
    arccos,
    arccosh,
    arcsin,
    arcsinh,
    arctan,
    arctan2,
    arctanh,
    around,
    array_equal,
    ceil,
    clip,
    cos,
    cosh,
    cumprod,
    cumsum,
    deg2rad,
    degrees,
    diff,
    divide,
    // Comparison
    equal,
    // Exp/log
    exp,
    exp2,
    expm1,
    fix,
    floor,
    fmod,
    gradient,
    greater,
    greater_equal,
    hypot,
    isclose,
    isfinite,
    isinf,
    // Float
    isnan,
    less,
    less_equal,
    log,
    log1p,
    log2,
    log10,
    // Logical
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    maximum,
    minimum,
    mod_,
    multiply,
    nan_to_num,
    negative,
    not_equal,
    power,
    rad2deg,
    radians,
    remainder,
    rint,
    // Rounding
    round,
    sign,
    // Trig
    sin,
    // Special
    sinc,
    sinh,
    sqrt,
    square,
    subtract,
    tan,
    tanh,
    trunc,
};

// Stats (reductions)
pub use ferray_stats::{
    argmax, argmin, max, mean, median, min, nanmax, nanmean, nanmin, nansum, prod, std_, sum, var,
};

// Traits used by downstream generics — users importing the prelude
// expected these to be in scope so trait methods like
// `.array_add()` / `.all()` / `.bitand()` resolve without extra
// imports (#220, #336).
pub use ferray_ufunc::{BitwiseCount, BitwiseOps, Logical, ShiftOps};

// Bitwise ops — added alongside the trait re-export because the
// integration tests expect `bitwise_count` to be directly callable
// from the prelude (#396).
pub use ferray_ufunc::{
    bitwise_and, bitwise_count, bitwise_not, bitwise_or, bitwise_xor, invert, left_shift,
    right_shift,
};

// `Complex<T>` is frequently used alongside ferray arrays; re-export
// here so `use ferray::prelude::*` gives it without pulling in a
// direct `num-complex` dependency (#333).
pub use num_complex::Complex;
