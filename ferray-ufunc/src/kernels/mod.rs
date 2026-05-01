// ferray-ufunc: SIMD kernel modules
//
// Provides SIMD-accelerated kernels for f32, f64, i32, i64, plus
// scalar fallbacks (#384 added integer SIMD coverage).

pub mod scalar;
pub mod simd_f32;
pub mod simd_f64;
pub mod simd_i32;
pub mod simd_i64;
