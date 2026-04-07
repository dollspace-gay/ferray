// ferray-ufunc: Common helper functions for ufunc implementations
//
// Provides generic unary/binary operation wrappers that handle contiguous
// vs non-contiguous arrays, SIMD dispatch, and broadcasting.

use ferray_core::Array;
use ferray_core::dimension::broadcast::broadcast_shapes;
use ferray_core::dimension::{Dimension, IxDyn};
use ferray_core::dtype::Element;
use ferray_core::error::{FerrayError, FerrayResult};
use rayon::prelude::*;

use crate::parallel::THRESHOLD_COMPUTE_BOUND;

/// Target chunk size for parallel splits. Large enough that per-chunk
/// overhead (thread handoff, cache warmup) stays below a few percent of
/// the useful compute, small enough that work distributes evenly across
/// available cores. Empirically 64k elements is a good plateau.
const PARALLEL_CHUNK: usize = 65_536;

/// Fill `out` from `src` in parallel, applying `f` to each element, when
/// the total element count exceeds `threshold`. Falls through to a plain
/// serial loop below it.
///
/// Uses rayon's **global** pool via `par_chunks_mut`, not a custom ferray
/// pool — installing into a separate pool from outside its worker threads
/// adds hundreds of microseconds of fixed overhead per call, which
/// completely dominates the arithmetic for typical memory-bound ops.
#[inline]
fn parallel_unary_fill_threshold<T, U, F>(src: &[T], out: &mut [U], threshold: usize, f: F)
where
    T: Copy + Sync,
    U: Send,
    F: Fn(T) -> U + Sync + Send,
{
    let n = src.len();
    debug_assert_eq!(out.len(), n);
    if n >= threshold {
        out.par_chunks_mut(PARALLEL_CHUNK)
            .zip(src.par_chunks(PARALLEL_CHUNK))
            .for_each(|(out_chunk, in_chunk)| {
                for (o, &x) in out_chunk.iter_mut().zip(in_chunk.iter()) {
                    *o = f(x);
                }
            });
    } else {
        for (o, &x) in out.iter_mut().zip(src.iter()) {
            *o = f(x);
        }
    }
}

// Note: binary arithmetic ops (add / sub / mul / div) are memory-bandwidth-
// bound on modern CPUs — a single core already saturates the DRAM channel,
// and splitting the work across threads only hurts per `examples/bench_parallel.rs`.
// So `binary_float_op` / `binary_map_op` / `binary_mixed_op` are intentionally
// kept serial. Transcendentals that go through `unary_float_op_compute` or the
// slice kernels (`unary_slice_op_f64/f32`, `try_simd_f64_unary`) *do* benefit
// from parallelism and have parallel dispatch at the compute-bound threshold.

/// Apply a unary function elementwise, preserving dimension.
/// Works for any `T: Element + Float` (or any Copy type with the given fn).
///
/// When the input is contiguous, operates directly on the underlying slice
/// for better auto-vectorization and cache locality. This is the default
/// path for memory-bound ops (abs, neg, sign, sqrt fallback) — serial,
/// because a single core saturates memory bandwidth. For compute-bound
/// transcendentals (sin, cos, exp, log) use [`unary_float_op_compute`]
/// instead, which parallelizes above the 100k-element threshold.
#[inline]
pub fn unary_float_op<T, D>(
    input: &Array<T, D>,
    f: impl Fn(T) -> T,
) -> FerrayResult<Array<T, D>>
where
    T: Element + Copy,
    D: Dimension,
{
    if let Some(slice) = input.as_slice() {
        let n = slice.len();
        let mut data = Vec::with_capacity(n);
        // SAFETY: every element is written below before the Vec is read.
        #[allow(clippy::uninit_vec)]
        unsafe {
            data.set_len(n);
        }
        for (o, &x) in data.iter_mut().zip(slice.iter()) {
            *o = f(x);
        }
        Array::from_vec(input.dim().clone(), data)
    } else {
        let data: Vec<T> = input.iter().map(|&x| f(x)).collect();
        Array::from_vec(input.dim().clone(), data)
    }
}

/// Parallel variant of [`unary_float_op`] for compute-bound scalar kernels.
///
/// Intended for transcendentals (`sin`, `cos`, `exp`, `log`, `cr_*`) where
/// each element takes ≥10 ns of scalar work — large enough to amortize
/// rayon dispatch at the 100k-element mark. Memory-bound ops like `add`,
/// `sqrt`, or `square` should keep using [`unary_float_op`] instead since
/// their crossover is much higher and parallelism actively hurts below it.
#[inline]
pub fn unary_float_op_compute<T, D>(
    input: &Array<T, D>,
    f: impl Fn(T) -> T + Sync + Send,
) -> FerrayResult<Array<T, D>>
where
    T: Element + Copy,
    D: Dimension,
{
    if let Some(slice) = input.as_slice() {
        let n = slice.len();
        let mut data = Vec::with_capacity(n);
        // SAFETY: every element is written below before the Vec is read.
        #[allow(clippy::uninit_vec)]
        unsafe {
            data.set_len(n);
        }
        parallel_unary_fill_threshold(slice, &mut data, THRESHOLD_COMPUTE_BOUND, f);
        Array::from_vec(input.dim().clone(), data)
    } else {
        let data: Vec<T> = input.iter().map(|&x| f(x)).collect();
        Array::from_vec(input.dim().clone(), data)
    }
}

/// Apply a unary operation using a pre-written slice-to-slice kernel.
///
/// This is used for operations like sqrt, abs, neg where we have optimized
/// SIMD implementations that operate on contiguous `f64` slices directly.
/// Arrays above `THRESHOLD_COMPUTE_BOUND` are chunked across the ferray
/// Rayon pool so the SIMD kernel runs in parallel on disjoint regions.
#[inline]
pub fn unary_slice_op_f64<D>(
    input: &Array<f64, D>,
    kernel: fn(&[f64], &mut [f64]),
    scalar_fallback: fn(f64) -> f64,
) -> FerrayResult<Array<f64, D>>
where
    D: Dimension,
{
    let n = input.size();
    if let Some(slice) = input.as_slice() {
        // SAFETY: kernel writes all n elements. We allocate uninit memory and let
        // the kernel fill it, avoiding a pointless zeroing pass over 8*n bytes.
        let mut data = Vec::with_capacity(n);
        #[allow(clippy::uninit_vec)]
        unsafe {
            data.set_len(n);
        }
        run_slice_kernel_f64(slice, &mut data, kernel);
        Array::from_vec(input.dim().clone(), data)
    } else {
        let data: Vec<f64> = input.iter().map(|&x| scalar_fallback(x)).collect();
        Array::from_vec(input.dim().clone(), data)
    }
}

/// Apply a unary operation using a pre-written slice-to-slice kernel for f32.
#[inline]
pub fn unary_slice_op_f32<D>(
    input: &Array<f32, D>,
    kernel: fn(&[f32], &mut [f32]),
    scalar_fallback: fn(f32) -> f32,
) -> FerrayResult<Array<f32, D>>
where
    D: Dimension,
{
    let n = input.size();
    if let Some(slice) = input.as_slice() {
        let mut data = Vec::with_capacity(n);
        #[allow(clippy::uninit_vec)]
        unsafe {
            data.set_len(n);
        }
        run_slice_kernel_f32(slice, &mut data, kernel);
        Array::from_vec(input.dim().clone(), data)
    } else {
        let data: Vec<f32> = input.iter().map(|&x| scalar_fallback(x)).collect();
        Array::from_vec(input.dim().clone(), data)
    }
}

/// Run a SIMD f64 kernel on potentially-parallel chunks via rayon's global pool.
///
/// The kernel is a plain function pointer so it carries no captured state
/// and is trivially `Sync`.
#[inline]
fn run_slice_kernel_f64(src: &[f64], out: &mut [f64], kernel: fn(&[f64], &mut [f64])) {
    let n = src.len();
    debug_assert_eq!(out.len(), n);
    if n >= THRESHOLD_COMPUTE_BOUND {
        out.par_chunks_mut(PARALLEL_CHUNK)
            .zip(src.par_chunks(PARALLEL_CHUNK))
            .for_each(|(out_chunk, in_chunk)| {
                kernel(in_chunk, out_chunk);
            });
    } else {
        kernel(src, out);
    }
}

#[inline]
fn run_slice_kernel_f32(src: &[f32], out: &mut [f32], kernel: fn(&[f32], &mut [f32])) {
    let n = src.len();
    debug_assert_eq!(out.len(), n);
    if n >= THRESHOLD_COMPUTE_BOUND {
        out.par_chunks_mut(PARALLEL_CHUNK)
            .zip(src.par_chunks(PARALLEL_CHUNK))
            .for_each(|(out_chunk, in_chunk)| {
                kernel(in_chunk, out_chunk);
            });
    } else {
        kernel(src, out);
    }
}

/// Try to run a SIMD f64 kernel on a contiguous array.
///
/// Returns `None` if `T` is not `f64` or the array is not contiguous,
/// allowing the caller to fall back to the generic scalar path.
#[inline]
pub fn try_simd_f64_unary<T, D>(
    input: &Array<T, D>,
    kernel: fn(&[f64], &mut [f64]),
) -> Option<FerrayResult<Array<T, D>>>
where
    T: Element + Copy,
    D: Dimension,
{
    use std::any::TypeId;

    if TypeId::of::<T>() != TypeId::of::<f64>() {
        return None;
    }
    let slice = input.as_slice()?;
    let n = slice.len();
    // SAFETY: T is f64, verified by TypeId check above. f64 and T have
    // identical size, alignment, and bit representation.
    let f64_slice: &[f64] = unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const f64, n) };
    let mut output = Vec::with_capacity(n);
    #[allow(clippy::uninit_vec)]
    unsafe {
        output.set_len(n);
    }
    run_slice_kernel_f64(f64_slice, &mut output, kernel);
    // SAFETY: T is f64. Reinterpret Vec<f64> as Vec<T> without copying.
    let t_vec: Vec<T> = unsafe {
        let mut md = std::mem::ManuallyDrop::new(output);
        Vec::from_raw_parts(md.as_mut_ptr() as *mut T, n, n)
    };
    Some(Array::from_vec(input.dim().clone(), t_vec))
}

/// Apply a unary function that maps T -> U, preserving dimension.
#[inline]
pub fn unary_map_op<T, U, D>(input: &Array<T, D>, f: impl Fn(T) -> U) -> FerrayResult<Array<U, D>>
where
    T: Element + Copy,
    U: Element,
    D: Dimension,
{
    let data: Vec<U> = input.iter().map(|&x| f(x)).collect();
    Array::from_vec(input.dim().clone(), data)
}

/// Apply a binary function elementwise with NumPy broadcasting.
///
/// When `a` and `b` have identical shapes, takes the fast path (zip iter).
/// Otherwise, broadcasts both inputs to the common shape and reconstructs
/// the result `Dimension` via [`Dimension::from_dim_slice`]. Both inputs
/// share dimension type `D`, so the broadcast result has the same rank
/// (or, for `IxDyn`, the maximum of the two ranks).
///
/// For cross-rank broadcasting (e.g. `Ix1` + `Ix2`), use
/// [`binary_broadcast_op`] which returns `Array<T, IxDyn>`.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if the shapes are not broadcast-compatible.
#[inline]
pub fn binary_float_op<T, D>(
    a: &Array<T, D>,
    b: &Array<T, D>,
    f: impl Fn(T, T) -> T,
) -> FerrayResult<Array<T, D>>
where
    T: Element + Copy,
    D: Dimension,
{
    // Fast path: identical shapes — no broadcasting needed.
    if a.shape() == b.shape() {
        // Contiguous subpath: iterate over raw slices so the compiler
        // auto-vectorizes the inner loop. Memory-bound, no parallelism.
        if let (Some(a_slice), Some(b_slice)) = (a.as_slice(), b.as_slice()) {
            let data: Vec<T> = a_slice
                .iter()
                .zip(b_slice.iter())
                .map(|(&x, &y)| f(x, y))
                .collect();
            return Array::from_vec(a.dim().clone(), data);
        }
        let data: Vec<T> = a.iter().zip(b.iter()).map(|(&x, &y)| f(x, y)).collect();
        return Array::from_vec(a.dim().clone(), data);
    }

    // Broadcasting path.
    let target_shape = broadcast_shapes(a.shape(), b.shape()).map_err(|_| {
        FerrayError::shape_mismatch(format!(
            "binary op: shapes {:?} and {:?} are not broadcast-compatible",
            a.shape(),
            b.shape()
        ))
    })?;
    let a_view = a.broadcast_to(&target_shape)?;
    let b_view = b.broadcast_to(&target_shape)?;
    let data: Vec<T> = a_view
        .iter()
        .zip(b_view.iter())
        .map(|(&x, &y)| f(x, y))
        .collect();
    let result_dim = D::from_dim_slice(&target_shape).ok_or_else(|| {
        FerrayError::shape_mismatch(format!(
            "binary op: cannot represent broadcast result shape {:?} as the input dimension type",
            target_shape
        ))
    })?;
    Array::from_vec(result_dim, data)
}

/// Apply a binary function that maps `(T, T) -> U` with NumPy broadcasting.
///
/// Same shape semantics as [`binary_float_op`] but allows the output element
/// type to differ from the inputs. Used by comparison ops (`(T, T) -> bool`),
/// logical ops, and `isclose`/`isfinite`-style functions.
#[inline]
pub fn binary_map_op<T, U, D>(
    a: &Array<T, D>,
    b: &Array<T, D>,
    f: impl Fn(T, T) -> U,
) -> FerrayResult<Array<U, D>>
where
    T: Element + Copy,
    U: Element,
    D: Dimension,
{
    // Fast path: identical shapes — no broadcasting needed.
    if a.shape() == b.shape() {
        if let (Some(a_slice), Some(b_slice)) = (a.as_slice(), b.as_slice()) {
            let data: Vec<U> = a_slice
                .iter()
                .zip(b_slice.iter())
                .map(|(&x, &y)| f(x, y))
                .collect();
            return Array::from_vec(a.dim().clone(), data);
        }
        let data: Vec<U> = a.iter().zip(b.iter()).map(|(&x, &y)| f(x, y)).collect();
        return Array::from_vec(a.dim().clone(), data);
    }

    // Broadcasting path.
    let target_shape = broadcast_shapes(a.shape(), b.shape()).map_err(|_| {
        FerrayError::shape_mismatch(format!(
            "binary op: shapes {:?} and {:?} are not broadcast-compatible",
            a.shape(),
            b.shape()
        ))
    })?;
    let a_view = a.broadcast_to(&target_shape)?;
    let b_view = b.broadcast_to(&target_shape)?;
    let data: Vec<U> = a_view
        .iter()
        .zip(b_view.iter())
        .map(|(&x, &y)| f(x, y))
        .collect();
    let result_dim = D::from_dim_slice(&target_shape).ok_or_else(|| {
        FerrayError::shape_mismatch(format!(
            "binary op: cannot represent broadcast result shape {:?} as the input dimension type",
            target_shape
        ))
    })?;
    Array::from_vec(result_dim, data)
}

/// Apply a binary function on two arrays of different element types with
/// broadcasting. Used for ops where the operands have distinct types
/// (e.g. `left_shift::<T, u32>`, `ldexp::<T, i32>`).
#[inline]
pub fn binary_mixed_op<T, U, V, D>(
    a: &Array<T, D>,
    b: &Array<U, D>,
    f: impl Fn(T, U) -> V,
) -> FerrayResult<Array<V, D>>
where
    T: Element + Copy,
    U: Element + Copy,
    V: Element,
    D: Dimension,
{
    // Fast path: identical shapes.
    if a.shape() == b.shape() {
        if let (Some(a_slice), Some(b_slice)) = (a.as_slice(), b.as_slice()) {
            let data: Vec<V> = a_slice
                .iter()
                .zip(b_slice.iter())
                .map(|(&x, &y)| f(x, y))
                .collect();
            return Array::from_vec(a.dim().clone(), data);
        }
        let data: Vec<V> = a.iter().zip(b.iter()).map(|(&x, &y)| f(x, y)).collect();
        return Array::from_vec(a.dim().clone(), data);
    }

    // Broadcasting path.
    let target_shape = broadcast_shapes(a.shape(), b.shape()).map_err(|_| {
        FerrayError::shape_mismatch(format!(
            "binary mixed op: shapes {:?} and {:?} are not broadcast-compatible",
            a.shape(),
            b.shape()
        ))
    })?;
    let a_view = a.broadcast_to(&target_shape)?;
    let b_view = b.broadcast_to(&target_shape)?;
    let data: Vec<V> = a_view
        .iter()
        .zip(b_view.iter())
        .map(|(&x, &y)| f(x, y))
        .collect();
    let result_dim = D::from_dim_slice(&target_shape).ok_or_else(|| {
        FerrayError::shape_mismatch(format!(
            "binary mixed op: cannot represent broadcast result shape {:?} as the input dimension type",
            target_shape
        ))
    })?;
    Array::from_vec(result_dim, data)
}

/// Apply a binary function with broadcasting support.
/// Returns an IxDyn array with the broadcast shape.
pub fn binary_broadcast_op<T, D1, D2>(
    a: &Array<T, D1>,
    b: &Array<T, D2>,
    f: impl Fn(T, T) -> T,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Copy,
    D1: Dimension,
    D2: Dimension,
{
    let shape = broadcast_shapes(a.shape(), b.shape())?;
    let a_view = a.broadcast_to(&shape)?;
    let b_view = b.broadcast_to(&shape)?;
    let data: Vec<T> = a_view
        .iter()
        .zip(b_view.iter())
        .map(|(&x, &y)| f(x, y))
        .collect();
    Array::from_vec(IxDyn::from(&shape[..]), data)
}

/// Apply a binary function with broadcasting, mapping to output type U.
pub fn binary_broadcast_map_op<T, U, D1, D2>(
    a: &Array<T, D1>,
    b: &Array<T, D2>,
    f: impl Fn(T, T) -> U,
) -> FerrayResult<Array<U, IxDyn>>
where
    T: Element + Copy,
    U: Element,
    D1: Dimension,
    D2: Dimension,
{
    let shape = broadcast_shapes(a.shape(), b.shape())?;
    let a_view = a.broadcast_to(&shape)?;
    let b_view = b.broadcast_to(&shape)?;
    let data: Vec<U> = a_view
        .iter()
        .zip(b_view.iter())
        .map(|(&x, &y)| f(x, y))
        .collect();
    Array::from_vec(IxDyn::from(&shape[..]), data)
}

// ---------------------------------------------------------------------------
// f16 helpers — f32-promoted operations (feature-gated)
// ---------------------------------------------------------------------------

/// Apply a unary f32 function to an f16 array via promotion.
///
/// Each element is promoted to f32, the function is applied, and
/// the result is converted back to f16.
#[cfg(feature = "f16")]
#[inline]
pub fn unary_f16_op<D>(
    input: &Array<half::f16, D>,
    f: impl Fn(f32) -> f32,
) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    let data: Vec<half::f16> = input
        .iter()
        .map(|&x| half::f16::from_f32(f(x.to_f32())))
        .collect();
    Array::from_vec(input.dim().clone(), data)
}

/// Apply a unary f32 function to an f16 array, returning a bool array.
#[cfg(feature = "f16")]
#[inline]
pub fn unary_f16_to_bool_op<D>(
    input: &Array<half::f16, D>,
    f: impl Fn(f32) -> bool,
) -> FerrayResult<Array<bool, D>>
where
    D: Dimension,
{
    let data: Vec<bool> = input.iter().map(|&x| f(x.to_f32())).collect();
    Array::from_vec(input.dim().clone(), data)
}

/// Apply a binary f32 function to two f16 arrays via promotion, with broadcasting.
#[cfg(feature = "f16")]
#[inline]
pub fn binary_f16_op<D>(
    a: &Array<half::f16, D>,
    b: &Array<half::f16, D>,
    f: impl Fn(f32, f32) -> f32,
) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    binary_float_op(a, b, |x, y| half::f16::from_f32(f(x.to_f32(), y.to_f32())))
}

/// Apply a binary f32 function to two f16 arrays, returning a bool array, with broadcasting.
#[cfg(feature = "f16")]
#[inline]
pub fn binary_f16_to_bool_op<D>(
    a: &Array<half::f16, D>,
    b: &Array<half::f16, D>,
    f: impl Fn(f32, f32) -> bool,
) -> FerrayResult<Array<bool, D>>
where
    D: Dimension,
{
    binary_map_op(a, b, |x, y| f(x.to_f32(), y.to_f32()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::{Ix1, Ix2};

    #[test]
    fn unary_op_works() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 4.0, 9.0]).unwrap();
        let r = unary_float_op(&a, f64::sqrt).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn binary_op_same_shape() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![4.0, 5.0, 6.0]).unwrap();
        let r = binary_float_op(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn binary_op_shape_mismatch() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![4.0, 5.0]).unwrap();
        assert!(binary_float_op(&a, &b, |x, y| x + y).is_err());
    }

    #[test]
    fn binary_broadcast_works() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 1]), vec![1.0, 2.0]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![10.0, 20.0, 30.0]).unwrap();
        let r = binary_broadcast_op(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        let s: Vec<f64> = r.iter().copied().collect();
        assert_eq!(s, vec![11.0, 21.0, 31.0, 12.0, 22.0, 32.0]);
    }

    #[test]
    fn binary_float_op_broadcasts_within_same_rank() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([3, 1]), vec![1.0, 2.0, 3.0]).unwrap();
        let b =
            Array::<f64, Ix2>::from_vec(Ix2::new([1, 4]), vec![10.0, 20.0, 30.0, 40.0]).unwrap();
        let r = binary_float_op(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(r.shape(), &[3, 4]);
        assert_eq!(
            r.iter().copied().collect::<Vec<_>>(),
            vec![
                11.0, 21.0, 31.0, 41.0, 12.0, 22.0, 32.0, 42.0, 13.0, 23.0, 33.0, 43.0,
            ]
        );
    }

    #[test]
    fn binary_map_op_broadcasts_within_same_rank() {
        let a = Array::<i32, Ix2>::from_vec(Ix2::new([2, 1]), vec![1, 5]).unwrap();
        let b = Array::<i32, Ix2>::from_vec(Ix2::new([1, 3]), vec![3, 5, 7]).unwrap();
        let r = binary_map_op(&a, &b, |x, y| x < y).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(
            r.iter().copied().collect::<Vec<_>>(),
            vec![true, true, true, false, false, true]
        );
    }

    #[test]
    fn binary_mixed_op_broadcasts() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 1]), vec![1.0, 4.0]).unwrap();
        let b = Array::<i32, Ix2>::from_vec(Ix2::new([1, 3]), vec![1, 2, 3]).unwrap();
        let r = binary_mixed_op(&a, &b, |x, n| x * (1 << n) as f64).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(
            r.iter().copied().collect::<Vec<_>>(),
            vec![2.0, 4.0, 8.0, 8.0, 16.0, 32.0]
        );
    }

    #[test]
    fn binary_op_incompatible_shapes_error() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(binary_float_op(&a, &b, |x, y| x + y).is_err());
        assert!(binary_map_op(&a, &b, |x, y| x == y).is_err());
    }
}
