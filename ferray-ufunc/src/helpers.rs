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

/// Reinterpret a concrete `Array<SRC, D>` back to `Array<DST, D>` when
/// the caller has already proven (via [`std::any::TypeId`] equality or
/// equivalent) that `SRC` and `DST` are the same type. Exists so that
/// the TypeId-dispatched f32/f64 fast paths scattered across
/// [`crate::ops::explog`], [`crate::ops::floatintrinsic`] etc. share
/// one well-documented safety boundary instead of inlining bare
/// `transmute_copy` calls in every function (see issues #159, #160).
///
/// # Safety
/// `SRC` and `DST` must be the same runtime type. The caller must verify
/// this with `TypeId::of::<SRC>() == TypeId::of::<DST>()` before
/// invoking; violating the precondition is instant UB. The extra
/// `size_of`/`align_of` assertions are a cheap runtime sanity check.
#[inline]
pub(crate) unsafe fn reinterpret_array<SRC, DST, D>(arr: Array<SRC, D>) -> Array<DST, D>
where
    SRC: Element,
    DST: Element,
    D: Dimension,
{
    // If we ever land here with types that don't match in size/align,
    // that is a bug in the caller's TypeId dispatch — not something
    // users of the library can trigger.
    debug_assert_eq!(std::mem::size_of::<SRC>(), std::mem::size_of::<DST>());
    debug_assert_eq!(std::mem::align_of::<SRC>(), std::mem::align_of::<DST>());
    // SAFETY: guaranteed by caller; see function docs.
    unsafe { std::mem::transmute_copy(&std::mem::ManuallyDrop::new(arr)) }
}

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
// So `binary_elementwise_op` / `binary_map_op` / `binary_mixed_op` are intentionally
// kept serial. Transcendentals that go through `unary_float_op_compute` or the
// slice kernels (`unary_slice_op_f64/f32`, `try_simd_f64_unary`) *do* benefit
// from parallelism and have parallel dispatch at the compute-bound threshold.

/// Borrow the input as a contiguous slice when possible, otherwise
/// materialize the elements into a fresh row-major Vec.
///
/// Used by the unary fast paths so non-contiguous inputs (e.g. transposed
/// or strided views) can still feed SIMD / parallel slice kernels —
/// previously these inputs fell straight through to the generic
/// `input.iter().map().collect()` scalar path with no chance of SIMD
/// acceleration (#385). The cost on the non-contig branch is one
/// allocation + one element-by-element copy, which is still far cheaper
/// than scalar evaluation of a transcendental kernel over millions of
/// elements.
///
/// On the contig branch the returned `Cow::Borrowed` aliases the
/// underlying buffer with no copy at all, so the existing C-contig fast
/// path keeps its zero-overhead behaviour.
#[inline]
pub(crate) fn contig_input<T, D>(input: &Array<T, D>) -> std::borrow::Cow<'_, [T]>
where
    T: Element + Copy,
    D: Dimension,
{
    if let Some(slice) = input.as_slice() {
        std::borrow::Cow::Borrowed(slice)
    } else {
        std::borrow::Cow::Owned(input.iter().copied().collect())
    }
}

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
pub fn unary_float_op<T, D>(input: &Array<T, D>, f: impl Fn(T) -> T) -> FerrayResult<Array<T, D>>
where
    T: Element + Copy,
    D: Dimension,
{
    // Run the same auto-vectorizable loop on contig and non-contig inputs.
    // For non-contig views `contig_input` materializes a row-major Vec
    // first (#385) so the loop below can still benefit from auto-SIMD
    // instead of going through the generic per-element iterator path.
    let src = contig_input(input);
    let n = src.len();
    let mut data = Vec::with_capacity(n);
    // SAFETY: every element is written below before the Vec is read.
    #[allow(clippy::uninit_vec)]
    unsafe {
        data.set_len(n);
    }
    for (o, &x) in data.iter_mut().zip(src.iter()) {
        *o = f(x);
    }
    Array::from_vec(input.dim().clone(), data)
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
    // Materialize non-contig inputs once so the parallel kernel can run
    // on the resulting Vec — for compute-bound transcendentals this is
    // a clear win even with the extra allocation (#385).
    let src = contig_input(input);
    let n = src.len();
    let mut data = Vec::with_capacity(n);
    // SAFETY: every element is written below before the Vec is read.
    #[allow(clippy::uninit_vec)]
    unsafe {
        data.set_len(n);
    }
    parallel_unary_fill_threshold(&src, &mut data, THRESHOLD_COMPUTE_BOUND, f);
    Array::from_vec(input.dim().clone(), data)
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
) -> FerrayResult<Array<f64, D>>
where
    D: Dimension,
{
    let n = input.size();
    // Always go through the SIMD slice kernel — for non-contig inputs we
    // pay one allocation to materialize a row-major Vec<f64> and then the
    // kernel runs on bulk slices instead of falling back to a per-element
    // scalar loop (#385). Previously this function took a separate
    // `scalar_fallback: fn(f64) -> f64` for the non-contig branch; that
    // parameter has been removed because the contig_input helper makes
    // the slice kernel work uniformly.
    let src = contig_input(input);
    // SAFETY: kernel writes all n elements. We allocate uninit memory and let
    // the kernel fill it, avoiding a pointless zeroing pass over 8*n bytes.
    let mut data = Vec::with_capacity(n);
    #[allow(clippy::uninit_vec)]
    unsafe {
        data.set_len(n);
    }
    run_slice_kernel_f64(&src, &mut data, kernel);
    Array::from_vec(input.dim().clone(), data)
}

/// Apply a unary operation using a pre-written slice-to-slice kernel for f32.
#[inline]
pub fn unary_slice_op_f32<D>(
    input: &Array<f32, D>,
    kernel: fn(&[f32], &mut [f32]),
) -> FerrayResult<Array<f32, D>>
where
    D: Dimension,
{
    let n = input.size();
    // See unary_slice_op_f64 for the rationale on dropping the
    // scalar_fallback parameter (#385).
    let src = contig_input(input);
    let mut data = Vec::with_capacity(n);
    #[allow(clippy::uninit_vec)]
    unsafe {
        data.set_len(n);
    }
    run_slice_kernel_f32(&src, &mut data, kernel);
    Array::from_vec(input.dim().clone(), data)
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

/// Try to run a SIMD f64 kernel on an array, regardless of memory layout.
///
/// Returns `None` only when `T` is not `f64`. For non-contig f64 inputs
/// the elements are first materialized into a fresh row-major Vec<f64>
/// (#385) and the kernel runs on that — previously this short-circuited
/// to `None` for any non-contig view, forcing callers down the scalar
/// fallback path even on transposed/strided f64 arrays where the SIMD
/// win is large.
///
/// The result is always a fresh C-contig owned array.
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

    // Borrow the f64 slice if the input is contig, otherwise materialize
    // a row-major Vec<f64>. The Cow's Owned variant is held alive for the
    // duration of the kernel call below; we need an explicit binding so
    // the borrow lasts long enough.
    let n = input.size();
    let src_t = contig_input(input);
    // SAFETY: T is f64, verified by TypeId check above. f64 and T have
    // identical size, alignment, and bit representation.
    let src_f64: &[f64] = unsafe { std::slice::from_raw_parts(src_t.as_ptr().cast::<f64>(), n) };

    let mut output = Vec::with_capacity(n);
    #[allow(clippy::uninit_vec)]
    unsafe {
        output.set_len(n);
    }
    run_slice_kernel_f64(src_f64, &mut output, kernel);

    let cap = output.capacity();
    // SAFETY: T is f64. Reinterpret Vec<f64> as Vec<T> without copying.
    let t_vec: Vec<T> = unsafe {
        let mut md = std::mem::ManuallyDrop::new(output);
        Vec::from_raw_parts(md.as_mut_ptr().cast::<T>(), n, cap)
    };
    Some(Array::from_vec(input.dim().clone(), t_vec))
}

/// f32 sibling of [`try_simd_f64_unary`]. Returns `None` if `T` is not
/// `f32`; otherwise dispatches the slice kernel and returns the result
/// wrapped in `Some`.
#[inline]
pub fn try_simd_f32_unary<T, D>(
    input: &Array<T, D>,
    kernel: fn(&[f32], &mut [f32]),
) -> Option<FerrayResult<Array<T, D>>>
where
    T: Element + Copy,
    D: Dimension,
{
    use std::any::TypeId;

    if TypeId::of::<T>() != TypeId::of::<f32>() {
        return None;
    }

    let n = input.size();
    let src_t = contig_input(input);
    // SAFETY: T is f32, verified by TypeId check above.
    let src_f32: &[f32] = unsafe { std::slice::from_raw_parts(src_t.as_ptr().cast::<f32>(), n) };

    let mut output = Vec::with_capacity(n);
    #[allow(clippy::uninit_vec)]
    unsafe {
        output.set_len(n);
    }
    run_slice_kernel_f32(src_f32, &mut output, kernel);

    let cap = output.capacity();
    // SAFETY: T is f32.
    let t_vec: Vec<T> = unsafe {
        let mut md = std::mem::ManuallyDrop::new(output);
        Vec::from_raw_parts(md.as_mut_ptr().cast::<T>(), n, cap)
    };

    Some(Array::from_vec(input.dim().clone(), t_vec))
}

/// Run a SIMD-dispatched binary kernel on two contiguous f64 slices.
///
/// Same scheduling shape as [`run_slice_kernel_f64`]: parallel via rayon
/// when `n >= THRESHOLD_COMPUTE_BOUND`, single-threaded otherwise.
#[inline]
fn run_slice_kernel_binary_f64(
    a: &[f64],
    b: &[f64],
    out: &mut [f64],
    kernel: fn(&[f64], &[f64], &mut [f64]),
) {
    let n = a.len();
    debug_assert_eq!(b.len(), n);
    debug_assert_eq!(out.len(), n);
    if n >= THRESHOLD_COMPUTE_BOUND {
        out.par_chunks_mut(PARALLEL_CHUNK)
            .zip(a.par_chunks(PARALLEL_CHUNK))
            .zip(b.par_chunks(PARALLEL_CHUNK))
            .for_each(|((out_chunk, a_chunk), b_chunk)| {
                kernel(a_chunk, b_chunk, out_chunk);
            });
    } else {
        kernel(a, b, out);
    }
}

#[inline]
fn run_slice_kernel_binary_f32(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    kernel: fn(&[f32], &[f32], &mut [f32]),
) {
    let n = a.len();
    debug_assert_eq!(b.len(), n);
    debug_assert_eq!(out.len(), n);
    if n >= THRESHOLD_COMPUTE_BOUND {
        out.par_chunks_mut(PARALLEL_CHUNK)
            .zip(a.par_chunks(PARALLEL_CHUNK))
            .zip(b.par_chunks(PARALLEL_CHUNK))
            .for_each(|((out_chunk, a_chunk), b_chunk)| {
                kernel(a_chunk, b_chunk, out_chunk);
            });
    } else {
        kernel(a, b, out);
    }
}

/// Binary sibling of [`try_simd_f64_unary`].
///
/// Returns `None` if `T` is not `f64` or shapes differ (broadcasting
/// must be handled by the caller). Otherwise materialises both inputs
/// via `contig_input`, runs the SIMD slice kernel, and returns the
/// result wrapped in `Some`.
#[inline]
pub fn try_simd_f64_binary<T, D>(
    a: &Array<T, D>,
    b: &Array<T, D>,
    kernel: fn(&[f64], &[f64], &mut [f64]),
) -> Option<FerrayResult<Array<T, D>>>
where
    T: Element + Copy,
    D: Dimension,
{
    use std::any::TypeId;

    if TypeId::of::<T>() != TypeId::of::<f64>() {
        return None;
    }
    if a.shape() != b.shape() {
        // Broadcasting needs the generic helper. Returning None lets
        // the caller fall through to it.
        return None;
    }

    let n = a.size();
    let a_src = contig_input(a);
    let b_src = contig_input(b);
    // SAFETY: T is f64, verified by TypeId check above.
    let a_f64: &[f64] = unsafe { std::slice::from_raw_parts(a_src.as_ptr().cast::<f64>(), n) };
    let b_f64: &[f64] = unsafe { std::slice::from_raw_parts(b_src.as_ptr().cast::<f64>(), n) };

    let mut output = Vec::with_capacity(n);
    #[allow(clippy::uninit_vec)]
    unsafe {
        output.set_len(n);
    }
    run_slice_kernel_binary_f64(a_f64, b_f64, &mut output, kernel);

    let cap = output.capacity();
    // SAFETY: T is f64. Reinterpret Vec<f64> as Vec<T>.
    let t_vec: Vec<T> = unsafe {
        let mut md = std::mem::ManuallyDrop::new(output);
        Vec::from_raw_parts(md.as_mut_ptr().cast::<T>(), n, cap)
    };
    Some(Array::from_vec(a.dim().clone(), t_vec))
}

/// f32 sibling of [`try_simd_f64_binary`].
#[inline]
pub fn try_simd_f32_binary<T, D>(
    a: &Array<T, D>,
    b: &Array<T, D>,
    kernel: fn(&[f32], &[f32], &mut [f32]),
) -> Option<FerrayResult<Array<T, D>>>
where
    T: Element + Copy,
    D: Dimension,
{
    use std::any::TypeId;

    if TypeId::of::<T>() != TypeId::of::<f32>() {
        return None;
    }
    if a.shape() != b.shape() {
        return None;
    }

    let n = a.size();
    let a_src = contig_input(a);
    let b_src = contig_input(b);
    // SAFETY: T is f32, verified by TypeId check above.
    let a_f32: &[f32] = unsafe { std::slice::from_raw_parts(a_src.as_ptr().cast::<f32>(), n) };
    let b_f32: &[f32] = unsafe { std::slice::from_raw_parts(b_src.as_ptr().cast::<f32>(), n) };

    let mut output = Vec::with_capacity(n);
    #[allow(clippy::uninit_vec)]
    unsafe {
        output.set_len(n);
    }
    run_slice_kernel_binary_f32(a_f32, b_f32, &mut output, kernel);

    let cap = output.capacity();
    // SAFETY: T is f32.
    let t_vec: Vec<T> = unsafe {
        let mut md = std::mem::ManuallyDrop::new(output);
        Vec::from_raw_parts(md.as_mut_ptr().cast::<T>(), n, cap)
    };
    Some(Array::from_vec(a.dim().clone(), t_vec))
}

/// Apply a unary function that maps T -> U, preserving dimension.
#[inline]
pub fn unary_map_op<T, U, D>(input: &Array<T, D>, f: impl Fn(T) -> U) -> FerrayResult<Array<U, D>>
where
    T: Element + Copy,
    U: Element,
    D: Dimension,
{
    // Materialize non-contig inputs first (#385) so the inner loop iterates
    // a flat slice — much friendlier to auto-vectorization than the
    // generic ndarray iterator and uniform with the other unary helpers.
    let src = contig_input(input);
    let data: Vec<U> = src.iter().map(|&x| f(x)).collect();
    Array::from_vec(input.dim().clone(), data)
}

/// Apply a binary function elementwise with `NumPy` broadcasting.
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
pub fn binary_elementwise_op<T, D>(
    a: &Array<T, D>,
    b: &Array<T, D>,
    f: impl Fn(T, T) -> T,
) -> FerrayResult<Array<T, D>>
where
    T: Element + Copy,
    D: Dimension,
{
    // Fast path: identical shapes — no broadcasting needed. Materialize
    // any non-contig operand into a row-major Vec first (#385) so the
    // inner zip-loop sees two flat slices and can auto-vectorize.
    if a.shape() == b.shape() {
        let a_src = contig_input(a);
        let b_src = contig_input(b);
        let data: Vec<T> = a_src
            .iter()
            .zip(b_src.iter())
            .map(|(&x, &y)| f(x, y))
            .collect();
        return Array::from_vec(a.dim().clone(), data);
    }

    // Broadcasting path. Broadcast views are stride-tricked, never
    // contig — there is no contig fast path to take here, so we walk
    // them via the iterator interface.
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
            "binary op: cannot represent broadcast result shape {target_shape:?} as the input dimension type"
        ))
    })?;
    Array::from_vec(result_dim, data)
}

/// Apply a binary function that maps `(T, T) -> U` with `NumPy` broadcasting.
///
/// Same shape semantics as [`binary_elementwise_op`] but allows the output element
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
    // Materialize non-contig operands first (#385) so the inner zip-loop
    // sees flat slices.
    if a.shape() == b.shape() {
        let a_src = contig_input(a);
        let b_src = contig_input(b);
        let data: Vec<U> = a_src
            .iter()
            .zip(b_src.iter())
            .map(|(&x, &y)| f(x, y))
            .collect();
        return Array::from_vec(a.dim().clone(), data);
    }

    // Broadcasting path. Broadcast views are stride-tricked, never
    // contig — there is no contig fast path to take here.
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
            "binary op: cannot represent broadcast result shape {target_shape:?} as the input dimension type"
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
    // Fast path: identical shapes. Materialize non-contig operands so the
    // zip-loop runs over flat slices (#385).
    if a.shape() == b.shape() {
        let a_src = contig_input(a);
        let b_src = contig_input(b);
        let data: Vec<V> = a_src
            .iter()
            .zip(b_src.iter())
            .map(|(&x, &y)| f(x, y))
            .collect();
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
            "binary mixed op: cannot represent broadcast result shape {target_shape:?} as the input dimension type"
        ))
    })?;
    Array::from_vec(result_dim, data)
}

/// Apply a binary function with broadcasting support.
/// Returns an `IxDyn` array with the broadcast shape.
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
    binary_elementwise_op(a, b, |x, y| half::f16::from_f32(f(x.to_f32(), y.to_f32())))
}

/// Define a unary `*_f16` entry point that promotes elements to `f32`,
/// applies `$f32_fn`, and converts the result back to `f16`. Eliminates
/// the six-line `pub fn ..._f16<D>(input) { helpers::unary_f16_op(input, ...) }`
/// pattern previously duplicated across every `ops::*` module
/// (see issue #142).
///
/// The macro itself is always defined; feature-gate the generated
/// function by including `#[cfg(feature = "f16")]` in the attribute list
/// passed to the macro call.
macro_rules! unary_f16_fn {
    (
        $(#[$attr:meta])*
        $name:ident,
        $f32_fn:expr
    ) => {
        $(#[$attr])*
        pub fn $name<D>(
            input: &::ferray_core::Array<half::f16, D>,
        ) -> ::ferray_core::error::FerrayResult<
            ::ferray_core::Array<half::f16, D>,
        >
        where
            D: ::ferray_core::dimension::Dimension,
        {
            $crate::helpers::unary_f16_op(input, $f32_fn)
        }
    };
}

/// Define a binary `*_f16` entry point analogous to [`unary_f16_fn!`].
macro_rules! binary_f16_fn {
    (
        $(#[$attr:meta])*
        $name:ident,
        $f32_fn:expr
    ) => {
        $(#[$attr])*
        pub fn $name<D>(
            a: &::ferray_core::Array<half::f16, D>,
            b: &::ferray_core::Array<half::f16, D>,
        ) -> ::ferray_core::error::FerrayResult<
            ::ferray_core::Array<half::f16, D>,
        >
        where
            D: ::ferray_core::dimension::Dimension,
        {
            $crate::helpers::binary_f16_op(a, b, $f32_fn)
        }
    };
}

pub(crate) use binary_f16_fn;
pub(crate) use unary_f16_fn;

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

// ---------------------------------------------------------------------------
// In-place (`_into`) helpers — write results into a caller-owned buffer.
//
// These back the `_into` public ops that implement NumPy's `out=` parameter.
// On the hot path (same-shape contiguous inputs, contiguous output) there is
// zero allocation: the kernel runs straight from input slices into the output
// slice. Broadcasting or non-contiguous layouts are rejected with a clear
// error — users who need those should call the allocating variants instead.
// ---------------------------------------------------------------------------

/// Shared shape/layout validation for the in-place variants.
#[inline]
fn check_into_shapes<U: Element, D2: Dimension>(
    out: &Array<U, D2>,
    input_shape: &[usize],
    op_name: &str,
) -> FerrayResult<()> {
    if out.shape() != input_shape {
        return Err(FerrayError::shape_mismatch(format!(
            "{op_name}_into: out shape {:?} does not match input shape {:?}",
            out.shape(),
            input_shape
        )));
    }
    Ok(())
}

/// Apply a unary function elementwise, writing the result into `out` in
/// place. Both `input` and `out` must be contiguous and have the same shape.
///
/// Returns an error if shapes mismatch or either array is non-contiguous.
/// Memory-bound default: no parallel dispatch. Use [`unary_float_op_into_compute`]
/// for transcendentals.
#[inline]
pub fn unary_float_op_into<T, D>(
    input: &Array<T, D>,
    out: &mut Array<T, D>,
    op_name: &str,
    f: impl Fn(T) -> T,
) -> FerrayResult<()>
where
    T: Element + Copy,
    D: Dimension,
{
    check_into_shapes::<T, D>(out, input.shape(), op_name)?;
    let in_slice = input.as_slice().ok_or_else(|| {
        FerrayError::invalid_value(format!(
            "{op_name}_into: input must be contiguous (C-order); call {op_name}() for strided arrays"
        ))
    })?;
    let out_slice = out.as_slice_mut().ok_or_else(|| {
        FerrayError::invalid_value(format!(
            "{op_name}_into: out must be contiguous (C-order); call {op_name}() for strided output"
        ))
    })?;
    for (o, &x) in out_slice.iter_mut().zip(in_slice.iter()) {
        *o = f(x);
    }
    Ok(())
}

/// Compute-bound variant of [`unary_float_op_into`] — parallelizes the
/// contiguous write above `THRESHOLD_COMPUTE_BOUND` for transcendentals
/// (`exp_into`, `sin_into`, `log_into`, etc.).
#[inline]
pub fn unary_float_op_into_compute<T, D>(
    input: &Array<T, D>,
    out: &mut Array<T, D>,
    op_name: &str,
    f: impl Fn(T) -> T + Sync + Send,
) -> FerrayResult<()>
where
    T: Element + Copy,
    D: Dimension,
{
    check_into_shapes::<T, D>(out, input.shape(), op_name)?;
    let in_slice = input.as_slice().ok_or_else(|| {
        FerrayError::invalid_value(format!(
            "{op_name}_into: input must be contiguous (C-order); call {op_name}() for strided arrays"
        ))
    })?;
    let out_slice = out.as_slice_mut().ok_or_else(|| {
        FerrayError::invalid_value(format!(
            "{op_name}_into: out must be contiguous (C-order); call {op_name}() for strided output"
        ))
    })?;
    parallel_unary_fill_threshold(in_slice, out_slice, THRESHOLD_COMPUTE_BOUND, f);
    Ok(())
}

/// Apply a binary function elementwise, writing the result into `out` in
/// place. All three arrays must be contiguous and identically shaped.
///
/// Broadcasting is **not** supported here; use [`binary_elementwise_op`] if you
/// need it. The no-broadcast constraint is what makes this zero-allocation.
#[inline]
pub fn binary_elementwise_op_into<T, D>(
    a: &Array<T, D>,
    b: &Array<T, D>,
    out: &mut Array<T, D>,
    op_name: &str,
    f: impl Fn(T, T) -> T,
) -> FerrayResult<()>
where
    T: Element + Copy,
    D: Dimension,
{
    if a.shape() != b.shape() {
        return Err(FerrayError::shape_mismatch(format!(
            "{op_name}_into: input shapes {:?} and {:?} differ (broadcasting not supported; use {op_name}() instead)",
            a.shape(),
            b.shape()
        )));
    }
    check_into_shapes::<T, D>(out, a.shape(), op_name)?;
    let a_slice = a.as_slice().ok_or_else(|| {
        FerrayError::invalid_value(format!("{op_name}_into: a must be contiguous (C-order)"))
    })?;
    let b_slice = b.as_slice().ok_or_else(|| {
        FerrayError::invalid_value(format!("{op_name}_into: b must be contiguous (C-order)"))
    })?;
    let out_slice = out.as_slice_mut().ok_or_else(|| {
        FerrayError::invalid_value(format!("{op_name}_into: out must be contiguous (C-order)"))
    })?;
    for ((o, &x), &y) in out_slice.iter_mut().zip(a_slice.iter()).zip(b_slice.iter()) {
        *o = f(x, y);
    }
    Ok(())
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
        let r = binary_elementwise_op(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn binary_op_shape_mismatch() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![4.0, 5.0]).unwrap();
        assert!(binary_elementwise_op(&a, &b, |x, y| x + y).is_err());
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
    fn binary_elementwise_op_broadcasts_within_same_rank() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([3, 1]), vec![1.0, 2.0, 3.0]).unwrap();
        let b =
            Array::<f64, Ix2>::from_vec(Ix2::new([1, 4]), vec![10.0, 20.0, 30.0, 40.0]).unwrap();
        let r = binary_elementwise_op(&a, &b, |x, y| x + y).unwrap();
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
        let r = binary_mixed_op(&a, &b, |x, n| x * f64::from(1 << n)).unwrap();
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
        assert!(binary_elementwise_op(&a, &b, |x, y| x + y).is_err());
        assert!(binary_map_op(&a, &b, |x, y| x == y).is_err());
    }

    // -----------------------------------------------------------------------
    // Stride-aware non-contig path coverage (#385)
    //
    // Owned ferray arrays from from_vec are always C-contig, so we use
    // from_vec_f to construct a Fortran-order array — its `as_slice()`
    // returns None, exercising the contig_input materialization branch.
    // -----------------------------------------------------------------------

    #[test]
    fn contig_input_borrows_for_c_order() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        // C-contig owned arrays must take the borrow branch.
        let cow = contig_input(&a);
        match cow {
            std::borrow::Cow::Borrowed(_) => {}
            std::borrow::Cow::Owned(_) => panic!("expected borrow for C-contig array"),
        }
    }

    #[test]
    fn contig_input_materializes_for_fortran_order() {
        // Fortran-order owned array — as_slice() returns None for the
        // C-order request, so contig_input must materialize a fresh
        // row-major Vec. The materialized values must match the logical
        // C-order traversal of the input.
        let f = Array::<f64, Ix2>::from_vec_f(Ix2::new([2, 3]), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
            .unwrap();
        // Logical layout is [[1,2,3],[4,5,6]] regardless of memory order.
        assert_eq!(
            f.iter().copied().collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        // f.as_slice() must return None — that's the precondition we want
        // to exercise.
        assert!(f.as_slice().is_none());

        let cow = contig_input(&f);
        match cow {
            std::borrow::Cow::Owned(v) => {
                // The materialized buffer must be in C order.
                assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            }
            std::borrow::Cow::Borrowed(_) => {
                panic!("expected materialized owned Vec for Fortran-order input")
            }
        }
    }

    #[test]
    fn unary_float_op_works_on_fortran_layout() {
        let f = Array::<f64, Ix2>::from_vec_f(Ix2::new([2, 3]), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
            .unwrap();
        // sqrt over the logical [[1,2,3],[4,5,6]] → [[1,√2,√3],[2,√5,√6]]
        let r = unary_float_op(&f, f64::sqrt).unwrap();
        let s = r.as_slice().unwrap();
        let expected: Vec<f64> = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .map(|x| x.sqrt())
            .collect();
        assert_eq!(s, expected.as_slice());
    }

    #[test]
    fn unary_map_op_works_on_fortran_layout() {
        let f = Array::<f64, Ix2>::from_vec_f(Ix2::new([2, 3]), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
            .unwrap();
        // Mapping each element to its truncated integer.
        let r = unary_map_op(&f, |x| x as i32).unwrap();
        let s = r.as_slice().unwrap();
        assert_eq!(s, &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn binary_elementwise_op_handles_fortran_lhs() {
        // F-order LHS, C-order RHS, same shape — must produce a correct
        // C-order result without losing track of the logical positions.
        let a = Array::<f64, Ix2>::from_vec_f(Ix2::new([2, 3]), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
            .unwrap();
        let b =
            Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
                .unwrap();
        let r = binary_elementwise_op(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[11.0, 22.0, 33.0, 44.0, 55.0, 66.0]);
    }

    #[test]
    fn binary_elementwise_op_handles_two_fortran_inputs() {
        let a = Array::<f64, Ix2>::from_vec_f(Ix2::new([2, 3]), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
            .unwrap();
        let b = Array::<f64, Ix2>::from_vec_f(
            Ix2::new([2, 3]),
            vec![10.0, 40.0, 20.0, 50.0, 30.0, 60.0],
        )
        .unwrap();
        let r = binary_elementwise_op(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[11.0, 22.0, 33.0, 44.0, 55.0, 66.0]);
    }

    #[test]
    fn binary_map_op_handles_fortran_layout() {
        // Comparison ops should also work on F-order inputs.
        let a = Array::<f64, Ix2>::from_vec_f(Ix2::new([2, 3]), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
            .unwrap();
        let b = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 1.0, 4.0, 5.0, 5.0, 5.0])
            .unwrap();
        let r = binary_map_op(&a, &b, |x, y| x > y).unwrap();
        assert_eq!(
            r.iter().copied().collect::<Vec<_>>(),
            vec![false, true, false, false, false, true]
        );
    }

    #[test]
    fn try_simd_f64_unary_runs_on_fortran_layout() {
        // Previously this short-circuited to None on non-contig f64. Now
        // it materializes the input and runs the SIMD kernel against the
        // materialized buffer.
        fn double_kernel(src: &[f64], dst: &mut [f64]) {
            for (o, &x) in dst.iter_mut().zip(src.iter()) {
                *o = x * 2.0;
            }
        }
        let f = Array::<f64, Ix2>::from_vec_f(Ix2::new([2, 3]), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
            .unwrap();
        let r = try_simd_f64_unary(&f, double_kernel)
            .expect("try_simd_f64_unary should succeed for f64 input")
            .unwrap();
        assert_eq!(r.as_slice().unwrap(), &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    }
}
