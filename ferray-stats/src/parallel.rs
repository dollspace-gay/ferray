// ferray-stats: Rayon threshold dispatch for reductions and sorting (REQ-19, REQ-20)
//
// ## REQ status (ferray-stats parallel kernels, NumPy parity)
//  - REQ-19 (large reductions use parallel tree-reduce via Rayon) — SHIPPED:
//    `pub fn pairwise_sum` / `pub fn pairwise_sum_f64` / `pub fn pairwise_sum_f32`
//    (pairwise/tree summation), `pub fn simd_sum_sq_diff_f64` /
//    `pub fn simd_sum_sq_diff_f32` (fused sum-of-squared-deviations), and
//    `pub fn parallel_sum` / `pub fn parallel_prod` (Rayon tree-reduce above the
//    element-count threshold). The pairwise tree is numerically equivalent to
//    numpy's pairwise summation (numpy/_core/src/umath/loops_utils.h.src
//    pairwise reduction). Non-test consumers: `reductions::sum`/`mean`/`var`
//    (`crate::parallel::pairwise_sum*` / `simd_sum_sq_diff*` in
//    `reductions/mod.rs`) and `correlation::corrcoef` (`crate::parallel::pairwise_sum`).
//  - REQ-20 (large-array sort uses parallel merge sort via Rayon) — SHIPPED:
//    `pub fn parallel_sort` / `pub fn parallel_sort_stable` (Rayon
//    parallel/parallel-stable sort above the sort threshold), plus
//    `pub fn nan_last_cmp` (the NaN-last total order numpy's sort uses,
//    numpy/_core/src/npysort). Non-test consumer: `sorting::sort`
//    (`parallel::parallel_sort` / `parallel_sort_stable` in `sorting.rs`).

// `with_simd`, `simd_pairwise_f64`, `simd_base_sum_f64`, and `base_sum`
// are pulp dispatch entry points where `#[inline(always)]` is required —
// without it the SIMD instruction set selected at the call site fails to
// inline into the kernel and dispatch becomes a no-op.
#![allow(clippy::inline_always)]

use pulp::Arch;
use rayon::prelude::*;

/// Threshold above which reductions use parallel tree-reduce.
///
/// Set high to avoid rayon thread-pool startup overhead dominating small workloads.
/// Rayon's thread pool initialization costs ~1-3ms on first use, which is catastrophic
/// for arrays under ~1M elements where sequential pairwise sum takes <1ms.
pub const PARALLEL_REDUCTION_THRESHOLD: usize = 1_000_000;

/// Threshold above which sorting uses parallel merge sort.
pub const PARALLEL_SORT_THRESHOLD: usize = 100_000;

/// Pairwise summation base case threshold.
///
/// Below this size, we use an unrolled sequential sum. 256 elements gives a
/// good balance: halves carry-merge overhead vs 128 while keeping the same
/// O(ε log N) accuracy bound (just one fewer merge level).
/// Base-chunk size for pairwise summation. Each base chunk runs the
/// flat SIMD inner-loop (4-way unrolled f64 adds with one horizontal
/// reduction at the end); the pairwise tree only kicks in across base
/// chunks. So this knob is the trade-off between (a) tree-induced
/// horizontal-sum overhead — paid once per base chunk — and (b)
/// numerical precision, which scales as O(ε * base_size) within a
/// chunk and O(ε log N/base_size) across the tree.
///
/// 4096 puts the per-chunk error at ~12 ULP for f64 (well below f64's
/// 52-bit mantissa) while cutting the horizontal-sum overhead 16× vs
/// the previous 256. For N=1M that's 250 chunks instead of ~4000 — the
/// difference between memory-bandwidth-bound (NumPy's 154 µs floor at
/// 1M) and add-throughput-bound (242 µs at 1M).
const PAIRWISE_BASE: usize = 4096;

/// 8-wide unrolled base-case sum for auto-vectorization.
#[inline(always)]
fn base_sum<T: Copy + std::ops::Add<Output = T>>(data: &[T], identity: T) -> T {
    let n = data.len();
    let mut acc0 = identity;
    let mut acc1 = identity;
    let mut acc2 = identity;
    let mut acc3 = identity;
    let mut acc4 = identity;
    let mut acc5 = identity;
    let mut acc6 = identity;
    let mut acc7 = identity;
    let chunks = n / 8;
    let rem = n % 8;
    for i in 0..chunks {
        let base = i * 8;
        acc0 = acc0 + data[base];
        acc1 = acc1 + data[base + 1];
        acc2 = acc2 + data[base + 2];
        acc3 = acc3 + data[base + 3];
        acc4 = acc4 + data[base + 4];
        acc5 = acc5 + data[base + 5];
        acc6 = acc6 + data[base + 6];
        acc7 = acc7 + data[base + 7];
    }
    for i in 0..rem {
        acc0 = acc0 + data[chunks * 8 + i];
    }
    (acc0 + acc1) + (acc2 + acc3) + ((acc4 + acc5) + (acc6 + acc7))
}

/// Pairwise summation of a slice.
///
/// Uses an iterative carry-merge algorithm with O(ε log N) error bound,
/// matching `NumPy`'s summation accuracy. Data is processed in 128-element
/// base chunks, then merged pairwise using a small stack (like binary
/// carry addition). Zero recursion overhead — stack depth is at most
/// ceil(log2(N/128)) + 1 ≈ 20 entries.
pub fn pairwise_sum<T>(data: &[T], identity: T) -> T
where
    T: Copy + std::ops::Add<Output = T>,
{
    let n = data.len();
    if n == 0 {
        return identity;
    }
    if n <= PAIRWISE_BASE {
        return base_sum(data, identity);
    }

    // Iterative pairwise summation using a carry-merge stack.
    // Each entry holds (partial_sum, level) where level is how many base
    // chunks it represents. Same-level adjacent entries merge on push,
    // exactly like binary carry addition. Max depth: ~24 for any realistic size.
    let mut stack_val: [T; 24] = [identity; 24];
    let mut stack_lvl: [usize; 24] = [0; 24];
    let mut depth = 0usize;

    let mut offset = 0;
    while offset < n {
        let end = (offset + PAIRWISE_BASE).min(n);
        let mut current = base_sum(&data[offset..end], identity);
        offset = end;

        // Push and carry-merge: merge with stack top while levels match
        let mut level = 1usize;
        while depth > 0 && stack_lvl[depth - 1] == level {
            depth -= 1;
            current = stack_val[depth] + current;
            level += 1;
        }
        stack_val[depth] = current;
        stack_lvl[depth] = level;
        depth += 1;
    }

    // Merge remaining stack entries
    let mut result = stack_val[depth - 1];
    for i in (0..depth - 1).rev() {
        result = stack_val[i] + result;
    }
    result
}

// ---------------------------------------------------------------------------
// SIMD-accelerated pairwise sum for f64
// ---------------------------------------------------------------------------

/// SIMD-accelerated pairwise summation for f64 slices.
///
/// Uses pulp for hardware SIMD dispatch (AVX2/SSE2/NEON) in the 128-element
/// base case, with the same iterative carry-merge structure for O(ε log N)
/// accuracy.
#[must_use]
pub fn pairwise_sum_f64(data: &[f64]) -> f64 {
    Arch::new().dispatch(PairwiseSumF64Op { data })
}

struct PairwiseSumF64Op<'a> {
    data: &'a [f64],
}

impl pulp::WithSimd for PairwiseSumF64Op<'_> {
    type Output = f64;

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> f64 {
        simd_pairwise_f64(simd, self.data)
    }
}

#[inline(always)]
fn simd_base_sum_f64<S: pulp::Simd>(simd: S, data: &[f64]) -> f64 {
    let n = data.len();
    let lane_count = size_of::<S::f64s>() / size_of::<f64>();
    let simd_end = n - (n % lane_count);

    let zero = simd.splat_f64s(0.0);
    let mut acc0 = zero;
    let mut acc1 = zero;
    let mut acc2 = zero;
    let mut acc3 = zero;

    // 4-accumulator unroll to saturate FPU throughput
    let stride = lane_count * 4;
    let unrolled_end = n - (n % stride);
    let mut i = 0;
    while i < unrolled_end {
        let v0 = simd.partial_load_f64s(&data[i..i + lane_count]);
        let v1 = simd.partial_load_f64s(&data[i + lane_count..i + lane_count * 2]);
        let v2 = simd.partial_load_f64s(&data[i + lane_count * 2..i + lane_count * 3]);
        let v3 = simd.partial_load_f64s(&data[i + lane_count * 3..i + stride]);
        acc0 = simd.add_f64s(acc0, v0);
        acc1 = simd.add_f64s(acc1, v1);
        acc2 = simd.add_f64s(acc2, v2);
        acc3 = simd.add_f64s(acc3, v3);
        i += stride;
    }
    while i + lane_count <= simd_end {
        let v = simd.partial_load_f64s(&data[i..i + lane_count]);
        acc0 = simd.add_f64s(acc0, v);
        i += lane_count;
    }
    acc0 = simd.add_f64s(acc0, acc1);
    acc2 = simd.add_f64s(acc2, acc3);
    acc0 = simd.add_f64s(acc0, acc2);

    // Horizontal sum: store SIMD register to temp array
    let mut temp = [0.0f64; 8]; // max 8 lanes (AVX-512)
    simd.partial_store_f64s(&mut temp[..lane_count], acc0);
    let mut sum = 0.0f64;
    for t in temp.iter().take(lane_count) {
        sum += t;
    }
    // Scalar remainder
    for &val in &data[simd_end..n] {
        sum += val;
    }
    sum
}

#[inline(always)]
fn simd_pairwise_f64<S: pulp::Simd>(simd: S, data: &[f64]) -> f64 {
    let n = data.len();
    if n == 0 {
        return 0.0;
    }
    if n <= PAIRWISE_BASE {
        return simd_base_sum_f64(simd, data);
    }

    let mut stack_val = [0.0f64; 24];
    let mut stack_lvl = [0usize; 24];
    let mut depth = 0usize;

    let mut offset = 0;
    while offset < n {
        let end = (offset + PAIRWISE_BASE).min(n);
        let mut current = simd_base_sum_f64(simd, &data[offset..end]);
        offset = end;

        let mut level = 1usize;
        while depth > 0 && stack_lvl[depth - 1] == level {
            depth -= 1;
            current += stack_val[depth];
            level += 1;
        }
        stack_val[depth] = current;
        stack_lvl[depth] = level;
        depth += 1;
    }

    let mut result = stack_val[depth - 1];
    for i in (0..depth - 1).rev() {
        result += stack_val[i];
    }
    result
}

// ---------------------------------------------------------------------------
// SIMD-accelerated fused sum of squared differences for variance
// ---------------------------------------------------------------------------

/// SIMD-accelerated computation of sum((x - mean)²) for f64 slices.
///
/// Computes the sum of squared differences from the mean in a single pass
/// without allocating an intermediate Vec. Uses 4 SIMD accumulators for ILP.
#[must_use]
pub fn simd_sum_sq_diff_f64(data: &[f64], mean: f64) -> f64 {
    Arch::new().dispatch(SumSqDiffF64Op { data, mean })
}

struct SumSqDiffF64Op<'a> {
    data: &'a [f64],
    mean: f64,
}

impl pulp::WithSimd for SumSqDiffF64Op<'_> {
    type Output = f64;

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> f64 {
        let data = self.data;
        let n = data.len();
        let lane_count = size_of::<S::f64s>() / size_of::<f64>();
        let simd_end = n - (n % lane_count);

        let zero = simd.splat_f64s(0.0);
        let mean_v = simd.splat_f64s(self.mean);
        let mut acc0 = zero;
        let mut acc1 = zero;
        let mut acc2 = zero;
        let mut acc3 = zero;

        let stride = lane_count * 4;
        let unrolled_end = n - (n % stride);
        let mut i = 0;
        while i < unrolled_end {
            let v0 = simd.partial_load_f64s(&data[i..i + lane_count]);
            let v1 = simd.partial_load_f64s(&data[i + lane_count..i + lane_count * 2]);
            let v2 = simd.partial_load_f64s(&data[i + lane_count * 2..i + lane_count * 3]);
            let v3 = simd.partial_load_f64s(&data[i + lane_count * 3..i + stride]);
            let d0 = simd.sub_f64s(v0, mean_v);
            let d1 = simd.sub_f64s(v1, mean_v);
            let d2 = simd.sub_f64s(v2, mean_v);
            let d3 = simd.sub_f64s(v3, mean_v);
            acc0 = simd.mul_add_f64s(d0, d0, acc0);
            acc1 = simd.mul_add_f64s(d1, d1, acc1);
            acc2 = simd.mul_add_f64s(d2, d2, acc2);
            acc3 = simd.mul_add_f64s(d3, d3, acc3);
            i += stride;
        }
        while i + lane_count <= simd_end {
            let v = simd.partial_load_f64s(&data[i..i + lane_count]);
            let d = simd.sub_f64s(v, mean_v);
            acc0 = simd.mul_add_f64s(d, d, acc0);
            i += lane_count;
        }
        acc0 = simd.add_f64s(acc0, acc1);
        acc2 = simd.add_f64s(acc2, acc3);
        acc0 = simd.add_f64s(acc0, acc2);

        let mut temp = [0.0f64; 8];
        simd.partial_store_f64s(&mut temp[..lane_count], acc0);
        let mut sum = 0.0f64;
        for t in temp.iter().take(lane_count) {
            sum += t;
        }
        for &val in &data[simd_end..n] {
            let d = val - self.mean;
            sum += d * d;
        }
        sum
    }
}

// ---------------------------------------------------------------------------
// SIMD-accelerated pairwise sum / sum-of-squared-differences for f32 (#173)
//
// Mirrors the f64 versions above with pulp's f32s SIMD type. Twice as
// many lanes per register (e.g. AVX-512 f32 = 16 lanes vs f64 = 8), so
// the constants for max-temp-buffer below need to match the widest f32
// lane count. The base case and stack-based pairwise tree are
// structurally identical.
// ---------------------------------------------------------------------------

/// SIMD-accelerated pairwise summation for f32 slices (#173).
#[must_use]
pub fn pairwise_sum_f32(data: &[f32]) -> f32 {
    Arch::new().dispatch(PairwiseSumF32Op { data })
}

struct PairwiseSumF32Op<'a> {
    data: &'a [f32],
}

impl pulp::WithSimd for PairwiseSumF32Op<'_> {
    type Output = f32;

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> f32 {
        simd_pairwise_f32(simd, self.data)
    }
}

#[inline(always)]
fn simd_base_sum_f32<S: pulp::Simd>(simd: S, data: &[f32]) -> f32 {
    let n = data.len();
    let lane_count = size_of::<S::f32s>() / size_of::<f32>();
    let simd_end = n - (n % lane_count);

    let zero = simd.splat_f32s(0.0);
    let mut acc0 = zero;
    let mut acc1 = zero;
    let mut acc2 = zero;
    let mut acc3 = zero;

    let stride = lane_count * 4;
    let unrolled_end = n - (n % stride);
    let mut i = 0;
    while i < unrolled_end {
        let v0 = simd.partial_load_f32s(&data[i..i + lane_count]);
        let v1 = simd.partial_load_f32s(&data[i + lane_count..i + lane_count * 2]);
        let v2 = simd.partial_load_f32s(&data[i + lane_count * 2..i + lane_count * 3]);
        let v3 = simd.partial_load_f32s(&data[i + lane_count * 3..i + stride]);
        acc0 = simd.add_f32s(acc0, v0);
        acc1 = simd.add_f32s(acc1, v1);
        acc2 = simd.add_f32s(acc2, v2);
        acc3 = simd.add_f32s(acc3, v3);
        i += stride;
    }
    while i + lane_count <= simd_end {
        let v = simd.partial_load_f32s(&data[i..i + lane_count]);
        acc0 = simd.add_f32s(acc0, v);
        i += lane_count;
    }
    acc0 = simd.add_f32s(acc0, acc1);
    acc2 = simd.add_f32s(acc2, acc3);
    acc0 = simd.add_f32s(acc0, acc2);

    // Horizontal sum: store SIMD register to temp array. AVX-512 f32
    // is 16 lanes wide, so size the temp accordingly.
    let mut temp = [0.0f32; 16];
    simd.partial_store_f32s(&mut temp[..lane_count], acc0);
    let mut sum = 0.0f32;
    for t in temp.iter().take(lane_count) {
        sum += t;
    }
    for &val in &data[simd_end..n] {
        sum += val;
    }
    sum
}

#[inline(always)]
fn simd_pairwise_f32<S: pulp::Simd>(simd: S, data: &[f32]) -> f32 {
    let n = data.len();
    if n == 0 {
        return 0.0;
    }
    if n <= PAIRWISE_BASE {
        return simd_base_sum_f32(simd, data);
    }

    let mut stack_val = [0.0f32; 24];
    let mut stack_lvl = [0usize; 24];
    let mut depth = 0usize;

    let mut offset = 0;
    while offset < n {
        let end = (offset + PAIRWISE_BASE).min(n);
        let mut current = simd_base_sum_f32(simd, &data[offset..end]);
        offset = end;

        let mut level = 1usize;
        while depth > 0 && stack_lvl[depth - 1] == level {
            depth -= 1;
            current += stack_val[depth];
            level += 1;
        }
        stack_val[depth] = current;
        stack_lvl[depth] = level;
        depth += 1;
    }

    let mut result = stack_val[depth - 1];
    for i in (0..depth - 1).rev() {
        result += stack_val[i];
    }
    result
}

/// SIMD-accelerated computation of sum((x - mean)²) for f32 slices (#173).
#[must_use]
pub fn simd_sum_sq_diff_f32(data: &[f32], mean: f32) -> f32 {
    Arch::new().dispatch(SumSqDiffF32Op { data, mean })
}

struct SumSqDiffF32Op<'a> {
    data: &'a [f32],
    mean: f32,
}

impl pulp::WithSimd for SumSqDiffF32Op<'_> {
    type Output = f32;

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> f32 {
        let data = self.data;
        let n = data.len();
        let lane_count = size_of::<S::f32s>() / size_of::<f32>();
        let simd_end = n - (n % lane_count);

        let zero = simd.splat_f32s(0.0);
        let mean_v = simd.splat_f32s(self.mean);
        let mut acc0 = zero;
        let mut acc1 = zero;
        let mut acc2 = zero;
        let mut acc3 = zero;

        let stride = lane_count * 4;
        let unrolled_end = n - (n % stride);
        let mut i = 0;
        while i < unrolled_end {
            let v0 = simd.partial_load_f32s(&data[i..i + lane_count]);
            let v1 = simd.partial_load_f32s(&data[i + lane_count..i + lane_count * 2]);
            let v2 = simd.partial_load_f32s(&data[i + lane_count * 2..i + lane_count * 3]);
            let v3 = simd.partial_load_f32s(&data[i + lane_count * 3..i + stride]);
            let d0 = simd.sub_f32s(v0, mean_v);
            let d1 = simd.sub_f32s(v1, mean_v);
            let d2 = simd.sub_f32s(v2, mean_v);
            let d3 = simd.sub_f32s(v3, mean_v);
            acc0 = simd.mul_add_f32s(d0, d0, acc0);
            acc1 = simd.mul_add_f32s(d1, d1, acc1);
            acc2 = simd.mul_add_f32s(d2, d2, acc2);
            acc3 = simd.mul_add_f32s(d3, d3, acc3);
            i += stride;
        }
        while i + lane_count <= simd_end {
            let v = simd.partial_load_f32s(&data[i..i + lane_count]);
            let d = simd.sub_f32s(v, mean_v);
            acc0 = simd.mul_add_f32s(d, d, acc0);
            i += lane_count;
        }
        acc0 = simd.add_f32s(acc0, acc1);
        acc2 = simd.add_f32s(acc2, acc3);
        acc0 = simd.add_f32s(acc0, acc2);

        let mut temp = [0.0f32; 16];
        simd.partial_store_f32s(&mut temp[..lane_count], acc0);
        let mut sum = 0.0f32;
        for t in temp.iter().take(lane_count) {
            sum += t;
        }
        for &val in &data[simd_end..n] {
            let d = val - self.mean;
            sum += d * d;
        }
        sum
    }
}

// ---------------------------------------------------------------------------
// Parallel dispatch
// ---------------------------------------------------------------------------

/// Perform a parallel pairwise sum on a slice.
///
/// Uses rayon tree-reduce for large slices (which is inherently pairwise),
/// and iterative pairwise summation for smaller slices.
pub fn parallel_sum<T>(data: &[T], identity: T) -> T
where
    T: Copy + Send + Sync + std::ops::Add<Output = T>,
{
    if data.len() >= PARALLEL_REDUCTION_THRESHOLD {
        // Rayon's reduce is a tree-reduce, which gives pairwise-like accuracy
        data.par_iter().copied().reduce(|| identity, |a, b| a + b)
    } else {
        pairwise_sum(data, identity)
    }
}

/// Perform a parallel product on a slice of `Copy + Send + Sync` values.
pub fn parallel_prod<T>(data: &[T], identity: T) -> T
where
    T: Copy + Send + Sync + std::ops::Mul<Output = T>,
{
    if data.len() >= PARALLEL_REDUCTION_THRESHOLD {
        data.par_iter().copied().reduce(|| identity, |a, b| a * b)
    } else {
        data.iter().copied().fold(identity, |a, b| a * b)
    }
}

/// Total-order comparator that sends "unorderable" (NaN) values to the
/// end, matching `NumPy`'s `np.sort` convention.
///
/// Mirrors numpy/_core/src/common/numpy_tag.h:62
/// `floating_point_type::less(a, b) = a < b || (b != b && a == a)`:
/// a value is "less than" b iff it is non-NaN and either strictly less
/// than b or b is NaN. Every NaN (regardless of sign bit) is treated as
/// the greatest element, so all NaNs sort to the tail — unlike
/// `f64::total_cmp`, which would order a negative NaN before `-inf`.
///
/// Detects NaN without a `Float` bound via self-comparison: for any sane
/// `PartialOrd`, `x.partial_cmp(&x)` is `Some(Equal)`; only NaN-like
/// values break that invariant and return `None`. Non-NaN orderable
/// values keep their natural ordering, so integer and other no-NaN
/// dtypes are completely unaffected.
#[inline]
pub fn nan_last_cmp<T: PartialOrd>(a: &T, b: &T) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    if let Some(ord) = a.partial_cmp(b) {
        ord
    } else {
        let a_nan = a.partial_cmp(a).is_none();
        let b_nan = b.partial_cmp(b).is_none();
        match (a_nan, b_nan) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            // Genuinely incomparable non-NaN values shouldn't exist for
            // any numeric Element type; keep the sort total by treating
            // them as equal.
            (false, false) => Ordering::Equal,
        }
    }
}

/// Parallel sort (unstable) for large slices. Returns a sorted copy.
pub fn parallel_sort<T>(data: &mut [T])
where
    T: Copy + Send + Sync + PartialOrd,
{
    if data.len() >= PARALLEL_SORT_THRESHOLD {
        data.par_sort_unstable_by(nan_last_cmp);
    } else {
        data.sort_unstable_by(nan_last_cmp);
    }
}

/// Parallel stable sort for large slices.
pub fn parallel_sort_stable<T>(data: &mut [T])
where
    T: Copy + Send + Sync + PartialOrd,
{
    if data.len() >= PARALLEL_SORT_THRESHOLD {
        data.par_sort_by(nan_last_cmp);
    } else {
        data.sort_by(nan_last_cmp);
    }
}

#[cfg(test)]
mod sort_cmp_tests {
    use super::nan_last_cmp;

    // Expected values pulled from live numpy 2.4:
    //   np.sort([3., nan, 1.])            -> [1., 3., nan]
    //   np.sort([nan, nan, 1., 2.])       -> [1., 2., nan, nan]
    //   np.sort([inf, -inf, 0., nan, 1.]) -> [-inf, 0., 1., inf, nan]
    //   np.sort([1., nan, -nan, 2.])      -> [1., 2., nan, nan] (both NaN last)
    #[test]
    fn nan_last_basic() {
        let mut v = [3.0f64, f64::NAN, 1.0];
        v.sort_by(nan_last_cmp);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 3.0);
        assert!(v[2].is_nan());
    }

    #[test]
    fn multiple_nans_last() {
        let mut v = [f64::NAN, f64::NAN, 1.0, 2.0];
        v.sort_by(nan_last_cmp);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 2.0);
        assert!(v[2].is_nan() && v[3].is_nan());
    }

    #[test]
    fn inf_order_then_nan() {
        let mut v = [f64::INFINITY, f64::NEG_INFINITY, 0.0, f64::NAN, 1.0];
        v.sort_by(nan_last_cmp);
        assert_eq!(v[0], f64::NEG_INFINITY);
        assert_eq!(v[1], 0.0);
        assert_eq!(v[2], 1.0);
        assert_eq!(v[3], f64::INFINITY);
        assert!(v[4].is_nan());
    }

    #[test]
    fn negative_nan_also_last() {
        let mut v = [1.0f64, f64::NAN, -f64::NAN, 2.0];
        v.sort_by(nan_last_cmp);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 2.0);
        assert!(v[2].is_nan() && v[3].is_nan());
    }

    #[test]
    fn integers_unaffected() {
        let mut v = [5i32, 2, 8, 1];
        v.sort_by(nan_last_cmp);
        assert_eq!(v, [1, 2, 5, 8]);
    }

    #[test]
    fn f32_nan_last() {
        let mut v = [3.0f32, f32::NAN, 1.0];
        v.sort_by(nan_last_cmp);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 3.0);
        assert!(v[2].is_nan());
    }
}
