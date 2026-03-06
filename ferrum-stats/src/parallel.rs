// ferrum-stats: Rayon threshold dispatch for reductions and sorting (REQ-19, REQ-20)

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
/// Below this size, we use an unrolled sequential sum. Matches NumPy's approach.
const PAIRWISE_BASE: usize = 128;

/// Pairwise summation of a slice.
///
/// Uses a recursive divide-and-conquer approach with O(ε log N) error bound,
/// matching NumPy's summation algorithm. This is significantly more accurate than
/// naive sequential summation (O(Nε)) for large arrays.
pub fn pairwise_sum<T>(data: &[T], identity: T) -> T
where
    T: Copy + std::ops::Add<Output = T>,
{
    let n = data.len();
    if n == 0 {
        return identity;
    }
    if n <= PAIRWISE_BASE {
        // Base case: unrolled sequential sum (8-wide) for vectorization
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
    } else {
        // Recursive case: split in half
        let mid = n / 2;
        pairwise_sum(&data[..mid], identity) + pairwise_sum(&data[mid..], identity)
    }
}

/// Perform a parallel pairwise sum on a slice.
///
/// Uses rayon tree-reduce for large slices (which is inherently pairwise),
/// and recursive pairwise summation for smaller slices.
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

/// Parallel sort (unstable) for large slices. Returns a sorted copy.
pub fn parallel_sort<T>(data: &mut [T])
where
    T: Copy + Send + Sync + PartialOrd,
{
    if data.len() >= PARALLEL_SORT_THRESHOLD {
        data.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    } else {
        data.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }
}

/// Parallel stable sort for large slices.
pub fn parallel_sort_stable<T>(data: &mut [T])
where
    T: Copy + Send + Sync + PartialOrd,
{
    if data.len() >= PARALLEL_SORT_THRESHOLD {
        data.par_sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    } else {
        data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }
}
