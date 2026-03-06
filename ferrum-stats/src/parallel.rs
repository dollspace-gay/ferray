// ferrum-stats: Rayon threshold dispatch for reductions and sorting (REQ-19, REQ-20)

use rayon::prelude::*;

/// Threshold above which reductions use parallel tree-reduce.
pub const PARALLEL_REDUCTION_THRESHOLD: usize = 10_000;

/// Threshold above which sorting uses parallel merge sort.
pub const PARALLEL_SORT_THRESHOLD: usize = 100_000;

/// Perform a parallel sum on a slice of `Copy + Send + Sync` values.
///
/// Falls back to sequential sum when the slice length is below the threshold.
pub fn parallel_sum<T>(data: &[T], identity: T) -> T
where
    T: Copy + Send + Sync + std::ops::Add<Output = T>,
{
    if data.len() >= PARALLEL_REDUCTION_THRESHOLD {
        data.par_iter()
            .copied()
            .reduce(|| identity, |a, b| a + b)
    } else {
        data.iter().copied().fold(identity, |a, b| a + b)
    }
}

/// Perform a parallel product on a slice of `Copy + Send + Sync` values.
pub fn parallel_prod<T>(data: &[T], identity: T) -> T
where
    T: Copy + Send + Sync + std::ops::Mul<Output = T>,
{
    if data.len() >= PARALLEL_REDUCTION_THRESHOLD {
        data.par_iter()
            .copied()
            .reduce(|| identity, |a, b| a * b)
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
