// ferrum-ufunc: Rayon parallel dispatch (REQ-21, REQ-22)
//
// Large arrays are parallelized via a ferrum-owned Rayon ThreadPool.
// Thresholds: ~100k for memory-bound ops, ~50k for compute-bound ops.

use rayon::ThreadPool;
use std::sync::OnceLock;

/// Memory-bound threshold (e.g., add, multiply): 100k elements.
pub const THRESHOLD_MEMORY_BOUND: usize = 100_000;

/// Compute-bound threshold (e.g., sin, exp): 50k elements.
pub const THRESHOLD_COMPUTE_BOUND: usize = 50_000;

/// The ferrum-owned Rayon thread pool, lazily initialized.
static POOL: OnceLock<ThreadPool> = OnceLock::new();

/// Get or create the ferrum thread pool.
pub fn pool() -> &'static ThreadPool {
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .thread_name(|idx| format!("ferrum-worker-{idx}"))
            .build()
            .expect("failed to create ferrum Rayon thread pool")
    })
}

/// Run a closure on the ferrum thread pool if the element count exceeds the
/// threshold, otherwise run it on the current thread.
#[inline]
pub fn maybe_parallel<F, R>(count: usize, threshold: usize, f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    if count >= threshold {
        pool().install(f)
    } else {
        f()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_creates_successfully() {
        let p = pool();
        assert!(p.current_num_threads() > 0);
    }

    #[test]
    fn maybe_parallel_below_threshold() {
        let result = maybe_parallel(10, THRESHOLD_MEMORY_BOUND, || 42);
        assert_eq!(result, 42);
    }

    #[test]
    fn maybe_parallel_above_threshold() {
        let result = maybe_parallel(200_000, THRESHOLD_MEMORY_BOUND, || 99);
        assert_eq!(result, 99);
    }
}
