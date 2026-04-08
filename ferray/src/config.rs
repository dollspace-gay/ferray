// ferray: Thread pool configuration and parallel thresholds (REQ-6, REQ-7, REQ-8)

use std::sync::OnceLock;

use ferray_core::error::{FerrayError, FerrayResult};
use rayon::ThreadPool;

/// Global ferray thread pool, initialized once.
static GLOBAL_POOL: OnceLock<ThreadPool> = OnceLock::new();

/// Pool cache for `with_num_threads`. Keyed by thread count.
/// Uses a simple mutex-protected HashMap since pool creation is rare.
static POOL_CACHE: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<usize, std::sync::Arc<ThreadPool>>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// Configure the global ferray Rayon thread pool.
///
/// This can only be called once — subsequent calls return an error.
/// If never called, Rayon's default thread count (usually number of CPUs) is used.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` wrapped with the rayon build
/// error if thread-pool creation fails, or
/// `"ferray thread pool already initialized"` if `set_num_threads`
/// has already been called (#218 — this used to return
/// `Result<_, String>` which forced callers to juggle two error
/// types alongside the rest of the ferray API).
pub fn set_num_threads(n: usize) -> FerrayResult<()> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .map_err(|e| FerrayError::invalid_value(format!("failed to create thread pool: {e}")))?;
    GLOBAL_POOL.set(pool).map_err(|_| {
        FerrayError::invalid_value("ferray thread pool already initialized")
    })
}

/// Execute a closure on a thread pool with `n` threads.
///
/// Pools are cached — calling this 100 times with the same `n` reuses
/// the same pool, avoiding the cost of repeated thread creation (AC-8).
///
/// # Errors
/// Returns a `FerrayError::InvalidValue` wrapping any rayon build
/// failure or lock-poisoning event (#218).
pub fn with_num_threads<F, R>(n: usize, f: F) -> FerrayResult<R>
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    let mut cache = POOL_CACHE.lock().map_err(|e| {
        FerrayError::invalid_value(format!("pool cache lock poisoned: {e}"))
    })?;
    let pool = if let Some(existing) = cache.get(&n) {
        existing.clone()
    } else {
        let new_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .map_err(|e| {
                FerrayError::invalid_value(format!(
                    "failed to create cached thread pool: {e}"
                ))
            })?;
        let arc = std::sync::Arc::new(new_pool);
        cache.insert(n, arc.clone());
        arc
    };
    drop(cache); // release lock before running user code
    Ok(pool.install(f))
}

/// Parallel threshold for elementwise (memory-bound) operations.
///
/// Arrays with fewer elements than this are processed sequentially.
pub const PARALLEL_THRESHOLD_ELEMENTWISE: usize = 100_000;

/// Parallel threshold for compute-bound operations (transcendentals, etc.).
pub const PARALLEL_THRESHOLD_COMPUTE: usize = 50_000;

/// Parallel threshold for reductions (sum, mean, etc.).
pub const PARALLEL_THRESHOLD_REDUCTION: usize = 10_000;

/// Parallel threshold for sorting operations.
pub const PARALLEL_THRESHOLD_SORT: usize = 100_000;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_num_threads_caches_pool_ac8() {
        // Call with_num_threads 100 times — should NOT create 100 pools
        for _ in 0..100 {
            let result = with_num_threads(2, || 42).unwrap();
            assert_eq!(result, 42);
        }
        // Verify only one pool exists for n=2
        let cache = POOL_CACHE.lock().unwrap();
        assert_eq!(cache.len(), 1);
        assert!(cache.contains_key(&2));
    }

    #[test]
    fn threshold_constants() {
        assert_eq!(PARALLEL_THRESHOLD_ELEMENTWISE, 100_000);
        assert_eq!(PARALLEL_THRESHOLD_COMPUTE, 50_000);
        assert_eq!(PARALLEL_THRESHOLD_REDUCTION, 10_000);
        assert_eq!(PARALLEL_THRESHOLD_SORT, 100_000);
    }
}
