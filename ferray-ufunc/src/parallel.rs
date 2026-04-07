// ferray-ufunc: Rayon parallel dispatch thresholds.
//
// ufunc kernels use rayon's **global** thread pool (not a separate ferray
// pool) because `ThreadPool::install` across pools costs several hundred
// microseconds per call — enough to wipe out any gain on small arrays. The
// global pool is normally already hot, so `par_chunks_mut` dispatch is
// cheap and rayon's work-stealing handles load balancing for free.
//
// The thresholds were picked empirically from `bench_parallel` on a
// modern x86_64 core (see `ferray-ufunc/examples/bench_parallel.rs`):
//
// * Memory-bound ops (add / sub / mul / div) saturate a single core's
//   memory bandwidth at a few hundred MB/s per lane, so parallelism only
//   helps once the problem size is large enough that the DRAM cost
//   dwarfs fixed dispatch overhead. That crossover sits around 1M
//   elements on a typical 4-channel desktop; below that, running serial
//   is strictly faster.
// * Compute-bound ops (sin / cos / exp / log / sqrt with scalar kernels)
//   are far heavier per element, so a lower crossover near 100k is fine.

/// Memory-bound threshold (e.g., add, multiply): 1M elements.
///
/// Below this the serial path's auto-vectorized loop is uncontested by
/// DRAM bandwidth and there is nothing for extra threads to do.
pub const THRESHOLD_MEMORY_BOUND: usize = 1_000_000;

/// Compute-bound threshold (e.g., sin, exp): 100k elements.
///
/// Scalar transcendental kernels cost 5–20ns per element, so 100k
/// elements is ~1ms of work — enough to amortize rayon dispatch.
pub const THRESHOLD_COMPUTE_BOUND: usize = 100_000;

// Compile-time sanity check that the constants stay in the expected
// relative ordering — catches accidental value swaps during tuning.
const _: () = {
    assert!(THRESHOLD_COMPUTE_BOUND < THRESHOLD_MEMORY_BOUND);
    assert!(THRESHOLD_COMPUTE_BOUND >= 10_000);
};
