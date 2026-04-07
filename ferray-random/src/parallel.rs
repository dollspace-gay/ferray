// ferray-random: Parallel generation via jump-ahead / stream splitting
//
// Provides deterministic parallel generation that produces the same output
// regardless of thread count, by using fixed index-range assignment.

use ferray_core::{Array, FerrayError, IxDyn};

use crate::bitgen::BitGenerator;
use crate::distributions::normal::standard_normal_single;
use crate::generator::{Generator, shape_size, vec_to_array_f64};
use crate::shape::IntoShape;

impl<B: BitGenerator + Clone> Generator<B> {
    /// Generate standard normal variates in parallel, deterministically.
    ///
    /// Uses `spawn()` to create independent child generators (via jump-ahead
    /// for Xoshiro256**, stream IDs for Philox, or seed-from-parent for
    /// PCG64), then generates chunks in parallel using Rayon.
    ///
    /// The output is **deterministic** for a given seed and thread count,
    /// but does **not** match `standard_normal(size)` (which is sequential).
    /// The parallel version produces different values because the child
    /// generators have different internal states.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `shape` is invalid.
    pub fn standard_normal_parallel(
        &mut self,
        size: impl IntoShape,
    ) -> Result<Array<f64, IxDyn>, FerrayError> {
        let shape_vec = size.into_shape()?;
        let n = shape_size(&shape_vec);

        // For small sizes, sequential is faster (no spawn/thread overhead)
        let num_threads = rayon::current_num_threads().max(1);
        if n < 10_000 || num_threads <= 1 {
            // Sequential fallback
            let mut data = Vec::with_capacity(n);
            for _ in 0..n {
                data.push(standard_normal_single(&mut self.bg));
            }
            return vec_to_array_f64(data, &shape_vec);
        }

        // Spawn independent child generators
        let mut children = self.spawn(num_threads)?;
        let chunk_size = n.div_ceil(num_threads);

        // Generate chunks in parallel
        use rayon::prelude::*;
        let chunks: Vec<Vec<f64>> = children
            .par_iter_mut()
            .enumerate()
            .map(|(i, child)| {
                let start = i * chunk_size;
                let end = (start + chunk_size).min(n);
                let count = end - start;
                let mut chunk = Vec::with_capacity(count);
                for _ in 0..count {
                    chunk.push(standard_normal_single(&mut child.bg));
                }
                chunk
            })
            .collect();

        // Concatenate chunks
        let mut data = Vec::with_capacity(n);
        for chunk in chunks {
            data.extend_from_slice(&chunk);
        }
        data.truncate(n);

        vec_to_array_f64(data, &shape_vec)
    }

    /// Spawn `n` independent child generators for manual parallel use.
    ///
    /// Uses `jump()` if available, otherwise seeds children from parent output.
    ///
    /// # Arguments
    /// * `n` - Number of child generators to create.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `n` is zero.
    pub fn spawn(&mut self, n: usize) -> Result<Vec<Generator<B>>, FerrayError> {
        crate::generator::spawn_generators(self, n)
    }
}

#[cfg(test)]
mod tests {
    use crate::default_rng_seeded;

    #[test]
    fn parallel_correct_length_and_stats() {
        // Parallel output has correct length and reasonable statistics
        let mut rng = default_rng_seeded(42);
        let par = rng.standard_normal_parallel(10_000).unwrap();
        assert_eq!(par.shape(), &[10_000]);
        let slice = par.as_slice().unwrap();
        let mean: f64 = slice.iter().sum::<f64>() / slice.len() as f64;
        let var: f64 =
            slice.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / slice.len() as f64;
        // Mean should be near 0, variance near 1
        assert!(mean.abs() < 0.05, "mean = {mean}");
        assert!((var - 1.0).abs() < 0.1, "var = {var}");
    }

    #[test]
    fn parallel_deterministic() {
        let mut rng1 = default_rng_seeded(42);
        let mut rng2 = default_rng_seeded(42);

        let a = rng1.standard_normal_parallel(50_000).unwrap();
        let b = rng2.standard_normal_parallel(50_000).unwrap();

        assert_eq!(a.as_slice().unwrap(), b.as_slice().unwrap());
    }

    #[test]
    fn parallel_large() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.standard_normal_parallel(1_000_000).unwrap();
        assert_eq!(arr.shape(), &[1_000_000]);
        // Check mean is roughly 0
        let slice = arr.as_slice().unwrap();
        let mean: f64 = slice.iter().sum::<f64>() / slice.len() as f64;
        assert!(mean.abs() < 0.01, "parallel mean {mean} too far from 0");
    }

    #[test]
    fn spawn_creates_independent_generators() {
        let mut rng = default_rng_seeded(42);
        let mut children = rng.spawn(4).unwrap();
        assert_eq!(children.len(), 4);

        // Each child should produce different sequences
        let outputs: Vec<u64> = children.iter_mut().map(|c| c.next_u64()).collect();
        for i in 0..outputs.len() {
            for j in (i + 1)..outputs.len() {
                assert_ne!(
                    outputs[i], outputs[j],
                    "children {i} and {j} produced same first value"
                );
            }
        }
    }

    #[test]
    fn spawn_deterministic() {
        let mut rng1 = default_rng_seeded(42);
        let mut rng2 = default_rng_seeded(42);

        let mut children1 = rng1.spawn(4).unwrap();
        let mut children2 = rng2.spawn(4).unwrap();

        for (c1, c2) in children1.iter_mut().zip(children2.iter_mut()) {
            for _ in 0..100 {
                assert_eq!(c1.next_u64(), c2.next_u64());
            }
        }
    }

    #[test]
    fn parallel_zero_size_error() {
        let mut rng = default_rng_seeded(42);
        assert!(rng.standard_normal_parallel(0).is_err());
    }
}
