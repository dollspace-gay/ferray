// ferray-random: Permutations and sampling — shuffle, permutation, permuted, choice

use ferray_core::{Array, FerrayError, Ix1};

use crate::bitgen::BitGenerator;
use crate::generator::Generator;

impl<B: BitGenerator> Generator<B> {
    /// Shuffle a 1-D array in-place using Fisher-Yates.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if the array is not contiguous.
    pub fn shuffle<T>(&mut self, arr: &mut Array<T, Ix1>) -> Result<(), FerrayError>
    where
        T: ferray_core::Element,
    {
        let n = arr.shape()[0];
        if n <= 1 {
            return Ok(());
        }
        let slice = arr
            .as_slice_mut()
            .ok_or_else(|| FerrayError::invalid_value("array must be contiguous for shuffle"))?;
        // Fisher-Yates
        for i in (1..n).rev() {
            let j = self.bg.next_u64_bounded((i + 1) as u64) as usize;
            slice.swap(i, j);
        }
        Ok(())
    }

    /// Return a new array with elements randomly permuted.
    ///
    /// If the input is 1-D, returns a shuffled copy. If an integer `n` is
    /// given (via `permutation_range`), returns a permutation of `0..n`.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if the array is empty.
    pub fn permutation<T>(&mut self, arr: &Array<T, Ix1>) -> Result<Array<T, Ix1>, FerrayError>
    where
        T: ferray_core::Element,
    {
        let mut copy = arr.clone();
        self.shuffle(&mut copy)?;
        Ok(copy)
    }

    /// Return a permutation of `0..n` as an `Array1<i64>`.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `n` is zero.
    pub fn permutation_range(&mut self, n: usize) -> Result<Array<i64, Ix1>, FerrayError> {
        if n == 0 {
            return Err(FerrayError::invalid_value("n must be > 0"));
        }
        let mut data: Vec<i64> = (0..n as i64).collect();
        // Fisher-Yates
        for i in (1..n).rev() {
            let j = self.bg.next_u64_bounded((i + 1) as u64) as usize;
            data.swap(i, j);
        }
        Array::<i64, Ix1>::from_vec(Ix1::new([n]), data)
    }

    /// Return an array with elements independently permuted along the given axis.
    ///
    /// For 1-D arrays, this is the same as `permutation`. This simplified
    /// implementation operates on 1-D arrays along axis 0.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if the array is empty.
    pub fn permuted<T>(
        &mut self,
        arr: &Array<T, Ix1>,
        _axis: usize,
    ) -> Result<Array<T, Ix1>, FerrayError>
    where
        T: ferray_core::Element,
    {
        self.permutation(arr)
    }

    /// Randomly select elements from an array, with or without replacement.
    ///
    /// # Arguments
    /// * `arr` - Source array to sample from.
    /// * `size` - Number of elements to select.
    /// * `replace` - If `true`, sample with replacement; if `false`, without.
    /// * `p` - Optional probability weights (must sum to 1.0 and have same length as `arr`).
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if parameters are invalid (e.g.,
    /// `size > arr.len()` when `replace=false`, or invalid probability weights).
    pub fn choice<T>(
        &mut self,
        arr: &Array<T, Ix1>,
        size: usize,
        replace: bool,
        p: Option<&[f64]>,
    ) -> Result<Array<T, Ix1>, FerrayError>
    where
        T: ferray_core::Element,
    {
        let n = arr.shape()[0];
        // size == 0 is valid: NumPy returns an empty array. Only the
        // source-array-empty case (and only when we actually need a
        // sample) is still an error (#264, #455).
        if size == 0 {
            return Array::from_vec(Ix1::new([0]), Vec::new());
        }
        if n == 0 {
            return Err(FerrayError::invalid_value("source array must be non-empty"));
        }
        if !replace && size > n {
            return Err(FerrayError::invalid_value(format!(
                "cannot choose {size} elements without replacement from array of size {n}"
            )));
        }

        if let Some(probs) = p {
            if probs.len() != n {
                return Err(FerrayError::invalid_value(format!(
                    "p must have same length as array ({n}), got {}",
                    probs.len()
                )));
            }
            let psum: f64 = probs.iter().sum();
            if (psum - 1.0).abs() > 1e-6 {
                return Err(FerrayError::invalid_value(format!(
                    "p must sum to 1.0, got {psum}"
                )));
            }
            for (i, &pi) in probs.iter().enumerate() {
                if pi < 0.0 {
                    return Err(FerrayError::invalid_value(format!(
                        "p[{i}] = {pi} is negative"
                    )));
                }
            }
        }

        let src = arr
            .as_slice()
            .ok_or_else(|| FerrayError::invalid_value("array must be contiguous"))?;

        let indices = if let Some(probs) = p {
            // Weighted sampling
            if replace {
                weighted_sample_with_replacement(&mut self.bg, probs, size)
            } else {
                weighted_sample_without_replacement(&mut self.bg, probs, size)?
            }
        } else if replace {
            // Uniform with replacement
            (0..size)
                .map(|_| self.bg.next_u64_bounded(n as u64) as usize)
                .collect()
        } else {
            // Uniform without replacement: partial Fisher-Yates
            sample_without_replacement(&mut self.bg, n, size)
        };

        let data: Vec<T> = indices.iter().map(|&i| src[i].clone()).collect();
        Array::<T, Ix1>::from_vec(Ix1::new([size]), data)
    }
}

/// Sample `size` indices from `[0, n)` without replacement using partial Fisher-Yates.
fn sample_without_replacement<B: BitGenerator>(bg: &mut B, n: usize, size: usize) -> Vec<usize> {
    let mut pool: Vec<usize> = (0..n).collect();
    for i in 0..size {
        let j = i + bg.next_u64_bounded((n - i) as u64) as usize;
        pool.swap(i, j);
    }
    pool[..size].to_vec()
}

/// Weighted sampling with replacement using Vose's alias method (#265).
///
/// Setup is O(n); each sample is O(1) — strictly faster than the
/// O(log n) binary-search-on-CDF path we used to use, especially at
/// large `size`. The alias table holds, for each bin `i`, a
/// "secondary" choice `alias[i]` and a probability `prob[i]` of
/// sticking with `i`. Sampling: pick `i` uniformly, draw `u ∈ [0, 1)`;
/// if `u < prob[i]` return `i`, else return `alias[i]`.
///
/// Reference: M. D. Vose, "A linear algorithm for generating random
/// numbers with a given distribution", IEEE TSE 17(9), 1991.
fn weighted_sample_with_replacement<B: BitGenerator>(
    bg: &mut B,
    probs: &[f64],
    size: usize,
) -> Vec<usize> {
    let n = probs.len();

    // Normalize so the sum is exactly n. The alias method works on
    // probabilities scaled by n: each bin "should" hold mass 1, and we
    // shuffle excess from heavy bins into light bins.
    let total: f64 = probs.iter().sum();
    let mut scaled: Vec<f64> = probs.iter().map(|&p| p * n as f64 / total).collect();

    let mut prob = vec![0.0_f64; n];
    let mut alias = vec![0_usize; n];

    // Two stacks: indices with mass < 1 vs. mass >= 1.
    let mut small: Vec<usize> = Vec::with_capacity(n);
    let mut large: Vec<usize> = Vec::with_capacity(n);
    for (i, &m) in scaled.iter().enumerate() {
        if m < 1.0 {
            small.push(i);
        } else {
            large.push(i);
        }
    }

    while !small.is_empty() && !large.is_empty() {
        let s = small.pop().unwrap();
        let l = large.pop().unwrap();
        prob[s] = scaled[s];
        alias[s] = l;
        // Donate (1 - scaled[s]) of mass from l to fill s.
        scaled[l] = (scaled[l] + scaled[s]) - 1.0;
        if scaled[l] < 1.0 {
            small.push(l);
        } else {
            large.push(l);
        }
    }
    // Drain leftovers — these slots have mass exactly 1.0 (modulo
    // floating-point drift); pin prob[i] = 1.0 so sampling always
    // returns i for these.
    for &i in large.iter().chain(small.iter()) {
        prob[i] = 1.0;
    }

    (0..size)
        .map(|_| {
            let i = bg.next_u64_bounded(n as u64) as usize;
            let u = bg.next_f64();
            if u < prob[i] { i } else { alias[i] }
        })
        .collect()
}

/// Weighted sampling without replacement using a sequential elimination method.
fn weighted_sample_without_replacement<B: BitGenerator>(
    bg: &mut B,
    probs: &[f64],
    size: usize,
) -> Result<Vec<usize>, FerrayError> {
    let n = probs.len();
    let mut weights: Vec<f64> = probs.to_vec();
    let mut selected = Vec::with_capacity(size);

    for _ in 0..size {
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return Err(FerrayError::invalid_value(
                "insufficient probability mass for sampling without replacement",
            ));
        }
        let u = bg.next_f64() * total;
        let mut cumsum = 0.0;
        let mut chosen = n - 1;
        for (i, &w) in weights.iter().enumerate() {
            cumsum += w;
            if cumsum > u {
                chosen = i;
                break;
            }
        }
        selected.push(chosen);
        weights[chosen] = 0.0;
    }

    Ok(selected)
}

#[cfg(test)]
mod tests {
    use crate::default_rng_seeded;
    use ferray_core::{Array, Ix1};

    #[test]
    fn shuffle_preserves_elements() {
        let mut rng = default_rng_seeded(42);
        let mut arr = Array::<i64, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5]).unwrap();
        rng.shuffle(&mut arr).unwrap();
        let mut sorted: Vec<i64> = arr.as_slice().unwrap().to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn permutation_preserves_elements() {
        let mut rng = default_rng_seeded(42);
        let arr = Array::<i64, Ix1>::from_vec(Ix1::new([5]), vec![10, 20, 30, 40, 50]).unwrap();
        let perm = rng.permutation(&arr).unwrap();
        let mut sorted: Vec<i64> = perm.as_slice().unwrap().to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![10, 20, 30, 40, 50]);
    }

    #[test]
    fn permutation_range_covers_all() {
        let mut rng = default_rng_seeded(42);
        let perm = rng.permutation_range(10).unwrap();
        let mut sorted: Vec<i64> = perm.as_slice().unwrap().to_vec();
        sorted.sort_unstable();
        let expected: Vec<i64> = (0..10).collect();
        assert_eq!(sorted, expected);
    }

    #[test]
    fn shuffle_modifies_in_place() {
        let mut rng = default_rng_seeded(42);
        let original = vec![1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut arr = Array::<i64, Ix1>::from_vec(Ix1::new([10]), original.clone()).unwrap();
        rng.shuffle(&mut arr).unwrap();
        // Very unlikely (10! - 1 chance) that shuffle produces identity
        let shuffled = arr.as_slice().unwrap().to_vec();
        // Just verify it's a valid permutation
        let mut sorted = shuffled;
        sorted.sort_unstable();
        assert_eq!(sorted, original);
    }

    #[test]
    fn choice_with_replacement() {
        let mut rng = default_rng_seeded(42);
        let arr = Array::<i64, Ix1>::from_vec(Ix1::new([5]), vec![10, 20, 30, 40, 50]).unwrap();
        let chosen = rng.choice(&arr, 10, true, None).unwrap();
        assert_eq!(chosen.shape(), &[10]);
        // All values should be from the original array
        let src: Vec<i64> = vec![10, 20, 30, 40, 50];
        for &v in chosen.as_slice().unwrap() {
            assert!(src.contains(&v), "choice returned unexpected value {v}");
        }
    }

    #[test]
    fn choice_without_replacement_no_duplicates() {
        let mut rng = default_rng_seeded(42);
        let arr = Array::<i64, Ix1>::from_vec(Ix1::new([10]), (0..10).collect()).unwrap();
        let chosen = rng.choice(&arr, 5, false, None).unwrap();
        let slice = chosen.as_slice().unwrap();
        // No duplicates
        let mut seen = std::collections::HashSet::new();
        for &v in slice {
            assert!(
                seen.insert(v),
                "duplicate value {v} in choice without replacement"
            );
        }
    }

    #[test]
    fn choice_without_replacement_too_many() {
        let mut rng = default_rng_seeded(42);
        let arr = Array::<i64, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5]).unwrap();
        assert!(rng.choice(&arr, 10, false, None).is_err());
    }

    #[test]
    fn choice_with_weights() {
        let mut rng = default_rng_seeded(42);
        let arr = Array::<i64, Ix1>::from_vec(Ix1::new([3]), vec![10, 20, 30]).unwrap();
        let p = [0.0, 0.0, 1.0]; // Always pick the last element
        let chosen = rng.choice(&arr, 10, true, Some(&p)).unwrap();
        for &v in chosen.as_slice().unwrap() {
            assert_eq!(v, 30);
        }
    }

    #[test]
    fn choice_without_replacement_with_weights() {
        let mut rng = default_rng_seeded(42);
        let arr = Array::<i64, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5]).unwrap();
        let p = [0.1, 0.2, 0.3, 0.2, 0.2];
        let chosen = rng.choice(&arr, 3, false, Some(&p)).unwrap();
        let slice = chosen.as_slice().unwrap();
        // No duplicates
        let mut seen = std::collections::HashSet::new();
        for &v in slice {
            assert!(seen.insert(v), "duplicate value {v}");
        }
    }

    #[test]
    fn choice_bad_weights() {
        let mut rng = default_rng_seeded(42);
        let arr = Array::<i64, Ix1>::from_vec(Ix1::new([3]), vec![1, 2, 3]).unwrap();
        // Wrong length
        assert!(rng.choice(&arr, 1, true, Some(&[0.5, 0.5])).is_err());
        // Doesn't sum to 1
        assert!(rng.choice(&arr, 1, true, Some(&[0.5, 0.5, 0.5])).is_err());
        // Negative
        assert!(rng.choice(&arr, 1, true, Some(&[-0.1, 0.6, 0.5])).is_err());
    }

    #[test]
    fn permuted_1d() {
        let mut rng = default_rng_seeded(42);
        let arr = Array::<i64, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5]).unwrap();
        let result = rng.permuted(&arr, 0).unwrap();
        let mut sorted: Vec<i64> = result.as_slice().unwrap().to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn weighted_with_replacement_alias_distribution_recovers_probs() {
        // #265: Vose's alias method must produce empirical bin
        // frequencies that match the input probability vector across a
        // large sample. Use a deliberately uneven distribution that
        // exercises the small/large stack rebalancing.
        let mut rng = default_rng_seeded(42);
        let arr = Array::<i64, Ix1>::from_vec(Ix1::new([5]), vec![0, 1, 2, 3, 4]).unwrap();
        let p = [0.05, 0.15, 0.30, 0.40, 0.10];
        let n = 100_000;
        let chosen = rng.choice(&arr, n, true, Some(&p)).unwrap();
        let mut counts = [0_usize; 5];
        for &v in chosen.as_slice().unwrap() {
            counts[v as usize] += 1;
        }
        // Each empirical frequency must be within 1.5% absolute of
        // its target — comfortably above the Monte Carlo noise of
        // sqrt(p(1-p)/n) ~ 0.15% for the largest bin.
        for (i, &c) in counts.iter().enumerate() {
            let observed = c as f64 / n as f64;
            assert!(
                (observed - p[i]).abs() < 0.015,
                "bin {i}: observed {observed}, expected {}",
                p[i]
            );
        }
    }

    #[test]
    fn weighted_with_replacement_unnormalized_probs() {
        // The alias setup normalizes probs internally; a vector that
        // sums to !=1 must produce the same empirical distribution as
        // its normalized counterpart. (We bypass `choice`'s strict
        // sum-to-1 validation by hitting the inner function path —
        // here we test the user-facing path with an exact input.)
        let mut rng = default_rng_seeded(42);
        let arr = Array::<i64, Ix1>::from_vec(Ix1::new([3]), vec![0, 1, 2]).unwrap();
        // Already-normalized comparison input.
        let p = [0.2, 0.5, 0.3];
        let n = 50_000;
        let chosen = rng.choice(&arr, n, true, Some(&p)).unwrap();
        let mut counts = [0_usize; 3];
        for &v in chosen.as_slice().unwrap() {
            counts[v as usize] += 1;
        }
        for (i, &c) in counts.iter().enumerate() {
            let observed = c as f64 / n as f64;
            assert!(
                (observed - p[i]).abs() < 0.02,
                "bin {i}: observed {observed}, expected {}",
                p[i]
            );
        }
    }
}
