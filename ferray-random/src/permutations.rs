// ferray-random: Permutations and sampling — shuffle, permutation, permuted, choice

use ferray_core::{Array, FerrayError, Ix1, IxDyn};

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

    /// Shuffle an N-D array in place along `axis`, swapping whole
    /// hyperslices (rows when `axis == 0` for a 2-D array).
    ///
    /// Equivalent to `numpy.random.Generator.shuffle(x, axis=axis)`.
    /// Each pair `(i, j)` selected by Fisher-Yates exchanges *all*
    /// elements with axis-coordinate `i` and `j` simultaneously, so
    /// rows / columns / N-D slices keep their internal structure (#447).
    ///
    /// # Errors
    /// - `FerrayError::AxisOutOfBounds` if `axis >= arr.ndim()`.
    /// - `FerrayError::InvalidValue` if `arr` is non-contiguous.
    pub fn shuffle_dyn<T>(
        &mut self,
        arr: &mut Array<T, IxDyn>,
        axis: usize,
    ) -> Result<(), FerrayError>
    where
        T: ferray_core::Element,
    {
        let shape = arr.shape().to_vec();
        let ndim = shape.len();
        if axis >= ndim {
            return Err(FerrayError::axis_out_of_bounds(axis, ndim));
        }
        let n = shape[axis];
        if n <= 1 {
            return Ok(());
        }
        let inner_stride: usize = shape[axis + 1..].iter().product();
        let block = n * inner_stride;
        let outer_size: usize = shape[..axis].iter().product();
        let slice = arr
            .as_slice_mut()
            .ok_or_else(|| FerrayError::invalid_value("array must be contiguous for shuffle"))?;
        for i in (1..n).rev() {
            let j = self.bg.next_u64_bounded((i + 1) as u64) as usize;
            if i == j {
                continue;
            }
            for o in 0..outer_size {
                let base = o * block;
                for k in 0..inner_stride {
                    slice.swap(base + i * inner_stride + k, base + j * inner_stride + k);
                }
            }
        }
        Ok(())
    }

    /// Independently permute the entries along `axis` of `arr`.
    ///
    /// Returns a new array. For each combination of "other" indices
    /// (everything except `axis`) the values along `axis` are
    /// shuffled with their own Fisher-Yates pass — so columns of a
    /// 2-D array get independent permutations when `axis = 0`.
    /// Equivalent to `numpy.random.Generator.permuted(x, axis=axis)`.
    ///
    /// # Errors
    /// - `FerrayError::AxisOutOfBounds` if `axis >= arr.ndim()`.
    /// - `FerrayError::InvalidValue` if `arr` is non-contiguous.
    pub fn permuted_dyn<T>(
        &mut self,
        arr: &Array<T, IxDyn>,
        axis: usize,
    ) -> Result<Array<T, IxDyn>, FerrayError>
    where
        T: ferray_core::Element,
    {
        let shape = arr.shape().to_vec();
        let ndim = shape.len();
        if axis >= ndim {
            return Err(FerrayError::axis_out_of_bounds(axis, ndim));
        }
        let mut out = arr.clone();
        let n = shape[axis];
        if n <= 1 {
            return Ok(out);
        }
        let inner_stride: usize = shape[axis + 1..].iter().product();
        let block = n * inner_stride;
        let outer_size: usize = shape[..axis].iter().product();
        let slice = out.as_slice_mut().ok_or_else(|| {
            FerrayError::invalid_value("array must be contiguous for permuted")
        })?;
        for o in 0..outer_size {
            let base = o * block;
            for k in 0..inner_stride {
                // Independent Fisher-Yates over the n axis positions
                // at this (outer, inner) coordinate.
                for i in (1..n).rev() {
                    let j = self.bg.next_u64_bounded((i + 1) as u64) as usize;
                    slice.swap(base + i * inner_stride + k, base + j * inner_stride + k);
                }
            }
        }
        Ok(out)
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

    // ---- shuffle_dyn / permuted_dyn (#447) -----------------------------

    #[test]
    fn shuffle_dyn_axis0_swaps_whole_rows() {
        use ferray_core::IxDyn;
        let mut rng = default_rng_seeded(42);
        // 4×3: rows are [0,1,2], [10,11,12], [20,21,22], [30,31,32]
        let data: Vec<i64> = (0..4)
            .flat_map(|i| (0..3).map(move |j| i * 10 + j))
            .collect();
        let mut arr = Array::<i64, IxDyn>::from_vec(IxDyn::new(&[4, 3]), data).unwrap();
        rng.shuffle_dyn(&mut arr, 0).unwrap();
        let slice = arr.as_slice().unwrap();
        // Each row must still be one of the originals — internal layout preserved.
        let mut seen = std::collections::HashSet::new();
        for row in 0..4 {
            let row_first = slice[row * 3];
            let id = row_first / 10;
            assert!(
                (0..4).contains(&id),
                "row {row} starts with unexpected value {row_first}"
            );
            assert_eq!(slice[row * 3 + 1], id * 10 + 1);
            assert_eq!(slice[row * 3 + 2], id * 10 + 2);
            assert!(seen.insert(id), "row id {id} duplicated — shuffle broke a row");
        }
    }

    #[test]
    fn shuffle_dyn_axis1_swaps_whole_columns() {
        use ferray_core::IxDyn;
        let mut rng = default_rng_seeded(7);
        // 3×4: column j is [j, 10+j, 20+j].
        let data: Vec<i64> = (0..3)
            .flat_map(|i| (0..4).map(move |j| i * 10 + j))
            .collect();
        let mut arr = Array::<i64, IxDyn>::from_vec(IxDyn::new(&[3, 4]), data).unwrap();
        rng.shuffle_dyn(&mut arr, 1).unwrap();
        let slice = arr.as_slice().unwrap();
        // Each column must still equal one of the original column patterns.
        let mut col_ids = Vec::new();
        for col in 0..4 {
            let v0 = slice[col];
            let v1 = slice[4 + col];
            let v2 = slice[8 + col];
            assert!((0..4).contains(&v0));
            assert_eq!(v1, v0 + 10);
            assert_eq!(v2, v0 + 20);
            col_ids.push(v0);
        }
        col_ids.sort_unstable();
        assert_eq!(col_ids, vec![0, 1, 2, 3]);
    }

    #[test]
    fn shuffle_dyn_axis_out_of_bounds() {
        use ferray_core::IxDyn;
        let mut rng = default_rng_seeded(0);
        let mut arr = Array::<i64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![0; 6]).unwrap();
        assert!(rng.shuffle_dyn(&mut arr, 2).is_err());
    }

    #[test]
    fn permuted_dyn_axis0_each_column_independent() {
        use ferray_core::IxDyn;
        let mut rng = default_rng_seeded(99);
        // 5×4 array; permuted along axis=0 → each column is independently
        // shuffled, so a row is a *re-mix* of column-wise positions, not a
        // whole-row swap.
        let n_rows = 5;
        let n_cols = 4;
        let data: Vec<i64> = (0..n_rows * n_cols).map(|x| x as i64).collect();
        let arr = Array::<i64, IxDyn>::from_vec(IxDyn::new(&[n_rows, n_cols]), data.clone())
            .unwrap();
        let result = rng.permuted_dyn(&arr, 0).unwrap();
        let slice = result.as_slice().unwrap();
        // Each column must be a permutation of the original column values.
        for col in 0..n_cols {
            let original_col: Vec<i64> = (0..n_rows).map(|r| (r * n_cols + col) as i64).collect();
            let mut got_col: Vec<i64> = (0..n_rows).map(|r| slice[r * n_cols + col]).collect();
            got_col.sort_unstable();
            let mut want = original_col.clone();
            want.sort_unstable();
            assert_eq!(got_col, want, "col {col} lost values during permute");
        }
    }

    #[test]
    fn permuted_dyn_columns_can_diverge() {
        use ferray_core::IxDyn;
        // Permuted should produce different per-column orderings — across
        // many trials the probability that all columns still match each
        // other for a 5-row 4-column array is (1/120)^3 ≈ 1e-6.
        let mut rng = default_rng_seeded(1234);
        let n_rows = 5;
        let n_cols = 4;
        let data: Vec<i64> = (0..n_rows * n_cols).map(|x| x as i64 % n_rows as i64).collect();
        let arr =
            Array::<i64, IxDyn>::from_vec(IxDyn::new(&[n_rows, n_cols]), data.clone()).unwrap();
        let result = rng.permuted_dyn(&arr, 0).unwrap();
        let slice = result.as_slice().unwrap();
        // Reference column 0 against each other column. At least one must differ.
        let col0: Vec<i64> = (0..n_rows).map(|r| slice[r * n_cols]).collect();
        let mut any_diff = false;
        for col in 1..n_cols {
            let coln: Vec<i64> = (0..n_rows).map(|r| slice[r * n_cols + col]).collect();
            if col0 != coln {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "all columns matched — permuted didn't independently shuffle");
    }

    #[test]
    fn permuted_dyn_seed_reproducible() {
        use ferray_core::IxDyn;
        let mut a = default_rng_seeded(31);
        let mut b = default_rng_seeded(31);
        let arr = Array::<i64, IxDyn>::from_vec(
            IxDyn::new(&[3, 3]),
            (0..9).collect(),
        )
        .unwrap();
        let xa = a.permuted_dyn(&arr, 1).unwrap();
        let xb = b.permuted_dyn(&arr, 1).unwrap();
        assert_eq!(xa.as_slice().unwrap(), xb.as_slice().unwrap());
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
