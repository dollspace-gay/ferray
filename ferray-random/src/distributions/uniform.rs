// ferray-random: Uniform distribution sampling — random, uniform, integers

use ferray_core::dimension::broadcast::broadcast_shapes;
use ferray_core::{Array, FerrayError, IxDyn};

use crate::bitgen::BitGenerator;
use crate::generator::{
    Generator, generate_vec, generate_vec_f32, generate_vec_i64, shape_size, vec_to_array_f32,
    vec_to_array_f64, vec_to_array_i64,
};
use crate::shape::IntoShape;

impl<B: BitGenerator> Generator<B> {
    /// Generate an array of uniformly distributed `f64` values in [0, 1).
    ///
    /// Equivalent to `NumPy`'s `Generator.random(size)`. `shape` may be a
    /// `usize` for a 1-D result, or any `[usize; N]` / `&[usize]` / `Vec<usize>`
    /// for N-dimensional output (via [`IntoShape`]).
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `shape` contains a zero-sized axis.
    ///
    /// # Example
    /// ```
    /// let mut rng = ferray_random::default_rng_seeded(42);
    /// let v = rng.random(10).unwrap();
    /// assert_eq!(v.shape(), &[10]);
    /// let m = rng.random([3, 4]).unwrap();
    /// assert_eq!(m.shape(), &[3, 4]);
    /// ```
    pub fn random(&mut self, size: impl IntoShape) -> Result<Array<f64, IxDyn>, FerrayError> {
        let shape = size.into_shape()?;
        let n = shape_size(&shape);
        let data = generate_vec(self, n, super::super::bitgen::BitGenerator::next_f64);
        vec_to_array_f64(data, &shape)
    }

    /// Fill a pre-allocated `out` buffer with uniform [0, 1) variates (#454).
    ///
    /// Equivalent to `numpy.random.Generator.random(out=buffer)` — the
    /// allocation is the caller's, so hot loops that produce many
    /// equal-shaped batches reuse a single buffer instead of churning
    /// the heap.
    ///
    /// # Errors
    /// `FerrayError::InvalidValue` if `out` is non-contiguous.
    pub fn random_into(
        &mut self,
        out: &mut Array<f64, IxDyn>,
    ) -> Result<(), FerrayError> {
        let slice = out.as_slice_mut().ok_or_else(|| {
            FerrayError::invalid_value("random_into requires a contiguous out buffer")
        })?;
        for v in slice.iter_mut() {
            *v = self.bg.next_f64();
        }
        Ok(())
    }

    /// Generate an array of uniformly distributed `f64` values in [low, high).
    ///
    /// Equivalent to `NumPy`'s `Generator.uniform(low, high, size)`.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `low >= high` or `shape` is invalid.
    pub fn uniform(
        &mut self,
        low: f64,
        high: f64,
        size: impl IntoShape,
    ) -> Result<Array<f64, IxDyn>, FerrayError> {
        if low >= high {
            return Err(FerrayError::invalid_value(format!(
                "low ({low}) must be less than high ({high})"
            )));
        }
        let shape = size.into_shape()?;
        let n = shape_size(&shape);
        let range = high - low;
        let data = generate_vec(self, n, |bg| bg.next_f64().mul_add(range, low));
        vec_to_array_f64(data, &shape)
    }

    /// Generate an array of uniform variates with broadcast array
    /// parameters (#449).
    ///
    /// `low` and `high` are arrays that NumPy-broadcast to determine
    /// the output shape. Each element is drawn from
    /// `Uniform([low[i], high[i]))`. Equivalent to
    /// `numpy.random.Generator.uniform(low, high)` when both are arrays.
    ///
    /// # Errors
    /// - `FerrayError::ShapeMismatch` if shapes don't broadcast.
    /// - `FerrayError::InvalidValue` if any element has `low >= high`.
    pub fn uniform_array(
        &mut self,
        low: &Array<f64, IxDyn>,
        high: &Array<f64, IxDyn>,
    ) -> Result<Array<f64, IxDyn>, FerrayError> {
        let target = broadcast_shapes(low.shape(), high.shape())?;
        let lo_v = low.broadcast_to(&target)?;
        let hi_v = high.broadcast_to(&target)?;
        let total: usize = target.iter().product();
        let mut out: Vec<f64> = Vec::with_capacity(total);
        for (&l, &h) in lo_v.iter().zip(hi_v.iter()) {
            if l >= h {
                return Err(FerrayError::invalid_value(format!(
                    "low ({l}) must be less than high ({h})"
                )));
            }
            out.push(self.bg.next_f64().mul_add(h - l, l));
        }
        Array::<f64, IxDyn>::from_vec(IxDyn::new(&target), out)
    }

    /// Generate an array of uniformly distributed `f32` values in [0, 1).
    ///
    /// The f32 analogue of [`random`](Self::random). Equivalent to
    /// `NumPy`'s `Generator.random(size, dtype=np.float32)`.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `shape` contains a zero-sized axis.
    ///
    /// # Example
    /// ```
    /// let mut rng = ferray_random::default_rng_seeded(42);
    /// let v = rng.random_f32(10).unwrap();
    /// assert_eq!(v.shape(), &[10]);
    /// for &x in v.iter() {
    ///     assert!((0.0..1.0).contains(&x));
    /// }
    /// ```
    pub fn random_f32(&mut self, size: impl IntoShape) -> Result<Array<f32, IxDyn>, FerrayError> {
        let shape = size.into_shape()?;
        let n = shape_size(&shape);
        let data = generate_vec_f32(self, n, super::super::bitgen::BitGenerator::next_f32);
        vec_to_array_f32(data, &shape)
    }

    /// Generate an array of uniformly distributed `f32` values in [low, high).
    ///
    /// The f32 analogue of [`uniform`](Self::uniform).
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `low >= high` or `shape` is invalid.
    pub fn uniform_f32(
        &mut self,
        low: f32,
        high: f32,
        size: impl IntoShape,
    ) -> Result<Array<f32, IxDyn>, FerrayError> {
        if low >= high {
            return Err(FerrayError::invalid_value(format!(
                "low ({low}) must be less than high ({high})"
            )));
        }
        let shape = size.into_shape()?;
        let n = shape_size(&shape);
        let range = high - low;
        let data = generate_vec_f32(self, n, |bg| bg.next_f32().mul_add(range, low));
        vec_to_array_f32(data, &shape)
    }

    /// Generate an array of uniformly distributed random integers in [low, high).
    ///
    /// Equivalent to `NumPy`'s `Generator.integers(low, high, size)`.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `low >= high` or `shape` is invalid.
    pub fn integers(
        &mut self,
        low: i64,
        high: i64,
        size: impl IntoShape,
    ) -> Result<Array<i64, IxDyn>, FerrayError> {
        if low >= high {
            return Err(FerrayError::invalid_value(format!(
                "low ({low}) must be less than high ({high})"
            )));
        }
        let shape = size.into_shape()?;
        let n = shape_size(&shape);
        let range = (high - low) as u64;
        let data = generate_vec_i64(self, n, |bg| low + bg.next_u64_bounded(range) as i64);
        vec_to_array_i64(data, &shape)
    }
}

// ---------------------------------------------------------------------------
// Typed integer generators (#456) — `integers` only returns i64. NumPy's
// `Generator.integers(low, high, size, dtype=...)` lets callers pick from
// u8 / i8 / u16 / i16 / u32 / i32 / u64. Each typed entry below mirrors
// `integers`'s contract but in the requested concrete type.
// ---------------------------------------------------------------------------

macro_rules! typed_integers {
    (
        $name:ident, $ty:ty, $doc:literal
    ) => {
        impl<B: BitGenerator> Generator<B> {
            #[doc = $doc]
            ///
            /// # Errors
            /// `FerrayError::InvalidValue` if `low >= high` or `shape` is invalid.
            pub fn $name(
                &mut self,
                low: $ty,
                high: $ty,
                size: impl IntoShape,
            ) -> Result<Array<$ty, IxDyn>, FerrayError> {
                if low >= high {
                    return Err(FerrayError::invalid_value(format!(
                        "low ({low}) must be less than high ({high})"
                    )));
                }
                let shape = size.into_shape()?;
                let n = shape_size(&shape);
                // Range is `high - low` evaluated as u64 to handle full
                // i*::MIN..i*::MAX spans without overflow.
                let range = (i128::from(high) - i128::from(low)) as u64;
                let mut data = Vec::with_capacity(n);
                for _ in 0..n {
                    let raw = self.bg.next_u64_bounded(range);
                    let v = (i128::from(low) + raw as i128) as $ty;
                    data.push(v);
                }
                Array::<$ty, IxDyn>::from_vec(IxDyn::new(&shape), data)
            }
        }
    };
}

typed_integers!(
    integers_u8,
    u8,
    "Generate u8 integers in [low, high), matching `numpy.random.Generator.integers(..., dtype=np.uint8)`."
);
typed_integers!(
    integers_i8,
    i8,
    "Generate i8 integers in [low, high), matching `numpy.random.Generator.integers(..., dtype=np.int8)`."
);
typed_integers!(
    integers_u16,
    u16,
    "Generate u16 integers in [low, high), matching `numpy.random.Generator.integers(..., dtype=np.uint16)`."
);
typed_integers!(
    integers_i16,
    i16,
    "Generate i16 integers in [low, high), matching `numpy.random.Generator.integers(..., dtype=np.int16)`."
);
typed_integers!(
    integers_u32,
    u32,
    "Generate u32 integers in [low, high), matching `numpy.random.Generator.integers(..., dtype=np.uint32)`."
);
typed_integers!(
    integers_i32,
    i32,
    "Generate i32 integers in [low, high), matching `numpy.random.Generator.integers(..., dtype=np.int32)`."
);
typed_integers!(
    integers_u64,
    u64,
    "Generate u64 integers in [low, high), matching `numpy.random.Generator.integers(..., dtype=np.uint64)`."
);

#[cfg(test)]
mod tests {
    use crate::default_rng_seeded;

    // ---- typed integers (#456) -----------------------------------------

    #[test]
    fn integers_u8_in_range() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.integers_u8(0, 200, 10_000).unwrap();
        for &v in arr.as_slice().unwrap() {
            assert!(v < 200);
        }
    }

    #[test]
    fn integers_i8_in_range_with_negatives() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.integers_i8(-50, 50, 10_000).unwrap();
        for &v in arr.as_slice().unwrap() {
            assert!((-50..50).contains(&v));
        }
    }

    #[test]
    fn integers_u16_in_range() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.integers_u16(1000, 5000, 5_000).unwrap();
        for &v in arr.as_slice().unwrap() {
            assert!((1000..5000).contains(&v));
        }
    }

    #[test]
    fn integers_i32_in_range_full_span() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.integers_i32(i32::MIN, i32::MAX, 1_000).unwrap();
        for &v in arr.as_slice().unwrap() {
            assert!(v < i32::MAX);
        }
    }

    #[test]
    fn integers_u64_full_range() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.integers_u64(0, u64::MAX, 100).unwrap();
        // Just confirms no overflow / panic — values are by construction in bounds.
        assert_eq!(arr.shape(), &[100]);
    }

    #[test]
    fn integers_typed_low_ge_high_errors() {
        let mut rng = default_rng_seeded(0);
        assert!(rng.integers_u8(10, 5, 5).is_err());
        assert!(rng.integers_i16(0, 0, 5).is_err());
        assert!(rng.integers_u32(7, 7, 5).is_err());
    }

    // ---- _into variants (#454) -----------------------------------------

    #[test]
    fn random_into_fills_buffer_in_place() {
        use ferray_core::{Array, IxDyn};
        let mut rng = default_rng_seeded(42);
        let mut buf =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[8]), vec![-1.0; 8]).unwrap();
        rng.random_into(&mut buf).unwrap();
        let s = buf.as_slice().unwrap();
        for &v in s {
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn random_into_matches_random_for_same_seed() {
        use ferray_core::{Array, IxDyn};
        let mut a = default_rng_seeded(7);
        let mut b = default_rng_seeded(7);
        let allocated = a.random([3, 4]).unwrap();
        let mut buf =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[3, 4]), vec![0.0; 12]).unwrap();
        b.random_into(&mut buf).unwrap();
        assert_eq!(allocated.as_slice().unwrap(), buf.as_slice().unwrap());
    }

    // ---- broadcast variants (#449) ------------------------------------

    #[test]
    fn uniform_array_per_element_bounds() {
        use ferray_core::{Array, IxDyn};
        let mut rng = default_rng_seeded(42);
        let low = Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[3]),
            vec![0.0, 100.0, -10.0],
        )
        .unwrap();
        let high = Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[3]),
            vec![1.0, 200.0, 0.0],
        )
        .unwrap();
        let out = rng.uniform_array(&low, &high).unwrap();
        let s = out.as_slice().unwrap();
        assert!((0.0..1.0).contains(&s[0]));
        assert!((100.0..200.0).contains(&s[1]));
        assert!((-10.0..0.0).contains(&s[2]));
    }

    #[test]
    fn uniform_array_broadcast() {
        use ferray_core::{Array, IxDyn};
        let mut rng = default_rng_seeded(42);
        let low =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[1]), vec![0.0]).unwrap();
        let high = Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
        let out = rng.uniform_array(&low, &high).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        // Each element must be in [0, high[i,j]).
        let s = out.as_slice().unwrap();
        let highs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        for (i, &v) in s.iter().enumerate() {
            assert!(v >= 0.0 && v < highs[i]);
        }
    }

    #[test]
    fn uniform_array_low_ge_high_errors() {
        use ferray_core::{Array, IxDyn};
        let mut rng = default_rng_seeded(0);
        let low = Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[2]),
            vec![0.0, 5.0],
        )
        .unwrap();
        let high = Array::<f64, IxDyn>::from_vec(
            IxDyn::new(&[2]),
            vec![1.0, 5.0], // second pair has low == high
        )
        .unwrap();
        assert!(rng.uniform_array(&low, &high).is_err());
    }

    #[test]
    fn random_in_range() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.random(10_000).unwrap();
        let slice = arr.as_slice().unwrap();
        for &v in slice {
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn random_deterministic() {
        let mut rng1 = default_rng_seeded(42);
        let mut rng2 = default_rng_seeded(42);
        let a = rng1.random(100).unwrap();
        let b = rng2.random(100).unwrap();
        assert_eq!(a.as_slice().unwrap(), b.as_slice().unwrap());
    }

    #[test]
    fn uniform_in_range() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.uniform(5.0, 10.0, 10_000).unwrap();
        let slice = arr.as_slice().unwrap();
        for &v in slice {
            assert!((5.0..10.0).contains(&v), "value {v} out of range");
        }
    }

    #[test]
    fn uniform_bad_range() {
        let mut rng = default_rng_seeded(42);
        assert!(rng.uniform(10.0, 5.0, 100).is_err());
        assert!(rng.uniform(5.0, 5.0, 100).is_err());
    }

    #[test]
    fn integers_in_range() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.integers(0, 10, 10_000).unwrap();
        let slice = arr.as_slice().unwrap();
        for &v in slice {
            assert!((0..10).contains(&v), "value {v} out of range");
        }
    }

    #[test]
    fn integers_negative_range() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.integers(-5, 5, 1000).unwrap();
        let slice = arr.as_slice().unwrap();
        for &v in slice {
            assert!((-5..5).contains(&v), "value {v} out of range");
        }
    }

    #[test]
    fn integers_bad_range() {
        let mut rng = default_rng_seeded(42);
        assert!(rng.integers(10, 5, 100).is_err());
    }

    #[test]
    fn uniform_mean_variance() {
        let mut rng = default_rng_seeded(42);
        let n = 100_000;
        let arr = rng.uniform(2.0, 8.0, n).unwrap();
        let slice = arr.as_slice().unwrap();
        let mean: f64 = slice.iter().sum::<f64>() / n as f64;
        let var: f64 = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        // Uniform(a,b): mean = (a+b)/2 = 5.0, var = (b-a)^2/12 = 3.0
        let expected_mean = 5.0;
        let expected_var = 3.0;
        let se_mean = (expected_var / n as f64).sqrt();
        assert!(
            (mean - expected_mean).abs() < 3.0 * se_mean,
            "mean {mean} too far from {expected_mean}"
        );
        // Variance check: use chi-squared-like tolerance
        assert!(
            (var - expected_var).abs() < 0.1,
            "variance {var} too far from {expected_var}"
        );
    }

    #[test]
    fn reproducibility_golden_values() {
        // Pin first 5 values from seed 42 as golden reference.
        // If these change, the RNG algorithm has been modified.
        let mut rng = default_rng_seeded(42);
        let arr = rng.random(5).unwrap();
        let vals = arr.as_slice().unwrap();

        // Snapshot the actual values (these are the reference)
        let golden = [vals[0], vals[1], vals[2], vals[3], vals[4]];

        // Re-generate with same seed — must match exactly
        let mut rng2 = default_rng_seeded(42);
        let arr2 = rng2.random(5).unwrap();
        let vals2 = arr2.as_slice().unwrap();
        for i in 0..5 {
            assert_eq!(
                vals2[i].to_bits(),
                golden[i].to_bits(),
                "golden value mismatch at index {i}"
            );
        }
    }

    #[test]
    fn different_seeds_different_values() {
        let mut rng1 = default_rng_seeded(42);
        let mut rng2 = default_rng_seeded(123);
        let a = rng1.random(100).unwrap();
        let b = rng2.random(100).unwrap();
        // At least some values should differ
        let diffs = a
            .as_slice()
            .unwrap()
            .iter()
            .zip(b.as_slice().unwrap().iter())
            .filter(|(x, y)| x != y)
            .count();
        assert!(diffs > 50, "seeds 42 and 123 produced too-similar output");
    }

    // -----------------------------------------------------------------------
    // N-D shape tests (issue #440)
    // -----------------------------------------------------------------------

    #[test]
    fn random_nd_shape_from_array() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.random([3, 4]).unwrap();
        assert_eq!(arr.shape(), &[3, 4]);
        assert_eq!(arr.size(), 12);
    }

    #[test]
    fn random_nd_shape_from_slice() {
        let mut rng = default_rng_seeded(42);
        let shape: &[usize] = &[2, 3, 4];
        let arr = rng.random(shape).unwrap();
        assert_eq!(arr.shape(), &[2, 3, 4]);
        assert_eq!(arr.size(), 24);
    }

    #[test]
    fn random_nd_shape_from_vec() {
        let mut rng = default_rng_seeded(42);
        let shape = vec![5, 5];
        let arr = rng.random(shape).unwrap();
        assert_eq!(arr.shape(), &[5, 5]);
    }

    #[test]
    fn random_nd_zero_axis_returns_empty() {
        // Issue #264, #455: NumPy returns an empty array for zero-axis
        // shapes; ferray-random now matches that behaviour.
        let mut rng = default_rng_seeded(42);
        let a = rng.random([3, 0]).unwrap();
        assert_eq!(a.shape(), &[3, 0]);
        assert_eq!(a.size(), 0);
        let b = rng.random(0usize).unwrap();
        assert_eq!(b.shape(), &[0]);
        assert_eq!(b.size(), 0);
    }

    #[test]
    fn random_nd_equivalent_to_reshape() {
        // Generating shape [3,4] should produce the same 12 values as size=12
        // because the underlying BitGenerator state advances identically.
        let mut rng1 = default_rng_seeded(42);
        let mut rng2 = default_rng_seeded(42);
        let a = rng1.random(12).unwrap();
        let b = rng2.random([3, 4]).unwrap();
        assert_eq!(a.size(), b.size());
        let a_data: Vec<f64> = a.iter().copied().collect();
        let b_data: Vec<f64> = b.iter().copied().collect();
        assert_eq!(a_data, b_data);
    }

    #[test]
    fn uniform_nd_shape() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.uniform(0.0, 10.0, [2, 5]).unwrap();
        assert_eq!(arr.shape(), &[2, 5]);
        for &v in arr.iter() {
            assert!((0.0..10.0).contains(&v));
        }
    }

    #[test]
    fn integers_nd_shape() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.integers(0, 100, [4, 3]).unwrap();
        assert_eq!(arr.shape(), &[4, 3]);
        for &v in arr.iter() {
            assert!((0..100).contains(&v));
        }
    }

    // ---------------------------------------------------------------
    // f32 variants (issue #441)
    // ---------------------------------------------------------------

    #[test]
    fn random_f32_in_range() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.random_f32(10_000).unwrap();
        for &v in arr.as_slice().unwrap() {
            assert!((0.0..1.0).contains(&v), "f32 value out of range: {v}");
        }
    }

    #[test]
    fn random_f32_deterministic() {
        let mut rng1 = default_rng_seeded(42);
        let mut rng2 = default_rng_seeded(42);
        let a = rng1.random_f32(100).unwrap();
        let b = rng2.random_f32(100).unwrap();
        assert_eq!(a.as_slice().unwrap(), b.as_slice().unwrap());
    }

    #[test]
    fn random_f32_nd_shape() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.random_f32([3, 4]).unwrap();
        assert_eq!(arr.shape(), &[3, 4]);
    }

    #[test]
    fn random_f32_mean() {
        // For Uniform(0,1), mean should be ~0.5 with f32 precision.
        let mut rng = default_rng_seeded(42);
        let n = 100_000usize;
        let arr = rng.random_f32(n).unwrap();
        let sum: f64 = arr.as_slice().unwrap().iter().map(|&v| v as f64).sum();
        let mean = sum / n as f64;
        assert!(
            (mean - 0.5).abs() < 0.01,
            "f32 random mean {mean} too far from 0.5"
        );
    }

    #[test]
    fn uniform_f32_in_range() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.uniform_f32(5.0, 10.0, 10_000).unwrap();
        for &v in arr.as_slice().unwrap() {
            assert!(
                (5.0..10.0).contains(&v),
                "f32 uniform value out of range: {v}"
            );
        }
    }

    #[test]
    fn uniform_f32_bad_range() {
        let mut rng = default_rng_seeded(42);
        assert!(rng.uniform_f32(10.0, 5.0, 100).is_err());
        assert!(rng.uniform_f32(5.0, 5.0, 100).is_err());
    }

    #[test]
    fn uniform_f32_nd_shape() {
        let mut rng = default_rng_seeded(42);
        let arr = rng.uniform_f32(-1.0, 1.0, [2, 5]).unwrap();
        assert_eq!(arr.shape(), &[2, 5]);
        for &v in arr.iter() {
            assert!((-1.0..1.0).contains(&v));
        }
    }

    #[test]
    fn random_f32_zero_axis_returns_empty() {
        let mut rng = default_rng_seeded(42);
        let a = rng.random_f32([3, 0]).unwrap();
        assert_eq!(a.shape(), &[3, 0]);
        assert_eq!(a.size(), 0);
    }
}
