// ferray-ma: Masked reductions (REQ-4)
//
// mean, sum, min, max, var, std, count — all skip masked elements.
//
// Whole-array variants (`sum`, `mean`, ...) return a scalar `T`.
// Per-axis variants (`sum_axis`, `mean_axis`, ...) return a `MaskedArray<T, IxDyn>`
// where each output position holds the reduction of one lane along the
// chosen axis. Lanes that contain only masked elements are themselves
// masked in the output (and hold the source array's `fill_value`).

use ferray_core::Array;
use ferray_core::dimension::{Dimension, IxDyn};
use ferray_core::dtype::Element;
use ferray_core::error::{FerrayError, FerrayResult};
use num_traits::Float;

use crate::MaskedArray;

// ---------------------------------------------------------------------------
// Internal helpers for axis-aware reductions
// ---------------------------------------------------------------------------

/// Compute row-major strides for `shape`.
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    let mut s = vec![1usize; n];
    for i in (0..n.saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

/// Increment a multi-index in row-major order. Returns false on overflow.
fn increment_multi(multi: &mut [usize], shape: &[usize]) -> bool {
    for d in (0..multi.len()).rev() {
        multi[d] += 1;
        if multi[d] < shape[d] {
            return true;
        }
        multi[d] = 0;
    }
    false
}

/// Apply a per-lane masked reduction along `axis`.
///
/// `kernel` receives a `&[(T, bool)]` slice (data + mask) for each lane and
/// returns either `Some(value)` (the reduction result) or `None` if every
/// element in the lane was masked. Masked output positions are filled with
/// `fill_value`.
fn reduce_axis<T, D, F>(
    ma: &MaskedArray<T, D>,
    axis: usize,
    fill_value: T,
    kernel: F,
) -> FerrayResult<MaskedArray<T, IxDyn>>
where
    T: Element + Copy,
    D: Dimension,
    F: Fn(&[(T, bool)]) -> Option<T>,
{
    let ndim = ma.ndim();
    if axis >= ndim {
        return Err(FerrayError::axis_out_of_bounds(axis, ndim));
    }
    let shape = ma.shape();
    let axis_len = shape[axis];

    // Output shape: drop the reduced axis.
    let out_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter_map(|(i, &s)| if i == axis { None } else { Some(s) })
        .collect();
    let out_size: usize = if out_shape.is_empty() {
        1
    } else {
        out_shape.iter().product()
    };

    // Materialize source data + mask in row-major order so we can index by
    // computed flat indices regardless of the source memory layout.
    let src_data: Vec<T> = ma.data().iter().copied().collect();
    let src_mask: Vec<bool> = ma.mask().iter().copied().collect();
    let strides = compute_strides(shape);

    let mut out_data = Vec::with_capacity(out_size);
    let mut out_mask = Vec::with_capacity(out_size);
    let mut out_multi = vec![0usize; out_shape.len()];
    let mut in_multi = vec![0usize; ndim];
    let mut lane: Vec<(T, bool)> = Vec::with_capacity(axis_len);

    for _ in 0..out_size {
        // Map output multi-index back into the input multi-index by inserting
        // a placeholder at `axis`.
        let mut out_dim = 0;
        for (d, idx) in in_multi.iter_mut().enumerate() {
            if d == axis {
                *idx = 0;
            } else {
                *idx = out_multi[out_dim];
                out_dim += 1;
            }
        }

        lane.clear();
        for k in 0..axis_len {
            in_multi[axis] = k;
            let flat = in_multi
                .iter()
                .zip(strides.iter())
                .map(|(i, s)| i * s)
                .sum::<usize>();
            lane.push((src_data[flat], src_mask[flat]));
        }

        if let Some(value) = kernel(&lane) {
            out_data.push(value);
            out_mask.push(false);
        } else {
            out_data.push(fill_value);
            out_mask.push(true);
        }

        if !out_shape.is_empty() {
            increment_multi(&mut out_multi, &out_shape);
        }
    }

    let data_arr = Array::<T, IxDyn>::from_vec(IxDyn::new(&out_shape), out_data)?;
    let mask_arr = Array::<bool, IxDyn>::from_vec(IxDyn::new(&out_shape), out_mask)?;
    let mut result = MaskedArray::new(data_arr, mask_arr)?;
    result.set_fill_value(fill_value);
    Ok(result)
}

/// Per-axis count of unmasked elements. Returns a plain `Array<u64, IxDyn>`
/// (not masked, since count is always defined) with the reduced axis dropped.
fn count_axis<T, D>(ma: &MaskedArray<T, D>, axis: usize) -> FerrayResult<Array<u64, IxDyn>>
where
    T: Element + Copy,
    D: Dimension,
{
    let ndim = ma.ndim();
    if axis >= ndim {
        return Err(FerrayError::axis_out_of_bounds(axis, ndim));
    }
    let shape = ma.shape();
    let axis_len = shape[axis];
    let out_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter_map(|(i, &s)| if i == axis { None } else { Some(s) })
        .collect();
    let out_size: usize = if out_shape.is_empty() {
        1
    } else {
        out_shape.iter().product()
    };

    let src_mask: Vec<bool> = ma.mask().iter().copied().collect();
    let strides = compute_strides(shape);
    let mut out: Vec<u64> = Vec::with_capacity(out_size);
    let mut out_multi = vec![0usize; out_shape.len()];
    let mut in_multi = vec![0usize; ndim];

    for _ in 0..out_size {
        let mut out_dim = 0;
        for (d, idx) in in_multi.iter_mut().enumerate() {
            if d == axis {
                *idx = 0;
            } else {
                *idx = out_multi[out_dim];
                out_dim += 1;
            }
        }

        let mut count: u64 = 0;
        for k in 0..axis_len {
            in_multi[axis] = k;
            let flat = in_multi
                .iter()
                .zip(strides.iter())
                .map(|(i, s)| i * s)
                .sum::<usize>();
            if !src_mask[flat] {
                count += 1;
            }
        }
        out.push(count);

        if !out_shape.is_empty() {
            increment_multi(&mut out_multi, &out_shape);
        }
    }

    Array::<u64, IxDyn>::from_vec(IxDyn::new(&out_shape), out)
}

impl<T, D> MaskedArray<T, D>
where
    T: Element + Copy,
    D: Dimension,
{
    /// Count the number of unmasked (valid) elements.
    ///
    /// # Errors
    /// This function does not currently error but returns `Result` for API
    /// consistency.
    pub fn count(&self) -> FerrayResult<usize> {
        let n = self
            .data()
            .iter()
            .zip(self.mask().iter())
            .filter(|(_, m)| !**m)
            .count();
        Ok(n)
    }

    /// Count the number of unmasked elements per lane along `axis`.
    ///
    /// Returns a plain `Array<u64, IxDyn>` (not masked, since count is
    /// always defined) with the reduced axis removed.
    ///
    /// # Errors
    /// Returns `FerrayError::AxisOutOfBounds` if `axis >= ndim`.
    pub fn count_axis(&self, axis: usize) -> FerrayResult<Array<u64, IxDyn>> {
        count_axis(self, axis)
    }
}

impl<T, D> MaskedArray<T, D>
where
    T: Element + Float,
    D: Dimension,
{
    /// Compute the sum of unmasked elements.
    ///
    /// Returns zero if all elements are masked.
    ///
    /// # Errors
    /// Returns an error only for internal failures.
    pub fn sum(&self) -> FerrayResult<T> {
        let zero = num_traits::zero::<T>();
        let s = self
            .data()
            .iter()
            .zip(self.mask().iter())
            .filter(|(_, m)| !**m)
            .fold(zero, |acc, (v, _)| acc + *v);
        Ok(s)
    }

    /// Compute the mean of unmasked elements.
    ///
    /// Returns `NaN` if no elements are unmasked.
    ///
    /// # Errors
    /// Returns an error only for internal failures.
    pub fn mean(&self) -> FerrayResult<T> {
        let zero = num_traits::zero::<T>();
        let (sum, count) = self
            .data()
            .iter()
            .zip(self.mask().iter())
            .filter(|(_, m)| !**m)
            .fold((zero, 0usize), |(s, c), (v, _)| (s + *v, c + 1));
        if count == 0 {
            return Ok(T::nan());
        }
        // #267: T::from(count).unwrap_or(one) silently returned the
        // sum as the "mean" if the conversion ever failed. Surface a
        // typed error instead so a downstream NaN/garbage isn't
        // misattributed to upstream data.
        let n = T::from(count).ok_or_else(|| {
            FerrayError::invalid_value(format!(
                "cannot convert unmasked count {count} to element type"
            ))
        })?;
        Ok(sum / n)
    }

    /// Compute the minimum of unmasked elements.
    ///
    /// NaN values in unmasked elements are propagated (returns NaN), matching `NumPy`.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if no elements are unmasked.
    pub fn min(&self) -> FerrayResult<T> {
        self.data()
            .iter()
            .zip(self.mask().iter())
            .filter(|(_, m)| !**m)
            .map(|(v, _)| *v)
            .fold(None, |acc: Option<T>, v| {
                Some(match acc {
                    Some(a) => {
                        // NaN-propagating: if comparison is unordered, propagate NaN
                        if a <= v {
                            a
                        } else if a > v {
                            v
                        } else {
                            a
                        }
                    }
                    None => v,
                })
            })
            .ok_or_else(|| FerrayError::invalid_value("min: all elements are masked"))
    }

    /// Compute the maximum of unmasked elements.
    ///
    /// NaN values in unmasked elements are propagated (returns NaN), matching `NumPy`.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if no elements are unmasked.
    pub fn max(&self) -> FerrayResult<T> {
        self.data()
            .iter()
            .zip(self.mask().iter())
            .filter(|(_, m)| !**m)
            .map(|(v, _)| *v)
            .fold(None, |acc: Option<T>, v| {
                Some(match acc {
                    Some(a) => {
                        if a >= v {
                            a
                        } else if a < v {
                            v
                        } else {
                            a
                        }
                    }
                    None => v,
                })
            })
            .ok_or_else(|| FerrayError::invalid_value("max: all elements are masked"))
    }

    /// Compute the variance of unmasked elements (population variance, ddof=0).
    ///
    /// Returns `NaN` if no elements are unmasked.
    ///
    /// # Errors
    /// Returns an error only for internal failures.
    pub fn var(&self) -> FerrayResult<T> {
        self.var_ddof(0)
    }

    /// Compute the variance of unmasked elements with a delta degrees-of-freedom
    /// adjustment.
    ///
    /// `ddof = 0` is the population variance (divides by `n`).
    /// `ddof = 1` is Bessel's correction for sample variance (divides
    /// by `n - 1`). Matches `numpy.ma.var(ddof=...)` (#270).
    ///
    /// Returns `NaN` if `count <= ddof` (insufficient unmasked elements).
    ///
    /// # Errors
    /// Returns an error only for internal failures.
    pub fn var_ddof(&self, ddof: usize) -> FerrayResult<T> {
        let mean = self.mean()?;
        if mean.is_nan() {
            return Ok(T::nan());
        }
        let zero = num_traits::zero::<T>();
        let (sum_sq, count) = self
            .data()
            .iter()
            .zip(self.mask().iter())
            .filter(|(_, m)| !**m)
            .fold((zero, 0usize), |(s, c), (v, _)| {
                let d = *v - mean;
                (s + d * d, c + 1)
            });
        if count <= ddof {
            return Ok(T::nan());
        }
        let n = T::from(count - ddof).ok_or_else(|| {
            FerrayError::invalid_value(format!(
                "cannot convert (count - ddof) = {} to element type",
                count - ddof
            ))
        })?;
        Ok(sum_sq / n)
    }

    /// Compute the standard deviation of unmasked elements (population, ddof=0).
    ///
    /// Returns `NaN` if no elements are unmasked.
    ///
    /// # Errors
    /// Returns an error only for internal failures.
    pub fn std(&self) -> FerrayResult<T> {
        Ok(self.var()?.sqrt())
    }

    /// Compute the standard deviation of unmasked elements with a delta
    /// degrees-of-freedom adjustment.
    ///
    /// `ddof = 0` is the population std; `ddof = 1` is the sample std
    /// (Bessel's correction). Matches `numpy.ma.std(ddof=...)` (#270).
    ///
    /// Returns `NaN` if `count <= ddof`.
    ///
    /// # Errors
    /// Returns an error only for internal failures.
    pub fn std_ddof(&self, ddof: usize) -> FerrayResult<T> {
        Ok(self.var_ddof(ddof)?.sqrt())
    }

    // -----------------------------------------------------------------------
    // Per-axis reductions (issue #500)
    //
    // Each lane along `axis` is reduced independently. Lanes containing only
    // masked elements produce a masked output position holding `fill_value`.
    // -----------------------------------------------------------------------

    /// Sum unmasked elements along `axis`. Returns a masked array with the
    /// reduced axis removed; lanes that are entirely masked produce a
    /// masked output position holding `fill_value`.
    pub fn sum_axis(&self, axis: usize) -> FerrayResult<MaskedArray<T, IxDyn>> {
        let zero = num_traits::zero::<T>();
        let fill = self.fill_value();
        reduce_axis(self, axis, fill, |lane| {
            let mut acc = zero;
            let mut any = false;
            for &(v, m) in lane {
                if !m {
                    acc = acc + v;
                    any = true;
                }
            }
            if any { Some(acc) } else { None }
        })
    }

    /// Mean of unmasked elements along `axis`. All-masked lanes are masked
    /// in the output.
    pub fn mean_axis(&self, axis: usize) -> FerrayResult<MaskedArray<T, IxDyn>> {
        let zero = num_traits::zero::<T>();
        let fill = self.fill_value();
        reduce_axis(self, axis, fill, |lane| {
            let mut acc = zero;
            let mut count = 0usize;
            for &(v, m) in lane {
                if !m {
                    acc = acc + v;
                    count += 1;
                }
            }
            if count == 0 {
                None
            } else {
                // #267: a failed count→T conversion would silently
                // divide by 1, returning the sum as the "mean". Mask
                // the cell instead — for f32/f64 the conversion never
                // fails, so this only triggers on pathological types.
                T::from(count).map(|n| acc / n)
            }
        })
    }

    /// Min of unmasked elements along `axis`. NaN-propagating per `NumPy`.
    pub fn min_axis(&self, axis: usize) -> FerrayResult<MaskedArray<T, IxDyn>> {
        let fill = self.fill_value();
        reduce_axis(self, axis, fill, |lane| {
            let mut acc: Option<T> = None;
            for &(v, m) in lane {
                if !m {
                    acc = Some(match acc {
                        Some(a) => {
                            // NaN-propagating: if comparison is unordered, return NaN
                            if a <= v {
                                a
                            } else if a > v {
                                v
                            } else {
                                a
                            }
                        }
                        None => v,
                    });
                }
            }
            acc
        })
    }

    /// Max of unmasked elements along `axis`. NaN-propagating per `NumPy`.
    pub fn max_axis(&self, axis: usize) -> FerrayResult<MaskedArray<T, IxDyn>> {
        let fill = self.fill_value();
        reduce_axis(self, axis, fill, |lane| {
            let mut acc: Option<T> = None;
            for &(v, m) in lane {
                if !m {
                    acc = Some(match acc {
                        Some(a) => {
                            if a >= v {
                                a
                            } else if a < v {
                                v
                            } else {
                                a
                            }
                        }
                        None => v,
                    });
                }
            }
            acc
        })
    }

    /// Population variance (ddof=0) of unmasked elements along `axis`.
    pub fn var_axis(&self, axis: usize) -> FerrayResult<MaskedArray<T, IxDyn>> {
        self.var_axis_ddof(axis, 0)
    }

    /// Variance with `ddof` adjustment along `axis` (#270).
    ///
    /// `ddof = 1` matches numpy's sample variance (Bessel's correction).
    /// Lanes with `count <= ddof` produce a masked output position
    /// holding `fill_value`.
    pub fn var_axis_ddof(
        &self,
        axis: usize,
        ddof: usize,
    ) -> FerrayResult<MaskedArray<T, IxDyn>> {
        let zero = num_traits::zero::<T>();
        let fill = self.fill_value();
        reduce_axis(self, axis, fill, |lane| {
            let mut acc = zero;
            let mut count = 0usize;
            for &(v, m) in lane {
                if !m {
                    acc = acc + v;
                    count += 1;
                }
            }
            if count <= ddof {
                return None;
            }
            // For the running mean we still divide by `count`, not
            // `count - ddof` — ddof only applies to the final variance
            // denominator (matches numpy's behavior).
            let n_mean = T::from(count)?;
            let mean = acc / n_mean;
            let mut sum_sq = zero;
            for &(v, m) in lane {
                if !m {
                    let d = v - mean;
                    sum_sq = sum_sq + d * d;
                }
            }
            let n_var = T::from(count - ddof)?;
            Some(sum_sq / n_var)
        })
    }

    /// Population standard deviation (ddof=0) of unmasked elements along `axis`.
    pub fn std_axis(&self, axis: usize) -> FerrayResult<MaskedArray<T, IxDyn>> {
        self.std_axis_ddof(axis, 0)
    }

    /// Standard deviation with `ddof` adjustment along `axis` (#270).
    pub fn std_axis_ddof(
        &self,
        axis: usize,
        ddof: usize,
    ) -> FerrayResult<MaskedArray<T, IxDyn>> {
        let result = self.var_axis_ddof(axis, ddof)?;
        let fill = self.fill_value();
        let mask = result.mask().clone();
        let new_data: Vec<T> = result
            .data()
            .iter()
            .zip(result.mask().iter())
            .map(|(v, m)| if *m { fill } else { v.sqrt() })
            .collect();
        let data_arr = Array::<T, IxDyn>::from_vec(IxDyn::new(result.shape()), new_data)?;
        let mut out = MaskedArray::new(data_arr, mask)?;
        out.set_fill_value(fill);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::{Ix1, Ix2};

    fn ma2d(rows: usize, cols: usize, data: Vec<f64>, mask: Vec<bool>) -> MaskedArray<f64, Ix2> {
        let d = Array::<f64, Ix2>::from_vec(Ix2::new([rows, cols]), data).unwrap();
        let m = Array::<bool, Ix2>::from_vec(Ix2::new([rows, cols]), mask).unwrap();
        MaskedArray::new(d, m).unwrap()
    }

    // ---- #500: per-axis reductions ----

    #[test]
    fn sum_axis_drops_axis() {
        // 2x3 array, no masks. axis=0 sums columns, axis=1 sums rows.
        let ma = ma2d(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![false; 6]);
        let s0 = ma.sum_axis(0).unwrap();
        assert_eq!(s0.shape(), &[3]);
        let d0: Vec<f64> = s0.data().iter().copied().collect();
        assert_eq!(d0, vec![5.0, 7.0, 9.0]);

        let s1 = ma.sum_axis(1).unwrap();
        assert_eq!(s1.shape(), &[2]);
        let d1: Vec<f64> = s1.data().iter().copied().collect();
        assert_eq!(d1, vec![6.0, 15.0]);
    }

    #[test]
    fn sum_axis_skips_masked() {
        // 2x3 array. Mask out (0, 1) so column 1 and row 0 lose one element.
        let ma = ma2d(
            2,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![false, true, false, false, false, false],
        );
        // axis=0 (per-column): col0 = 1+4=5, col1 = 5 (only row 1), col2 = 3+6=9
        let s0 = ma.sum_axis(0).unwrap();
        let d0: Vec<f64> = s0.data().iter().copied().collect();
        assert_eq!(d0, vec![5.0, 5.0, 9.0]);
        let m0: Vec<bool> = s0.mask().iter().copied().collect();
        assert_eq!(m0, vec![false, false, false]);
    }

    #[test]
    fn sum_axis_all_masked_lane_is_masked() {
        // 2x3 array, mask out entire column 1.
        let ma = ma2d(
            2,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![false, true, false, false, true, false],
        );
        let s0 = ma.sum_axis(0).unwrap();
        let m0: Vec<bool> = s0.mask().iter().copied().collect();
        assert_eq!(m0, vec![false, true, false]);
    }

    #[test]
    fn mean_axis_skips_masked() {
        let ma = ma2d(
            2,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![false, true, false, false, false, false],
        );
        // mean axis 1 (per-row): row 0 = (1+3)/2 = 2.0, row 1 = (4+5+6)/3 = 5.0
        let m1 = ma.mean_axis(1).unwrap();
        let d: Vec<f64> = m1.data().iter().copied().collect();
        assert_eq!(d, vec![2.0, 5.0]);
    }

    #[test]
    fn min_max_axis() {
        let ma = ma2d(2, 3, vec![3.0, 1.0, 5.0, 2.0, 4.0, 0.0], vec![false; 6]);
        let mn = ma.min_axis(0).unwrap();
        let mx = ma.max_axis(0).unwrap();
        let mn_d: Vec<f64> = mn.data().iter().copied().collect();
        let mx_d: Vec<f64> = mx.data().iter().copied().collect();
        assert_eq!(mn_d, vec![2.0, 1.0, 0.0]);
        assert_eq!(mx_d, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn count_axis_basic() {
        // 2x3 array. Mask: (0,1) and (1,2). col0:2, col1:1, col2:1.
        let ma = ma2d(
            2,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![false, true, false, false, false, true],
        );
        let c0 = ma.count_axis(0).unwrap();
        let v: Vec<u64> = c0.iter().copied().collect();
        assert_eq!(v, vec![2u64, 1, 1]);
    }

    #[test]
    fn axis_out_of_bounds_errors() {
        let ma = ma2d(2, 3, vec![0.0; 6], vec![false; 6]);
        assert!(ma.sum_axis(2).is_err());
    }

    #[test]
    fn var_std_axis() {
        // Two rows of [1, 2, 3, 4, 5] — variance along axis 1 should be 2.0 each.
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let ma = ma2d(2, 5, data, vec![false; 10]);
        let v = ma.var_axis(1).unwrap();
        let s = ma.std_axis(1).unwrap();
        let v_d: Vec<f64> = v.data().iter().copied().collect();
        let s_d: Vec<f64> = s.data().iter().copied().collect();
        for &x in &v_d {
            assert!((x - 2.0).abs() < 1e-12);
        }
        for &x in &s_d {
            assert!((x - 2.0_f64.sqrt()).abs() < 1e-12);
        }
    }

    // ---- #501: fill_value ----

    #[test]
    fn fill_value_default_is_zero() {
        let d = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let m = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false; 3]).unwrap();
        let ma = MaskedArray::new(d, m).unwrap();
        assert_eq!(ma.fill_value(), 0.0);
    }

    #[test]
    fn with_fill_value_sets_field() {
        let d = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let m = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false; 3]).unwrap();
        let ma = MaskedArray::new(d, m).unwrap().with_fill_value(99.0);
        assert_eq!(ma.fill_value(), 99.0);
    }

    #[test]
    fn filled_default_uses_stored_fill_value() {
        let d = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let m =
            Array::<bool, Ix1>::from_vec(Ix1::new([4]), vec![false, true, false, true]).unwrap();
        let ma = MaskedArray::new(d, m).unwrap().with_fill_value(-1.0);
        let filled = ma.filled_default().unwrap();
        let v: Vec<f64> = filled.iter().copied().collect();
        assert_eq!(v, vec![1.0, -1.0, 3.0, -1.0]);
    }

    #[test]
    fn arithmetic_uses_fill_value() {
        // (Adding two masked arrays) — result data at masked positions should
        // be the receiver's fill_value, not zero.
        use crate::masked_add;
        let d_a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let m_a = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false, true, false]).unwrap();
        let d_b = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![10.0, 20.0, 30.0]).unwrap();
        let m_b = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false; 3]).unwrap();
        let a = MaskedArray::new(d_a, m_a).unwrap().with_fill_value(-999.0);
        let b = MaskedArray::new(d_b, m_b).unwrap();
        let r = masked_add(&a, &b).unwrap();
        let r_d: Vec<f64> = r.data().iter().copied().collect();
        assert_eq!(r_d, vec![11.0, -999.0, 33.0]);
        assert_eq!(r.fill_value(), -999.0);
    }

    // ---- #504: broadcasting in masked arithmetic ----

    #[test]
    fn masked_add_broadcasts_within_same_rank() {
        use crate::masked_add;
        // (3, 1) + (1, 4) -> (3, 4) — both Ix2.
        let d_a = Array::<f64, Ix2>::from_vec(Ix2::new([3, 1]), vec![1.0, 2.0, 3.0]).unwrap();
        let m_a = Array::<bool, Ix2>::from_vec(Ix2::new([3, 1]), vec![false; 3]).unwrap();
        let d_b =
            Array::<f64, Ix2>::from_vec(Ix2::new([1, 4]), vec![10.0, 20.0, 30.0, 40.0]).unwrap();
        let m_b = Array::<bool, Ix2>::from_vec(Ix2::new([1, 4]), vec![false; 4]).unwrap();
        let a = MaskedArray::new(d_a, m_a).unwrap();
        let b = MaskedArray::new(d_b, m_b).unwrap();
        let r = masked_add(&a, &b).unwrap();
        assert_eq!(r.shape(), &[3, 4]);
        let r_d: Vec<f64> = r.data().iter().copied().collect();
        assert_eq!(
            r_d,
            vec![
                11.0, 21.0, 31.0, 41.0, // row 0 = 1 + {10,20,30,40}
                12.0, 22.0, 32.0, 42.0, // row 1 = 2 + ...
                13.0, 23.0, 33.0, 43.0, // row 2 = 3 + ...
            ]
        );
        let r_m: Vec<bool> = r.mask().iter().copied().collect();
        assert_eq!(r_m, vec![false; 12]);
    }

    #[test]
    fn masked_sub_broadcasts_with_mask_union() {
        use crate::masked_sub;
        // Mask one element in `a`. After broadcasting (3,1) -> (3,4),
        // the masked row becomes a fully-masked row in the result.
        let d_a = Array::<f64, Ix2>::from_vec(Ix2::new([3, 1]), vec![10.0, 20.0, 30.0]).unwrap();
        let m_a = Array::<bool, Ix2>::from_vec(Ix2::new([3, 1]), vec![false, true, false]).unwrap();
        let d_b = Array::<f64, Ix2>::from_vec(Ix2::new([1, 4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let m_b = Array::<bool, Ix2>::from_vec(Ix2::new([1, 4]), vec![false; 4]).unwrap();
        let a = MaskedArray::new(d_a, m_a).unwrap();
        let b = MaskedArray::new(d_b, m_b).unwrap();
        let r = masked_sub(&a, &b).unwrap();
        let r_m: Vec<bool> = r.mask().iter().copied().collect();
        // Row 1 is fully masked (4 positions).
        assert_eq!(
            r_m,
            vec![
                false, false, false, false, // row 0
                true, true, true, true, // row 1 (masked)
                false, false, false, false, // row 2
            ]
        );
    }

    // ---- #276: all-masked whole-array reductions ----
    //
    // Pin the current semantics when every element is masked:
    //   sum   -> 0 (neutral element of the fold)
    //   mean  -> NaN
    //   var   -> NaN
    //   std   -> NaN
    //   min   -> error (FerrayError::InvalidValue)
    //   max   -> error (FerrayError::InvalidValue)

    fn all_masked_ma1d(n: usize) -> MaskedArray<f64, Ix1> {
        let d = Array::<f64, Ix1>::from_vec(Ix1::new([n]), vec![1.0; n]).unwrap();
        let m = Array::<bool, Ix1>::from_vec(Ix1::new([n]), vec![true; n]).unwrap();
        MaskedArray::new(d, m).unwrap()
    }

    #[test]
    fn sum_all_masked_returns_zero() {
        let ma = all_masked_ma1d(4);
        assert_eq!(ma.sum().unwrap(), 0.0);
    }

    #[test]
    fn mean_all_masked_returns_nan() {
        let ma = all_masked_ma1d(4);
        assert!(ma.mean().unwrap().is_nan());
    }

    #[test]
    fn var_all_masked_returns_nan() {
        let ma = all_masked_ma1d(4);
        assert!(ma.var().unwrap().is_nan());
    }

    #[test]
    fn std_all_masked_returns_nan() {
        let ma = all_masked_ma1d(4);
        assert!(ma.std().unwrap().is_nan());
    }

    #[test]
    fn min_max_all_masked_error() {
        // Documenting the asymmetry: sum/mean/var/std fall through with
        // 0/NaN sentinels, but min/max have no neutral element and error.
        let ma = all_masked_ma1d(4);
        assert!(ma.min().is_err());
        assert!(ma.max().is_err());
    }

    #[test]
    fn sum_var_std_all_masked_2d_matches_1d() {
        // Same semantics hold for multi-dimensional whole-array reductions.
        let d = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![9.0; 6]).unwrap();
        let m = Array::<bool, Ix2>::from_vec(Ix2::new([2, 3]), vec![true; 6]).unwrap();
        let ma = MaskedArray::new(d, m).unwrap();
        assert_eq!(ma.sum().unwrap(), 0.0);
        assert!(ma.var().unwrap().is_nan());
        assert!(ma.std().unwrap().is_nan());
    }

    #[test]
    fn masked_add_broadcast_incompatible_errors() {
        use crate::masked_add;
        let d_a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let m_a = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false; 3]).unwrap();
        let d_b = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let m_b = Array::<bool, Ix1>::from_vec(Ix1::new([4]), vec![false; 4]).unwrap();
        let a = MaskedArray::new(d_a, m_a).unwrap();
        let b = MaskedArray::new(d_b, m_b).unwrap();
        assert!(masked_add(&a, &b).is_err());
    }

    // ----- ddof=1 sample variance/std (#270) -----------------------------

    #[test]
    fn var_ddof_zero_matches_default_var() {
        let data = Array::<f64, Ix1>::from_vec(
            Ix1::new([5]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        )
        .unwrap();
        let mask = Array::<bool, Ix1>::from_vec(Ix1::new([5]), vec![false; 5]).unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        let v0 = ma.var().unwrap();
        let v_explicit = ma.var_ddof(0).unwrap();
        assert!((v0 - v_explicit).abs() < 1e-14);
    }

    #[test]
    fn var_ddof_one_is_bessel_corrected() {
        // [1,2,3,4,5] mean = 3; squared deviations sum = 10.
        // ddof=0: 10/5 = 2.0
        // ddof=1: 10/4 = 2.5 (Bessel)
        let data = Array::<f64, Ix1>::from_vec(
            Ix1::new([5]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        )
        .unwrap();
        let mask = Array::<bool, Ix1>::from_vec(Ix1::new([5]), vec![false; 5]).unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        let v0 = ma.var_ddof(0).unwrap();
        let v1 = ma.var_ddof(1).unwrap();
        assert!((v0 - 2.0).abs() < 1e-14, "ddof=0: expected 2.0, got {v0}");
        assert!((v1 - 2.5).abs() < 1e-14, "ddof=1: expected 2.5, got {v1}");
    }

    #[test]
    fn var_ddof_skips_masked_elements() {
        // [1,2,_,4,5] with index 2 masked: mean = 3, sq deviations = 10, count = 4.
        // ddof=0: 10/4 = 2.5; ddof=1: 10/3.
        let data = Array::<f64, Ix1>::from_vec(
            Ix1::new([5]),
            vec![1.0, 2.0, 99.0, 4.0, 5.0],
        )
        .unwrap();
        let mask = Array::<bool, Ix1>::from_vec(
            Ix1::new([5]),
            vec![false, false, true, false, false],
        )
        .unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        let v0 = ma.var_ddof(0).unwrap();
        let v1 = ma.var_ddof(1).unwrap();
        assert!((v0 - 2.5).abs() < 1e-14);
        assert!((v1 - 10.0 / 3.0).abs() < 1e-14);
    }

    #[test]
    fn var_ddof_returns_nan_when_count_le_ddof() {
        // 1 unmasked element + ddof=1 → division by zero would occur;
        // numpy returns NaN.
        let data =
            Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let mask =
            Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![true, false, true]).unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        let v = ma.var_ddof(1).unwrap();
        assert!(v.is_nan(), "expected NaN, got {v}");
    }

    #[test]
    fn std_ddof_is_sqrt_of_var_ddof() {
        let data = Array::<f64, Ix1>::from_vec(
            Ix1::new([5]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        )
        .unwrap();
        let mask = Array::<bool, Ix1>::from_vec(Ix1::new([5]), vec![false; 5]).unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        let s1 = ma.std_ddof(1).unwrap();
        let v1 = ma.var_ddof(1).unwrap();
        assert!((s1 - v1.sqrt()).abs() < 1e-14);
    }

    #[test]
    fn var_axis_ddof_one_per_row() {
        use ferray_core::dimension::Ix2;
        // Two rows: [1,2,3] and [10,20,30]; both unmasked.
        // Row 0: mean=2, sum_sq = 1+0+1 = 2; ddof=1 → 2.
        // Row 1: mean=20, sum_sq = 100+0+100 = 200; ddof=1 → 100.
        let data = Array::<f64, Ix2>::from_vec(
            Ix2::new([2, 3]),
            vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        )
        .unwrap();
        let mask = Array::<bool, Ix2>::from_vec(Ix2::new([2, 3]), vec![false; 6]).unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        let v = ma.var_axis_ddof(1, 1).unwrap();
        let vs: Vec<f64> = v.data().iter().copied().collect();
        assert_eq!(vs.len(), 2);
        assert!((vs[0] - 1.0).abs() < 1e-12);
        assert!((vs[1] - 100.0).abs() < 1e-12);
    }
}
