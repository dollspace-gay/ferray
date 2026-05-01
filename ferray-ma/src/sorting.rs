// ferray-ma: Masked sort and argsort (REQ-13, REQ-14)
//
// Sorting unmasked elements while pushing masked elements to the end.

use ferray_core::Array;
use ferray_core::dimension::{Dimension, Ix1, IxDyn};
use ferray_core::dtype::Element;
use ferray_core::error::{FerrayError, FerrayResult};

use crate::MaskedArray;

impl<T, D> MaskedArray<T, D>
where
    T: Element + PartialOrd + Copy,
    D: Dimension,
{
    /// Sort the masked array (flattened), placing masked elements at the end.
    ///
    /// Returns a new 1-D `MaskedArray` where:
    /// - Unmasked elements are sorted in ascending order
    /// - Masked elements come after all unmasked elements
    ///
    /// # Errors
    /// Returns an error only for internal failures.
    pub fn sort(&self) -> FerrayResult<MaskedArray<T, Ix1>> {
        let mut unmasked: Vec<T> = Vec::new();
        let mut masked_vals: Vec<T> = Vec::new();

        for (v, m) in self.data().iter().zip(self.mask().iter()) {
            if *m {
                masked_vals.push(*v);
            } else {
                unmasked.push(*v);
            }
        }

        // Sort unmasked elements using partial_cmp to handle floats
        unmasked.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let unmasked_count = unmasked.len();
        let total = unmasked_count + masked_vals.len();

        // Build result: sorted unmasked followed by masked
        let mut result_data = Vec::with_capacity(total);
        result_data.extend_from_slice(&unmasked);
        result_data.extend_from_slice(&masked_vals);

        let mut result_mask = Vec::with_capacity(total);
        result_mask.extend(std::iter::repeat_n(false, unmasked_count));
        result_mask.extend(std::iter::repeat_n(true, masked_vals.len()));

        let data_arr = Array::from_vec(Ix1::new([total]), result_data)?;
        let mask_arr = Array::from_vec(Ix1::new([total]), result_mask)?;
        MaskedArray::new(data_arr, mask_arr)
    }

    /// Return the indices that would sort the masked array, with masked
    /// elements placed at the end.
    ///
    /// Returns a 1-D `Array<u64, Ix1>` of indices, matching the index
    /// dtype used elsewhere in ferray (#269). Previous versions returned
    /// `Vec<usize>` which was inconsistent with [`MaskedArray::sort`]
    /// (returns a `MaskedArray`) and forced callers to wrap manually
    /// for any downstream array op.
    ///
    /// # Errors
    /// Returns an error only for internal failures.
    pub fn argsort(&self) -> FerrayResult<Array<u64, Ix1>> {
        let vals: Vec<T> = self.data().iter().copied().collect();
        let masks: Vec<bool> = self.mask().iter().copied().collect();

        // Separate indices into unmasked and masked
        let mut unmasked_indices: Vec<usize> = Vec::new();
        let mut masked_indices: Vec<usize> = Vec::new();

        for (i, m) in masks.iter().enumerate() {
            if *m {
                masked_indices.push(i);
            } else {
                unmasked_indices.push(i);
            }
        }

        // Sort unmasked indices by their data values
        unmasked_indices.sort_by(|a, b| {
            vals[*a]
                .partial_cmp(&vals[*b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Concatenate: sorted unmasked indices, then masked indices.
        let total = unmasked_indices.len() + masked_indices.len();
        let mut result: Vec<u64> = Vec::with_capacity(total);
        for &i in &unmasked_indices {
            result.push(i as u64);
        }
        for &i in &masked_indices {
            result.push(i as u64);
        }

        Array::from_vec(Ix1::new([total]), result)
    }

    /// Sort the masked array along `axis`, placing masked elements at
    /// the end of each lane (#271).
    ///
    /// Each 1-D slice along `axis` is sorted independently — unmasked
    /// values ascend, masked values trail. The output preserves the
    /// input shape (no flattening) and produces an `IxDyn` mask
    /// reflecting the new positions of masked entries.
    ///
    /// # Errors
    /// Returns `FerrayError::AxisOutOfBounds` if `axis >= self.ndim()`.
    pub fn sort_axis(&self, axis: usize) -> FerrayResult<MaskedArray<T, IxDyn>> {
        let ndim = self.ndim();
        if axis >= ndim {
            return Err(FerrayError::axis_out_of_bounds(axis, ndim));
        }
        let shape = self.shape().to_vec();
        let axis_len = shape[axis];
        let total: usize = shape.iter().product();

        // Materialize source data and mask in row-major flat order.
        let src_data: Vec<T> = self.data().iter().copied().collect();
        let src_mask: Vec<bool> = self.mask().iter().copied().collect();
        let mut strides = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        let mut out_data = vec![src_data[0]; total];
        let mut out_mask = vec![false; total];

        // Iterate each lane along `axis` by walking the multi-index
        // over the "outer" axes (all but `axis`), then sweeping the
        // axis from 0..axis_len for each.
        let outer_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .filter_map(|(i, &s)| if i == axis { None } else { Some(s) })
            .collect();
        let outer_size: usize = if outer_shape.is_empty() {
            1
        } else {
            outer_shape.iter().product()
        };

        let mut outer_multi = vec![0usize; outer_shape.len()];
        for _ in 0..outer_size {
            // Gather the lane's (value, mask) pairs and their flat indices.
            let mut lane: Vec<(T, bool, usize)> = Vec::with_capacity(axis_len);
            for k in 0..axis_len {
                let mut flat = 0usize;
                let mut o = 0usize;
                for (i, &stride) in strides.iter().enumerate() {
                    if i == axis {
                        flat += stride * k;
                    } else {
                        flat += stride * outer_multi[o];
                        o += 1;
                    }
                }
                lane.push((src_data[flat], src_mask[flat], flat));
            }
            // Partition: unmasked first, then masked. Sort unmasked
            // ascending. Masked entries keep relative input order.
            let mut unmasked: Vec<(T, usize)> = Vec::new();
            let mut masked: Vec<(T, usize)> = Vec::new();
            for (v, m, flat) in lane {
                if m {
                    masked.push((v, flat));
                } else {
                    unmasked.push((v, flat));
                }
            }
            unmasked.sort_by(|a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            // Write back: lane position k gets the k-th value, with
            // mask=false for unmasked positions and mask=true after.
            for (k, (v, _flat)) in unmasked.iter().chain(masked.iter()).enumerate() {
                let mut flat = 0usize;
                let mut o = 0usize;
                for (i, &stride) in strides.iter().enumerate() {
                    if i == axis {
                        flat += stride * k;
                    } else {
                        flat += stride * outer_multi[o];
                        o += 1;
                    }
                }
                out_data[flat] = *v;
                out_mask[flat] = k >= unmasked.len();
            }

            // Increment the outer multi-index.
            for i in (0..outer_shape.len()).rev() {
                outer_multi[i] += 1;
                if outer_multi[i] < outer_shape[i] {
                    break;
                }
                outer_multi[i] = 0;
            }
        }

        let data_arr = Array::<T, IxDyn>::from_vec(IxDyn::new(&shape), out_data)?;
        let mask_arr = Array::<bool, IxDyn>::from_vec(IxDyn::new(&shape), out_mask)?;
        MaskedArray::new(data_arr, mask_arr)
    }
}
