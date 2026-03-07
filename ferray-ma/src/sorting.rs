// ferray-ma: Masked sort and argsort (REQ-13, REQ-14)
//
// Sorting unmasked elements while pushing masked elements to the end.

use ferray_core::Array;
use ferray_core::dimension::{Dimension, Ix1};
use ferray_core::dtype::Element;
use ferray_core::error::FerrayResult;

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
    /// Returns a `Vec<usize>` of indices.
    ///
    /// # Errors
    /// Returns an error only for internal failures.
    pub fn argsort(&self) -> FerrayResult<Vec<usize>> {
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

        // Concatenate: sorted unmasked indices, then masked indices
        let mut result = Vec::with_capacity(unmasked_indices.len() + masked_indices.len());
        result.extend_from_slice(&unmasked_indices);
        result.extend_from_slice(&masked_indices);

        Ok(result)
    }
}
