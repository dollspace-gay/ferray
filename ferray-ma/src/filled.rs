// ferray-ma: filled and compressed (REQ-5, REQ-6)

use ferray_core::Array;
use ferray_core::dimension::{Dimension, Ix1};
use ferray_core::dtype::Element;
use ferray_core::error::FerrayResult;

use crate::MaskedArray;

impl<T, D> MaskedArray<T, D>
where
    T: Element + Copy,
    D: Dimension,
{
    /// Return a regular array with masked positions replaced by `fill_value`.
    ///
    /// Unmasked positions retain their original data values.
    ///
    /// # Errors
    /// Returns an error only for internal failures.
    pub fn filled(&self, fill_value: T) -> FerrayResult<Array<T, D>> {
        let data: Vec<T> = self
            .data()
            .iter()
            .zip(self.mask().iter())
            .map(|(v, m)| if *m { fill_value } else { *v })
            .collect();
        Array::from_vec(self.dim().clone(), data)
    }

    /// Return a regular array with masked positions replaced by the array's
    /// stored [`MaskedArray::fill_value`].
    ///
    /// Equivalent to `NumPy`'s `arr.filled()` with no argument. Use [`MaskedArray::filled`]
    /// to override the fill value for a single call.
    ///
    /// # Errors
    /// Returns an error only for internal failures.
    pub fn filled_default(&self) -> FerrayResult<Array<T, D>> {
        self.filled(self.fill_value)
    }

    /// Return a 1-D array containing only the unmasked elements.
    ///
    /// The order is the logical (row-major) iteration order of the
    /// original array, with masked elements removed.
    ///
    /// # Errors
    /// Returns an error only for internal failures.
    pub fn compressed(&self) -> FerrayResult<Array<T, Ix1>> {
        let data: Vec<T> = self
            .data()
            .iter()
            .zip(self.mask().iter())
            .filter(|(_, m)| !**m)
            .map(|(v, _)| *v)
            .collect();
        let len = data.len();
        Array::from_vec(Ix1::new([len]), data)
    }
}
