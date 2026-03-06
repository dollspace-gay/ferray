// ferrum-strings: StringArray<D> type definition (REQ-1, REQ-2)
//
// StringArray is a specialized array type backed by Vec<String>.
// String does not implement Element, so we cannot use NdArray<String, D>.
// Instead we store shape metadata alongside a flat Vec<String>.

use ferrum_core::dimension::{Dimension, Ix1, Ix2, IxDyn};
use ferrum_core::error::{FerrumError, FerrumResult};

/// A specialized N-dimensional array of strings.
///
/// Unlike [`ferrum_core::Array`], this type does not require `Element` —
/// it stores `Vec<String>` directly with shape metadata for indexing.
///
/// The data is stored in row-major (C) order.
#[derive(Debug, Clone)]
pub struct StringArray<D: Dimension> {
    /// Flat storage of string data in row-major order.
    data: Vec<String>,
    /// The shape of this array.
    dim: D,
}

/// 1-dimensional string array.
pub type StringArray1 = StringArray<Ix1>;

/// 2-dimensional string array.
pub type StringArray2 = StringArray<Ix2>;

impl<D: Dimension> StringArray<D> {
    /// Create a new `StringArray` from a flat vector of strings and a shape.
    ///
    /// # Errors
    /// Returns `FerrumError::ShapeMismatch` if `data.len()` does not equal
    /// the product of the shape dimensions.
    pub fn from_vec(dim: D, data: Vec<String>) -> FerrumResult<Self> {
        let expected = dim.size();
        if data.len() != expected {
            return Err(FerrumError::shape_mismatch(format!(
                "data length {} does not match shape {:?} (expected {})",
                data.len(),
                dim.as_slice(),
                expected,
            )));
        }
        Ok(Self { data, dim })
    }

    /// Create a `StringArray` filled with empty strings.
    ///
    /// # Errors
    /// This function is infallible for valid shapes but returns `Result`
    /// for API consistency.
    pub fn empty(dim: D) -> FerrumResult<Self> {
        let size = dim.size();
        let data = vec![String::new(); size];
        Ok(Self { data, dim })
    }

    /// Return the shape as a slice.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.dim.as_slice()
    }

    /// Return the number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.dim.ndim()
    }

    /// Return the total number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Return `true` if the array has no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Return a reference to the dimension descriptor.
    #[inline]
    pub fn dim(&self) -> &D {
        &self.dim
    }

    /// Return a reference to the flat data.
    #[inline]
    pub fn as_slice(&self) -> &[String] {
        &self.data
    }

    /// Return a mutable reference to the flat data.
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [String] {
        &mut self.data
    }

    /// Consume self and return the underlying `Vec<String>`.
    #[inline]
    pub fn into_vec(self) -> Vec<String> {
        self.data
    }

    /// Apply a function to each element, producing a new `StringArray`.
    pub fn map<F>(&self, f: F) -> FerrumResult<StringArray<D>>
    where
        F: Fn(&str) -> String,
    {
        let data: Vec<String> = self.data.iter().map(|s| f(s)).collect();
        StringArray::from_vec(self.dim.clone(), data)
    }

    /// Apply a function to each element, producing a `Vec<T>`.
    ///
    /// This is a lower-level helper used by search and boolean operations
    /// that need to produce typed arrays (e.g., `Array<bool, D>`).
    pub fn map_to_vec<T, F>(&self, f: F) -> Vec<T>
    where
        F: Fn(&str) -> T,
    {
        self.data.iter().map(|s| f(s)).collect()
    }

    /// Iterate over all elements.
    pub fn iter(&self) -> std::slice::Iter<'_, String> {
        self.data.iter()
    }
}

impl<D: Dimension> PartialEq for StringArray<D> {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim && self.data == other.data
    }
}

impl<D: Dimension> Eq for StringArray<D> {}

// ---------------------------------------------------------------------------
// Construction from string slices (REQ-2)
// ---------------------------------------------------------------------------

impl StringArray<Ix1> {
    /// Create a 1-D `StringArray` from a slice of string-like values.
    ///
    /// # Examples
    /// ```ignore
    /// let a = StringArray1::from_slice(&["hello", "world"]).unwrap();
    /// ```
    pub fn from_slice(items: &[&str]) -> FerrumResult<Self> {
        let data: Vec<String> = items.iter().map(|s| (*s).to_string()).collect();
        let dim = Ix1::new([data.len()]);
        Self::from_vec(dim, data)
    }
}

impl StringArray<Ix2> {
    /// Create a 2-D `StringArray` from nested slices.
    ///
    /// # Errors
    /// Returns `FerrumError::ShapeMismatch` if inner slices have different lengths.
    pub fn from_rows(rows: &[&[&str]]) -> FerrumResult<Self> {
        if rows.is_empty() {
            return Self::from_vec(Ix2::new([0, 0]), Vec::new());
        }
        let ncols = rows[0].len();
        for (i, row) in rows.iter().enumerate() {
            if row.len() != ncols {
                return Err(FerrumError::shape_mismatch(format!(
                    "row {} has length {} but row 0 has length {}",
                    i,
                    row.len(),
                    ncols
                )));
            }
        }
        let nrows = rows.len();
        let data: Vec<String> = rows
            .iter()
            .flat_map(|row| row.iter().map(|s| (*s).to_string()))
            .collect();
        Self::from_vec(Ix2::new([nrows, ncols]), data)
    }
}

impl StringArray<IxDyn> {
    /// Create a dynamic-rank `StringArray` from a flat vec and a dynamic shape.
    pub fn from_vec_dyn(shape: &[usize], data: Vec<String>) -> FerrumResult<Self> {
        Self::from_vec(IxDyn::new(shape), data)
    }
}

/// Create a 1-D `StringArray` from a slice of strings — the primary
/// constructor matching `numpy.strings.array(...)`.
///
/// # Errors
/// This function is infallible for valid inputs but returns `Result`
/// for API consistency.
pub fn array(items: &[&str]) -> FerrumResult<StringArray1> {
    StringArray1::from_slice(items)
}

// ---------------------------------------------------------------------------
// Broadcasting helpers for binary string operations
// ---------------------------------------------------------------------------

use ferrum_core::dimension::broadcast::broadcast_shapes;

/// Result of broadcasting two arrays: the output shape and paired indices.
pub(crate) type BroadcastResult = (Vec<usize>, Vec<(usize, usize)>);

/// Compute the broadcast result of two `StringArray`s, returning paired
/// element indices into the flat data of each array.
///
/// Returns `(broadcast_shape, pairs)` where each pair is `(idx_a, idx_b)`.
pub(crate) fn broadcast_binary<Da: Dimension, Db: Dimension>(
    a: &StringArray<Da>,
    b: &StringArray<Db>,
) -> FerrumResult<BroadcastResult> {
    let shape_a = a.shape();
    let shape_b = b.shape();
    let out_shape = broadcast_shapes(shape_a, shape_b)?;
    let out_size: usize = out_shape.iter().product();

    let strides_a = compute_strides(shape_a);
    let strides_b = compute_strides(shape_b);

    let mut pairs = Vec::with_capacity(out_size);
    for linear in 0..out_size {
        let multi = linear_to_multi(linear, &out_shape);
        let idx_a = multi_to_broadcast_linear(&multi, shape_a, &strides_a);
        let idx_b = multi_to_broadcast_linear(&multi, shape_b, &strides_b);
        pairs.push((idx_a, idx_b));
    }

    Ok((out_shape, pairs))
}

/// Compute C-order strides from a shape.
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    if ndim == 0 {
        return vec![];
    }
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Convert a linear index to multi-dimensional indices given a shape.
fn linear_to_multi(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut indices = vec![0usize; ndim];
    for i in (0..ndim).rev() {
        if shape[i] > 0 {
            indices[i] = linear % shape[i];
            linear /= shape[i];
        }
    }
    indices
}

/// Convert multi-dimensional indices to a linear index, applying broadcasting
/// (clamping indices to 0 for dimensions of size 1).
fn multi_to_broadcast_linear(multi: &[usize], src_shape: &[usize], src_strides: &[usize]) -> usize {
    let out_ndim = multi.len();
    let src_ndim = src_shape.len();
    let pad = out_ndim.saturating_sub(src_ndim);

    let mut linear = 0usize;
    for i in 0..src_ndim {
        let idx = multi[i + pad];
        // Broadcast: if src dimension is 1, always use index 0
        let effective = if src_shape[i] == 1 { 0 } else { idx };
        linear += effective * src_strides[i];
    }
    linear
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_from_slice() {
        let a = array(&["hello", "world"]).unwrap();
        assert_eq!(a.shape(), &[2]);
        assert_eq!(a.len(), 2);
        assert_eq!(a.as_slice()[0], "hello");
        assert_eq!(a.as_slice()[1], "world");
    }

    #[test]
    fn create_from_vec() {
        let a = StringArray1::from_vec(Ix1::new([3]), vec!["a".into(), "b".into(), "c".into()])
            .unwrap();
        assert_eq!(a.shape(), &[3]);
    }

    #[test]
    fn shape_mismatch_error() {
        let res = StringArray1::from_vec(Ix1::new([5]), vec!["a".into(), "b".into()]);
        assert!(res.is_err());
    }

    #[test]
    fn empty_array() {
        let a = StringArray1::empty(Ix1::new([4])).unwrap();
        assert_eq!(a.len(), 4);
        assert!(a.as_slice().iter().all(|s| s.is_empty()));
    }

    #[test]
    fn map_strings() {
        let a = array(&["hello", "world"]).unwrap();
        let b = a.map(|s| s.to_uppercase()).unwrap();
        assert_eq!(b.as_slice()[0], "HELLO");
        assert_eq!(b.as_slice()[1], "WORLD");
    }

    #[test]
    fn from_rows_2d() {
        let a = StringArray2::from_rows(&[&["a", "b"], &["c", "d"]]).unwrap();
        assert_eq!(a.shape(), &[2, 2]);
        assert_eq!(a.as_slice(), &["a", "b", "c", "d"]);
    }

    #[test]
    fn from_rows_ragged_error() {
        let res = StringArray2::from_rows(&[&["a", "b"], &["c"]]);
        assert!(res.is_err());
    }

    #[test]
    fn equality() {
        let a = array(&["x", "y"]).unwrap();
        let b = array(&["x", "y"]).unwrap();
        let c = array(&["x", "z"]).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn broadcast_binary_scalar() {
        let a = array(&["hello", "world"]).unwrap();
        let b = array(&["!"]).unwrap();
        let (shape, pairs) = broadcast_binary(&a, &b).unwrap();
        assert_eq!(shape, vec![2]);
        assert_eq!(pairs, vec![(0, 0), (1, 0)]);
    }

    #[test]
    fn broadcast_binary_same_shape() {
        let a = array(&["a", "b", "c"]).unwrap();
        let b = array(&["x", "y", "z"]).unwrap();
        let (shape, pairs) = broadcast_binary(&a, &b).unwrap();
        assert_eq!(shape, vec![3]);
        assert_eq!(pairs, vec![(0, 0), (1, 1), (2, 2)]);
    }

    #[test]
    fn into_vec() {
        let a = array(&["a", "b"]).unwrap();
        let v = a.into_vec();
        assert_eq!(v, vec!["a".to_string(), "b".to_string()]);
    }
}
