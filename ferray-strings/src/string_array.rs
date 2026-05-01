// ferray-strings: StringArray<D> type definition (REQ-1, REQ-2)
//
// StringArray is a specialized array type backed by Vec<String>.
// String does not implement Element, so we cannot use NdArray<String, D>.
// Instead we store shape metadata alongside a flat Vec<String>.

use ferray_core::dimension::{Dimension, Ix1, Ix2, IxDyn};
use ferray_core::error::{FerrayError, FerrayResult};

/// A specialized N-dimensional array of strings.
///
/// Unlike [`ferray_core::Array`], this type does not require `Element` —
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
    /// Returns `FerrayError::ShapeMismatch` if `data.len()` does not equal
    /// the product of the shape dimensions.
    pub fn from_vec(dim: D, data: Vec<String>) -> FerrayResult<Self> {
        let expected = dim.size();
        if data.len() != expected {
            return Err(FerrayError::shape_mismatch(format!(
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
    pub fn empty(dim: D) -> FerrayResult<Self> {
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
    pub const fn dim(&self) -> &D {
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
    pub fn map<F>(&self, f: F) -> FerrayResult<Self>
    where
        F: Fn(&str) -> String,
    {
        let data: Vec<String> = self.data.iter().map(|s| f(s)).collect();
        Self::from_vec(self.dim.clone(), data)
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

    // -----------------------------------------------------------------
    // Shape operations (#514) — parallel to `ferray_core::Array`
    //
    // StringArray can't reuse `Array<String, D>` because `String`
    // isn't an `Element` (the trait is sealed inside ferray-core and
    // `String` isn't `Copy`). Instead we mirror the shape API that
    // `Array` exposes so callers can write shape-manipulation code
    // that looks the same for string and numeric arrays.
    // -----------------------------------------------------------------

    /// Reshape this array to a new dimension type / shape. The total
    /// element count must be unchanged.
    ///
    /// Since strings are cheap to move (they're owned), reshape just
    /// rebuilds the array around the existing buffer. No data copy.
    ///
    /// # Errors
    /// Returns [`FerrayError::ShapeMismatch`] if the new shape's
    /// element count does not match `self.len()`.
    pub fn reshape<D2: Dimension>(self, new_dim: D2) -> FerrayResult<StringArray<D2>> {
        StringArray::<D2>::from_vec(new_dim, self.data)
    }

    /// Flatten to a 1-D `StringArray1` of length `self.len()`. The
    /// row-major traversal order is preserved.
    ///
    /// This is the string analogue of `ndarray::Array::flatten` /
    /// `NumPy`'s `arr.flatten()`.
    pub fn flatten(self) -> StringArray1 {
        let n = self.data.len();
        StringArray::<Ix1>::from_vec(Ix1::new([n]), self.data)
            .expect("flatten: length check is trivially satisfied")
    }

    /// Convert to a dynamic-rank `StringArray<IxDyn>`. Useful when
    /// the rank isn't known until runtime, or when interoperating
    /// with code that only accepts `IxDyn`.
    pub fn into_dyn(self) -> StringArray<IxDyn> {
        let shape = self.dim.as_slice().to_vec();
        StringArray::<IxDyn>::from_vec(IxDyn::new(&shape), self.data)
            .expect("into_dyn: shape length check is trivially satisfied")
    }

    /// Look up an element by multi-dimensional index. Returns `None`
    /// if the index is out of bounds.
    ///
    /// Indexing is row-major (C-order): for a `(rows, cols)` array,
    /// index `[r, c]` maps to `data[r * cols + c]`.
    pub fn get(&self, idx: &[usize]) -> Option<&String> {
        let shape = self.dim.as_slice();
        if idx.len() != shape.len() {
            return None;
        }
        let mut flat = 0usize;
        let mut stride = 1usize;
        // Walk dimensions right-to-left (row-major).
        for (i, (&dim, &k)) in shape.iter().zip(idx.iter()).enumerate().rev() {
            if k >= dim {
                return None;
            }
            if i == shape.len() - 1 {
                flat += k;
            } else {
                flat += k * stride;
            }
            stride *= dim;
        }
        self.data.get(flat)
    }
}

impl<D: Dimension> PartialEq for StringArray<D> {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim && self.data == other.data
    }
}

impl<D: Dimension> Eq for StringArray<D> {}

// Display: print like NumPy's `array(["a", "b", "c"])` (#278).
impl<D: Dimension> std::fmt::Display for StringArray<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "array([")?;
        for (i, s) in self.data.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{s:?}")?;
        }
        write!(f, "])")
    }
}

// IntoIterator for &StringArray yields &String (#278).
impl<'a, D: Dimension> IntoIterator for &'a StringArray<D> {
    type Item = &'a String;
    type IntoIter = std::slice::Iter<'a, String>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

// IntoIterator for StringArray yields owned Strings.
impl<D: Dimension> IntoIterator for StringArray<D> {
    type Item = String;
    type IntoIter = std::vec::IntoIter<String>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

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
    pub fn from_slice(items: &[&str]) -> FerrayResult<Self> {
        let data: Vec<String> = items.iter().map(|s| (*s).to_string()).collect();
        let dim = Ix1::new([data.len()]);
        Self::from_vec(dim, data)
    }
}

impl StringArray<Ix2> {
    /// Transpose a 2-D `StringArray`: swap rows and columns.
    ///
    /// Walks the elements into a new buffer — a `(r, c)` cell of the
    /// input becomes `(c, r)` of the output. Strings are cloned to
    /// avoid disturbing the original array.
    pub fn transpose(&self) -> FerrayResult<Self> {
        let shape = self.shape();
        let (nrows, ncols) = (shape[0], shape[1]);
        let mut data = Vec::with_capacity(nrows * ncols);
        for c in 0..ncols {
            for r in 0..nrows {
                data.push(self.data[r * ncols + c].clone());
            }
        }
        Self::from_vec(Ix2::new([ncols, nrows]), data)
    }

    /// Create a 2-D `StringArray` from nested slices.
    ///
    /// # Errors
    /// Returns `FerrayError::ShapeMismatch` if inner slices have different lengths.
    pub fn from_rows(rows: &[&[&str]]) -> FerrayResult<Self> {
        if rows.is_empty() {
            return Self::from_vec(Ix2::new([0, 0]), Vec::new());
        }
        let ncols = rows[0].len();
        for (i, row) in rows.iter().enumerate() {
            if row.len() != ncols {
                return Err(FerrayError::shape_mismatch(format!(
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
    pub fn from_vec_dyn(shape: &[usize], data: Vec<String>) -> FerrayResult<Self> {
        Self::from_vec(IxDyn::new(shape), data)
    }
}

/// Create a 1-D `StringArray` from a slice of strings — the primary
/// constructor matching `numpy.strings.array(...)`.
///
/// # Errors
/// This function is infallible for valid inputs but returns `Result`
/// for API consistency.
pub fn array(items: &[&str]) -> FerrayResult<StringArray1> {
    StringArray1::from_slice(items)
}

// ---------------------------------------------------------------------------
// Broadcasting helpers for binary string operations
// ---------------------------------------------------------------------------

use ferray_core::dimension::broadcast::broadcast_shapes;

/// Streaming pair iterator produced by [`broadcast_binary`] (#281).
///
/// Walks the broadcast output shape in row-major order, producing
/// `(idx_a, idx_b)` flat-index pairs on demand instead of allocating
/// the full `out_size`-sized `Vec` up front.
pub(crate) struct BroadcastIter {
    out_shape: Vec<usize>,
    shape_a: Vec<usize>,
    shape_b: Vec<usize>,
    strides_a: Vec<usize>,
    strides_b: Vec<usize>,
    out_size: usize,
    linear: usize,
}

impl Iterator for BroadcastIter {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.linear >= self.out_size {
            return None;
        }
        let multi = linear_to_multi(self.linear, &self.out_shape);
        let idx_a = multi_to_broadcast_linear(&multi, &self.shape_a, &self.strides_a);
        let idx_b = multi_to_broadcast_linear(&multi, &self.shape_b, &self.strides_b);
        self.linear += 1;
        Some((idx_a, idx_b))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.out_size - self.linear;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for BroadcastIter {}

/// Compute the broadcast result of two `StringArray`s, returning the
/// output shape and a lazy iterator of paired flat indices.
///
/// Replaces the previous version that materialized every `(idx_a, idx_b)`
/// pair into a `Vec` of size `out_shape.iter().product()`. Callers
/// iterate sequentially (see `concat::add`), so streaming pairs is
/// strictly memory-equivalent and avoids the intermediate allocation
/// for large broadcast results (#281).
pub(crate) fn broadcast_binary<Da: Dimension, Db: Dimension>(
    a: &StringArray<Da>,
    b: &StringArray<Db>,
) -> FerrayResult<(Vec<usize>, BroadcastIter)> {
    let shape_a = a.shape().to_vec();
    let shape_b = b.shape().to_vec();
    let out_shape = broadcast_shapes(&shape_a, &shape_b)?;
    let out_size: usize = out_shape.iter().product();

    let strides_a = compute_strides(&shape_a);
    let strides_b = compute_strides(&shape_b);

    let iter = BroadcastIter {
        out_shape: out_shape.clone(),
        shape_a,
        shape_b,
        strides_a,
        strides_b,
        out_size,
        linear: 0,
    };
    Ok((out_shape, iter))
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
        assert!(a.as_slice().iter().all(std::string::String::is_empty));
    }

    #[test]
    fn map_strings() {
        let a = array(&["hello", "world"]).unwrap();
        let b = a.map(str::to_uppercase).unwrap();
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
        let collected: Vec<(usize, usize)> = pairs.collect();
        assert_eq!(collected, vec![(0, 0), (1, 0)]);
    }

    #[test]
    fn broadcast_binary_same_shape() {
        let a = array(&["a", "b", "c"]).unwrap();
        let b = array(&["x", "y", "z"]).unwrap();
        let (shape, pairs) = broadcast_binary(&a, &b).unwrap();
        assert_eq!(shape, vec![3]);
        let collected: Vec<(usize, usize)> = pairs.collect();
        assert_eq!(collected, vec![(0, 0), (1, 1), (2, 2)]);
    }

    #[test]
    fn broadcast_binary_iter_size_hint() {
        // #281: ExactSizeIterator size_hint should match out_size.
        let a = array(&["hello", "world"]).unwrap();
        let b = array(&["!"]).unwrap();
        let (_shape, pairs) = broadcast_binary(&a, &b).unwrap();
        assert_eq!(pairs.size_hint(), (2, Some(2)));
        assert_eq!(pairs.len(), 2);
    }

    #[test]
    fn into_vec() {
        let a = array(&["a", "b"]).unwrap();
        let v = a.into_vec();
        assert_eq!(v, vec!["a".to_string(), "b".to_string()]);
    }

    // ---- shape operations (#514) ----

    #[test]
    fn reshape_1d_to_2d() {
        let a = array(&["a", "b", "c", "d", "e", "f"]).unwrap();
        let b = a.reshape(Ix2::new([2, 3])).unwrap();
        assert_eq!(b.shape(), &[2, 3]);
        assert_eq!(b.as_slice(), &["a", "b", "c", "d", "e", "f"]);
    }

    #[test]
    fn reshape_wrong_size_errors() {
        let a = array(&["a", "b", "c"]).unwrap();
        assert!(a.reshape(Ix2::new([2, 2])).is_err());
    }

    #[test]
    fn flatten_2d_to_1d() {
        let a = StringArray2::from_rows(&[&["a", "b"], &["c", "d"]]).unwrap();
        let f = a.flatten();
        assert_eq!(f.shape(), &[4]);
        assert_eq!(f.as_slice(), &["a", "b", "c", "d"]);
    }

    #[test]
    fn into_dyn_preserves_shape() {
        let a = StringArray2::from_rows(&[&["x", "y"], &["z", "w"]]).unwrap();
        let d = a.into_dyn();
        assert_eq!(d.shape(), &[2, 2]);
        assert_eq!(d.as_slice(), &["x", "y", "z", "w"]);
    }

    #[test]
    fn transpose_2x3() {
        // [["a","b","c"], ["d","e","f"]] -> [["a","d"], ["b","e"], ["c","f"]]
        let a = StringArray2::from_rows(&[&["a", "b", "c"], &["d", "e", "f"]]).unwrap();
        let t = a.transpose().unwrap();
        assert_eq!(t.shape(), &[3, 2]);
        assert_eq!(t.as_slice(), &["a", "d", "b", "e", "c", "f"]);
    }

    #[test]
    fn transpose_square_is_involution() {
        let a = StringArray2::from_rows(&[&["1", "2"], &["3", "4"]]).unwrap();
        let t = a.transpose().unwrap();
        let tt = t.transpose().unwrap();
        assert_eq!(tt.as_slice(), a.as_slice());
    }

    #[test]
    fn get_1d() {
        let a = array(&["zero", "one", "two"]).unwrap();
        assert_eq!(a.get(&[0]).unwrap(), "zero");
        assert_eq!(a.get(&[1]).unwrap(), "one");
        assert_eq!(a.get(&[2]).unwrap(), "two");
        assert_eq!(a.get(&[3]), None); // out of bounds
        assert_eq!(a.get(&[0, 0]), None); // wrong rank
    }

    #[test]
    fn get_2d() {
        let a = StringArray2::from_rows(&[&["a", "b", "c"], &["d", "e", "f"]]).unwrap();
        assert_eq!(a.get(&[0, 0]).unwrap(), "a");
        assert_eq!(a.get(&[0, 2]).unwrap(), "c");
        assert_eq!(a.get(&[1, 1]).unwrap(), "e");
        assert_eq!(a.get(&[2, 0]), None); // row out of bounds
        assert_eq!(a.get(&[0, 3]), None); // col out of bounds
    }
}
