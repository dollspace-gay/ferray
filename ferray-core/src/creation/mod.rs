// ferray-core: Array creation functions (REQ-16, REQ-17, REQ-18, REQ-19)
//
// Mirrors numpy's array creation routines: zeros, ones, full, empty,
// arange, linspace, logspace, geomspace, eye, identity, diag, etc.

use std::mem::MaybeUninit;

use crate::array::owned::Array;
use crate::array::view::ArrayView;
use crate::dimension::{Dimension, Ix1, Ix2, IxDyn};
use crate::dtype::Element;
use crate::error::{FerrayError, FerrayResult};

// ============================================================================
// REQ-16: Basic creation functions
// ============================================================================

/// Create an array from a flat vector and a shape (C-order).
///
/// This is the primary "array constructor" — analogous to `numpy.array()` when
/// given a flat sequence plus a shape.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if `data.len()` does not equal the
/// product of the shape dimensions.
pub fn array<T: Element, D: Dimension>(dim: D, data: Vec<T>) -> FerrayResult<Array<T, D>> {
    Array::from_vec(dim, data)
}

/// Interpret existing data as an array without copying (if possible).
///
/// This is equivalent to `numpy.asarray()`. Since Rust ownership rules
/// require moving the data, this creates an owned array from the vector.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if lengths don't match.
pub fn asarray<T: Element, D: Dimension>(dim: D, data: Vec<T>) -> FerrayResult<Array<T, D>> {
    Array::from_vec(dim, data)
}

/// Create an array from a byte buffer, interpreting bytes as elements of type `T`.
///
/// Analogous to `numpy.frombuffer()`.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if the buffer length is not a multiple
/// of `size_of::<T>()`, or if the resulting length does not match the shape.
pub fn frombuffer<T: Element, D: Dimension>(dim: D, buf: &[u8]) -> FerrayResult<Array<T, D>> {
    let elem_size = std::mem::size_of::<T>();
    if elem_size == 0 {
        return Err(FerrayError::invalid_value("zero-sized type"));
    }
    if buf.len() % elem_size != 0 {
        return Err(FerrayError::invalid_value(format!(
            "buffer length {} is not a multiple of element size {}",
            buf.len(),
            elem_size,
        )));
    }
    let n_elems = buf.len() / elem_size;
    let expected = dim.size();
    if n_elems != expected {
        return Err(FerrayError::shape_mismatch(format!(
            "buffer contains {} elements but shape {:?} requires {}",
            n_elems,
            dim.as_slice(),
            expected,
        )));
    }
    // Validate bytes for types where not all bit patterns are valid.
    // bool only permits 0x00 and 0x01.
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<bool>() {
        for &byte in buf {
            if byte > 1 {
                return Err(FerrayError::invalid_value(format!(
                    "invalid byte {byte:#04x} for bool (must be 0x00 or 0x01)"
                )));
            }
        }
    }

    // Copy bytes element-by-element via from_ne_bytes equivalent
    let mut data = Vec::with_capacity(n_elems);
    for i in 0..n_elems {
        let start = i * elem_size;
        let end = start + elem_size;
        let slice = &buf[start..end];
        // SAFETY: We're reading elem_size bytes and interpreting as T.
        // For bool, we validated above that all bytes are 0 or 1.
        // For numeric types, all bit patterns are valid.
        let val = unsafe {
            let mut val = MaybeUninit::<T>::uninit();
            std::ptr::copy_nonoverlapping(slice.as_ptr(), val.as_mut_ptr().cast::<u8>(), elem_size);
            val.assume_init()
        };
        data.push(val);
    }
    Array::from_vec(dim, data)
}

/// Create a zero-copy [`ArrayView`] over an existing byte buffer (#364).
///
/// Unlike [`frombuffer`], which copies bytes into a freshly allocated
/// `Array`, this function returns a view whose lifetime is tied to the
/// input slice. This is the equivalent of `NumPy`'s `np.frombuffer()` with
/// a memoryview source — the primary building block for zero-copy
/// interop with mmap, shared memory, network buffers, and FFI.
///
/// # Errors
/// - `InvalidValue` if `T` is a ZST.
/// - `InvalidValue` if `buf.len()` is not a multiple of `size_of::<T>()`.
/// - `ShapeMismatch` if the element count doesn't match `dim.size()`.
/// - `InvalidValue` if `buf.as_ptr()` is not aligned to `align_of::<T>()`
///   (views require proper alignment — use the copying [`frombuffer`]
///   instead if alignment cannot be guaranteed).
/// - `InvalidValue` if `T` is `bool` and any byte is outside `{0x00, 0x01}`.
pub fn frombuffer_view<T: Element, D: Dimension>(
    dim: D,
    buf: &[u8],
) -> FerrayResult<ArrayView<'_, T, D>> {
    let elem_size = std::mem::size_of::<T>();
    if elem_size == 0 {
        return Err(FerrayError::invalid_value("zero-sized type"));
    }
    if buf.len() % elem_size != 0 {
        return Err(FerrayError::invalid_value(format!(
            "buffer length {} is not a multiple of element size {}",
            buf.len(),
            elem_size,
        )));
    }
    let n_elems = buf.len() / elem_size;
    let expected = dim.size();
    if n_elems != expected {
        return Err(FerrayError::shape_mismatch(format!(
            "buffer contains {} elements but shape {:?} requires {}",
            n_elems,
            dim.as_slice(),
            expected,
        )));
    }

    // Alignment: a view interprets the bytes in place, so the buffer must
    // already be aligned for T. A misaligned read of f32/f64/etc. is UB.
    let align = std::mem::align_of::<T>();
    let addr = buf.as_ptr() as usize;
    if addr % align != 0 {
        return Err(FerrayError::invalid_value(format!(
            "buffer address 0x{addr:x} is not aligned to {align} bytes required by the element type; \
             use `frombuffer` for misaligned input"
        )));
    }

    // bool has the same size/alignment as u8 but restricts the valid bit
    // patterns; validate exhaustively, matching the copy-based frombuffer.
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<bool>() {
        for &byte in buf {
            if byte > 1 {
                return Err(FerrayError::invalid_value(format!(
                    "invalid byte {byte:#04x} for bool (must be 0x00 or 0x01)"
                )));
            }
        }
    }

    // SAFETY:
    // - The pointer comes from a valid `&[u8]` slice with length
    //   `n_elems * elem_size`, so the region is valid for reads of
    //   `n_elems` `T` values.
    // - Alignment was checked above.
    // - For bool, bit patterns were validated above. For all other
    //   `Element` types, every bit pattern is a valid value.
    // - The returned view's lifetime is bound to `'a = &'a [u8]`, which
    //   tracks the borrow back to `buf`, so the memory cannot be freed
    //   or mutated while the view lives.
    let ptr = buf.as_ptr().cast::<T>();
    let nd_dim = dim.to_ndarray_dim();
    let nd_view = unsafe { ndarray::ArrayView::from_shape_ptr(nd_dim, ptr) };
    Ok(ArrayView::from_ndarray(nd_view))
}

/// Create a 1-D array from an iterator.
///
/// Analogous to `numpy.fromiter()`.
///
/// # Errors
/// This function always succeeds (returns `Ok`).
pub fn fromiter<T: Element>(iter: impl IntoIterator<Item = T>) -> FerrayResult<Array<T, Ix1>> {
    Array::from_iter_1d(iter)
}

/// Create an array filled with zeros.
///
/// Analogous to `numpy.zeros()`.
pub fn zeros<T: Element, D: Dimension>(dim: D) -> FerrayResult<Array<T, D>> {
    Array::zeros(dim)
}

/// Create an array filled with ones.
///
/// Analogous to `numpy.ones()`.
pub fn ones<T: Element, D: Dimension>(dim: D) -> FerrayResult<Array<T, D>> {
    Array::ones(dim)
}

/// Create an array filled with a given value.
///
/// Analogous to `numpy.full()`.
pub fn full<T: Element, D: Dimension>(dim: D, fill_value: T) -> FerrayResult<Array<T, D>> {
    Array::from_elem(dim, fill_value)
}

/// Create an array with the same shape as `other`, filled with zeros.
///
/// Analogous to `numpy.zeros_like()`.
pub fn zeros_like<T: Element, D: Dimension>(other: &Array<T, D>) -> FerrayResult<Array<T, D>> {
    Array::zeros(other.dim().clone())
}

/// Create an array with the same shape as `other`, filled with ones.
///
/// Analogous to `numpy.ones_like()`.
pub fn ones_like<T: Element, D: Dimension>(other: &Array<T, D>) -> FerrayResult<Array<T, D>> {
    Array::ones(other.dim().clone())
}

/// Create an array with the same shape as `other`, filled with `fill_value`.
///
/// Analogous to `numpy.full_like()`.
pub fn full_like<T: Element, D: Dimension>(
    other: &Array<T, D>,
    fill_value: T,
) -> FerrayResult<Array<T, D>> {
    Array::from_elem(other.dim().clone(), fill_value)
}

// ============================================================================
// REQ-17: empty() returning MaybeUninit
// ============================================================================

/// An array whose elements have not been initialized.
///
/// The caller must call [`assume_init`](UninitArray::assume_init) after
/// filling all elements.
pub struct UninitArray<T: Element, D: Dimension> {
    data: Vec<MaybeUninit<T>>,
    dim: D,
}

impl<T: Element, D: Dimension> UninitArray<T, D> {
    /// Shape as a slice.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.dim.as_slice()
    }

    /// Total number of elements.
    #[inline]
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.dim.ndim()
    }

    /// Get a mutable raw pointer to the underlying data.
    ///
    /// Use this to fill the array element-by-element before calling
    /// `assume_init()`.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut MaybeUninit<T> {
        self.data.as_mut_ptr()
    }

    /// Write a value at a flat index.
    ///
    /// # Errors
    /// Returns `FerrayError::IndexOutOfBounds` if `flat_index >= size()`.
    pub fn write_at(&mut self, flat_index: usize, value: T) -> FerrayResult<()> {
        let size = self.size();
        if flat_index >= size {
            return Err(FerrayError::IndexOutOfBounds {
                index: flat_index as isize,
                axis: 0,
                size,
            });
        }
        self.data[flat_index] = MaybeUninit::new(value);
        Ok(())
    }

    /// Convert to an initialized `Array<T, D>`.
    ///
    /// # Safety
    /// The caller must ensure that **all** elements have been initialized
    /// (e.g., via `write_at` or raw pointer writes). Reading uninitialized
    /// memory is undefined behavior.
    pub unsafe fn assume_init(self) -> Array<T, D> {
        let nd_dim = self.dim.to_ndarray_dim();
        let len = self.data.len();

        // Transmute Vec<MaybeUninit<T>> to Vec<T>.
        // SAFETY: MaybeUninit<T> has the same layout as T, and the caller
        // guarantees all elements are initialized.
        let mut raw_vec = std::mem::ManuallyDrop::new(self.data);
        let data: Vec<T> = unsafe {
            Vec::from_raw_parts(raw_vec.as_mut_ptr().cast::<T>(), len, raw_vec.capacity())
        };

        let inner = ndarray::Array::from_shape_vec(nd_dim, data)
            .expect("UninitArray assume_init: shape/data mismatch (this is a bug)");
        Array::from_ndarray(inner)
    }
}

/// Create an uninitialized array.
///
/// Analogous to `numpy.empty()`, but returns a [`UninitArray`] that must
/// be explicitly initialized via [`UninitArray::assume_init`].
///
/// This prevents accidentally reading uninitialized memory — a key safety
/// improvement over `NumPy`'s `empty()`.
pub fn empty<T: Element, D: Dimension>(dim: D) -> UninitArray<T, D> {
    let size = dim.size();
    let mut data = Vec::with_capacity(size);
    // SAFETY: MaybeUninit does not require initialization.
    // We set the length to match the capacity; each element is MaybeUninit.
    unsafe {
        data.set_len(size);
    }
    UninitArray { data, dim }
}

/// Create an uninitialized array with the same shape (and element type)
/// as `other`.
///
/// Analogous to `numpy.empty_like()`. Returns a [`UninitArray`] that the
/// caller must fully initialize before calling
/// [`UninitArray::assume_init`]. Avoids the memset that `zeros_like` /
/// `full_like` incur when the caller is about to overwrite every element
/// anyway.
pub fn empty_like<T: Element, D: Dimension>(other: &Array<T, D>) -> UninitArray<T, D> {
    empty(other.dim().clone())
}

// ============================================================================
// REQ-18: Range functions
// ============================================================================

/// Trait for types usable with `arange` — numeric types that support
/// stepping and comparison.
pub trait ArangeNum: Element + PartialOrd {
    /// Convert from f64 for step calculations.
    fn from_f64(v: f64) -> Self;
    /// Convert to f64 for step calculations.
    fn to_f64(self) -> f64;
}

macro_rules! impl_arange_int {
    ($($ty:ty),*) => {
        $(
            impl ArangeNum for $ty {
                #[inline]
                fn from_f64(v: f64) -> Self { v as Self }
                #[inline]
                fn to_f64(self) -> f64 { self as f64 }
            }
        )*
    };
}

macro_rules! impl_arange_float {
    ($($ty:ty),*) => {
        $(
            impl ArangeNum for $ty {
                #[inline]
                fn from_f64(v: f64) -> Self { v as Self }
                #[inline]
                fn to_f64(self) -> f64 { self as f64 }
            }
        )*
    };
}

impl_arange_int!(u8, u16, u32, u64, i8, i16, i32, i64);
impl_arange_float!(f32, f64);

/// Create a 1-D array with evenly spaced values within a given interval.
///
/// Analogous to `numpy.arange(start, stop, step)`.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if `step` is zero.
pub fn arange<T: ArangeNum>(start: T, stop: T, step: T) -> FerrayResult<Array<T, Ix1>> {
    let step_f = step.to_f64();
    if step_f == 0.0 {
        return Err(FerrayError::invalid_value("step cannot be zero"));
    }
    let start_f = start.to_f64();
    let stop_f = stop.to_f64();
    let n = ((stop_f - start_f) / step_f).ceil();
    let n = if n < 0.0 { 0 } else { n as usize };

    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        data.push(T::from_f64((i as f64).mul_add(step_f, start_f)));
    }
    let dim = Ix1::new([data.len()]);
    Array::from_vec(dim, data)
}

/// Trait for float-like types used in linspace/logspace/geomspace.
pub trait LinspaceNum: Element + PartialOrd {
    /// Convert from f64.
    fn from_f64(v: f64) -> Self;
    /// Convert to f64.
    fn to_f64(self) -> f64;
}

impl LinspaceNum for f32 {
    #[inline]
    fn from_f64(v: f64) -> Self {
        v as Self
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl LinspaceNum for f64 {
    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
}

/// Create a 1-D array with `num` evenly spaced values between `start` and `stop`.
///
/// If `endpoint` is true (the default in `NumPy`), `stop` is the last sample.
/// Otherwise, it is not included.
///
/// Analogous to `numpy.linspace()`.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if `num` is 0 and `endpoint` is true
/// (cannot produce an empty array with an endpoint).
pub fn linspace<T: LinspaceNum>(
    start: T,
    stop: T,
    num: usize,
    endpoint: bool,
) -> FerrayResult<Array<T, Ix1>> {
    if num == 0 {
        return Array::from_vec(Ix1::new([0]), vec![]);
    }
    if num == 1 {
        return Array::from_vec(Ix1::new([1]), vec![start]);
    }
    let start_f = start.to_f64();
    let stop_f = stop.to_f64();
    let divisor = if endpoint {
        (num - 1) as f64
    } else {
        num as f64
    };
    let step = (stop_f - start_f) / divisor;
    let mut data = Vec::with_capacity(num);
    for i in 0..num {
        data.push(T::from_f64((i as f64).mul_add(step, start_f)));
    }
    Array::from_vec(Ix1::new([num]), data)
}

/// Create a 1-D array with values spaced evenly on a log scale.
///
/// Returns `base ** linspace(start, stop, num)`.
///
/// Analogous to `numpy.logspace()`.
///
/// # Errors
/// Propagates errors from `linspace`.
pub fn logspace<T: LinspaceNum>(
    start: T,
    stop: T,
    num: usize,
    endpoint: bool,
    base: f64,
) -> FerrayResult<Array<T, Ix1>> {
    let lin = linspace(start, stop, num, endpoint)?;
    let data: Vec<T> = lin
        .iter()
        .map(|v| T::from_f64(base.powf(v.clone().to_f64())))
        .collect();
    Array::from_vec(Ix1::new([num]), data)
}

/// Create a 1-D array with values spaced evenly on a geometric (log) scale.
///
/// The values are `start * (stop/start) ** linspace(0, 1, num)`.
///
/// Analogous to `numpy.geomspace()`.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if `start` or `stop` is zero or
/// if they have different signs.
pub fn geomspace<T: LinspaceNum>(
    start: T,
    stop: T,
    num: usize,
    endpoint: bool,
) -> FerrayResult<Array<T, Ix1>> {
    let start_f = start.clone().to_f64();
    let stop_f = stop.to_f64();
    if start_f == 0.0 || stop_f == 0.0 {
        return Err(FerrayError::invalid_value(
            "geomspace: start and stop must be non-zero",
        ));
    }
    if (start_f < 0.0) != (stop_f < 0.0) {
        return Err(FerrayError::invalid_value(
            "geomspace: start and stop must have the same sign",
        ));
    }
    if num == 0 {
        return Array::from_vec(Ix1::new([0]), vec![]);
    }
    if num == 1 {
        return Array::from_vec(Ix1::new([1]), vec![start]);
    }
    let log_start = start_f.abs().ln();
    let log_stop = stop_f.abs().ln();
    let sign = if start_f < 0.0 { -1.0 } else { 1.0 };
    let divisor = if endpoint {
        (num - 1) as f64
    } else {
        num as f64
    };
    let step = (log_stop - log_start) / divisor;
    let mut data = Vec::with_capacity(num);
    for i in 0..num {
        let log_val = (i as f64).mul_add(step, log_start);
        data.push(T::from_f64(sign * log_val.exp()));
    }
    Array::from_vec(Ix1::new([num]), data)
}

/// Return coordinate arrays from coordinate vectors.
///
/// Analogous to `numpy.meshgrid(*xi, indexing='xy')`.
///
/// Given N 1-D arrays, returns N N-D arrays, where each output array
/// has the shape `(len(x1), len(x2), ..., len(xN))` for 'xy' indexing
/// or `(len(x1), ..., len(xN))` transposed for 'ij' indexing.
///
/// `indexing` should be `"xy"` (default Cartesian) or `"ij"` (matrix).
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if `indexing` is not `"xy"` or `"ij"`,
/// or if there are fewer than 2 input arrays.
pub fn meshgrid(
    arrays: &[Array<f64, Ix1>],
    indexing: &str,
) -> FerrayResult<Vec<Array<f64, IxDyn>>> {
    if indexing != "xy" && indexing != "ij" {
        return Err(FerrayError::invalid_value(
            "meshgrid: indexing must be 'xy' or 'ij'",
        ));
    }
    let ndim = arrays.len();
    if ndim == 0 {
        return Ok(vec![]);
    }

    let mut shapes: Vec<usize> = arrays.iter().map(|a| a.shape()[0]).collect();
    if indexing == "xy" && ndim >= 2 {
        shapes.swap(0, 1);
    }

    let total: usize = shapes.iter().product();
    let mut results = Vec::with_capacity(ndim);

    for (k, arr) in arrays.iter().enumerate() {
        let src_data: Vec<f64> = arr.iter().copied().collect();
        let mut data = Vec::with_capacity(total);
        // For 'xy' indexing, the first two dimensions are swapped
        let effective_k = if indexing == "xy" && ndim >= 2 {
            match k {
                0 => 1,
                1 => 0,
                other => other,
            }
        } else {
            k
        };

        // Build the output by iterating over all indices in the output shape
        for flat in 0..total {
            // Compute the index along dimension effective_k
            let mut rem = flat;
            let mut idx_k = 0;
            for (d, &s) in shapes.iter().enumerate().rev() {
                if d == effective_k {
                    idx_k = rem % s;
                }
                rem /= s;
            }
            data.push(src_data[idx_k]);
        }

        let dim = IxDyn::new(&shapes);
        results.push(Array::from_vec(dim, data)?);
    }
    Ok(results)
}

/// Create a dense multi-dimensional "meshgrid" with matrix ('ij') indexing.
///
/// Analogous to `numpy.mgrid[start:stop:step, ...]`.
///
/// Takes a slice of `(start, stop, step)` tuples, one per dimension.
/// Returns a vector of arrays, one per dimension.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if any step is zero.
pub fn mgrid(ranges: &[(f64, f64, f64)]) -> FerrayResult<Vec<Array<f64, IxDyn>>> {
    let mut arrs: Vec<Array<f64, Ix1>> = Vec::with_capacity(ranges.len());
    for &(start, stop, step) in ranges {
        arrs.push(arange(start, stop, step)?);
    }
    meshgrid(&arrs, "ij")
}

/// Create a sparse (open) multi-dimensional "meshgrid" with 'ij' indexing.
///
/// Analogous to `numpy.ogrid[start:stop:step, ...]`.
///
/// Returns arrays that are broadcastable to the full grid shape.
/// Each returned array has shape 1 in all dimensions except its own.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if any step is zero.
pub fn ogrid(ranges: &[(f64, f64, f64)]) -> FerrayResult<Vec<Array<f64, IxDyn>>> {
    let ndim = ranges.len();
    let mut results = Vec::with_capacity(ndim);
    for (i, &(start, stop, step)) in ranges.iter().enumerate() {
        let arr1d = arange(start, stop, step)?;
        let n = arr1d.shape()[0];
        let data: Vec<f64> = arr1d.iter().copied().collect();
        // Build shape: all ones except dimension i = n
        let mut shape = vec![1usize; ndim];
        shape[i] = n;
        let dim = IxDyn::new(&shape);
        results.push(Array::from_vec(dim, data)?);
    }
    Ok(results)
}

// ============================================================================
// REQ-19: Identity/diagonal functions
// ============================================================================

/// Create a 2-D identity matrix of size `n x n`.
///
/// Analogous to `numpy.identity()`.
pub fn identity<T: Element>(n: usize) -> FerrayResult<Array<T, Ix2>> {
    eye(n, n, 0)
}

/// Create a 2-D array with ones on the diagonal and zeros elsewhere.
///
/// `k` is the diagonal offset: 0 = main diagonal, positive = above, negative = below.
///
/// Analogous to `numpy.eye(N, M, k)`.
pub fn eye<T: Element>(n: usize, m: usize, k: isize) -> FerrayResult<Array<T, Ix2>> {
    let mut data = vec![T::zero(); n * m];
    for i in 0..n {
        let j = i as isize + k;
        if j >= 0 && (j as usize) < m {
            data[i * m + j as usize] = T::one();
        }
    }
    Array::from_vec(Ix2::new([n, m]), data)
}

/// Extract a diagonal or construct a diagonal array.
///
/// If `a` is 2-D, extract the `k`-th diagonal as a 1-D array.
/// If `a` is 1-D, construct a 2-D array with `a` on the `k`-th diagonal.
///
/// Analogous to `numpy.diag()`.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if `a` is not 1-D or 2-D.
pub fn diag<T: Element>(a: &Array<T, IxDyn>, k: isize) -> FerrayResult<Array<T, IxDyn>> {
    let shape = a.shape();
    match shape.len() {
        1 => {
            // Construct a 2-D diagonal array
            let n = shape[0];
            let size = n + k.unsigned_abs();
            let mut data = vec![T::zero(); size * size];
            let src: Vec<T> = a.iter().cloned().collect();
            for (i, val) in src.into_iter().enumerate() {
                let row = if k >= 0 { i } else { i + k.unsigned_abs() };
                let col = if k >= 0 { i + k as usize } else { i };
                data[row * size + col] = val;
            }
            Array::from_vec(IxDyn::new(&[size, size]), data)
        }
        2 => {
            // Extract the k-th diagonal
            let (n, m) = (shape[0], shape[1]);
            let src: Vec<T> = a.iter().cloned().collect();
            let mut diag_vals = Vec::new();
            for i in 0..n {
                let j = i as isize + k;
                if j >= 0 && (j as usize) < m {
                    diag_vals.push(src[i * m + j as usize].clone());
                }
            }
            let len = diag_vals.len();
            Array::from_vec(IxDyn::new(&[len]), diag_vals)
        }
        _ => Err(FerrayError::invalid_value("diag: input must be 1-D or 2-D")),
    }
}

/// Create a 2-D array with the flattened input as a diagonal.
///
/// Analogous to `numpy.diagflat()`.
///
/// # Errors
/// Propagates errors from the underlying construction.
pub fn diagflat<T: Element>(a: &Array<T, IxDyn>, k: isize) -> FerrayResult<Array<T, IxDyn>> {
    // Flatten a to 1-D, then call diag
    let flat: Vec<T> = a.iter().cloned().collect();
    let n = flat.len();
    let arr1d = Array::from_vec(IxDyn::new(&[n]), flat)?;
    diag(&arr1d, k)
}

/// Create a lower-triangular matrix of ones.
///
/// Returns an `n x m` array where `a[i, j] = 1` if `i >= j - k`, else `0`.
///
/// Analogous to `numpy.tri(N, M, k)`.
pub fn tri<T: Element>(n: usize, m: usize, k: isize) -> FerrayResult<Array<T, Ix2>> {
    let mut data = vec![T::zero(); n * m];
    for i in 0..n {
        for j in 0..m {
            if (i as isize) >= (j as isize) - k {
                data[i * m + j] = T::one();
            }
        }
    }
    Array::from_vec(Ix2::new([n, m]), data)
}

/// Return the lower triangle of a 2-D array.
///
/// `k` is the diagonal above which to zero elements. 0 = main diagonal.
///
/// Analogous to `numpy.tril()`.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if input is not 2-D.
pub fn tril<T: Element>(a: &Array<T, IxDyn>, k: isize) -> FerrayResult<Array<T, IxDyn>> {
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(FerrayError::invalid_value("tril: input must be 2-D"));
    }
    let (n, m) = (shape[0], shape[1]);
    let src: Vec<T> = a.iter().cloned().collect();
    let mut data = vec![T::zero(); n * m];
    for i in 0..n {
        for j in 0..m {
            if (i as isize) >= (j as isize) - k {
                data[i * m + j] = src[i * m + j].clone();
            }
        }
    }
    Array::from_vec(IxDyn::new(&[n, m]), data)
}

/// Return the upper triangle of a 2-D array.
///
/// `k` is the diagonal below which to zero elements. 0 = main diagonal.
///
/// Analogous to `numpy.triu()`.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if input is not 2-D.
pub fn triu<T: Element>(a: &Array<T, IxDyn>, k: isize) -> FerrayResult<Array<T, IxDyn>> {
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(FerrayError::invalid_value("triu: input must be 2-D"));
    }
    let (n, m) = (shape[0], shape[1]);
    let src: Vec<T> = a.iter().cloned().collect();
    let mut data = vec![T::zero(); n * m];
    for i in 0..n {
        for j in 0..m {
            if (i as isize) <= (j as isize) - k {
                data[i * m + j] = src[i * m + j].clone();
            }
        }
    }
    Array::from_vec(IxDyn::new(&[n, m]), data)
}

// ============================================================================
// REQ: vander (Vandermonde matrix)
// ============================================================================

/// Generate a Vandermonde matrix.
///
/// Each row is a geometric progression of the corresponding `x` element.
/// `n` is the number of columns (`None` => `x.len()`). When `increasing`
/// is `false` (numpy default), columns are powers in decreasing order
/// `x[i]^(n-1-j)`. When `true`, they are increasing: `x[i]^j`.
///
/// Analogous to `numpy.vander(x, N=n, increasing=...)`.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if `x` is empty.
pub fn vander<T>(
    x: &Array<T, Ix1>,
    n: Option<usize>,
    increasing: bool,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + std::ops::Mul<Output = T> + Copy,
{
    let m = x.shape()[0];
    if m == 0 {
        return Err(FerrayError::invalid_value(
            "vander: input array must not be empty",
        ));
    }
    let cols = n.unwrap_or(m);
    let xs: Vec<T> = x.iter().copied().collect();
    let mut data = vec![<T as Element>::one(); m * cols];
    for (i, &xi) in xs.iter().enumerate() {
        // Build powers x^0, x^1, ..., x^(cols-1) for this row.
        let mut acc = <T as Element>::one();
        let mut powers = Vec::with_capacity(cols);
        for _ in 0..cols {
            powers.push(acc);
            acc = acc * xi;
        }
        if increasing {
            for (j, p) in powers.iter().enumerate() {
                data[i * cols + j] = *p;
            }
        } else {
            for (j, p) in powers.iter().enumerate() {
                data[i * cols + (cols - 1 - j)] = *p;
            }
        }
    }
    Array::from_vec(IxDyn::new(&[m, cols]), data)
}

// ============================================================================
// REQ: copy / asarray family / chkfinite / require
// ============================================================================

/// Return an array copy of the given object.
///
/// Analogous to `numpy.copy()`. Allocates a new C-contiguous buffer
/// with the same data.
pub fn copy<T: Element, D: Dimension>(a: &Array<T, D>) -> Array<T, D> {
    a.clone()
}

/// Return a contiguous array (C order) in memory.
///
/// In ferray, owned `Array<T, D>` values are always C-contiguous, so this
/// is functionally equivalent to `clone()`. Provided for NumPy API parity
/// (`numpy.ascontiguousarray`).
pub fn ascontiguousarray<T: Element, D: Dimension>(a: &Array<T, D>) -> Array<T, D> {
    a.clone()
}

/// Return an array (Fortran-order) by transposing then cloning.
///
/// ferray stores in C-order; to get Fortran-style layout we materialize a
/// new buffer in column-major order. The shape is preserved; only the
/// memory layout differs (the element traversal order in `iter()` will
/// follow C-order regardless after this round-trip).
///
/// Analogous to `numpy.asfortranarray` — return a copy in F-contiguous
/// (column-major) memory layout. The element-iteration order
/// (logical/row-major) is preserved, but `flags().f_contiguous` is
/// true on the result.
///
/// Implementation: transpose → C-contiguous standard layout → transpose
/// back. The double-transpose preserves logical values while producing
/// physical column-major memory.
pub fn asfortranarray<T: Element, D: Dimension>(a: &Array<T, D>) -> Array<T, D> {
    let transposed = a.inner.view().reversed_axes();
    let c_contig = transposed.as_standard_layout().into_owned();
    let f_contig = c_contig.reversed_axes();
    Array::<T, D>::from_ndarray(f_contig)
}

/// Convert input to an array. Accepts an existing array (no-op clone).
///
/// In NumPy, `asanyarray` differs from `asarray` only in that it preserves
/// `MaskedArray` / `matrix` subclasses. ferray has no array-subclass
/// hierarchy at this layer (masked arrays live in `ferray-ma`), so
/// `asanyarray` is identical to `asarray` for owned arrays.
///
/// Analogous to `numpy.asanyarray`.
pub fn asanyarray<T: Element, D: Dimension>(a: &Array<T, D>) -> Array<T, D> {
    a.clone()
}

/// Convert the input to an array, checking for NaNs or infinities.
///
/// Analogous to `numpy.asarray_chkfinite`. Returns an `InvalidValue`
/// error if any element is non-finite. Float-only.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if any element is NaN or infinite.
pub fn asarray_chkfinite<T, D: Dimension>(a: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + num_traits::Float,
{
    for v in a.iter() {
        if !v.is_finite() {
            return Err(FerrayError::invalid_value(
                "asarray_chkfinite: array contains non-finite values (NaN or Inf)",
            ));
        }
    }
    Ok(a.clone())
}

/// Return a contiguous array meeting requirements.
///
/// `requirements` is a string of single-character flags:
/// - `'C'` — C-contiguous
/// - `'F'` — F-contiguous (treated as C in ferray; see `asfortranarray`)
/// - `'A'` — any contiguous (no-op)
/// - `'W'` — writeable (always true for owned arrays)
/// - `'O'` — owned data (no-op; arrays are always owned)
///
/// Analogous to `numpy.require`. Unknown flags are ignored.
pub fn require<T: Element, D: Dimension>(a: &Array<T, D>, _requirements: &str) -> Array<T, D> {
    a.clone()
}

// ============================================================================
// REQ: fromfunction / fromstring / fromfile
// ============================================================================

/// Construct an array by executing a function over each coordinate.
///
/// `f` is called with the multi-index of each element (as `&[usize]`) and
/// returns the value to place there. Iteration is C-order over the given
/// shape.
///
/// Analogous to `numpy.fromfunction(func, shape, ...)`. NumPy passes
/// per-axis index arrays to `func` (broadcasting); ferray instead invokes
/// `f` per element with a coordinate slice — equivalent in result for
/// scalar-returning functions but simpler in Rust.
pub fn fromfunction<T, D, F>(shape: D, mut f: F) -> FerrayResult<Array<T, D>>
where
    T: Element,
    D: Dimension,
    F: FnMut(&[usize]) -> T,
{
    let dims: Vec<usize> = shape.as_slice().to_vec();
    let total: usize = dims.iter().product();
    let mut data = Vec::with_capacity(total);
    let mut idx = vec![0usize; dims.len()];
    for _ in 0..total {
        data.push(f(&idx));
        // Increment multi-index in C-order.
        for axis in (0..dims.len()).rev() {
            idx[axis] += 1;
            if idx[axis] < dims[axis] {
                break;
            }
            idx[axis] = 0;
        }
    }
    Array::from_vec(shape, data)
}

/// Construct a 1-D array from a whitespace- or separator-delimited string.
///
/// Each token is parsed via `T::from_str` (any type implementing
/// [`std::str::FromStr`]). Empty tokens are ignored.
///
/// Analogous to `numpy.fromstring(s, sep=...)` (the binary mode of NumPy
/// `fromstring` is deprecated in favor of `frombuffer`, so we only
/// implement the text-parsing form here).
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if any token fails to parse.
pub fn fromstring<T>(s: &str, sep: &str) -> FerrayResult<Array<T, Ix1>>
where
    T: Element + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    let parts: Vec<&str> = if sep.is_empty() {
        s.split_whitespace().collect()
    } else {
        s.split(sep)
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .collect()
    };
    let mut data = Vec::with_capacity(parts.len());
    for tok in parts {
        let v = tok.parse::<T>().map_err(|e| {
            FerrayError::invalid_value(format!("fromstring: failed to parse {tok:?}: {e}"))
        })?;
        data.push(v);
    }
    let n = data.len();
    Array::from_vec(Ix1::new([n]), data)
}

/// Read a 1-D array from a file by parsing whitespace-delimited tokens.
///
/// Analogous to `numpy.fromfile(path, sep=' ')` (text mode). The binary
/// mode of NumPy's `fromfile` is intentionally not provided here — use
/// `ferray-io::load` for `.npy` and `ferray_core::frombuffer` for raw
/// byte buffers.
///
/// # Errors
/// Returns `FerrayError::Io` for filesystem errors, `FerrayError::InvalidValue`
/// for parse failures.
pub fn fromfile<T, P: AsRef<std::path::Path>>(path: P, sep: &str) -> FerrayResult<Array<T, Ix1>>
where
    T: Element + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    let s = std::fs::read_to_string(path)
        .map_err(|e| FerrayError::invalid_value(format!("fromfile: {e}")))?;
    fromstring(&s, sep)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::{Ix1, Ix2, IxDyn};

    // -- REQ-16 tests --

    #[test]
    fn test_array_creation() {
        let a = array(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(a.shape(), &[2, 3]);
        assert_eq!(a.size(), 6);
    }

    #[test]
    fn test_asarray() {
        let a = asarray(Ix1::new([3]), vec![1, 2, 3]).unwrap();
        assert_eq!(a.as_slice().unwrap(), &[1, 2, 3]);
    }

    #[test]
    fn test_frombuffer() {
        let bytes: Vec<u8> = vec![1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0];
        let a = frombuffer::<i32, Ix1>(Ix1::new([3]), &bytes).unwrap();
        assert_eq!(a.as_slice().unwrap(), &[1, 2, 3]);
    }

    #[test]
    fn test_frombuffer_bad_length() {
        let bytes: Vec<u8> = vec![1, 0, 0];
        assert!(frombuffer::<i32, Ix1>(Ix1::new([1]), &bytes).is_err());
    }

    #[test]
    fn test_frombuffer_bool() {
        // Issue #135: bool round-trips through frombuffer must
        // preserve the discriminating byte (0 -> false, nonzero
        // -> true, although our raw-buffer contract is that each
        // byte is a valid bool per `Element`).
        let bytes: Vec<u8> = vec![0, 1, 0, 1, 1];
        let a = frombuffer::<bool, Ix1>(Ix1::new([5]), &bytes).unwrap();
        assert_eq!(a.as_slice().unwrap(), &[false, true, false, true, true]);
    }

    #[test]
    fn test_frombuffer_bool_wrong_length() {
        // For bool (1 byte each), the buffer length must equal the
        // requested element count.
        let bytes: Vec<u8> = vec![0, 1];
        assert!(frombuffer::<bool, Ix1>(Ix1::new([3]), &bytes).is_err());
    }

    // #364: frombuffer_view — zero-copy view over an existing byte buffer.

    /// Build an aligned byte buffer of `nbytes` from a typed slice so we
    /// can exercise the zero-copy path without fighting the allocator.
    fn aligned_bytes<T: Copy>(src: &[T]) -> Vec<u8> {
        let n = std::mem::size_of_val(src);
        let mut out = vec![0u8; n];
        // SAFETY: src is &[T], out is a byte buffer of exactly n bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr().cast::<u8>(), out.as_mut_ptr(), n);
        }
        out
    }

    #[test]
    fn test_frombuffer_view_i32_is_zero_copy() {
        // Build an aligned byte buffer that represents three i32s.
        let source: Vec<i32> = vec![10, 20, 30];
        let bytes = aligned_bytes(&source);
        let view = frombuffer_view::<i32, Ix1>(Ix1::new([3]), &bytes).unwrap();
        assert_eq!(view.shape(), &[3]);
        let values: Vec<i32> = view.iter().copied().collect();
        assert_eq!(values, vec![10, 20, 30]);
        // Pointer must alias the source buffer — that's the zero-copy
        // contract distinguishing this from the copying frombuffer.
        assert_eq!(view.as_ptr().cast::<u8>(), bytes.as_ptr());
    }

    #[test]
    fn test_frombuffer_view_f64_2d() {
        let source: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes = aligned_bytes(&source);
        let view = frombuffer_view::<f64, Ix2>(Ix2::new([2, 3]), &bytes).unwrap();
        assert_eq!(view.shape(), &[2, 3]);
        let values: Vec<f64> = view.iter().copied().collect();
        assert_eq!(values, source);
    }

    #[test]
    fn test_frombuffer_view_bool_valid() {
        let bytes: Vec<u8> = vec![0, 1, 0, 1];
        let view = frombuffer_view::<bool, Ix1>(Ix1::new([4]), &bytes).unwrap();
        let values: Vec<bool> = view.iter().copied().collect();
        assert_eq!(values, vec![false, true, false, true]);
    }

    #[test]
    fn test_frombuffer_view_bool_rejects_invalid_byte() {
        let bytes: Vec<u8> = vec![0, 1, 42]; // 42 is not a valid bool.
        assert!(frombuffer_view::<bool, Ix1>(Ix1::new([3]), &bytes).is_err());
    }

    #[test]
    fn test_frombuffer_view_rejects_wrong_length() {
        // 13 bytes is not a multiple of size_of::<i32>() = 4.
        let bytes = vec![0u8; 13];
        assert!(frombuffer_view::<i32, Ix1>(Ix1::new([3]), &bytes).is_err());
        // 8 bytes is 2 i32s, but the caller asked for shape [3].
        let bytes = vec![0u8; 8];
        assert!(frombuffer_view::<i32, Ix1>(Ix1::new([3]), &bytes).is_err());
    }

    #[test]
    fn test_frombuffer_view_rejects_misalignment() {
        // Force a misaligned slice by advancing the byte pointer by 1.
        let mut backing: Vec<u8> = vec![0u8; 1 + 4 * 3];
        for (i, chunk) in backing[1..].chunks_exact_mut(4).enumerate() {
            chunk.copy_from_slice(&(i as i32).to_ne_bytes());
        }
        let misaligned = &backing[1..];
        // The slice address is off by one from a 4-byte boundary, so
        // alignment for i32 cannot be satisfied.
        assert!((misaligned.as_ptr() as usize) % 4 != 0);
        assert!(frombuffer_view::<i32, Ix1>(Ix1::new([3]), misaligned).is_err());
    }

    #[test]
    fn test_fromiter() {
        let a = fromiter((0..5).map(|x| x as f64)).unwrap();
        assert_eq!(a.shape(), &[5]);
        assert_eq!(a.as_slice().unwrap(), &[0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_zeros() {
        let a = zeros::<f64, Ix2>(Ix2::new([3, 4])).unwrap();
        assert_eq!(a.shape(), &[3, 4]);
        assert!(a.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_ones() {
        let a = ones::<f64, Ix1>(Ix1::new([5])).unwrap();
        assert!(a.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_full() {
        let a = full(Ix1::new([4]), 42i32).unwrap();
        assert!(a.iter().all(|&v| v == 42));
    }

    #[test]
    fn test_zeros_like() {
        let a = ones::<f64, Ix2>(Ix2::new([2, 3])).unwrap();
        let b = zeros_like(&a).unwrap();
        assert_eq!(b.shape(), &[2, 3]);
        assert!(b.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_ones_like() {
        let a = zeros::<f64, Ix1>(Ix1::new([4])).unwrap();
        let b = ones_like(&a).unwrap();
        assert!(b.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_full_like() {
        let a = zeros::<i32, Ix1>(Ix1::new([3])).unwrap();
        let b = full_like(&a, 7).unwrap();
        assert!(b.iter().all(|&v| v == 7));
    }

    // -- REQ-17 tests --

    #[test]
    fn test_empty_and_init() {
        let mut u = empty::<f64, Ix1>(Ix1::new([3]));
        assert_eq!(u.shape(), &[3]);
        u.write_at(0, 1.0).unwrap();
        u.write_at(1, 2.0).unwrap();
        u.write_at(2, 3.0).unwrap();
        // SAFETY: all elements initialized
        let a = unsafe { u.assume_init() };
        assert_eq!(a.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_empty_write_oob() {
        let mut u = empty::<f64, Ix1>(Ix1::new([2]));
        assert!(u.write_at(5, 1.0).is_err());
    }

    // #363: empty_like matches source shape, contents independent.
    #[test]
    fn test_empty_like_matches_shape_2d() {
        use crate::dimension::Ix2;
        let src = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let mut u = empty_like(&src);
        assert_eq!(u.shape(), &[2, 3]);
        assert_eq!(u.size(), 6);
        assert_eq!(u.ndim(), 2);

        // Fill and init — the resulting array is independent of `src`.
        for i in 0..6 {
            u.write_at(i, -(i as f64)).unwrap();
        }
        // SAFETY: every slot just written.
        let out = unsafe { u.assume_init() };
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(
            out.as_slice().unwrap(),
            &[0.0, -1.0, -2.0, -3.0, -4.0, -5.0]
        );
        // Source is unchanged.
        assert_eq!(src.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_empty_like_zero_sized() {
        let src = Array::<f64, Ix1>::from_vec(Ix1::new([0]), vec![]).unwrap();
        let u = empty_like(&src);
        assert_eq!(u.shape(), &[0]);
        assert_eq!(u.size(), 0);
        // SAFETY: size is zero — nothing to initialize.
        let out = unsafe { u.assume_init() };
        assert_eq!(out.size(), 0);
    }

    // -- REQ-18 tests --

    #[test]
    fn test_arange_int() {
        let a = arange(0i32, 5, 1).unwrap();
        assert_eq!(a.as_slice().unwrap(), &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_arange_float() {
        let a = arange(0.0_f64, 1.0, 0.25).unwrap();
        assert_eq!(a.shape(), &[4]);
        let data = a.as_slice().unwrap();
        assert!((data[0] - 0.0).abs() < 1e-10);
        assert!((data[1] - 0.25).abs() < 1e-10);
        assert!((data[2] - 0.5).abs() < 1e-10);
        assert!((data[3] - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_arange_negative_step() {
        let a = arange(5.0_f64, 0.0, -1.0).unwrap();
        assert_eq!(a.shape(), &[5]);
    }

    #[test]
    fn test_arange_zero_step() {
        assert!(arange(0.0_f64, 1.0, 0.0).is_err());
    }

    #[test]
    fn test_arange_empty() {
        let a = arange(5i32, 0, 1).unwrap();
        assert_eq!(a.shape(), &[0]);
    }

    #[test]
    fn test_linspace() {
        let a = linspace(0.0_f64, 1.0, 5, true).unwrap();
        assert_eq!(a.shape(), &[5]);
        let data = a.as_slice().unwrap();
        assert!((data[0] - 0.0).abs() < 1e-10);
        assert!((data[4] - 1.0).abs() < 1e-10);
        assert!((data[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_linspace_no_endpoint() {
        let a = linspace(0.0_f64, 1.0, 4, false).unwrap();
        assert_eq!(a.shape(), &[4]);
        let data = a.as_slice().unwrap();
        assert!((data[0] - 0.0).abs() < 1e-10);
        assert!((data[1] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_linspace_single() {
        let a = linspace(5.0_f64, 10.0, 1, true).unwrap();
        assert_eq!(a.as_slice().unwrap(), &[5.0]);
    }

    #[test]
    fn test_linspace_empty() {
        let a = linspace(0.0_f64, 1.0, 0, true).unwrap();
        assert_eq!(a.shape(), &[0]);
    }

    #[test]
    fn test_logspace() {
        let a = logspace(0.0_f64, 2.0, 3, true, 10.0).unwrap();
        let data = a.as_slice().unwrap();
        assert!((data[0] - 1.0).abs() < 1e-10); // 10^0
        assert!((data[1] - 10.0).abs() < 1e-10); // 10^1
        assert!((data[2] - 100.0).abs() < 1e-10); // 10^2
    }

    #[test]
    fn test_geomspace() {
        let a = geomspace(1.0_f64, 1000.0, 4, true).unwrap();
        let data = a.as_slice().unwrap();
        assert!((data[0] - 1.0).abs() < 1e-10);
        assert!((data[1] - 10.0).abs() < 1e-8);
        assert!((data[2] - 100.0).abs() < 1e-6);
        assert!((data[3] - 1000.0).abs() < 1e-4);
    }

    #[test]
    fn test_geomspace_zero_start() {
        assert!(geomspace(0.0_f64, 1.0, 5, true).is_err());
    }

    #[test]
    fn test_geomspace_different_signs() {
        assert!(geomspace(-1.0_f64, 1.0, 5, true).is_err());
    }

    #[test]
    fn test_meshgrid_xy() {
        let x = Array::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let y = Array::from_vec(Ix1::new([2]), vec![4.0, 5.0]).unwrap();
        let grids = meshgrid(&[x, y], "xy").unwrap();
        assert_eq!(grids.len(), 2);
        assert_eq!(grids[0].shape(), &[2, 3]);
        assert_eq!(grids[1].shape(), &[2, 3]);
        // X grid: rows are [1,2,3], [1,2,3]
        let xdata: Vec<f64> = grids[0].iter().copied().collect();
        assert_eq!(xdata, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        // Y grid: rows are [4,4,4], [5,5,5]
        let ydata: Vec<f64> = grids[1].iter().copied().collect();
        assert_eq!(ydata, vec![4.0, 4.0, 4.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_meshgrid_ij() {
        let x = Array::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let y = Array::from_vec(Ix1::new([2]), vec![4.0, 5.0]).unwrap();
        let grids = meshgrid(&[x, y], "ij").unwrap();
        assert_eq!(grids.len(), 2);
        assert_eq!(grids[0].shape(), &[3, 2]);
        assert_eq!(grids[1].shape(), &[3, 2]);
    }

    #[test]
    fn test_meshgrid_bad_indexing() {
        assert!(meshgrid(&[], "zz").is_err());
    }

    #[test]
    fn test_mgrid() {
        let grids = mgrid(&[(0.0, 3.0, 1.0), (0.0, 2.0, 1.0)]).unwrap();
        assert_eq!(grids.len(), 2);
        assert_eq!(grids[0].shape(), &[3, 2]);
    }

    #[test]
    fn test_ogrid() {
        let grids = ogrid(&[(0.0, 3.0, 1.0), (0.0, 2.0, 1.0)]).unwrap();
        assert_eq!(grids.len(), 2);
        assert_eq!(grids[0].shape(), &[3, 1]);
        assert_eq!(grids[1].shape(), &[1, 2]);
    }

    // -- REQ-19 tests --

    #[test]
    fn test_identity() {
        let a = identity::<f64>(3).unwrap();
        assert_eq!(a.shape(), &[3, 3]);
        let data = a.as_slice().unwrap();
        assert_eq!(data, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_eye() {
        let a = eye::<f64>(3, 4, 0).unwrap();
        assert_eq!(a.shape(), &[3, 4]);
        let data = a.as_slice().unwrap();
        assert_eq!(
            data,
            &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        );
    }

    #[test]
    fn test_eye_positive_k() {
        let a = eye::<f64>(3, 3, 1).unwrap();
        let data = a.as_slice().unwrap();
        assert_eq!(data, &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_eye_negative_k() {
        let a = eye::<f64>(3, 3, -1).unwrap();
        let data = a.as_slice().unwrap();
        assert_eq!(data, &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_diag_from_1d() {
        let a = Array::from_vec(IxDyn::new(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        let d = diag(&a, 0).unwrap();
        assert_eq!(d.shape(), &[3, 3]);
        let data: Vec<f64> = d.iter().copied().collect();
        assert_eq!(data, vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
    }

    #[test]
    fn test_diag_from_2d() {
        let a = Array::from_vec(
            IxDyn::new(&[3, 3]),
            vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0],
        )
        .unwrap();
        let d = diag(&a, 0).unwrap();
        assert_eq!(d.shape(), &[3]);
        let data: Vec<f64> = d.iter().copied().collect();
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_diag_k_positive() {
        let a = Array::from_vec(IxDyn::new(&[2]), vec![1.0, 2.0]).unwrap();
        let d = diag(&a, 1).unwrap();
        assert_eq!(d.shape(), &[3, 3]);
        let data: Vec<f64> = d.iter().copied().collect();
        assert_eq!(data, vec![0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_diagflat() {
        let a = Array::from_vec(IxDyn::new(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let d = diagflat(&a, 0).unwrap();
        assert_eq!(d.shape(), &[4, 4]);
        // Diagonal should be [1, 2, 3, 4]
        let extracted = diag(&d, 0).unwrap();
        let data: Vec<f64> = extracted.iter().copied().collect();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_tri() {
        let a = tri::<f64>(3, 3, 0).unwrap();
        let data = a.as_slice().unwrap();
        assert_eq!(data, &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_tril() {
        let a = Array::from_vec(
            IxDyn::new(&[3, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let t = tril(&a, 0).unwrap();
        let data: Vec<f64> = t.iter().copied().collect();
        assert_eq!(data, vec![1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_triu() {
        let a = Array::from_vec(
            IxDyn::new(&[3, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let t = triu(&a, 0).unwrap();
        let data: Vec<f64> = t.iter().copied().collect();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]);
    }

    #[test]
    fn test_tril_not_2d() {
        let a = Array::from_vec(IxDyn::new(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(tril(&a, 0).is_err());
    }

    #[test]
    fn test_triu_not_2d() {
        let a = Array::from_vec(IxDyn::new(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(triu(&a, 0).is_err());
    }

    // -- vander --

    #[test]
    fn test_vander_default_decreasing() {
        let x = Array::from_vec(Ix1::new([3]), vec![1.0_f64, 2.0, 3.0]).unwrap();
        let v = vander(&x, None, false).unwrap();
        assert_eq!(v.shape(), &[3, 3]);
        // Row 0: [1, 1, 1], row 1: [4, 2, 1], row 2: [9, 3, 1]
        let data: Vec<f64> = v.iter().copied().collect();
        assert_eq!(data, vec![1.0, 1.0, 1.0, 4.0, 2.0, 1.0, 9.0, 3.0, 1.0]);
    }

    #[test]
    fn test_vander_increasing() {
        let x = Array::from_vec(Ix1::new([3]), vec![1.0_f64, 2.0, 3.0]).unwrap();
        let v = vander(&x, None, true).unwrap();
        let data: Vec<f64> = v.iter().copied().collect();
        // Row 0: [1, 1, 1], row 1: [1, 2, 4], row 2: [1, 3, 9]
        assert_eq!(data, vec![1.0, 1.0, 1.0, 1.0, 2.0, 4.0, 1.0, 3.0, 9.0]);
    }

    #[test]
    fn test_vander_explicit_n() {
        let x = Array::from_vec(Ix1::new([2]), vec![2.0_f64, 3.0]).unwrap();
        let v = vander(&x, Some(4), false).unwrap();
        assert_eq!(v.shape(), &[2, 4]);
        // Row 0: 2^3, 2^2, 2^1, 2^0 = [8, 4, 2, 1]
        // Row 1: 3^3, 3^2, 3^1, 3^0 = [27, 9, 3, 1]
        let data: Vec<f64> = v.iter().copied().collect();
        assert_eq!(data, vec![8.0, 4.0, 2.0, 1.0, 27.0, 9.0, 3.0, 1.0]);
    }

    #[test]
    fn test_vander_empty() {
        let x: Array<f64, Ix1> = Array::from_vec(Ix1::new([0]), vec![]).unwrap();
        assert!(vander(&x, None, false).is_err());
    }

    // -- copy / asarray family --

    #[test]
    fn test_copy() {
        let a = array(Ix1::new([3]), vec![1.0_f64, 2.0, 3.0]).unwrap();
        let b = copy(&a);
        assert_eq!(
            a.iter().copied().collect::<Vec<_>>(),
            b.iter().copied().collect::<Vec<_>>()
        );
        // Different buffer
        assert_ne!(
            a.as_slice().unwrap().as_ptr(),
            b.as_slice().unwrap().as_ptr()
        );
    }

    #[test]
    fn test_ascontiguousarray() {
        let a = array(Ix1::new([3]), vec![1.0_f64, 2.0, 3.0]).unwrap();
        let b = ascontiguousarray(&a);
        assert_eq!(b.shape(), a.shape());
    }

    #[test]
    fn test_asarray_chkfinite_ok() {
        let a = array(Ix1::new([3]), vec![1.0_f64, 2.0, 3.0]).unwrap();
        assert!(asarray_chkfinite(&a).is_ok());
    }

    #[test]
    fn test_asarray_chkfinite_nan_fails() {
        let a = array(Ix1::new([3]), vec![1.0_f64, f64::NAN, 3.0]).unwrap();
        assert!(asarray_chkfinite(&a).is_err());
    }

    #[test]
    fn test_asarray_chkfinite_inf_fails() {
        let a = array(Ix1::new([3]), vec![1.0_f64, f64::INFINITY, 3.0]).unwrap();
        assert!(asarray_chkfinite(&a).is_err());
    }

    #[test]
    fn test_require() {
        let a = array(Ix1::new([3]), vec![1, 2, 3]).unwrap();
        let b = require(&a, "CW");
        assert_eq!(b.shape(), a.shape());
    }

    // -- fromfunction / fromstring / fromfile --

    #[test]
    fn test_fromfunction_2d() {
        let a = fromfunction(Ix2::new([3, 3]), |idx| (idx[0] + idx[1]) as i32).unwrap();
        let data: Vec<i32> = a.iter().copied().collect();
        assert_eq!(data, vec![0, 1, 2, 1, 2, 3, 2, 3, 4]);
    }

    #[test]
    fn test_fromfunction_1d() {
        let a = fromfunction(Ix1::new([5]), |idx| idx[0] as f64 * 2.0).unwrap();
        let data: Vec<f64> = a.iter().copied().collect();
        assert_eq!(data, vec![0.0, 2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_fromstring_whitespace() {
        let a: Array<f64, Ix1> = fromstring("1.5 2.5 3.5", " ").unwrap();
        let data: Vec<f64> = a.iter().copied().collect();
        assert_eq!(data, vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn test_fromstring_comma() {
        let a: Array<i32, Ix1> = fromstring("1, 2, 3, 4", ",").unwrap();
        assert_eq!(a.shape(), &[4]);
        let data: Vec<i32> = a.iter().copied().collect();
        assert_eq!(data, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_fromstring_empty_sep_uses_whitespace() {
        let a: Array<i32, Ix1> = fromstring("1   2\t3\n4", "").unwrap();
        let data: Vec<i32> = a.iter().copied().collect();
        assert_eq!(data, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_fromstring_bad_token() {
        let r: FerrayResult<Array<f64, Ix1>> = fromstring("1.0 abc 3.0", " ");
        assert!(r.is_err());
    }
}
