// ferray-core: Display/Debug formatting with NumPy-style output (REQ-39)

use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::dimension::Dimension;
use crate::dtype::Element;

use super::arc::ArcArray;
use super::cow::CowArray;
use super::owned::Array;
use super::view::ArrayView;

// ---------------------------------------------------------------------------
// Global print options
// ---------------------------------------------------------------------------

static PRINT_PRECISION: AtomicUsize = AtomicUsize::new(8);
static PRINT_THRESHOLD: AtomicUsize = AtomicUsize::new(1000);
static PRINT_LINEWIDTH: AtomicUsize = AtomicUsize::new(75);
static PRINT_EDGEITEMS: AtomicUsize = AtomicUsize::new(3);

/// Configure how arrays are printed.
///
/// Matches `NumPy`'s `set_printoptions`:
/// - `precision`: number of decimal places for floats (default 8)
/// - `threshold`: total element count above which truncation kicks in (default 1000)
/// - `linewidth`: max characters per line (default 75)
/// - `edgeitems`: number of items shown at each edge when truncated (default 3)
pub fn set_print_options(precision: usize, threshold: usize, linewidth: usize, edgeitems: usize) {
    PRINT_PRECISION.store(precision, Ordering::SeqCst);
    PRINT_THRESHOLD.store(threshold, Ordering::SeqCst);
    PRINT_LINEWIDTH.store(linewidth, Ordering::SeqCst);
    PRINT_EDGEITEMS.store(edgeitems, Ordering::SeqCst);
}

/// Get current print options as `(precision, threshold, linewidth, edgeitems)`.
pub fn get_print_options() -> (usize, usize, usize, usize) {
    (
        PRINT_PRECISION.load(Ordering::SeqCst),
        PRINT_THRESHOLD.load(Ordering::SeqCst),
        PRINT_LINEWIDTH.load(Ordering::SeqCst),
        PRINT_EDGEITEMS.load(Ordering::SeqCst),
    )
}

// ---------------------------------------------------------------------------
// Core formatting logic
// ---------------------------------------------------------------------------

/// Format an array's data for display, handling truncation for large arrays.
fn format_array_data<T: Element, D: Dimension>(
    inner: &ndarray::ArrayBase<impl ndarray::Data<Elem = T>, D::NdarrayDim>,
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    let shape = inner.shape();
    let ndim = shape.len();
    let size: usize = shape.iter().product();
    let (precision, threshold, _linewidth, edgeitems) = get_print_options();

    if ndim == 0 {
        // Scalar
        // 0-d arrays always have exactly one element
        let val = inner.iter().next().unwrap_or_else(|| unreachable!());
        write!(f, "{val}")?;
        return Ok(());
    }

    let truncate = size > threshold;

    write!(f, "array(")?;
    format_recursive(inner, shape, &[], truncate, edgeitems, precision, f)?;
    write!(f, ")")?;
    Ok(())
}

/// Recursively format nested brackets.
fn format_recursive<T: fmt::Display>(
    data: &ndarray::ArrayBase<impl ndarray::Data<Elem = T>, impl ndarray::Dimension>,
    shape: &[usize],
    indices: &[usize],
    truncate: bool,
    edgeitems: usize,
    precision: usize,
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    let depth = indices.len();
    let ndim = shape.len();
    let n = shape[depth];
    let show_all = !truncate || n <= 2 * edgeitems;

    write!(f, "[")?;
    if depth == ndim - 1 {
        // Innermost dimension: print elements
        if show_all {
            for i in 0..n {
                if i > 0 {
                    write!(f, ", ")?;
                }
                let mut idx = indices.to_vec();
                idx.push(i);
                write_element_at(data, &idx, precision, f)?;
            }
        } else {
            for i in 0..edgeitems {
                if i > 0 {
                    write!(f, ", ")?;
                }
                let mut idx = indices.to_vec();
                idx.push(i);
                write_element_at(data, &idx, precision, f)?;
            }
            write!(f, ", ..., ")?;
            for i in (n - edgeitems)..n {
                if i > n - edgeitems {
                    write!(f, ", ")?;
                }
                let mut idx = indices.to_vec();
                idx.push(i);
                write_element_at(data, &idx, precision, f)?;
            }
        }
    } else {
        // Outer dimension: recurse
        let indent = " ".repeat(depth + 7); // "array(" = 6 chars + 1 for [

        if show_all {
            for i in 0..n {
                if i > 0 {
                    write!(f, ",\n{indent}")?;
                }
                let mut idx = indices.to_vec();
                idx.push(i);
                format_recursive(data, shape, &idx, truncate, edgeitems, precision, f)?;
            }
        } else {
            for i in 0..edgeitems {
                if i > 0 {
                    write!(f, ",\n{indent}")?;
                }
                let mut idx = indices.to_vec();
                idx.push(i);
                format_recursive(data, shape, &idx, truncate, edgeitems, precision, f)?;
            }
            write!(f, ",\n{indent}...")?;
            for i in (n - edgeitems)..n {
                write!(f, ",\n{indent}")?;
                let mut idx = indices.to_vec();
                idx.push(i);
                format_recursive(data, shape, &idx, truncate, edgeitems, precision, f)?;
            }
        }
    }
    write!(f, "]")?;
    Ok(())
}

/// Write a single element given multi-dimensional indices.
fn write_element_at<T: fmt::Display>(
    data: &ndarray::ArrayBase<impl ndarray::Data<Elem = T>, impl ndarray::Dimension>,
    indices: &[usize],
    _precision: usize,
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    // Convert indices to ndarray's indexing — use dynamic indexing
    let nd_idx = ndarray::IxDyn(indices);
    let dyn_view = data.view().into_dyn();
    let val = &dyn_view[nd_idx];
    write!(f, "{val}")
}

// ---------------------------------------------------------------------------
// Display / Debug for Array<T, D>
// ---------------------------------------------------------------------------

impl<T: Element, D: Dimension> fmt::Display for Array<T, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_array_data::<T, D>(&self.inner, f)
    }
}

impl<T: Element, D: Dimension> fmt::Debug for Array<T, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Array(dtype={}, shape={:?}, ", T::dtype(), self.shape())?;
        format_array_data::<T, D>(&self.inner, f)?;
        write!(f, ")")
    }
}

// ---------------------------------------------------------------------------
// Display / Debug for ArrayView
// ---------------------------------------------------------------------------

impl<T: Element, D: Dimension> fmt::Display for ArrayView<'_, T, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_array_data::<T, D>(&self.inner, f)
    }
}

impl<T: Element, D: Dimension> fmt::Debug for ArrayView<'_, T, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ArrayView(dtype={}, shape={:?}, ",
            T::dtype(),
            self.shape()
        )?;
        format_array_data::<T, D>(&self.inner, f)?;
        write!(f, ")")
    }
}

// ---------------------------------------------------------------------------
// Display / Debug for ArcArray
// ---------------------------------------------------------------------------

impl<T: Element, D: Dimension> fmt::Display for ArcArray<T, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Build a temporary ndarray view for formatting
        let nd_dim = self.dim().to_ndarray_dim();
        let slice = self.as_slice();
        let view =
            ndarray::ArrayView::from_shape(nd_dim, slice).expect("ArcArray shape consistent");
        format_array_data::<T, D>(&view, f)
    }
}

impl<T: Element, D: Dimension> fmt::Debug for ArcArray<T, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ArcArray(dtype={}, shape={:?}, refs={}, ",
            T::dtype(),
            self.shape(),
            self.ref_count()
        )?;
        fmt::Display::fmt(self, f)?;
        write!(f, ")")
    }
}

// ---------------------------------------------------------------------------
// Display / Debug for CowArray
// ---------------------------------------------------------------------------

impl<T: Element, D: Dimension> fmt::Display for CowArray<'_, T, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CowArray::Borrowed(v) => fmt::Display::fmt(v, f),
            CowArray::Owned(a) => fmt::Display::fmt(a, f),
        }
    }
}

impl<T: Element, D: Dimension> fmt::Debug for CowArray<'_, T, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CowArray::Borrowed(v) => {
                write!(f, "CowArray::Borrowed(")?;
                fmt::Debug::fmt(v, f)?;
                write!(f, ")")
            }
            CowArray::Owned(a) => {
                write!(f, "CowArray::Owned(")?;
                fmt::Debug::fmt(a, f)?;
                write!(f, ")")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::{Ix1, Ix2};

    #[test]
    fn display_1d() {
        let arr = Array::<i32, Ix1>::from_vec(Ix1::new([4]), vec![1, 2, 3, 4]).unwrap();
        let s = format!("{arr}");
        assert!(s.contains("[1, 2, 3, 4]"));
        assert!(s.starts_with("array("));
    }

    #[test]
    fn display_2d() {
        let arr = Array::<i32, Ix2>::from_vec(Ix2::new([2, 3]), vec![1, 2, 3, 4, 5, 6]).unwrap();
        let s = format!("{arr}");
        assert!(s.contains("[1, 2, 3]"));
        assert!(s.contains("[4, 5, 6]"));
    }

    #[test]
    fn debug_format() {
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![1.0, 2.0]).unwrap();
        let s = format!("{arr:?}");
        assert!(s.contains("dtype=float64"));
        assert!(s.contains("shape=[2]"));
    }

    #[test]
    fn truncated_display() {
        // Set low threshold to force truncation
        set_print_options(8, 5, 75, 2);

        let arr = Array::<i32, Ix1>::from_vec(Ix1::new([10]), (0..10).collect()).unwrap();
        let s = format!("{arr}");
        assert!(s.contains("..."));

        // Reset to defaults
        set_print_options(8, 1000, 75, 3);
    }

    #[test]
    fn arc_display() {
        let arr = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![10, 20, 30]).unwrap();
        let arc = ArcArray::from_owned(arr);
        let s = format!("{arc}");
        assert!(s.contains("[10, 20, 30]"));
    }

    #[test]
    fn cow_display() {
        let arr = Array::<i32, Ix1>::from_vec(Ix1::new([2]), vec![7, 8]).unwrap();
        let cow = CowArray::from_owned(arr);
        let s = format!("{cow}");
        assert!(s.contains("[7, 8]"));
    }
}
