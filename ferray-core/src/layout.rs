// ferray-core: Memory layout enum (REQ-1)

/// Describes the memory layout of an N-dimensional array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryLayout {
    /// Row-major (C-style) contiguous layout.
    C,
    /// Column-major (Fortran-style) contiguous layout.
    Fortran,
    /// Non-contiguous or custom stride layout.
    Custom,
}

impl MemoryLayout {
    /// Returns `true` if the layout is C-contiguous (row-major).
    #[inline]
    #[must_use]
    pub fn is_c_contiguous(self) -> bool {
        self == Self::C
    }

    /// Returns `true` if the layout is Fortran-contiguous (column-major).
    #[inline]
    #[must_use]
    pub fn is_f_contiguous(self) -> bool {
        self == Self::Fortran
    }

    /// Returns `true` if the layout is neither C nor Fortran contiguous.
    #[inline]
    #[must_use]
    pub fn is_custom(self) -> bool {
        self == Self::Custom
    }
}

impl core::fmt::Display for MemoryLayout {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::C => write!(f, "C"),
            Self::Fortran => write!(f, "F"),
            Self::Custom => write!(f, "Custom"),
        }
    }
}

/// Classify the memory layout for an ndarray-backed array given
/// whether it is already known to be C-standard and its current shape +
/// element strides. Used by `Array::layout`, `ArrayView::layout` and
/// `ArrayViewMut::layout` so all three share one implementation
/// (see issue #127).
#[cfg(feature = "std")]
#[inline]
pub(crate) fn classify_layout(
    is_standard: bool,
    shape: &[usize],
    strides: &[isize],
) -> MemoryLayout {
    if is_standard {
        MemoryLayout::C
    } else {
        detect_layout(shape, strides)
    }
}

/// Determine memory layout from shape and strides.
#[cfg(feature = "std")]
pub(crate) fn detect_layout(shape: &[usize], strides: &[isize]) -> MemoryLayout {
    if shape.is_empty() {
        return MemoryLayout::C; // scalar-like
    }

    let is_c = is_c_contiguous(shape, strides);
    let is_f = is_f_contiguous(shape, strides);

    if is_c {
        MemoryLayout::C
    } else if is_f {
        MemoryLayout::Fortran
    } else {
        MemoryLayout::Custom
    }
}

#[cfg(feature = "std")]
fn is_c_contiguous(shape: &[usize], strides: &[isize]) -> bool {
    if shape.len() != strides.len() {
        return false;
    }
    let ndim = shape.len();
    if ndim == 0 {
        return true;
    }
    let mut expected: isize = 1;
    for i in (0..ndim).rev() {
        if shape[i] == 0 {
            return true; // empty array is contiguous by convention
        }
        if shape[i] != 1 && strides[i] != expected {
            return false;
        }
        expected = strides[i] * shape[i] as isize;
    }
    true
}

#[cfg(feature = "std")]
fn is_f_contiguous(shape: &[usize], strides: &[isize]) -> bool {
    if shape.len() != strides.len() {
        return false;
    }
    let ndim = shape.len();
    if ndim == 0 {
        return true;
    }
    let mut expected: isize = 1;
    for i in 0..ndim {
        if shape[i] == 0 {
            return true; // empty array is contiguous by convention
        }
        if shape[i] != 1 && strides[i] != expected {
            return false;
        }
        expected = strides[i] * shape[i] as isize;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_c_contiguous() {
        // 3x4 C-contiguous: strides = [4, 1]
        assert_eq!(detect_layout(&[3, 4], &[4, 1]), MemoryLayout::C);
    }

    #[test]
    fn detect_f_contiguous() {
        // 3x4 F-contiguous: strides = [1, 3]
        assert_eq!(detect_layout(&[3, 4], &[1, 3]), MemoryLayout::Fortran);
    }

    #[test]
    fn detect_custom() {
        // non-contiguous strides
        assert_eq!(detect_layout(&[3, 4], &[8, 2]), MemoryLayout::Custom);
    }

    #[test]
    fn detect_empty() {
        assert_eq!(detect_layout(&[], &[]), MemoryLayout::C);
    }

    #[test]
    fn display() {
        assert_eq!(MemoryLayout::C.to_string(), "C");
        assert_eq!(MemoryLayout::Fortran.to_string(), "F");
        assert_eq!(MemoryLayout::Custom.to_string(), "Custom");
    }
}
