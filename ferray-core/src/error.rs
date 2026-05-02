// ferray-core: Error types (REQ-27)

use core::fmt;

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::{string::String, string::ToString, vec::Vec};

/// The primary error type for all ferray operations.
///
/// This enum is `#[non_exhaustive]`, so new variants may be added
/// in minor releases without breaking downstream code.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum FerrayError {
    /// Operand shapes are incompatible for the requested operation.
    #[error("shape mismatch: {message}")]
    ShapeMismatch {
        /// Human-readable description of the mismatch.
        message: String,
    },

    /// Broadcasting failed because shapes cannot be reconciled.
    #[error("broadcast failure: cannot broadcast shapes {shape_a:?} and {shape_b:?}")]
    BroadcastFailure {
        /// First shape.
        shape_a: Vec<usize>,
        /// Second shape.
        shape_b: Vec<usize>,
    },

    /// An axis index exceeded the array's dimensionality.
    #[error("axis {axis} is out of bounds for array with {ndim} dimensions")]
    AxisOutOfBounds {
        /// The invalid axis.
        axis: usize,
        /// Number of dimensions.
        ndim: usize,
    },

    /// An element index exceeded the array's extent along some axis.
    #[error("index {index} is out of bounds for axis {axis} with size {size}")]
    IndexOutOfBounds {
        /// The invalid index.
        index: isize,
        /// The axis along which the index was applied.
        axis: usize,
        /// The size of that axis.
        size: usize,
    },

    /// A matrix was singular when an invertible one was required.
    #[error("singular matrix: {message}")]
    SingularMatrix {
        /// Diagnostic context.
        message: String,
    },

    /// An iterative algorithm did not converge within its budget.
    #[error("convergence failure after {iterations} iterations: {message}")]
    ConvergenceFailure {
        /// Number of iterations attempted.
        iterations: usize,
        /// Diagnostic context.
        message: String,
    },

    /// The requested dtype is invalid or unsupported for this operation.
    #[error("invalid dtype: {message}")]
    InvalidDtype {
        /// Diagnostic context.
        message: String,
    },

    /// A computation produced NaN / Inf when finite results were required.
    #[error("numerical instability: {message}")]
    NumericalInstability {
        /// Diagnostic context.
        message: String,
    },

    /// An I/O operation failed.
    #[error("I/O error: {message}")]
    IoError {
        /// Diagnostic context.
        message: String,
    },

    /// A function argument was invalid.
    #[error("invalid value: {message}")]
    InvalidValue {
        /// Diagnostic context.
        message: String,
    },
}

/// Convenience alias used throughout ferray.
pub type FerrayResult<T> = Result<T, FerrayError>;

impl FerrayError {
    /// Returns `true` when this error originates from a linear-algebra
    /// operation (singular matrix, non-convergence, numerical
    /// instability) — the rough Rust equivalent of catching numpy's
    /// `LinAlgError` exception (#423).
    ///
    /// Use this to distinguish linalg-specific failures from generic
    /// shape / dtype / I/O errors when writing recovery code:
    ///
    /// ```ignore
    /// match ferray_linalg::solve(&a, &b) {
    ///     Ok(x) => x,
    ///     Err(e) if e.is_linalg_error() => fall_back_to_pinv(&a, &b)?,
    ///     Err(e) => return Err(e),
    /// }
    /// ```
    ///
    /// The predicate covers `SingularMatrix`, `ConvergenceFailure`,
    /// and `NumericalInstability`. Shape / dtype / index / I/O errors
    /// stay outside the linalg category.
    #[must_use]
    pub const fn is_linalg_error(&self) -> bool {
        matches!(
            self,
            Self::SingularMatrix { .. }
                | Self::ConvergenceFailure { .. }
                | Self::NumericalInstability { .. }
        )
    }
}

impl FerrayError {
    /// Create a `ShapeMismatch` error with a formatted message.
    pub fn shape_mismatch(msg: impl fmt::Display) -> Self {
        Self::ShapeMismatch {
            message: msg.to_string(),
        }
    }

    /// Create a `BroadcastFailure` error.
    #[must_use]
    pub fn broadcast_failure(a: &[usize], b: &[usize]) -> Self {
        Self::BroadcastFailure {
            shape_a: a.to_vec(),
            shape_b: b.to_vec(),
        }
    }

    /// Create an `AxisOutOfBounds` error.
    #[must_use]
    pub const fn axis_out_of_bounds(axis: usize, ndim: usize) -> Self {
        Self::AxisOutOfBounds { axis, ndim }
    }

    /// Create an `IndexOutOfBounds` error.
    #[must_use]
    pub const fn index_out_of_bounds(index: isize, axis: usize, size: usize) -> Self {
        Self::IndexOutOfBounds { index, axis, size }
    }

    /// Create an `InvalidDtype` error with a formatted message.
    pub fn invalid_dtype(msg: impl fmt::Display) -> Self {
        Self::InvalidDtype {
            message: msg.to_string(),
        }
    }

    /// Create an `InvalidValue` error with a formatted message.
    pub fn invalid_value(msg: impl fmt::Display) -> Self {
        Self::InvalidValue {
            message: msg.to_string(),
        }
    }

    /// Create an `IoError` from a formatted message.
    pub fn io_error(msg: impl fmt::Display) -> Self {
        Self::IoError {
            message: msg.to_string(),
        }
    }
}

#[cfg(feature = "std")]
impl From<std::io::Error> for FerrayError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError {
            message: e.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_shape_mismatch() {
        let e = FerrayError::shape_mismatch("expected (3,4), got (3,5)");
        assert!(e.to_string().contains("expected (3,4), got (3,5)"));
    }

    #[test]
    fn error_display_axis_out_of_bounds() {
        let e = FerrayError::axis_out_of_bounds(5, 3);
        assert!(e.to_string().contains("axis 5"));
        assert!(e.to_string().contains("3 dimensions"));
    }

    #[test]
    fn error_display_broadcast_failure() {
        let e = FerrayError::broadcast_failure(&[4, 3], &[2, 5]);
        let s = e.to_string();
        assert!(s.contains("[4, 3]"));
        assert!(s.contains("[2, 5]"));
    }

    #[test]
    fn error_is_non_exhaustive() {
        // Verify the enum is non_exhaustive by using a wildcard
        // in a match from an "external" perspective. Inside this crate
        // the compiler knows all variants, so we just verify construction.
        let e = FerrayError::invalid_value("test");
        assert!(matches!(e, FerrayError::InvalidValue { .. }));

        let e2 = FerrayError::shape_mismatch("bad shape");
        assert!(matches!(e2, FerrayError::ShapeMismatch { .. }));
    }

    #[test]
    fn from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let ferray_err: FerrayError = io_err.into();
        assert!(ferray_err.to_string().contains("file missing"));
    }

    // ----- is_linalg_error predicate (#423) -----------------------------

    #[test]
    fn is_linalg_error_returns_true_for_linalg_variants() {
        assert!(
            FerrayError::SingularMatrix {
                message: "test".into()
            }
            .is_linalg_error()
        );
        assert!(
            FerrayError::ConvergenceFailure {
                iterations: 100,
                message: "test".into()
            }
            .is_linalg_error()
        );
        assert!(
            FerrayError::NumericalInstability {
                message: "test".into()
            }
            .is_linalg_error()
        );
    }

    #[test]
    fn is_linalg_error_returns_false_for_other_variants() {
        assert!(!FerrayError::shape_mismatch("test").is_linalg_error());
        assert!(!FerrayError::invalid_dtype("test").is_linalg_error());
        assert!(!FerrayError::invalid_value("test").is_linalg_error());
        assert!(!FerrayError::io_error("test").is_linalg_error());
        assert!(!FerrayError::axis_out_of_bounds(2, 1).is_linalg_error());
    }
}
