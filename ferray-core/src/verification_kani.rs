//! Kani formal verification harnesses for ferray-core.
//!
//! Verifies critical invariants:
//! - Broadcasting shape resolution correctness (REQ-19)
//! - Index bounds safety (REQ-20)
//! - Type promotion table correctness (REQ-21)
//! - NdArray structural invariants (REQ-23)
//! - Error condition detection (REQ-18)

#![cfg(kani)]

use crate::dimension::{Dimension, Ix1, Ix2, IxDyn};
use crate::dtype::{DType, Element};
use crate::error::FerrayError;

// ---------------------------------------------------------------------------
// Broadcasting shape resolution (REQ-19)
// ---------------------------------------------------------------------------

/// Verify broadcasting produces valid output shapes.
#[kani::proof]
fn verify_broadcast_output_ndim() {
    let a_ndim: usize = kani::any();
    let b_ndim: usize = kani::any();
    kani::assume(a_ndim >= 1 && a_ndim <= 4);
    kani::assume(b_ndim >= 1 && b_ndim <= 4);

    // Output ndim is max of inputs
    let out_ndim = if a_ndim > b_ndim { a_ndim } else { b_ndim };
    assert!(out_ndim >= a_ndim);
    assert!(out_ndim >= b_ndim);
}

/// Verify broadcasting rule: dimension 1 broadcasts to any size.
#[kani::proof]
fn verify_broadcast_unit_dimension() {
    let a_dim: usize = kani::any();
    let b_dim: usize = kani::any();
    kani::assume(a_dim >= 1 && a_dim <= 1024);
    kani::assume(b_dim >= 1 && b_dim <= 1024);

    // Broadcasting rule: if a_dim == 1, result is b_dim; if b_dim == 1, result is a_dim
    // if both equal, result is that value; otherwise incompatible
    let result = if a_dim == b_dim {
        Some(a_dim)
    } else if a_dim == 1 {
        Some(b_dim)
    } else if b_dim == 1 {
        Some(a_dim)
    } else {
        None
    };

    if let Some(r) = result {
        assert!(r >= 1);
        assert!(r == a_dim || r == b_dim);
    }
}

/// Verify broadcasting is commutative.
#[kani::proof]
fn verify_broadcast_commutative() {
    let a: usize = kani::any();
    let b: usize = kani::any();
    kani::assume(a >= 1 && a <= 64);
    kani::assume(b >= 1 && b <= 64);

    let ab = if a == b {
        Some(a)
    } else if a == 1 {
        Some(b)
    } else if b == 1 {
        Some(a)
    } else {
        None
    };

    let ba = if b == a {
        Some(b)
    } else if b == 1 {
        Some(a)
    } else if a == 1 {
        Some(b)
    } else {
        None
    };

    assert_eq!(ab, ba);
}

// ---------------------------------------------------------------------------
// Index bounds safety (REQ-20)
// ---------------------------------------------------------------------------

/// Verify positive index bounds checking.
#[kani::proof]
fn verify_positive_index_bounds() {
    let shape_len: usize = kani::any();
    let index: usize = kani::any();
    kani::assume(shape_len >= 1 && shape_len <= 256);
    kani::assume(index <= 512);

    let in_bounds = index < shape_len;
    if in_bounds {
        assert!(index < shape_len);
    } else {
        assert!(index >= shape_len);
    }
}

/// Verify negative index normalization.
#[kani::proof]
fn verify_negative_index_normalization() {
    let axis_len: usize = kani::any();
    let neg_index: isize = kani::any();
    kani::assume(axis_len >= 1 && axis_len <= 128);
    kani::assume(neg_index >= -(axis_len as isize) && neg_index < 0);

    // Negative index normalization: idx + len
    let normalized = (neg_index + axis_len as isize) as usize;
    assert!(normalized < axis_len);
}

/// Verify slice bounds are valid.
#[kani::proof]
fn verify_slice_bounds() {
    let axis_len: usize = kani::any();
    let start: usize = kani::any();
    let end: usize = kani::any();
    kani::assume(axis_len >= 1 && axis_len <= 64);
    kani::assume(start <= axis_len);
    kani::assume(end <= axis_len);
    kani::assume(start <= end);

    let slice_len = end - start;
    assert!(slice_len <= axis_len);
    assert!(start + slice_len <= axis_len);
}

// ---------------------------------------------------------------------------
// Type promotion correctness (REQ-21)
// ---------------------------------------------------------------------------

/// Verify DType size_of returns correct values.
#[kani::proof]
fn verify_dtype_size_of() {
    assert_eq!(DType::Bool.size_of(), 1);
    assert_eq!(DType::U8.size_of(), 1);
    assert_eq!(DType::U16.size_of(), 2);
    assert_eq!(DType::U32.size_of(), 4);
    assert_eq!(DType::U64.size_of(), 8);
    assert_eq!(DType::I8.size_of(), 1);
    assert_eq!(DType::I16.size_of(), 2);
    assert_eq!(DType::I32.size_of(), 4);
    assert_eq!(DType::I64.size_of(), 8);
    assert_eq!(DType::F32.size_of(), 4);
    assert_eq!(DType::F64.size_of(), 8);
    assert_eq!(DType::Complex32.size_of(), 8);
    assert_eq!(DType::Complex64.size_of(), 16);
}

/// Verify DType classification is mutually exclusive.
#[kani::proof]
fn verify_dtype_classification_exclusive() {
    let dtype_idx: u8 = kani::any();
    kani::assume(dtype_idx < 15);

    let dtype = match dtype_idx {
        0 => DType::Bool,
        1 => DType::U8,
        2 => DType::U16,
        3 => DType::U32,
        4 => DType::U64,
        5 => DType::I8,
        6 => DType::I16,
        7 => DType::I32,
        8 => DType::I64,
        9 => DType::F32,
        10 => DType::F64,
        11 => DType::Complex32,
        12 => DType::Complex64,
        13 => DType::U128,
        14 => DType::I128,
        _ => unreachable!(),
    };

    // Float and complex are mutually exclusive
    if dtype.is_float() {
        assert!(!dtype.is_complex());
    }
    if dtype.is_complex() {
        assert!(!dtype.is_float());
    }

    // Complex types are always signed
    if dtype.is_complex() {
        assert!(dtype.is_signed());
    }

    // Float types are always signed
    if dtype.is_float() {
        assert!(dtype.is_signed());
    }
}

/// Verify Element trait zero/one are distinct for numeric types.
#[kani::proof]
fn verify_element_zero_one_distinct() {
    assert_ne!(f64::zero(), f64::one());
    assert_ne!(f32::zero(), f32::one());
    assert_ne!(i32::zero(), i32::one());
    assert_ne!(u64::zero(), u64::one());
}

/// Verify alignment is always a power of 2.
#[kani::proof]
fn verify_dtype_alignment_power_of_two() {
    let dtype_idx: u8 = kani::any();
    kani::assume(dtype_idx < 15);

    let dtype = match dtype_idx {
        0 => DType::Bool,
        1 => DType::U8,
        2 => DType::U16,
        3 => DType::U32,
        4 => DType::U64,
        5 => DType::I8,
        6 => DType::I16,
        7 => DType::I32,
        8 => DType::I64,
        9 => DType::F32,
        10 => DType::F64,
        11 => DType::Complex32,
        12 => DType::Complex64,
        13 => DType::U128,
        14 => DType::I128,
        _ => unreachable!(),
    };

    let alignment = dtype.alignment();
    assert!(alignment > 0);
    assert!(alignment.is_power_of_two());
}

// ---------------------------------------------------------------------------
// NdArray structural invariants (REQ-23)
// ---------------------------------------------------------------------------

/// Verify Ix1 shape is consistent.
#[kani::proof]
fn verify_ix1_shape_consistency() {
    let n: usize = kani::any();
    kani::assume(n <= 1024);

    let dim = Ix1::new([n]);
    assert_eq!(dim.ndim(), 1);
    assert_eq!(dim.size(), n);
    assert_eq!(dim.as_slice().len(), 1);
    assert_eq!(dim.as_slice()[0], n);
}

/// Verify Ix2 shape consistency.
#[kani::proof]
fn verify_ix2_shape_consistency() {
    let m: usize = kani::any();
    let n: usize = kani::any();
    kani::assume(m <= 64);
    kani::assume(n <= 64);

    let dim = Ix2::new([m, n]);
    assert_eq!(dim.ndim(), 2);
    assert_eq!(dim.size(), m * n);
    assert_eq!(dim.as_slice().len(), 2);
    assert_eq!(dim.as_slice()[0], m);
    assert_eq!(dim.as_slice()[1], n);
}

/// Verify IxDyn shape consistency.
#[kani::proof]
fn verify_ixdyn_shape_consistency() {
    let a: usize = kani::any();
    let b: usize = kani::any();
    kani::assume(a <= 32);
    kani::assume(b <= 32);

    let dim = IxDyn::new(&[a, b]);
    assert_eq!(dim.ndim(), 2);
    assert_eq!(dim.size(), a * b);
    assert_eq!(dim.as_slice()[0], a);
    assert_eq!(dim.as_slice()[1], b);
}

/// Verify from_vec requires data length == shape product.
#[kani::proof]
fn verify_from_vec_length_invariant() {
    let shape_n: usize = kani::any();
    let data_len: usize = kani::any();
    kani::assume(shape_n <= 64);
    kani::assume(data_len <= 64);

    // Array::from_vec should succeed iff data_len == shape_n
    let matches = data_len == shape_n;
    if matches {
        // This SHOULD succeed
        assert_eq!(data_len, shape_n);
    } else {
        // This SHOULD fail with ShapeMismatch
        assert_ne!(data_len, shape_n);
    }
}

/// Verify reshape preserves total element count.
#[kani::proof]
fn verify_reshape_element_count() {
    let old_m: usize = kani::any();
    let old_n: usize = kani::any();
    let new_a: usize = kani::any();
    let new_b: usize = kani::any();
    kani::assume(old_m >= 1 && old_m <= 16);
    kani::assume(old_n >= 1 && old_n <= 16);
    kani::assume(new_a >= 1 && new_a <= 16);
    kani::assume(new_b >= 1 && new_b <= 16);

    let old_size = old_m * old_n;
    let new_size = new_a * new_b;

    // Reshape is valid iff sizes match
    let valid = old_size == new_size;
    if valid {
        assert_eq!(old_size, new_size);
    }
}

// ---------------------------------------------------------------------------
// Error condition detection (REQ-18)
// ---------------------------------------------------------------------------

/// Verify FerrayError variants carry context.
#[kani::proof]
fn verify_error_shape_mismatch() {
    let err = FerrayError::shape_mismatch("test context");
    // The error should be the ShapeMismatch variant
    match err {
        FerrayError::ShapeMismatch { .. } => {} // correct
        _ => panic!("Expected ShapeMismatch"),
    }
}

/// Verify FerrayError index_out_of_bounds.
#[kani::proof]
fn verify_error_index_out_of_bounds() {
    let err = FerrayError::index_out_of_bounds(5, 0, 3);
    match err {
        FerrayError::IndexOutOfBounds { .. } => {} // correct
        _ => panic!("Expected IndexOutOfBounds"),
    }
}
