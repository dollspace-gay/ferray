//! ACToR-CRITIC divergence pins for `ferray_core::dtype::promotion::result_type`
//! against the NumPy 2.4.5 live oracle (NEP-50 promotion, no value-based casting).
//!
//! API under audit:
//! `pub fn result_type(a: DType, b: DType) -> FerrayResult<DType>`
//! in `ferray-core/src/dtype/promotion.rs`.
//!
//! Every expected value in this file is the result of a LIVE numpy call
//! (`numpy.promote_types`) — never copied from ferray's own output (R-CHAR-3).
//! The governing contract is numpy's integer `common_dtype` slot: when no
//! signed integer type is wide enough to losslessly hold both a uint64 and a
//! signed integer, numpy falls back to `float64` (there is no int128 dtype in
//! numpy). See numpy/_core/src/multiarray/convert_datatype.c
//! `PyArray_PromoteTypes` (line 1045) → `PyArray_CommonDType`, and the kind
//! ordering in `dtype_kind_to_ordering` (line 652): unsigned(1) < signed(2) <
//! float(4). The live-call evidence is cited per-test below.

use ferray_core::DType;
use ferray_core::dtype::promotion::result_type;

// ---------------------------------------------------------------------------
// DIVERGENCE CLASS: uint64 + signed integer -> float64 (NEP-50 special case)
//
// numpy has no int128. The signed type that would hold the union of uint64
// (0..2^64-1) and a signed int (down to -2^63) needs > 64 signed bits, which
// numpy does not provide, so it promotes to float64. ferray instead promotes
// to its custom I128 (and the U128 cases to I256), which is NOT what numpy
// does for the native uint64/int64 dtypes that the Python API exposes.
// ---------------------------------------------------------------------------

/// numpy live: `np.promote_types('uint64','int64')` -> `float64`.
/// ferray currently returns I128 (lossless, but NOT numpy's contract).
#[test]
fn uint64_int64_promotes_to_f64_like_numpy() {
    assert_eq!(
        result_type(DType::U64, DType::I64).unwrap(),
        DType::F64,
        "np.promote_types('uint64','int64') == float64"
    );
}

/// numpy live: `np.promote_types('uint64','int8')` -> `float64`.
#[test]
fn uint64_int8_promotes_to_f64_like_numpy() {
    assert_eq!(
        result_type(DType::U64, DType::I8).unwrap(),
        DType::F64,
        "np.promote_types('uint64','int8') == float64"
    );
}

/// numpy live: `np.promote_types('uint64','int16')` -> `float64`.
#[test]
fn uint64_int16_promotes_to_f64_like_numpy() {
    assert_eq!(
        result_type(DType::U64, DType::I16).unwrap(),
        DType::F64,
        "np.promote_types('uint64','int16') == float64"
    );
}

/// numpy live: `np.promote_types('uint64','int32')` -> `float64`.
#[test]
fn uint64_int32_promotes_to_f64_like_numpy() {
    assert_eq!(
        result_type(DType::U64, DType::I32).unwrap(),
        DType::F64,
        "np.promote_types('uint64','int32') == float64"
    );
}

/// Commutativity: numpy live `np.promote_types('int64','uint64')` -> `float64`.
/// Pins that the divergence is symmetric (ferray returns I128 either way).
#[test]
fn int64_uint64_promotes_to_f64_like_numpy_commutative() {
    assert_eq!(
        result_type(DType::I64, DType::U64).unwrap(),
        DType::F64,
        "np.promote_types('int64','uint64') == float64 (commutative)"
    );
}

// ---------------------------------------------------------------------------
// SANITY (expected to PASS — confirms these classes already match numpy, so
// they are correctly NOT pinned as divergences). Kept temporarily for the
// audit; if any of these FAIL, that is a NEW divergence class to pin.
// ---------------------------------------------------------------------------
#[test]
fn sanity_matches_numpy() {
    // int16 + float32 -> float32 (numpy live)
    assert_eq!(result_type(DType::I16, DType::F32).unwrap(), DType::F32);
    // int32 + float32 -> float64 (numpy live)
    assert_eq!(result_type(DType::I32, DType::F32).unwrap(), DType::F64);
    // uint8 + float32 -> float32 (numpy live)
    assert_eq!(result_type(DType::U8, DType::F32).unwrap(), DType::F32);
    // uint32 + int32 -> int64 (numpy live)
    assert_eq!(result_type(DType::U32, DType::I32).unwrap(), DType::I64);
    // uint16 + int8 -> int32 (numpy live)
    assert_eq!(result_type(DType::U16, DType::I8).unwrap(), DType::I32);
    // int32 + complex64 -> complex128 (numpy live)
    assert_eq!(
        result_type(DType::I32, DType::Complex32).unwrap(),
        DType::Complex64
    );
    // int8 + complex64 -> complex64 (numpy live)
    assert_eq!(
        result_type(DType::I8, DType::Complex32).unwrap(),
        DType::Complex32
    );
    // float64 + complex64 -> complex128 (numpy live)
    assert_eq!(
        result_type(DType::F64, DType::Complex32).unwrap(),
        DType::Complex64
    );
    // float32 + complex64 -> complex64 (numpy live)
    assert_eq!(
        result_type(DType::F32, DType::Complex32).unwrap(),
        DType::Complex32
    );
    // bool + int8 -> int8 ; bool+bool -> bool (numpy live)
    assert_eq!(result_type(DType::Bool, DType::I8).unwrap(), DType::I8);
    assert_eq!(result_type(DType::Bool, DType::Bool).unwrap(), DType::Bool);
    // uint32 + complex64 -> complex128 (numpy live)
    assert_eq!(
        result_type(DType::U32, DType::Complex32).unwrap(),
        DType::Complex64
    );
}
