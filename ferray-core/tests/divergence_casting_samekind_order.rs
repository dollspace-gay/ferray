//! ACToR-CRITIC divergence pins for `ferray_core::dtype::casting::can_cast`
//! under `CastKind::SameKind`, against the NumPy 2.4.4 live oracle
//! (`numpy.can_cast(from, to, casting='same_kind')`).
//!
//! API under audit:
//! `pub fn can_cast(from: DType, to: DType, casting: CastKind) -> FerrayResult<bool>`
//! in `ferray-core/src/dtype/casting.rs`, with `CastKind::SameKind`.
//!
//! Every expected value in this file is the result of a LIVE numpy call
//! (`numpy.can_cast(..., casting='same_kind')`, numpy 2.4.4) — never copied
//! from ferray's own output (R-CHAR-3). The live values were generated with
//! `ferray-python/.venv/bin/python -c "import numpy as np; ..."`.
//!
//! GOVERNING CONTRACT — numpy's `same_kind` is a KIND-ORDERING relation, not a
//! kind-equality relation. From
//! `numpy/_core/src/multiarray/convert_datatype.c`:
//!   - `dtype_kind_to_ordering` (lines 653-690) assigns DISTINCT rungs:
//!     bool 'b' -> 0, unsigned 'u' -> 1, signed 'i' -> 2,
//!     float 'f' -> 4, complex 'c' -> 5.
//!   - The numeric default cast-safety logic (lines 2344-2351) returns
//!     `NPY_SAME_KIND_CASTING` when the cast is not safe but
//!     `dtype_kind_to_ordering(from) <= dtype_kind_to_ordering(to)`.
//!   - `PyArray_CanCastTypeTo` (line 698) returns true for `same_kind` iff the
//!     min cast safety is <= same_kind, i.e. iff `ord(from) <= ord(to)`.
//!
//! ferray's `is_same_kind` (casting.rs:88) instead uses
//! `is_safe_cast(from,to) || dtype_kind(from) == dtype_kind(to)` where
//! `dtype_kind` (casting.rs:94) lumps ALL integers (signed + unsigned) into a
//! single bucket (1) and treats float (2) / complex (3) as buckets that never
//! cross. This produces TWO divergence classes vs numpy:
//!
//!   CLASS A — signed -> unsigned: ferray allows it (same integer bucket), but
//!     numpy FORBIDS it under same_kind because signed 'i'(2) > unsigned 'u'(1)
//!     so the cast goes "backwards" down the ordering. (Sign loss.)
//!
//!   CLASS B — non-safe int/uint/float -> wider float/complex: ferray FORBIDS
//!     it (different `dtype_kind` buckets and the cast is not safe), but numpy
//!     ALLOWS it because the ordering goes UP (e.g. int 2 <= float 4,
//!     float 4 <= complex 5). These are precision-losing but same-kind-permitted
//!     widenings: int32->float32, int64->float32, uint64->float32,
//!     int32->complex64, float64->complex64, etc.
//!
//! These tests are deliberately LEFT UNMARKED (not `#[ignore]`d): the casting
//! safety lattice is public ABI surface (`arr.cast(CastKind::SameKind)` gates
//! on it), so a wrong same_kind verdict silently rejects or admits casts that
//! numpy admits or rejects — a release-blocker.
//!
//! Blocker: crosslink #<filed-below>.

use ferray_core::DType;
use ferray_core::dtype::casting::{CastKind, can_cast};

// ===========================================================================
// CLASS A — signed -> unsigned must be FALSE under same_kind (sign change is
// "backwards" in numpy's kind ordering: signed 'i'(2) > unsigned 'u'(1)).
// ferray returns TRUE (both are in dtype_kind bucket 1).
// ===========================================================================

/// numpy live: `np.can_cast('int8','uint8','same_kind')` -> False.
#[test]
fn same_kind_int8_to_uint8_matches_numpy() {
    assert!(
        !can_cast(DType::I8, DType::U8, CastKind::SameKind).unwrap(),
        "np.can_cast('int8','uint8','same_kind') == False (signed 'i'>unsigned 'u')"
    );
}

/// numpy live: `np.can_cast('int16','uint8','same_kind')` -> False.
#[test]
fn same_kind_int16_to_uint8_matches_numpy() {
    assert!(
        !can_cast(DType::I16, DType::U8, CastKind::SameKind).unwrap(),
        "np.can_cast('int16','uint8','same_kind') == False (signed -> unsigned)"
    );
}

/// numpy live: `np.can_cast('int32','uint16','same_kind')` -> False.
#[test]
fn same_kind_int32_to_uint16_matches_numpy() {
    assert!(
        !can_cast(DType::I32, DType::U16, CastKind::SameKind).unwrap(),
        "np.can_cast('int32','uint16','same_kind') == False (signed -> unsigned)"
    );
}

/// numpy live: `np.can_cast('int64','uint32','same_kind')` -> False.
#[test]
fn same_kind_int64_to_uint32_matches_numpy() {
    assert!(
        !can_cast(DType::I64, DType::U32, CastKind::SameKind).unwrap(),
        "np.can_cast('int64','uint32','same_kind') == False (signed -> unsigned)"
    );
}

/// numpy live: `np.can_cast('int64','uint64','same_kind')` -> False.
#[test]
fn same_kind_int64_to_uint64_matches_numpy() {
    assert!(
        !can_cast(DType::I64, DType::U64, CastKind::SameKind).unwrap(),
        "np.can_cast('int64','uint64','same_kind') == False (signed -> unsigned)"
    );
}

// ===========================================================================
// CLASS B — non-safe int/uint/float -> wider float/complex must be TRUE under
// same_kind (ordering goes UP: int 2 <= float 4 <= complex 5). ferray returns
// FALSE because the cast is not `is_safe_cast` AND the dtype_kind buckets
// differ (int bucket 1 != float bucket 2 != complex bucket 3).
// ===========================================================================

/// numpy live: `np.can_cast('int32','float32','same_kind')` -> True.
/// (int32 -> float32 is NOT safe — loses precision above 2^24 — yet numpy
/// permits it under same_kind because signed 'i'(2) <= float 'f'(4).)
#[test]
fn same_kind_int32_to_float32_matches_numpy() {
    assert!(
        can_cast(DType::I32, DType::F32, CastKind::SameKind).unwrap(),
        "np.can_cast('int32','float32','same_kind') == True (ord int 2 <= float 4)"
    );
}

/// numpy live: `np.can_cast('int64','float32','same_kind')` -> True.
#[test]
fn same_kind_int64_to_float32_matches_numpy() {
    assert!(
        can_cast(DType::I64, DType::F32, CastKind::SameKind).unwrap(),
        "np.can_cast('int64','float32','same_kind') == True (ord int 2 <= float 4)"
    );
}

/// numpy live: `np.can_cast('uint32','float32','same_kind')` -> True.
#[test]
fn same_kind_uint32_to_float32_matches_numpy() {
    assert!(
        can_cast(DType::U32, DType::F32, CastKind::SameKind).unwrap(),
        "np.can_cast('uint32','float32','same_kind') == True (ord uint 1 <= float 4)"
    );
}

/// numpy live: `np.can_cast('uint64','float32','same_kind')` -> True.
#[test]
fn same_kind_uint64_to_float32_matches_numpy() {
    assert!(
        can_cast(DType::U64, DType::F32, CastKind::SameKind).unwrap(),
        "np.can_cast('uint64','float32','same_kind') == True (ord uint 1 <= float 4)"
    );
}

/// numpy live: `np.can_cast('int32','complex64','same_kind')` -> True.
/// (`complex64` == ferray `DType::Complex32`.)
#[test]
fn same_kind_int32_to_complex64_matches_numpy() {
    assert!(
        can_cast(DType::I32, DType::Complex32, CastKind::SameKind).unwrap(),
        "np.can_cast('int32','complex64','same_kind') == True (ord int 2 <= complex 5)"
    );
}

/// numpy live: `np.can_cast('int64','complex64','same_kind')` -> True.
#[test]
fn same_kind_int64_to_complex64_matches_numpy() {
    assert!(
        can_cast(DType::I64, DType::Complex32, CastKind::SameKind).unwrap(),
        "np.can_cast('int64','complex64','same_kind') == True (ord int 2 <= complex 5)"
    );
}

/// numpy live: `np.can_cast('uint32','complex64','same_kind')` -> True.
#[test]
fn same_kind_uint32_to_complex64_matches_numpy() {
    assert!(
        can_cast(DType::U32, DType::Complex32, CastKind::SameKind).unwrap(),
        "np.can_cast('uint32','complex64','same_kind') == True (ord uint 1 <= complex 5)"
    );
}

/// numpy live: `np.can_cast('uint64','complex64','same_kind')` -> True.
#[test]
fn same_kind_uint64_to_complex64_matches_numpy() {
    assert!(
        can_cast(DType::U64, DType::Complex32, CastKind::SameKind).unwrap(),
        "np.can_cast('uint64','complex64','same_kind') == True (ord uint 1 <= complex 5)"
    );
}

/// numpy live: `np.can_cast('float64','complex64','same_kind')` -> True.
/// (float64 -> complex64 is NOT safe — narrows the f64 real part to f32 — yet
/// numpy permits it under same_kind because float 'f'(4) <= complex 'c'(5).)
#[test]
fn same_kind_float64_to_complex64_matches_numpy() {
    assert!(
        can_cast(DType::F64, DType::Complex32, CastKind::SameKind).unwrap(),
        "np.can_cast('float64','complex64','same_kind') == True (ord float 4 <= complex 5)"
    );
}

// ===========================================================================
// CONTROL CASES — these MUST already match numpy and PASS. They guard against
// an over-broad fix that would flip unsigned->signed (legitimately True) to
// False, or that would make complex->float (legitimately False) True.
// ===========================================================================

/// numpy live: `np.can_cast('uint8','int8','same_kind')` -> True.
/// (unsigned 'u'(1) <= signed 'i'(2): going UP the ordering is allowed.)
#[test]
fn control_same_kind_uint8_to_int8_true() {
    assert!(
        can_cast(DType::U8, DType::I8, CastKind::SameKind).unwrap(),
        "np.can_cast('uint8','int8','same_kind') == True (ord uint 1 <= int 2)"
    );
}

/// numpy live: `np.can_cast('complex64','float64','same_kind')` -> False.
/// (complex 'c'(5) > float 'f'(4): going DOWN the ordering is forbidden.)
#[test]
fn control_same_kind_complex64_to_float64_false() {
    assert!(
        !can_cast(DType::Complex32, DType::F64, CastKind::SameKind).unwrap(),
        "np.can_cast('complex64','float64','same_kind') == False (ord complex 5 > float 4)"
    );
}

/// numpy live: `np.can_cast('float64','int32','same_kind')` -> False.
/// (float 'f'(4) > signed 'i'(2): going DOWN the ordering is forbidden.)
#[test]
fn control_same_kind_float64_to_int32_false() {
    assert!(
        !can_cast(DType::F64, DType::I32, CastKind::SameKind).unwrap(),
        "np.can_cast('float64','int32','same_kind') == False (ord float 4 > int 2)"
    );
}
