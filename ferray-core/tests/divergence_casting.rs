//! ACToR-CRITIC divergence pins for `ferray_core::dtype::casting::can_cast`
//! against the NumPy 2.4.5 live oracle (`numpy.can_cast`).
//!
//! API under audit:
//! `pub fn can_cast(from: DType, to: DType, casting: CastKind) -> FerrayResult<bool>`
//! in `ferray-core/src/dtype/casting.rs`, with
//! `CastKind::{No, Equiv, Safe, SameKind, Unsafe}`.
//!
//! Every expected value in this file is the result of a LIVE numpy call
//! (`numpy.can_cast(from, to, casting)`, numpy 2.4.5) — never copied from
//! ferray's own output (R-CHAR-3).
//!
//! GOVERNING CONTRACT (numpy `same_kind` semantics):
//! numpy's `'same_kind'` casting is "this cast is OK under `'safe'`, OR the
//! cast stays within the same kind". In numpy/_core/src/multiarray/
//! convert_datatype.c, `PyArray_CheckCastSafety` first checks the safe cast
//! result and only then falls back to the same-kind kind-ordering check
//! (`PyArray_MinCastSafety` / `dtype_kind_to_ordering`). Concretely numpy's
//! casting lattice is ordered no < equiv < safe < same_kind < unsafe, so a
//! cast that is allowed at the `'safe'` level is ALWAYS allowed at
//! `'same_kind'`. ferray's `is_same_kind` instead ONLY compares a kind
//! category (`dtype_kind`) and drops the "safe ⊆ same_kind" subsumption, so
//! cross-kind-but-safe casts (int→float, int→complex, float→complex) and ALL
//! bool→numeric casts wrongly report `false` under `CastKind::SameKind`.
//!
//! Additionally, numpy treats `bool` as safely castable to EVERY numeric type
//! (bool is the bottom of the casting lattice); ferray assigns bool its own
//! `dtype_kind` (0) that only matches itself, so `bool -> <numeric>` under
//! `same_kind` diverges.
//!
//! Blocker: crosslink #777 (single root cause — `is_same_kind` must subsume
//! `is_safe_cast`).

use ferray_core::DType;
use ferray_core::dtype::casting::{CastKind, can_cast};

// ---------------------------------------------------------------------------
// DIVERGENCE CLASS A: same_kind must subsume safe — cross-kind safe casts.
//
// numpy: a cast that is safe is also allowed under same_kind (the lattice
// orders safe < same_kind). int->float / int->complex / float->complex are
// all safe (widening within numpy's promotion table) and therefore True under
// same_kind. ferray's is_same_kind only checks dtype_kind equality, so these
// return False.
// ---------------------------------------------------------------------------

/// numpy live: `np.can_cast('int8','float64','same_kind')` -> True.
#[test]
fn same_kind_int8_to_float64_matches_numpy() {
    assert!(
        can_cast(DType::I8, DType::F64, CastKind::SameKind).unwrap(),
        "np.can_cast('int8','float64','same_kind') == True (safe ⊆ same_kind)"
    );
}

/// numpy live: `np.can_cast('int16','float32','same_kind')` -> True.
#[test]
fn same_kind_int16_to_float32_matches_numpy() {
    assert!(
        can_cast(DType::I16, DType::F32, CastKind::SameKind).unwrap(),
        "np.can_cast('int16','float32','same_kind') == True (safe ⊆ same_kind)"
    );
}

/// numpy live: `np.can_cast('uint8','float64','same_kind')` -> True.
#[test]
fn same_kind_uint8_to_float64_matches_numpy() {
    assert!(
        can_cast(DType::U8, DType::F64, CastKind::SameKind).unwrap(),
        "np.can_cast('uint8','float64','same_kind') == True (safe ⊆ same_kind)"
    );
}

/// numpy live: `np.can_cast('int32','complex128','same_kind')` -> True.
#[test]
fn same_kind_int32_to_complex128_matches_numpy() {
    assert!(
        can_cast(DType::I32, DType::Complex64, CastKind::SameKind).unwrap(),
        "np.can_cast('int32','complex128','same_kind') == True (safe ⊆ same_kind)"
    );
}

/// numpy live: `np.can_cast('float64','complex128','same_kind')` -> True.
#[test]
fn same_kind_float64_to_complex128_matches_numpy() {
    assert!(
        can_cast(DType::F64, DType::Complex64, CastKind::SameKind).unwrap(),
        "np.can_cast('float64','complex128','same_kind') == True (safe ⊆ same_kind)"
    );
}

// ---------------------------------------------------------------------------
// DIVERGENCE CLASS B: bool is the bottom of the casting lattice.
//
// numpy treats bool as safely castable to every numeric type, so bool->X is
// True under both 'safe' and 'same_kind'. ferray gives bool its own dtype_kind
// (0) that only matches itself, so bool->int / bool->float / bool->complex are
// wrongly False under same_kind.
// ---------------------------------------------------------------------------

/// numpy live: `np.can_cast('bool','int8','same_kind')` -> True.
#[test]
fn same_kind_bool_to_int8_matches_numpy() {
    assert!(
        can_cast(DType::Bool, DType::I8, CastKind::SameKind).unwrap(),
        "np.can_cast('bool','int8','same_kind') == True (bool is lattice bottom)"
    );
}

/// numpy live: `np.can_cast('bool','float64','same_kind')` -> True.
#[test]
fn same_kind_bool_to_float64_matches_numpy() {
    assert!(
        can_cast(DType::Bool, DType::F64, CastKind::SameKind).unwrap(),
        "np.can_cast('bool','float64','same_kind') == True (bool is lattice bottom)"
    );
}

/// numpy live: `np.can_cast('bool','complex128','same_kind')` -> True.
#[test]
fn same_kind_bool_to_complex128_matches_numpy() {
    assert!(
        can_cast(DType::Bool, DType::Complex64, CastKind::SameKind).unwrap(),
        "np.can_cast('bool','complex128','same_kind') == True (bool is lattice bottom)"
    );
}
