//! ACToR critic RE-AUDIT of builder commit c245558e (REQ-23 unary int->float
//! promotion + REQ-24 floor/ceil/trunc/round int identity).
//!
//! All expected values are live `numpy` 2.4.5 oracle results (R-CHAR-3),
//! cross-checked against `numpy/_core/code_generators/generate_umath.py` and
//! `numpy/_core/fromnumeric.py`. Never literal-copied from ferray.
//!
//! DIVERGENCE FIXED (bool-input `round`/`around` output dtype):
//! `np.round`/`np.around` dispatch to `ndarray.round()`
//! (`fromnumeric.py:3674,3691` — `_wrapfunc(a, 'round', ...)`), NOT the
//! `floor`/`ceil`/`trunc` ufuncs. For bool input `ndarray.round()` casts to
//! **float16** (`np.round(np.array([True,False])).dtype == float16`, live
//! numpy 2.4.5) — the SAME promotion the `rint` ufunc applies (`rint`
//! registers no `TD(bints)`, `generate_umath.py:1021`). The original builder's
//! `round_int`/`around_int` returned bool input UNCHANGED in the bool dtype
//! (REQ-24 "int identity"), which is correct for the genuine integer-loop
//! ufuncs floor/ceil/trunc/fix (those DO register `TD(bints)` first,
//! `generate_umath.py:1011`) but WRONG for round/around on bool. The fix adds
//! `round_bool`/`around_bool` (bool→float16 via `rint_promote`); the pins below
//! call that path and assert the promoted output dtype/values.

// The bool-input round/around promotion target is `half::f16`, which only
// exists under the `f16` feature — gate the whole audit on it so the test
// crate compiles (and is empty) when run without `--features f16`.
#![cfg(feature = "f16")]

use ferray_core::Array;
use ferray_core::dimension::Ix1;
use ferray_core::dtype::{DType, Element};

fn abool(v: Vec<bool>) -> Array<bool, Ix1> {
    Array::<bool, Ix1>::from_vec(Ix1::new([v.len()]), v).unwrap()
}

// ---------------------------------------------------------------------------
// PINNED DIVERGENCE — bool-input round/around output dtype.
//
// numpy 2.4.5 (live):
//   np.round(np.array([True, False, True])).dtype    == float16
//   np.around(np.array([True, False, True])).dtype   == float16
//   np.round(np.array([True, False, True])).tolist() == [1.0, 0.0, 1.0]
// Source: fromnumeric.py:3674 round -> _wrapfunc(a,'round',...) ->
//   ndarray.round, whose bool kernel promotes to float16 (same as the rint
//   ufunc, generate_umath.py:1021 has no TD(bints)).
//
// Fix: `round_bool`/`around_bool<bool>` return Array<half::f16> via the
// `rint_promote` (PromoteFloat) path, so the output element type is `f16`
// (DType::F16) with values `[1.0, 0.0, 1.0]` — the int8..uint64 `round_int`/
// `around_int` identity path is unchanged (still input-dtype identity).
//
// The assertions below inspect the REAL output array's element dtype against
// the numpy-true float16 result, and the promoted values. numpy yields
// float16; ferray's `round_bool`/`around_bool` now yield float16 too.
// ---------------------------------------------------------------------------

/// `np.round(bool)` -> float16 `[1.0,0.0,1.0]`; `round_bool(bool)` matches.
#[test]
fn divergence_round_bool_promotes_to_f16_not_bool_identity() {
    // numpy 2.4.5: np.round(np.array([True,False,True])).dtype == float16
    const NUMPY_ROUND_BOOL_DTYPE: DType = DType::F16;

    let out = ferray_ufunc::round_bool(&abool(vec![true, false, true])).unwrap();
    // round_bool<bool> yields Array<half::f16>; inspect its REAL element dtype.
    let actual_dtype = <half::f16 as Element>::dtype();
    // Promoted values: round-half-to-even of 1.0/0.0 is the identity.
    let vals: Vec<f32> = out.as_slice().unwrap().iter().map(|x| x.to_f32()).collect();
    assert_eq!(vals, &[1.0, 0.0, 1.0]);

    // numpy promotes bool round -> float16; ferray now matches.
    assert_eq!(
        actual_dtype, NUMPY_ROUND_BOOL_DTYPE,
        "np.round(bool) -> float16 (fromnumeric.py:3674 -> ndarray.round), \
         ferray round_bool(bool) -> float16"
    );
}

/// `np.around(bool)` -> float16 (alias of round); `around_bool(bool)` matches.
#[test]
fn divergence_around_bool_promotes_to_f16_not_bool_identity() {
    // numpy 2.4.5: np.around(np.array([True,False])).dtype == float16
    const NUMPY_AROUND_BOOL_DTYPE: DType = DType::F16;

    let out = ferray_ufunc::around_bool(&abool(vec![true, false])).unwrap();
    let actual_dtype = <half::f16 as Element>::dtype();
    let vals: Vec<f32> = out.as_slice().unwrap().iter().map(|x| x.to_f32()).collect();
    assert_eq!(vals, &[1.0, 0.0]);

    assert_eq!(
        actual_dtype, NUMPY_AROUND_BOOL_DTYPE,
        "np.around(bool) -> float16 (fromnumeric.py:3691 alias of round), \
         ferray around_bool(bool) -> float16"
    );
}
