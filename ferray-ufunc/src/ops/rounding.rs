//! ferray-ufunc: Rounding functions
//!
//! round (banker's rounding!), floor, ceil, trunc, fix, rint, around
//!
//! ## REQ status — REQ-24 floor/ceil/trunc/round integer-input identity
//!
//! SHIPPED: `floor_int`/`ceil_int`/`trunc_int`/`round_int`/`fix_int`/
//! `around_int` accept `T: Element + Copy` (any int/bool) and return the input
//! UNCHANGED in the INPUT-int-dtype via `round_identity` — `np.floor(int32)→
//! int32`, `np.round(int64)→int64`. Mirrors NumPy registering `TD(bints)`
//! FIRST for floor/ceil/trunc (floor `generate_umath.py:1011`, ceil `:983`,
//! trunc `:993`). BOOL EXCEPTION for round/around: `floor`/`ceil`/`trunc`/`fix`
//! keep bool as bool (those `TD(bints)` loops cover bool), but `round`/`around`
//! on bool PROMOTE to **float16** like `rint` — `round`/`around` dispatch to
//! `ndarray.round` whose bool kernel has no `TD(bints)`
//! (`generate_umath.py:1021`, same as `rint`), so `np.round(bool)→float16`
//! `[1.0,0.0,1.0]`. That bool path is `round_bool`/`around_bool` here, routing
//! through `rint_promote` (`PromoteFloat`, bool→f16). int8..uint64 round/around
//! stay input-dtype identity. This is the OPPOSITE of `rint`, which has NO
//! `TD(bints)` for ANY integer and promotes every integer input to float
//! (REQ-23) — served by `rint_promote` in `promoted.rs`. The float
//! `floor`/`ceil`/`trunc`/`round`/`rint` (`T: Float`) are untouched (f32/f64
//! byte-identical). Consumer: the `*_int`/`*_bool` entries are the
//! integer/bool-rounding public surface re-exported from `lib.rs`. Verified by
//! `tests/divergence_unary_promote.rs` and `divergence_unary_promote_audit.rs`.

use ferray_core::Array;
use ferray_core::dimension::Dimension;
use ferray_core::dtype::Element;
use ferray_core::error::FerrayResult;
use num_traits::Float;

use crate::helpers::unary_float_op;

/// Banker's rounding: round half to even (AC-9).
///
/// `round(0.5) == 0`, `round(1.5) == 2`, `round(2.5) == 2`.
fn bankers_round<T: Float>(x: T) -> T {
    // Check if x is exactly at a .5 boundary
    let half = T::from(0.5).unwrap();
    let two = T::from(2.0).unwrap();

    // Get the fractional part: x - floor(x)
    let floored = x.floor();
    let frac = x - floored;

    // Check if fractional part is exactly 0.5
    if frac == half {
        // At exact .5 -- round to even
        let ceiled = x.ceil();
        // Check which of floor/ceil is even
        // A number is even if dividing by 2 and flooring gives back the same
        if (floored / two).floor() * two == floored {
            floored
        } else {
            ceiled
        }
    } else if frac == -half {
        // Negative half case: x is negative, frac = x - floor(x) could be 0.5 for negatives
        // Actually for negative numbers like -0.5: floor(-0.5) = -1, frac = -0.5 - (-1) = 0.5
        // So the above branch handles it. This branch is for safety.
        x.ceil()
    } else {
        // Not at a .5 boundary, standard rounding is fine
        x.round()
    }
}

/// Elementwise banker's rounding (round half to even).
///
/// This matches `NumPy`'s `np.round` / `np.around` behavior.
/// AC-9: `round(0.5)==0`, `round(1.5)==2`.
pub fn round<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op(input, bankers_round)
}

/// Alias for [`round`] -- matches `NumPy`'s `around`.
pub fn around<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    round(input)
}

/// Alias for [`round`] -- matches `NumPy`'s `rint`.
pub fn rint<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    round(input)
}

/// Elementwise floor (round toward negative infinity).
pub fn floor<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op(input, T::floor)
}

/// Elementwise ceiling (round toward positive infinity).
pub fn ceil<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op(input, T::ceil)
}

/// Elementwise truncation (round toward zero).
pub fn trunc<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op(input, T::trunc)
}

/// Elementwise fix: round toward zero (same as trunc for real numbers).
pub fn fix<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    trunc(input)
}

// ---------------------------------------------------------------------------
// Integer-input rounding (REQ-24): floor/ceil/trunc/round/fix/around are
// INT-IDENTITY on integer/bool input.
//
// NumPy registers `TD(bints)` FIRST for floor/ceil/trunc (and the rounding
// resolver routes round/around/fix the same way), so an integer or bool
// input array selects the integer loop and is returned UNCHANGED, in the
// INPUT dtype — `np.floor(int64)->int64`, `np.ceil(int32)->int32`,
// `np.floor(bool)->bool`
// (numpy/_core/code_generators/generate_umath.py:983 ceil, :993 trunc,
// :1011 floor — `TD(bints)` is the first registered loop). This is the
// OPPOSITE of `rint`, which has NO `TD(bints)` (`generate_umath.py:1021`)
// and promotes integer input to float (REQ-23, `rint_promote`). ferray's
// float `floor`/`ceil`/`trunc`/`round` above are `T: Float` and reject
// integer input at compile time, so these `*_int` siblings carry the
// integer contract; the float entry points are untouched (f32/f64
// byte-identical).
//
// `T: Element + Copy` (no `Float`) accepts every integer and bool element
// type. The op is a pure identity copy preserving shape, layout, and dtype.
// ---------------------------------------------------------------------------

/// Identity copy preserving the input dtype/shape — the shared kernel for
/// the integer-input rounding ops (REQ-24).
#[inline]
fn round_identity<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Copy,
    D: Dimension,
{
    let data: Vec<T> = input.iter().copied().collect();
    Array::from_vec(input.dim().clone(), data)
}

/// `floor` on integer/bool input: returns the values unchanged in the input
/// dtype (REQ-24, `TD(bints)` first). `np.floor(int64 [1,2,4]) == [1,2,4]`
/// with `.dtype == int64`; `np.floor(bool [T,F]).dtype == bool`.
pub fn floor_int<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Copy,
    D: Dimension,
{
    round_identity(input)
}

/// `ceil` on integer/bool input: int-identity (REQ-24).
/// `np.ceil(int32 [5]).dtype == int32`.
pub fn ceil_int<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Copy,
    D: Dimension,
{
    round_identity(input)
}

/// `trunc` on integer/bool input: int-identity (REQ-24).
pub fn trunc_int<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Copy,
    D: Dimension,
{
    round_identity(input)
}

/// `round` on integer/bool input: int-identity (REQ-24). Unlike `rint`
/// (REQ-23 float-promote), `round`/`around` keep integer input integer.
pub fn round_int<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Copy,
    D: Dimension,
{
    round_identity(input)
}

/// `around` (alias of `round`) on integer/bool input: int-identity (REQ-24).
pub fn around_int<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Copy,
    D: Dimension,
{
    round_identity(input)
}

// ---------------------------------------------------------------------------
// bool-input round/around: promote to float16 (NOT bool identity).
//
// Unlike floor/ceil/trunc/fix (which register `TD(bints)` FIRST, so bool stays
// bool — handled by the `*_int` identity above), `round`/`around` dispatch to
// `ndarray.round()` whose bool kernel promotes to float16 — the SAME promotion
// the `rint` ufunc applies (`rint` registers `TD('e', f='rint')` FIRST with NO
// `TD(bints)`, `generate_umath.py:1021`). Live numpy 2.4.5:
//   np.round(np.array([True,False,True])).dtype  == float16
//   np.round(np.array([True,False,True])).tolist() == [1.0, 0.0, 1.0]
// So bool round/around route through the existing `PromoteFloat` machinery
// (`crate::rint_promote`, bool->float16) rather than `round_identity`. The
// round-half-to-even of the promoted 0.0/1.0 values is the identity, so the
// f16 output equals numpy's `[1.0, 0.0, 1.0]`. The non-bool integer dtypes
// (int8..uint64) keep their input-dtype identity via `round_int`/`around_int`.
// ---------------------------------------------------------------------------

/// `round` on bool input: promotes to float16 (REQ-24, bool exception).
///
/// `np.round(np.array([True,False,True]))` is `float16 [1.0,0.0,1.0]` — bool
/// round has no `TD(bints)` loop and promotes like `rint`
/// (`generate_umath.py:1021`), unlike `floor`/`ceil`/`trunc` which keep bool.
#[cfg(feature = "f16")]
pub fn round_bool<D>(input: &Array<bool, D>) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    crate::rint_promote(input)
}

/// `around` (alias of `round`) on bool input: promotes to float16 (REQ-24,
/// bool exception). See [`round_bool`].
#[cfg(feature = "f16")]
pub fn around_bool<D>(input: &Array<bool, D>) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    round_bool(input)
}

/// `fix` (alias of `trunc`) on integer/bool input: int-identity (REQ-24).
pub fn fix_int<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Copy,
    D: Dimension,
{
    round_identity(input)
}

// ---------------------------------------------------------------------------
// f16 variants (f32-promoted) — generated via the shared unary_f16_fn!
// macro (#142).
// ---------------------------------------------------------------------------

use crate::helpers::unary_f16_fn;

unary_f16_fn!(
    /// Elementwise floor for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    floor_f16,
    f32::floor
);
unary_f16_fn!(
    /// Elementwise ceiling for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    ceil_f16,
    f32::ceil
);
unary_f16_fn!(
    /// Elementwise truncation for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    trunc_f16,
    f32::trunc
);
unary_f16_fn!(
    /// Elementwise banker's rounding for f16 arrays via f32 promotion.
    ///
    /// Reuses the generic [`bankers_round`] via monomorphization on
    /// `f32`; the hand-rolled f32 copy was deleted in #144.
    #[cfg(feature = "f16")]
    round_f16,
    bankers_round::<f32>
);

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_util::arr1;

    #[test]
    fn test_bankers_round_half_to_even_ac9() {
        // AC-9: round(0.5)==0, round(1.5)==2
        let a = arr1(vec![0.5, 1.5, 2.5, 3.5, -0.5, -1.5]);
        let r = round(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert_eq!(s[0], 0.0); // 0.5 -> 0 (even)
        assert_eq!(s[1], 2.0); // 1.5 -> 2 (even)
        assert_eq!(s[2], 2.0); // 2.5 -> 2 (even)
        assert_eq!(s[3], 4.0); // 3.5 -> 4 (even)
        assert_eq!(s[4], 0.0); // -0.5 -> 0 (even)
        assert_eq!(s[5], -2.0); // -1.5 -> -2 (even)
    }

    #[test]
    fn test_round_normal() {
        let a = arr1(vec![1.2, 2.7, -1.3, -2.8]);
        let r = round(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert_eq!(s[0], 1.0);
        assert_eq!(s[1], 3.0);
        assert_eq!(s[2], -1.0);
        assert_eq!(s[3], -3.0);
    }

    #[test]
    fn test_floor() {
        let a = arr1(vec![1.7, -1.7, 0.0]);
        let r = floor(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert_eq!(s[0], 1.0);
        assert_eq!(s[1], -2.0);
        assert_eq!(s[2], 0.0);
    }

    #[test]
    fn test_ceil() {
        let a = arr1(vec![1.2, -1.2, 0.0]);
        let r = ceil(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert_eq!(s[0], 2.0);
        assert_eq!(s[1], -1.0);
        assert_eq!(s[2], 0.0);
    }

    #[test]
    fn test_trunc() {
        let a = arr1(vec![1.9, -1.9, 0.0]);
        let r = trunc(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert_eq!(s[0], 1.0);
        assert_eq!(s[1], -1.0);
        assert_eq!(s[2], 0.0);
    }

    #[test]
    fn test_fix() {
        let a = arr1(vec![2.9, -2.9]);
        let r = fix(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert_eq!(s[0], 2.0);
        assert_eq!(s[1], -2.0);
    }

    #[test]
    fn test_around_alias() {
        let a = arr1(vec![0.5, 1.5]);
        let r = around(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert_eq!(s[0], 0.0);
        assert_eq!(s[1], 2.0);
    }

    #[test]
    fn test_rint_alias() {
        let a = arr1(vec![0.5, 1.5]);
        let r = rint(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert_eq!(s[0], 0.0);
        assert_eq!(s[1], 2.0);
    }

    // ----------------------------------------------------------------------
    // f32 sibling tests (#152) — every rounding op exercised on f32 to
    // verify the SIMD f32 path and confirm bit-exact rounding behaviour
    // matches the f64 path on values both representable.
    // ----------------------------------------------------------------------

    use ferray_core::Array;
    use ferray_core::dimension::Ix1;

    fn arr1_f32(data: Vec<f32>) -> Array<f32, Ix1> {
        Array::<f32, Ix1>::from_vec(Ix1::new([data.len()]), data).unwrap()
    }

    #[test]
    fn test_bankers_round_half_to_even_f32() {
        let a = arr1_f32(vec![0.5, 1.5, 2.5, 3.5, -0.5, -1.5]);
        let r = round(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert_eq!(s[0], 0.0);
        assert_eq!(s[1], 2.0);
        assert_eq!(s[2], 2.0);
        assert_eq!(s[3], 4.0);
        assert_eq!(s[4], 0.0);
        assert_eq!(s[5], -2.0);
    }

    #[test]
    fn test_round_normal_f32() {
        let a = arr1_f32(vec![1.2, 2.7, -1.3, -2.8]);
        let r = round(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert_eq!(s[0], 1.0);
        assert_eq!(s[1], 3.0);
        assert_eq!(s[2], -1.0);
        assert_eq!(s[3], -3.0);
    }

    #[test]
    fn test_floor_f32() {
        let a = arr1_f32(vec![1.7, -1.7, 0.0]);
        let r = floor(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert_eq!(s[0], 1.0);
        assert_eq!(s[1], -2.0);
        assert_eq!(s[2], 0.0);
    }

    #[test]
    fn test_ceil_f32() {
        let a = arr1_f32(vec![1.2, -1.2, 0.0]);
        let r = ceil(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert_eq!(s[0], 2.0);
        assert_eq!(s[1], -1.0);
        assert_eq!(s[2], 0.0);
    }

    #[test]
    fn test_trunc_f32() {
        let a = arr1_f32(vec![1.9, -1.9, 0.0]);
        let r = trunc(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert_eq!(s[0], 1.0);
        assert_eq!(s[1], -1.0);
        assert_eq!(s[2], 0.0);
    }

    #[test]
    fn test_fix_f32() {
        let a = arr1_f32(vec![2.9, -2.9]);
        let r = fix(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert_eq!(s[0], 2.0);
        assert_eq!(s[1], -2.0);
    }

    #[test]
    fn test_around_alias_f32() {
        let a = arr1_f32(vec![0.5, 1.5]);
        let r = around(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert_eq!(s[0], 0.0);
        assert_eq!(s[1], 2.0);
    }

    #[test]
    fn test_rint_alias_f32() {
        let a = arr1_f32(vec![0.5, 1.5]);
        let r = rint(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert_eq!(s[0], 0.0);
        assert_eq!(s[1], 2.0);
    }
}
