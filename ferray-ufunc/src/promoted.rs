//! Mixed-type (promoted) variants of arithmetic ufuncs.
//!
//! `NumPy`'s ufunc type resolver automatically promotes operands to a
//! common result type — `np.add(int32_arr, float64_arr)` just works.
//! ferray-core already carries the compile-time promotion machinery
//! (`Promoted` for result-type resolution, `PromoteTo` for element
//! casting), so this module is glue: cast each operand to the promoted
//! output type, then invoke the existing same-type ufunc.
//!
//! Scope: the four basic arithmetic ops (add / subtract / multiply /
//! divide) and elementwise max/min. These cover the vast majority of
//! `NumPy` calls with mixed dtypes. Unary ufuncs don't need promotion
//! because there's only one operand, and the float-only transcendentals
//! (sin, exp, log, …) can't meaningfully accept integer inputs without
//! first promoting them — callers that need that can `.cast::<f64>()`
//! explicitly.
//!
//! All helpers here take same-shape inputs (no broadcasting on the
//! promoted path — composing promotion with broadcasting is not hard
//! but is deferred until there's a concrete caller that wants it).
//!
//! ## REQ status — NEP-50 mixed-width integer promotion
//!
//! SHIPPED: `add_promoted` / `subtract_promoted` / `multiply_promoted`
//! resolve the output dtype via `Promoted` and run the wrapping integer
//! kernel (`WrappingArith`) so `int8 + int64 -> int64`, `int32 * int64 ->
//! int64` work and overflow wraps like NumPy's fixed-width integers.
//! `divide_promoted` stays float-output (NumPy true-division): integer
//! true-division is the same-type `divide` via `TrueDivide`.
//!
//! ## REQ status — REQ-23 unary float-ufunc int/bool→float promotion
//!
//! SHIPPED: the `PromoteFloat` trait + the `*_promote` family (`sqrt_promote`,
//! `exp_promote`, `rint_promote`, `sin_promote`, … 24 entry points generated
//! by `unary_promote_fn!`) accept integer/bool input and promote to the
//! smallest-safe-cast float (`PromoteFloat::Out`: i16/u16→f32, i32/i64/u32/
//! u64→f64; bool/i8/u8→half::f16 under the `f16` feature). `unary_promote_float`
//! casts int→compute-float, runs the existing `T: Float` ufunc unchanged, and
//! narrows. f32/f64 float callers are byte-identical (the existing kernels are
//! never modified). Mirrors NumPy's float-only loop registration
//! (`generate_umath.py:907-1027`: `sqrt`/`exp`/`rint` register no `TD(bints)`).
//! Consumer: `rint_promote` is the production callee for the rounding crate's
//! integer-`rint` contract; the family is the workspace's int-accepting unary
//! ufunc surface. Verified by `tests/divergence_unary_promote.rs`.
//!
//! ## REQ status — REQ-25 binary float-ufunc int/bool→float promotion
//!
//! SHIPPED: the binary `*_promote` family (`hypot_promote`, `arctan2_promote`,
//! `logaddexp_promote`, `logaddexp2_promote`, `copysign_promote`,
//! `nextafter_promote`) accepts a same-type integer/bool input PAIR and promotes
//! to the smallest-safe-cast float, reusing the REQ-23 `PromoteFloat` trait
//! unchanged (`PromoteFloat::Out`: i16/u16→f32, i32/i64/u32/u64→f64; bool/i8/u8
//! →half::f16 under `f16`). The shared `fn binary_promote_float in promoted.rs`
//! casts both operands int→compute-float, runs the EXISTING generic `T: Float`
//! binary ufunc (`crate::hypot`/`crate::arctan2`/…) monomorphised at the compute
//! float, and narrows — so existing f32/f64 callers are byte-identical. Mirrors
//! NumPy's float-only loop registration (`generate_umath.py`: hypot :1057-1063,
//! arctan2 :1030-1037, logaddexp/logaddexp2 :710-721, copysign/nextafter
//! :1107-1119 — all `TD(flts)`, no `TD(bints)`). Mixed-width integer pairs
//! resolve via NEP-50 then to float (`np.hypot(int16, int32)`→float64); the
//! same-`T` binary kernels cover the same-width contract. Consumer: re-exported
//! from `lib.rs` as the int-accepting binary-ufunc public surface (the
//! ferray-python binary-float dispatch target). Verified by
//! `tests/divergence_binary_promote.rs`.

use ferray_core::Array;
use ferray_core::dimension::Dimension;
use ferray_core::dtype::Element;
use ferray_core::dtype::promotion::{PromoteTo, Promoted};
use ferray_core::error::{FerrayError, FerrayResult};
use num_traits::Float;

use crate::helpers::binary_elementwise_op;
use crate::ops::arithmetic::WrappingArith;

// ---------------------------------------------------------------------------
// Unary float-ufunc integer/bool input promotion (REQ-23)
//
// NumPy accepts integer and bool input across the entire unary float-ufunc
// family (exp/log/sqrt/sin/.../rint). These ops register only float loops
// (`generate_umath.py`: e.g. `sqrt` starts at `TD('e', ...)` then
// `TD('fdg'+cmplx)`, with NO `TD(bints)`), so an integer/bool input
// safe-casts UP to the smallest float loop that can hold its kind. NumPy
// resolves that to: bool/int8/uint8 -> float16, int16/uint16 -> float32,
// int32/int64/uint32/uint64 -> float64
// (numpy/_core/code_generators/generate_umath.py:907-1027). This mirrors
// ferray-core's `Promoted` int+float table (e.g. `i16 + f32 -> f32`,
// `i32 + f32 -> f64`) projected onto the "promote against the smallest
// float" rule.
//
// `PromoteFloat` carries that mapping at the type level. The f32/f64-output
// kinds cast the integer up to the compute float and run the EXISTING
// generic `T: Float` ufunc unchanged, so existing f32/f64 callers are
// byte-identical (this module never touches their code path). The
// f16-output kinds (bool/int8/uint8) compute the math in f32 and narrow the
// result to `half::f16` — exactly how NumPy's float16 ufunc loop works
// (`astype={'e': 'f'}` in the `TD('e', ...)` entries) and how ferray's
// existing `*_f16` ops behave; `half::f16` additionally lacks a `CrMath`
// impl, so routing through the f32 kernel is also the only correct option
// for the transcendentals.
// ---------------------------------------------------------------------------

/// Type-level mapping from an integer/bool input element to the floating
/// output kind NumPy promotes it to for unary float ufuncs (REQ-23).
///
/// `Compute` is the float type the kernel actually runs in (f32 for the
/// f16- and f32-output kinds, f64 for the f64-output kind); `Out` is the
/// element type of the returned array. For the f32/f64 kinds `Out ==
/// Compute`; for the f16 kinds the f32 result is narrowed via
/// [`narrow`](PromoteFloat::narrow).
///
/// This trait is implemented only for the integer and bool element types —
/// floats are intentionally excluded so the public `*_promote` entry points
/// reject `f32`/`f64` input (those callers use the existing `sqrt`/`exp`/…
/// directly, with no promotion).
pub trait PromoteFloat: Element + Copy {
    /// The float type the kernel computes in.
    type Compute: Element + Copy + Float;
    /// The element type of the promoted output array (`f16`/`f32`/`f64`).
    type Out: Element + Copy;

    /// Widen one input element to the compute float.
    fn to_compute(self) -> Self::Compute;
    /// Narrow one computed element to the output type.
    fn narrow(c: Self::Compute) -> Self::Out;
}

/// f32/f64-output kinds: `Out == Compute`, narrow is the identity.
macro_rules! impl_promote_float_same {
    ($int:ty => $flt:ty) => {
        impl PromoteFloat for $int {
            type Compute = $flt;
            type Out = $flt;
            #[inline]
            fn to_compute(self) -> $flt {
                <$int as PromoteTo<$flt>>::promote(self)
            }
            #[inline]
            fn narrow(c: $flt) -> $flt {
                c
            }
        }
    };
}

// int16/uint16 -> float32 (numpy: `i16 + f32 -> f32`).
impl_promote_float_same!(i16 => f32);
impl_promote_float_same!(u16 => f32);
// int32/int64/uint32/uint64 -> float64 (numpy: `i32 + f32 -> f64`, etc.).
impl_promote_float_same!(i32 => f64);
impl_promote_float_same!(i64 => f64);
impl_promote_float_same!(u32 => f64);
impl_promote_float_same!(u64 => f64);

/// f16-output kinds (bool/int8/uint8): compute in f32, narrow to `half::f16`.
/// Feature-gated because `half::f16` only exists under the `f16` feature.
#[cfg(feature = "f16")]
macro_rules! impl_promote_float_f16 {
    ($int:ty) => {
        impl PromoteFloat for $int {
            type Compute = f32;
            type Out = half::f16;
            #[inline]
            fn to_compute(self) -> f32 {
                <$int as PromoteTo<f32>>::promote(self)
            }
            #[inline]
            fn narrow(c: f32) -> half::f16 {
                half::f16::from_f32(c)
            }
        }
    };
}

#[cfg(feature = "f16")]
impl_promote_float_f16!(bool);
#[cfg(feature = "f16")]
impl_promote_float_f16!(i8);
#[cfg(feature = "f16")]
impl_promote_float_f16!(u8);

/// Cast an integer/bool input array to its compute float, run a generic
/// unary float kernel `op` (the existing `T: Float` ufunc monomorphised at
/// the compute type), and narrow the result to the promoted output type.
///
/// Shared by every `*_promote` entry point in this module so the int->float
/// promotion logic lives in exactly one place. The `op` closure receives a
/// `&Array<Compute, D>` and returns `Array<Compute, D>`; for the f32/f64
/// kinds `narrow` is the identity so the existing kernel's bytes flow
/// straight through.
#[inline]
fn unary_promote_float<T, D>(
    input: &Array<T, D>,
    op: impl Fn(&Array<T::Compute, D>) -> FerrayResult<Array<T::Compute, D>>,
) -> FerrayResult<Array<<T as PromoteFloat>::Out, D>>
where
    T: PromoteFloat,
    D: Dimension,
{
    // Cast int/bool -> compute float, reusing the contiguous fast path.
    let compute: Array<T::Compute, D> = if let Some(slice) = input.as_slice() {
        let data: Vec<T::Compute> = slice.iter().map(|&x| x.to_compute()).collect();
        Array::from_vec(input.dim().clone(), data)?
    } else {
        let data: Vec<T::Compute> = input.iter().map(|&x| x.to_compute()).collect();
        Array::from_vec(input.dim().clone(), data)?
    };
    let result = op(&compute)?;
    // Narrow to the output kind (identity for f32/f64, f32->f16 for f16).
    let out: Array<<T as PromoteFloat>::Out, D> = if let Some(slice) = result.as_slice() {
        let data: Vec<<T as PromoteFloat>::Out> = slice.iter().map(|&c| T::narrow(c)).collect();
        Array::from_vec(result.dim().clone(), data)?
    } else {
        let data: Vec<<T as PromoteFloat>::Out> = result.iter().map(|&c| T::narrow(c)).collect();
        Array::from_vec(result.dim().clone(), data)?
    };
    Ok(out)
}

/// Generate a `pub fn <name>_promote` int/bool-accepting entry point that
/// promotes per REQ-23 and routes through the existing float ufunc
/// `$op_path`. `$op_path` must be the crate's generic `T: Float` (or
/// `T: Float + CrMath`) unary ufunc — it is called monomorphised at the
/// resolved compute float, so f32/f64 results are byte-identical to a direct
/// float call.
macro_rules! unary_promote_fn {
    (
        $(#[$attr:meta])*
        $name:ident,
        $op_path:path
    ) => {
        $(#[$attr])*
        pub fn $name<T, D>(
            input: &Array<T, D>,
        ) -> FerrayResult<Array<<T as PromoteFloat>::Out, D>>
        where
            T: PromoteFloat,
            T::Compute: crate::cr_math::CrMath,
            D: Dimension,
        {
            unary_promote_float(input, |c| $op_path(c))
        }
    };
}

// exp / log family (need CrMath on the compute float — f32 and f64 both
// impl CrMath, so the `T::Compute: CrMath` bound is always satisfiable).
unary_promote_fn!(
    /// `exp` with NumPy int/bool->float promotion (REQ-23). `np.exp(int64
    /// [1,2,4]) -> float64`; `np.exp(uint8 [1]) -> float16`.
    exp_promote,
    crate::exp
);
unary_promote_fn!(
    /// `exp2` with int/bool->float promotion (REQ-23).
    exp2_promote,
    crate::exp2
);
unary_promote_fn!(
    /// `expm1` with int/bool->float promotion (REQ-23).
    expm1_promote,
    crate::expm1
);
unary_promote_fn!(
    /// `log` with int/bool->float promotion (REQ-23).
    log_promote,
    crate::log
);
unary_promote_fn!(
    /// `log2` with int/bool->float promotion (REQ-23).
    log2_promote,
    crate::log2
);
unary_promote_fn!(
    /// `log10` with int/bool->float promotion (REQ-23).
    log10_promote,
    crate::log10
);
unary_promote_fn!(
    /// `log1p` with int/bool->float promotion (REQ-23).
    log1p_promote,
    crate::log1p
);

// The remaining REQ-23 ops route through generic `T: Float` ufuncs that do
// not need `CrMath`; they still satisfy the macro's bound because f32/f64
// both impl CrMath. Using one macro keeps every entry point uniform.
unary_promote_fn!(
    /// `sqrt` with int/bool->float promotion (REQ-23). `np.sqrt(int64
    /// [1,2,4]) -> float64 [1.0, √2, 2.0]`; `np.sqrt(bool [T,F,T]) ->
    /// float16 [1.0, 0.0, 1.0]`.
    sqrt_promote,
    crate::sqrt
);
unary_promote_fn!(
    /// `cbrt` with int/bool->float promotion (REQ-23).
    cbrt_promote,
    crate::cbrt
);
unary_promote_fn!(
    /// `fabs` with int/bool->float promotion (REQ-23).
    fabs_promote,
    crate::fabs
);
unary_promote_fn!(
    /// `rint` with int/bool->float promotion (REQ-23). Unlike
    /// floor/ceil/trunc/round (REQ-24, int identity), `rint` registers NO
    /// `TD(bints)` (`generate_umath.py:1021`), so int input promotes to
    /// float: `np.rint(int64 [1,2,4]) -> float64 [1.0,2.0,4.0]`.
    rint_promote,
    crate::rint
);
unary_promote_fn!(
    /// `sin` with int/bool->float promotion (REQ-23).
    sin_promote,
    crate::sin
);
unary_promote_fn!(
    /// `cos` with int/bool->float promotion (REQ-23).
    cos_promote,
    crate::cos
);
unary_promote_fn!(
    /// `tan` with int/bool->float promotion (REQ-23).
    tan_promote,
    crate::tan
);
unary_promote_fn!(
    /// `arcsin` with int/bool->float promotion (REQ-23).
    arcsin_promote,
    crate::arcsin
);
unary_promote_fn!(
    /// `arccos` with int/bool->float promotion (REQ-23).
    arccos_promote,
    crate::arccos
);
unary_promote_fn!(
    /// `arctan` with int/bool->float promotion (REQ-23).
    arctan_promote,
    crate::arctan
);
unary_promote_fn!(
    /// `sinh` with int/bool->float promotion (REQ-23).
    sinh_promote,
    crate::sinh
);
unary_promote_fn!(
    /// `cosh` with int/bool->float promotion (REQ-23).
    cosh_promote,
    crate::cosh
);
unary_promote_fn!(
    /// `tanh` with int/bool->float promotion (REQ-23).
    tanh_promote,
    crate::tanh
);
unary_promote_fn!(
    /// `arcsinh` with int/bool->float promotion (REQ-23).
    arcsinh_promote,
    crate::arcsinh
);
unary_promote_fn!(
    /// `arccosh` with int/bool->float promotion (REQ-23).
    arccosh_promote,
    crate::arccosh
);
unary_promote_fn!(
    /// `arctanh` with int/bool->float promotion (REQ-23).
    arctanh_promote,
    crate::arctanh
);

// ---------------------------------------------------------------------------
// Binary float-ufunc integer/bool input promotion (REQ-25)
//
// NumPy accepts integer and bool input across the binary float-ufunc family
// hypot/arctan2/logaddexp/logaddexp2/copysign/nextafter. These ops register
// only float loops with NO `TD(bints)` (`generate_umath.py`: `hypot`
// :1057-1063, `arctan2` :1030-1037, `logaddexp` :710-715, `logaddexp2`
// :716-721, `copysign` :1107-1113, `nextafter` :1114-1119), so an integer/bool
// input pair safe-casts UP to the smallest float loop that can hold its kind —
// the SAME smallest-safe-cast-float rule as the unary REQ-23 family:
// bool/int8/uint8 -> float16, int16/uint16 -> float32,
// int32/int64/uint32/uint64 -> float64.
//
// This reuses the existing `PromoteFloat` trait (REQ-23) unchanged: both
// operands share the same element type `T`, so they resolve to the same
// `T::Compute`/`T::Out`. For SAME-WIDTH integer pairs (the core contract) this
// is exactly numpy's result kind. For MIXED-WIDTH integer pairs (e.g. int16
// paired with int32), numpy first promotes the two integers together via NEP-50
// `result_type` (-> int32) and then to its float loop (-> float64); confirmed
// live: `np.hypot(np.int16([3]), np.int32([4])).dtype == float64`. The binary
// float ufuncs ferray exposes (`crate::hypot` etc.) require same-type operands
// `&Array<T, D>, &Array<T, D>`, so this same-`T` entry-point family covers the
// same-width contract; a caller with genuinely mixed integer widths casts the
// narrower operand up first (the NEP-50 integer promotion already lives in
// ferray-core's `Promoted` table) and then calls the matching `*_promote`.
//
// Like the unary path, the f32/f64-output kinds run the EXISTING generic
// `T: Float` binary ufunc unchanged (so f32/f64 callers are byte-identical and
// this module never touches their code path), and the f16-output kinds compute
// in f32 then narrow to `half::f16`.
// ---------------------------------------------------------------------------

/// Cast a same-type integer/bool input pair to its compute float, run a
/// generic binary float kernel `op` (the existing `T: Float` ufunc
/// monomorphised at the compute type), and narrow the result to the promoted
/// output type.
///
/// Mirrors [`unary_promote_float`] for two operands: both inputs share the
/// element type `T`, so they resolve to the same `T::Compute`/`T::Out`. The
/// `op` closure receives two `&Array<Compute, D>` and returns
/// `Array<Compute, D>`; for the f32/f64 kinds `narrow` is the identity so the
/// existing kernel's bytes flow straight through.
///
/// # Errors
/// Propagates [`FerrayError::ShapeMismatch`] from the underlying binary kernel
/// (same-shape inputs are required, no broadcasting on the promoted path).
#[inline]
fn binary_promote_float<T, D>(
    a: &Array<T, D>,
    b: &Array<T, D>,
    op: impl Fn(&Array<T::Compute, D>, &Array<T::Compute, D>) -> FerrayResult<Array<T::Compute, D>>,
) -> FerrayResult<Array<<T as PromoteFloat>::Out, D>>
where
    T: PromoteFloat,
    D: Dimension,
{
    let a_compute = cast_to_compute(a)?;
    let b_compute = cast_to_compute(b)?;
    let result = op(&a_compute, &b_compute)?;
    // Narrow to the output kind (identity for f32/f64, f32->f16 for f16).
    let out: Array<<T as PromoteFloat>::Out, D> = if let Some(slice) = result.as_slice() {
        let data: Vec<<T as PromoteFloat>::Out> = slice.iter().map(|&c| T::narrow(c)).collect();
        Array::from_vec(result.dim().clone(), data)?
    } else {
        let data: Vec<<T as PromoteFloat>::Out> = result.iter().map(|&c| T::narrow(c)).collect();
        Array::from_vec(result.dim().clone(), data)?
    };
    Ok(out)
}

/// Widen one integer/bool input array to its `PromoteFloat::Compute` float,
/// reusing the contiguous fast path. Shared by the binary promote entry points.
#[inline]
fn cast_to_compute<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T::Compute, D>>
where
    T: PromoteFloat,
    D: Dimension,
{
    if let Some(slice) = input.as_slice() {
        let data: Vec<T::Compute> = slice.iter().map(|&x| x.to_compute()).collect();
        Array::from_vec(input.dim().clone(), data)
    } else {
        let data: Vec<T::Compute> = input.iter().map(|&x| x.to_compute()).collect();
        Array::from_vec(input.dim().clone(), data)
    }
}

/// Generate a `pub fn <name>_promote` int/bool-accepting binary entry point
/// that promotes per REQ-25 and routes through the existing float binary ufunc
/// `$op_path`. `$op_path` must be the crate's generic `T: Float` (or
/// `T: Float + CrMath`) binary ufunc — it is called monomorphised at the
/// resolved compute float, so f32/f64 results are byte-identical to a direct
/// float call. The `T::Compute: CrMath` bound is always satisfiable (f32/f64
/// both impl `CrMath`), keeping every entry point uniform even for the ops
/// (`copysign`/`nextafter`) whose kernel does not itself need `CrMath`.
macro_rules! binary_promote_fn {
    (
        $(#[$attr:meta])*
        $name:ident,
        $op_path:path
    ) => {
        $(#[$attr])*
        pub fn $name<T, D>(
            a: &Array<T, D>,
            b: &Array<T, D>,
        ) -> FerrayResult<Array<<T as PromoteFloat>::Out, D>>
        where
            T: PromoteFloat,
            T::Compute: crate::cr_math::CrMath,
            D: Dimension,
        {
            binary_promote_float(a, b, |x, y| $op_path(x, y))
        }
    };
}

binary_promote_fn!(
    /// `hypot` with NumPy int/bool->float promotion (REQ-25).
    /// `np.hypot(int64 [3,4], int64 [4,3]) -> float64 [5.0, 5.0]`;
    /// `np.hypot(int16 [3], int16 [4]) -> float32`;
    /// `np.hypot(uint8 ...) -> float16`.
    hypot_promote,
    crate::hypot
);
binary_promote_fn!(
    /// `arctan2` with int/bool->float promotion (REQ-25).
    /// `np.arctan2(int64 [1,1], int64 [1,1]) -> float64`.
    arctan2_promote,
    crate::arctan2
);
binary_promote_fn!(
    /// `logaddexp` with int/bool->float promotion (REQ-25).
    /// `np.logaddexp(int64 [0,0], int64 [0,0]) -> float64`.
    logaddexp_promote,
    crate::logaddexp
);
binary_promote_fn!(
    /// `logaddexp2` with int/bool->float promotion (REQ-25).
    logaddexp2_promote,
    crate::logaddexp2
);
binary_promote_fn!(
    /// `copysign` with int/bool->float promotion (REQ-25).
    /// `np.copysign(int64 [1,2], int64 [-1,1]) -> float64 [-1.0, 2.0]`.
    copysign_promote,
    crate::copysign
);
binary_promote_fn!(
    /// `nextafter` with int/bool->float promotion (REQ-25).
    /// `np.nextafter(int64 [1], int64 [2]) -> float64`.
    nextafter_promote,
    crate::nextafter
);

/// Cast every element of `a` from `A` to the target type `Out`, producing
/// a fresh array. This is the tiny bridge that lets a mixed-type op
/// route both operands through a same-shape same-type kernel.
#[inline]
fn cast_array<A, Out, D>(a: &Array<A, D>) -> FerrayResult<Array<Out, D>>
where
    A: Element + Copy + PromoteTo<Out>,
    Out: Element + Copy,
    D: Dimension,
{
    if let Some(slice) = a.as_slice() {
        let data: Vec<Out> = slice.iter().map(|&x| x.promote()).collect();
        Array::from_vec(a.dim().clone(), data)
    } else {
        let data: Vec<Out> = a.iter().map(|&x| x.promote()).collect();
        Array::from_vec(a.dim().clone(), data)
    }
}

// ---------------------------------------------------------------------------
// Add / Subtract / Multiply / Divide
// ---------------------------------------------------------------------------

/// Elementwise addition with NumPy-style type promotion.
///
/// `add_promoted(Array<i32>, Array<f64>)` promotes the i32 to f64
/// (per the `Promoted` trait) and returns `Array<f64>`. Integer-only
/// mixed widths promote too (`int8 + int64 -> int64`) and wrap on overflow.
///
/// Both inputs must have the same shape. For broadcasting, cast
/// explicitly and use the existing [`crate::add`].
///
/// # Errors
/// Returns [`FerrayError::ShapeMismatch`] if shapes differ.
pub fn add_promoted<A, B, D>(
    a: &Array<A, D>,
    b: &Array<B, D>,
) -> FerrayResult<Array<<A as Promoted<B>>::Output, D>>
where
    A: Element + Copy + Promoted<B> + PromoteTo<<A as Promoted<B>>::Output>,
    B: Element + Copy + PromoteTo<<A as Promoted<B>>::Output>,
    <A as Promoted<B>>::Output: Element + Copy + WrappingArith,
    D: Dimension,
{
    if a.shape() != b.shape() {
        return Err(FerrayError::shape_mismatch(format!(
            "add_promoted: shapes {:?} and {:?} differ",
            a.shape(),
            b.shape()
        )));
    }
    let a_cast = cast_array::<A, <A as Promoted<B>>::Output, D>(a)?;
    let b_cast = cast_array::<B, <A as Promoted<B>>::Output, D>(b)?;
    binary_elementwise_op(&a_cast, &b_cast, WrappingArith::wadd)
}

/// Elementwise subtraction with NumPy-style type promotion.
pub fn subtract_promoted<A, B, D>(
    a: &Array<A, D>,
    b: &Array<B, D>,
) -> FerrayResult<Array<<A as Promoted<B>>::Output, D>>
where
    A: Element + Copy + Promoted<B> + PromoteTo<<A as Promoted<B>>::Output>,
    B: Element + Copy + PromoteTo<<A as Promoted<B>>::Output>,
    <A as Promoted<B>>::Output: Element + Copy + WrappingArith,
    D: Dimension,
{
    if a.shape() != b.shape() {
        return Err(FerrayError::shape_mismatch(format!(
            "subtract_promoted: shapes {:?} and {:?} differ",
            a.shape(),
            b.shape()
        )));
    }
    let a_cast = cast_array::<A, <A as Promoted<B>>::Output, D>(a)?;
    let b_cast = cast_array::<B, <A as Promoted<B>>::Output, D>(b)?;
    binary_elementwise_op(&a_cast, &b_cast, WrappingArith::wsub)
}

/// Elementwise multiplication with NumPy-style type promotion.
pub fn multiply_promoted<A, B, D>(
    a: &Array<A, D>,
    b: &Array<B, D>,
) -> FerrayResult<Array<<A as Promoted<B>>::Output, D>>
where
    A: Element + Copy + Promoted<B> + PromoteTo<<A as Promoted<B>>::Output>,
    B: Element + Copy + PromoteTo<<A as Promoted<B>>::Output>,
    <A as Promoted<B>>::Output: Element + Copy + WrappingArith,
    D: Dimension,
{
    if a.shape() != b.shape() {
        return Err(FerrayError::shape_mismatch(format!(
            "multiply_promoted: shapes {:?} and {:?} differ",
            a.shape(),
            b.shape()
        )));
    }
    let a_cast = cast_array::<A, <A as Promoted<B>>::Output, D>(a)?;
    let b_cast = cast_array::<B, <A as Promoted<B>>::Output, D>(b)?;
    binary_elementwise_op(&a_cast, &b_cast, WrappingArith::wmul)
}

/// Elementwise division with NumPy-style type promotion.
///
/// This path requires the promoted output to be a float (`int + float`,
/// `float + float`). For `int + int`, NumPy's `np.divide` is *true*
/// division returning `float64` — that is the same-type [`crate::divide`]
/// via the [`crate::TrueDivide`] trait, not `Promoted::Output` (which would
/// be an integer). Use [`crate::divide`] for integer true-division.
pub fn divide_promoted<A, B, D>(
    a: &Array<A, D>,
    b: &Array<B, D>,
) -> FerrayResult<Array<<A as Promoted<B>>::Output, D>>
where
    A: Element + Copy + Promoted<B> + PromoteTo<<A as Promoted<B>>::Output>,
    B: Element + Copy + PromoteTo<<A as Promoted<B>>::Output>,
    <A as Promoted<B>>::Output: Element + Copy + Float,
    D: Dimension,
{
    if a.shape() != b.shape() {
        return Err(FerrayError::shape_mismatch(format!(
            "divide_promoted: shapes {:?} and {:?} differ",
            a.shape(),
            b.shape()
        )));
    }
    let a_cast = cast_array::<A, <A as Promoted<B>>::Output, D>(a)?;
    let b_cast = cast_array::<B, <A as Promoted<B>>::Output, D>(b)?;
    binary_elementwise_op(&a_cast, &b_cast, |x, y| x / y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::{Ix1, Ix2};

    #[test]
    fn add_i8_i64_promotes_to_i64() {
        // np.add(int8, int64) -> int64 (NEP-50 mixed-width integer promotion).
        // Live numpy: np.add(np.array([1,2],np.int8), np.array([10,20],np.int64))
        //   -> array([11, 22]) dtype=int64
        let a = Array::<i8, Ix1>::from_vec(Ix1::new([2]), vec![1i8, 2]).unwrap();
        let b = Array::<i64, Ix1>::from_vec(Ix1::new([2]), vec![10i64, 20]).unwrap();
        let c = add_promoted(&a, &b).unwrap();
        let slice: &[i64] = c.as_slice().unwrap();
        assert_eq!(slice, &[11, 22]);
    }

    #[test]
    fn multiply_i32_i64_promotes_to_i64() {
        // np.multiply(int32, int64) -> int64.
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([2]), vec![3i32, 4]).unwrap();
        let b = Array::<i64, Ix1>::from_vec(Ix1::new([2]), vec![10i64, 20]).unwrap();
        let c = multiply_promoted(&a, &b).unwrap();
        let slice: &[i64] = c.as_slice().unwrap();
        assert_eq!(slice, &[30, 80]);
    }

    #[test]
    fn add_i32_f64_promotes_to_f64() {
        // np.add(int32_arr, float64_arr) → float64
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![1i32, 2, 3]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![0.5, 1.5, 2.5]).unwrap();
        let c = add_promoted(&a, &b).unwrap();
        // Type-check: the return type is Array<f64, Ix1>
        let slice: &[f64] = c.as_slice().unwrap();
        assert_eq!(slice, &[1.5, 3.5, 5.5]);
    }

    #[test]
    fn add_f32_f64_promotes_to_f64() {
        let a = Array::<f32, Ix1>::from_vec(Ix1::new([3]), vec![1.0f32, 2.0, 3.0]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![0.5, 0.5, 0.5]).unwrap();
        let c = add_promoted(&a, &b).unwrap();
        let slice: &[f64] = c.as_slice().unwrap();
        assert_eq!(slice, &[1.5, 2.5, 3.5]);
    }

    #[test]
    fn subtract_i16_f32_promotes_to_f32() {
        let a = Array::<i16, Ix1>::from_vec(Ix1::new([3]), vec![10i16, 20, 30]).unwrap();
        let b = Array::<f32, Ix1>::from_vec(Ix1::new([3]), vec![1.5f32, 2.5, 3.5]).unwrap();
        let c = subtract_promoted(&a, &b).unwrap();
        let slice: &[f32] = c.as_slice().unwrap();
        assert_eq!(slice, &[8.5, 17.5, 26.5]);
    }

    #[test]
    fn multiply_u8_f64_promotes_to_f64() {
        let a = Array::<u8, Ix1>::from_vec(Ix1::new([3]), vec![2u8, 3, 4]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![0.5, 0.5, 0.5]).unwrap();
        let c = multiply_promoted(&a, &b).unwrap();
        assert_eq!(c.as_slice().unwrap(), &[1.0, 1.5, 2.0]);
    }

    #[test]
    fn divide_f32_f64_promotes_to_f64() {
        let a = Array::<f32, Ix1>::from_vec(Ix1::new([3]), vec![10.0f32, 20.0, 30.0]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![2.0, 4.0, 5.0]).unwrap();
        let c = divide_promoted(&a, &b).unwrap();
        let slice: &[f64] = c.as_slice().unwrap();
        assert_eq!(slice, &[5.0, 5.0, 6.0]);
    }

    #[test]
    fn same_type_path_is_identity() {
        // f64 + f64 should still work (Promoted<f64>::Output = f64).
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![4.0, 5.0, 6.0]).unwrap();
        let c = add_promoted(&a, &b).unwrap();
        assert_eq!(c.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn promoted_2d_shape_preserved() {
        let a = Array::<i32, Ix2>::from_vec(Ix2::new([2, 3]), vec![1i32, 2, 3, 4, 5, 6]).unwrap();
        let b = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
            .unwrap();
        let c = add_promoted(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.as_slice().unwrap(), &[1.5, 2.5, 3.5, 4.5, 5.5, 6.5]);
    }

    #[test]
    fn shape_mismatch_errors() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![1i32, 2, 3]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(add_promoted(&a, &b).is_err());
    }
}
