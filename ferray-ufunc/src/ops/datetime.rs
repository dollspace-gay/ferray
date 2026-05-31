// ferray-ufunc: datetime64 / timedelta64 predicates and arithmetic
//
// Provides:
//   - isnat() for the time element types — the "is Not a Time"
//     predicate that returns true where a value equals the NaT sentinel
//     (i64::MIN).
//   - Element-wise arithmetic kernels:
//       datetime - datetime → timedelta
//       datetime + timedelta → datetime
//       datetime - timedelta → datetime
//       timedelta + timedelta → timedelta
//       timedelta - timedelta → timedelta
//
// Same-unit operands required (NumPy promotes to the finer unit; we
// surface a ShapeMismatch with a clear message and let callers cast
// explicitly — that matches ferray's "no implicit precision changes"
// stance from the rest of the workspace).
//
// NaT propagation: any operand that is NaT yields NaT in the output
// (matching NumPy).

use ferray_core::Array;
use ferray_core::dimension::Dimension;
use ferray_core::dtype::{DateTime64, NAT, TimeUnit, Timedelta64};
use ferray_core::error::{FerrayError, FerrayResult};

/// Element-wise: true where the input is the NaT ("Not a Time") sentinel.
///
/// Equivalent to `numpy.isnat` on a `datetime64` array. NumPy raises
/// `TypeError` for any other dtype; in ferray that constraint is
/// statically enforced — call sites that try to pass a non-time array
/// fail to type-check.
///
/// # Errors
/// Returns an error only if internal array construction fails.
pub fn isnat_datetime<D: Dimension>(input: &Array<DateTime64, D>) -> FerrayResult<Array<bool, D>> {
    let data: Vec<bool> = input.iter().map(|v| v.is_nat()).collect();
    Array::from_vec(input.dim().clone(), data)
}

/// Element-wise: true where the input is the NaT sentinel.
///
/// Equivalent to `numpy.isnat` on a `timedelta64` array.
///
/// # Errors
/// Returns an error only if internal array construction fails.
pub fn isnat_timedelta<D: Dimension>(
    input: &Array<Timedelta64, D>,
) -> FerrayResult<Array<bool, D>> {
    let data: Vec<bool> = input.iter().map(|v| v.is_nat()).collect();
    Array::from_vec(input.dim().clone(), data)
}

// ===========================================================================
// Arithmetic kernels
// ===========================================================================

/// Helper: combine two i64 ticks with NaT propagation. If either side is
/// NaT, the result is NaT; otherwise the closure runs.
#[inline]
fn nat_propagate(a: i64, b: i64, op: impl FnOnce(i64, i64) -> i64) -> i64 {
    if a == NAT || b == NAT { NAT } else { op(a, b) }
}

fn check_same_shape<A, B, DA, DB>(
    a: &Array<A, DA>,
    b: &Array<B, DB>,
    name: &str,
) -> FerrayResult<()>
where
    A: ferray_core::Element,
    B: ferray_core::Element,
    DA: Dimension,
    DB: Dimension,
{
    if a.shape() != b.shape() {
        return Err(FerrayError::shape_mismatch(format!(
            "{name}: shapes {:?} and {:?} differ",
            a.shape(),
            b.shape()
        )));
    }
    Ok(())
}

/// Element-wise `datetime - datetime → timedelta` (same unit assumed).
///
/// NaT propagates: if either side is NaT at position `i`, the output is
/// NaT at that position. Mirrors NumPy's `datetime64 - datetime64`
/// semantics.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if the shapes differ.
pub fn sub_datetime<D: Dimension>(
    a: &Array<DateTime64, D>,
    b: &Array<DateTime64, D>,
) -> FerrayResult<Array<Timedelta64, D>> {
    check_same_shape(a, b, "sub_datetime")?;
    let data: Vec<Timedelta64> = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| Timedelta64(nat_propagate(x.0, y.0, |p, q| p.wrapping_sub(q))))
        .collect();
    Array::from_vec(a.dim().clone(), data)
}

/// Element-wise `datetime + timedelta → datetime`.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if the shapes differ.
pub fn add_datetime_timedelta<D: Dimension>(
    a: &Array<DateTime64, D>,
    b: &Array<Timedelta64, D>,
) -> FerrayResult<Array<DateTime64, D>> {
    check_same_shape(a, b, "add_datetime_timedelta")?;
    let data: Vec<DateTime64> = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| DateTime64(nat_propagate(x.0, y.0, |p, q| p.wrapping_add(q))))
        .collect();
    Array::from_vec(a.dim().clone(), data)
}

/// Element-wise `datetime - timedelta → datetime`.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if the shapes differ.
pub fn sub_datetime_timedelta<D: Dimension>(
    a: &Array<DateTime64, D>,
    b: &Array<Timedelta64, D>,
) -> FerrayResult<Array<DateTime64, D>> {
    check_same_shape(a, b, "sub_datetime_timedelta")?;
    let data: Vec<DateTime64> = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| DateTime64(nat_propagate(x.0, y.0, |p, q| p.wrapping_sub(q))))
        .collect();
    Array::from_vec(a.dim().clone(), data)
}

/// Element-wise `timedelta + timedelta → timedelta`.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if the shapes differ.
pub fn add_timedelta<D: Dimension>(
    a: &Array<Timedelta64, D>,
    b: &Array<Timedelta64, D>,
) -> FerrayResult<Array<Timedelta64, D>> {
    check_same_shape(a, b, "add_timedelta")?;
    let data: Vec<Timedelta64> = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| Timedelta64(nat_propagate(x.0, y.0, |p, q| p.wrapping_add(q))))
        .collect();
    Array::from_vec(a.dim().clone(), data)
}

/// Element-wise `timedelta - timedelta → timedelta`.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if the shapes differ.
pub fn sub_timedelta<D: Dimension>(
    a: &Array<Timedelta64, D>,
    b: &Array<Timedelta64, D>,
) -> FerrayResult<Array<Timedelta64, D>> {
    check_same_shape(a, b, "sub_timedelta")?;
    let data: Vec<Timedelta64> = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| Timedelta64(nat_propagate(x.0, y.0, |p, q| p.wrapping_sub(q))))
        .collect();
    Array::from_vec(a.dim().clone(), data)
}

// ===========================================================================
// Implicit unit promotion
// ===========================================================================
//
// The promoted-* variants below take both operands together with their
// units, rescale both to the finer unit using `TimeUnit::finer` /
// `TimeUnit::scale_to`, and then call the same-unit kernel above. They
// return `(result, target_unit)` so callers (e.g. DynArray dispatch in
// ferray-numpy-interop) know which unit to tag the output array with.
//
// NaT propagation is preserved: rescaling treats `NAT` as "still NaT"
// regardless of the multiplicative factor.

#[inline]
fn rescale_datetime(arr: &[DateTime64], factor: i64) -> Vec<DateTime64> {
    arr.iter()
        .map(|v| {
            if v.is_nat() {
                DateTime64(NAT)
            } else {
                DateTime64(v.0.wrapping_mul(factor))
            }
        })
        .collect()
}

#[inline]
fn rescale_timedelta(arr: &[Timedelta64], factor: i64) -> Vec<Timedelta64> {
    arr.iter()
        .map(|v| {
            if v.is_nat() {
                Timedelta64(NAT)
            } else {
                Timedelta64(v.0.wrapping_mul(factor))
            }
        })
        .collect()
}

/// Element-wise `datetime[unit_a] - datetime[unit_b] → timedelta[finer]`.
///
/// Both operands are rescaled to `TimeUnit::finer(unit_a, unit_b)` before
/// the per-element subtraction. Returns `(result, target_unit)`.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if the shapes differ.
pub fn sub_datetime_promoted<D: Dimension>(
    a: &Array<DateTime64, D>,
    unit_a: TimeUnit,
    b: &Array<DateTime64, D>,
    unit_b: TimeUnit,
) -> FerrayResult<(Array<Timedelta64, D>, TimeUnit)> {
    check_same_shape(a, b, "sub_datetime_promoted")?;
    let target = unit_a.finer(unit_b);
    let scale_a = unit_a.scale_to(target).ok_or_else(|| {
        FerrayError::invalid_value(format!(
            "sub_datetime_promoted: cannot rescale {unit_a:?} → {target:?} (non-divisible)"
        ))
    })?;
    let scale_b = unit_b.scale_to(target).ok_or_else(|| {
        FerrayError::invalid_value(format!(
            "sub_datetime_promoted: cannot rescale {unit_b:?} → {target:?} (non-divisible)"
        ))
    })?;
    let a_data: Vec<DateTime64> = a.iter().copied().collect();
    let b_data: Vec<DateTime64> = b.iter().copied().collect();
    let a_rescaled = if scale_a == 1 {
        a_data
    } else {
        rescale_datetime(&a_data, scale_a)
    };
    let b_rescaled = if scale_b == 1 {
        b_data
    } else {
        rescale_datetime(&b_data, scale_b)
    };
    let data: Vec<Timedelta64> = a_rescaled
        .iter()
        .zip(b_rescaled.iter())
        .map(|(x, y)| Timedelta64(nat_propagate(x.0, y.0, |p, q| p.wrapping_sub(q))))
        .collect();
    let arr = Array::from_vec(a.dim().clone(), data)?;
    Ok((arr, target))
}

/// Element-wise `datetime[unit_a] + timedelta[unit_b] → datetime[finer]`.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if the shapes differ.
pub fn add_datetime_timedelta_promoted<D: Dimension>(
    a: &Array<DateTime64, D>,
    unit_a: TimeUnit,
    b: &Array<Timedelta64, D>,
    unit_b: TimeUnit,
) -> FerrayResult<(Array<DateTime64, D>, TimeUnit)> {
    check_same_shape(a, b, "add_datetime_timedelta_promoted")?;
    let target = unit_a.finer(unit_b);
    let scale_a = unit_a.scale_to(target).ok_or_else(|| {
        FerrayError::invalid_value(format!(
            "add_datetime_timedelta_promoted: cannot rescale {unit_a:?} → {target:?}"
        ))
    })?;
    let scale_b = unit_b.scale_to(target).ok_or_else(|| {
        FerrayError::invalid_value(format!(
            "add_datetime_timedelta_promoted: cannot rescale {unit_b:?} → {target:?}"
        ))
    })?;
    let a_data: Vec<DateTime64> = a.iter().copied().collect();
    let b_data: Vec<Timedelta64> = b.iter().copied().collect();
    let a_rescaled = if scale_a == 1 {
        a_data
    } else {
        rescale_datetime(&a_data, scale_a)
    };
    let b_rescaled = if scale_b == 1 {
        b_data
    } else {
        rescale_timedelta(&b_data, scale_b)
    };
    let data: Vec<DateTime64> = a_rescaled
        .iter()
        .zip(b_rescaled.iter())
        .map(|(x, y)| DateTime64(nat_propagate(x.0, y.0, |p, q| p.wrapping_add(q))))
        .collect();
    let arr = Array::from_vec(a.dim().clone(), data)?;
    Ok((arr, target))
}

/// Element-wise `timedelta[unit_a] + timedelta[unit_b] → timedelta[finer]`.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if the shapes differ.
pub fn add_timedelta_promoted<D: Dimension>(
    a: &Array<Timedelta64, D>,
    unit_a: TimeUnit,
    b: &Array<Timedelta64, D>,
    unit_b: TimeUnit,
) -> FerrayResult<(Array<Timedelta64, D>, TimeUnit)> {
    check_same_shape(a, b, "add_timedelta_promoted")?;
    let target = unit_a.finer(unit_b);
    let scale_a = unit_a.scale_to(target).ok_or_else(|| {
        FerrayError::invalid_value(format!(
            "add_timedelta_promoted: cannot rescale {unit_a:?} → {target:?}"
        ))
    })?;
    let scale_b = unit_b.scale_to(target).ok_or_else(|| {
        FerrayError::invalid_value(format!(
            "add_timedelta_promoted: cannot rescale {unit_b:?} → {target:?}"
        ))
    })?;
    let a_data: Vec<Timedelta64> = a.iter().copied().collect();
    let b_data: Vec<Timedelta64> = b.iter().copied().collect();
    let a_rescaled = if scale_a == 1 {
        a_data
    } else {
        rescale_timedelta(&a_data, scale_a)
    };
    let b_rescaled = if scale_b == 1 {
        b_data
    } else {
        rescale_timedelta(&b_data, scale_b)
    };
    let data: Vec<Timedelta64> = a_rescaled
        .iter()
        .zip(b_rescaled.iter())
        .map(|(x, y)| Timedelta64(nat_propagate(x.0, y.0, |p, q| p.wrapping_add(q))))
        .collect();
    let arr = Array::from_vec(a.dim().clone(), data)?;
    Ok((arr, target))
}

// ===========================================================================
// timedelta numeric arithmetic (REQ-2, #942)
// ===========================================================================
//
// numpy registers these timedelta numeric loops (generate_umath.py:386-1046):
//   multiply:     'mq'->'m', 'qm'->'m', 'md'->'m', 'dm'->'m'
//   divide:       'mq'->'m', 'md'->'m', 'mm'->'d'
//   floor_divide: 'mq'->'m', 'md'->'m', 'mm'->'q'
//   remainder:    'mm'->'m'   (no 'mq'/'md' — td % int RAISES)
// The scalar (int/float) kernels below cover the `mq`/`qm`/`md`/`dm` loops
// (`mul_timedelta_scalar_*` / `div_timedelta_scalar_*`); the timedelta ÷
// timedelta kernels cover `mm` (`truediv_timedelta` -> f64,
// `floordiv_timedelta` -> i64, `mod_timedelta` -> timedelta).
//
// Semantics mirror the numpy C loops in
// numpy/_core/src/umath/loops.c.src EXACTLY (verified live, numpy 2.4.5):
//   * NaT (`i64::MIN`) propagates: any NaT operand -> NaT result (or `nan`
//     for the float-ratio loop, or `0` for the floor-divide loop — which is
//     how numpy's `TIMEDELTA_mm_q_floor_divide` handles NaT, loops.c.src:1110).
//   * `td * float` / `td / float` cast the double result back to int64 via a
//     C cast = TRUNCATION TOWARD ZERO (`TIMEDELTA_md_m_multiply`,
//     loops.c.src:944 `(npy_timedelta)result`), NOT round-half-even; a
//     non-finite double (inf/nan from `*0`-style overflow or `/0.0`) -> NaT.
//   * `td / int` is integer division (`in1 / in2`, trunc toward zero); a zero
//     divisor -> NaT (`TIMEDELTA_mq_m_divide`, loops.c.src:1014).
//   * `td // td` is FLOOR division (round toward -inf); NaT either operand or a
//     zero divisor -> `0` (`TIMEDELTA_mm_q_floor_divide`, loops.c.src:1104).
//   * `td % td` is Python floor-modulo (the sign of the result follows the
//     DIVISOR); NaT either operand or a zero divisor -> NaT
//     (`TIMEDELTA_mm_m_remainder`, loops.c.src:1075 "handle mixed case the
//     way Python does").
// The `mm` ratio/floor/mod kernels promote both operands to the finer unit
// first (`TimeUnit::finer` / `scale_to`), matching numpy's common-unit cast.

/// Rescale a timedelta tick to a finer `target` unit, returning the rescaled
/// `Vec<i64>` ticks (NaT preserved). Used by the `mm` kernels below.
#[inline]
fn rescale_td_ticks(arr: &[Timedelta64], factor: i64) -> Vec<i64> {
    arr.iter()
        .map(|v| {
            if v.is_nat() {
                NAT
            } else {
                v.0.wrapping_mul(factor)
            }
        })
        .collect()
}

/// Promote two timedelta operands to their common finer unit, returning the
/// rescaled int64 tick buffers `(a_ticks, b_ticks)` and the target unit.
///
/// Mirrors numpy's common-unit cast ahead of a `timedelta op timedelta` loop
/// (`PyUFunc_DivisionTypeResolver` / `PyUFunc_RemainderTypeResolver` promote
/// to the GCD unit; for the units ferray models that is `TimeUnit::finer`).
///
/// # Errors
/// `FerrayError::ShapeMismatch` if the shapes differ;
/// `FerrayError::InvalidValue` if a unit cannot be rescaled to the target.
fn promote_td_pair<D: Dimension>(
    a: &Array<Timedelta64, D>,
    unit_a: TimeUnit,
    b: &Array<Timedelta64, D>,
    unit_b: TimeUnit,
    name: &str,
) -> FerrayResult<(Vec<i64>, Vec<i64>, TimeUnit)> {
    check_same_shape(a, b, name)?;
    let target = unit_a.finer(unit_b);
    let scale_a = unit_a.scale_to(target).ok_or_else(|| {
        FerrayError::invalid_value(format!("{name}: cannot rescale {unit_a:?} → {target:?}"))
    })?;
    let scale_b = unit_b.scale_to(target).ok_or_else(|| {
        FerrayError::invalid_value(format!("{name}: cannot rescale {unit_b:?} → {target:?}"))
    })?;
    let a_data: Vec<Timedelta64> = a.iter().copied().collect();
    let b_data: Vec<Timedelta64> = b.iter().copied().collect();
    let a_ticks = if scale_a == 1 {
        a_data.iter().map(|v| v.0).collect()
    } else {
        rescale_td_ticks(&a_data, scale_a)
    };
    let b_ticks = if scale_b == 1 {
        b_data.iter().map(|v| v.0).collect()
    } else {
        rescale_td_ticks(&b_data, scale_b)
    };
    Ok((a_ticks, b_ticks, target))
}

/// Element-wise `timedelta * int → timedelta` (same unit; the integer is a
/// pure scalar multiplier, no unit). NaT propagates.
///
/// Mirrors `TIMEDELTA_mq_m_multiply` / `TIMEDELTA_qm_m_multiply`
/// (loops.c.src:902/918): `in1 * in2`, NaT -> NaT. The result keeps the
/// timedelta's own unit.
///
/// # Errors
/// Returns an error only if internal array construction fails.
pub fn mul_timedelta_scalar_i64<D: Dimension>(
    a: &Array<Timedelta64, D>,
    k: i64,
) -> FerrayResult<Array<Timedelta64, D>> {
    let data: Vec<Timedelta64> = a
        .iter()
        .map(|v| {
            if v.is_nat() {
                Timedelta64(NAT)
            } else {
                Timedelta64(v.0.wrapping_mul(k))
            }
        })
        .collect();
    Array::from_vec(a.dim().clone(), data)
}

/// Element-wise `timedelta * float → timedelta` (same unit). The double
/// product is cast back to int64 via TRUNCATION TOWARD ZERO; a non-finite
/// product yields NaT. NaT propagates.
///
/// Mirrors `TIMEDELTA_md_m_multiply` / `TIMEDELTA_dm_m_multiply`
/// (loops.c.src:933/954): `result = in1 * in2; isfinite ? (i64)result : NaT`.
///
/// # Errors
/// Returns an error only if internal array construction fails.
pub fn mul_timedelta_scalar_f64<D: Dimension>(
    a: &Array<Timedelta64, D>,
    k: f64,
) -> FerrayResult<Array<Timedelta64, D>> {
    let data: Vec<Timedelta64> = a
        .iter()
        .map(|v| {
            if v.is_nat() {
                Timedelta64(NAT)
            } else {
                let r = v.0 as f64 * k;
                if r.is_finite() {
                    Timedelta64(r as i64)
                } else {
                    Timedelta64(NAT)
                }
            }
        })
        .collect();
    Array::from_vec(a.dim().clone(), data)
}

/// Element-wise `timedelta / int → timedelta` (same unit). Integer division,
/// truncating toward zero; a zero divisor yields NaT. NaT propagates.
///
/// Mirrors `TIMEDELTA_mq_m_divide` (loops.c.src:976): `in1 / in2`, with
/// `in1 == NaT || in2 == 0 -> NaT`.
///
/// # Errors
/// Returns an error only if internal array construction fails.
pub fn div_timedelta_scalar_i64<D: Dimension>(
    a: &Array<Timedelta64, D>,
    k: i64,
) -> FerrayResult<Array<Timedelta64, D>> {
    let data: Vec<Timedelta64> = a
        .iter()
        .map(|v| {
            if v.is_nat() || k == 0 {
                Timedelta64(NAT)
            } else {
                Timedelta64(v.0.wrapping_div(k))
            }
        })
        .collect();
    Array::from_vec(a.dim().clone(), data)
}

/// Element-wise `timedelta / float → timedelta` (same unit). The double
/// quotient is cast back to int64 via TRUNCATION TOWARD ZERO; a non-finite
/// quotient (e.g. divide-by-zero) yields NaT. NaT propagates.
///
/// Mirrors `TIMEDELTA_md_m_divide` (loops.c.src:1025): `result = in1 / in2;
/// isfinite ? (i64)result : NaT`.
///
/// # Errors
/// Returns an error only if internal array construction fails.
pub fn div_timedelta_scalar_f64<D: Dimension>(
    a: &Array<Timedelta64, D>,
    k: f64,
) -> FerrayResult<Array<Timedelta64, D>> {
    let data: Vec<Timedelta64> = a
        .iter()
        .map(|v| {
            if v.is_nat() {
                Timedelta64(NAT)
            } else {
                let r = v.0 as f64 / k;
                if r.is_finite() {
                    Timedelta64(r as i64)
                } else {
                    Timedelta64(NAT)
                }
            }
        })
        .collect();
    Array::from_vec(a.dim().clone(), data)
}

/// Element-wise `timedelta[ua] / timedelta[ub] → float64` (the ratio). Both
/// operands are first promoted to the finer common unit; the ratio is then
/// `a as f64 / b as f64`. NaT in either operand yields `nan`.
///
/// Mirrors `TIMEDELTA_mm_d_divide` (loops.c.src:1046): `NaT either -> nan`,
/// else `(double)in1 / (double)in2` (a zero divisor therefore yields
/// `±inf`/`nan`, exactly as the IEEE division does — verified live:
/// `td/td0 -> inf`).
///
/// # Errors
/// `FerrayError::ShapeMismatch` if the shapes differ; `InvalidValue` if a
/// unit cannot be rescaled to the common finer unit.
pub fn truediv_timedelta<D: Dimension>(
    a: &Array<Timedelta64, D>,
    unit_a: TimeUnit,
    b: &Array<Timedelta64, D>,
    unit_b: TimeUnit,
) -> FerrayResult<Array<f64, D>> {
    let (at, bt, _target) = promote_td_pair(a, unit_a, b, unit_b, "truediv_timedelta")?;
    let data: Vec<f64> = at
        .iter()
        .zip(bt.iter())
        .map(|(&x, &y)| {
            if x == NAT || y == NAT {
                f64::NAN
            } else {
                x as f64 / y as f64
            }
        })
        .collect();
    Array::from_vec(a.dim().clone(), data)
}

/// Element-wise `timedelta[ua] // timedelta[ub] → int64` (FLOOR division,
/// round toward −∞). Both operands are promoted to the finer common unit.
/// NaT in either operand, or a zero divisor, yields `0`.
///
/// Mirrors `TIMEDELTA_mm_q_floor_divide` (loops.c.src:1089): NaT either
/// operand or a zero divisor -> `0`; otherwise the quotient is corrected
/// downward for negative results so it floors (loops.c.src:1128).
/// `i64::div_euclid` does NOT match (it rounds remainder toward +∞); the
/// floor is computed directly.
///
/// # Errors
/// `FerrayError::ShapeMismatch` if the shapes differ; `InvalidValue` if a
/// unit cannot be rescaled to the common finer unit.
pub fn floordiv_timedelta<D: Dimension>(
    a: &Array<Timedelta64, D>,
    unit_a: TimeUnit,
    b: &Array<Timedelta64, D>,
    unit_b: TimeUnit,
) -> FerrayResult<Array<i64, D>> {
    let (at, bt, _target) = promote_td_pair(a, unit_a, b, unit_b, "floordiv_timedelta")?;
    let data: Vec<i64> = at
        .iter()
        .zip(bt.iter())
        .map(|(&x, &y)| {
            if x == NAT || y == NAT || y == 0 {
                0
            } else {
                // Floor division (round toward -inf): trunc quotient, then
                // step down when the signs differ and the division is inexact
                // (mirrors numpy's loops.c.src:1128 correction).
                let q = x.wrapping_div(y);
                if (x % y != 0) && ((x < 0) != (y < 0)) {
                    q - 1
                } else {
                    q
                }
            }
        })
        .collect();
    Array::from_vec(a.dim().clone(), data)
}

/// Element-wise `timedelta[ua] % timedelta[ub] → timedelta[finer]` (Python
/// floor-modulo: the sign of the result follows the DIVISOR). Both operands
/// are promoted to the finer common unit; the result carries that unit.
/// Returns `(result, target_unit)`. NaT in either operand, or a zero divisor,
/// yields NaT.
///
/// Mirrors `TIMEDELTA_mm_m_remainder` (loops.c.src:1061): `rem = in1 % in2`,
/// then if the operand signs differ and `rem != 0`, `rem + in2` — i.e. Python
/// floor-modulo (`-7 % 2 == 1`, `7 % -2 == -1`, verified live).
///
/// # Errors
/// `FerrayError::ShapeMismatch` if the shapes differ; `InvalidValue` if a
/// unit cannot be rescaled to the common finer unit.
pub fn mod_timedelta<D: Dimension>(
    a: &Array<Timedelta64, D>,
    unit_a: TimeUnit,
    b: &Array<Timedelta64, D>,
    unit_b: TimeUnit,
) -> FerrayResult<(Array<Timedelta64, D>, TimeUnit)> {
    let (at, bt, target) = promote_td_pair(a, unit_a, b, unit_b, "mod_timedelta")?;
    let data: Vec<Timedelta64> = at
        .iter()
        .zip(bt.iter())
        .map(|(&x, &y)| {
            if x == NAT || y == NAT || y == 0 {
                Timedelta64(NAT)
            } else {
                let rem = x.wrapping_rem(y);
                if rem == 0 || ((x > 0) == (y > 0)) {
                    Timedelta64(rem)
                } else {
                    Timedelta64(rem.wrapping_add(y))
                }
            }
        })
        .collect();
    let arr = Array::from_vec(a.dim().clone(), data)?;
    Ok((arr, target))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::Ix1;

    #[test]
    fn isnat_datetime_flags_min_int() {
        let data = vec![
            DateTime64(0),
            DateTime64::nat(),
            DateTime64(1_700_000_000_000_000_000),
            DateTime64::nat(),
        ];
        let arr = Array::<DateTime64, Ix1>::from_vec(Ix1::new([4]), data).unwrap();
        let r = isnat_datetime(&arr).unwrap();
        let v: Vec<bool> = r.iter().copied().collect();
        assert_eq!(v, vec![false, true, false, true]);
    }

    #[test]
    fn isnat_timedelta_flags_min_int() {
        let data = vec![Timedelta64(1_000), Timedelta64::nat(), Timedelta64(0)];
        let arr = Array::<Timedelta64, Ix1>::from_vec(Ix1::new([3]), data).unwrap();
        let r = isnat_timedelta(&arr).unwrap();
        let v: Vec<bool> = r.iter().copied().collect();
        assert_eq!(v, vec![false, true, false]);
    }

    #[test]
    fn isnat_datetime_all_finite() {
        let data = vec![DateTime64(1), DateTime64(2), DateTime64(3)];
        let arr = Array::<DateTime64, Ix1>::from_vec(Ix1::new([3]), data).unwrap();
        let r = isnat_datetime(&arr).unwrap();
        let v: Vec<bool> = r.iter().copied().collect();
        assert_eq!(v, vec![false, false, false]);
    }

    #[test]
    fn isnat_datetime_all_nat() {
        let data = vec![DateTime64::nat(); 5];
        let arr = Array::<DateTime64, Ix1>::from_vec(Ix1::new([5]), data).unwrap();
        let r = isnat_datetime(&arr).unwrap();
        let v: Vec<bool> = r.iter().copied().collect();
        assert_eq!(v, vec![true; 5]);
    }

    // ---- Arithmetic kernels ----

    #[test]
    fn sub_datetime_basic() {
        let a = Array::<DateTime64, Ix1>::from_vec(
            Ix1::new([3]),
            vec![DateTime64(100), DateTime64(200), DateTime64(50)],
        )
        .unwrap();
        let b = Array::<DateTime64, Ix1>::from_vec(
            Ix1::new([3]),
            vec![DateTime64(40), DateTime64(150), DateTime64(50)],
        )
        .unwrap();
        let r = sub_datetime(&a, &b).unwrap();
        let v: Vec<i64> = r.iter().map(|x| x.0).collect();
        assert_eq!(v, vec![60, 50, 0]);
    }

    #[test]
    fn sub_datetime_propagates_nat() {
        let a = Array::<DateTime64, Ix1>::from_vec(
            Ix1::new([3]),
            vec![DateTime64(100), DateTime64::nat(), DateTime64(50)],
        )
        .unwrap();
        let b = Array::<DateTime64, Ix1>::from_vec(
            Ix1::new([3]),
            vec![DateTime64(40), DateTime64(150), DateTime64::nat()],
        )
        .unwrap();
        let r = sub_datetime(&a, &b).unwrap();
        let v: Vec<i64> = r.iter().map(|x| x.0).collect();
        assert_eq!(v[0], 60);
        assert_eq!(v[1], NAT);
        assert_eq!(v[2], NAT);
    }

    #[test]
    fn add_datetime_timedelta_basic() {
        let a = Array::<DateTime64, Ix1>::from_vec(
            Ix1::new([2]),
            vec![DateTime64(1000), DateTime64(2000)],
        )
        .unwrap();
        let b = Array::<Timedelta64, Ix1>::from_vec(
            Ix1::new([2]),
            vec![Timedelta64(50), Timedelta64::nat()],
        )
        .unwrap();
        let r = add_datetime_timedelta(&a, &b).unwrap();
        let v: Vec<i64> = r.iter().map(|x| x.0).collect();
        assert_eq!(v[0], 1050);
        assert_eq!(v[1], NAT); // NaT propagates
    }

    #[test]
    fn sub_datetime_timedelta_basic() {
        let a = Array::<DateTime64, Ix1>::from_vec(
            Ix1::new([2]),
            vec![DateTime64(1000), DateTime64(500)],
        )
        .unwrap();
        let b = Array::<Timedelta64, Ix1>::from_vec(
            Ix1::new([2]),
            vec![Timedelta64(100), Timedelta64(50)],
        )
        .unwrap();
        let r = sub_datetime_timedelta(&a, &b).unwrap();
        let v: Vec<i64> = r.iter().map(|x| x.0).collect();
        assert_eq!(v, vec![900, 450]);
    }

    #[test]
    fn add_sub_timedelta_basic() {
        let a = Array::<Timedelta64, Ix1>::from_vec(
            Ix1::new([2]),
            vec![Timedelta64(10), Timedelta64(20)],
        )
        .unwrap();
        let b = Array::<Timedelta64, Ix1>::from_vec(
            Ix1::new([2]),
            vec![Timedelta64(3), Timedelta64(7)],
        )
        .unwrap();
        let s = add_timedelta(&a, &b).unwrap();
        let sv: Vec<i64> = s.iter().map(|x| x.0).collect();
        assert_eq!(sv, vec![13, 27]);
        let d = sub_timedelta(&a, &b).unwrap();
        let dv: Vec<i64> = d.iter().map(|x| x.0).collect();
        assert_eq!(dv, vec![7, 13]);
    }

    #[test]
    fn arithmetic_shape_mismatch_errors() {
        let a =
            Array::<DateTime64, Ix1>::from_vec(Ix1::new([2]), vec![DateTime64(1), DateTime64(2)])
                .unwrap();
        let b = Array::<DateTime64, Ix1>::from_vec(
            Ix1::new([3]),
            vec![DateTime64(1), DateTime64(2), DateTime64(3)],
        )
        .unwrap();
        assert!(sub_datetime(&a, &b).is_err());
    }

    // ---- Promoted (mixed-unit) arithmetic ----

    #[test]
    fn timeunit_finer_picks_finer() {
        assert_eq!(TimeUnit::Ns.finer(TimeUnit::Us), TimeUnit::Ns);
        assert_eq!(TimeUnit::Us.finer(TimeUnit::Ns), TimeUnit::Ns);
        assert_eq!(TimeUnit::S.finer(TimeUnit::Ms), TimeUnit::Ms);
        assert_eq!(TimeUnit::Ns.finer(TimeUnit::Ns), TimeUnit::Ns);
    }

    #[test]
    fn timeunit_scale_to_correct_factors() {
        assert_eq!(TimeUnit::Us.scale_to(TimeUnit::Ns), Some(1_000));
        assert_eq!(TimeUnit::S.scale_to(TimeUnit::Us), Some(1_000_000));
        assert_eq!(TimeUnit::Ns.scale_to(TimeUnit::S), None);
        assert_eq!(TimeUnit::Ns.scale_to(TimeUnit::Ns), Some(1));
    }

    #[test]
    fn sub_datetime_promoted_us_minus_ns() {
        let a =
            Array::<DateTime64, Ix1>::from_vec(Ix1::new([2]), vec![DateTime64(5), DateTime64(10)])
                .unwrap();
        let b = Array::<DateTime64, Ix1>::from_vec(
            Ix1::new([2]),
            vec![DateTime64(1234), DateTime64(7777)],
        )
        .unwrap();
        let (result, unit) = sub_datetime_promoted(&a, TimeUnit::Us, &b, TimeUnit::Ns).unwrap();
        assert_eq!(unit, TimeUnit::Ns);
        let v: Vec<i64> = result.iter().map(|x| x.0).collect();
        assert_eq!(v, vec![5_000 - 1_234, 10_000 - 7_777]);
    }

    #[test]
    fn sub_datetime_promoted_propagates_nat() {
        let a = Array::<DateTime64, Ix1>::from_vec(
            Ix1::new([2]),
            vec![DateTime64(5), DateTime64::nat()],
        )
        .unwrap();
        let b = Array::<DateTime64, Ix1>::from_vec(
            Ix1::new([2]),
            vec![DateTime64(1000), DateTime64(2000)],
        )
        .unwrap();
        let (result, _) = sub_datetime_promoted(&a, TimeUnit::Us, &b, TimeUnit::Ns).unwrap();
        let v: Vec<i64> = result.iter().map(|x| x.0).collect();
        assert_eq!(v[0], 5_000 - 1_000);
        assert_eq!(v[1], NAT);
    }

    #[test]
    fn add_datetime_timedelta_promoted_basic() {
        let a = Array::<DateTime64, Ix1>::from_vec(Ix1::new([1]), vec![DateTime64(2)]).unwrap();
        let b = Array::<Timedelta64, Ix1>::from_vec(Ix1::new([1]), vec![Timedelta64(500)]).unwrap();
        let (result, unit) =
            add_datetime_timedelta_promoted(&a, TimeUnit::S, &b, TimeUnit::Ms).unwrap();
        assert_eq!(unit, TimeUnit::Ms);
        // 2 s = 2000 ms + 500 ms = 2500 ms
        assert_eq!(result.iter().next().unwrap().0, 2_500);
    }

    #[test]
    fn add_timedelta_promoted_basic() {
        let a = Array::<Timedelta64, Ix1>::from_vec(Ix1::new([1]), vec![Timedelta64(3)]).unwrap();
        let b = Array::<Timedelta64, Ix1>::from_vec(Ix1::new([1]), vec![Timedelta64(250)]).unwrap();
        let (result, unit) = add_timedelta_promoted(&a, TimeUnit::S, &b, TimeUnit::Ms).unwrap();
        assert_eq!(unit, TimeUnit::Ms);
        assert_eq!(result.iter().next().unwrap().0, 3_250);
    }

    // ---- Array-level operator overloads ----

    #[test]
    fn array_sub_datetime_via_operator() {
        // The operator overload below routes to sub_datetime on `&Array - &Array`.
        let a = Array::<DateTime64, Ix1>::from_vec(
            Ix1::new([3]),
            vec![DateTime64(100), DateTime64(200), DateTime64(50)],
        )
        .unwrap();
        let b = Array::<DateTime64, Ix1>::from_vec(
            Ix1::new([3]),
            vec![DateTime64(40), DateTime64(150), DateTime64(50)],
        )
        .unwrap();
        let result = (&a - &b).unwrap();
        let v: Vec<i64> = result.iter().map(|x| x.0).collect();
        assert_eq!(v, vec![60, 50, 0]);
    }

    #[test]
    fn array_add_datetime_timedelta_via_operator() {
        let a = Array::<DateTime64, Ix1>::from_vec(
            Ix1::new([2]),
            vec![DateTime64(1000), DateTime64(2000)],
        )
        .unwrap();
        let b = Array::<Timedelta64, Ix1>::from_vec(
            Ix1::new([2]),
            vec![Timedelta64(50), Timedelta64(75)],
        )
        .unwrap();
        let result = (&a + &b).unwrap();
        let v: Vec<i64> = result.iter().map(|x| x.0).collect();
        assert_eq!(v, vec![1050, 2075]);
    }

    #[test]
    fn array_timedelta_arith_via_operators() {
        let a = Array::<Timedelta64, Ix1>::from_vec(
            Ix1::new([2]),
            vec![Timedelta64(10), Timedelta64(20)],
        )
        .unwrap();
        let b = Array::<Timedelta64, Ix1>::from_vec(
            Ix1::new([2]),
            vec![Timedelta64(3), Timedelta64(7)],
        )
        .unwrap();
        let s = (&a + &b).unwrap();
        let sv: Vec<i64> = s.iter().map(|x| x.0).collect();
        assert_eq!(sv, vec![13, 27]);
        let d = (&a - &b).unwrap();
        let dv: Vec<i64> = d.iter().map(|x| x.0).collect();
        assert_eq!(dv, vec![7, 13]);
    }

    #[test]
    fn promoted_same_unit_passes_through() {
        let a = Array::<DateTime64, Ix1>::from_vec(
            Ix1::new([3]),
            vec![DateTime64(100), DateTime64(200), DateTime64(50)],
        )
        .unwrap();
        let b = Array::<DateTime64, Ix1>::from_vec(
            Ix1::new([3]),
            vec![DateTime64(40), DateTime64(150), DateTime64(50)],
        )
        .unwrap();
        let basic = sub_datetime(&a, &b).unwrap();
        let (promoted, unit) = sub_datetime_promoted(&a, TimeUnit::Ns, &b, TimeUnit::Ns).unwrap();
        assert_eq!(unit, TimeUnit::Ns);
        let bv: Vec<i64> = basic.iter().map(|x| x.0).collect();
        let pv: Vec<i64> = promoted.iter().map(|x| x.0).collect();
        assert_eq!(bv, pv);
    }
}

#[cfg(test)]
mod arith_tests {
    // timedelta numeric arithmetic (REQ-2, #942). Expected values derived LIVE
    // from numpy 2.4.5 (R-CHAR-3): np.timedelta64(6,'D')*3 -> 18; td*float /
    // td/float TRUNCATE toward zero; td/td -> float; td//td -> int (floor);
    // td%td -> td (Python floor-mod, sign follows divisor); NaT + zero divisor.
    use super::*;
    use ferray_core::dimension::Ix1;

    fn td1(vals: &[i64]) -> Array<Timedelta64, Ix1> {
        let data: Vec<Timedelta64> = vals.iter().map(|&v| Timedelta64(v)).collect();
        Array::<Timedelta64, Ix1>::from_vec(Ix1::new([vals.len()]), data).unwrap()
    }

    fn first_td(a: Array<Timedelta64, Ix1>) -> i64 {
        a.iter().next().unwrap().0
    }

    fn first_i64(a: Array<i64, Ix1>) -> i64 {
        *a.iter().next().unwrap()
    }

    #[test]
    fn mul_timedelta_scalar_i64_exact() {
        // np.timedelta64(6,'D')*3 == timedelta64(18,'D'); NaT propagates.
        let r = mul_timedelta_scalar_i64(&td1(&[6, 5, NAT]), 3).unwrap();
        let v: Vec<i64> = r.iter().map(|x| x.0).collect();
        assert_eq!(v, vec![18, 15, NAT]);
    }

    #[test]
    fn mul_timedelta_scalar_f64_truncates_toward_zero() {
        // numpy casts the double product to int64 (trunc toward zero), live:
        //   td(5,'D')*1.5 -> 7 (7.5 trunc), td(5,'D')*2.5 -> 12 (12.5 trunc),
        //   td(1,'D')*0.5 -> 0, td(1,'D')*-2.9 -> -2.
        assert_eq!(
            first_td(mul_timedelta_scalar_f64(&td1(&[5]), 1.5).unwrap()),
            7
        );
        assert_eq!(
            first_td(mul_timedelta_scalar_f64(&td1(&[5]), 2.5).unwrap()),
            12
        );
        assert_eq!(
            first_td(mul_timedelta_scalar_f64(&td1(&[1]), 0.5).unwrap()),
            0
        );
        assert_eq!(
            first_td(mul_timedelta_scalar_f64(&td1(&[1]), -2.9).unwrap()),
            -2
        );
        assert_eq!(
            first_td(mul_timedelta_scalar_f64(&td1(&[NAT]), 2.0).unwrap()),
            NAT
        );
    }

    #[test]
    fn div_timedelta_scalar_i64_trunc_and_zero() {
        // td(6,'D')/2 -> 3; td(-7,'D')/2 -> -3 (trunc toward zero, live);
        // zero divisor -> NaT; NaT -> NaT.
        assert_eq!(
            first_td(div_timedelta_scalar_i64(&td1(&[6]), 2).unwrap()),
            3
        );
        assert_eq!(
            first_td(div_timedelta_scalar_i64(&td1(&[-7]), 2).unwrap()),
            -3
        );
        assert_eq!(
            first_td(div_timedelta_scalar_i64(&td1(&[6]), 0).unwrap()),
            NAT
        );
        assert_eq!(
            first_td(div_timedelta_scalar_i64(&td1(&[NAT]), 2).unwrap()),
            NAT
        );
    }

    #[test]
    fn div_timedelta_scalar_f64_trunc_and_nonfinite() {
        // td(5,'D')/2.0 -> 2 (2.5 trunc), td(5,'D')/2.5 -> 2; /0.0 -> NaT.
        assert_eq!(
            first_td(div_timedelta_scalar_f64(&td1(&[5]), 2.0).unwrap()),
            2
        );
        assert_eq!(
            first_td(div_timedelta_scalar_f64(&td1(&[5]), 2.5).unwrap()),
            2
        );
        assert_eq!(
            first_td(div_timedelta_scalar_f64(&td1(&[5]), 0.0).unwrap()),
            NAT
        );
    }

    #[test]
    fn truediv_timedelta_ratio_and_nat() {
        // td(6,'D')/td(2,'D') -> 3.0; NaT either -> nan; td/td0 -> inf (live).
        let r = truediv_timedelta(&td1(&[6]), TimeUnit::D, &td1(&[2]), TimeUnit::D).unwrap();
        assert_eq!(*r.iter().next().unwrap(), 3.0);
        let rn = truediv_timedelta(&td1(&[NAT]), TimeUnit::D, &td1(&[2]), TimeUnit::D).unwrap();
        assert!(rn.iter().next().unwrap().is_nan());
        let rz = truediv_timedelta(&td1(&[5]), TimeUnit::D, &td1(&[0]), TimeUnit::D).unwrap();
        assert!(rz.iter().next().unwrap().is_infinite());
    }

    #[test]
    fn truediv_timedelta_cross_unit() {
        // td(1,'D')/td(12,'h') -> 2.0 (promote D->h: 24h/12h). Live.
        let r = truediv_timedelta(&td1(&[1]), TimeUnit::D, &td1(&[12]), TimeUnit::H).unwrap();
        assert_eq!(*r.iter().next().unwrap(), 2.0);
    }

    #[test]
    fn floordiv_timedelta_floor_and_zero() {
        // td(7,'D')//td(2,'D') -> 3; td(-7,'D')//td(2,'D') -> -4 (floor, live);
        // NaT either or zero divisor -> 0.
        assert_eq!(
            first_i64(
                floordiv_timedelta(&td1(&[7]), TimeUnit::D, &td1(&[2]), TimeUnit::D).unwrap()
            ),
            3
        );
        assert_eq!(
            first_i64(
                floordiv_timedelta(&td1(&[-7]), TimeUnit::D, &td1(&[2]), TimeUnit::D).unwrap()
            ),
            -4
        );
        assert_eq!(
            first_i64(
                floordiv_timedelta(&td1(&[5]), TimeUnit::D, &td1(&[0]), TimeUnit::D).unwrap()
            ),
            0
        );
        assert_eq!(
            first_i64(
                floordiv_timedelta(&td1(&[NAT]), TimeUnit::D, &td1(&[2]), TimeUnit::D).unwrap()
            ),
            0
        );
    }

    #[test]
    fn floordiv_timedelta_cross_unit() {
        // td(1,'D')//td(5,'h') -> 4 (24h // 5h). Live.
        assert_eq!(
            first_i64(
                floordiv_timedelta(&td1(&[1]), TimeUnit::D, &td1(&[5]), TimeUnit::H).unwrap()
            ),
            4
        );
    }

    #[test]
    fn mod_timedelta_python_floor_mod() {
        // td(7,'D')%td(4,'D') -> 3; Python floor-mod: -7%2 -> 1, 7%-2 -> -1
        // (sign follows divisor, live); NaT either or zero divisor -> NaT.
        let (r, u) = mod_timedelta(&td1(&[7]), TimeUnit::D, &td1(&[4]), TimeUnit::D).unwrap();
        assert_eq!(u, TimeUnit::D);
        assert_eq!(first_td(r), 3);
        assert_eq!(
            first_td(
                mod_timedelta(&td1(&[-7]), TimeUnit::D, &td1(&[2]), TimeUnit::D)
                    .unwrap()
                    .0
            ),
            1
        );
        assert_eq!(
            first_td(
                mod_timedelta(&td1(&[7]), TimeUnit::D, &td1(&[-2]), TimeUnit::D)
                    .unwrap()
                    .0
            ),
            -1
        );
        assert_eq!(
            first_td(
                mod_timedelta(&td1(&[NAT]), TimeUnit::D, &td1(&[2]), TimeUnit::D)
                    .unwrap()
                    .0
            ),
            NAT
        );
        assert_eq!(
            first_td(
                mod_timedelta(&td1(&[5]), TimeUnit::D, &td1(&[0]), TimeUnit::D)
                    .unwrap()
                    .0
            ),
            NAT
        );
    }

    #[test]
    fn mod_timedelta_cross_unit() {
        // td(1,'D')%td(5,'h') -> timedelta64(4,'h') (24h % 5h = 4h, finer unit).
        let (r, u) = mod_timedelta(&td1(&[1]), TimeUnit::D, &td1(&[5]), TimeUnit::H).unwrap();
        assert_eq!(u, TimeUnit::H);
        assert_eq!(first_td(r), 4);
    }
}
