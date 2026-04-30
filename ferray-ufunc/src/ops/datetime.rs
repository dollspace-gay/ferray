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
