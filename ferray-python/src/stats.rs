//! Bindings for the `numpy` statistics surface — reductions,
//! nan-aware reductions, sort/search, set operations, cumulative
//! reductions, and `where`.
//!
//! Reductions return a `numpy.ndarray` (0-D when `axis=None`). NumPy
//! itself returns 0-D scalars (e.g. `numpy.int64`) for the same case;
//! a 0-D ndarray and a NumPy scalar are arithmetically interchangeable
//! and the deviation is invisible to almost all callers. If exact
//! parity is needed for a specific call, do `result.item()` to drop to
//! a Python scalar.
//!
//! Index-returning functions (`argmin`, `argmax`, `argsort`,
//! `count_nonzero`, `searchsorted`) are cast from ferray's internal
//! `u64` to `int64` so they match NumPy's `intp` default on 64-bit
//! platforms.

use ferray_core::array::aliases::{Array1, ArrayD};
use ferray_core::array::reductions::ReduceAcc;
use ferray_core::dimension::{Ix1, IxDyn};
use ferray_linalg as fl;
use ferray_numpy_interop::{AsFerray, IntoNumPy};
use num_complex::Complex;
use numpy::{PyReadonlyArray1, PyReadonlyArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::conv::{
    DynMarshal, as_ndarray, coerce_dtype, dtype_name, extract_q, ferr_to_pyerr, normalize_axis,
    promote_linalg_input,
};
use crate::fft::{complex_ferray_to_pyarray, complex_pyarray_to_ferray};
use crate::{
    match_dtype_all, match_dtype_all_complex, match_dtype_float, match_dtype_numeric,
    match_dtype_orderable,
};

// ---------------------------------------------------------------------------
// Complex-input dispatch for reductions (#925).
//
// The prior binding made every reduction real-only, with two distinct failure
// classes on complex input (both verified live, numpy 2.4.4):
//
//   * `sum`/`prod`/`cumsum`/`cumprod` dispatch `match_dtype_numeric!`, which has
//     NO complex arm — a complex input raised `TypeError: unsupported dtype for
//     numeric op`. numpy COMPUTES these as complex scalars/arrays.
//   * `mean`/`var`/`std`/`average` coerced the input to `float64`/`f32` ahead of
//     `match_dtype_float!`, and numpy's complex->real cast DROPS the imaginary
//     part with a `ComplexWarning` — an active R-CODE-4 boundary corruption
//     (`fr.mean([1+2j,3-1j])` returned `1.0`; numpy returns `(2+0.5j)`).
//
// The complex library substrate already exists: `impl ReduceAcc for Complex<f32>
// /Complex<f64>` (`ferray-core/src/array/reductions.rs`, `Acc = Self`) lets
// `ferray_stats::sum`/`prod`/`cumsum`/`cumprod` fold complex directly, keeping
// the input width (c64->c64, c128->c128 — verified live, numpy 2.4.4 does NOT
// promote a complex sum/prod's width). `mean` is NOT foldable directly
// (`ferray_stats::mean` is `T: Float`-bounded and `Complex` is not `Float`), so
// it is composed as the complex sum divided by the count, mirroring ma #873's
// `ma_complex_mean`: with a purely-real divisor the complex-division loop
// reduces to a reciprocal-MULTIPLY (`re*(1/n)`, `im*(1/n)`), bit-identical to
// numpy's `dsum / cnt` and one ULP below the naive `re/n`. numpy's MAIN-array
// `mean` PRESERVES the complex width (c64->c64, c128->c128 — verified live,
// unlike numpy.ma which promotes to c128). `var`/`std` over complex return a
// REAL result `mean(|x - mean|^2)`, width-following (c64->float32,
// c128->float64 — verified live, again unlike ma's always-float64), computed
// via `|z| = hypot(re, im)` THEN squared to match numpy's
// `absolute(danom) ** 2` last-bit.
// ---------------------------------------------------------------------------

/// `true` for the two complex dtype names ferray marshals
/// (`complex64`/`complex128`, plus their `c8`/`c16` aliases).
fn is_complex_dtype(dt: &str) -> bool {
    matches!(dt, "complex128" | "c16" | "complex64" | "c8")
}

/// The real scalar arithmetic a complex `mean`/`var`/`std` needs, sealed to the
/// two real component widths ferray marshals (`f32`/`f64`). `num-traits` is not
/// a direct dependency of `ferray-python`, so this provides the small set of
/// `Float` operations (`hypot`, `sqrt`, identity/count conversions) used by the
/// complex reduction helpers without pulling a new crate into the manifest.
trait ReductionReal:
    Copy
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
{
    const ZERO: Self;
    const ONE: Self;
    /// `NaN` of this real width — the empty/all-NaN-lane result for the
    /// complex `nanmean`/`median` helpers (numpy returns `nan+nanj` there).
    const NAN: Self;
    fn from_usize(n: usize) -> Self;
    fn sqrt(self) -> Self;
}

impl ReductionReal for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const NAN: Self = f32::NAN;
    fn from_usize(n: usize) -> Self {
        n as f32
    }
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
}

impl ReductionReal for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const NAN: Self = f64::NAN;
    fn from_usize(n: usize) -> Self {
        n as f64
    }
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
}

/// Which fold a complex `sum`/`prod`/`cumsum`/`cumprod` dispatches.
#[derive(Clone, Copy)]
enum ComplexFold {
    Sum,
    Prod,
    CumSum,
    CumProd,
}

/// Dispatch a complex `sum`/`prod`/`cumsum`/`cumprod` over both complex widths
/// via the existing complex `ReduceAcc` (`Acc = Self`, so the result keeps the
/// input width). `sum`/`prod` produce a 0-D (or axis-reduced) complex array;
/// `cumsum`/`cumprod` keep the input shape. numpy: `np.sum([1+2j,3+4j])==(4+6j)`,
/// `np.prod([1+2j,3+4j])==(-5+10j)`, `np.cumsum([1+2j,3+4j])==[1+2j,4+6j]`
/// (live, numpy 2.4.4) — all keep the input complex width.
fn complex_fold_dispatch<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    dt: &str,
    axis: Option<usize>,
    fold: ComplexFold,
) -> PyResult<Bound<'py, PyAny>> {
    match dt {
        "complex128" | "c16" => {
            let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(arr)?;
            let r: ArrayD<Complex<f64>> = match fold {
                ComplexFold::Sum => ferray_stats::sum(&fa, axis).map_err(ferr_to_pyerr)?,
                ComplexFold::Prod => ferray_stats::prod(&fa, axis).map_err(ferr_to_pyerr)?,
                ComplexFold::CumSum => ferray_stats::cumsum(&fa, axis).map_err(ferr_to_pyerr)?,
                ComplexFold::CumProd => ferray_stats::cumprod(&fa, axis).map_err(ferr_to_pyerr)?,
            };
            complex_ferray_to_pyarray(py, r)
        }
        "complex64" | "c8" => {
            let fa: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(arr)?;
            let r: ArrayD<Complex<f32>> = match fold {
                ComplexFold::Sum => ferray_stats::sum(&fa, axis).map_err(ferr_to_pyerr)?,
                ComplexFold::Prod => ferray_stats::prod(&fa, axis).map_err(ferr_to_pyerr)?,
                ComplexFold::CumSum => ferray_stats::cumsum(&fa, axis).map_err(ferr_to_pyerr)?,
                ComplexFold::CumProd => ferray_stats::cumprod(&fa, axis).map_err(ferr_to_pyerr)?,
            };
            complex_ferray_to_pyarray(py, r)
        }
        other => Err(PyTypeError::new_err(format!(
            "complex_fold_dispatch: expected a complex dtype, got {other:?}"
        ))),
    }
}

/// Element-wise complex mean of an axis-lane (or the whole array when
/// `axis=None`): the complex sum divided by the lane count via a
/// reciprocal-MULTIPLY (`re*(1/n)`, `im*(1/n)`), the form numpy's `dsum / cnt`
/// complex-division loop reduces to for a purely-real divisor (ma #873). Keeps
/// the input complex width `T` (numpy main-array `mean` preserves c64/c128).
fn complex_mean_dispatch<'py, T>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default + ReductionReal,
    Complex<T>: ferray_core::Element + numpy::Element,
    Complex<T>: ReduceAcc<Acc = Complex<T>> + Send + Sync,
{
    let fa: ArrayD<Complex<T>> = complex_pyarray_to_ferray::<T>(arr)?;
    let n: usize = match axis {
        None => fa.size(),
        Some(ax) => *fa.shape().get(ax).ok_or_else(|| {
            ferr_to_pyerr(ferray_core::FerrayError::invalid_value(
                "complex mean: axis out of bounds",
            ))
        })?,
    };
    if n == 0 {
        return Err(ferr_to_pyerr(ferray_core::FerrayError::invalid_value(
            "cannot compute mean of empty array",
        )));
    }
    let summed: ArrayD<Complex<T>> = ferray_stats::sum(&fa, axis).map_err(ferr_to_pyerr)?;
    // Reciprocal-multiply per component: `re*(1/n)`, `im*(1/n)` — numpy's
    // `dsum / cnt` for a real divisor (ma #873, one ULP below naive `re/n`).
    let scl = T::ONE / T::from_usize(n);
    let shape: Vec<usize> = summed.shape().to_vec();
    let data: Vec<Complex<T>> = summed
        .iter()
        .map(|z| Complex::new(z.re * scl, z.im * scl))
        .collect();
    let r = ArrayD::<Complex<T>>::from_vec(IxDyn::new(&shape), data).map_err(ferr_to_pyerr)?;
    complex_ferray_to_pyarray(py, r)
}

/// Element-wise complex variance of an axis-lane (or whole array): the REAL
/// `mean(|x - mean|^2)`, where `|z| = hypot(re, im)` THEN squared (numpy's
/// `absolute(danom) ** 2`, ma #873). `ddof` divides by `n - ddof`. The result
/// element type `R` is the REAL width following the complex input (c64->f32,
/// c128->f64, verified live numpy 2.4.4). When `want_std`, the per-lane square
/// root is taken. Returns a real ndarray (NOT the complex helper) so no
/// imaginary part is fabricated (R-CODE-4).
fn complex_var_std_dispatch<'py, T>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    axis: Option<usize>,
    ddof: usize,
    want_std: bool,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default + ReductionReal,
    ArrayD<T>: IntoNumPy<T, IxDyn>,
    Complex<T>: ferray_core::Element + numpy::Element,
{
    let fa: ArrayD<Complex<T>> = complex_pyarray_to_ferray::<T>(arr)?;
    let shape: Vec<usize> = fa.shape().to_vec();
    let ndim = shape.len();

    // Compute one REAL variance value over a slice of complex elements.
    let var_of = |vals: &[Complex<T>]| -> Option<T> {
        let n = vals.len();
        if n == 0 || n <= ddof {
            return None;
        }
        // Reciprocal-multiply complex mean (matches `complex_mean_dispatch`).
        let scl = T::ONE / T::from_usize(n);
        let mut mre = T::ZERO;
        let mut mim = T::ZERO;
        for z in vals {
            mre = mre + z.re;
            mim = mim + z.im;
        }
        let (mre, mim) = (mre * scl, mim * scl);
        let mut sq_sum = T::ZERO;
        for z in vals {
            let dr = z.re - mre;
            let di = z.im - mim;
            // `|x - mean|^2` as `dr*dr + di*di` — numpy's MAIN-array `_var`
            // computes `(x * conjugate(x)).real` (numpy/_core/_methods.py:_var),
            // i.e. `re^2 + im^2`, NOT the `hypot(re,im)^2` numpy.ma uses (the two
            // paths differ in the last bit; `np.var([1+2j,3+4j])` is EXACTLY 2.0
            // only with `re^2 + im^2`, verified live numpy 2.4.4).
            sq_sum = sq_sum + dr * dr + di * di;
        }
        let denom = T::from_usize(n - ddof);
        let v = sq_sum / denom;
        Some(if want_std { v.sqrt() } else { v })
    };

    match axis {
        None => {
            let vals: Vec<Complex<T>> = fa.iter().copied().collect();
            let v = var_of(&vals).ok_or_else(|| {
                ferr_to_pyerr(ferray_core::FerrayError::invalid_value(
                    "complex var/std: degrees of freedom <= 0",
                ))
            })?;
            let r = ArrayD::<T>::from_vec(IxDyn::new(&[]), vec![v]).map_err(ferr_to_pyerr)?;
            Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
        }
        Some(ax) => {
            if ax >= ndim {
                return Err(ferr_to_pyerr(ferray_core::FerrayError::invalid_value(
                    "complex var/std: axis out of bounds",
                )));
            }
            let out_shape: Vec<usize> = shape
                .iter()
                .enumerate()
                .filter_map(|(d, &s)| if d == ax { None } else { Some(s) })
                .collect();
            let axis_len = shape[ax];
            let flat: Vec<Complex<T>> = fa.iter().copied().collect();
            // Row-major strides into the flat buffer.
            let mut strides = vec![1usize; ndim];
            for d in (0..ndim.saturating_sub(1)).rev() {
                strides[d] = strides[d + 1] * shape[d + 1];
            }
            // The axes that survive the reduction, in order, with their extents.
            let out_dims: Vec<usize> = (0..ndim).filter(|&d| d != ax).collect();
            let out_extents: Vec<usize> = out_dims.iter().map(|&d| shape[d]).collect();
            let out_n: usize = out_extents.iter().product::<usize>().max(1);
            let mut out: Vec<T> = Vec::with_capacity(out_n);
            // Linearise each of the `out_n` output coordinates (mixed-radix over
            // `out_extents`) into a base flat offset, then walk the reduced axis.
            for lin in 0..out_n {
                let mut rem = lin;
                let mut base = 0usize;
                for (i, &d) in out_dims.iter().enumerate() {
                    let ext = out_extents[i];
                    let c = rem % ext;
                    rem /= ext;
                    base += c * strides[d];
                }
                let lane: Vec<Complex<T>> = (0..axis_len)
                    .map(|k| flat[base + k * strides[ax]])
                    .collect();
                let v = var_of(&lane).ok_or_else(|| {
                    ferr_to_pyerr(ferray_core::FerrayError::invalid_value(
                        "complex var/std: degrees of freedom <= 0",
                    ))
                })?;
                out.push(v);
            }
            let r = ArrayD::<T>::from_vec(IxDyn::new(&out_shape), out).map_err(ferr_to_pyerr)?;
            Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
        }
    }
}

/// `np.average(complex, weights=w)` over `axis` (or the whole array): the
/// complex weighted mean `sum(a*w) / sum(w)`, computed in `complex128` (numpy
/// promotes a weighted complex average to complex128 regardless of the input
/// width — verified live numpy 2.4.4: a `complex64` input weighted average is
/// `complex128`). The complex input is read at its native width then widened to
/// `Complex<f64>`; `w` is coerced to `float64`. The division by the (real)
/// weight sum is a reciprocal-multiply per component, matching numpy's complex
/// `wsum`-division for a real divisor.
fn complex_weighted_average<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    weights: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    // Widen the complex input to complex128 regardless of native width.
    let arr128 = coerce_dtype(py, arr, "complex128")?;
    let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(&arr128)?;
    let shape: Vec<usize> = fa.shape().to_vec();
    let ndim = shape.len();

    let warr = coerce_dtype(py, weights, "float64")?;
    let wview: PyReadonlyArrayDyn<f64> = warr.extract()?;
    let fw: ArrayD<f64> = wview.as_ferray().map_err(ferr_to_pyerr)?;

    // Weighted mean over a paired slice of (value, weight): `sum(v*w)/sum(w)`,
    // divide-by-real as a reciprocal-multiply (one ULP below naive per-component
    // divide, matching numpy's complex `wsum` division for a real divisor).
    let wavg = |vals: &[Complex<f64>], ws: &[f64]| -> PyResult<Complex<f64>> {
        let mut acc = Complex::<f64>::new(0.0, 0.0);
        let mut wsum = 0.0f64;
        for (v, &w) in vals.iter().zip(ws.iter()) {
            acc.re += v.re * w;
            acc.im += v.im * w;
            wsum += w;
        }
        if wsum == 0.0 {
            return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                "Weights sum to zero, can't be normalized",
            ));
        }
        let scl = 1.0 / wsum;
        Ok(Complex::new(acc.re * scl, acc.im * scl))
    };

    match axis {
        None => {
            // numpy requires `weights` to match the flattened length here.
            let vals: Vec<Complex<f64>> = fa.iter().copied().collect();
            let ws: Vec<f64> = fw.iter().copied().collect();
            if ws.len() != vals.len() {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Axis must be specified when shapes of a and weights differ.",
                ));
            }
            let r = wavg(&vals, &ws)?;
            let rd = ArrayD::<Complex<f64>>::from_vec(IxDyn::new(&[]), vec![r])
                .map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, rd)
        }
        Some(ax) => {
            if ax >= ndim {
                return Err(ferr_to_pyerr(ferray_core::FerrayError::invalid_value(
                    "complex average: axis out of bounds",
                )));
            }
            // numpy allows `weights` to be 1-D along `axis` (length == shape[ax])
            // or full-shape. Build a per-lane weight lookup.
            let axis_len = shape[ax];
            let ws_flat: Vec<f64> = fw.iter().copied().collect();
            let w_is_1d = ws_flat.len() == axis_len && fw.ndim() == 1;
            if !w_is_1d && ws_flat.len() != fa.size() {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Axis must be specified when shapes of a and weights differ.",
                ));
            }
            let out_shape: Vec<usize> = shape
                .iter()
                .enumerate()
                .filter_map(|(d, &s)| if d == ax { None } else { Some(s) })
                .collect();
            let flat: Vec<Complex<f64>> = fa.iter().copied().collect();
            let mut strides = vec![1usize; ndim];
            for d in (0..ndim.saturating_sub(1)).rev() {
                strides[d] = strides[d + 1] * shape[d + 1];
            }
            let out_dims: Vec<usize> = (0..ndim).filter(|&d| d != ax).collect();
            let out_extents: Vec<usize> = out_dims.iter().map(|&d| shape[d]).collect();
            let out_n: usize = out_extents.iter().product::<usize>().max(1);
            let mut out: Vec<Complex<f64>> = Vec::with_capacity(out_n);
            for lin in 0..out_n {
                let mut rem = lin;
                let mut base = 0usize;
                for (i, &d) in out_dims.iter().enumerate() {
                    let ext = out_extents[i];
                    let c = rem % ext;
                    rem /= ext;
                    base += c * strides[d];
                }
                let mut lane = Vec::with_capacity(axis_len);
                let mut lane_w = Vec::with_capacity(axis_len);
                for k in 0..axis_len {
                    let off = base + k * strides[ax];
                    lane.push(flat[off]);
                    lane_w.push(if w_is_1d { ws_flat[k] } else { ws_flat[off] });
                }
                out.push(wavg(&lane, &lane_w)?);
            }
            let rd = ArrayD::<Complex<f64>>::from_vec(IxDyn::new(&out_shape), out)
                .map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, rd)
        }
    }
}

// ---------------------------------------------------------------------------
// Complex ORDERING ops (#932): sort / argsort / unique / min / max / ptp /
// argmin / argmax over complex input.
//
// The prior binding dispatched every ordering op through
// `match_dtype_orderable!`, which has NO complex arm — a complex input raised
// `TypeError: unsupported dtype for ordering op`. numpy orders complex
// LEXICOGRAPHICALLY: compare the real part first, then the imaginary part
// (numpy's `npy_sort.c.src` complex comparators; verified live numpy 2.4.5:
// `np.sort([3+1j,1+2j,1+1j]) == [1+1j,1+2j,3+1j]`).
//
// A complex value is "NaN" if EITHER component is NaN (numpy's `np.isnan`
// complex contract). Two distinct NaN dispositions, both verified live:
//   * sort / argsort / unique — NaN-complex sorts LAST
//     (`np.sort([1+1j,nan+0j,2+0j]) == [1+1j,2+0j,nan+0j]`), mirroring the #918
//     real nan-last comparator.
//   * min / max / argmin / argmax / ptp — NaN PROPAGATES: the FIRST NaN-complex
//     element/index wins (`np.min([3+1j,nan+0j,1+1j]) == nan+0j`,
//     `np.argmin(...) == 1`), matching numpy's reduction-nan semantics.
//
// All width-preserving (c64->c64, c128->c128; argsort/argmin return int64
// indices, verified live). This is binding-side only — `Complex` is not
// `PartialOrd`, so the library `match_dtype_orderable!`/`nan_last_cmp` path
// cannot carry it; the lexicographic comparator lives here over the flattened
// (or per-axis-lane) complex data, reusing the #924/#874 lexicographic pattern.
//
// #938 (D-class) extends the same `cplx_sort_cmp` order to the SORTED/SET ops
// (`searchsorted` / `partition` / `intersect1d` / `union1d` / `setdiff1d` /
// `setxor1d` / `isin`) — each routed via a `is_complex` gate to a
// `complex_*_dispatch` helper below (`ferray_stats` set/sort fns need
// `T: PartialOrd`, absent for `Complex`). `count_nonzero` additionally moved to
// the `match_dtype_all_complex!` + `DynMarshal` seam (#933) since it needs only
// `PartialEq` (`z != 0` = `re != 0 || im != 0`). Consumers: the matching
// top-level `#[pyfunction]`s in `lib.rs` `_ferray`. numpy live 2.4.5:
// `np.searchsorted([1,2,3],1+1j)==1`, `np.intersect1d([3+4j,1-2j],[1-2j])==[1-2j]`,
// `np.count_nonzero([3+4j,1-2j])==2`. Pinned green:
// `tests/test_divergence_complex_converge_audit.py` (searchsorted/partition/
// intersect1d/union1d/setdiff1d/isin/count_nonzero) + `test_expansion_complex_dclass.py`.
// ---------------------------------------------------------------------------

/// `true` if either component of `z` is NaN — numpy's complex `isnan` contract
/// (`np.isnan(complex(1, nan)) == True`), reused by every complex ordering op.
fn cplx_is_nan<T: PartialOrd + Copy>(z: &Complex<T>) -> bool {
    z.re.partial_cmp(&z.re).is_none() || z.im.partial_cmp(&z.im).is_none()
}

/// Total order for a SORT/ARGSORT/UNIQUE over complex: lexicographic
/// `(real, then imag)` with NaN-complex (either part NaN) sorted LAST. Mirrors
/// the #918 real `nan_last_cmp` extended to two components. `Complex` is not
/// `PartialOrd`, so the order is computed directly from the parts. Ties (equal
/// real, equal imag — and two NaN-complex) compare `Equal`, giving a stable
/// dedup for `unique`.
fn cplx_sort_cmp<T: PartialOrd + Copy>(a: &Complex<T>, b: &Complex<T>) -> core::cmp::Ordering {
    use core::cmp::Ordering;
    let an = cplx_is_nan(a);
    let bn = cplx_is_nan(b);
    match (an, bn) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater, // NaN sorts last
        (false, true) => Ordering::Less,
        (false, false) => match a.re.partial_cmp(&b.re) {
            Some(Ordering::Equal) | None => a.im.partial_cmp(&b.im).unwrap_or(Ordering::Equal),
            Some(o) => o,
        },
    }
}

/// `true` when `a` is the lexicographic MIN/MAX winner over `b` for a
/// min/max/argmin/argmax/ptp reduction — NaN PROPAGATES (a NaN-complex beats
/// any finite value, and the FIRST NaN wins on ties). `want_max` flips the
/// finite comparison. Used with a fold that keeps the first index on a tie, so
/// numpy's first-occurrence semantics hold (verified live: the first NaN-complex
/// index is returned).
fn cplx_reduce_wins<T: PartialOrd + Copy>(a: &Complex<T>, b: &Complex<T>, want_max: bool) -> bool {
    let an = cplx_is_nan(a);
    let bn = cplx_is_nan(b);
    if an || bn {
        // NaN propagates: `a` wins ONLY when `a` is NaN and the incumbent `b` is
        // not — so the FIRST NaN-complex becomes the extremum and no later NaN
        // displaces it (numpy first-occurrence: `np.argmin([3+1j,nan,1+1j,nan])
        // == 1`, live).
        return an && !bn;
    }
    if want_max {
        a.re > b.re || (a.re == b.re && a.im > b.im)
    } else {
        a.re < b.re || (a.re == b.re && a.im < b.im)
    }
}

/// Index of the lexicographic min/max element of a complex slice, NaN
/// propagating (first NaN-complex wins). The slice is non-empty.
fn cplx_arg_extremum<T: PartialOrd + Copy>(vals: &[Complex<T>], want_max: bool) -> usize {
    let mut best = 0usize;
    for (i, z) in vals.iter().enumerate().skip(1) {
        if cplx_reduce_wins(z, &vals[best], want_max) {
            best = i;
        }
    }
    best
}

/// Walk every output coordinate of an axis reduction over a row-major flat
/// complex buffer, calling `f` with each lane's elements and pushing its
/// result. Shared by the complex min/max/ptp/argmin/argmax/sort/unique axis
/// arms. Returns `(out_shape, results)`.
fn cplx_axis_lanes<T, R>(
    fa: &ArrayD<Complex<T>>,
    ax: usize,
    mut f: impl FnMut(&[Complex<T>]) -> R,
) -> PyResult<(Vec<usize>, Vec<R>)>
where
    T: Copy,
    Complex<T>: ferray_core::Element,
{
    let shape: Vec<usize> = fa.shape().to_vec();
    let ndim = shape.len();
    if ax >= ndim {
        return Err(ferr_to_pyerr(ferray_core::FerrayError::invalid_value(
            "complex ordering op: axis out of bounds",
        )));
    }
    let out_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter_map(|(d, &s)| if d == ax { None } else { Some(s) })
        .collect();
    let axis_len = shape[ax];
    let flat: Vec<Complex<T>> = fa.iter().copied().collect();
    let mut strides = vec![1usize; ndim];
    for d in (0..ndim.saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    let out_dims: Vec<usize> = (0..ndim).filter(|&d| d != ax).collect();
    let out_extents: Vec<usize> = out_dims.iter().map(|&d| shape[d]).collect();
    let out_n: usize = out_extents.iter().product::<usize>().max(1);
    let mut results: Vec<R> = Vec::with_capacity(out_n);
    for lin in 0..out_n {
        let mut rem = lin;
        let mut base = 0usize;
        for (i, &d) in out_dims.iter().enumerate() {
            let ext = out_extents[i];
            let c = rem % ext;
            rem /= ext;
            base += c * strides[d];
        }
        let lane: Vec<Complex<T>> = (0..axis_len)
            .map(|k| flat[base + k * strides[ax]])
            .collect();
        results.push(f(&lane));
    }
    Ok((out_shape, results))
}

/// Complex `sort`: lexicographic with NaN-complex last. `axis=None` flattens to
/// a 1-D result (matching the real `ferray_stats::sort` axis=None contract);
/// an integer axis sorts each lane in place, keeping the input shape.
fn complex_sort_dispatch<'py, T>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default + PartialOrd,
    Complex<T>: ferray_core::Element + numpy::Element,
{
    let fa: ArrayD<Complex<T>> = complex_pyarray_to_ferray::<T>(arr)?;
    match axis {
        None => {
            let mut data: Vec<Complex<T>> = fa.iter().copied().collect();
            let n = data.len();
            data.sort_by(cplx_sort_cmp);
            let r =
                ArrayD::<Complex<T>>::from_vec(IxDyn::new(&[n]), data).map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, r)
        }
        Some(ax) => {
            let shape: Vec<usize> = fa.shape().to_vec();
            let ndim = shape.len();
            if ax >= ndim {
                return Err(ferr_to_pyerr(ferray_core::FerrayError::invalid_value(
                    "complex sort: axis out of bounds",
                )));
            }
            let axis_len = shape[ax];
            let mut flat: Vec<Complex<T>> = fa.iter().copied().collect();
            let mut strides = vec![1usize; ndim];
            for d in (0..ndim.saturating_sub(1)).rev() {
                strides[d] = strides[d + 1] * shape[d + 1];
            }
            let out_dims: Vec<usize> = (0..ndim).filter(|&d| d != ax).collect();
            let out_extents: Vec<usize> = out_dims.iter().map(|&d| shape[d]).collect();
            let out_n: usize = out_extents.iter().product::<usize>().max(1);
            for lin in 0..out_n {
                let mut rem = lin;
                let mut base = 0usize;
                for (i, &d) in out_dims.iter().enumerate() {
                    let ext = out_extents[i];
                    let c = rem % ext;
                    rem /= ext;
                    base += c * strides[d];
                }
                let mut lane: Vec<Complex<T>> = (0..axis_len)
                    .map(|k| flat[base + k * strides[ax]])
                    .collect();
                lane.sort_by(cplx_sort_cmp);
                for (k, v) in lane.into_iter().enumerate() {
                    flat[base + k * strides[ax]] = v;
                }
            }
            let r =
                ArrayD::<Complex<T>>::from_vec(IxDyn::new(&shape), flat).map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, r)
        }
    }
}

/// Complex `argsort`: lexicographic-with-nan-last indices (int64). `axis=None`
/// flattens; an integer axis argsorts each lane (axis-local indices), keeping
/// the input shape. A stable sort gives numpy's first-occurrence tie order.
fn complex_argsort_dispatch<'py, T>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default + PartialOrd,
    Complex<T>: ferray_core::Element + numpy::Element,
{
    let fa: ArrayD<Complex<T>> = complex_pyarray_to_ferray::<T>(arr)?;
    match axis {
        None => {
            let data: Vec<Complex<T>> = fa.iter().copied().collect();
            let n = data.len();
            let mut idx: Vec<u64> = (0..n as u64).collect();
            idx.sort_by(|&i, &j| cplx_sort_cmp(&data[i as usize], &data[j as usize]));
            let r = ArrayD::<u64>::from_vec(IxDyn::new(&[n]), idx).map_err(ferr_to_pyerr)?;
            u64_arrd_to_i64_pyarray(py, r)
        }
        Some(ax) => {
            let shape: Vec<usize> = fa.shape().to_vec();
            let (_out, lanes) = cplx_axis_lanes(&fa, ax, |lane| {
                let mut idx: Vec<u64> = (0..lane.len() as u64).collect();
                idx.sort_by(|&i, &j| cplx_sort_cmp(&lane[i as usize], &lane[j as usize]));
                idx
            })?;
            // Scatter the per-lane axis-local indices back into the input shape.
            let ndim = shape.len();
            let mut out_flat = vec![0u64; fa.size()];
            let mut strides = vec![1usize; ndim];
            for d in (0..ndim.saturating_sub(1)).rev() {
                strides[d] = strides[d + 1] * shape[d + 1];
            }
            let out_dims: Vec<usize> = (0..ndim).filter(|&d| d != ax).collect();
            let out_extents: Vec<usize> = out_dims.iter().map(|&d| shape[d]).collect();
            for (lin, lane_idx) in lanes.iter().enumerate() {
                let mut rem = lin;
                let mut base = 0usize;
                for (i, &d) in out_dims.iter().enumerate() {
                    let ext = out_extents[i];
                    let c = rem % ext;
                    rem /= ext;
                    base += c * strides[d];
                }
                for (k, &v) in lane_idx.iter().enumerate() {
                    out_flat[base + k * strides[ax]] = v;
                }
            }
            let r = ArrayD::<u64>::from_vec(IxDyn::new(&shape), out_flat).map_err(ferr_to_pyerr)?;
            u64_arrd_to_i64_pyarray(py, r)
        }
    }
}

/// Complex `unique`: lexicographic sort + dedup (NaN-complex last; numpy keeps
/// at most one NaN-complex — verified live `np.unique([nan+0j,nan+0j])` is a
/// single `nan+0j`). 1-D result, width preserved (numpy `unique` always
/// flattens).
fn complex_unique_dispatch<'py, T>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default + PartialOrd,
    Complex<T>: ferray_core::Element + numpy::Element,
{
    let fa: ArrayD<Complex<T>> = complex_pyarray_to_ferray::<T>(arr)?;
    let mut data: Vec<Complex<T>> = fa.iter().copied().collect();
    data.sort_by(cplx_sort_cmp);
    // Dedup adjacent equals (including two NaN-complex, which compare Equal).
    let mut out: Vec<Complex<T>> = Vec::with_capacity(data.len());
    for z in data {
        match out.last() {
            Some(prev) if cplx_sort_cmp(prev, &z) == core::cmp::Ordering::Equal => {}
            _ => out.push(z),
        }
    }
    let n = out.len();
    let r = ArrayD::<Complex<T>>::from_vec(IxDyn::new(&[n]), out).map_err(ferr_to_pyerr)?;
    complex_ferray_to_pyarray(py, r)
}

/// Extract a complex numpy array as a flat row-major `Vec<Complex<T>>`.
fn complex_flat_1d<'py, T>(arr: &Bound<'py, PyAny>) -> PyResult<Vec<Complex<T>>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default + PartialOrd,
    Complex<T>: ferray_core::Element + numpy::Element,
{
    let fa: ArrayD<Complex<T>> = complex_pyarray_to_ferray::<T>(arr)?;
    Ok(fa.iter().copied().collect())
}

/// Emit a flat `Vec<Complex<T>>` as a 1-D complex numpy array.
fn complex_vec_to_pyarray_1d<'py, T>(
    py: Python<'py>,
    data: Vec<Complex<T>>,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default + PartialOrd,
    Complex<T>: ferray_core::Element + numpy::Element,
{
    let n = data.len();
    let r = ArrayD::<Complex<T>>::from_vec(IxDyn::new(&[n]), data).map_err(ferr_to_pyerr)?;
    complex_ferray_to_pyarray(py, r)
}

/// Lexicographic sort + adjacent dedup of a complex vector (#932 order). Mirrors
/// numpy's `np.unique` used inside the set ops.
fn cplx_sorted_unique<T>(mut data: Vec<Complex<T>>) -> Vec<Complex<T>>
where
    T: PartialOrd + Copy,
{
    data.sort_by(cplx_sort_cmp);
    let mut out: Vec<Complex<T>> = Vec::with_capacity(data.len());
    for z in data {
        match out.last() {
            Some(prev) if cplx_sort_cmp(prev, &z) == core::cmp::Ordering::Equal => {}
            _ => out.push(z),
        }
    }
    out
}

/// Complex 1-D set operations (union/intersect/diff/xor) via the #932
/// lexicographic order. Each returns a SORTED UNIQUE complex array, matching
/// numpy's `union1d`/`intersect1d`/`setdiff1d`/`setxor1d` over complex (live).
fn complex_set_op_dispatch<'py, T>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    op: CplxSetOp,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default + PartialOrd,
    Complex<T>: ferray_core::Element + numpy::Element,
{
    let ua = cplx_sorted_unique(complex_flat_1d::<T>(a)?);
    let ub = cplx_sorted_unique(complex_flat_1d::<T>(b)?);
    let eq = |x: &Complex<T>, y: &Complex<T>| cplx_sort_cmp(x, y) == core::cmp::Ordering::Equal;
    let in_b = |z: &Complex<T>| ub.iter().any(|y| eq(z, y));
    let in_a = |z: &Complex<T>| ua.iter().any(|x| eq(z, x));
    let out: Vec<Complex<T>> = match op {
        CplxSetOp::Union => {
            let mut all = ua;
            all.extend(ub);
            cplx_sorted_unique(all)
        }
        CplxSetOp::Intersect => ua.into_iter().filter(|z| in_b(z)).collect(),
        CplxSetOp::Diff => ua.into_iter().filter(|z| !in_b(z)).collect(),
        CplxSetOp::Xor => {
            let mut r: Vec<Complex<T>> = ua.iter().copied().filter(|z| !in_b(z)).collect();
            r.extend(ub.iter().copied().filter(|z| !in_a(z)));
            cplx_sorted_unique(r)
        }
    };
    complex_vec_to_pyarray_1d(py, out)
}

/// Complex `isin`: per-element membership of `element` in `test_elements` by
/// exact complex equality. Matches numpy `np.isin` over complex (live).
fn complex_isin_dispatch<'py, T>(
    py: Python<'py>,
    element: &Bound<'py, PyAny>,
    test_elements: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default + PartialOrd,
    Complex<T>: ferray_core::Element + numpy::Element,
{
    let fa: ArrayD<Complex<T>> = complex_pyarray_to_ferray::<T>(element)?;
    let shape: Vec<usize> = fa.shape().to_vec();
    let tb: Vec<Complex<T>> = complex_flat_1d::<T>(test_elements)?;
    let eq = |x: &Complex<T>, y: &Complex<T>| cplx_sort_cmp(x, y) == core::cmp::Ordering::Equal;
    let mask: Vec<bool> = fa.iter().map(|z| tb.iter().any(|y| eq(z, y))).collect();
    let r = ArrayD::<bool>::from_vec(IxDyn::new(&shape), mask).map_err(ferr_to_pyerr)?;
    Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// Complex `partition`: a full lexicographic sort satisfies numpy's partition
/// postcondition (kth element in final position, smaller-before/larger-after).
/// Matches `np.partition` over complex (live).
fn complex_partition_dispatch<'py, T>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    kth: usize,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default + PartialOrd,
    Complex<T>: ferray_core::Element + numpy::Element,
{
    let mut data: Vec<Complex<T>> = complex_flat_1d::<T>(arr)?;
    let n = data.len();
    if kth >= n {
        return Err(ferr_to_pyerr(ferray_core::FerrayError::invalid_value(
            format!("kth(={kth}) out of bounds ({n})"),
        )));
    }
    data.sort_by(cplx_sort_cmp);
    complex_vec_to_pyarray_1d(py, data)
}

/// Complex `searchsorted`: lexicographic insertion points of `v` into the
/// (assumed sorted) complex array `a`. `side='left'`/`'right'` matches numpy's
/// bisect-left / bisect-right over the #932 lexicographic order (live).
fn complex_searchsorted_dispatch<'py, T>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    v: &Bound<'py, PyAny>,
    side: &str,
    v_scalar: bool,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default + PartialOrd,
    Complex<T>: ferray_core::Element + numpy::Element,
{
    use core::cmp::Ordering;
    let right = match side {
        "left" => false,
        "right" => true,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "side must be 'left' or 'right', got {other:?}"
            )));
        }
    };
    let base: Vec<Complex<T>> = complex_flat_1d::<T>(a)?;
    let needles: Vec<Complex<T>> = complex_flat_1d::<T>(v)?;
    let mut out: Vec<u64> = Vec::with_capacity(needles.len());
    for z in &needles {
        // bisect-left (side='left'): first index i with a[i] >= z.
        // bisect-right (side='right'): first index i with a[i] > z.
        let mut lo = 0usize;
        let mut hi = base.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let c = cplx_sort_cmp(&base[mid], z);
            let go_right = if right {
                c != Ordering::Greater // a[mid] <= z
            } else {
                c == Ordering::Less // a[mid] < z
            };
            if go_right {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        out.push(lo as u64);
    }
    if v_scalar {
        // Scalar `v` → 0-d int64 result (numpy returns a numpy scalar).
        let r = ArrayD::<u64>::from_vec(IxDyn::new(&[]), out).map_err(ferr_to_pyerr)?;
        return u64_arrd_to_i64_pyarray(py, r);
    }
    let r = Array1::<u64>::from_vec(Ix1::new([out.len()]), out).map_err(ferr_to_pyerr)?;
    u64_arr1_to_i64_pyarray(py, r)
}

/// Complex lexicographic `min`/`max` element (NaN propagating, first wins).
/// `axis=None` → 0-D complex; an integer axis collapses that axis.
fn complex_minmax_reduce<'py, T>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    axis: Option<usize>,
    want_max: bool,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default + PartialOrd,
    Complex<T>: ferray_core::Element + numpy::Element,
{
    let fa: ArrayD<Complex<T>> = complex_pyarray_to_ferray::<T>(arr)?;
    let pick = |lane: &[Complex<T>]| -> Complex<T> {
        let i = cplx_arg_extremum(lane, want_max);
        lane[i]
    };
    match axis {
        None => {
            let vals: Vec<Complex<T>> = fa.iter().copied().collect();
            if vals.is_empty() {
                return Err(ferr_to_pyerr(ferray_core::FerrayError::invalid_value(
                    "zero-size array to reduction operation",
                )));
            }
            let r = pick(&vals);
            let rd =
                ArrayD::<Complex<T>>::from_vec(IxDyn::new(&[]), vec![r]).map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, rd)
        }
        Some(ax) => {
            let (out_shape, vals) = cplx_axis_lanes(&fa, ax, |lane| pick(lane))?;
            let rd = ArrayD::<Complex<T>>::from_vec(IxDyn::new(&out_shape), vals)
                .map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, rd)
        }
    }
}

/// Complex lexicographic `argmin`/`argmax` (int64 index, NaN propagating, first
/// wins). `axis=None` → 0-D index; an integer axis collapses that axis.
fn complex_argextremum_reduce<'py, T>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    axis: Option<usize>,
    want_max: bool,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default + PartialOrd,
    Complex<T>: ferray_core::Element + numpy::Element,
{
    let fa: ArrayD<Complex<T>> = complex_pyarray_to_ferray::<T>(arr)?;
    match axis {
        None => {
            let vals: Vec<Complex<T>> = fa.iter().copied().collect();
            if vals.is_empty() {
                return Err(ferr_to_pyerr(ferray_core::FerrayError::invalid_value(
                    "attempt to get argmin/argmax of an empty sequence",
                )));
            }
            let i = cplx_arg_extremum(&vals, want_max) as u64;
            let rd = ArrayD::<u64>::from_vec(IxDyn::new(&[]), vec![i]).map_err(ferr_to_pyerr)?;
            u64_arrd_to_i64_pyarray(py, rd)
        }
        Some(ax) => {
            let (out_shape, vals) =
                cplx_axis_lanes(&fa, ax, |lane| cplx_arg_extremum(lane, want_max) as u64)?;
            let rd =
                ArrayD::<u64>::from_vec(IxDyn::new(&out_shape), vals).map_err(ferr_to_pyerr)?;
            u64_arrd_to_i64_pyarray(py, rd)
        }
    }
}

/// Complex `ptp` = lexicographic `max - min` (complex difference, NaN
/// propagating). `axis=None` → 0-D complex; an integer axis collapses that axis.
fn complex_ptp_reduce<'py, T>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default + PartialOrd,
    Complex<T>: ferray_core::Element + numpy::Element + core::ops::Sub<Output = Complex<T>>,
{
    let fa: ArrayD<Complex<T>> = complex_pyarray_to_ferray::<T>(arr)?;
    let ptp_of = |lane: &[Complex<T>]| -> Complex<T> {
        let lo = lane[cplx_arg_extremum(lane, false)];
        let hi = lane[cplx_arg_extremum(lane, true)];
        hi - lo
    };
    match axis {
        None => {
            let vals: Vec<Complex<T>> = fa.iter().copied().collect();
            if vals.is_empty() {
                return Err(ferr_to_pyerr(ferray_core::FerrayError::invalid_value(
                    "zero-size array to reduction operation",
                )));
            }
            let r = ptp_of(&vals);
            let rd =
                ArrayD::<Complex<T>>::from_vec(IxDyn::new(&[]), vec![r]).map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, rd)
        }
        Some(ax) => {
            let (out_shape, vals) = cplx_axis_lanes(&fa, ax, |lane| ptp_of(lane))?;
            let rd = ArrayD::<Complex<T>>::from_vec(IxDyn::new(&out_shape), vals)
                .map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, rd)
        }
    }
}

// ---------------------------------------------------------------------------
// Boolean reductions: all / any
// ---------------------------------------------------------------------------

/// `numpy.all(a, axis=None)` — test whether all elements are truthy.
///
/// `axis=None` returns a numpy `bool_` scalar (0-D bool array); an integer
/// `axis` collapses that axis, returning a bool ndarray
/// (numpy/_core/fromnumeric.py:2585 `all`, dispatching to
/// `numpy/_core/_methods.py:67` `_all` which `reduce`s `logical_and`). The
/// library `ferray_ufunc::all` is the whole-array fold; `all_axis` collapses
/// one axis. The whole-array `bool` is returned as a 0-D bool array (numpy's
/// `bool_` scalar and a 0-D bool array are interchangeable; `.item()` drops
/// to a Python bool if needed).
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn all<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let axis = norm_axis(py, &arr, axis)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        match axis {
            None => {
                let r = ferray_ufunc::all(&fa);
                ArrayD::<bool>::from_vec(IxDyn::new(&[]), vec![r])
                    .map_err(ferr_to_pyerr)?
                    .into_pyarray(py)
                    .map_err(ferr_to_pyerr)?
                    .into_any()
            }
            Some(ax) => {
                let r = ferray_ufunc::all_axis(&fa, ax).map_err(ferr_to_pyerr)?;
                r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
            }
        }
    }))
}

/// `numpy.any(a, axis=None)` — test whether any element is truthy.
///
/// Mirrors [`all`]: `axis=None` returns a 0-D bool, an integer `axis`
/// returns a bool ndarray (numpy/_core/fromnumeric.py:2479 `any` ->
/// `_methods.py:62` `_any` reducing `logical_or`).
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn any<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let axis = norm_axis(py, &arr, axis)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        match axis {
            None => {
                let r = ferray_ufunc::any(&fa);
                ArrayD::<bool>::from_vec(IxDyn::new(&[]), vec![r])
                    .map_err(ferr_to_pyerr)?
                    .into_pyarray(py)
                    .map_err(ferr_to_pyerr)?
                    .into_any()
            }
            Some(ax) => {
                let r = ferray_ufunc::any_axis(&fa, ax).map_err(ferr_to_pyerr)?;
                r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
            }
        }
    }))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Cast a `u64` ndarray to `i64` to match NumPy's default `intp`. The
/// values are always indices so the cast is non-lossy on 64-bit.
fn u64_arrd_to_i64_pyarray<'py>(py: Python<'py>, arr: ArrayD<u64>) -> PyResult<Bound<'py, PyAny>> {
    let shape = arr.shape().to_vec();
    let data: Vec<i64> = arr.iter().map(|&v| v as i64).collect();
    let cast = ArrayD::<i64>::from_vec(IxDyn::new(&shape), data).map_err(ferr_to_pyerr)?;
    Ok(cast.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

fn u64_arr1_to_i64_pyarray<'py>(py: Python<'py>, arr: Array1<u64>) -> PyResult<Bound<'py, PyAny>> {
    let n = arr.shape()[0];
    let data: Vec<i64> = arr.iter().map(|&v| v as i64).collect();
    let cast = Array1::<i64>::from_vec(Ix1::new([n]), data).map_err(ferr_to_pyerr)?;
    Ok(cast.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// True when the numpy array has zero elements (any axis of length 0).
/// Empty `mean`/`var`/`std`/`median` reductions over `axis=None` return
/// `nan` in NumPy (with a `RuntimeWarning`) rather than raising — see
/// numpy `_core/_methods.py:122` (`_mean`) and `:154-156` (`_var`).
fn is_empty_array(arr: &Bound<'_, PyAny>) -> PyResult<bool> {
    let size: usize = arr.getattr("size")?.extract()?;
    Ok(size == 0)
}

/// A 0-D `float64` array holding `nan`. This is the binding's scalar
/// return for an empty `mean`/`var`/`std`/`median` reduction, matching
/// NumPy's `nan` (float64, 0-D) result for an empty slice.
fn nan_scalar_f64<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    let r = ArrayD::<f64>::from_vec(IxDyn::new(&[]), vec![f64::NAN]).map_err(ferr_to_pyerr)?;
    Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

// ---------------------------------------------------------------------------
// Reductions: numeric (sum, prod) — accept any numeric dtype
// ---------------------------------------------------------------------------

/// Normalize a binding `axis` argument against an array's `ndim`, raising
/// `numpy.exceptions.AxisError` (subclass of ValueError AND IndexError) for
/// an out-of-bounds axis — matching numpy/exceptions.py:108
/// `class AxisError(ValueError, IndexError)`. ferray's library
/// `validate_axis` surfaces a `ValueError`, which `except IndexError` /
/// `except np.exceptions.AxisError` would silently miss, so the binding
/// validates at the boundary (R-DEV-2). A negative axis (valid in numpy) is
/// folded into `[0, ndim)`; `None` passes through.
fn norm_axis(
    py: Python<'_>,
    arr: &Bound<'_, PyAny>,
    axis: Option<isize>,
) -> PyResult<Option<usize>> {
    match axis {
        None => Ok(None),
        Some(ax) => {
            let ndim: usize = arr.getattr("ndim")?.extract()?;
            Ok(Some(normalize_axis(py, ax, ndim)?))
        }
    }
}

/// `numpy.sum(a, axis=None)`.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn sum<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let axis = norm_axis(py, &arr, axis)?;
    let dt = dtype_name(&arr)?;
    // datetime64/timedelta64 (REQ-4, #944): `sum(timedelta)->timedelta`,
    // `sum(datetime)` RAISES numpy's `UFuncTypeError`. Branch ahead of the
    // real-only macros, which would otherwise raise a ferray-side TypeError
    // (timedelta) or silently mis-handle.
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::time_reduce(py, &arr, crate::datetime::TimeReduce::Sum, axis);
    }
    // Complex sum folds via the complex `ReduceAcc` (`Acc = Self`), keeping the
    // input width (c64->c64, c128->c128) — numpy: `np.sum([1+2j,3+4j])==(4+6j)`,
    // complex64 stays complex64 (live, numpy 2.4.4). The real-only
    // `match_dtype_numeric!` below has no complex arm (would raise TypeError).
    if is_complex_dtype(dt.as_str()) {
        return complex_fold_dispatch(py, &arr, dt.as_str(), axis, ComplexFold::Sum);
    }
    // bool sums in the platform integer (int64) — numpy/_core/fromnumeric.py:2325
    // ("`a` is signed then the platform integer is used"). ferray's library
    // ReduceAcc maps bool -> i64, so dispatch bool to the same promoting path.
    if dt.as_str() == "bool" {
        let view: PyReadonlyArrayDyn<bool> = arr.extract()?;
        let fa: ArrayD<bool> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r = ferray_stats::sum(&fa, axis).map_err(ferr_to_pyerr)?;
        return Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any());
    }
    Ok(match_dtype_numeric!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r = ferray_stats::sum(&fa, axis).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.prod(a, axis=None)`.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn prod<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let axis = norm_axis(py, &arr, axis)?;
    let dt = dtype_name(&arr)?;
    // datetime64/timedelta64 (#944): `prod` is UNDEFINED for both — numpy
    // RAISES `UFuncTypeError` ('multiply' cannot use two datetimes/timedeltas,
    // live). Delegate so numpy's exact exception surfaces (R-DEV-2), rather than
    // the ferray-side TypeError the numeric macro would emit.
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::time_reduce(py, &arr, crate::datetime::TimeReduce::Prod, axis);
    }
    // Complex prod folds via the complex `ReduceAcc`, keeping width — numpy:
    // `np.prod([1+2j,3+4j])==(-5+10j)`, complex64 stays complex64 (live).
    if is_complex_dtype(dt.as_str()) {
        return complex_fold_dispatch(py, &arr, dt.as_str(), axis, ComplexFold::Prod);
    }
    // bool products promote to int64 like sum — numpy/_core/fromnumeric.py:2692.
    if dt.as_str() == "bool" {
        let view: PyReadonlyArrayDyn<bool> = arr.extract()?;
        let fa: ArrayD<bool> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r = ferray_stats::prod(&fa, axis).map_err(ferr_to_pyerr)?;
        return Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any());
    }
    Ok(match_dtype_numeric!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r = ferray_stats::prod(&fa, axis).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

// ---------------------------------------------------------------------------
// Reductions: orderable (min, max, ptp) — any orderable dtype
// ---------------------------------------------------------------------------

/// `numpy.min(a, axis=None)`.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn min<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    // datetime64/timedelta64 (REQ-3 ordering surfaced via #944): `min` ->
    // datetime/timedelta by int64 tick value, NaT propagating (live).
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::time_reduce(py, &arr, crate::datetime::TimeReduce::Min, axis);
    }
    // Complex min: lexicographic extremum (NaN propagates, first wins), width
    // preserved — numpy: `np.min([1+2j,3+1j]) == (1+2j)` (live). The real-only
    // `match_dtype_orderable!` below has no complex arm (would raise TypeError).
    match dt.as_str() {
        "complex128" | "c16" => return complex_minmax_reduce::<f64>(py, &arr, axis, false),
        "complex64" | "c8" => return complex_minmax_reduce::<f32>(py, &arr, axis, false),
        _ => {}
    }
    Ok(match_dtype_orderable!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r = ferray_stats::min(&fa, axis).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.max(a, axis=None)`.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn max<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    // datetime64/timedelta64 (#944): `max` -> datetime/timedelta by int64 tick
    // value, NaT propagating (live).
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::time_reduce(py, &arr, crate::datetime::TimeReduce::Max, axis);
    }
    // Complex max: lexicographic extremum (NaN propagates, first wins), width
    // preserved — numpy: `np.max([1+2j,3+1j]) == (3+1j)` (live).
    match dt.as_str() {
        "complex128" | "c16" => return complex_minmax_reduce::<f64>(py, &arr, axis, true),
        "complex64" | "c8" => return complex_minmax_reduce::<f32>(py, &arr, axis, true),
        _ => {}
    }
    Ok(match_dtype_orderable!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r = ferray_stats::max(&fa, axis).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.ptp(a, axis=None)` — peak-to-peak (max - min).
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn ptp<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    // datetime64/timedelta64 (REQ-4, #944): `ptp` -> timedelta ALWAYS
    // (datetime ptp = max-min is a timedelta; timedelta ptp stays timedelta),
    // NaT propagating — numpy `np.ptp(dt)==timedelta64(...)` (live).
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::time_reduce(py, &arr, crate::datetime::TimeReduce::Ptp, axis);
    }
    // Complex ptp: lexicographic max - lexicographic min (complex difference,
    // NaN propagates), width preserved — numpy: `np.ptp([1+2j,-3+4j,0+0j,2-1j])
    // == (5-5j)` (live).
    match dt.as_str() {
        "complex128" | "c16" => return complex_ptp_reduce::<f64>(py, &arr, axis),
        "complex64" | "c8" => return complex_ptp_reduce::<f32>(py, &arr, axis),
        _ => {}
    }
    Ok(match_dtype_numeric!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r = ferray_stats::ptp(&fa, axis).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

// ---------------------------------------------------------------------------
// Reductions: float-only (mean, var, std)
// ---------------------------------------------------------------------------

/// `numpy.mean(a, axis=None)` — float-only.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn mean<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    // datetime64/timedelta64 (REQ-4, #944): `mean(timedelta)->timedelta`
    // (int64 trunc-toward-zero of sum/n); `mean(datetime)` RAISES numpy's
    // `UFuncTypeError`. MUST branch BEFORE the f64 coercion below, which is the
    // silent-float corruption this REQ eliminates (`fr.mean(dt)->18268.5`).
    // Also ahead of the empty-array nan-float shortcut: an empty timedelta mean
    // RAISES, it does not return a float nan.
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::time_reduce(py, &arr, crate::datetime::TimeReduce::Mean, axis);
    }
    // Empty reduction: NumPy returns nan (float64, 0-D) + RuntimeWarning,
    // it does NOT raise (numpy/_core/_methods.py:122).
    if axis.is_none() && is_empty_array(&arr)? {
        return nan_scalar_f64(py);
    }
    let dt = dtype_name(&arr)?;
    // Complex mean: compute the complex sum / count (reciprocal-multiply,
    // ma #873), keeping the input width (c64->c64, c128->c128, verified live
    // numpy 2.4.4). MUST branch BEFORE the `coerce_dtype(arr,"float64")` below,
    // which numpy casts complex->float64 DROPPING the imaginary part with a
    // `ComplexWarning` — the R-CODE-4 corruption this REQ eliminates
    // (`fr.mean([1+2j,3-1j])` returned `1.0`; numpy returns `(2+0.5j)`).
    match dt.as_str() {
        "complex128" | "c16" => return complex_mean_dispatch::<f64>(py, &arr, axis),
        "complex64" | "c8" => return complex_mean_dispatch::<f32>(py, &arr, axis),
        _ => {}
    }
    // Promote integer inputs to float64 to match NumPy's mean semantics.
    let arr = if matches!(dt.as_str(), "float64" | "float32" | "f64" | "f32") {
        arr
    } else {
        coerce_dtype(py, &arr, "float64")?
    };
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r = ferray_stats::mean(&fa, axis).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.var(a, axis=None, ddof=0)`.
#[pyfunction]
#[pyo3(signature = (a, axis = None, ddof = 0))]
pub fn var<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
    ddof: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    // datetime64/timedelta64 (REQ-4, #944): `var` is UNDEFINED for both
    // (numpy RAISES `TypeError: ufunc 'square' not supported` for timedelta,
    // `UFuncTypeError` for datetime). Branch BEFORE the f64 coercion — returning
    // a float here is the silent-float corruption (`fr.var(td)->...`). The
    // delegate path surfaces numpy's exact exception (R-DEV-2).
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::time_reduce(py, &arr, crate::datetime::TimeReduce::Var, axis);
    }
    // Empty reduction: NumPy warns "Degrees of freedom <= 0" and returns
    // nan (float64, 0-D); it does NOT raise (numpy/_core/_methods.py:154).
    if axis.is_none() && is_empty_array(&arr)? {
        return nan_scalar_f64(py);
    }
    let dt = dtype_name(&arr)?;
    // Complex var: REAL result `mean(|x - mean|^2)` (numpy's
    // `absolute(danom)**2`, ma #873), width-following (c64->float32,
    // c128->float64, verified live numpy 2.4.4). MUST branch BEFORE the
    // `coerce_dtype(arr,"float64")` imag-discard below (R-CODE-4).
    match dt.as_str() {
        "complex128" | "c16" => {
            return complex_var_std_dispatch::<f64>(py, &arr, axis, ddof, false);
        }
        "complex64" | "c8" => {
            return complex_var_std_dispatch::<f32>(py, &arr, axis, ddof, false);
        }
        _ => {}
    }
    let arr = if matches!(dt.as_str(), "float64" | "float32" | "f64" | "f32") {
        arr
    } else {
        coerce_dtype(py, &arr, "float64")?
    };
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r = ferray_stats::var(&fa, axis, ddof).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.std(a, axis=None, ddof=0)`.
#[pyfunction]
#[pyo3(signature = (a, axis = None, ddof = 0))]
pub fn std<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
    ddof: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    // datetime64/timedelta64 (REQ-4, #944): `std` is UNDEFINED for both —
    // numpy RAISES (`TypeError 'square' not supported` for timedelta,
    // `UFuncTypeError` for datetime). Branch BEFORE the f64 coercion that was
    // the silent-float corruption (`fr.std(td)->0.5`). delegate -> numpy's exact
    // exception (R-DEV-2).
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::time_reduce(py, &arr, crate::datetime::TimeReduce::Std, axis);
    }
    // Empty reduction: `_std` delegates to `_var`, which returns nan
    // (float64, 0-D) for an empty slice; it does NOT raise
    // (numpy/_core/_methods.py:217 -> :154).
    if axis.is_none() && is_empty_array(&arr)? {
        return nan_scalar_f64(py);
    }
    let dt = dtype_name(&arr)?;
    // Complex std: `sqrt` of the REAL complex variance, width-following
    // (c64->float32, c128->float64, verified live). Branch BEFORE the f64
    // coercion imag-discard (R-CODE-4).
    match dt.as_str() {
        "complex128" | "c16" => {
            return complex_var_std_dispatch::<f64>(py, &arr, axis, ddof, true);
        }
        "complex64" | "c8" => {
            return complex_var_std_dispatch::<f32>(py, &arr, axis, ddof, true);
        }
        _ => {}
    }
    let arr = if matches!(dt.as_str(), "float64" | "float32" | "f64" | "f32") {
        arr
    } else {
        coerce_dtype(py, &arr, "float64")?
    };
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r = ferray_stats::std_(&fa, axis, ddof).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

// ---------------------------------------------------------------------------
// Argmin / argmax (return int64 index arrays)
// ---------------------------------------------------------------------------

/// `numpy.argmin(a, axis=None)`.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn argmin<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    // datetime64/timedelta64 (#944): `argmin` -> int64 index by int64 tick
    // ordering; a NaT-containing array returns the FIRST NaT index (live).
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::time_reduce(py, &arr, crate::datetime::TimeReduce::Argmin, axis);
    }
    // Complex argmin: lexicographic min index (NaN propagates, first wins) —
    // numpy: `np.argmin([1+2j,-3+4j,0+0j,2-1j]) == 2` (live).
    match dt.as_str() {
        "complex128" | "c16" => return complex_argextremum_reduce::<f64>(py, &arr, axis, false),
        "complex64" | "c8" => return complex_argextremum_reduce::<f32>(py, &arr, axis, false),
        _ => {}
    }
    let r: ArrayD<u64> = match_dtype_orderable!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        ferray_stats::argmin(&fa, axis).map_err(ferr_to_pyerr)?
    });
    u64_arrd_to_i64_pyarray(py, r)
}

/// `numpy.argmax(a, axis=None)`.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn argmax<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    // datetime64/timedelta64 (#944): `argmax` -> int64 index by int64 tick
    // ordering; a NaT-containing array returns the FIRST NaT index (numpy
    // treats NaT as the argmax winner too, live).
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::time_reduce(py, &arr, crate::datetime::TimeReduce::Argmax, axis);
    }
    // Complex argmax: lexicographic max index (NaN propagates, first wins) —
    // numpy: `np.argmax([1+2j,-3+4j,0+0j,2-1j]) == 1`? (live). lexicographic
    // max of that set is `2-1j` at index 3.
    match dt.as_str() {
        "complex128" | "c16" => return complex_argextremum_reduce::<f64>(py, &arr, axis, true),
        "complex64" | "c8" => return complex_argextremum_reduce::<f32>(py, &arr, axis, true),
        _ => {}
    }
    let r: ArrayD<u64> = match_dtype_orderable!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        ferray_stats::argmax(&fa, axis).map_err(ferr_to_pyerr)?
    });
    u64_arrd_to_i64_pyarray(py, r)
}

// ---------------------------------------------------------------------------
// NaN-aware reductions (float-only)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Complex nan-aware reductions (#930).
//
// `nansum`/`nanprod`/`nanmean` funnelled every input through
// `coerce_dtype(arr, "float64")`, which numpy casts complex -> float64 DROPPING
// the imaginary part with a `ComplexWarning` — an active R-CODE-4 boundary
// corruption (`fr.nansum([1+2j,...])` returned the real scalar `0.0`; numpy
// returns the complex `5j`). numpy COMPUTES these as complex: `nansum`/`nanprod`
// replace NaN-complex (either part NaN) with the identity (0 / 1) then fold,
// preserving the input complex width (c64->c64, c128->c128 — verified live numpy
// 2.4.5: `np.nansum([...],dtype=complex64).dtype == complex64`). `nanmean` is
// `nansum(non-nan) / count(non-nan)`. The complex `ReduceAcc` (`Acc = Self`) lets
// `ferray_stats::sum`/`prod` fold the NaN-replaced complex array directly,
// mirroring the #925 `complex_fold_dispatch`/`complex_mean_dispatch` design.
// ---------------------------------------------------------------------------

/// `true` if either component of `z` is NaN — numpy's `np.isnan` contract for a
/// complex value (verified live: `np.isnan(complex(1, nan)) == True`).
fn complex_is_nan<T: PartialOrd + Copy>(z: &Complex<T>) -> bool {
    z.re.partial_cmp(&z.re).is_none() || z.im.partial_cmp(&z.im).is_none()
}

/// Which nan-aware fold a complex `nansum`/`nanprod` dispatches.
#[derive(Clone, Copy)]
enum ComplexNanFold {
    Sum,
    Prod,
}

/// Dispatch a complex `nansum`/`nanprod` over both complex widths: replace each
/// NaN-complex element with the fold identity (`0` for sum, `1` for prod), then
/// fold via the existing complex `ReduceAcc` (`Acc = Self`, width preserved).
/// numpy: `np.nansum([1+2j,nan+0j,3+0j]) == (4+2j)`,
/// `np.nanprod([1+2j,nan+nanj,2-1j]) == (4+3j)` (NaN -> identity, live 2.4.5).
fn complex_nan_fold_dispatch<'py, T>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    axis: Option<usize>,
    fold: ComplexNanFold,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default + ReductionReal + PartialOrd,
    Complex<T>: ferray_core::Element + numpy::Element,
    Complex<T>: ReduceAcc<Acc = Complex<T>> + Send + Sync,
{
    let fa: ArrayD<Complex<T>> = complex_pyarray_to_ferray::<T>(arr)?;
    let ident = match fold {
        ComplexNanFold::Sum => Complex::new(T::ZERO, T::ZERO),
        ComplexNanFold::Prod => Complex::new(T::ONE, T::ZERO),
    };
    let shape: Vec<usize> = fa.shape().to_vec();
    let data: Vec<Complex<T>> = fa
        .iter()
        .map(|z| if complex_is_nan(z) { ident } else { *z })
        .collect();
    let cleaned =
        ArrayD::<Complex<T>>::from_vec(IxDyn::new(&shape), data).map_err(ferr_to_pyerr)?;
    let r: ArrayD<Complex<T>> = match fold {
        ComplexNanFold::Sum => ferray_stats::sum(&cleaned, axis).map_err(ferr_to_pyerr)?,
        ComplexNanFold::Prod => ferray_stats::prod(&cleaned, axis).map_err(ferr_to_pyerr)?,
    };
    complex_ferray_to_pyarray(py, r)
}

/// Dispatch a complex `nanmean` over both complex widths: the complex
/// `sum(non-nan) / count(non-nan)` per lane (reciprocal-multiply per component,
/// mirroring `complex_mean_dispatch`). numpy: `np.nanmean([1+2j,nan+0j,3+0j]) ==
/// (2+1j)` (sum `4+2j` over count `2`), width preserved (live 2.4.5). An
/// all-NaN lane has count `0`; numpy returns `nan+nanj` there (a RuntimeWarning,
/// not a raise) — matched by emitting `NaN + NaN*i` for that lane.
fn complex_nanmean_dispatch<'py, T>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default + ReductionReal + PartialOrd,
    Complex<T>: ferray_core::Element + numpy::Element,
{
    let fa: ArrayD<Complex<T>> = complex_pyarray_to_ferray::<T>(arr)?;
    let shape: Vec<usize> = fa.shape().to_vec();
    let ndim = shape.len();

    // Mean of the non-NaN complex elements of one lane (reciprocal-multiply).
    let mean_of = |vals: &[Complex<T>]| -> Complex<T> {
        let mut re = T::ZERO;
        let mut im = T::ZERO;
        let mut cnt = 0usize;
        for z in vals {
            if !complex_is_nan(z) {
                re = re + z.re;
                im = im + z.im;
                cnt += 1;
            }
        }
        if cnt == 0 {
            // All-NaN lane: numpy returns nan+nanj with a RuntimeWarning.
            let nan = T::NAN;
            return Complex::new(nan, nan);
        }
        let scl = T::ONE / T::from_usize(cnt);
        Complex::new(re * scl, im * scl)
    };

    match axis {
        None => {
            let vals: Vec<Complex<T>> = fa.iter().copied().collect();
            let r = mean_of(&vals);
            let rd =
                ArrayD::<Complex<T>>::from_vec(IxDyn::new(&[]), vec![r]).map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, rd)
        }
        Some(ax) => {
            if ax >= ndim {
                return Err(ferr_to_pyerr(ferray_core::FerrayError::invalid_value(
                    "complex nanmean: axis out of bounds",
                )));
            }
            let out_shape: Vec<usize> = shape
                .iter()
                .enumerate()
                .filter_map(|(d, &s)| if d == ax { None } else { Some(s) })
                .collect();
            let axis_len = shape[ax];
            let flat: Vec<Complex<T>> = fa.iter().copied().collect();
            let mut strides = vec![1usize; ndim];
            for d in (0..ndim.saturating_sub(1)).rev() {
                strides[d] = strides[d + 1] * shape[d + 1];
            }
            let out_dims: Vec<usize> = (0..ndim).filter(|&d| d != ax).collect();
            let out_extents: Vec<usize> = out_dims.iter().map(|&d| shape[d]).collect();
            let out_n: usize = out_extents.iter().product::<usize>().max(1);
            let mut out: Vec<Complex<T>> = Vec::with_capacity(out_n);
            for lin in 0..out_n {
                let mut rem = lin;
                let mut base = 0usize;
                for (i, &d) in out_dims.iter().enumerate() {
                    let ext = out_extents[i];
                    let c = rem % ext;
                    rem /= ext;
                    base += c * strides[d];
                }
                let lane: Vec<Complex<T>> = (0..axis_len)
                    .map(|k| flat[base + k * strides[ax]])
                    .collect();
                out.push(mean_of(&lane));
            }
            let rd = ArrayD::<Complex<T>>::from_vec(IxDyn::new(&out_shape), out)
                .map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, rd)
        }
    }
}

macro_rules! bind_nan_reduction {
    ($name:ident, $ferr_path:path) => {
        #[pyfunction]
        #[pyo3(signature = (a, axis = None))]
        pub fn $name<'py>(
            py: Python<'py>,
            a: &Bound<'py, PyAny>,
            axis: Option<usize>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let arr = as_ndarray(py, a)?;
            let dt = dtype_name(&arr)?;
            let arr = if matches!(dt.as_str(), "float64" | "float32" | "f64" | "f32") {
                arr
            } else {
                coerce_dtype(py, &arr, "float64")?
            };
            let dt = dtype_name(&arr)?;
            Ok(match_dtype_float!(dt.as_str(), T => {
                let view: PyReadonlyArrayDyn<T> = arr.extract()?;
                let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
                let r = $ferr_path(&fa, axis).map_err(ferr_to_pyerr)?;
                r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
            }))
        }
    };
    // Complex-loop form: a complex input dispatches to `$cplx` (numpy COMPUTES
    // these as complex, NaN->identity), keeping the input width, BEFORE the f64
    // coercion can drop the imaginary part (R-CODE-4). A datetime64/timedelta64
    // input dispatches to `$time` (compute-or-raise per numpy, #946) BEFORE the
    // same f64 coercion — the silent-float corruption fix for the nan-aware
    // reductions (`fr.nansum(td)`/`fr.nanmax(td)` returned a bare float).
    ($name:ident, $ferr_path:path, complex = $cplx:expr, time = $time:expr) => {
        #[pyfunction]
        #[pyo3(signature = (a, axis = None))]
        pub fn $name<'py>(
            py: Python<'py>,
            a: &Bound<'py, PyAny>,
            axis: Option<usize>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let arr = as_ndarray(py, a)?;
            if crate::datetime::is_time_array(&arr)? {
                return ($time)(py, &arr, axis);
            }
            let dt = dtype_name(&arr)?;
            match dt.as_str() {
                "complex128" | "c16" => return ($cplx)(py, &arr, axis, false),
                "complex64" | "c8" => return ($cplx)(py, &arr, axis, true),
                _ => {}
            }
            let arr = if matches!(dt.as_str(), "float64" | "float32" | "f64" | "f32") {
                arr
            } else {
                coerce_dtype(py, &arr, "float64")?
            };
            let dt = dtype_name(&arr)?;
            Ok(match_dtype_float!(dt.as_str(), T => {
                let view: PyReadonlyArrayDyn<T> = arr.extract()?;
                let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
                let r = $ferr_path(&fa, axis).map_err(ferr_to_pyerr)?;
                r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
            }))
        }
    };
}

/// Complex `nansum` arm: width-keyed [`complex_nan_fold_dispatch`] (`is_c64`
/// selects `f32` vs `f64` components).
fn nansum_complex<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    axis: Option<usize>,
    is_c64: bool,
) -> PyResult<Bound<'py, PyAny>> {
    if is_c64 {
        complex_nan_fold_dispatch::<f32>(py, arr, axis, ComplexNanFold::Sum)
    } else {
        complex_nan_fold_dispatch::<f64>(py, arr, axis, ComplexNanFold::Sum)
    }
}

/// Complex `nanprod` arm: width-keyed [`complex_nan_fold_dispatch`].
fn nanprod_complex<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    axis: Option<usize>,
    is_c64: bool,
) -> PyResult<Bound<'py, PyAny>> {
    if is_c64 {
        complex_nan_fold_dispatch::<f32>(py, arr, axis, ComplexNanFold::Prod)
    } else {
        complex_nan_fold_dispatch::<f64>(py, arr, axis, ComplexNanFold::Prod)
    }
}

/// Complex `nanmean` arm: width-keyed [`complex_nanmean_dispatch`].
fn nanmean_complex<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    axis: Option<usize>,
    is_c64: bool,
) -> PyResult<Bound<'py, PyAny>> {
    if is_c64 {
        complex_nanmean_dispatch::<f32>(py, arr, axis)
    } else {
        complex_nanmean_dispatch::<f64>(py, arr, axis)
    }
}

/// Complex `nanmin`/`nanmax` over a width `T`: lexicographic `(real, then imag)`
/// extremum over the NON-NaN-complex elements per lane (numpy SKIPS any complex
/// with a NaN part — `np.nanmax([3+4j,nan+0j,-5+12j]) == (3+4j)` lexicographically;
/// `np.nanmin` of the same is `(-5+12j)`, live numpy 2.4.4). Reuses
/// [`cplx_is_nan`] to drop NaN-complex, then picks the lexicographic winner. An
/// all-NaN lane has no finite element; numpy returns `nan+nanj` there (a
/// RuntimeWarning, not a raise) — matched by emitting `NaN + NaN*i`. Width
/// preserved (c64->c64, c128->c128).
fn complex_nan_minmax_dispatch<'py, T>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    axis: Option<usize>,
    want_max: bool,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default + ReductionReal + PartialOrd,
    Complex<T>: ferray_core::Element + numpy::Element,
{
    let fa: ArrayD<Complex<T>> = complex_pyarray_to_ferray::<T>(arr)?;
    let pick = |lane: &[Complex<T>]| -> Complex<T> {
        // Index of the lexicographic extremum among the non-NaN-complex values.
        let mut best: Option<usize> = None;
        for (i, z) in lane.iter().enumerate() {
            if cplx_is_nan(z) {
                continue;
            }
            match best {
                None => best = Some(i),
                Some(b) => {
                    // `cplx_reduce_wins` with NEITHER side NaN reduces to the
                    // pure lexicographic comparison (NaN already filtered out).
                    if cplx_reduce_wins(z, &lane[b], want_max) {
                        best = Some(i);
                    }
                }
            }
        }
        match best {
            Some(i) => lane[i],
            None => {
                let nan = T::NAN;
                Complex::new(nan, nan)
            }
        }
    };
    match axis {
        None => {
            let vals: Vec<Complex<T>> = fa.iter().copied().collect();
            if vals.is_empty() {
                return Err(ferr_to_pyerr(ferray_core::FerrayError::invalid_value(
                    "zero-size array to reduction operation",
                )));
            }
            let r = pick(&vals);
            let rd =
                ArrayD::<Complex<T>>::from_vec(IxDyn::new(&[]), vec![r]).map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, rd)
        }
        Some(ax) => {
            let (out_shape, vals) = cplx_axis_lanes(&fa, ax, |lane| pick(lane))?;
            let rd = ArrayD::<Complex<T>>::from_vec(IxDyn::new(&out_shape), vals)
                .map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, rd)
        }
    }
}

/// Complex `nanmin` arm: width-keyed [`complex_nan_minmax_dispatch`].
fn nanmin_complex<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    axis: Option<usize>,
    is_c64: bool,
) -> PyResult<Bound<'py, PyAny>> {
    if is_c64 {
        complex_nan_minmax_dispatch::<f32>(py, arr, axis, false)
    } else {
        complex_nan_minmax_dispatch::<f64>(py, arr, axis, false)
    }
}

/// Complex `nanmax` arm: width-keyed [`complex_nan_minmax_dispatch`].
fn nanmax_complex<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    axis: Option<usize>,
    is_c64: bool,
) -> PyResult<Bound<'py, PyAny>> {
    if is_c64 {
        complex_nan_minmax_dispatch::<f32>(py, arr, axis, true)
    } else {
        complex_nan_minmax_dispatch::<f64>(py, arr, axis, true)
    }
}

// datetime64/timedelta64 routing for the nan-aware reductions (#946),
// compute-or-raise per the live numpy 2.4 contract (R-CHAR-3):
//   nansum(td)->td (NaT propagates, no skip); nansum(dt) RAISES.
//   nanmin/nanmax(td)->td, (dt)->datetime; NaT skipped, all-NaT -> NaT.
//   nanmean(td)->td (NaT skipped, mean); nanmean(dt) RAISES.
//   nanprod(td/dt) RAISES.
// Each delegates the RAISE cases to numpy so its exact `UFuncTypeError`
// surfaces (R-DEV-2); the compute cases use the int64-tick transport.
fn nansum_time<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    crate::datetime::time_nan_reduce(py, arr, crate::datetime::TimeReduce::Sum, axis)
}

fn nanmin_time<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    crate::datetime::time_nan_reduce(py, arr, crate::datetime::TimeReduce::Min, axis)
}

fn nanmax_time<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    crate::datetime::time_nan_reduce(py, arr, crate::datetime::TimeReduce::Max, axis)
}

/// nanmean(td) skips NaT and means the remaining ticks (timedelta); nanmean(dt)
/// and an all-NaT timedelta delegate to numpy so its exact result/exception
/// surfaces (R-DEV-2). Routes through numpy directly: this op's compute path is
/// not one of the #946 pins, but the time-array branch still eliminates the
/// prior silent-float coercion by deferring to numpy's own datetime kernel.
fn nanmean_time<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let kwargs = pyo3::types::PyDict::new(py);
    if let Some(ax) = axis {
        kwargs.set_item("axis", ax)?;
    }
    np.call_method("nanmean", (arr,), Some(&kwargs))
}

/// nanprod over a datetime64/timedelta64 array RAISES in numpy (no `prod` over
/// time dtypes) — delegate so numpy's exact `UFuncTypeError` surfaces.
fn nanprod_time<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let kwargs = pyo3::types::PyDict::new(py);
    if let Some(ax) = axis {
        kwargs.set_item("axis", ax)?;
    }
    np.call_method("nanprod", (arr,), Some(&kwargs))
}

bind_nan_reduction!(
    nansum,
    ferray_stats::nansum,
    complex = nansum_complex,
    time = nansum_time
);
bind_nan_reduction!(
    nanmean,
    ferray_stats::nanmean,
    complex = nanmean_complex,
    time = nanmean_time
);
bind_nan_reduction!(
    nanmin,
    ferray_stats::nanmin,
    complex = nanmin_complex,
    time = nanmin_time
);
bind_nan_reduction!(
    nanmax,
    ferray_stats::nanmax,
    complex = nanmax_complex,
    time = nanmax_time
);
bind_nan_reduction!(
    nanprod,
    ferray_stats::nanprod,
    complex = nanprod_complex,
    time = nanprod_time
);

/// Complex `nanvar`/`nanstd` over a width `T`: the REAL `mean(|x - mean|^2)` over
/// the NON-NaN-complex elements per lane, where `|z|^2 = re^2 + im^2` (numpy's
/// `(x*conj(x)).real`, matching the plain-complex `complex_var_std_dispatch`).
/// `ddof` divides by `count - ddof`. numpy SKIPS NaN-complex and returns a REAL
/// result on the COMPLEX data — `np.nanvar([3+4j,1-2j,-5+12j]) == 44.444`, NOT
/// the variance of the real parts (live numpy 2.4.4). Result element type `R` is
/// the real width following the input (c64->f32, c128->f64). An all-NaN lane (or
/// `count <= ddof`) yields `NaN` (numpy returns nan + RuntimeWarning, not a raise).
fn complex_nan_var_std_dispatch<'py, T>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    axis: Option<usize>,
    ddof: usize,
    want_std: bool,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default + ReductionReal + PartialOrd,
    ArrayD<T>: IntoNumPy<T, IxDyn>,
    Complex<T>: ferray_core::Element + numpy::Element,
{
    let fa: ArrayD<Complex<T>> = complex_pyarray_to_ferray::<T>(arr)?;
    let shape: Vec<usize> = fa.shape().to_vec();
    let ndim = shape.len();

    // One REAL variance value over the non-NaN-complex elements of a lane.
    let var_of = |vals: &[Complex<T>]| -> T {
        let mut mre = T::ZERO;
        let mut mim = T::ZERO;
        let mut cnt = 0usize;
        for z in vals {
            if !complex_is_nan(z) {
                mre = mre + z.re;
                mim = mim + z.im;
                cnt += 1;
            }
        }
        if cnt == 0 || cnt <= ddof {
            return T::NAN;
        }
        let scl = T::ONE / T::from_usize(cnt);
        let (mre, mim) = (mre * scl, mim * scl);
        let mut sq_sum = T::ZERO;
        for z in vals {
            if complex_is_nan(z) {
                continue;
            }
            let dr = z.re - mre;
            let di = z.im - mim;
            // `|x - mean|^2` as `re^2 + im^2` (numpy `(x*conj(x)).real`).
            sq_sum = sq_sum + dr * dr + di * di;
        }
        let v = sq_sum / T::from_usize(cnt - ddof);
        if want_std { v.sqrt() } else { v }
    };

    match axis {
        None => {
            let vals: Vec<Complex<T>> = fa.iter().copied().collect();
            let v = var_of(&vals);
            let r = ArrayD::<T>::from_vec(IxDyn::new(&[]), vec![v]).map_err(ferr_to_pyerr)?;
            Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
        }
        Some(ax) => {
            if ax >= ndim {
                return Err(ferr_to_pyerr(ferray_core::FerrayError::invalid_value(
                    "complex nanvar/nanstd: axis out of bounds",
                )));
            }
            let out_shape: Vec<usize> = shape
                .iter()
                .enumerate()
                .filter_map(|(d, &s)| if d == ax { None } else { Some(s) })
                .collect();
            let axis_len = shape[ax];
            let flat: Vec<Complex<T>> = fa.iter().copied().collect();
            let mut strides = vec![1usize; ndim];
            for d in (0..ndim.saturating_sub(1)).rev() {
                strides[d] = strides[d + 1] * shape[d + 1];
            }
            let out_dims: Vec<usize> = (0..ndim).filter(|&d| d != ax).collect();
            let out_extents: Vec<usize> = out_dims.iter().map(|&d| shape[d]).collect();
            let out_n: usize = out_extents.iter().product::<usize>().max(1);
            let mut out: Vec<T> = Vec::with_capacity(out_n);
            for lin in 0..out_n {
                let mut rem = lin;
                let mut base = 0usize;
                for (i, &d) in out_dims.iter().enumerate() {
                    let ext = out_extents[i];
                    let c = rem % ext;
                    rem /= ext;
                    base += c * strides[d];
                }
                let lane: Vec<Complex<T>> = (0..axis_len)
                    .map(|k| flat[base + k * strides[ax]])
                    .collect();
                out.push(var_of(&lane));
            }
            let r = ArrayD::<T>::from_vec(IxDyn::new(&out_shape), out).map_err(ferr_to_pyerr)?;
            Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
        }
    }
}

/// `numpy.nanvar(a, axis=None, ddof=0)`.
#[pyfunction]
#[pyo3(signature = (a, axis = None, ddof = 0))]
pub fn nanvar<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
    ddof: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    // Complex nanvar: REAL variance over the non-NaN-complex data (NOT var of the
    // real parts). MUST branch BEFORE the float64 coercion that drops the
    // imaginary part (R-CODE-4 corruption: `np.nanvar([3+4j,1-2j,-5+12j])` is
    // 44.444, ferray returned 11.555 = var of the real parts).
    match dt.as_str() {
        "complex128" | "c16" => {
            return complex_nan_var_std_dispatch::<f64>(py, &arr, axis, ddof, false);
        }
        "complex64" | "c8" => {
            return complex_nan_var_std_dispatch::<f32>(py, &arr, axis, ddof, false);
        }
        _ => {}
    }
    let arr = if matches!(dt.as_str(), "float64" | "float32" | "f64" | "f32") {
        arr
    } else {
        coerce_dtype(py, &arr, "float64")?
    };
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r = ferray_stats::nanvar(&fa, axis, ddof).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.nanstd(a, axis=None, ddof=0)`.
#[pyfunction]
#[pyo3(signature = (a, axis = None, ddof = 0))]
pub fn nanstd<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
    ddof: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    // Complex nanstd: sqrt of the REAL complex nanvar (same R-CODE-4 guard).
    match dt.as_str() {
        "complex128" | "c16" => {
            return complex_nan_var_std_dispatch::<f64>(py, &arr, axis, ddof, true);
        }
        "complex64" | "c8" => {
            return complex_nan_var_std_dispatch::<f32>(py, &arr, axis, ddof, true);
        }
        _ => {}
    }
    let arr = if matches!(dt.as_str(), "float64" | "float32" | "f64" | "f32") {
        arr
    } else {
        coerce_dtype(py, &arr, "float64")?
    };
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r = ferray_stats::nanstd(&fa, axis, ddof).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// Complex `nanmedian` over a width `T`: lexicographically sort the NON-NaN-complex
/// elements of each lane and take the middle (odd count) or the complex average of
/// the two middles (even count). numpy SKIPS NaN-complex then medians the rest —
/// `np.nanmedian([3+4j,1-2j,-5+12j]) == (1-2j)`; with a NaN-complex present
/// `np.nanmedian([3+4j,nan+0j,-5+12j,nan+nanj]) == (-1+8j)` (the even-count
/// average of the two non-NaN values, live numpy 2.4.4). An all-NaN lane yields
/// `nan+nanj` (numpy's empty-after-skip result). Width preserved (`Complex<T>`).
fn complex_nanmedian_dispatch<'py, T>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default + ReductionReal + PartialOrd,
    Complex<T>: ferray_core::Element + numpy::Element,
{
    let fa: ArrayD<Complex<T>> = complex_pyarray_to_ferray::<T>(arr)?;
    let shape: Vec<usize> = fa.shape().to_vec();
    let ndim = shape.len();
    let half = T::ONE / (T::ONE + T::ONE);

    let median_of = |vals: &[Complex<T>]| -> Complex<T> {
        // Drop every NaN-complex first (numpy nan-skip), then median the rest.
        let mut sorted: Vec<Complex<T>> = vals
            .iter()
            .copied()
            .filter(|z| !complex_is_nan(z))
            .collect();
        let n = sorted.len();
        if n == 0 {
            let nan = T::NAN;
            return Complex::new(nan, nan);
        }
        sorted.sort_by(complex_lex_order);
        if n % 2 == 1 {
            sorted[n / 2]
        } else {
            let lo = sorted[n / 2 - 1];
            let hi = sorted[n / 2];
            Complex::new((lo.re + hi.re) * half, (lo.im + hi.im) * half)
        }
    };

    match axis {
        None => {
            let vals: Vec<Complex<T>> = fa.iter().copied().collect();
            let r = median_of(&vals);
            let rd =
                ArrayD::<Complex<T>>::from_vec(IxDyn::new(&[]), vec![r]).map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, rd)
        }
        Some(ax) => {
            if ax >= ndim {
                return Err(ferr_to_pyerr(ferray_core::FerrayError::invalid_value(
                    "complex nanmedian: axis out of bounds",
                )));
            }
            let (out_shape, vals) = cplx_axis_lanes(&fa, ax, |lane| median_of(lane))?;
            let rd = ArrayD::<Complex<T>>::from_vec(IxDyn::new(&out_shape), vals)
                .map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, rd)
        }
    }
}

/// `numpy.nanmedian(a, axis=None)`.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn nanmedian<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    // datetime64/timedelta64 (#946): `nanmedian(timedelta)->timedelta` (NaT
    // skipped, all-NaT lane -> NaT); `nanmedian(datetime)` RAISES numpy's
    // `UFuncTypeError`. Branch BEFORE the float64 coercion (silent-float fix).
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::time_median(py, &arr, axis, true);
    }
    let dt = dtype_name(&arr)?;
    // Complex nanmedian: lexicographic median over the non-NaN-complex data. MUST
    // branch BEFORE the float64 coercion that drops the imaginary part (R-CODE-4:
    // `np.nanmedian([3+4j,1-2j,-5+12j])` is `(1-2j)`, ferray returned real `1.0`).
    match dt.as_str() {
        "complex128" | "c16" => return complex_nanmedian_dispatch::<f64>(py, &arr, axis),
        "complex64" | "c8" => return complex_nanmedian_dispatch::<f32>(py, &arr, axis),
        _ => {}
    }
    let arr = if matches!(dt.as_str(), "float64" | "float32" | "f64" | "f32") {
        arr
    } else {
        coerce_dtype(py, &arr, "float64")?
    };
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r = ferray_stats::nanmedian(&fa, axis).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.nanargmin(a, axis=None)`.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn nanargmin<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    let arr = if matches!(dt.as_str(), "float64" | "float32" | "f64" | "f32") {
        arr
    } else {
        coerce_dtype(py, &arr, "float64")?
    };
    let dt = dtype_name(&arr)?;
    let r: ArrayD<u64> = match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        ferray_stats::nanargmin(&fa, axis).map_err(ferr_to_pyerr)?
    });
    u64_arrd_to_i64_pyarray(py, r)
}

/// `numpy.nanargmax(a, axis=None)`.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn nanargmax<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    let arr = if matches!(dt.as_str(), "float64" | "float32" | "f64" | "f32") {
        arr
    } else {
        coerce_dtype(py, &arr, "float64")?
    };
    let dt = dtype_name(&arr)?;
    let r: ArrayD<u64> = match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        ferray_stats::nanargmax(&fa, axis).map_err(ferr_to_pyerr)?
    });
    u64_arrd_to_i64_pyarray(py, r)
}

// ---------------------------------------------------------------------------
// Cumulative reductions (cumsum, cumprod, diff)
// ---------------------------------------------------------------------------

/// `numpy.cumsum(a, axis=None)`.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn cumsum<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    // datetime64/timedelta64 (REQ-4, #944): `cumsum(timedelta)->timedelta`
    // (NaT propagates forward); `cumsum(datetime)` RAISES numpy's
    // `UFuncTypeError`.
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::time_reduce(py, &arr, crate::datetime::TimeReduce::Cumsum, axis);
    }
    // Complex cumsum folds via the complex `ReduceAcc`, keeping width and shape
    // — numpy: `np.cumsum([1+2j,3+4j])==[1+2j,4+6j]`, complex64 stays complex64.
    if is_complex_dtype(dt.as_str()) {
        return complex_fold_dispatch(py, &arr, dt.as_str(), axis, ComplexFold::CumSum);
    }
    Ok(match_dtype_numeric!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        // numpy cumsum promotes narrow-int accumulators to the platform int
        // (fromnumeric.py:2853-2855); ferray_stats::cumsum now returns the
        // promoted dtype, so let the bound type follow it.
        let r = ferray_stats::cumsum(&fa, axis).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.cumprod(a, axis=None)`.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn cumprod<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    // Complex cumprod folds via the complex `ReduceAcc`, keeping width and shape
    // — numpy: `np.cumprod([1+2j,3+4j])==[1+2j,-5+10j]`.
    if is_complex_dtype(dt.as_str()) {
        return complex_fold_dispatch(py, &arr, dt.as_str(), axis, ComplexFold::CumProd);
    }
    Ok(match_dtype_numeric!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        // numpy cumprod promotes narrow-int accumulators to the platform int;
        // ferray_stats::cumprod now returns the promoted dtype.
        let r = ferray_stats::cumprod(&fa, axis).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.diff(a, n=1)` — discrete difference along the only axis.
/// ferray's diff is 1-D-only; multi-axis diff is deferred.
#[pyfunction]
#[pyo3(signature = (a, n = 1))]
pub fn diff<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, n: usize) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    // datetime64/timedelta64 diff -> timedelta64 (consecutive differences;
    // `datetime - datetime = timedelta`). Routed through the #947 time dispatch
    // (`crate::datetime::diff_time`) ahead of the real-only `match_dtype_numeric!`,
    // which would otherwise raise `TypeError` where numpy computes.
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::diff_time(py, &arr, n);
    }
    // Complex diff: numpy applies the SAME `a[1:] - a[:-1]` subtraction the real
    // path does (`numpy/lib/_function_base_impl.py:1538` `op = ... else
    // subtract`), dtype-preserving (c64->c64, c128->c128 — verified live, numpy
    // 2.4.5). The library `ferray_ufunc::diff` is generic over
    // `T: Sub + Copy`, which `Complex<f32/f64>` satisfy, so it folds complex
    // directly with NO imag discard (R-CODE-4). `match_dtype_numeric!` has no
    // complex arm, so a complex input would otherwise raise `TypeError`.
    if is_complex_dtype(dt.as_str()) {
        return complex_diff_dispatch(py, &arr, dt.as_str(), n);
    }
    Ok(match_dtype_numeric!(dt.as_str(), T => {
        let view: PyReadonlyArray1<T> = arr.extract()?;
        let fa: Array1<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: Array1<T> = ferray_ufunc::diff(&fa, n).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// Complex arm for `diff` (#935): n-th successive difference of a 1-D complex
/// array via the `Sub`-bounded `ferray_ufunc::diff`, keeping the input width.
/// numpy: `np.diff([1+2j,-3+4j,0j,2-1j]) == [-4+2j, 3-4j, 2-1j]` (live).
fn complex_diff_dispatch<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    dt: &str,
    n: usize,
) -> PyResult<Bound<'py, PyAny>> {
    macro_rules! diff_arm {
        ($Tc:ty) => {{
            let fa: ArrayD<Complex<$Tc>> = complex_pyarray_to_ferray::<$Tc>(arr)?;
            // `diff` is 1-D-only; rebuild the (already 1-D) array as `Array1` from
            // its flat data, mirroring the complex `correlate` arm's `from_vec`
            // (no `into_dimensionality` on ferray's `Array`).
            let data: Vec<Complex<$Tc>> = fa.iter().copied().collect();
            let len = data.len();
            let fa1: Array1<Complex<$Tc>> =
                Array1::from_vec(Ix1::new([len]), data).map_err(ferr_to_pyerr)?;
            let r: Array1<Complex<$Tc>> = ferray_ufunc::diff(&fa1, n).map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, r.into_dyn())
        }};
    }
    match dt {
        "complex128" | "c16" => diff_arm!(f64),
        "complex64" | "c8" => diff_arm!(f32),
        other => Err(PyTypeError::new_err(format!(
            "complex_diff_dispatch: expected a complex dtype, got {other:?}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Sort / argsort
// ---------------------------------------------------------------------------

/// `numpy.sort(a, axis=-1)` — sorted copy.
#[pyfunction]
#[pyo3(signature = (a, axis = -1))]
pub fn sort<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_stats::SortKind;
    let arr = as_ndarray(py, a)?;
    // numpy.sort defaults to `axis=-1` (sort along the last axis); an explicit
    // `axis=None` flattens (numpy/_core/fromnumeric.py `sort` docstring). The
    // dispatch helpers below take `Option<usize>` where `None` == flatten, so
    // fold the negative/default axis into `[0, ndim)` here (`-1` -> last axis).
    let axis = norm_axis(py, &arr, axis)?;
    let dt = dtype_name(&arr)?;
    // datetime64/timedelta64 sort: by int64 tick, NaT last (REQ-3, #943).
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::sort_time(py, &arr, axis);
    }
    // Complex sort: lexicographic (real, then imag) with NaN-complex last,
    // width preserved — numpy: `np.sort([3+1j,1+2j,1+1j]) == [1+1j,1+2j,3+1j]`
    // (live). The real-only `match_dtype_orderable!` has no complex arm.
    match dt.as_str() {
        "complex128" | "c16" => return complex_sort_dispatch::<f64>(py, &arr, axis),
        "complex64" | "c8" => return complex_sort_dispatch::<f32>(py, &arr, axis),
        _ => {}
    }
    Ok(match_dtype_orderable!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = ferray_stats::sort(&fa, axis, SortKind::Quick)
            .map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.argsort(a, axis=-1)`.
#[pyfunction]
#[pyo3(signature = (a, axis = -1))]
pub fn argsort<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    // numpy.argsort defaults to `axis=-1` (per-lane indices along the last
    // axis); an explicit `axis=None` flattens. Fold the negative/default axis
    // into `[0, ndim)` so the dispatch helpers (which take `Option<usize>`,
    // `None` == flatten) match numpy's default.
    let axis = norm_axis(py, &arr, axis)?;
    let dt = dtype_name(&arr)?;
    // datetime64/timedelta64 argsort: int64 indices by tick, NaT last (REQ-3).
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::argsort_time(py, &arr, axis);
    }
    // Complex argsort: lexicographic-with-nan-last indices (int64) — numpy:
    // `np.argsort([1+2j,-3+4j,0+0j,2-1j]) == [1,2,0,3]` (live).
    match dt.as_str() {
        "complex128" | "c16" => return complex_argsort_dispatch::<f64>(py, &arr, axis),
        "complex64" | "c8" => return complex_argsort_dispatch::<f32>(py, &arr, axis),
        _ => {}
    }
    let r: ArrayD<u64> = match_dtype_orderable!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        ferray_stats::argsort(&fa, axis).map_err(ferr_to_pyerr)?
    });
    u64_arrd_to_i64_pyarray(py, r)
}

/// `numpy.searchsorted(a, v, side="left")`.
#[pyfunction]
#[pyo3(signature = (a, v, side = "left"))]
pub fn searchsorted<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    v: &Bound<'py, PyAny>,
    side: &str,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_stats::Side;
    let s = match side {
        "left" => Side::Left,
        "right" => Side::Right,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "side must be 'left' or 'right', got {other:?}"
            )));
        }
    };
    let arr_a = as_ndarray(py, a)?;
    let dt = dtype_name(&arr_a)?;
    // datetime64/timedelta64 searchsorted: int64 insertion points, NaT last
    // (REQ-3, #943). Gated ahead of the real coerce (which would mis-coerce the
    // datetime query). numpy returns a 0-d result for a scalar `v`.
    if crate::datetime::is_time_array(&arr_a)? {
        let v_scalar = !v.hasattr("__len__")?;
        return crate::datetime::searchsorted_time(py, &arr_a, v, side, v_scalar);
    }
    let arr_v = coerce_dtype(py, v, dt.as_str())?;
    // Complex `searchsorted`: lexicographic insertion points into the (assumed
    // sorted) complex array, matching numpy's lexicographic complex order
    // (`np.searchsorted([1,2,3], 1+1j) == 1`, live). `ferray_stats::searchsorted`
    // needs `T: PartialOrd`, which `Complex<T>` lacks, so compose with the #932
    // `cplx_sort_cmp` lexicographic order.
    // numpy returns a 0-d result when `v` is a scalar, a 1-d result otherwise.
    let v_scalar = !v.hasattr("__len__")?;
    match dt.as_str() {
        "complex128" | "c16" => {
            return complex_searchsorted_dispatch::<f64>(py, &arr_a, &arr_v, side, v_scalar);
        }
        "complex64" | "c8" => {
            return complex_searchsorted_dispatch::<f32>(py, &arr_a, &arr_v, side, v_scalar);
        }
        _ => {}
    }
    let r: Array1<u64> = match_dtype_orderable!(dt.as_str(), T => {
        let va: PyReadonlyArray1<T> = arr_a.extract()?;
        let vv: PyReadonlyArray1<T> = arr_v.extract()?;
        let fa: Array1<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
        let fv: Array1<T> = vv.as_ferray().map_err(ferr_to_pyerr)?;
        ferray_stats::searchsorted(&fa, &fv, s).map_err(ferr_to_pyerr)?
    });
    u64_arr1_to_i64_pyarray(py, r)
}

// ---------------------------------------------------------------------------
// Unique / count_nonzero
// ---------------------------------------------------------------------------

/// `numpy.unique(a)` — sorted unique values.
#[pyfunction]
pub fn unique<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    // datetime64/timedelta64 unique: sorted-unique ticks, NaT last (one kept)
    // (REQ-3, #943).
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::unique_time(py, &arr);
    }
    // Complex unique: lexicographic sort + dedup (NaN-complex last, one kept),
    // width preserved — numpy: `np.unique([1+2j,-3+4j,1+2j,2-1j]) ==
    // [-3+4j,1+2j,2-1j]` (live).
    match dt.as_str() {
        "complex128" | "c16" => return complex_unique_dispatch::<f64>(py, &arr),
        "complex64" | "c8" => return complex_unique_dispatch::<f32>(py, &arr),
        _ => {}
    }
    Ok(match_dtype_orderable!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: Array1<T> = ferray_stats::unique_values(&fa).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.count_nonzero(a, axis=None)`.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn count_nonzero<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    // datetime64/timedelta64 count_nonzero -> int (count of ticks != 0; NaT is
    // nonzero). Routed through the #947 time dispatch ahead of the real-only
    // `match_dtype_all_complex!`, which would otherwise raise `TypeError`.
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::count_nonzero_time(py, &arr, axis);
    }
    // Complex `count_nonzero`: numpy counts `z != 0` (`re != 0 || im != 0`).
    // `ferray_stats::count_nonzero` needs only `T: Element + PartialEq + Copy`,
    // which `Complex<T>` satisfies, so route through the DynMarshal seam (#933)
    // — `np.count_nonzero([3+4j,1-2j,-5+12j]) == 3` (verified live).
    let r: ArrayD<u64> = match_dtype_all_complex!(dt.as_str(), T => {
        let fa: ArrayD<T> = T::extract_dyn(&arr)?;
        ferray_stats::count_nonzero(&fa, axis).map_err(ferr_to_pyerr)?
    });
    u64_arrd_to_i64_pyarray(py, r)
}

// ---------------------------------------------------------------------------
// Set operations (1-D)
// ---------------------------------------------------------------------------

/// Which 1-D set operation a complex dispatch performs (lexicographic via #932).
#[derive(Clone, Copy)]
enum CplxSetOp {
    Union,
    Intersect,
    Diff,
    Xor,
}

macro_rules! bind_set_op {
    ($name:ident, $ferr_path:path, $kind:expr) => {
        #[pyfunction]
        #[pyo3(signature = (a, b, assume_unique = false))]
        pub fn $name<'py>(
            py: Python<'py>,
            a: &Bound<'py, PyAny>,
            b: &Bound<'py, PyAny>,
            assume_unique: bool,
        ) -> PyResult<Bound<'py, PyAny>> {
            let arr_a = as_ndarray(py, a)?;
            let dt = dtype_name(&arr_a)?;
            let arr_b = coerce_dtype(py, b, dt.as_str())?;
            // Complex set ops compose from the #932 `cplx_sort_cmp` lexicographic
            // order (numpy sorts complex set results lexicographically). The
            // `ferray_stats` set ops require `T: PartialOrd`, absent for complex.
            match dt.as_str() {
                "complex128" | "c16" => {
                    return complex_set_op_dispatch::<f64>(py, &arr_a, &arr_b, $kind);
                }
                "complex64" | "c8" => {
                    return complex_set_op_dispatch::<f32>(py, &arr_a, &arr_b, $kind);
                }
                _ => {}
            }
            Ok(match_dtype_orderable!(dt.as_str(), T => {
                let va: PyReadonlyArray1<T> = arr_a.extract()?;
                let vb: PyReadonlyArray1<T> = arr_b.extract()?;
                let fa: Array1<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
                let fb: Array1<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
                let r: Array1<T> = $ferr_path(&fa, &fb, assume_unique).map_err(ferr_to_pyerr)?;
                r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
            }))
        }
    };
}

bind_set_op!(union1d, ferray_stats::union1d, CplxSetOp::Union);
bind_set_op!(intersect1d, ferray_stats::intersect1d, CplxSetOp::Intersect);
bind_set_op!(setdiff1d, ferray_stats::setdiff1d, CplxSetOp::Diff);
bind_set_op!(setxor1d, ferray_stats::setxor1d, CplxSetOp::Xor);

/// `numpy.in1d(ar1, ar2)` — bool array of presence.
#[pyfunction]
#[pyo3(signature = (a, b, assume_unique = false))]
pub fn in1d<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    assume_unique: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let arr_a = as_ndarray(py, a)?;
    let dt = dtype_name(&arr_a)?;
    let arr_b = coerce_dtype(py, b, dt.as_str())?;
    Ok(match_dtype_orderable!(dt.as_str(), T => {
        let va: PyReadonlyArray1<T> = arr_a.extract()?;
        let vb: PyReadonlyArray1<T> = arr_b.extract()?;
        let fa: Array1<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
        let fb: Array1<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
        let r: Array1<bool> = ferray_stats::in1d(&fa, &fb, assume_unique).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.isin(element, test_elements)`.
#[pyfunction]
#[pyo3(signature = (element, test_elements, assume_unique = false))]
pub fn isin<'py>(
    py: Python<'py>,
    element: &Bound<'py, PyAny>,
    test_elements: &Bound<'py, PyAny>,
    assume_unique: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let arr_a = as_ndarray(py, element)?;
    let dt = dtype_name(&arr_a)?;
    let arr_b = coerce_dtype(py, test_elements, dt.as_str())?;
    // Complex `isin`: per-element membership of `element` in `test_elements`,
    // by exact complex equality (numpy `np.isin` over complex, verified live:
    // `np.isin([3+4j,1-2j,-5+12j], [1-2j]) == [F, T, F]`). `ferray_stats::isin`
    // needs `T: PartialOrd`; compose membership directly from complex equality.
    match dt.as_str() {
        "complex128" | "c16" => return complex_isin_dispatch::<f64>(py, &arr_a, &arr_b),
        "complex64" | "c8" => return complex_isin_dispatch::<f32>(py, &arr_a, &arr_b),
        _ => {}
    }
    Ok(match_dtype_orderable!(dt.as_str(), T => {
        let va: PyReadonlyArray1<T> = arr_a.extract()?;
        let vb: PyReadonlyArray1<T> = arr_b.extract()?;
        let fa: Array1<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
        let fb: Array1<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
        let r: Array1<bool> = ferray_stats::isin(&fa, &fb, assume_unique).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

// ---------------------------------------------------------------------------
// where_
// ---------------------------------------------------------------------------

/// `numpy.partition(a, kth)` — 1-D partial sort: kth element in its
/// final sorted position, smaller before, larger after.
#[pyfunction]
pub fn partition<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    kth: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    // datetime64/timedelta64 partition: by int64 tick, NaT last (REQ-3, #943).
    // A full ordered copy satisfies the partition postcondition for any kth.
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::partition_time(py, &arr);
    }
    // Complex `partition`: numpy places the kth element of the lexicographically
    // sorted complex array in its final position. ferray's `partition` needs
    // `T: PartialOrd` (absent for `Complex<T>`); a full lexicographic sort
    // satisfies the partition postcondition exactly (the kth element is in place,
    // smaller-before / larger-after), matching `np.partition` (verified live).
    match dt.as_str() {
        "complex128" | "c16" => return complex_partition_dispatch::<f64>(py, &arr, kth),
        "complex64" | "c8" => return complex_partition_dispatch::<f32>(py, &arr, kth),
        _ => {}
    }
    Ok(match_dtype_orderable!(dt.as_str(), T => {
        let view: PyReadonlyArray1<T> = arr.extract()?;
        let fa: Array1<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: Array1<T> = ferray_stats::partition(&fa, kth).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.argpartition(a, kth)`.
#[pyfunction]
pub fn argpartition<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    kth: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    // datetime64/timedelta64 argpartition: stable argsort by int64 tick, NaT
    // last (REQ-3, #943) — a full ordered index satisfies argpartition for kth.
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::argpartition_time(py, &arr);
    }
    let r: Array1<u64> = match_dtype_orderable!(dt.as_str(), T => {
        let view: PyReadonlyArray1<T> = arr.extract()?;
        let fa: Array1<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        ferray_stats::argpartition(&fa, kth).map_err(ferr_to_pyerr)?
    });
    u64_arr1_to_i64_pyarray(py, r)
}

/// `numpy.lexsort(keys)` — indirect lexicographic sort. `keys` is a
/// sequence of 1-D arrays; the LAST key is the primary sort key.
#[pyfunction]
pub fn lexsort<'py>(py: Python<'py>, keys: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let list = keys.cast::<pyo3::types::PyList>()?;
    if list.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "lexsort: need at least one key array",
        ));
    }
    // Sniff dtype from the first key; coerce all to match.
    let first = as_ndarray(py, &list.get_item(0)?)?;
    let dt = dtype_name(&first)?;
    let r: Array1<u64> = match_dtype_orderable!(dt.as_str(), T => {
        let mut owned: Vec<Array1<T>> = Vec::with_capacity(list.len());
        for item in list.iter() {
            let coerced = coerce_dtype(py, &item, dt.as_str())?;
            let view: PyReadonlyArray1<T> = coerced.extract()?;
            owned.push(view.as_ferray().map_err(ferr_to_pyerr)?);
        }
        let refs: Vec<&Array1<T>> = owned.iter().collect();
        ferray_stats::lexsort(&refs).map_err(ferr_to_pyerr)?
    });
    u64_arr1_to_i64_pyarray(py, r)
}

/// `numpy.unique(a, return_index=False, return_inverse=False, return_counts=False)`.
///
/// When extra return flags are requested, NumPy returns a tuple. We
/// match that here — passing all three flags returns a 4-tuple
/// `(values, indices, inverse, counts)`.
#[pyfunction]
#[pyo3(signature = (a, return_index = false, return_inverse = false, return_counts = false))]
pub fn unique_extended<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    return_index: bool,
    return_inverse: bool,
    return_counts: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_orderable!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let result = ferray_stats::unique(&fa, return_index, return_inverse, return_counts)
            .map_err(ferr_to_pyerr)?;
        let values_py = result.values.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
        let mut tuple_items: Vec<Bound<'py, pyo3::types::PyAny>> = vec![values_py];
        if let Some(idx) = result.indices {
            tuple_items.push(u64_arr1_to_i64_pyarray(py, idx)?);
        }
        if let Some(inv) = result.inverse {
            tuple_items.push(u64_arr1_to_i64_pyarray(py, inv)?);
        }
        if let Some(counts) = result.counts {
            tuple_items.push(u64_arr1_to_i64_pyarray(py, counts)?);
        }
        if tuple_items.len() == 1 {
            tuple_items.into_iter().next().unwrap()
        } else {
            pyo3::types::PyTuple::new(py, tuple_items)?.into_any()
        }
    }))
}

// ---------------------------------------------------------------------------
// histogram family (#704)
// ---------------------------------------------------------------------------

/// Parse the `bins` argument into a ferray `Bins<T>`. Accepts an int
/// (number of equal-width bins) or a sequence of float edges.
fn parse_bins<'py, T>(bins: &Bound<'py, PyAny>) -> PyResult<ferray_stats::Bins<T>>
where
    T: 'py,
    f64: Into<T>,
{
    if let Ok(n) = bins.extract::<usize>() {
        return Ok(ferray_stats::Bins::Count(n));
    }
    let edges_f64: Vec<f64> = bins.extract()?;
    let edges: Vec<T> = edges_f64.into_iter().map(Into::into).collect();
    Ok(ferray_stats::Bins::Edges(edges))
}

/// `numpy.histogram(a, bins=10, range=None, density=False)` →
/// `(hist, bin_edges)`. Histograms float input only (integer auto-promoted).
#[pyfunction]
#[pyo3(signature = (a, bins = 10usize, range = None, density = false))]
pub fn histogram<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    bins: usize,
    range: Option<(f64, f64)>,
    density: bool,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::array::aliases::Array1;
    let arr = as_ndarray(py, a)?;
    // numpy.histogram on complex input raises IndexError (the complex->real cast
    // in `_search_sorted_inclusive` produces out-of-range bin indices), not a
    // silent real-part histogram (numpy/lib/_histograms_impl.py:853 casts complex
    // to intp). Reject complex to match numpy's observable failure (R-CODE-4 /
    // R-DEV-2: preserve numpy's exception type).
    if is_complex_dtype(dtype_name(&arr)?.as_str()) {
        return Err(pyo3::exceptions::PyIndexError::new_err(
            "index out of bounds for axis 0: numpy.histogram does not support complex input",
        ));
    }
    // datetime64/timedelta64 (#946): numpy's `histogram` RAISES on time inputs
    // (`DTypePromotionError`: a timedelta/datetime cannot be promoted with the
    // float64 bin edges — verified live). Delegate so numpy's EXACT exception
    // surfaces (R-DEV-2), eliminating the prior silent-float corruption
    // (`fr.histogram(td)` returned bogus float bin counts/edges).
    if crate::datetime::is_time_array(&arr)? {
        let np = py.import("numpy")?;
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("bins", bins)?;
        if let Some(r) = range {
            kwargs.set_item("range", r)?;
        }
        kwargs.set_item("density", density)?;
        return np.call_method("histogram", (&arr,), Some(&kwargs));
    }
    let arr_f64 = coerce_dtype(py, &arr, "float64")?;
    let view: PyReadonlyArray1<f64> = arr_f64.extract()?;
    let fa: Array1<f64> = view.as_ferray().map_err(ferr_to_pyerr)?;
    let bins_e = ferray_stats::Bins::Count(bins);
    let (h, e) = ferray_stats::histogram(&fa, bins_e, range, density).map_err(ferr_to_pyerr)?;
    let h_py = h.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    let e_py = e.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    Ok(pyo3::types::PyTuple::new(py, [h_py, e_py])?.into_any())
}

/// `numpy.histogram_bin_edges(a, bins=10, range=None)`.
#[pyfunction]
#[pyo3(signature = (a, bins = 10usize, range = None))]
pub fn histogram_bin_edges<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    bins: usize,
    range: Option<(f64, f64)>,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::array::aliases::Array1;
    let arr = as_ndarray(py, a)?;
    let arr_f64 = coerce_dtype(py, &arr, "float64")?;
    let view: PyReadonlyArray1<f64> = arr_f64.extract()?;
    let fa: Array1<f64> = view.as_ferray().map_err(ferr_to_pyerr)?;
    let bins_e = ferray_stats::Bins::Count(bins);
    let edges: Array1<f64> =
        ferray_stats::histogram_bin_edges(&fa, bins_e, range).map_err(ferr_to_pyerr)?;
    Ok(edges.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.histogram2d(x, y, bins=10)` → `(hist, x_edges, y_edges)`.
#[pyfunction]
#[pyo3(signature = (x, y, bins = (10usize, 10usize)))]
pub fn histogram2d<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    y: &Bound<'py, PyAny>,
    bins: (usize, usize),
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::array::aliases::Array1;
    let arr_x = coerce_dtype(py, x, "float64")?;
    let arr_y = coerce_dtype(py, y, "float64")?;
    let vx: PyReadonlyArray1<f64> = arr_x.extract()?;
    let vy: PyReadonlyArray1<f64> = arr_y.extract()?;
    let fx: Array1<f64> = vx.as_ferray().map_err(ferr_to_pyerr)?;
    let fy: Array1<f64> = vy.as_ferray().map_err(ferr_to_pyerr)?;
    let (h, ex, ey) = ferray_stats::histogram2d(&fx, &fy, bins).map_err(ferr_to_pyerr)?;
    let h_py = h.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    let ex_py = ex.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    let ey_py = ey.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    Ok(pyo3::types::PyTuple::new(py, [h_py, ex_py, ey_py])?.into_any())
}

/// `numpy.histogramdd(sample, bins)` → `(hist, edges)`.
#[pyfunction]
pub fn histogramdd<'py>(
    py: Python<'py>,
    sample: &Bound<'py, PyAny>,
    bins: Vec<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::array::aliases::Array2;
    let arr = coerce_dtype(py, sample, "float64")?;
    let view: numpy::PyReadonlyArray2<f64> = arr.extract()?;
    let fa: Array2<f64> = view.as_ferray().map_err(ferr_to_pyerr)?;
    let (h, edges) = ferray_stats::histogramdd(&fa, &bins).map_err(ferr_to_pyerr)?;
    let h_py = h.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    let edges_py: Vec<Bound<'py, PyAny>> = edges
        .into_iter()
        .map(|e| Ok(e.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()))
        .collect::<PyResult<Vec<_>>>()?;
    let edges_list = pyo3::types::PyList::new(py, edges_py)?.into_any();
    Ok(pyo3::types::PyTuple::new(py, [h_py, edges_list])?.into_any())
}

/// `numpy.bincount(x, weights=None, minlength=0)`.
#[pyfunction]
#[pyo3(signature = (x, weights = None, minlength = 0))]
pub fn bincount<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    weights: Option<&Bound<'py, PyAny>>,
    minlength: usize,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::array::aliases::Array1;
    let arr_x = coerce_dtype(py, x, "uint64")?;
    let vx: PyReadonlyArray1<u64> = arr_x.extract()?;
    let fx: Array1<u64> = vx.as_ferray().map_err(ferr_to_pyerr)?;
    if let Some(w) = weights {
        let arr_w = coerce_dtype(py, w, "float64")?;
        let vw: PyReadonlyArray1<f64> = arr_w.extract()?;
        let fw: Array1<f64> = vw.as_ferray().map_err(ferr_to_pyerr)?;
        let r: Array1<f64> =
            ferray_stats::bincount_weighted(&fx, &fw, minlength).map_err(ferr_to_pyerr)?;
        Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    } else {
        let r: Array1<u64> = ferray_stats::bincount_u64(&fx, minlength).map_err(ferr_to_pyerr)?;
        u64_arr1_to_i64_pyarray(py, r)
    }
}

/// `numpy.digitize(x, bins, right=False)`.
#[pyfunction]
#[pyo3(signature = (x, bins, right = false))]
pub fn digitize<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    bins: &Bound<'py, PyAny>,
    right: bool,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::array::aliases::Array1;
    // numpy.digitize raises `TypeError: x may not be complex` on complex `x`
    // (it cannot order complex against the bin edges). Reject before the float64
    // coercion would silently digitize the real parts (R-CODE-4 / R-DEV-2).
    let x_arr = as_ndarray(py, x)?;
    if is_complex_dtype(dtype_name(&x_arr)?.as_str()) {
        return Err(PyTypeError::new_err("x may not be complex"));
    }
    let arr_x = coerce_dtype(py, x, "float64")?;
    let arr_b = coerce_dtype(py, bins, "float64")?;
    let vx: PyReadonlyArray1<f64> = arr_x.extract()?;
    let vb: PyReadonlyArray1<f64> = arr_b.extract()?;
    let fx: Array1<f64> = vx.as_ferray().map_err(ferr_to_pyerr)?;
    let fb: Array1<f64> = vb.as_ferray().map_err(ferr_to_pyerr)?;
    let r: Array1<u64> = ferray_stats::digitize(&fx, &fb, right).map_err(ferr_to_pyerr)?;
    u64_arr1_to_i64_pyarray(py, r)
}

// Force the parse_bins helper to count as used (will be wired into a
// future histogram(bins=array_of_edges) variant; phase-2.1 work).
#[allow(dead_code)]
fn _force_parse_bins_used<T>(b: &Bound<'_, PyAny>) -> PyResult<ferray_stats::Bins<T>>
where
    T: 'static,
    f64: Into<T>,
{
    parse_bins::<T>(b)
}

// ---------------------------------------------------------------------------
// percentile / quantile / median / cov / corrcoef (#703)
// ---------------------------------------------------------------------------

/// Whether a `percentile`/`quantile` library call computes percentages
/// (`true`) or fractions (`false`). Picks which scalar-q library fn to call.
enum QuantileKind {
    Percentile,
    Quantile,
}

/// Shared implementation for `percentile`/`quantile`, honoring numpy's
/// `q : array_like of float` contract (numpy/lib/_function_base_impl.py:4083
/// percentile, :4284 quantile). A scalar `q` returns the bare reduction; a
/// sequence `q` computes one reduction per entry and stacks them along a new
/// leading axis via `numpy.stack` (`np.percentile(x, [25,50,75])` ->
/// shape `(3, ...)`), matching numpy exactly (R-DEV-3). The library
/// percentile/quantile are scalar-q, so the binding loops them.
fn quantile_dispatch<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    q: &Bound<'py, PyAny>,
    axis: Option<isize>,
    kind: QuantileKind,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let axis = norm_axis(py, &arr, axis)?;

    // datetime64/timedelta64 (#946): `percentile`/`quantile` interpolate the
    // int64 ticks (linear method) and re-tag the result — timedelta stays
    // timedelta, datetime stays datetime. Branch BEFORE the float64 coercion
    // below (the R-CODE-4 silent-float corruption: ferray returned a bare
    // float). `q` is rescaled to a `[0, 1]` fraction (percentile divides by
    // 100) and range-validated to mirror numpy's ValueError.
    if crate::datetime::is_time_array(&arr)? {
        let single_time = |q_val: f64| -> PyResult<Bound<'py, PyAny>> {
            let frac = match kind {
                QuantileKind::Percentile => {
                    if !(0.0..=100.0).contains(&q_val) {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "Percentiles must be in the range [0, 100]",
                        ));
                    }
                    q_val / 100.0
                }
                QuantileKind::Quantile => {
                    if !(0.0..=1.0).contains(&q_val) {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "Quantiles must be in the range [0, 1]",
                        ));
                    }
                    q_val
                }
            };
            crate::datetime::time_quantile(py, &arr, frac, axis)
        };
        return match extract_q(q)? {
            Err(scalar) => single_time(scalar),
            Ok(seq) => {
                let mut results: Vec<Bound<'py, PyAny>> = Vec::with_capacity(seq.len());
                for q_val in seq {
                    results.push(single_time(q_val)?);
                }
                let np = py.import("numpy")?;
                let list = pyo3::types::PyList::new(py, results)?;
                np.call_method1("stack", (list,))
            }
        };
    }

    let dt = dtype_name(&arr)?;
    let real_dt = if matches!(dt.as_str(), "float32" | "f32" | "float64" | "f64") {
        dt.as_str().to_string()
    } else {
        "float64".to_string()
    };
    let arr = coerce_dtype(py, &arr, &real_dt)?;

    // Compute one scalar-q reduction, returning a `Bound` pyarray.
    let single = |q_val: f64| -> PyResult<Bound<'py, PyAny>> {
        Ok(match_dtype_float!(real_dt.as_str(), T => {
            let view: PyReadonlyArrayDyn<T> = arr.extract()?;
            let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
            let r = match kind {
                QuantileKind::Percentile => ferray_stats::percentile(&fa, q_val as T, axis),
                QuantileKind::Quantile => ferray_stats::quantile(&fa, q_val as T, axis),
            }
            .map_err(ferr_to_pyerr)?;
            r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
        }))
    };

    match extract_q(q)? {
        Err(scalar) => single(scalar),
        Ok(seq) => {
            let mut results: Vec<Bound<'py, PyAny>> = Vec::with_capacity(seq.len());
            for q_val in seq {
                results.push(single(q_val)?);
            }
            // Stack per-q results along a new leading axis (numpy's q-axis).
            let np = py.import("numpy")?;
            let list = pyo3::types::PyList::new(py, results)?;
            np.call_method1("stack", (list,))
        }
    }
}

/// `numpy.percentile(a, q, axis=None)` — Linear interpolation method.
#[pyfunction]
#[pyo3(signature = (a, q, axis = None))]
pub fn percentile<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    q: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<Bound<'py, PyAny>> {
    quantile_dispatch(py, a, q, axis, QuantileKind::Percentile)
}

/// `numpy.quantile(a, q, axis=None)` — Linear interpolation method.
#[pyfunction]
#[pyo3(signature = (a, q, axis = None))]
pub fn quantile<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    q: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<Bound<'py, PyAny>> {
    quantile_dispatch(py, a, q, axis, QuantileKind::Quantile)
}

/// `numpy.median(a, axis=None)`.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn median<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    // Empty reduction: NumPy's `_median` warns and returns nan
    // (float64, 0-D); it does NOT raise
    // (numpy/lib/_function_base_impl.py:4003).
    if axis.is_none() && is_empty_array(&arr)? {
        return nan_scalar_f64(py);
    }
    // datetime64/timedelta64 (#946): `median(timedelta)->timedelta` (sort int64
    // ticks, average the two middles truncating toward zero); `median(datetime)`
    // RAISES numpy's `UFuncTypeError` (`add` of two datetimes is undefined).
    // Branch BEFORE the float64 coercion below — that coercion is the R-CODE-4
    // silent-float corruption this REQ eliminates (`fr.median(td)` returned a
    // bare float).
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::time_median(py, &arr, axis, false);
    }
    let dt = dtype_name(&arr)?;
    // Complex median: numpy sorts complex LEXICOGRAPHICALLY (real, then imag)
    // and takes the middle (odd) or the complex-average of the two middles
    // (even), preserving the input complex width (c64->c64, c128->c128 —
    // verified live numpy 2.4.5). MUST branch BEFORE the float64 coercion below,
    // which numpy casts complex->float64 DROPPING the imaginary part with a
    // `ComplexWarning` — the R-CODE-4 corruption this REQ eliminates
    // (`fr.median([1+2j,...])` returned `0.5`; numpy returns `(0.5+1j)`).
    match dt.as_str() {
        "complex128" | "c16" => return complex_median_dispatch::<f64>(py, &arr, axis),
        "complex64" | "c8" => return complex_median_dispatch::<f32>(py, &arr, axis),
        _ => {}
    }
    let real_dt = if matches!(dt.as_str(), "float32" | "f32" | "float64" | "f64") {
        dt.as_str().to_string()
    } else {
        "float64".to_string()
    };
    let arr = coerce_dtype(py, &arr, &real_dt)?;
    Ok(match_dtype_float!(real_dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r = ferray_stats::median(&fa, axis).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// Lexicographic order `(real, then imag)` for two complex values — numpy's
/// complex sort key (`numpy/_core/src/npysort`). Used by the complex median.
/// NaN parts compare as `Greater` so they sort to the END, matching numpy's
/// placement of complex NaN at the tail of a sorted array.
fn complex_lex_order<T: PartialOrd + Copy>(a: &Complex<T>, b: &Complex<T>) -> core::cmp::Ordering {
    use core::cmp::Ordering;
    // Order two REAL components; an unordered (NaN) comparison pushes the
    // NaN-bearing side last so complex NaN sorts to the tail (numpy's placement).
    let key = |x: T, y: T| -> Ordering {
        match x.partial_cmp(&y) {
            Some(o) => o,
            None => {
                let x_nan = x.partial_cmp(&x).is_none();
                if x_nan {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            }
        }
    };
    match key(a.re, b.re) {
        Ordering::Equal => key(a.im, b.im),
        ord => ord,
    }
}

/// Complex `median` over `axis` (or the whole array): lexicographically sort
/// each lane, then take the middle element (odd length) or the complex average
/// of the two middle elements (even length). A lane containing a NaN component
/// yields `nan+nanj` (numpy's `_median_nancheck` replaces a NaN-containing
/// median with NaN). Width preserved (`Complex<T>`).
fn complex_median_dispatch<'py, T>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default + ReductionReal + PartialOrd,
    Complex<T>: ferray_core::Element + numpy::Element,
{
    let fa: ArrayD<Complex<T>> = complex_pyarray_to_ferray::<T>(arr)?;
    let shape: Vec<usize> = fa.shape().to_vec();
    let ndim = shape.len();
    let half = T::ONE / (T::ONE + T::ONE);

    let median_of = |vals: &[Complex<T>]| -> Complex<T> {
        let n = vals.len();
        if n == 0 {
            let nan = T::NAN;
            return Complex::new(nan, nan);
        }
        // numpy propagates NaN: a NaN-containing lane has a NaN median.
        if vals.iter().any(complex_is_nan) {
            let nan = T::NAN;
            return Complex::new(nan, nan);
        }
        let mut sorted: Vec<Complex<T>> = vals.to_vec();
        sorted.sort_by(complex_lex_order);
        if n % 2 == 1 {
            sorted[n / 2]
        } else {
            let lo = sorted[n / 2 - 1];
            let hi = sorted[n / 2];
            // (lo + hi) / 2 per component (numpy averages the two middles).
            Complex::new((lo.re + hi.re) * half, (lo.im + hi.im) * half)
        }
    };

    match axis {
        None => {
            let vals: Vec<Complex<T>> = fa.iter().copied().collect();
            let r = median_of(&vals);
            let rd =
                ArrayD::<Complex<T>>::from_vec(IxDyn::new(&[]), vec![r]).map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, rd)
        }
        Some(ax) => {
            if ax >= ndim {
                return Err(ferr_to_pyerr(ferray_core::FerrayError::invalid_value(
                    "complex median: axis out of bounds",
                )));
            }
            let out_shape: Vec<usize> = shape
                .iter()
                .enumerate()
                .filter_map(|(d, &s)| if d == ax { None } else { Some(s) })
                .collect();
            let axis_len = shape[ax];
            let flat: Vec<Complex<T>> = fa.iter().copied().collect();
            let mut strides = vec![1usize; ndim];
            for d in (0..ndim.saturating_sub(1)).rev() {
                strides[d] = strides[d + 1] * shape[d + 1];
            }
            let out_dims: Vec<usize> = (0..ndim).filter(|&d| d != ax).collect();
            let out_extents: Vec<usize> = out_dims.iter().map(|&d| shape[d]).collect();
            let out_n: usize = out_extents.iter().product::<usize>().max(1);
            let mut out: Vec<Complex<T>> = Vec::with_capacity(out_n);
            for lin in 0..out_n {
                let mut rem = lin;
                let mut base = 0usize;
                for (i, &d) in out_dims.iter().enumerate() {
                    let ext = out_extents[i];
                    let c = rem % ext;
                    rem /= ext;
                    base += c * strides[d];
                }
                let lane: Vec<Complex<T>> = (0..axis_len)
                    .map(|k| flat[base + k * strides[ax]])
                    .collect();
                out.push(median_of(&lane));
            }
            let rd = ArrayD::<Complex<T>>::from_vec(IxDyn::new(&out_shape), out)
                .map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, rd)
        }
    }
}

/// `numpy.cov(m, rowvar=True, ddof=None)`.
#[pyfunction]
#[pyo3(signature = (m, rowvar = true, ddof = None))]
pub fn cov<'py>(
    py: Python<'py>,
    m: &Bound<'py, PyAny>,
    rowvar: bool,
    ddof: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::array::aliases::Array2;
    let arr = as_ndarray(py, m)?;
    let dt = dtype_name(&arr)?;
    let real_dt = if matches!(dt.as_str(), "float32" | "f32" | "float64" | "f64") {
        dt.as_str().to_string()
    } else {
        "float64".to_string()
    };
    let arr = coerce_dtype(py, &arr, &real_dt)?;
    Ok(match_dtype_float!(real_dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: Array2<T> = ferray_stats::cov(&fa, rowvar, ddof).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.corrcoef(x, rowvar=True)`.
#[pyfunction]
#[pyo3(signature = (x, rowvar = true))]
pub fn corrcoef<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    rowvar: bool,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::array::aliases::Array2;
    let arr = as_ndarray(py, x)?;
    let dt = dtype_name(&arr)?;
    let real_dt = if matches!(dt.as_str(), "float32" | "f32" | "float64" | "f64") {
        dt.as_str().to_string()
    } else {
        "float64".to_string()
    };
    let arr = coerce_dtype(py, &arr, &real_dt)?;
    Ok(match_dtype_float!(real_dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: Array2<T> = ferray_stats::corrcoef(&fa, rowvar).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.where(condition, x, y)` — three-argument form (broadcasting).
#[pyfunction]
#[pyo3(name = "where_")]
pub fn where_fn<'py>(
    py: Python<'py>,
    condition: &Bound<'py, PyAny>,
    x: &Bound<'py, PyAny>,
    y: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let cond = as_ndarray(py, condition)?;
    // datetime64/timedelta64 selection: `where(cond, dt|td, dt|td)` ->
    // dt|td. Routed through the #947 time dispatch ahead of the real-only
    // `match_dtype_all_complex!` (which would raise `TypeError` on the time
    // dtype), when BOTH x and y are time arrays of the same kind.
    if crate::datetime::is_time_pair_same_kind(py, x, y)? {
        return crate::datetime::where_time(py, &cond, x, y);
    }
    // Broadcast condition + x + y to a common shape via numpy.
    let np = py.import("numpy")?;
    let pair = np.call_method1("broadcast_arrays", (&cond, x, y))?;
    let bcast: Vec<Bound<PyAny>> = pair.extract()?;
    let cond_b = coerce_dtype(py, &bcast[0], "bool")?;
    // numpy promotes the *output* dtype via result_type(x, y) — so
    // `where(cond, real, complex)` and `where(cond, complex, real)` both
    // yield complex (numpy/_core/multiarray.py:198). Taking only x's
    // dtype would silently drop the imaginary part of a complex `y`.
    let dt: String = np
        .getattr("result_type")?
        .call1((&bcast[1], &bcast[2]))?
        .getattr("name")?
        .extract()?;
    let arr_x = coerce_dtype(py, &bcast[1], dt.as_str())?;
    let arr_y = coerce_dtype(py, &bcast[2], dt.as_str())?;
    let cond_view: PyReadonlyArrayDyn<bool> = cond_b.extract()?;
    let cond_fa: ArrayD<bool> = cond_view.as_ferray().map_err(ferr_to_pyerr)?;
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let fx: ArrayD<T> = T::extract_dyn(&arr_x)?;
        let fy: ArrayD<T> = T::extract_dyn(&arr_y)?;
        let r: ArrayD<T> = ferray_stats::where_(&cond_fa, &fx, &fy).map_err(ferr_to_pyerr)?;
        T::emit_dyn(py, r)?
    }))
}

// ---------------------------------------------------------------------------
// nan-aware cumulative reductions
// ---------------------------------------------------------------------------

/// Complex `nancumsum`/`nancumprod` over a width `T`: replace each NaN-complex
/// element with the fold identity (`0` for cumsum, `1` for cumprod) then run the
/// complex cumulative fold via the existing complex `ReduceAcc` (width-preserving).
/// numpy: `np.nancumsum([3+4j,1-2j,-5+12j]) == [3+4j,4+2j,-1+14j]`,
/// `np.nancumprod([3+4j,1-2j,-5+12j]) == [3+4j,11-2j,-31+142j]` (NaN -> identity,
/// the running value is unchanged across a NaN position — verified live numpy
/// 2.4.4). Shape + width preserved (R-CODE-4: no imaginary discard; the prior
/// path coerced to float64 and `nancumprod` ALSO mis-multiplied the real parts).
fn complex_nan_cumulative_dispatch<'py, T>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    axis: Option<usize>,
    is_prod: bool,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default + ReductionReal + PartialOrd,
    Complex<T>: ferray_core::Element + numpy::Element,
    Complex<T>: ReduceAcc<Acc = Complex<T>> + Send + Sync,
{
    let fa: ArrayD<Complex<T>> = complex_pyarray_to_ferray::<T>(arr)?;
    let ident = if is_prod {
        Complex::new(T::ONE, T::ZERO)
    } else {
        Complex::new(T::ZERO, T::ZERO)
    };
    let shape: Vec<usize> = fa.shape().to_vec();
    let data: Vec<Complex<T>> = fa
        .iter()
        .map(|z| if complex_is_nan(z) { ident } else { *z })
        .collect();
    let cleaned =
        ArrayD::<Complex<T>>::from_vec(IxDyn::new(&shape), data).map_err(ferr_to_pyerr)?;
    let r: ArrayD<Complex<T>> = if is_prod {
        ferray_stats::cumprod(&cleaned, axis).map_err(ferr_to_pyerr)?
    } else {
        ferray_stats::cumsum(&cleaned, axis).map_err(ferr_to_pyerr)?
    };
    complex_ferray_to_pyarray(py, r)
}

macro_rules! bind_nan_cumulative {
    ($name:ident, $ferr_path:path, $is_prod:expr) => {
        #[pyfunction]
        #[pyo3(signature = (a, axis = None))]
        pub fn $name<'py>(
            py: Python<'py>,
            a: &Bound<'py, PyAny>,
            axis: Option<usize>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let arr = as_ndarray(py, a)?;
            let dt = dtype_name(&arr)?;
            // Complex arm: NaN-complex -> fold identity, then complex cumulative
            // fold, keeping the input width BEFORE the f64 coercion can drop the
            // imaginary part (R-CODE-4).
            match dt.as_str() {
                "complex128" | "c16" => {
                    return complex_nan_cumulative_dispatch::<f64>(py, &arr, axis, $is_prod);
                }
                "complex64" | "c8" => {
                    return complex_nan_cumulative_dispatch::<f32>(py, &arr, axis, $is_prod);
                }
                _ => {}
            }
            let arr = if matches!(dt.as_str(), "float64" | "float32" | "f64" | "f32") {
                arr
            } else {
                coerce_dtype(py, &arr, "float64")?
            };
            let dt = dtype_name(&arr)?;
            Ok(match_dtype_float!(dt.as_str(), T => {
                let view: PyReadonlyArrayDyn<T> = arr.extract()?;
                let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
                let r = $ferr_path(&fa, axis).map_err(ferr_to_pyerr)?;
                r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
            }))
        }
    };
}

// `numpy.nancumsum` treats NaN as 0, `nancumprod` treats NaN as 1
// (numpy/lib/_nanfunctions_impl.py:825 nancumsum, :915 nancumprod). The
// library re-exports the ferray-ufunc cumulative-with-nan-skip kernels; the
// complex arm folds via the complex `ReduceAcc` after NaN->identity replacement.
bind_nan_cumulative!(nancumsum, ferray_stats::nancumsum, false);
bind_nan_cumulative!(nancumprod, ferray_stats::nancumprod, true);

// ---------------------------------------------------------------------------
// nan-aware percentile / quantile
// ---------------------------------------------------------------------------

/// Shared scalar-or-sequence dispatch for `nanpercentile`/`nanquantile`,
/// mirroring the `q : array_like of float` contract numpy documents for
/// `numpy.nanpercentile` (numpy/lib/_nanfunctions_impl.py:1156) and
/// `numpy.nanquantile` (:1382). A scalar `q` returns the bare reduction; a
/// sequence stacks one reduction per `q` along a new leading axis (matching
/// the non-nan `percentile`/`quantile` dispatch in [`quantile_dispatch`]).
/// The library `nanpercentile`/`nanquantile` are scalar-q, so the binding
/// loops them.
fn nan_quantile_dispatch<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    q: &Bound<'py, PyAny>,
    axis: Option<isize>,
    kind: QuantileKind,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let axis = norm_axis(py, &arr, axis)?;
    let dt = dtype_name(&arr)?;
    let real_dt = if matches!(dt.as_str(), "float32" | "f32" | "float64" | "f64") {
        dt.as_str().to_string()
    } else {
        "float64".to_string()
    };
    let arr = coerce_dtype(py, &arr, &real_dt)?;

    let single = |q_val: f64| -> PyResult<Bound<'py, PyAny>> {
        Ok(match_dtype_float!(real_dt.as_str(), T => {
            let view: PyReadonlyArrayDyn<T> = arr.extract()?;
            let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
            let r = match kind {
                QuantileKind::Percentile => ferray_stats::nanpercentile(&fa, q_val as T, axis),
                QuantileKind::Quantile => ferray_stats::nanquantile(&fa, q_val as T, axis),
            }
            .map_err(ferr_to_pyerr)?;
            r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
        }))
    };

    match extract_q(q)? {
        Err(scalar) => single(scalar),
        Ok(seq) => {
            let mut results: Vec<Bound<'py, PyAny>> = Vec::with_capacity(seq.len());
            for q_val in seq {
                results.push(single(q_val)?);
            }
            let np = py.import("numpy")?;
            let list = pyo3::types::PyList::new(py, results)?;
            np.call_method1("stack", (list,))
        }
    }
}

/// `numpy.nanpercentile(a, q, axis=None)` — percentile ignoring NaNs.
#[pyfunction]
#[pyo3(signature = (a, q, axis = None))]
pub fn nanpercentile<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    q: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<Bound<'py, PyAny>> {
    nan_quantile_dispatch(py, a, q, axis, QuantileKind::Percentile)
}

/// `numpy.nanquantile(a, q, axis=None)` — quantile ignoring NaNs.
#[pyfunction]
#[pyo3(signature = (a, q, axis = None))]
pub fn nanquantile<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    q: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<Bound<'py, PyAny>> {
    nan_quantile_dispatch(py, a, q, axis, QuantileKind::Quantile)
}

// ---------------------------------------------------------------------------
// average (weighted mean)
// ---------------------------------------------------------------------------

/// `numpy.average(a, axis=None, weights=None)`.
///
/// numpy/lib/_function_base_impl.py:560 `average` — with no `weights` this
/// is the plain `mean`; with `weights` it is `sum(a*w)/sum(w)` along `axis`.
/// Integer/bool input promotes to `float64` (numpy computes the weighted
/// average in inexact arithmetic). The library `ferray_stats::average`
/// requires `weights.shape() == a.shape()`, so the binding broadcasts a
/// 1-D `weights` against `a` via numpy before dispatch when an `axis` is
/// given, matching numpy's `weights` along a single axis.
#[pyfunction]
#[pyo3(signature = (a, axis = None, weights = None))]
pub fn average<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
    weights: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    // datetime64/timedelta64 (#946): numpy's `average` RAISES on time inputs
    // (unweighted `average(td)` -> ValueError "Could not convert object to NumPy
    // timedelta"; weighted -> DTypePromotionError; datetime -> UFuncTypeError —
    // all verified live). Delegate so numpy's EXACT exception surfaces (R-DEV-2),
    // eliminating the prior silent-float corruption (`fr.average(td) -> 5.0`).
    if crate::datetime::is_time_array(&arr)? {
        let np = py.import("numpy")?;
        let kwargs = pyo3::types::PyDict::new(py);
        if let Some(ax) = axis {
            kwargs.set_item("axis", ax)?;
        }
        if let Some(w) = weights {
            kwargs.set_item("weights", w)?;
        }
        return np.call_method("average", (&arr,), Some(&kwargs));
    }
    // Complex average: branch BEFORE the `coerce_dtype(arr,"float64")` below,
    // which drops the imaginary part (R-CODE-4). No-weights complex average is
    // the complex mean (width preserved, c64->c64/c128->c128). Weighted complex
    // average is `sum(a*w)/sum(w)` computed in complex128 (numpy promotes the
    // weighted complex average to complex128 — verified live numpy 2.4.4).
    let dt = dtype_name(&arr)?;
    if is_complex_dtype(dt.as_str()) {
        return match weights {
            None => match dt.as_str() {
                "complex128" | "c16" => complex_mean_dispatch::<f64>(py, &arr, axis),
                _ => complex_mean_dispatch::<f32>(py, &arr, axis),
            },
            Some(w) => complex_weighted_average(py, &arr, w, axis),
        };
    }
    let arr = coerce_dtype(py, &arr, "float64")?;
    let view: PyReadonlyArrayDyn<f64> = arr.extract()?;
    let fa: ArrayD<f64> = view.as_ferray().map_err(ferr_to_pyerr)?;
    let r = match weights {
        None => ferray_stats::average(&fa, None, axis).map_err(ferr_to_pyerr)?,
        Some(w) => {
            let warr = coerce_dtype(py, w, "float64")?;
            // numpy broadcasts a 1-D `weights` against the reduced axis.
            // When `weights` already matches `a`'s shape, broadcast is a
            // no-op; otherwise reshape it onto `axis` then broadcast.
            let np = py.import("numpy")?;
            let target_shape = arr.getattr("shape")?;
            let wshape: Vec<usize> = warr.getattr("shape")?.extract()?;
            let ashape: Vec<usize> = arr.getattr("shape")?.extract()?;
            let warr_b = if wshape == ashape {
                warr
            } else if let Some(ax) = axis {
                // Reshape 1-D weights to broadcast along `axis`, then expand.
                let ndim = ashape.len();
                let mut newshape = vec![1usize; ndim];
                if ax < ndim {
                    newshape[ax] = wshape.iter().product();
                }
                let reshaped = np.call_method1("reshape", (&warr, newshape))?;
                np.call_method1("broadcast_to", (reshaped, target_shape))?
            } else {
                np.call_method1("broadcast_to", (&warr, target_shape))?
            };
            let wview: PyReadonlyArrayDyn<f64> = warr_b.extract()?;
            let fw: ArrayD<f64> = wview.as_ferray().map_err(ferr_to_pyerr)?;
            ferray_stats::average(&fa, Some(&fw), axis).map_err(ferr_to_pyerr)?
        }
    };
    Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

// ---------------------------------------------------------------------------
// sort_complex
// ---------------------------------------------------------------------------

/// `numpy.sort_complex(a)` — sort a complex array lexicographically by real
/// part then imaginary part (numpy/lib/_function_base_impl.py:638). A real
/// input is upcast to complex first (numpy: `b = array(a, copy=True);
/// b.sort(); ... if not issubclass(b.dtype.type, complexfloating): ... return
/// b.astype(complex)` — a real array's result is the complex-typed sorted
/// values). The library `ferray_stats::sort_complex` operates on
/// `Complex<f64>` / `Complex<f32>`; the binding coerces the input to
/// `complex128` (or keeps `complex64`) before dispatch.
#[pyfunction]
pub fn sort_complex<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    use crate::fft::{complex_ferray_to_pyarray, complex_pyarray_to_ferray};
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    let target = if dt.as_str() == "complex64" {
        "complex64"
    } else {
        "complex128"
    };
    let arr = coerce_dtype(py, &arr, target)?;
    // The library `sort_complex` is Ix1 over the flattened input. Extract the
    // complex data into a 1-D ferray array, sort, and return as a fresh numpy
    // complex array via the same materialiser the fft bindings use.
    if target == "complex64" {
        let fa = complex_pyarray_to_ferray::<f32>(&arr)?;
        let n = fa.size();
        let flat: Vec<num_complex::Complex<f32>> = fa.iter().copied().collect();
        let a1 = Array1::<num_complex::Complex<f32>>::from_vec(Ix1::new([n]), flat)
            .map_err(ferr_to_pyerr)?;
        let r = ferray_stats::sort_complex(&a1).map_err(ferr_to_pyerr)?;
        let data: Vec<num_complex::Complex<f32>> = r.iter().copied().collect();
        let rd = ArrayD::<num_complex::Complex<f32>>::from_vec(IxDyn::new(&[n]), data)
            .map_err(ferr_to_pyerr)?;
        complex_ferray_to_pyarray(py, rd)
    } else {
        let fa = complex_pyarray_to_ferray::<f64>(&arr)?;
        let n = fa.size();
        let flat: Vec<num_complex::Complex<f64>> = fa.iter().copied().collect();
        let a1 = Array1::<num_complex::Complex<f64>>::from_vec(Ix1::new([n]), flat)
            .map_err(ferr_to_pyerr)?;
        let r = ferray_stats::sort_complex(&a1).map_err(ferr_to_pyerr)?;
        let data: Vec<num_complex::Complex<f64>> = r.iter().copied().collect();
        let rd = ArrayD::<num_complex::Complex<f64>>::from_vec(IxDyn::new(&[n]), data)
            .map_err(ferr_to_pyerr)?;
        complex_ferray_to_pyarray(py, rd)
    }
}

// ---------------------------------------------------------------------------
// cross (1-D 3-vector)
// ---------------------------------------------------------------------------

/// `numpy.cross(a, b)` — the legacy top-level cross product
/// (`numpy/_core/numeric.py:cross`).
///
/// Routes through `ferray_linalg::cross`, which computes the cross product
/// along the last axis and supports:
/// - **3-component** vectors `(...,3) × (...,3) → (...,3)`, including stacked
///   inputs that broadcast over their leading batch axes (#836);
/// - **2-component** vectors `(...,2) × (...,2)`, returning the scalar
///   z-component (the legacy form numpy 2.4.5 still allows with a
///   `DeprecationWarning`).
///
/// Integer inputs are promoted to `float64` via `promote_linalg_input`
/// (the library cross is float-typed); numpy's documented contract is the
/// numerical cross-product value, which is preserved.
#[pyfunction]
pub fn cross<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr_a = promote_linalg_input(py, a)?;
    let dt = dtype_name(&arr_a)?;
    let arr_b = coerce_dtype(py, b, dt.as_str())?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let va: PyReadonlyArrayDyn<T> = arr_a.extract()?;
        let vb: PyReadonlyArrayDyn<T> = arr_b.extract()?;
        let fa: ArrayD<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
        let fb: ArrayD<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fl::cross(&fa, &fb).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}
