//! Bindings for `numpy.ma` — masked arrays.
//!
//! Wraps `ferray_ma::MaskedArray<T, IxDyn>` as a Python class via a runtime-
//! typed `enum DynMa` over the 11 REAL numpy dtypes (bool, int8/16/32/64,
//! uint8/16/32/64, float32/64), mirroring `ferray_core::DynArray` and the
//! `match_dtype_all!` idiom in `conv.rs`. PyO3 classes cannot be generic, so
//! `PyMaskedArray.inner` is a `DynMa` and each `#[pymethods]` body dispatches
//! on the active variant via the `match_ma!` macro — preserving the input's
//! native dtype across the boundary (#853, R-CODE-4: no int→f64 cast).
//!
//! The float-only numpy.ma ufunc/reduction surface (`sin`, `add`, `sqrt`,
//! `sort`, the scalar `sum`/`mean`/`min`/`max`/`var`/`std` reductions, …)
//! computes in f64 — matching numpy.ma's promote-to-float contract for those
//! ops — via `PyMaskedArray::to_f64_ma`, then re-wraps as `DynMa::F64`.
//!
//! COMPLEX (c64/c128) and structured/datetime masked arrays are OUT OF SCOPE
//! (the ferray-ma masked surface + interop `NpElement` set exclude them); they
//! stay on the existing numpy.ma delegation paths.
//!
//! The class exposes:
//!
//! - Constructors via free functions: `masked_array`, `array`,
//!   `masked_where`, `masked_invalid`, `masked_equal`, `masked_greater`,
//!   `masked_greater_equal`, `masked_less`, `masked_less_equal`,
//!   `masked_not_equal`, `masked_inside`, `masked_outside`.
//! - Properties: `data`, `mask`, `shape`, `ndim`, `size`,
//!   `fill_value`, `dtype`.
//! - Methods: `count`, `sum`, `mean`, `min`, `max`, `var`, `std`
//!   (with optional axis), `filled`, `compressed`, `__repr__`,
//!   `__array__` (NumPy protocol — returns the underlying data verbatim,
//!   so `numpy.asarray(ma)` equals `ma.data`).
//!
//! Boundary conventions mapping the ferray-ma model onto numpy.ma's Python
//! API: all-masked full reductions return the `numpy.ma.masked` singleton
//! (not ferray-ma's NaN analog); `getmask` / `.mask` of an unmasked array
//! return the `numpy.ma.nomask` singleton; `masked_where` with a
//! shape-mismatched condition raises `IndexError`; `masked_equal` sets the
//! result's `fill_value` to the compared value.
//!
//! The numpy.ma expansion (refs #818) adds, composed over the ferray-ma
//! library (`fma::*`):
//!
//! - **Masked elementwise unary ufuncs**: `sin`, `cos`, `tan`, `arctan`,
//!   `sinh`, `cosh`, `tanh`, `arcsinh`, `exp`, `floor`, `ceil`, `around`,
//!   `negative`, `absolute`, `abs`, `fabs`, `conjugate`, plus the
//!   domain-masking `sqrt`, `log`, `log2`, `log10`, `arcsin`, `arccos`,
//!   `arccosh`, `arctanh` (out-of-domain inputs are auto-masked).
//! - **Masked elementwise binary ufuncs** (MA-or-scalar 2nd arg): `add`,
//!   `subtract`, `multiply`, `divide` (zero-denominator masked),
//!   `true_divide`, `floor_divide`, `power`, `arctan2`, `hypot`, `fmod`,
//!   `remainder`, `mod`, `maximum`, `minimum`, `bitwise_and/or/xor`.
//! - **Masked reductions**: `prod`, `median`, `ptp`, `argmin`, `argmax`,
//!   `count`, `average`, `all`, `any`, `anom`.
//! - **Masked creation**: `zeros`, `ones`, `empty`, `arange`, `identity`,
//!   `asarray`, `asanyarray`, `copy` (nomask).
//! - **Masked manipulation**: `reshape`, `ravel`, `transpose`, `squeeze`,
//!   `expand_dims`, `concatenate`, `diag`, `repeat`, `clip`.
//! - **Mask helpers / predicates**: `getmaskarray`, `make_mask`,
//!   `make_mask_none`, `mask_or`, `masked_values`, `masked_object`,
//!   `fix_invalid`, `is_mask`, `isMaskedArray`/`isMA`/`isarray`,
//!   `set_fill_value`, `default_fill_value`.
//!
//! The specialized-algorithm slice (refs #835) binds the existing ferray-ma
//! algorithms that match numpy.ma exactly: `sort` / `argsort` (masked
//! values trail), `take`, `trace`, `dot` (1-D inner product), `unique`.
//!
//! The residual numpy.ma slice (refs #837) adds, composed at the binding:
//!
//! - **`polyfit`** — least-squares fit dropping any masked `(x, y)` pair, then
//!   delegating to the ferray-polynomial `Poly::fit` (highest-first coeffs).
//! - **`convolve` / `correlate`** — masked 1-D convolution / cross-correlation
//!   over `ferray_ufunc::convolve` / `ferray_stats::correlate`, with the mask
//!   propagated per numpy's `_convolve_or_correlate` (`propagate_mask`).
//! - **`cov` / `corrcoef` two-variable `y` form** — `x` and `y` are stacked
//!   into one variable matrix (numpy's `_covhelper`) before the masked
//!   covariance / correlation.
//!
//! The composable manipulation/elementwise/alias/error-class batch
//! (refs #835 #818) adds, each composed at the binding by applying the matching
//! `numpy.<name>` op to the masked array's data and bool mask independently and
//! recombining (the generalized `diag` pattern, `np_data_mask_op`):
//!
//! - **Mask-propagating manipulation**: `atleast_1d`/`atleast_2d`/`atleast_3d`,
//!   `column_stack`, `dstack`, `vstack`/`hstack`/`stack`/`row_stack`,
//!   `diagonal`, `diagflat`, `swapaxes`, `resize`, `compress` (1-arg masked),
//!   `append` (flatten / axis), `empty_like`/`ones_like`/`zeros_like`
//!   (input mask preserved), `cumsum`/`cumprod` (masked → reduction identity,
//!   then accumulate, mask kept), `round`/`round_`.
//! - **Masked comparisons / logical ops**: `equal`, `not_equal`, `greater`,
//!   `greater_equal`, `less`, `less_equal`, `logical_and`/`logical_or`/
//!   `logical_xor` (mask = OR of operand masks), `logical_not`.
//! - **Reductions as free functions + aliases**: `max`/`amax`, `min`/`amin`,
//!   `sum`, `mean`, `std`, `var`, `product` (→`prod`), `alltrue` (→`all`),
//!   `sometrue` (→`any`), `anomalies` (→`anom`).
//! - **Inner/outer products**: `outer`/`outerproduct`, `inner`/`innerproduct`.
//! - **Shape inspectors**: `ndim`, `shape`, `size`. **Masked `angle`** (real).
//! - **Predicates / fill-value helpers**: `allequal`, `allclose`,
//!   `maximum_fill_value`, `minimum_fill_value`, `common_fill_value`.
//! - **Shared vocabulary** (re-exported from numpy.ma): the `MAError` /
//!   `MaskError` exception classes, the `MaskType` (numpy bool) type, and the
//!   `masked` / `masked_singleton` / `nomask` sentinel constants.
//!
//! The composable array-constructor / iteration / split / row-col-mask batch
//! (refs #835 #818) adds, each composed at the binding over numpy.ma's own
//! algorithm (building a `numpy.ma.MaskedArray` from the input data + bool mask,
//! running `numpy.ma.<func>`, and returning numpy's dtype-preserving result) or
//! over ferray-ma's `ma_mask_rowcols` (#835):
//!
//! - **Array-of-masked constructors**: `masked_all`/`masked_all_like`
//!   (all-True mask of a given shape/dtype), `fromfunction` (evaluate over the
//!   index grid), `indices` (masked integer index grid).
//! - **Masked iteration / split**: `apply_along_axis`/`apply_over_axes`
//!   (apply to masked slices), `ndenumerate` ((index, value) skipping masked
//!   when `compressed=True`, else yielding `masked`), `hsplit` (masked split).
//! - **Whole-row/col masking**: `mask_rows` (== `mask_rowcols(a, 0)`) and
//!   `mask_cols` (== `mask_rowcols(a, 1)`), wired to ferray-ma's
//!   `ma_mask_rowcols`.
//! - **Multi-axis compress**: `compress_nd` (suppress masked slices along any
//!   axis → plain ndarray). **`ids`** (`(id(data), id(mask))`). **`bool_`** —
//!   re-export of the numpy bool scalar type (shared vocabulary).
//!
//! The structured-mask + integer-bitwise batch (refs #835 #818) binds the
//! residual non-stateful numpy.ma surface as shared-vocabulary delegations to
//! numpy.ma's canonical algorithm (the f64-only `PyMaskedArray` cannot hold
//! structured / integer dtypes without a lossy f64 cast, R-CODE-4):
//!
//! - **Structured-mask dtype vocabulary**: `make_mask_descr` (all-bool mask
//!   dtype mirroring a possibly-structured dtype), `flatten_mask` (flatten a
//!   nested/structured mask to flat bool), `flatten_structured_array` (flatten
//!   a structured array to 2-D/1-D), `fromflex` (rebuild a MaskedArray from a
//!   flexible `_data`/`_mask` structured array — inverse of `toflex`), and the
//!   `mvoid` scalar type re-export (single masked structured/record element).
//! - **Buffer constructor**: `frombuffer` (MaskedArray over a buffer-protocol
//!   object, no mask).
//! - **Integer-dtype masked bitwise shifts**: `left_shift` / `right_shift`,
//!   delegated dtype-preservingly so the integer result + mask survive the
//!   boundary (no int→f64 cast).
//!
//! Functions still needing masked-algorithm support ferray-ma genuinely lacks
//! or whose ferray-ma form diverges from numpy.ma's mask semantics
//! (`vander`/`isin`/`in1d` mask handling, 2-D `dot` matmul, multi-axis
//! `argsort`, `where`, `choose`, `diff`, `ediff1d`,
//! `nonzero`, `clump_*`, `notmasked_*`, `flatnotmasked_*`) and the four
//! genuinely-stateful ops needing an in-place / hardness-state extension on a
//! ferray-ma `PyMaskedArray` (`harden_mask`/`soften_mask` state,
//! `put`/`putmask` in-place) are tracked as a ferray-ma library follow-up
//! under #835.

use ferray_core::Array;
use ferray_core::array::aliases::{Array1, ArrayD};
use ferray_core::dimension::{Ix1, Ix2, IxDyn};
use ferray_ma as fma;
use ferray_ma::MaskedArray as RustMa;
use ferray_numpy_interop::{AsFerray, IntoNumPy};
use ferray_polynomial as fp_poly;
use ferray_stats::correlation::{CorrelateMode, correlate as stats_correlate};
use ferray_ufunc::{ConvolveMode, convolve as ufunc_convolve};
use numpy::PyReadonlyArrayDyn;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::conv::{coerce_dtype, ferr_to_pyerr, ma_masked_singleton, ma_nomask};

// ---------------------------------------------------------------------------
// MaskedArray pyclass (f64-only main slice)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// DynMa — a runtime-typed masked array over the 11 real element dtypes
// (mirrors `ferray_core::DynArray` / the `match_dtype_all!` idiom in conv.rs).
//
// PyO3 classes cannot be generic, and the owning library `ferray_ma::Masked
// Array<T: Element, D>` IS generic; so the dtype-preservation contract is a
// binding-marshalling problem (#853). `DynMa` holds one `RustMa<T, IxDyn>`
// per real dtype; `PyMaskedArray.inner` is a `DynMa` (was the f64-only
// monomorphization). Every `#[pymethods]` body dispatches on the active
// variant via [`match_ma!`], binding the concrete `RustMa<T, IxDyn>` and a
// type alias `T`, so a single generic body is written once and Rust
// monomorphizes it per dtype — exactly the conv.rs `match_dtype_all!` shape.
//
// COMPLEX (c64/c128) and structured/datetime are OUT OF SCOPE here (the
// ferray-ma masked surface + interop `NpElement` set exclude them); those
// stay on the existing numpy.ma delegation / f64 paths and are unaffected.
// ---------------------------------------------------------------------------

/// A runtime-typed masked array over the 11 real numpy dtypes. Mirrors
/// `ferray_core::DynArray`; each variant wraps a `ferray_ma::MaskedArray<T,
/// IxDyn>` for the concrete `T`.
#[derive(Clone)]
pub enum DynMa {
    Bool(RustMa<bool, IxDyn>),
    I8(RustMa<i8, IxDyn>),
    I16(RustMa<i16, IxDyn>),
    I32(RustMa<i32, IxDyn>),
    I64(RustMa<i64, IxDyn>),
    U8(RustMa<u8, IxDyn>),
    U16(RustMa<u16, IxDyn>),
    U32(RustMa<u32, IxDyn>),
    U64(RustMa<u64, IxDyn>),
    F32(RustMa<f32, IxDyn>),
    F64(RustMa<f64, IxDyn>),
}

/// Dispatch a body over the active `DynMa` variant.
///
/// Binds the inner `RustMa<T, IxDyn>` to `$m` and a type alias `T` to the
/// concrete element type, so the body is written once and monomorphized per
/// dtype. Accepts a reference (`match_ma!(&dynma, m, T => { ... })`) or an
/// owned value. The body must be an expression of the same type in every arm.
macro_rules! match_ma {
    ($dyn:expr, $m:ident, $T:ident => $body:block) => {
        match $dyn {
            DynMa::Bool($m) => {
                #[allow(non_camel_case_types, dead_code)]
                type $T = bool;
                $body
            }
            DynMa::I8($m) => {
                #[allow(non_camel_case_types, dead_code)]
                type $T = i8;
                $body
            }
            DynMa::I16($m) => {
                #[allow(non_camel_case_types, dead_code)]
                type $T = i16;
                $body
            }
            DynMa::I32($m) => {
                #[allow(non_camel_case_types, dead_code)]
                type $T = i32;
                $body
            }
            DynMa::I64($m) => {
                #[allow(non_camel_case_types, dead_code)]
                type $T = i64;
                $body
            }
            DynMa::U8($m) => {
                #[allow(non_camel_case_types, dead_code)]
                type $T = u8;
                $body
            }
            DynMa::U16($m) => {
                #[allow(non_camel_case_types, dead_code)]
                type $T = u16;
                $body
            }
            DynMa::U32($m) => {
                #[allow(non_camel_case_types, dead_code)]
                type $T = u32;
                $body
            }
            DynMa::U64($m) => {
                #[allow(non_camel_case_types, dead_code)]
                type $T = u64;
                $body
            }
            DynMa::F32($m) => {
                #[allow(non_camel_case_types, dead_code)]
                type $T = f32;
                $body
            }
            DynMa::F64($m) => {
                #[allow(non_camel_case_types, dead_code)]
                type $T = f64;
                $body
            }
        }
    };
}

impl DynMa {
    /// The canonical numpy dtype name of the active variant (`"int64"`,
    /// `"float64"`, `"bool"`, …) — the value `numpy.ma.array(...).dtype.name`
    /// returns for the same data (R-CODE-4: the native dtype crosses the
    /// boundary, no int→f64 collapse).
    fn dtype_name(&self) -> &'static str {
        match self {
            DynMa::Bool(_) => "bool",
            DynMa::I8(_) => "int8",
            DynMa::I16(_) => "int16",
            DynMa::I32(_) => "int32",
            DynMa::I64(_) => "int64",
            DynMa::U8(_) => "uint8",
            DynMa::U16(_) => "uint16",
            DynMa::U32(_) => "uint32",
            DynMa::U64(_) => "uint64",
            DynMa::F32(_) => "float32",
            DynMa::F64(_) => "float64",
        }
    }

    fn shape(&self) -> Vec<usize> {
        match_ma!(self, m, T => { m.shape().to_vec() })
    }

    fn ndim(&self) -> usize {
        match_ma!(self, m, T => { m.ndim() })
    }

    fn size(&self) -> usize {
        match_ma!(self, m, T => { m.size() })
    }

    fn count(&self) -> PyResult<usize> {
        match_ma!(self, m, T => { m.count().map_err(ferr_to_pyerr) })
    }

    fn has_real_mask(&self) -> bool {
        match_ma!(self, m, T => { m.has_real_mask() })
    }

    fn is_hard_mask(&self) -> bool {
        match_ma!(self, m, T => { m.is_hard_mask() })
    }

    /// The bool mask buffer (`getmaskarray`: full array, never `nomask`), as a
    /// flat row-major `Vec`, regardless of the active dtype.
    fn mask_bits(&self) -> PyResult<ArrayD<bool>> {
        match_ma!(self, m, T => { fma::getmaskarray(m).map_err(ferr_to_pyerr) })
    }

    /// The flat bool mask as a `Vec`, or `None` when the array carries no real
    /// mask (the `nomask` sentinel).
    fn mask_opt_bits(&self) -> Option<Vec<bool>> {
        match_ma!(self, m, T => { m.mask_opt().map(|mk| mk.iter().copied().collect()) })
    }

    /// Number of masked elements (`size - count`), dtype-independent.
    fn count_masked(&self) -> PyResult<usize> {
        Ok(self.size() - self.count()?)
    }
}

#[pyclass(name = "MaskedArray", module = "ferray.ma", from_py_object)]
#[derive(Clone)]
pub struct PyMaskedArray {
    inner: DynMa,
}

/// Build a `DynMa` from an already-coerced numpy `ndarray` and an optional
/// real bool mask of the same shape, dispatching on the ndarray's dtype.
///
/// `arr` must already be a `numpy.ndarray` (route through `as_ndarray` first).
/// When `mask` is `Some`, the variant carries a REAL mask (`RustMa::new`);
/// when `None`, the variant carries the `nomask` sentinel (`RustMa::from_data`).
/// Any dtype outside the 11 real ones (complex / structured / datetime) raises
/// a `TypeError` rather than a lossy cast (R-CODE-4) — those stay on the numpy.ma
/// delegation paths.
fn build_dynma(arr: &Bound<'_, PyAny>, mask: Option<ArrayD<bool>>) -> PyResult<DynMa> {
    let dt = crate::conv::dtype_name(arr)?;
    macro_rules! build {
        ($ty:ty, $variant:ident) => {{
            let view = arr.extract::<PyReadonlyArrayDyn<$ty>>()?;
            let data_fa: ArrayD<$ty> = view.as_ferray().map_err(ferr_to_pyerr)?;
            let inner = match mask {
                Some(m) => RustMa::new(data_fa, m).map_err(ferr_to_pyerr)?,
                None => RustMa::from_data(data_fa).map_err(ferr_to_pyerr)?,
            };
            Ok(DynMa::$variant(inner))
        }};
    }
    match dt.as_str() {
        "bool" => build!(bool, Bool),
        "int8" | "i8" => build!(i8, I8),
        "int16" | "i16" => build!(i16, I16),
        "int32" | "i32" => build!(i32, I32),
        "int64" | "i64" => build!(i64, I64),
        "uint8" | "u8" => build!(u8, U8),
        "uint16" | "u16" => build!(u16, U16),
        "uint32" | "u32" => build!(u32, U32),
        "uint64" | "u64" => build!(u64, U64),
        "float32" | "f32" => build!(f32, F32),
        "float64" | "f64" => build!(f64, F64),
        other => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "ferray.ma does not yet support dtype {other:?} (supported: bool, \
             int8/16/32/64, uint8/16/32/64, float32/64; complex / structured are \
             a follow-up)"
        ))),
    }
}

/// Infer the dtype-preserving `DynMa` for a constructor input `data`, with an
/// optional explicit `dtype=` override.
///
/// `data` is normalized via `numpy.asarray` (no cast) so its native dtype is
/// read; an explicit `dtype` re-casts as numpy's `array(data, dtype=...)` does.
/// `mask` (already a real bool `ArrayD` of the data shape, or `None` for the
/// nomask sentinel) is carried into the variant. This replaces the old
/// `coerce_dtype(.., "float64")` int→f64 collapse (R-CODE-4).
fn dynma_from_input<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    dtype: Option<&str>,
    mask: Option<ArrayD<bool>>,
) -> PyResult<DynMa> {
    let arr = match dtype {
        Some(dt) => coerce_dtype(py, data, dt)?,
        None => crate::conv::as_ndarray(py, data)?,
    };
    build_dynma(&arr, mask)
}

impl PyMaskedArray {
    fn from_dynma(inner: DynMa) -> Self {
        Self { inner }
    }

    /// Wrap an f64 masked array as the `DynMa::F64` variant. The float-only
    /// numpy.ma ufunc/reduction/manipulation surface (`sin`, `add`, `sqrt`,
    /// `sort`, …) computes in f64 — matching numpy.ma's promote-to-float
    /// contract for those ops — and wraps its `RustMa<f64, IxDyn>` result here.
    fn from_inner(inner: RustMa<f64, IxDyn>) -> Self {
        Self {
            inner: DynMa::F64(inner),
        }
    }

    /// Reinterpret the array as a `RustMa<f64, IxDyn>` for the float-only
    /// ufunc/reduction surface, casting integer/bool data to f64 (matching
    /// numpy.ma's promote-to-float64 contract for transcendental/division ops).
    /// The mask is preserved verbatim. The f64 variant is returned directly
    /// (no copy of the data semantics).
    fn to_f64_ma(&self) -> PyResult<RustMa<f64, IxDyn>> {
        if let DynMa::F64(m) = &self.inner {
            return Ok(m.clone());
        }
        let mask = self.inner.mask_bits()?;
        let has_real = self.inner.has_real_mask();
        let data: Vec<f64> = match_ma!(&self.inner, m, T => {
            m.data().iter().map(|&v| dyn_to_f64::<T>(v)).collect()
        });
        let shape = self.inner.shape();
        let data_fa = ArrayD::<f64>::from_vec(IxDyn::new(&shape), data).map_err(ferr_to_pyerr)?;
        if has_real {
            RustMa::new(data_fa, mask).map_err(ferr_to_pyerr)
        } else {
            RustMa::from_data(data_fa).map_err(ferr_to_pyerr)
        }
    }

    /// True iff every element is masked (or the array is empty).
    ///
    /// numpy.ma's full reductions return the `masked` singleton in exactly
    /// this case (`numpy/ma/core.py:5250` `elif newmask: result = masked`,
    /// where `newmask` is the all-true reduction of the mask). The count of
    /// unmasked elements being zero is the ferray-side analog of that
    /// all-true `newmask`.
    fn all_masked(&self) -> PyResult<bool> {
        Ok(self.inner.count()? == 0)
    }

    /// Egress the data buffer as a numpy `ndarray` of the NATIVE dtype.
    fn data_pyarray<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match_ma!(&self.inner, m, T => {
            let d: ArrayD<T> = fma::getdata(m).map_err(ferr_to_pyerr)?;
            Ok(d.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
        })
    }
}

/// Cast a real-dtype element to f64 for the float-only ufunc surface. A free
/// generic fn keyed on the `IntoF64` trait below; bool maps `false`→0.0 /
/// `true`→1.0 (numpy's bool→float promotion).
fn dyn_to_f64<T: IntoF64>(v: T) -> f64 {
    v.into_f64()
}

/// Lossless-enough widening of a real masked element type to `f64` for the
/// float-only ufunc surface (numpy.ma promotes integer/bool inputs to float64
/// before sin/sqrt/division). Mirrors numpy's `result_type(x, float64)` cast.
trait IntoF64: Copy {
    fn into_f64(self) -> f64;
}
macro_rules! impl_into_f64 {
    ($($t:ty),*) => { $(impl IntoF64 for $t { fn into_f64(self) -> f64 { self as f64 } })* };
}
impl_into_f64!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);
impl IntoF64 for bool {
    fn into_f64(self) -> f64 {
        if self { 1.0 } else { 0.0 }
    }
}

/// Wrap a concrete `RustMa<Self, IxDyn>` back into the matching `DynMa`
/// variant. Lets a generic `match_ma!` body that produced a same-`T` masked
/// array (subscript gather) re-tag it without a second explicit match.
trait WrapDyn: ferray_core::Element {
    fn wrap(m: RustMa<Self, IxDyn>) -> DynMa;
}
macro_rules! impl_wrap_dyn {
    ($($t:ty => $variant:ident),* $(,)?) => {
        $(impl WrapDyn for $t {
            fn wrap(m: RustMa<Self, IxDyn>) -> DynMa { DynMa::$variant(m) }
        })*
    };
}
impl_wrap_dyn!(
    bool => Bool, i8 => I8, i16 => I16, i32 => I32, i64 => I64,
    u8 => U8, u16 => U16, u32 => U32, u64 => U64, f32 => F32, f64 => F64,
);

// ---------------------------------------------------------------------------
// Dtype-preserving masked reductions (#858, R-CODE-4/R-CODE-5)
//
// numpy.ma's `sum`/`prod`/`cumsum`/`cumprod` delegate to
// `self.filled(<identity>).<op>(...)` (`numpy/ma/core.py:5244` sum,
// `:5294` cumsum, `:5303` prod), so the result dtype is exactly numpy's
// reduction accumulator: narrow signed ints → int64, narrow unsigned →
// uint64, bool → int64, float stays float. That is `ferray_core::ReduceAcc`
// (i8/i16/i32/i64→i64, u*→u64, bool→i64, f32→f32, f64→f64). `min`/`max`
// delegate to `self.filled(<extreme>).<op>()` which KEEP the input dtype
// (`numpy/ma/core.py:5933` min — no accumulator promotion). The old
// `to_f64_ma` funnel collapsed every integer/bool reduction to float64; these
// helpers reduce over the native `DynMa` variant instead.
// ---------------------------------------------------------------------------

use ferray_core::array::reductions::ReduceAcc;

/// Additive / multiplicative identities for a `ReduceAcc::Acc` accumulator
/// type (the 4 reachable accumulators: `i64`, `u64`, `f32`, `f64`). Mirrors
/// `<Acc as num_traits::Zero/One>` without taking a direct `num-traits`
/// dependency on this crate; the reduction folds start from these.
trait AccIdentity: Copy {
    const ZERO: Self;
    const ONE: Self;
}
macro_rules! impl_acc_identity {
    ($($t:ty => $z:expr, $o:expr);* $(;)?) => {
        $(impl AccIdentity for $t { const ZERO: Self = $z; const ONE: Self = $o; })*
    };
}
impl_acc_identity!(i64 => 0, 1; u64 => 0, 1; f32 => 0.0, 1.0; f64 => 0.0, 1.0);

/// Additive (`ZERO`) / multiplicative (`ONE`) identities in the array's NATIVE
/// element type `T`, used to fill masked slots before a cumulative op —
/// numpy.ma's `filled(0)` / `filled(1)` (`numpy/ma/core.py:5294`). `bool` maps
/// `false`/`true` (numpy's bool 0/1).
trait MaIdentity: Copy {
    const ZERO: Self;
    const ONE: Self;
}
macro_rules! impl_ma_identity {
    ($($t:ty => $z:expr, $o:expr);* $(;)?) => {
        $(impl MaIdentity for $t { const ZERO: Self = $z; const ONE: Self = $o; })*
    };
}
impl_ma_identity!(
    i8 => 0, 1; i16 => 0, 1; i32 => 0, 1; i64 => 0, 1;
    u8 => 0, 1; u16 => 0, 1; u32 => 0, 1; u64 => 0, 1;
    f32 => 0.0, 1.0; f64 => 0.0, 1.0; bool => false, true;
);

/// Egress a single reduction result as a **numpy scalar of element type `E`**
/// (not a dtype-agnostic Python `int`/`float`). numpy.ma's scalar reductions
/// return a typed numpy scalar (`np.int8`, `np.uint64`, `np.float32`, …) whose
/// `dtype` the dtype-audit pins read; a bare Rust `i8`/`u64`/`f32` egressed via
/// `into_bound_py_any` would collapse to Python `int`/`float` → `int64`/
/// `float64` under `np.asarray`. Building a 1-element `ArrayD<E>` and indexing
/// `[0]` yields a numpy scalar that preserves `E`'s dtype (#858, R-CODE-4/5).
fn scalar_pyobject<'py, E>(py: Python<'py>, value: E) -> PyResult<Bound<'py, PyAny>>
where
    E: ferray_core::Element,
    ArrayD<E>: IntoNumPy<E, IxDyn>,
{
    let arr = ArrayD::<E>::from_vec(IxDyn::new(&[1]), vec![value]).map_err(ferr_to_pyerr)?;
    let py_arr = arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    py_arr.get_item(0)
}

/// Sum the unmasked elements of `m`, widening each to numpy's reduction
/// accumulator `<T as ReduceAcc>::Acc` (narrow int → i64/u64, bool → i64,
/// float unchanged) and folding with `+` — exactly `filled(0).sum()`
/// (`numpy/ma/core.py:5244`), but skipping masked slots rather than adding 0.
fn ma_sum_acc<T>(m: &RustMa<T, IxDyn>) -> <T as ReduceAcc>::Acc
where
    T: ferray_core::Element + ReduceAcc,
    <T as ReduceAcc>::Acc: AccIdentity,
{
    let mut acc = <<T as ReduceAcc>::Acc as AccIdentity>::ZERO;
    for (&v, &mk) in m.data().iter().zip(m.mask().iter()) {
        if !mk {
            acc = acc + v.widen();
        }
    }
    acc
}

/// Product of the unmasked elements of `m`, widened to the reduction
/// accumulator and folded with `*` — `filled(1).prod()`
/// (`numpy/ma/core.py:5303`/`:5340`).
fn ma_prod_acc<T>(m: &RustMa<T, IxDyn>) -> <T as ReduceAcc>::Acc
where
    T: ferray_core::Element + ReduceAcc,
    <T as ReduceAcc>::Acc: AccIdentity,
{
    let mut acc = <<T as ReduceAcc>::Acc as AccIdentity>::ONE;
    for (&v, &mk) in m.data().iter().zip(m.mask().iter()) {
        if !mk {
            acc = acc * v.widen();
        }
    }
    acc
}

/// Minimum (`max=false`) / maximum (`max=true`) unmasked element of `m`, kept
/// in the INPUT dtype `T` (numpy.ma `min`/`max` delegate to
/// `filled(<extreme>).min()` with NO accumulator promotion,
/// `numpy/ma/core.py:5933`). Returns `None` when every element is masked
/// (caller maps that to the `numpy.ma.masked` singleton). NaN-propagating for
/// float dtypes, matching the ferray-ma `Float` reductions.
fn ma_extremum<T>(m: &RustMa<T, IxDyn>, want_max: bool) -> Option<T>
where
    T: ferray_core::Element + Copy + PartialOrd,
{
    let mut acc: Option<T> = None;
    for (&v, &mk) in m.data().iter().zip(m.mask().iter()) {
        if mk {
            continue;
        }
        acc = Some(match acc {
            None => v,
            Some(a) => {
                // NaN-propagating: an unordered comparison keeps the running
                // value if it is already NaN, else adopts the NaN `v`.
                if want_max {
                    if a >= v {
                        a
                    } else if a < v {
                        v
                    } else {
                        a
                    }
                } else if a <= v {
                    a
                } else if a > v {
                    v
                } else {
                    a
                }
            }
        });
    }
    acc
}

/// Coerce a single Python scalar to the masked array's element type `T` by
/// routing it through `numpy.asarray(value, dtype)` (numpy's cast of a
/// `fill_value` / RHS scalar to `self.dtype`, R-CODE-4 — no hard-f64 funnel),
/// raveling to 1-D, and reading the first element via `as_ferray`. `dt` is the
/// array's native dtype name.
fn coerce_scalar<'py, T>(py: Python<'py>, value: &Bound<'py, PyAny>, dt: &str) -> PyResult<T>
where
    T: ferray_numpy_interop::numpy_conv::NpElement + Copy,
{
    let arr = coerce_dtype(py, value, dt)?;
    let raveled = arr.call_method0("ravel")?;
    let view: PyReadonlyArrayDyn<T> = raveled.extract()?;
    let fa = view.as_ferray().map_err(ferr_to_pyerr)?;
    fa.iter()
        .next()
        .copied()
        .ok_or_else(|| PyValueError::new_err("fill value must be a scalar (got an empty array)"))
}

#[pymethods]
impl PyMaskedArray {
    #[new]
    #[pyo3(signature = (data, mask = None, dtype = None))]
    fn py_new<'py>(
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        mask: Option<&Bound<'py, PyAny>>,
        dtype: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Self> {
        // Optional explicit `dtype=` (`fr.ma.array([1,2,3], dtype=np.int32)`),
        // normalized to a canonical numpy dtype name (mirrors numpy's
        // `array(data, dtype=...)`).
        let dt_name: Option<String> = crate::conv::normalize_opt_dtype(py, dtype)?;
        let dt_ref: Option<&str> = dt_name.as_deref();

        // A `ferray.ma.MaskedArray` source is handled directly from its `inner`
        // (no `numpy.asarray` round-trip): numpy's `MaskedArray.__new__` copies
        // the source `_data`/`_mask` (`numpy/ma/core.py:2820`), so a `mask=None`
        // fr→fr round-trip must carry the source mask (and its real-mask /
        // nomask identity AND its native dtype) over verbatim.
        if let Ok(src) = data.extract::<PyMaskedArray>() {
            // An explicit `dtype=` re-casts the source data; without it the
            // source's native dtype variant is preserved.
            if dt_ref.is_some() || mask.is_some() {
                // Combine the explicit mask (if any) with the source mask
                // (keep_mask=True default, `numpy/ma/core.py:3008`).
                let src_mask_opt = src.inner.mask_opt_bits();
                let combined_mask: Option<ArrayD<bool>> = match mask {
                    None => src_mask_opt
                        .map(|bits| ArrayD::<bool>::from_vec(IxDyn::new(&src.inner.shape()), bits))
                        .transpose()
                        .map_err(ferr_to_pyerr)?,
                    Some(m) => {
                        let mask_fa = extract_bool_array(py, m)?;
                        Some(match src_mask_opt {
                            Some(bits) => {
                                let src_fa =
                                    ArrayD::<bool>::from_vec(IxDyn::new(&src.inner.shape()), bits)
                                        .map_err(ferr_to_pyerr)?;
                                or_masks(&mask_fa, &src_fa)?
                            }
                            None => mask_fa,
                        })
                    }
                };
                let inner = match dt_ref {
                    // Re-cast through numpy: egress source data → asarray(dt).
                    Some(_) => dynma_from_input(py, &src.data_pyarray(py)?, dt_ref, combined_mask)?,
                    // Keep native dtype: re-wrap the source variant with the
                    // (possibly combined) mask.
                    None => rewrap_with_mask(&src.inner, combined_mask)?,
                };
                return Ok(Self { inner });
            }
            return Ok(src);
        }

        // numpy distinguishes `nomask` (the singleton, from `ma.array(data)`
        // with no `mask=`) from an explicit all-False mask (`mask=[0, ...]`):
        // `MaskedArray._mask is nomask` for the former and a real bool array
        // for the latter (`numpy/ma/core.py:1468`). We preserve that by routing
        // `mask=None` through the `from_data` nomask sentinel and only
        // materializing a real mask for an explicit `mask=` or a masked source.
        //
        // EXCEPTION (mask round-trip, #851): when `mask=None` but `data` is
        // ITSELF a masked input carrying a REAL mask (a `numpy.ma.MaskedArray`),
        // numpy's `MaskedArray.__new__` copies the source `_mask`
        // (`numpy/ma/core.py:2820`). We mirror that via `source_real_mask`.
        let mask_arg: Option<ArrayD<bool>> = match mask {
            None => source_real_mask(py, data)?,
            Some(m) => {
                let mask_fa = extract_bool_array(py, m)?;
                // keep_mask=True default: OR an explicit mask with the source's
                // own mask when `data` is a masked source (`core.py:3008`).
                Some(match source_real_mask(py, data)? {
                    Some(src_mask) => or_masks(&mask_fa, &src_mask)?,
                    None => mask_fa,
                })
            }
        };
        let inner = dynma_from_input(py, data, dt_ref, mask_arg)?;
        Ok(Self { inner })
    }

    /// Underlying data buffer as a `numpy.ndarray` of the NATIVE dtype.
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.data_pyarray(py)
    }

    /// Mask as a `numpy.ndarray` of bool with the same shape as `data`,
    /// or the `numpy.ma.nomask` singleton when the array has no real mask.
    ///
    /// numpy's `MaskedArray.mask` getter returns `self._mask`, and `getmask`
    /// yields the `nomask` constant ONLY when `_mask is nomask`
    /// (`numpy/ma/core.py:1468`) — i.e. keyed off the real-mask identity, not
    /// the masked-element count. An array built with an explicit `mask=`
    /// (even all-False, e.g. `mask=[0, 0, 0]`) carries a real bool `_mask`,
    /// so `.mask` returns `array([False, ...])`, not the singleton. The
    /// binding mirrors that by branching on `has_real_mask()` (R-DEV-3).
    #[getter]
    fn mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if !self.inner.has_real_mask() {
            return ma_nomask(py);
        }
        let mask = self.inner.mask_bits()?;
        Ok(mask.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    }

    /// The replacement value used for masked positions by `filled()`, in the
    /// array's native dtype (numpy's per-dtype `default_filler`,
    /// `numpy/ma/core.py:166` — `1e20` float / `999999` int / `True` bool).
    #[getter]
    fn fill_value<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match_ma!(&self.inner, m, T => {
            m.fill_value().into_bound_py_any(py)
        })
    }

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(pyo3::types::PyTuple::new(py, self.inner.shape())?.into_any())
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    #[getter]
    fn size(&self) -> usize {
        self.inner.size()
    }

    /// The array's native numpy dtype name (`"int64"`, `"float64"`, `"bool"`,
    /// …) — NOT the literal `"float64"` of the old f64 pin (#853, R-CODE-4).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype_name()
    }

    /// Number of unmasked elements.
    fn count(&self) -> PyResult<usize> {
        self.inner.count()
    }

    /// Sum of unmasked elements (full reduction). An all-masked array
    /// reduces to the `numpy.ma.masked` singleton (`numpy/ma/core.py:5250`).
    /// The result dtype is numpy's reduction accumulator
    /// (`ferray_core::ReduceAcc`): narrow ints → int64, unsigned → uint64,
    /// bool → int64, float unchanged — exactly `filled(0).sum()`
    /// (`numpy/ma/core.py:5244`), NOT a float64 collapse (#858, R-CODE-4).
    fn sum<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.all_masked()? {
            return ma_masked_singleton(py);
        }
        match_ma!(&self.inner, m, T => {
            scalar_pyobject(py, ma_sum_acc(m))
        })
    }

    /// Mean of unmasked elements (full reduction). All-masked reduces to the
    /// `numpy.ma.masked` singleton (`numpy/ma/core.py:5417`). float64.
    fn mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.all_masked()? {
            return ma_masked_singleton(py);
        }
        let v = self.to_f64_ma()?.mean().map_err(ferr_to_pyerr)?;
        Ok(v.into_pyobject(py)?.into_any())
    }

    /// Minimum unmasked element. All-masked reduces to the `numpy.ma.masked`
    /// singleton (`numpy/ma/core.py:5942`). Keeps the INPUT dtype — numpy.ma
    /// `min` delegates to `filled(maximum_fill).min()` with no accumulator
    /// promotion (`numpy/ma/core.py:5933`), so int8 stays int8 (#858).
    fn min<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.all_masked()? {
            return ma_masked_singleton(py);
        }
        match_ma!(&self.inner, m, T => {
            match ma_extremum(m, false) {
                Some(v) => scalar_pyobject(py, v),
                None => ma_masked_singleton(py),
            }
        })
    }

    /// Maximum unmasked element. All-masked reduces to the `numpy.ma.masked`
    /// singleton (`numpy/ma/core.py:6047`). Keeps the INPUT dtype (numpy.ma
    /// `max` = `filled(minimum_fill).max()`, no promotion) (#858).
    fn max<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.all_masked()? {
            return ma_masked_singleton(py);
        }
        match_ma!(&self.inner, m, T => {
            match ma_extremum(m, true) {
                Some(v) => scalar_pyobject(py, v),
                None => ma_masked_singleton(py),
            }
        })
    }

    /// Variance of unmasked elements. All-masked (count==0) reduces to the
    /// `numpy.ma.masked` singleton (numpy's `_var` yields `masked`). Computed
    /// in f64 (numpy promotes integer/bool input to float for `var`/`std`).
    fn var<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.all_masked()? {
            return ma_masked_singleton(py);
        }
        let v = self.to_f64_ma()?.var().map_err(ferr_to_pyerr)?;
        Ok(v.into_pyobject(py)?.into_any())
    }

    /// Standard deviation of unmasked elements. All-masked (count==0) reduces
    /// to the `numpy.ma.masked` singleton. Computed in f64 (promote-to-float).
    fn std<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.all_masked()? {
            return ma_masked_singleton(py);
        }
        let v = self.to_f64_ma()?.std().map_err(ferr_to_pyerr)?;
        Ok(v.into_pyobject(py)?.into_any())
    }

    /// Sum along an axis. Returns a `MaskedArray` (mask flags axes
    /// where every input was masked). Computed in f64 (the ferray-ma axis
    /// reductions are `T: Float`-bound).
    fn sum_axis(&self, axis: usize) -> PyResult<Self> {
        Ok(Self::from_inner(
            self.to_f64_ma()?.sum_axis(axis).map_err(ferr_to_pyerr)?,
        ))
    }

    fn mean_axis(&self, axis: usize) -> PyResult<Self> {
        Ok(Self::from_inner(
            self.to_f64_ma()?.mean_axis(axis).map_err(ferr_to_pyerr)?,
        ))
    }

    fn min_axis(&self, axis: usize) -> PyResult<Self> {
        Ok(Self::from_inner(
            self.to_f64_ma()?.min_axis(axis).map_err(ferr_to_pyerr)?,
        ))
    }

    fn max_axis(&self, axis: usize) -> PyResult<Self> {
        Ok(Self::from_inner(
            self.to_f64_ma()?.max_axis(axis).map_err(ferr_to_pyerr)?,
        ))
    }

    /// Replace masked entries with `fill_value`, return as a regular
    /// `numpy.ndarray` of the array's native dtype. `fill_value` (when given)
    /// is coerced to the array dtype (numpy casts the fill to `self.dtype`),
    /// rather than the old hard-f64 param (R-CODE-4).
    #[pyo3(signature = (fill_value = None))]
    fn filled<'py>(
        &self,
        py: Python<'py>,
        fill_value: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        match_ma!(&self.inner, m, T => {
            let arr: ArrayD<T> = match fill_value {
                Some(v) => {
                    let fv = coerce_scalar::<T>(py, v, self.inner.dtype_name())?;
                    m.filled(fv).map_err(ferr_to_pyerr)?
                }
                None => m.filled_default().map_err(ferr_to_pyerr)?,
            };
            Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
        })
    }

    /// Return a 1-D `numpy.ndarray` of unmasked elements (native dtype).
    fn compressed<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match_ma!(&self.inner, m, T => {
            let arr: Array1<T> = m.compressed().map_err(ferr_to_pyerr)?;
            Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
        })
    }

    fn __repr__(&self) -> String {
        let masked = self
            .inner
            .count()
            .map(|c| self.inner.size() - c)
            .unwrap_or(0);
        format!(
            "MaskedArray(shape={:?}, masked={}, dtype={})",
            self.inner.shape(),
            masked,
            self.inner.dtype_name(),
        )
    }

    fn __len__(&self) -> usize {
        if self.inner.ndim() == 0 {
            0
        } else {
            self.inner.shape()[0]
        }
    }

    /// NumPy `__array__` protocol — return the underlying data verbatim,
    /// keeping the original values at masked positions (NOT the fill value).
    ///
    /// numpy's `MaskedArray.__array__` yields `self._data`, so
    /// `np.asarray(ma)` equals `ma.data` — the masked slots keep their
    /// stored values rather than being substituted. The binding mirrors that
    /// (R-DEV-3); `filled()` is the explicit fill-substituting path.
    ///
    /// numpy 2.x (NEP 56) passes `dtype=` and `copy=` keyword arguments when
    /// driving an object through the `__array__` protocol — `np.array(obj)`
    /// / `np.asarray(obj, dtype=...)` call `obj.__array__(dtype=None,
    /// copy=None)`. The signature accepts both so numpy interop egress
    /// succeeds (verified live: `numpy.ma.MaskedArray.__array__` accepts the
    /// same kwargs). `dtype` casts the returned ndarray (numpy `astype`);
    /// `copy=True` forces an independent copy. The no-arg / bare
    /// `np.asarray(ma)` path is unchanged (still `self._data`).
    #[pyo3(signature = (dtype = None, copy = None))]
    fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<&Bound<'py, PyAny>>,
        copy: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut out = self.data(py)?;
        if let Some(dt) = dtype {
            // `astype(dtype)` returns a fresh, correctly-typed ndarray (and is
            // itself a copy), so the `copy=` request is already satisfied.
            out = out.call_method1("astype", (dt,))?;
            return Ok(out);
        }
        // No dtype: honour `copy=True` by returning an independent copy;
        // `copy=False`/`None` may alias (a freshly-built ndarray is fine).
        if let Some(c) = copy
            && c.is_truthy()?
        {
            out = out.call_method0("copy")?;
        }
        Ok(out)
    }

    /// `MaskedArray.harden_mask()` — force the mask hard, preventing
    /// subsequent assignments from clearing mask bits, and return `self`.
    ///
    /// numpy's method sets `self._hardmask = True; return self`
    /// (`numpy/ma/core.py:3635`). The `&mut self` receiver mutates the
    /// underlying `MaskedArray` in the pyclass cell in place (NOT a clone),
    /// so the caller's object becomes hard. The method returns `None`;
    /// callers wanting the array back use the module-level
    /// `ferray.ma.harden_mask(a)` which returns `a`.
    fn harden_mask(&mut self) -> PyResult<()> {
        match_ma!(&mut self.inner, m, T => { m.harden_mask().map_err(ferr_to_pyerr) })
    }

    /// `MaskedArray.soften_mask()` — force the mask soft, allowing subsequent
    /// assignments to clear mask bits (`numpy/ma/core.py:3653`). Mutates in
    /// place via `&mut self`.
    fn soften_mask(&mut self) -> PyResult<()> {
        match_ma!(&mut self.inner, m, T => { m.soften_mask().map_err(ferr_to_pyerr) })
    }

    /// `MaskedArray.hardmask` property — whether the mask is hard
    /// (`numpy/ma/core.py:3656`).
    #[getter]
    fn hardmask(&self) -> bool {
        self.inner.is_hard_mask()
    }

    /// `MaskedArray.put(indices, values, mode='raise')` — set flat-indexed
    /// positions in place (`numpy/ma/core.py:4837`). Mutates `self` via
    /// `&mut self`; returns `None` like numpy. Values are coerced to the
    /// array's native dtype (R-CODE-4 — no hard-f64 funnel).
    #[pyo3(signature = (indices, values, mode = "raise"))]
    fn put<'py>(
        &mut self,
        py: Python<'py>,
        indices: &Bound<'py, PyAny>,
        values: &Bound<'py, PyAny>,
        mode: &str,
    ) -> PyResult<()> {
        let idx = extract_indices(py, indices)?;
        let put_mode = fma::PutMode::parse(mode).map_err(ferr_to_pyerr)?;
        let dt = self.inner.dtype_name();
        let vmask = values_mask(py, values)?;
        match_ma!(&mut self.inner, m, T => {
            let vals = values_data::<T>(py, values, dt)?;
            m.put(&idx, &vals, vmask.as_deref(), put_mode)
                .map_err(put_err_to_pyerr)
        })
    }

    /// `MaskedArray.putmask`-style in-place conditional assignment used by the
    /// module-level `ferray.ma.putmask(a, mask, values)`
    /// (`numpy/ma/core.py:7537`). Mutates `self` via `&mut self`; returns
    /// `None` like numpy. Values are coerced to the array's native dtype.
    fn putmask<'py>(
        &mut self,
        py: Python<'py>,
        mask: &Bound<'py, PyAny>,
        values: &Bound<'py, PyAny>,
    ) -> PyResult<()> {
        let mask_arr = extract_bool_array(py, mask)?;
        let mask_flat: Vec<bool> = mask_arr.iter().copied().collect();
        let dt = self.inner.dtype_name();
        let vmask = values_mask(py, values)?;
        match_ma!(&mut self.inner, m, T => {
            let vals = values_data::<T>(py, values, dt)?;
            m.putmask(&mask_flat, &vals, vmask.as_deref())
                .map_err(ferr_to_pyerr)
        })
    }

    /// `MaskedArray.__getitem__(indx)` — subscripting
    /// (`numpy/ma/core.py:3277`).
    ///
    /// numpy computes `dout = self.data[indx]` and `mout = self._mask[indx]`,
    /// then: a SCALAR result that is masked returns the `numpy.ma.masked`
    /// singleton, an unmasked scalar returns the plain value, and a non-scalar
    /// result (slice / fancy / bool / a row of a 2-D array) returns a new
    /// `MaskedArray` carrying the gathered data and mask (`core.py:3334`-`3400`).
    ///
    /// The index is resolved against numpy itself via a flat-position grid
    /// `numpy.arange(size).reshape(shape)[indx]`: numpy raises its exact
    /// `IndexError` for out-of-bounds / negative-out-of-bounds, a 0-d result
    /// flags the scalar case, and a higher-rank result yields both the result
    /// shape and the gathered flat C-order positions to gather over the
    /// library's contiguous `data()`/`mask()` buffers (R-CODE-4 — numpy's
    /// indexing semantics are reused, not reimplemented).
    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        indx: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let resolved = resolve_index(py, &self.inner.shape(), indx)?;
        let has_real = self.inner.has_real_mask();
        let mask_flat: Option<Vec<bool>> = self.inner.mask_opt_bits();
        match_ma!(&self.inner, m, T => {
            let data: Vec<T> = m.data().iter().copied().collect();
            match resolved {
                ResolvedIndex::Scalar(pos) => {
                    let masked = mask_flat
                        .as_ref()
                        .is_some_and(|mk| mk.get(pos).copied().unwrap_or(false));
                    if masked {
                        ma_masked_singleton(py)
                    } else {
                        data[pos].into_bound_py_any(py)
                    }
                }
                ResolvedIndex::Gather { positions, shape } => {
                    let sub_data: Vec<T> = positions.iter().map(|&p| data[p]).collect();
                    let data_fa = ArrayD::<T>::from_vec(IxDyn::new(&shape), sub_data)
                        .map_err(ferr_to_pyerr)?;
                    // numpy: `if _mask is not nomask: mout = _mask[indx]`
                    // (`numpy/ma/core.py:3317`). When the SOURCE carries a REAL
                    // mask, the result's mask is ALWAYS `_mask[indx]` (the #849
                    // pattern, keyed off `has_real_mask()`, NEVER off any-True
                    // bits); a nomask source gathers to a nomask sub-array.
                    let inner = match (has_real, mask_flat.as_ref()) {
                        (true, Some(mk)) => {
                            let sub_mask: Vec<bool> = positions.iter().map(|&p| mk[p]).collect();
                            let mask_fa = ArrayD::<bool>::from_vec(IxDyn::new(&shape), sub_mask)
                                .map_err(ferr_to_pyerr)?;
                            T::wrap(RustMa::new(data_fa, mask_fa).map_err(ferr_to_pyerr)?)
                        }
                        // `has_real` is derived from `has_real_mask()`, the same
                        // source as `mask_opt_bits()`, so a real mask always
                        // yields `Some`; the `None` arm gathers to a nomask
                        // sub-array rather than panicking (R-CODE-2/R-APG-1).
                        _ => T::wrap(RustMa::from_data(data_fa).map_err(ferr_to_pyerr)?),
                    };
                    Ok(Py::new(py, Self::from_dynma(inner))?.into_bound(py).into_any())
                }
            }
        })
    }

    /// `MaskedArray.__setitem__(indx, value)` — in-place subscript assignment
    /// (`numpy/ma/core.py:3406`).
    ///
    /// - `a[i] = numpy.ma.masked` masks the targeted positions, materializing a
    ///   real mask if the array was a nomask sentinel
    ///   (`core.py:3427`-`3436`; #848/#849).
    /// - Otherwise the new value's data is written. Under a SOFT mask the
    ///   targeted positions are unmasked except where the value itself is masked
    ///   (`core.py:3450`-`3457`, `_mask[indx] = mval`). Under a HARD mask the
    ///   existing mask is never cleared and already-masked positions keep their
    ///   data (`core.py:3461`-`3472`, `copyto(..., where=~mindx)`); the
    ///   hard-mask suppression of the clear is already enforced by
    ///   `set_mask_flat`.
    ///
    /// The index is resolved via the same flat-position grid as `__getitem__`,
    /// and the right-hand side is broadcast to the resolved result shape with
    /// numpy (`numpy.broadcast_to`) so `a[1:3] = [7, 8]`, `a[a > 1] = 0`, and a
    /// masked-array RHS all align element-for-element with the gathered target
    /// positions.
    fn __setitem__<'py>(
        &mut self,
        py: Python<'py>,
        indx: &Bound<'py, PyAny>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<()> {
        let resolved = resolve_index(py, &self.inner.shape(), indx)?;
        let (positions, result_shape) = match resolved {
            ResolvedIndex::Scalar(p) => (vec![p], Vec::new()),
            ResolvedIndex::Gather { positions, shape } => (positions, shape),
        };

        // `a[i] = numpy.ma.masked` — mask the targeted positions. `set_mask_flat`
        // materializes a real mask on a nomask sentinel and is a no-op clear
        // under a hard mask, exactly mirroring numpy's `_mask[indx] = True`
        // after a lazy `make_mask_none` (core.py:3427).
        let masked_singleton = ma_masked_singleton(py)?;
        if value.is(&masked_singleton) {
            for &p in &positions {
                set_mask_flat_dyn(&mut self.inner, p, true)?;
            }
            return Ok(());
        }

        let is_hard = self.inner.is_hard_mask();
        let had_real_mask = self.inner.has_real_mask();
        let existing: Option<Vec<bool>> = self.inner.mask_opt_bits();
        let dt = self.inner.dtype_name();
        // The RHS mask (broadcast) — dtype-independent.
        let vmask = broadcast_value_mask(py, value, &result_shape, positions.len())?;

        // Write the data in the array's native dtype (R-CODE-4 — the RHS is
        // coerced to `self.dtype`, not funneled through f64).
        match_ma!(&mut self.inner, m, T => {
            let vals = broadcast_value_data::<T>(py, value, &result_shape, positions.len(), dt)?;
            let buf = m
                .data_mut()
                .ok_or_else(|| PyValueError::new_err("masked array data is not contiguous"))?;
            for (k, &p) in positions.iter().enumerate() {
                if is_hard {
                    // Hard mask: an already-masked position keeps its data.
                    let was_masked = existing.as_ref().map(|e| e[p]).unwrap_or(false);
                    if !was_masked {
                        buf[p] = vals[k];
                    }
                } else {
                    buf[p] = vals[k];
                }
            }
        });

        // Mask updates after the data write. Under a soft mask each written
        // position is unmasked unless the RHS value was itself masked; a hard
        // mask suppresses clears (set_mask_flat enforces the gate).
        if vmask.is_some() || had_real_mask || is_hard {
            for (k, &p) in positions.iter().enumerate() {
                let rhs_masked = vmask.as_ref().map(|vm| vm[k]).unwrap_or(false);
                set_mask_flat_dyn(&mut self.inner, p, rhs_masked)?;
            }
        }
        Ok(())
    }

    /// Unary negation (`-a`), `numpy.ma`'s `__neg__` → `_MaskedUnaryOperation`
    /// over `numpy.negative` (`numpy/ma/core.py:955`, `:1010`).
    ///
    /// The op is applied to the FULL native-dtype data buffer (masked
    /// positions included — numpy transforms the data under the mask:
    /// `(-np.ma.array([-1, 2, -3], mask=[0, 1, 0])).data == [1, -2, 3]`,
    /// verified live numpy 2.4.4), and the result carries the operand's mask
    /// via [`carry_unary_mask`] (a REAL mask, even from a nomask source —
    /// numpy's `getmaskarray(a)` materializes an all-False mask, so
    /// `np.ma.getmask(-n) is np.ma.nomask` is `False`; verified live).
    ///
    /// Dtype is preserved (R-CODE-4): float → `negative` (Float),
    /// signed/unsigned int → `negative_int` (C-wrapping, so `-uint8(2)`
    /// wraps to `254`). `bool` raises `TypeError`, matching numpy 2.4
    /// (`-np.ma.array([True]) → TypeError: boolean negative … not supported`).
    fn __neg__(&self) -> PyResult<Self> {
        use ferray_ufunc::ops::arithmetic::{negative, negative_int};
        let inner = match &self.inner {
            DynMa::F32(m) => carry_unary_mask(m, negative(m.data()).map_err(ferr_to_pyerr)?)?,
            DynMa::F64(m) => carry_unary_mask(m, negative(m.data()).map_err(ferr_to_pyerr)?)?,
            DynMa::I8(m) => carry_unary_mask(m, negative_int(m.data()).map_err(ferr_to_pyerr)?)?,
            DynMa::I16(m) => carry_unary_mask(m, negative_int(m.data()).map_err(ferr_to_pyerr)?)?,
            DynMa::I32(m) => carry_unary_mask(m, negative_int(m.data()).map_err(ferr_to_pyerr)?)?,
            DynMa::I64(m) => carry_unary_mask(m, negative_int(m.data()).map_err(ferr_to_pyerr)?)?,
            DynMa::U8(m) => carry_unary_mask(m, negative_int(m.data()).map_err(ferr_to_pyerr)?)?,
            DynMa::U16(m) => carry_unary_mask(m, negative_int(m.data()).map_err(ferr_to_pyerr)?)?,
            DynMa::U32(m) => carry_unary_mask(m, negative_int(m.data()).map_err(ferr_to_pyerr)?)?,
            DynMa::U64(m) => carry_unary_mask(m, negative_int(m.data()).map_err(ferr_to_pyerr)?)?,
            DynMa::Bool(_) => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "numpy boolean negative, the `-` operator, is not supported, use the `~` \
                     operator or the logical_not function instead",
                ));
            }
        };
        Ok(Self::from_dynma(inner))
    }

    /// Unary plus (`+a`), `numpy.ma`'s `__pos__` → `numpy.positive`. Returns a
    /// mask-preserving copy (identity transform on the data); dtype preserved.
    ///
    /// `bool` raises `TypeError` — numpy 2.4 has no `positive` loop for bool
    /// (`+np.ma.array([True]) → numpy.exceptions._UFuncNoLoopError`, a
    /// `TypeError` subclass; verified live).
    fn __pos__(&self) -> PyResult<Self> {
        use ferray_ufunc::ops::arithmetic::positive;
        let inner = match &self.inner {
            DynMa::F32(m) => carry_unary_mask(m, positive(m.data()).map_err(ferr_to_pyerr)?)?,
            DynMa::F64(m) => carry_unary_mask(m, positive(m.data()).map_err(ferr_to_pyerr)?)?,
            DynMa::I8(m) => carry_unary_mask(m, m.data().clone())?,
            DynMa::I16(m) => carry_unary_mask(m, m.data().clone())?,
            DynMa::I32(m) => carry_unary_mask(m, m.data().clone())?,
            DynMa::I64(m) => carry_unary_mask(m, m.data().clone())?,
            DynMa::U8(m) => carry_unary_mask(m, m.data().clone())?,
            DynMa::U16(m) => carry_unary_mask(m, m.data().clone())?,
            DynMa::U32(m) => carry_unary_mask(m, m.data().clone())?,
            DynMa::U64(m) => carry_unary_mask(m, m.data().clone())?,
            DynMa::Bool(_) => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "the numpy boolean positive, the `+` operator, is not supported",
                ));
            }
        };
        Ok(Self::from_dynma(inner))
    }

    /// Absolute value (`abs(a)`), `numpy.ma`'s `__abs__` → `numpy.absolute`,
    /// mask-preserving, dtype-preserving.
    ///
    /// float → `absolute` (Float SIMD path); signed int → `absolute_int`
    /// (C-wrapping, so `abs(int8::MIN)` wraps like numpy); unsigned int →
    /// identity (numpy `abs(uint)` is the value unchanged); `bool` → identity
    /// (`abs(np.ma.array([True, False])).dtype == bool`; `abs(True) == True`,
    /// verified live numpy 2.4.4).
    fn __abs__(&self) -> PyResult<Self> {
        use ferray_ufunc::ops::arithmetic::{absolute, absolute_int};
        let inner = match &self.inner {
            DynMa::F32(m) => carry_unary_mask(m, absolute(m.data()).map_err(ferr_to_pyerr)?)?,
            DynMa::F64(m) => carry_unary_mask(m, absolute(m.data()).map_err(ferr_to_pyerr)?)?,
            DynMa::I8(m) => carry_unary_mask(m, absolute_int(m.data()).map_err(ferr_to_pyerr)?)?,
            DynMa::I16(m) => carry_unary_mask(m, absolute_int(m.data()).map_err(ferr_to_pyerr)?)?,
            DynMa::I32(m) => carry_unary_mask(m, absolute_int(m.data()).map_err(ferr_to_pyerr)?)?,
            DynMa::I64(m) => carry_unary_mask(m, absolute_int(m.data()).map_err(ferr_to_pyerr)?)?,
            DynMa::U8(m) => carry_unary_mask(m, m.data().clone())?,
            DynMa::U16(m) => carry_unary_mask(m, m.data().clone())?,
            DynMa::U32(m) => carry_unary_mask(m, m.data().clone())?,
            DynMa::U64(m) => carry_unary_mask(m, m.data().clone())?,
            DynMa::Bool(m) => carry_unary_mask(m, m.data().clone())?,
        };
        Ok(Self::from_dynma(inner))
    }

    /// Bitwise NOT (`~a`), `numpy.ma`'s `__invert__` → `numpy.invert`,
    /// mask-preserving, dtype-preserving.
    ///
    /// Defined for integer + bool dtypes (`~int8(-1) == 0` two's-complement;
    /// `~np.ma.array([True, False]) == [False, True]`). FLOAT dtypes raise
    /// `TypeError` — numpy 2.4 has no `invert` loop for float
    /// (`~np.ma.array([1.0]) → TypeError: ufunc 'invert' not supported`;
    /// verified live).
    fn __invert__(&self) -> PyResult<Self> {
        use ferray_ufunc::ops::bitwise::invert;
        let inner = match &self.inner {
            DynMa::Bool(m) => carry_unary_mask(m, invert(m.data()).map_err(ferr_to_pyerr)?)?,
            DynMa::I8(m) => carry_unary_mask(m, invert(m.data()).map_err(ferr_to_pyerr)?)?,
            DynMa::I16(m) => carry_unary_mask(m, invert(m.data()).map_err(ferr_to_pyerr)?)?,
            DynMa::I32(m) => carry_unary_mask(m, invert(m.data()).map_err(ferr_to_pyerr)?)?,
            DynMa::I64(m) => carry_unary_mask(m, invert(m.data()).map_err(ferr_to_pyerr)?)?,
            DynMa::U8(m) => carry_unary_mask(m, invert(m.data()).map_err(ferr_to_pyerr)?)?,
            DynMa::U16(m) => carry_unary_mask(m, invert(m.data()).map_err(ferr_to_pyerr)?)?,
            DynMa::U32(m) => carry_unary_mask(m, invert(m.data()).map_err(ferr_to_pyerr)?)?,
            DynMa::U64(m) => carry_unary_mask(m, invert(m.data()).map_err(ferr_to_pyerr)?)?,
            DynMa::F32(_) | DynMa::F64(_) => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "ufunc 'invert' not supported for the input types, and the inputs could not \
                     be safely coerced to any supported types according to the casting rule \
                     ''safe''",
                ));
            }
        };
        Ok(Self::from_dynma(inner))
    }

    // -- Binary arithmetic dunders (#862, REQ-1 R-A + REQ-6 R-F) --------------
    //
    // Each delegates to [`binary_op`] in either direction. `self` is always the
    // receiver; `reflected = true` computes `other OP self` (numpy.ma's
    // `__radd__` → `add(other, self)`, `numpy/ma/core.py:4322`). The compute,
    // dtype promotion (NEP-50 via `numpy.result_type`), mask union, and
    // domain masking are all done once in `binary_op` (mirroring
    // `_MaskedBinaryOperation.__call__` / `_DomainedBinaryOperation.__call__`,
    // `numpy/ma/core.py:1062` / `:1207`).

    /// `a + b` — `numpy.ma`'s `__add__` → `add` (`numpy/ma/core.py:4313`).
    fn __add__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op(py, &self.inner, other, BinOp::Add, false)
    }
    /// `b + a` — reflected `__radd__` → `add(other, self)` (`:4322`).
    fn __radd__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op(py, &self.inner, other, BinOp::Add, true)
    }
    /// `a - b` — `__sub__` → `subtract` (`numpy/ma/core.py:4332`).
    fn __sub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op(py, &self.inner, other, BinOp::Sub, false)
    }
    /// `b - a` — reflected `__rsub__` → `subtract(other, self)`.
    fn __rsub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op(py, &self.inner, other, BinOp::Sub, true)
    }
    /// `a * b` — `__mul__` → `multiply` (`numpy/ma/core.py:4342`).
    fn __mul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op(py, &self.inner, other, BinOp::Mul, false)
    }
    /// `b * a` — reflected `__rmul__` → `multiply(other, self)`.
    fn __rmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op(py, &self.inner, other, BinOp::Mul, true)
    }
    /// `a / b` — `__truediv__` → `true_divide` (`numpy/ma/core.py:4362`).
    /// True (float) division: integer operands promote to float64 (NEP-50),
    /// and divisor-zero / non-finite positions are domain-masked
    /// (`_DomainedBinaryOperation`, `numpy/ma/core.py:1207`).
    fn __truediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op(py, &self.inner, other, BinOp::TrueDiv, false)
    }
    /// `b / a` — reflected `__rtruediv__` → `true_divide(other, self)`.
    fn __rtruediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op(py, &self.inner, other, BinOp::TrueDiv, true)
    }
    /// `a // b` — `__floordiv__` → `floor_divide` (`numpy/ma/core.py:4378`),
    /// divisor-zero domain-masked.
    fn __floordiv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op(py, &self.inner, other, BinOp::FloorDiv, false)
    }
    /// `b // a` — reflected `__rfloordiv__` → `floor_divide(other, self)`.
    fn __rfloordiv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op(py, &self.inner, other, BinOp::FloorDiv, true)
    }
    /// `a % b` — `__mod__` → `remainder` (`numpy/ma/core.py:4386`),
    /// divisor-zero domain-masked.
    fn __mod__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op(py, &self.inner, other, BinOp::Mod, false)
    }
    /// `b % a` — reflected `__rmod__` → `remainder(other, self)`.
    fn __rmod__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        binary_op(py, &self.inner, other, BinOp::Mod, true)
    }
    /// `a ** b` — `__pow__` → `power` (`numpy/ma/core.py:4394`). The optional
    /// ternary-pow `modulo` argument is unsupported (numpy.ma ignores it too).
    fn __pow__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        modulo: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        if !modulo.is_none() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "pow() 3rd argument not allowed unless all arguments are integers",
            ));
        }
        binary_op(py, &self.inner, other, BinOp::Pow, false)
    }
    /// `b ** a` — reflected `__rpow__` → `power(other, self)`.
    fn __rpow__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        modulo: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        if !modulo.is_none() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "pow() 3rd argument not allowed unless all arguments are integers",
            ));
        }
        binary_op(py, &self.inner, other, BinOp::Pow, true)
    }
}

/// Wrap a unary-op result (already in the source's native dtype `T`) back into
/// a `DynMa`, carrying the operand's mask.
///
/// Mirrors numpy's `_MaskedUnaryOperation.__call__` (`numpy/ma/core.py:1010`):
/// `m = getmaskarray(a)` materializes a REAL all-False mask even for a nomask
/// operand, so `np.ma.getmask(-n) is np.ma.nomask` is `False` (verified live
/// numpy 2.4.4). We mirror that exactly by carrying `fma::getmaskarray(src)`
/// (always a real mask of the operand shape) onto the transformed data via
/// `RustMa::new`. The result data is the op applied to the WHOLE buffer
/// (masked positions transformed too), matching numpy's data-under-the-mask.
fn carry_unary_mask<T>(src: &RustMa<T, IxDyn>, data: ArrayD<T>) -> PyResult<DynMa>
where
    T: ferray_core::Element + WrapDyn,
{
    let mask = fma::getmaskarray(src).map_err(ferr_to_pyerr)?;
    let out = RustMa::new(data, mask).map_err(ferr_to_pyerr)?;
    Ok(T::wrap(out))
}

/// The seven `numpy.ma` binary arithmetic operators, each delegating to the
/// matching `ferray_ufunc` op (`numpy/ma/core.py:1290`-`:1324`). `TrueDiv`,
/// `FloorDiv`, and `Mod` are the *domained* ops (`_DomainedBinaryOperation`):
/// they additionally mask divisor-zero / non-finite result positions.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BinOp {
    Add,
    Sub,
    Mul,
    TrueDiv,
    FloorDiv,
    Mod,
    Pow,
}

impl BinOp {
    /// Whether this op is *domained* (`/`, `//`, `%`) — masks the result where
    /// the divisor is zero / the result is non-finite, and ALWAYS materializes
    /// a real mask even from two nomask operands (`_DomainedBinaryOperation`,
    /// `numpy/ma/core.py:1207`). The non-domained ops (`+`, `-`, `*`, `**`)
    /// keep `nomask` when both operands are nomask (`numpy/ma/core.py:1077`).
    fn is_domained(self) -> bool {
        matches!(self, BinOp::TrueDiv | BinOp::FloorDiv | BinOp::Mod)
    }
}

/// Compute one of the seven masked binary arithmetic operators, mirroring
/// numpy.ma's `_MaskedBinaryOperation.__call__` (`numpy/ma/core.py:1062`) and,
/// for `/`/`//`/`%`, `_DomainedBinaryOperation.__call__` (`:1207`).
///
/// Operands are materialized as ndarrays exactly as numpy.ma does
/// (`getdata(b) == np.asarray(b)` — a Python scalar becomes its natural
/// `int64`/`float64` array, NOT an NEP-50 weak scalar, so `int8_ma + 1` is
/// `int64`, verified live numpy 2.4.5). The operand-common dtype is
/// `numpy.result_type(left_data, right_data)` and the op computes over both
/// operands cast to it, via the matching `ferray_ufunc` op (NEP-50 promotion
/// to float for `/`; integer dtype kept for `//`/`%`/`**`). `reflected`
/// computes `other OP self` (the `__r*__` dunders, `:4322`).
///
/// Mask: `union(getmaskarray(left), getmaskarray(right))` (broadcast),
/// additionally OR'd with the divisor-zero / non-finite domain mask for the
/// domained ops. Masked-position DATA is reverted to the left operand's data
/// `da` (`np.copyto(result, da, where=m)`, `:1096` / the `result += m*da`
/// reconstruction at `:1233`).
fn binary_op(
    py: Python<'_>,
    recv: &DynMa,
    other: &Bound<'_, PyAny>,
    op: BinOp,
    reflected: bool,
) -> PyResult<PyMaskedArray> {
    // Receiver operand as (ndarray, mask, has-real-mask).
    let recv_data = dynma_data_pyarray(py, recv)?;
    let recv_mask = recv.mask_bits()?;
    let recv_real = recv.has_real_mask();

    // Other operand: data as `np.asarray(other)` (native dtype, no weak
    // scalar) + its full bool mask. A `fr.ma.MaskedArray` operand is read from
    // its `inner` (numpy.ma's `getmaskarray` would NOT recognize it); a
    // `numpy.ma.MaskedArray` / plain array-like uses `numpy.ma.getmaskarray`
    // (all-False for a plain array-like, the source mask otherwise).
    let (other_data, other_mask, other_real) = if let Ok(m) = other.extract::<PyMaskedArray>() {
        (
            m.data_pyarray(py)?,
            m.inner.mask_bits()?,
            m.inner.has_real_mask(),
        )
    } else {
        let data = crate::conv::as_ndarray(py, other)?;
        let np_ma = py.import("numpy")?.getattr("ma")?;
        let mask_obj = np_ma.call_method1("getmaskarray", (other,))?;
        let mask = extract_bool_array(py, &mask_obj)?;
        let real = source_real_mask(py, other)?.is_some();
        (data, mask, real)
    };

    // Left/right ordering: reflected ops compute `other OP self`.
    let (ld, lm, l_real, rd, rm, r_real) = if reflected {
        (
            &other_data,
            other_mask,
            other_real,
            &recv_data,
            recv_mask,
            recv_real,
        )
    } else {
        (
            &recv_data,
            recv_mask,
            recv_real,
            &other_data,
            other_mask,
            other_real,
        )
    };

    // Operand-common dtype = numpy.result_type(left_data, right_data)
    // (NEP-50, matching numpy.ma materializing both operands as arrays first).
    let operand_dt = crate::conv::binary_result_dtype(py, ld, rd)?;

    // Cast each operand's data to the common dtype and wrap as a same-dtype
    // DynMa carrying its mask. `build_dynma` rejects complex/structured.
    let left = build_dynma(&coerce_dtype(py, ld, &operand_dt)?, Some(lm))?;
    let right = build_dynma(&coerce_dtype(py, rd, &operand_dt)?, Some(rm))?;

    // Broadcast the two operand shapes (numpy raises ValueError on mismatch).
    let bshape = ferray_core::dimension::broadcast::broadcast_shapes(&left.shape(), &right.shape())
        .map_err(|_| {
            PyValueError::new_err(format!(
                "operands could not be broadcast together with shapes {:?} {:?}",
                left.shape(),
                right.shape()
            ))
        })?;

    // Compute data + the union/domain mask + the masked-position `da` revert,
    // dispatched over the common operand dtype.
    let computed = compute_binary(&left, &right, &bshape, op)?;

    // Real-mask identity: domained ops always materialize a real mask; the
    // additive/multiplicative ops keep nomask only when BOTH operands are
    // nomask (`numpy/ma/core.py:1077`).
    let want_real = op.is_domained() || l_real || r_real || computed.domain_any;
    let inner = if want_real {
        rewrap_dynma_with_mask(computed.data, computed.mask)?
    } else {
        strip_mask(computed.data)?
    };
    Ok(PyMaskedArray::from_dynma(inner))
}

/// The product of [`compute_binary`]: the result data (already reverted to the
/// left operand at masked positions) and the full result mask (operand union
/// `|` domain). `domain_any` records whether the domain contributed any masked
/// position (so a non-domained op can still tell whether a real mask is owed).
struct ComputedBinary {
    data: DynMa,
    mask: ArrayD<bool>,
    domain_any: bool,
}

/// Broadcast both operands' data + masks to `bshape`, run `op` over the common
/// operand dtype, and assemble the result data (masked positions reverted to
/// the left operand) and the union/domain mask.
///
/// `left` and `right` are the SAME `DynMa` variant (both cast to the
/// operand-common dtype upstream). The result dtype follows numpy: `/` promotes
/// integer/bool operands to `float64` (`TrueDivide`); `//`/`%`/`**` keep the
/// integer dtype (and promote `bool` to `int8`, matching numpy.ma); `+`/`-`/`*`
/// keep the operand dtype.
fn compute_binary(
    left: &DynMa,
    right: &DynMa,
    bshape: &[usize],
    op: BinOp,
) -> PyResult<ComputedBinary> {
    use ferray_core::manipulation::broadcast_to;
    use ferray_ufunc::ops::arithmetic::{
        add, divide, floor_divide, floor_divide_int, mod_, mod_int, multiply, power, power_int,
        subtract,
    };

    // Broadcast the two bool masks to the result shape and union them.
    let lmask = broadcast_to(&left.mask_bits()?, bshape).map_err(ferr_to_pyerr)?;
    let rmask = broadcast_to(&right.mask_bits()?, bshape).map_err(ferr_to_pyerr)?;
    let union: Vec<bool> = lmask
        .iter()
        .zip(rmask.iter())
        .map(|(&a, &b)| a || b)
        .collect();
    let n = union.len();

    // A float arm: `+`/`-`/`*` keep the float dtype; `/`/`//`/`%` are domained
    // (divisor-zero / non-finite masked); `**` is `power`. The result variant
    // equals the operand float variant (`numpy/ma/core.py:1290`-`:1324`).
    macro_rules! float_arm {
        ($l:expr, $r:expr, $T:ty, $variant:ident) => {{
            let la: ArrayD<$T> = broadcast_to($l.data(), bshape).map_err(ferr_to_pyerr)?;
            let ra: ArrayD<$T> = broadcast_to($r.data(), bshape).map_err(ferr_to_pyerr)?;
            let mut out = match op {
                BinOp::Add => add(&la, &ra).map_err(ferr_to_pyerr)?,
                BinOp::Sub => subtract(&la, &ra).map_err(ferr_to_pyerr)?,
                BinOp::Mul => multiply(&la, &ra).map_err(ferr_to_pyerr)?,
                BinOp::TrueDiv => divide(&la, &ra).map_err(ferr_to_pyerr)?,
                BinOp::FloorDiv => floor_divide(&la, &ra).map_err(ferr_to_pyerr)?,
                BinOp::Mod => mod_(&la, &ra).map_err(ferr_to_pyerr)?,
                BinOp::Pow => power(&la, &ra).map_err(ferr_to_pyerr)?,
            };
            let mut domain = vec![false; n];
            if op.is_domained() {
                let zero = <$T as ferray_core::Element>::zero();
                for (i, (&div, r)) in ra.iter().zip(out.iter()).enumerate() {
                    if div == zero || !r.is_finite() {
                        domain[i] = true;
                    }
                }
            }
            let full = or_vecs(&union, &domain);
            revert_masked(
                out.as_slice_mut().ok_or_else(non_contig)?,
                la.as_slice().ok_or_else(non_contig)?,
                &full,
            );
            let data = Array::from_vec(IxDyn::new(bshape), out.iter().copied().collect())
                .map_err(ferr_to_pyerr)?;
            (
                DynMa::$variant(RustMa::from_data(data).map_err(ferr_to_pyerr)?),
                domain,
            )
        }};
    }

    // An integer arm: `+`/`-`/`*` wrap; `//`/`%` keep the int dtype with
    // divisor-zero masked; `**` is `power_int`; `/` promotes to `float64`
    // (`TrueDivide::Output == f64`) and is domain-masked on non-finite results.
    macro_rules! int_arm {
        ($l:expr, $r:expr, $T:ty, $variant:ident) => {{
            let la: ArrayD<$T> = broadcast_to($l.data(), bshape).map_err(ferr_to_pyerr)?;
            let ra: ArrayD<$T> = broadcast_to($r.data(), bshape).map_err(ferr_to_pyerr)?;
            let zero = <$T as ferray_core::Element>::zero();
            if op == BinOp::TrueDiv {
                let mut out = divide(&la, &ra).map_err(ferr_to_pyerr)?;
                let mut domain = vec![false; n];
                for (i, (&div, r)) in ra.iter().zip(out.iter()).enumerate() {
                    if div == zero || !r.is_finite() {
                        domain[i] = true;
                    }
                }
                let la_f64: Vec<f64> = la.iter().map(|&v| dyn_to_f64::<$T>(v)).collect();
                let full = or_vecs(&union, &domain);
                revert_masked(out.as_slice_mut().ok_or_else(non_contig)?, &la_f64, &full);
                let data = Array::from_vec(IxDyn::new(bshape), out.iter().copied().collect())
                    .map_err(ferr_to_pyerr)?;
                (
                    DynMa::F64(RustMa::from_data(data).map_err(ferr_to_pyerr)?),
                    domain,
                )
            } else {
                let mut domain = vec![false; n];
                if op.is_domained() {
                    for (i, &div) in ra.iter().enumerate() {
                        if div == zero {
                            domain[i] = true;
                        }
                    }
                }
                let mut out = match op {
                    BinOp::Add => add(&la, &ra).map_err(ferr_to_pyerr)?,
                    BinOp::Sub => subtract(&la, &ra).map_err(ferr_to_pyerr)?,
                    BinOp::Mul => multiply(&la, &ra).map_err(ferr_to_pyerr)?,
                    BinOp::FloorDiv => floor_divide_int(&la, &ra).map_err(ferr_to_pyerr)?,
                    BinOp::Mod => mod_int(&la, &ra).map_err(ferr_to_pyerr)?,
                    BinOp::Pow => power_int(&la, &ra).map_err(ferr_to_pyerr)?,
                    BinOp::TrueDiv => add(&la, &ra).map_err(ferr_to_pyerr)?, // unreachable
                };
                let full = or_vecs(&union, &domain);
                revert_masked(
                    out.as_slice_mut().ok_or_else(non_contig)?,
                    la.as_slice().ok_or_else(non_contig)?,
                    &full,
                );
                let data = Array::from_vec(IxDyn::new(bshape), out.iter().copied().collect())
                    .map_err(ferr_to_pyerr)?;
                (
                    DynMa::$variant(RustMa::from_data(data).map_err(ferr_to_pyerr)?),
                    domain,
                )
            }
        }};
    }

    let (data, domain): (DynMa, Vec<bool>) = match (left, right) {
        (DynMa::F32(l), DynMa::F32(r)) => float_arm!(l, r, f32, F32),
        (DynMa::F64(l), DynMa::F64(r)) => float_arm!(l, r, f64, F64),
        (DynMa::I8(l), DynMa::I8(r)) => int_arm!(l, r, i8, I8),
        (DynMa::I16(l), DynMa::I16(r)) => int_arm!(l, r, i16, I16),
        (DynMa::I32(l), DynMa::I32(r)) => int_arm!(l, r, i32, I32),
        (DynMa::I64(l), DynMa::I64(r)) => int_arm!(l, r, i64, I64),
        (DynMa::U8(l), DynMa::U8(r)) => int_arm!(l, r, u8, U8),
        (DynMa::U16(l), DynMa::U16(r)) => int_arm!(l, r, u16, U16),
        (DynMa::U32(l), DynMa::U32(r)) => int_arm!(l, r, u32, U32),
        (DynMa::U64(l), DynMa::U64(r)) => int_arm!(l, r, u64, U64),
        (DynMa::Bool(l), DynMa::Bool(r)) => {
            let la = broadcast_to(l.data(), bshape).map_err(ferr_to_pyerr)?;
            let ra = broadcast_to(r.data(), bshape).map_err(ferr_to_pyerr)?;
            match op {
                BinOp::Add | BinOp::Mul => {
                    let mut out: Vec<bool> = la
                        .iter()
                        .zip(ra.iter())
                        .map(|(&x, &y)| if op == BinOp::Add { x || y } else { x && y })
                        .collect();
                    let la_slice: Vec<bool> = la.iter().copied().collect();
                    revert_masked(&mut out, &la_slice, &union);
                    let arr =
                        ArrayD::<bool>::from_vec(IxDyn::new(bshape), out).map_err(ferr_to_pyerr)?;
                    (
                        DynMa::Bool(RustMa::from_data(arr).map_err(ferr_to_pyerr)?),
                        vec![false; n],
                    )
                }
                BinOp::Sub => {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "numpy boolean subtract, the `-` operator, is not supported, use the \
                         bitwise_xor, the `^` operator, or the logical_xor function instead.",
                    ));
                }
                BinOp::TrueDiv | BinOp::FloorDiv | BinOp::Mod | BinOp::Pow => {
                    // Promote bool operands to int8 (numpy.ma's bool loop for
                    // `/`/`//`/`%`/`**`), preserving the operand union mask, and
                    // recurse through the i8 compute path. The recursion folds
                    // union+domain into its mask; surface that as the domain so
                    // the outer union (identical) is idempotent.
                    let li = ArrayD::<i8>::from_vec(
                        IxDyn::new(bshape),
                        la.iter().map(|&v| v as i8).collect(),
                    )
                    .map_err(ferr_to_pyerr)?;
                    let ri = ArrayD::<i8>::from_vec(
                        IxDyn::new(bshape),
                        ra.iter().map(|&v| v as i8).collect(),
                    )
                    .map_err(ferr_to_pyerr)?;
                    let union_arr = ArrayD::<bool>::from_vec(IxDyn::new(bshape), union.clone())
                        .map_err(ferr_to_pyerr)?;
                    let lm = DynMa::I8(RustMa::new(li, union_arr.clone()).map_err(ferr_to_pyerr)?);
                    let rm = DynMa::I8(RustMa::new(ri, union_arr).map_err(ferr_to_pyerr)?);
                    let computed = compute_binary(&lm, &rm, bshape, op)?;
                    let domain: Vec<bool> = computed.mask.iter().copied().collect();
                    (computed.data, domain)
                }
            }
        }
        _ => {
            return Err(PyValueError::new_err(
                "internal: binary operand dtypes were not unified before compute",
            ));
        }
    };

    let domain_any = domain.iter().any(|&d| d);
    let mask = ArrayD::<bool>::from_vec(IxDyn::new(bshape), or_vecs(&union, &domain))
        .map_err(ferr_to_pyerr)?;
    Ok(ComputedBinary {
        data,
        mask,
        domain_any,
    })
}

/// Elementwise OR of two equal-length bool slices.
fn or_vecs(a: &[bool], b: &[bool]) -> Vec<bool> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x || y).collect()
}

/// Revert `result` to the left operand `la` at every masked position, building
/// the masked-result data buffer numpy keeps (`np.copyto(result, da, where=m)`,
/// `numpy/ma/core.py:1096`; the domained engine reconstructs the same via
/// `result += m * da`, `:1233`). `la` and `result` share `bshape`.
fn revert_masked<T: Copy>(result: &mut [T], la: &[T], mask: &[bool]) {
    for ((r, &d), &m) in result.iter_mut().zip(la.iter()).zip(mask.iter()) {
        if m {
            *r = d;
        }
    }
}

/// Diagnostic for a non-contiguous internal buffer (broadcast results are
/// materialized C-contiguous, so this is defensive — never hit in practice).
fn non_contig() -> PyErr {
    PyValueError::new_err("internal: expected a contiguous result buffer")
}

/// Egress a `DynMa`'s data buffer as a native-dtype numpy ndarray (the
/// free-function analog of `PyMaskedArray::data_pyarray`).
fn dynma_data_pyarray<'py>(py: Python<'py>, d: &DynMa) -> PyResult<Bound<'py, PyAny>> {
    match_ma!(d, m, T => {
        let arr: ArrayD<T> = fma::getdata(m).map_err(ferr_to_pyerr)?;
        Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    })
}

/// Re-wrap a freshly-computed result `DynMa` (whose own mask is meaningless)
/// with the supplied real bool mask, preserving the result dtype variant.
fn rewrap_dynma_with_mask(data: DynMa, mask: ArrayD<bool>) -> PyResult<DynMa> {
    rewrap_with_mask(&data, Some(mask))
}

/// Re-wrap a result `DynMa` with the `nomask` sentinel (a `from_data`
/// all-False, real-mask-less array), for non-domained ops over two nomask
/// operands (`numpy/ma/core.py:1077`).
fn strip_mask(data: DynMa) -> PyResult<DynMa> {
    macro_rules! strip {
        ($sm:expr, $variant:ident) => {{
            let d = $sm.data().clone();
            Ok(DynMa::$variant(
                RustMa::from_data(d).map_err(ferr_to_pyerr)?,
            ))
        }};
    }
    match &data {
        DynMa::Bool(sm) => strip!(sm, Bool),
        DynMa::I8(sm) => strip!(sm, I8),
        DynMa::I16(sm) => strip!(sm, I16),
        DynMa::I32(sm) => strip!(sm, I32),
        DynMa::I64(sm) => strip!(sm, I64),
        DynMa::U8(sm) => strip!(sm, U8),
        DynMa::U16(sm) => strip!(sm, U16),
        DynMa::U32(sm) => strip!(sm, U32),
        DynMa::U64(sm) => strip!(sm, U64),
        DynMa::F32(sm) => strip!(sm, F32),
        DynMa::F64(sm) => strip!(sm, F64),
    }
}

/// `set_mask_flat` on the active `DynMa` variant (the hard-mask clear gate is
/// enforced inside the library method, dtype-independently).
fn set_mask_flat_dyn(d: &mut DynMa, idx: usize, value: bool) -> PyResult<()> {
    match_ma!(d, m, T => { m.set_mask_flat(idx, value).map_err(put_err_to_pyerr) })
}

// ---------------------------------------------------------------------------
// Free constructors (numpy.ma.* free functions)
// ---------------------------------------------------------------------------

fn extract_data<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<ArrayD<f64>> {
    let arr = coerce_dtype(py, obj, "float64")?;
    let view: PyReadonlyArrayDyn<f64> = arr.extract()?;
    view.as_ferray().map_err(ferr_to_pyerr)
}

fn extract_bool_array<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<ArrayD<bool>> {
    let arr = coerce_dtype(py, obj, "bool")?;
    let view: PyReadonlyArrayDyn<bool> = arr.extract()?;
    view.as_ferray().map_err(ferr_to_pyerr)
}

/// Outcome of resolving a Python subscript index against a masked array's
/// shape — either a single flat (C-order) position (numpy's "scalar" case) or
/// a gather of flat positions plus the result shape (slice / fancy / bool /
/// multi-dim row, numpy's non-scalar case). See [`resolve_index`].
enum ResolvedIndex {
    Scalar(usize),
    Gather {
        positions: Vec<usize>,
        shape: Vec<usize>,
    },
}

/// Resolve a Python subscript index against a masked array of the given
/// `shape` into flat C-order positions, delegating ALL indexing semantics to
/// numpy (R-CODE-4 — negative indices, out-of-bounds `IndexError`, slices,
/// fancy / boolean indexing all behave exactly as numpy does).
///
/// The technique mirrors numpy's own `dout = self.data[indx]` (`numpy/ma/
/// core.py:3288`): a flat-position grid `numpy.arange(size).reshape(shape)` is
/// subscripted with `indx`. A 0-d result is numpy's scalar case
/// (`core.py:3334`, `scalar_expected`) and carries the single flat position; a
/// higher-rank result yields both the gathered flat positions (C-order
/// `ravel()`) and the result `shape` to build the sub-`MaskedArray`. numpy
/// raises the appropriate `IndexError` here for any out-of-range index before
/// we touch the data buffer.
fn resolve_index<'py>(
    py: Python<'py>,
    shape: &[usize],
    indx: &Bound<'py, PyAny>,
) -> PyResult<ResolvedIndex> {
    let size: usize = shape.iter().product();
    let np = py.import("numpy")?;
    let grid = np.call_method1("arange", (size,))?;
    let grid = grid.call_method1("reshape", (shape.to_vec(),))?;
    // `grid[indx]` reproduces numpy's exact indexing — including raising the
    // correct `IndexError` for an out-of-bounds / negative-out-of-bounds index.
    let picked = grid.get_item(indx)?;
    if picked
        .call_method0("__array__")?
        .getattr("ndim")?
        .extract::<usize>()?
        == 0
    {
        let pos: usize = picked.call_method0("item")?.extract()?;
        return Ok(ResolvedIndex::Scalar(pos));
    }
    let arr = picked.call_method0("__array__")?;
    let result_shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    let view: PyReadonlyArrayDyn<i64> = arr
        .call_method1("astype", ("int64",))?
        .call_method0("ravel")?
        .extract()?;
    let positions: Vec<usize> = view
        .as_ferray()
        .map_err(ferr_to_pyerr)?
        .iter()
        .map(|&v| v as usize)
        .collect();
    Ok(ResolvedIndex::Gather {
        positions,
        shape: result_shape,
    })
}

/// Broadcast a `__setitem__` right-hand side to the resolved target shape and
/// return its flat (C-order) `Vec<f64>` data plus, when the RHS carries a mask
/// (a `ferray.ma.MaskedArray` or `numpy.ma.MaskedArray`), the parallel flat
/// `Vec<bool>` of mask bits.
///
/// numpy assigns `dval = getattr(value, '_data', value)` broadcast across the
/// indexed region and `mval = getmask(value)` likewise (`numpy/ma/
/// core.py:3439`-`3441`). The boundary mirrors that with `numpy.broadcast_to`
/// so `a[1:3] = [7, 8]`, `a[a > 1] = 0` (scalar broadcast) and a masked-array
/// RHS each align element-for-element with the `count` gathered positions. The
/// mask is read through `numpy.ma.getmaskarray` so a masked RHS re-masks the
/// target, matching `_mask[indx] = mval`.
/// The DATA buffer of a `__setitem__` right-hand side, stripped of any mask,
/// cast to the target dtype `dt`, broadcast to `result_shape`, and flattened.
/// numpy assigns `dval = getattr(value, '_data', value)` broadcast across the
/// indexed region (`numpy/ma/core.py:3439`); the cast to `dt` mirrors numpy's
/// RHS coercion to `self.dtype` (R-CODE-4 — no hard-f64 funnel).
fn broadcast_value_data<'py, T>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
    result_shape: &[usize],
    count: usize,
    dt: &str,
) -> PyResult<Vec<T>>
where
    T: ferray_numpy_interop::numpy_conv::NpElement + Copy,
{
    let np = py.import("numpy")?;
    let np_ma = np.getattr("ma")?;
    let fr_ma = value.extract::<PyMaskedArray>().ok();
    let is_np_ma = value.is_instance(&np_ma.getattr("MaskedArray")?)?;
    let data_obj = if let Some(ref m) = fr_ma {
        m.data(py)?
    } else if is_np_ma {
        np_ma.call_method1("getdata", (value,))?
    } else {
        value.clone()
    };
    let data_t = coerce_dtype(py, &data_obj, dt)?;
    let data_bc = np
        .call_method1("broadcast_to", (data_t, result_shape.to_vec()))?
        .call_method0("ravel")?;
    let view: PyReadonlyArrayDyn<T> = data_bc.extract()?;
    let vals: Vec<T> = view
        .as_ferray()
        .map_err(ferr_to_pyerr)?
        .iter()
        .copied()
        .collect();
    if vals.len() != count {
        return Err(PyValueError::new_err(format!(
            "could not broadcast value into the {count} indexed position(s)",
        )));
    }
    Ok(vals)
}

/// The MASK of a `__setitem__` right-hand side, broadcast to `result_shape`
/// and flattened — or `None` when the RHS carries no mask. Dtype-independent.
fn broadcast_value_mask<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
    result_shape: &[usize],
    _count: usize,
) -> PyResult<Option<Vec<bool>>> {
    let np = py.import("numpy")?;
    let np_ma = np.getattr("ma")?;
    let fr_ma = value.extract::<PyMaskedArray>().ok();
    let is_np_ma = value.is_instance(&np_ma.getattr("MaskedArray")?)?;
    let mask_obj = if let Some(ref m) = fr_ma {
        let mk = m.mask(py)?;
        if mk.is(&ma_nomask(py)?) {
            Some(np_ma.call_method1("getmaskarray", (m.data(py)?,))?)
        } else {
            Some(mk)
        }
    } else if is_np_ma {
        Some(np_ma.call_method1("getmaskarray", (value,))?)
    } else {
        None
    };
    let Some(mask_obj) = mask_obj else {
        return Ok(None);
    };
    let mask_bc = np
        .call_method1("broadcast_to", (mask_obj, result_shape.to_vec()))?
        .call_method0("ravel")?;
    let view: PyReadonlyArrayDyn<bool> = mask_bc.extract()?;
    Ok(Some(
        view.as_ferray()
            .map_err(ferr_to_pyerr)?
            .iter()
            .copied()
            .collect(),
    ))
}

/// Coerce a Python index argument (scalar / list / ndarray) into a flat
/// `Vec<isize>` in C (row-major) order, preserving negative indices so the
/// library's `PutMode` resolution can apply numpy's from-the-end semantics.
///
/// numpy's `put` interprets `indices` as integers against the flattened
/// (C-order) data (`numpy/ma/core.py:4841` `self._data.flat[n]`), so we route
/// through `numpy.asarray(..., intp).ravel()` to flatten any-rank input the
/// same way.
fn extract_indices<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Vec<isize>> {
    let arr = coerce_dtype(py, obj, "int64")?;
    // `ravel()` flattens in C order so a 2-D index array matches numpy's
    // `.flat` indexing convention.
    let raveled = arr.call_method0("ravel")?;
    let view: PyReadonlyArrayDyn<i64> = raveled.extract()?;
    let fa: ArrayD<i64> = view.as_ferray().map_err(ferr_to_pyerr)?;
    Ok(fa.iter().map(|&v| v as isize).collect())
}

/// The flat DATA of a `put`/`putmask` `values` argument, coerced to the target
/// dtype `dt` (the array's native dtype). A `ferray.ma.MaskedArray` /
/// `numpy.ma.MaskedArray` operand contributes its `_data`; a plain array-like
/// is taken verbatim. Flattened in C order to match numpy's `.flat`/`copyto`
/// element order (R-CODE-4 — cast to `dt`, not f64).
fn values_data<'py, T>(py: Python<'py>, obj: &Bound<'py, PyAny>, dt: &str) -> PyResult<Vec<T>>
where
    T: ferray_numpy_interop::numpy_conv::NpElement + Copy,
{
    let np_ma = py.import("numpy")?.getattr("ma")?;
    let data_obj = if obj.extract::<PyMaskedArray>().is_ok() {
        obj.extract::<PyMaskedArray>()?.data(py)?
    } else if obj.is_instance(&np_ma.getattr("MaskedArray")?)? {
        np_ma.call_method1("getdata", (obj,))?
    } else {
        obj.clone()
    };
    let arr = coerce_dtype(py, &data_obj, dt)?;
    let raveled = arr.call_method0("ravel")?;
    let view: PyReadonlyArrayDyn<T> = raveled.extract()?;
    Ok(view
        .as_ferray()
        .map_err(ferr_to_pyerr)?
        .iter()
        .copied()
        .collect())
}

/// The flat MASK of a `put`/`putmask` `values` argument, or `None` when the
/// operand carries no real mask. A masked `values` re-masks the written target
/// positions (`numpy/ma/core.py:4918`, `:7589`). Dtype-independent.
fn values_mask<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Option<Vec<bool>>> {
    if let Ok(m) = obj.extract::<PyMaskedArray>() {
        let mask = m.inner.mask_opt_bits();
        return Ok(mask.filter(|bits| bits.iter().any(|&b| b)));
    }
    let np_ma = py.import("numpy")?.getattr("ma")?;
    if obj.is_instance(&np_ma.getattr("MaskedArray")?)? {
        let mask_obj = np_ma.call_method1("getmaskarray", (obj,))?;
        let mask_fa = extract_bool_array(py, &mask_obj)?;
        let bits: Vec<bool> = mask_fa.iter().copied().collect();
        return Ok(if bits.iter().any(|&b| b) {
            Some(bits)
        } else {
            None
        });
    }
    Ok(None)
}

/// When a `MaskedArray` is constructed with `mask=None` from a non-ferray
/// source, detect whether that source is ITSELF a `numpy.ma.MaskedArray`
/// carrying a REAL mask, and if so return that mask (shaped like `data_fa`) so
/// the constructor can carry it over via [`RustMa::new`]. Returns `None` for
/// any source that is unmasked or carries only the `nomask` sentinel — those
/// keep the `from_data` nomask path. (A `ferray.ma.MaskedArray` source is
/// handled earlier in `py_new` directly from its `inner`, so it never reaches
/// here.)
///
/// numpy's `MaskedArray.__new__` copies the source `_data`/`_mask` from a masked
/// input (`numpy/ma/core.py:2820`), so `ma.array(np.ma.array(d, mask=m))`
/// preserves `m`. The discriminator is `numpy.ma.getmask(data) is not nomask` —
/// true for a real bool mask (including an explicit all-False one, #849) and
/// false for the `nomask` singleton (a plain ndarray / list / `mask`-less input,
/// #848). The `nomask`-distinction is read off the sentinel, NOT off any-True
/// bits, so an explicit all-False source round-trips as a real all-False mask.
///
/// Reuses the existing mask-extraction machinery: `numpy.ma.getmask` (sentinel
/// probe) + `numpy.ma.getmaskarray` (full bool array) — the same primitives
/// [`extract_values_with_mask`]/[`coerce_to_ma_impl`] already use.
fn source_real_mask<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
) -> PyResult<Option<ArrayD<bool>>> {
    // A numpy.ma.MaskedArray source: `numpy.ma.getmask` returns the `nomask`
    // sentinel (`numpy.ma.nomask`) for a mask-less array and a bool ndarray for
    // a real mask. Probe the sentinel by identity; only materialize the full
    // mask (via `getmaskarray`) when it is a real mask. The mask shape agrees
    // with the data by construction (a numpy.ma array and its `getmaskarray`
    // share a shape); a genuine mismatch surfaces as the library's
    // `ShapeMismatch` at `RustMa::new` rather than here.
    let np_ma = py.import("numpy")?.getattr("ma")?;
    if obj.is_instance(&np_ma.getattr("MaskedArray")?)? {
        let getmask = np_ma.call_method1("getmask", (obj,))?;
        let nomask = np_ma.getattr("nomask")?;
        if getmask.is(&nomask) {
            return Ok(None);
        }
        let mask_obj = np_ma.call_method1("getmaskarray", (obj,))?;
        let mask_fa = extract_bool_array(py, &mask_obj)?;
        return Ok(Some(mask_fa));
    }
    // Plain ndarray / list / scalar — not masked, keep the nomask path.
    Ok(None)
}

/// Re-wrap a `DynMa` source with a (possibly new) mask, preserving its native
/// dtype variant. `Some(mask)` materializes a REAL mask of that dtype;
/// `None` reproduces the source's own mask state (nomask vs real) verbatim.
fn rewrap_with_mask(src: &DynMa, mask: Option<ArrayD<bool>>) -> PyResult<DynMa> {
    let Some(m) = mask else {
        return Ok(src.clone());
    };
    macro_rules! rewrap {
        ($sm:expr, $variant:ident) => {{
            let data = $sm.data().clone();
            Ok(DynMa::$variant(
                RustMa::new(data, m).map_err(ferr_to_pyerr)?,
            ))
        }};
    }
    match src {
        DynMa::Bool(sm) => rewrap!(sm, Bool),
        DynMa::I8(sm) => rewrap!(sm, I8),
        DynMa::I16(sm) => rewrap!(sm, I16),
        DynMa::I32(sm) => rewrap!(sm, I32),
        DynMa::I64(sm) => rewrap!(sm, I64),
        DynMa::U8(sm) => rewrap!(sm, U8),
        DynMa::U16(sm) => rewrap!(sm, U16),
        DynMa::U32(sm) => rewrap!(sm, U32),
        DynMa::U64(sm) => rewrap!(sm, U64),
        DynMa::F32(sm) => rewrap!(sm, F32),
        DynMa::F64(sm) => rewrap!(sm, F64),
    }
}

/// Elementwise OR of an explicit mask with a source mask, mirroring numpy's
/// `keep_mask=True` default (`numpy/ma/core.py:3008` `np.logical_or(mask,
/// _data._mask)`). Both masks share the data's shape by construction (the
/// explicit `mask=` is coerced to the data shape and `source_real_mask` only
/// returns a shape-matched source mask); a length mismatch surfaces as the
/// library's `ShapeMismatch` rather than a panic.
fn or_masks(explicit: &ArrayD<bool>, source: &ArrayD<bool>) -> PyResult<ArrayD<bool>> {
    if explicit.shape() != source.shape() {
        return Err(PyValueError::new_err(format!(
            "mask shape {:?} does not match source mask shape {:?}",
            explicit.shape(),
            source.shape(),
        )));
    }
    let combined: Vec<bool> = explicit
        .iter()
        .zip(source.iter())
        .map(|(&a, &b)| a || b)
        .collect();
    ArrayD::<bool>::from_vec(IxDyn::new(explicit.shape()), combined).map_err(ferr_to_pyerr)
}

/// Map a `put` library error to the matching CPython exception: an
/// out-of-bounds index under `mode='raise'` surfaces as `IndexError`
/// (numpy's `put` raises `IndexError`, confirmed live), every other
/// `FerrayError` falls through to [`ferr_to_pyerr`] (→ `ValueError`).
fn put_err_to_pyerr(e: ferray_core::FerrayError) -> PyErr {
    if matches!(e, ferray_core::FerrayError::IndexOutOfBounds { .. }) {
        return PyIndexError::new_err(e.to_string());
    }
    ferr_to_pyerr(e)
}

/// `numpy.ma.array(data, mask=None, dtype=None)`.
#[pyfunction]
#[pyo3(signature = (data, mask = None, dtype = None))]
pub fn array<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    mask: Option<&Bound<'py, PyAny>>,
    dtype: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyMaskedArray> {
    PyMaskedArray::py_new(py, data, mask, dtype)
}

/// `numpy.ma.masked_array(data, mask=None, dtype=None)` — alias for `array`.
#[pyfunction]
#[pyo3(signature = (data, mask = None, dtype = None))]
pub fn masked_array<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    mask: Option<&Bound<'py, PyAny>>,
    dtype: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyMaskedArray> {
    PyMaskedArray::py_new(py, data, mask, dtype)
}

/// `numpy.ma.masked_where(condition, data)`.
///
/// A `condition` whose shape doesn't match `data` raises `IndexError`, not
/// `ValueError`: numpy applies the boolean mask as a fancy index
/// (`numpy/ma/core.py:1885`), and a boolean-index/broadcast shape mismatch
/// surfaces as `IndexError` (confirmed live: `np.ma.masked_where([T,F],
/// [1.,2.,3.])` raises `IndexError`). ferray-ma reports the mismatch as a
/// `FerrayError::ShapeMismatch`, which [`ferr_to_pyerr`] would map to
/// `ValueError`; the binding remaps it to `IndexError` here (R-DEV-2).
#[pyfunction]
pub fn masked_where<'py>(
    py: Python<'py>,
    condition: &Bound<'py, PyAny>,
    data: &Bound<'py, PyAny>,
) -> PyResult<PyMaskedArray> {
    let cond = extract_bool_array(py, condition)?;
    let data_fa = extract_data(py, data)?;
    let inner = fma::masked_where(&cond, &data_fa).map_err(shape_mismatch_to_index_error)?;
    Ok(PyMaskedArray::from_inner(inner))
}

/// Map a ferray-ma `ShapeMismatch` (from `masked_where`'s shape check) to a
/// `IndexError`, mirroring numpy's boolean-index failure; every other
/// `FerrayError` falls through to [`ferr_to_pyerr`].
fn shape_mismatch_to_index_error(e: ferray_core::FerrayError) -> PyErr {
    if matches!(e, ferray_core::FerrayError::ShapeMismatch { .. }) {
        return PyIndexError::new_err(e.to_string());
    }
    ferr_to_pyerr(e)
}

/// `numpy.ma.masked_invalid(data)` — mask NaN and inf values.
#[pyfunction]
pub fn masked_invalid<'py>(py: Python<'py>, data: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    let data_fa = extract_data(py, data)?;
    let inner = fma::masked_invalid(&data_fa).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(inner))
}

macro_rules! bind_masked_compare {
    ($name:ident, $ferr_path:path) => {
        #[pyfunction]
        pub fn $name<'py>(
            py: Python<'py>,
            x: &Bound<'py, PyAny>,
            value: f64,
        ) -> PyResult<PyMaskedArray> {
            let data_fa = extract_data(py, x)?;
            let inner = $ferr_path(&data_fa, value).map_err(ferr_to_pyerr)?;
            Ok(PyMaskedArray::from_inner(inner))
        }
    };
}

/// `numpy.ma.masked_equal(x, value)` — mask where equal to `value`.
///
/// numpy additionally sets the result's `fill_value` to the compared value
/// (`numpy/ma/core.py:2172` `output.fill_value = value`), so a later
/// `filled()` substitutes that value at masked slots. The binding mirrors
/// that by setting the ferray-ma `fill_value` after constructing the mask.
#[pyfunction]
pub fn masked_equal<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    value: f64,
) -> PyResult<PyMaskedArray> {
    let data_fa = extract_data(py, x)?;
    let mut inner = fma::masked_equal(&data_fa, value).map_err(ferr_to_pyerr)?;
    inner.set_fill_value(value);
    Ok(PyMaskedArray::from_inner(inner))
}

bind_masked_compare!(masked_not_equal, fma::masked_not_equal);
bind_masked_compare!(masked_greater, fma::masked_greater);
bind_masked_compare!(masked_greater_equal, fma::masked_greater_equal);
bind_masked_compare!(masked_less, fma::masked_less);
bind_masked_compare!(masked_less_equal, fma::masked_less_equal);

/// `numpy.ma.masked_inside(x, low, high)`.
#[pyfunction]
pub fn masked_inside<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    low: f64,
    high: f64,
) -> PyResult<PyMaskedArray> {
    let data_fa = extract_data(py, x)?;
    let inner = fma::masked_inside(&data_fa, low, high).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(inner))
}

/// `numpy.ma.masked_outside(x, low, high)`.
#[pyfunction]
pub fn masked_outside<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    low: f64,
    high: f64,
) -> PyResult<PyMaskedArray> {
    let data_fa = extract_data(py, x)?;
    let inner = fma::masked_outside(&data_fa, low, high).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(inner))
}

/// `numpy.ma.count_masked(a)` — number of masked elements (full reduction).
#[pyfunction]
pub fn count_masked(a: &PyMaskedArray) -> PyResult<usize> {
    a.inner.count_masked()
}

/// `numpy.ma.is_masked(a)` — true iff at least one element is masked.
#[pyfunction]
pub fn is_masked(a: &PyMaskedArray) -> PyResult<bool> {
    Ok(a.inner.count_masked()? > 0)
}

/// `numpy.ma.getmask(a)` — return mask as a numpy bool ndarray.
#[pyfunction]
pub fn getmask<'py>(py: Python<'py>, a: &PyMaskedArray) -> PyResult<Bound<'py, PyAny>> {
    a.mask(py)
}

/// `numpy.ma.getdata(a)` — return data buffer as a numpy float64 ndarray.
#[pyfunction]
pub fn getdata<'py>(py: Python<'py>, a: &PyMaskedArray) -> PyResult<Bound<'py, PyAny>> {
    a.data(py)
}

/// `numpy.ma.filled(a, fill_value=None)` — replace masked with fill (native
/// dtype; `fill_value` coerced to the array dtype).
#[pyfunction]
#[pyo3(signature = (a, fill_value = None))]
pub fn filled<'py>(
    py: Python<'py>,
    a: &PyMaskedArray,
    fill_value: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    a.filled(py, fill_value)
}

/// `numpy.ma.compressed(a)`.
#[pyfunction]
pub fn compressed<'py>(py: Python<'py>, a: &PyMaskedArray) -> PyResult<Bound<'py, PyAny>> {
    a.compressed(py)
}

/// Raises a clean error when callers pass a non-MaskedArray to a
/// function expecting one. Used internally — kept here so the diagnostic
/// message stays consistent with the rest of the module.
#[allow(dead_code)]
fn require_ma<'py>(obj: &Bound<'py, PyAny>) -> PyResult<()> {
    if obj.extract::<PyMaskedArray>().is_err() {
        return Err(PyValueError::new_err(
            "expected a ferray.ma.MaskedArray; use ferray.ma.masked_array(data, mask) to construct one",
        ));
    }
    Ok(())
}

// ===========================================================================
// numpy.ma expansion (refs #818): masked elementwise ufuncs, reductions,
// creation, manipulation, and mask helpers — composed over the existing
// ferray-ma library (`fma::*`) and ferray-core creation.
//
// Every binding here re-uses the f64-only `PyMaskedArray` wrapper. The
// numpy.ma contract was verified live against numpy 2.4.5 (R-CHAR-3 — the
// pytest oracle constructs expected values from `numpy.ma.*` directly).
// ===========================================================================

/// Coerce any Python object into a `RustMa<f64, IxDyn>`.
///
/// A `ferray.ma.MaskedArray` is unwrapped directly (preserving its mask);
/// any other array-like (list, ndarray, numpy.ma.MaskedArray, scalar) is
/// routed through `numpy.asarray(..., float64)` and wrapped with an
/// all-false (`nomask`) mask. This is the masked analog of the plain
/// `extract_data` path and lets the elementwise/binary ufuncs accept the
/// same operands `numpy.ma.add(ma, 3)` / `numpy.ma.sqrt([4., 9.])` do.
fn coerce_to_ma<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<RustMa<f64, IxDyn>> {
    coerce_to_ma_impl(py, obj, false)
}

/// Coerce a Python operand to a dtype-preserving [`DynMa`] (native dtype +
/// mask), the integer/bool-faithful analog of [`coerce_to_ma`]. A
/// `ferray.ma.MaskedArray` keeps its variant verbatim; any other array-like is
/// normalized via `numpy.asarray` (no cast) and routed through `build_dynma`,
/// reading a `numpy.ma` operand's mask via `numpy.ma.getmaskarray` (#858,
/// R-CODE-4 — no float64 funnel).
fn coerce_to_dynma<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<DynMa> {
    if let Ok(m) = obj.extract::<PyMaskedArray>() {
        return Ok(m.inner);
    }
    let arr = crate::conv::as_ndarray(py, obj)?;
    let np_ma = py.import("numpy")?.getattr("ma")?;
    let mask_obj = np_ma.call_method1("getmaskarray", (obj,))?;
    let mask_fa = extract_bool_array(py, &mask_obj)?;
    build_dynma(&arr, Some(mask_fa))
}

/// Like [`coerce_to_ma`] but, for a `numpy.ma.MaskedArray` operand, reads its
/// mask via `numpy.ma.getmaskarray` rather than discarding it.
///
/// The #818/#835 elementwise bindings predate masked-input support and wrap a
/// numpy.ma operand with `nomask` (the `coerce_to_ma` convention). The residual
/// `convolve`/`correlate`/`polyfit`/`cov`/`corrcoef` slice (refs #837) must
/// honour the incoming mask to match numpy.ma exactly, so it uses this
/// mask-preserving variant. A `ferray.ma.MaskedArray` (already carrying its
/// mask) and any non-masked array-like behave identically to `coerce_to_ma`.
fn coerce_to_ma_npmask<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
) -> PyResult<RustMa<f64, IxDyn>> {
    coerce_to_ma_impl(py, obj, true)
}

fn coerce_to_ma_impl<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    read_np_mask: bool,
) -> PyResult<RustMa<f64, IxDyn>> {
    if let Ok(m) = obj.extract::<PyMaskedArray>() {
        // The float-only ufunc/reduction surface computes in f64 (numpy.ma
        // promotes integer/bool input to float for these ops); cast the native
        // variant to f64, preserving the mask.
        return m.to_f64_ma();
    }
    let data_fa = extract_data(py, obj)?;
    let n = data_fa.size();
    let shape = data_fa.shape().to_vec();
    let mask_fa = if read_np_mask {
        // `numpy.ma.getmaskarray(obj)` returns a full bool array of `obj`'s
        // shape (all-False for a plain ndarray / nomask input), so the mask
        // of a numpy.ma operand is carried across the boundary.
        let np_ma = py.import("numpy")?.getattr("ma")?;
        let mask_obj = np_ma.call_method1("getmaskarray", (obj,))?;
        let mask_arr = coerce_dtype(py, &mask_obj, "bool")?;
        let mask_view: PyReadonlyArrayDyn<bool> = mask_arr.extract()?;
        mask_view.as_ferray().map_err(ferr_to_pyerr)?
    } else {
        ArrayD::<bool>::from_vec(IxDyn::new(&shape), vec![false; n]).map_err(ferr_to_pyerr)?
    };
    RustMa::new(data_fa, mask_fa).map_err(ferr_to_pyerr)
}

// ---------------------------------------------------------------------------
// Masked unary elementwise ufuncs
// ---------------------------------------------------------------------------

/// Generate a unary masked ufunc that applies `$f` (an `Fn(f64) -> f64`)
/// to the unmasked data and propagates the mask.
macro_rules! ma_unary {
    ($name:ident, $f:expr, $doc:literal) => {
        #[doc = $doc]
        #[pyfunction]
        pub fn $name<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
            let m = coerce_to_ma(py, a)?;
            let out = fma::masked_unary(&m, $f).map_err(ferr_to_pyerr)?;
            Ok(PyMaskedArray::from_inner(out))
        }
    };
}

/// Generate a unary masked ufunc whose result additionally masks any
/// position where the data is out of the function's domain, delegating to
/// the ferray-ma `*_domain` wrapper `$dom`.
macro_rules! ma_unary_domain {
    ($name:ident, $dom:path, $doc:literal) => {
        #[doc = $doc]
        #[pyfunction]
        pub fn $name<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
            let m = coerce_to_ma(py, a)?;
            let out = $dom(&m).map_err(ferr_to_pyerr)?;
            Ok(PyMaskedArray::from_inner(out))
        }
    };
}

ma_unary!(
    sin,
    f64::sin,
    "`numpy.ma.sin` — elementwise sine, mask propagated."
);
ma_unary!(
    cos,
    f64::cos,
    "`numpy.ma.cos` — elementwise cosine, mask propagated."
);
ma_unary!(
    tan,
    f64::tan,
    "`numpy.ma.tan` — elementwise tangent, mask propagated."
);
ma_unary!(
    arctan,
    f64::atan,
    "`numpy.ma.arctan` — elementwise arctangent."
);
ma_unary!(
    sinh,
    f64::sinh,
    "`numpy.ma.sinh` — elementwise hyperbolic sine."
);
ma_unary!(
    cosh,
    f64::cosh,
    "`numpy.ma.cosh` — elementwise hyperbolic cosine."
);
ma_unary!(
    tanh,
    f64::tanh,
    "`numpy.ma.tanh` — elementwise hyperbolic tangent."
);
ma_unary!(
    arcsinh,
    f64::asinh,
    "`numpy.ma.arcsinh` — inverse hyperbolic sine."
);
ma_unary!(exp, f64::exp, "`numpy.ma.exp` — elementwise exponential.");
ma_unary!(floor, f64::floor, "`numpy.ma.floor` — elementwise floor.");
ma_unary!(ceil, f64::ceil, "`numpy.ma.ceil` — elementwise ceiling.");
ma_unary!(
    negative,
    |x: f64| -x,
    "`numpy.ma.negative` — elementwise negation."
);
ma_unary!(
    absolute,
    f64::abs,
    "`numpy.ma.absolute` — elementwise absolute value."
);
ma_unary!(abs, f64::abs, "`numpy.ma.abs` — alias of `absolute`.");
ma_unary!(
    fabs,
    f64::abs,
    "`numpy.ma.fabs` — elementwise float absolute value."
);
ma_unary!(
    conjugate,
    |x: f64| x,
    "`numpy.ma.conjugate` — conjugate; on real (f64) data the identity, mask propagated."
);

ma_unary_domain!(
    sqrt,
    fma::sqrt_domain,
    "`numpy.ma.sqrt` — sqrt; negative inputs are masked."
);
ma_unary_domain!(
    log,
    fma::log_domain,
    "`numpy.ma.log` — natural log; non-positive inputs masked."
);
ma_unary_domain!(
    log2,
    fma::log2_domain,
    "`numpy.ma.log2` — base-2 log; non-positive inputs masked."
);
ma_unary_domain!(
    log10,
    fma::log10_domain,
    "`numpy.ma.log10` — base-10 log; non-positive inputs masked."
);
ma_unary_domain!(
    arcsin,
    fma::arcsin_domain,
    "`numpy.ma.arcsin` — arcsine; `|x| > 1` inputs masked."
);
ma_unary_domain!(
    arccos,
    fma::arccos_domain,
    "`numpy.ma.arccos` — arccosine; `|x| > 1` inputs masked."
);
ma_unary_domain!(
    arccosh,
    fma::arccosh_domain,
    "`numpy.ma.arccosh` — arccosh; `x < 1` inputs masked."
);
ma_unary_domain!(
    arctanh,
    fma::arctanh_domain,
    "`numpy.ma.arctanh` — arctanh; `|x| >= 1` inputs masked."
);

/// `numpy.ma.around(a, decimals=0)` — round to `decimals` places, mask
/// propagated. NumPy's masked `around` rounds the underlying data with the
/// half-to-even rule (`numpy.round`) and keeps the mask.
#[pyfunction]
#[pyo3(signature = (a, decimals = 0))]
pub fn around<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    decimals: i32,
) -> PyResult<PyMaskedArray> {
    let m = coerce_to_ma(py, a)?;
    let factor = 10f64.powi(decimals);
    // Half-to-even (banker's rounding), matching numpy.round.
    let round_even = move |x: f64| {
        let scaled = x * factor;
        let r = scaled.round();
        // `f64::round` rounds half away from zero; correct the exact-half
        // case to half-to-even to mirror numpy.
        let adjusted = if (scaled - scaled.floor() - 0.5).abs() < f64::EPSILON {
            let lower = scaled.floor();
            if (lower as i64) % 2 == 0 {
                lower
            } else {
                lower + 1.0
            }
        } else {
            r
        };
        adjusted / factor
    };
    let out = fma::masked_unary(&m, round_even).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(out))
}

// ---------------------------------------------------------------------------
// Masked binary elementwise ufuncs
// ---------------------------------------------------------------------------

/// Generate a binary masked ufunc applying `$f` (an `Fn(f64, f64) -> f64`)
/// to two operands, each coerced via [`coerce_to_ma`] so a scalar / plain
/// array second argument works, and OR-ing the two masks.
macro_rules! ma_binary {
    ($name:ident, $f:expr, $opname:literal, $doc:literal) => {
        #[doc = $doc]
        #[pyfunction]
        pub fn $name<'py>(
            py: Python<'py>,
            a: &Bound<'py, PyAny>,
            b: &Bound<'py, PyAny>,
        ) -> PyResult<PyMaskedArray> {
            let ma = coerce_to_ma(py, a)?;
            let mb = coerce_to_ma(py, b)?;
            let out = fma::masked_binary(&ma, &mb, $f, $opname).map_err(ferr_to_pyerr)?;
            Ok(PyMaskedArray::from_inner(out))
        }
    };
}

ma_binary!(
    add,
    |x: f64, y: f64| x + y,
    "add",
    "`numpy.ma.add` — elementwise sum, mask union."
);
ma_binary!(
    subtract,
    |x: f64, y: f64| x - y,
    "subtract",
    "`numpy.ma.subtract` — elementwise difference."
);
ma_binary!(
    multiply,
    |x: f64, y: f64| x * y,
    "multiply",
    "`numpy.ma.multiply` — elementwise product."
);
ma_binary!(
    arctan2,
    f64::atan2,
    "arctan2",
    "`numpy.ma.arctan2` — elementwise arctangent of y/x."
);
ma_binary!(
    hypot,
    f64::hypot,
    "hypot",
    "`numpy.ma.hypot` — elementwise hypotenuse."
);
ma_binary!(
    power,
    f64::powf,
    "power",
    "`numpy.ma.power` — elementwise power."
);
ma_binary!(
    fmod,
    |x: f64, y: f64| x % y,
    "fmod",
    "`numpy.ma.fmod` — C-style float modulo (sign of dividend)."
);
ma_binary!(
    remainder,
    |x: f64, y: f64| x - y * (x / y).floor(),
    "remainder",
    "`numpy.ma.remainder` — Python/numpy modulo (sign of divisor)."
);
ma_binary!(
    floor_divide,
    |x: f64, y: f64| (x / y).floor(),
    "floor_divide",
    "`numpy.ma.floor_divide` — elementwise floor division."
);
ma_binary!(
    maximum,
    |x: f64, y: f64| if x.is_nan() || y.is_nan() {
        f64::NAN
    } else {
        x.max(y)
    },
    "maximum",
    "`numpy.ma.maximum` — elementwise maximum (NaN-propagating)."
);
ma_binary!(
    minimum,
    |x: f64, y: f64| if x.is_nan() || y.is_nan() {
        f64::NAN
    } else {
        x.min(y)
    },
    "minimum",
    "`numpy.ma.minimum` — elementwise minimum (NaN-propagating)."
);
ma_binary!(
    bitwise_and,
    |x: f64, y: f64| ((x as i64) & (y as i64)) as f64,
    "bitwise_and",
    "`numpy.ma.bitwise_and` — elementwise bitwise AND (integer-valued data)."
);
ma_binary!(
    bitwise_or,
    |x: f64, y: f64| ((x as i64) | (y as i64)) as f64,
    "bitwise_or",
    "`numpy.ma.bitwise_or` — elementwise bitwise OR (integer-valued data)."
);
ma_binary!(
    bitwise_xor,
    |x: f64, y: f64| ((x as i64) ^ (y as i64)) as f64,
    "bitwise_xor",
    "`numpy.ma.bitwise_xor` — elementwise bitwise XOR (integer-valued data)."
);

/// `numpy.ma.divide(a, b)` — true division; positions where the denominator
/// is exactly zero are masked (numpy's `divide` domain rule). Delegates to
/// the ferray-ma `divide_domain` wrapper.
#[pyfunction]
pub fn divide<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<PyMaskedArray> {
    let ma = coerce_to_ma(py, a)?;
    let mb = coerce_to_ma(py, b)?;
    let out = fma::divide_domain(&ma, &mb).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(out))
}

/// `numpy.ma.true_divide(a, b)` — alias of [`divide`].
#[pyfunction]
pub fn true_divide<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<PyMaskedArray> {
    divide(py, a, b)
}

/// `numpy.ma.mod(a, b)` — alias of [`remainder`] (numpy/ma/core.py exposes
/// `mod = remainder`).
#[pyfunction]
pub fn mod_<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<PyMaskedArray> {
    remainder(py, a, b)
}

// ---------------------------------------------------------------------------
// Masked reductions (free functions accepting a MaskedArray)
// ---------------------------------------------------------------------------

/// `numpy.ma.prod(a)` — product of unmasked elements (full reduction).
/// All-masked reduces to the `numpy.ma.masked` singleton.
#[pyfunction]
pub fn prod<'py>(py: Python<'py>, a: &PyMaskedArray) -> PyResult<Bound<'py, PyAny>> {
    if a.all_masked()? {
        return ma_masked_singleton(py);
    }
    // Dtype-preserving: numpy.ma `prod` = `filled(1).prod()`
    // (`numpy/ma/core.py:5303`), result dtype = the ReduceAcc accumulator
    // (int8→int64, uint8→uint64, bool→int64, float unchanged) (#858).
    match_ma!(&a.inner, m, T => {
        scalar_pyobject(py, ma_prod_acc(m))
    })
}

/// `numpy.ma.median(a)` — median of unmasked elements. All-masked reduces
/// to the `numpy.ma.masked` singleton. Computed in f64 (promote-to-float).
#[pyfunction]
pub fn median<'py>(py: Python<'py>, a: &PyMaskedArray) -> PyResult<Bound<'py, PyAny>> {
    if a.all_masked()? {
        return ma_masked_singleton(py);
    }
    let v = a.to_f64_ma()?.median().map_err(ferr_to_pyerr)?;
    Ok(v.into_pyobject(py)?.into_any())
}

/// `numpy.ma.ptp(a)` — peak-to-peak (max − min) of unmasked elements.
/// All-masked reduces to the `numpy.ma.masked` singleton.
#[pyfunction]
pub fn ptp<'py>(py: Python<'py>, a: &PyMaskedArray) -> PyResult<Bound<'py, PyAny>> {
    if a.all_masked()? {
        return ma_masked_singleton(py);
    }
    let v = a.to_f64_ma()?.ptp().map_err(ferr_to_pyerr)?;
    Ok(v.into_pyobject(py)?.into_any())
}

/// `numpy.ma.argmin(a)` — index of the minimum unmasked element (flat).
#[pyfunction]
pub fn argmin<'py>(py: Python<'py>, a: &PyMaskedArray) -> PyResult<Bound<'py, PyAny>> {
    let v = a.to_f64_ma()?.argmin().map_err(ferr_to_pyerr)?;
    Ok(v.into_pyobject(py)?.into_any())
}

/// `numpy.ma.argmax(a)` — index of the maximum unmasked element (flat).
#[pyfunction]
pub fn argmax<'py>(py: Python<'py>, a: &PyMaskedArray) -> PyResult<Bound<'py, PyAny>> {
    let v = a.to_f64_ma()?.argmax().map_err(ferr_to_pyerr)?;
    Ok(v.into_pyobject(py)?.into_any())
}

/// `numpy.ma.count(a)` — number of unmasked elements.
#[pyfunction]
pub fn count(a: &PyMaskedArray) -> PyResult<usize> {
    a.inner.count()
}

/// `numpy.ma.average(a, weights=None)` — weighted average over unmasked
/// elements. All-masked reduces to the `numpy.ma.masked` singleton.
#[pyfunction]
#[pyo3(signature = (a, weights = None))]
pub fn average<'py>(
    py: Python<'py>,
    a: &PyMaskedArray,
    weights: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    if a.all_masked()? {
        return ma_masked_singleton(py);
    }
    let af = a.to_f64_ma()?;
    let v = match weights {
        None => af.average(None).map_err(ferr_to_pyerr)?,
        Some(w) => {
            let w_fa = extract_data(py, w)?;
            af.average(Some(&w_fa)).map_err(ferr_to_pyerr)?
        }
    };
    Ok(v.into_pyobject(py)?.into_any())
}

/// `numpy.ma.all(a)` — true iff every unmasked element is truthy
/// (non-zero). Masked elements are ignored (numpy treats `masked` as
/// `True` for `all`).
#[pyfunction]
pub fn all(a: &PyMaskedArray) -> PyResult<bool> {
    let m = a.to_f64_ma()?;
    let data = fma::getdata(&m).map_err(ferr_to_pyerr)?;
    let mask = fma::getmaskarray(&m).map_err(ferr_to_pyerr)?;
    Ok(data.iter().zip(mask.iter()).all(|(&v, &mk)| mk || v != 0.0))
}

/// `numpy.ma.any(a)` — true iff at least one unmasked element is truthy.
#[pyfunction]
pub fn any(a: &PyMaskedArray) -> PyResult<bool> {
    let m = a.to_f64_ma()?;
    let data = fma::getdata(&m).map_err(ferr_to_pyerr)?;
    let mask = fma::getmaskarray(&m).map_err(ferr_to_pyerr)?;
    Ok(data
        .iter()
        .zip(mask.iter())
        .any(|(&v, &mk)| !mk && v != 0.0))
}

/// `numpy.ma.anom(a)` — deviations from the unmasked mean (a.k.a.
/// `anomalies`). Returns a `MaskedArray` of `x − mean(x)` (float64).
#[pyfunction]
pub fn anom(a: &PyMaskedArray) -> PyResult<PyMaskedArray> {
    let out = a.to_f64_ma()?.anom().map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(out))
}

// ---------------------------------------------------------------------------
// Masked creation
// ---------------------------------------------------------------------------

/// Wrap a freshly-created `ArrayD<f64>` (no mask) as a `PyMaskedArray`.
fn from_unmasked(data: ArrayD<f64>) -> PyResult<PyMaskedArray> {
    let inner = RustMa::from_data(data).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(inner))
}

/// Promote a 1-D `MaskedArray<f64, Ix1>` (produced by `ravel`/`repeat`) to
/// the dynamic-dimension `RustMa<f64, IxDyn>` the `PyMaskedArray` wrapper
/// holds, preserving the data and mask.
fn ix1_ma_to_dyn(m: RustMa<f64, ferray_core::dimension::Ix1>) -> PyResult<RustMa<f64, IxDyn>> {
    let data = fma::getdata(&m).map_err(ferr_to_pyerr)?.into_dyn();
    let mask = fma::getmaskarray(&m).map_err(ferr_to_pyerr)?.into_dyn();
    RustMa::new(data, mask).map_err(ferr_to_pyerr)
}

/// `numpy.ma.zeros(shape)` — masked array of zeros with `nomask`.
#[pyfunction]
pub fn zeros<'py>(py: Python<'py>, shape: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    let arr = coerce_dtype(py, &creation_call(py, "zeros", shape)?, "float64")?;
    let view: PyReadonlyArrayDyn<f64> = arr.extract()?;
    from_unmasked(view.as_ferray().map_err(ferr_to_pyerr)?)
}

/// `numpy.ma.ones(shape)` — masked array of ones with `nomask`.
#[pyfunction]
pub fn ones<'py>(py: Python<'py>, shape: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    let arr = coerce_dtype(py, &creation_call(py, "ones", shape)?, "float64")?;
    let view: PyReadonlyArrayDyn<f64> = arr.extract()?;
    from_unmasked(view.as_ferray().map_err(ferr_to_pyerr)?)
}

/// `numpy.ma.empty(shape)` — masked array of uninitialized (here: zero)
/// data with `nomask`. numpy fills with arbitrary bytes; ferray returns a
/// deterministic zero buffer, which is a valid `empty` (R-DEV-7).
#[pyfunction]
pub fn empty<'py>(py: Python<'py>, shape: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    zeros(py, shape)
}

/// `numpy.ma.arange(stop)` / `arange(start, stop, step)` — masked range
/// with `nomask`. Built on `numpy.arange` (float64) then wrapped.
#[pyfunction]
#[pyo3(signature = (start, stop = None, step = None))]
pub fn arange<'py>(
    py: Python<'py>,
    start: &Bound<'py, PyAny>,
    stop: Option<&Bound<'py, PyAny>>,
    step: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyMaskedArray> {
    let np = py.import("numpy")?;
    let result = match (stop, step) {
        (None, _) => np.call_method1("arange", (start,))?,
        (Some(s), None) => np.call_method1("arange", (start, s))?,
        (Some(s), Some(st)) => np.call_method1("arange", (start, s, st))?,
    };
    let arr = coerce_dtype(py, &result, "float64")?;
    let view: PyReadonlyArrayDyn<f64> = arr.extract()?;
    from_unmasked(view.as_ferray().map_err(ferr_to_pyerr)?)
}

/// `numpy.ma.identity(n)` — masked identity matrix with `nomask`.
#[pyfunction]
pub fn identity<'py>(py: Python<'py>, n: usize) -> PyResult<PyMaskedArray> {
    let np = py.import("numpy")?;
    let result = np.call_method1("identity", (n,))?;
    let arr = coerce_dtype(py, &result, "float64")?;
    let view: PyReadonlyArrayDyn<f64> = arr.extract()?;
    from_unmasked(view.as_ferray().map_err(ferr_to_pyerr)?)
}

/// `numpy.ma.asarray(a)` / `asanyarray(a)` — interpret `a` as a masked
/// array. An existing `MaskedArray` is returned (clone); any other
/// array-like is wrapped with `nomask`.
#[pyfunction]
pub fn asarray<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    Ok(PyMaskedArray::from_inner(coerce_to_ma(py, a)?))
}

/// `numpy.ma.asanyarray(a)` — alias of [`asarray`] for the f64 binding.
#[pyfunction]
pub fn asanyarray<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    asarray(py, a)
}

/// `numpy.ma.copy(a)` — a copy of the masked array (data + mask).
#[pyfunction]
pub fn copy<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    Ok(PyMaskedArray::from_inner(coerce_to_ma(py, a)?))
}

/// Helper: call `numpy.<name>(shape)` accepting numpy's int-or-tuple shape.
fn creation_call<'py>(
    py: Python<'py>,
    name: &str,
    shape: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    py.import("numpy")?.call_method1(name, (shape,))
}

// ---------------------------------------------------------------------------
// Masked manipulation (mask propagated)
// ---------------------------------------------------------------------------

/// `numpy.ma.reshape(a, newshape)` — reshape data + mask.
#[pyfunction]
pub fn reshape<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    newshape: &Bound<'py, PyAny>,
) -> PyResult<PyMaskedArray> {
    let m = coerce_to_ma(py, a)?;
    let shape = crate::conv::extract_shape(newshape)?;
    let out = m.reshape(&shape).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(out))
}

/// `numpy.ma.ravel(a)` — flatten data + mask to 1-D.
#[pyfunction]
pub fn ravel<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    let m = coerce_to_ma(py, a)?;
    let out = m.ravel().map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(ix1_ma_to_dyn(out)?))
}

/// `numpy.ma.transpose(a)` — transpose data + mask (reverse axes).
#[pyfunction]
pub fn transpose<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    let m = coerce_to_ma(py, a)?;
    let out = m.transpose(None).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(out))
}

/// `numpy.ma.squeeze(a)` — drop all length-1 axes from data + mask.
#[pyfunction]
pub fn squeeze<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    let m = coerce_to_ma(py, a)?;
    let out = m.squeeze(None).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(out))
}

/// `numpy.ma.expand_dims(a, axis)` — insert a length-1 axis at `axis`.
#[pyfunction]
pub fn expand_dims<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: usize,
) -> PyResult<PyMaskedArray> {
    let m = coerce_to_ma(py, a)?;
    let out = m.expand_dims(axis).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(out))
}

/// `numpy.ma.concatenate(arrays, axis=0)` — concatenate masked arrays,
/// concatenating their masks alongside the data.
#[pyfunction]
#[pyo3(signature = (arrays, axis = 0))]
pub fn concatenate<'py>(
    py: Python<'py>,
    arrays: &Bound<'py, PyAny>,
    axis: usize,
) -> PyResult<PyMaskedArray> {
    let seq: Vec<Bound<'py, PyAny>> = arrays.extract()?;
    if seq.is_empty() {
        return Err(PyValueError::new_err(
            "ma.concatenate: need at least one array to concatenate",
        ));
    }
    let mut acc = coerce_to_ma(py, &seq[0])?;
    for item in &seq[1..] {
        let next = coerce_to_ma(py, item)?;
        acc = fma::ma_concatenate(&acc, &next, axis).map_err(ferr_to_pyerr)?;
    }
    Ok(PyMaskedArray::from_inner(acc))
}

/// `numpy.ma.diag(a, k=0)` — masked diagonal / 2-D diagonal embedding,
/// mirroring `numpy.diag` for the data and the mask separately.
#[pyfunction]
#[pyo3(signature = (a, k = 0))]
pub fn diag<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, k: i64) -> PyResult<PyMaskedArray> {
    let m = coerce_to_ma(py, a)?;
    let np = py.import("numpy")?;
    let data: ArrayD<f64> = fma::getdata(&m).map_err(ferr_to_pyerr)?;
    let mask: ArrayD<bool> = fma::getmaskarray(&m).map_err(ferr_to_pyerr)?;
    let data_py = data.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    let mask_py = mask.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    let data_diag = coerce_dtype(py, &np.call_method1("diag", (data_py, k))?, "float64")?;
    let mask_diag = coerce_dtype(py, &np.call_method1("diag", (mask_py, k))?, "bool")?;
    let data_fa: ArrayD<f64> = data_diag
        .extract::<PyReadonlyArrayDyn<f64>>()?
        .as_ferray()
        .map_err(ferr_to_pyerr)?;
    let mask_fa: ArrayD<bool> = mask_diag
        .extract::<PyReadonlyArrayDyn<bool>>()?
        .as_ferray()
        .map_err(ferr_to_pyerr)?;
    let inner = RustMa::new(data_fa, mask_fa).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(inner))
}

/// `numpy.ma.repeat(a, repeats)` — repeat each element `repeats` times,
/// repeating mask alongside the data (1-D result).
#[pyfunction]
pub fn repeat<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    repeats: usize,
) -> PyResult<PyMaskedArray> {
    let m = coerce_to_ma(py, a)?;
    let out = m.repeat(repeats).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(ix1_ma_to_dyn(out)?))
}

/// `numpy.ma.clip(a, a_min, a_max)` — clip unmasked data into
/// `[a_min, a_max]`, mask propagated.
#[pyfunction]
pub fn clip<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    a_min: f64,
    a_max: f64,
) -> PyResult<PyMaskedArray> {
    let m = coerce_to_ma(py, a)?;
    let out = m.clip(a_min, a_max).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(out))
}

// ---------------------------------------------------------------------------
// Mask helpers + predicates
// ---------------------------------------------------------------------------

/// `numpy.ma.getmaskarray(a)` — the mask as a full bool ndarray (never
/// `nomask`): an unmasked array yields an all-`False` array.
#[pyfunction]
pub fn getmaskarray<'py>(py: Python<'py>, a: &PyMaskedArray) -> PyResult<Bound<'py, PyAny>> {
    let mask = a.inner.mask_bits()?;
    Ok(mask.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.ma.make_mask(m)` — normalize an array-like into a boolean mask.
/// When every element is `False`, numpy collapses the result to the
/// `nomask` singleton (`numpy/ma/core.py` `make_mask` `if not m.any():
/// return nomask`); the binding mirrors that.
#[pyfunction]
pub fn make_mask<'py>(py: Python<'py>, m: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let mask = extract_bool_array(py, m)?;
    if !mask.iter().any(|&b| b) {
        return ma_nomask(py);
    }
    Ok(mask.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.ma.make_mask_none(shape)` — an all-`False` bool mask of `shape`.
#[pyfunction]
pub fn make_mask_none<'py>(
    py: Python<'py>,
    shape: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let dims = crate::conv::extract_shape(shape)?;
    let n: usize = dims.iter().product();
    let mask =
        ArrayD::<bool>::from_vec(IxDyn::new(&dims), vec![false; n]).map_err(ferr_to_pyerr)?;
    Ok(mask.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.ma.mask_or(m1, m2)` — elementwise OR of two boolean masks.
#[pyfunction]
pub fn mask_or<'py>(
    py: Python<'py>,
    m1: &Bound<'py, PyAny>,
    m2: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let a = extract_bool_array(py, m1)?;
    let b = extract_bool_array(py, m2)?;
    let out = fma::mask_or(&a, &b).map_err(ferr_to_pyerr)?;
    Ok(out.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.ma.masked_values(x, value, rtol=1e-5, atol=1e-8)` — mask where
/// `x` is approximately `value`. Sets the result's `fill_value` to `value`.
#[pyfunction]
#[pyo3(signature = (x, value, rtol = 1e-5, atol = 1e-8))]
pub fn masked_values<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    value: f64,
    rtol: f64,
    atol: f64,
) -> PyResult<PyMaskedArray> {
    let data_fa = extract_data(py, x)?;
    let mut inner = fma::masked_values(&data_fa, value, rtol, atol).map_err(ferr_to_pyerr)?;
    inner.set_fill_value(value);
    Ok(PyMaskedArray::from_inner(inner))
}

/// `numpy.ma.masked_object(x, value)` — mask where `x` exactly equals
/// `value`. For the f64 binding this is `masked_equal` with the result
/// `fill_value` set to `value` (numpy/ma/core.py `masked_object`).
#[pyfunction]
pub fn masked_object<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    value: f64,
) -> PyResult<PyMaskedArray> {
    let data_fa = extract_data(py, x)?;
    let mut inner = fma::masked_equal(&data_fa, value).map_err(ferr_to_pyerr)?;
    inner.set_fill_value(value);
    Ok(PyMaskedArray::from_inner(inner))
}

/// `numpy.ma.fix_invalid(a)` — mask NaN/inf positions; the masked data is
/// replaced with the fill value (numpy default `1e20`). Returns a
/// `MaskedArray`.
#[pyfunction]
pub fn fix_invalid<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    let m = coerce_to_ma(py, a)?;
    let data: ArrayD<f64> = fma::getdata(&m).map_err(ferr_to_pyerr)?;
    let out = fma::fix_invalid(&data, fma::default_fill_value_f64()).map_err(ferr_to_pyerr)?;
    // Combine with any pre-existing mask (numpy ORs the invalid mask in).
    let prior = fma::getmaskarray(&m).map_err(ferr_to_pyerr)?;
    let new_mask = fma::getmaskarray(&out).map_err(ferr_to_pyerr)?;
    let combined = fma::mask_or(&prior, &new_mask).map_err(ferr_to_pyerr)?;
    let out_data: ArrayD<f64> = fma::getdata(&out).map_err(ferr_to_pyerr)?;
    let inner = RustMa::new(out_data, combined).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(inner))
}

/// `numpy.ma.is_mask(m)` — true iff `m` is a plain boolean ndarray (or the
/// `nomask` singleton). numpy returns `False` for a non-bool array or a
/// MaskedArray. The binding mirrors numpy: a numpy bool ndarray / `nomask`
/// is a mask; anything else is not.
#[pyfunction]
pub fn is_mask<'py>(py: Python<'py>, m: &Bound<'py, PyAny>) -> PyResult<bool> {
    // nomask singleton.
    if m.is(&ma_nomask(py)?) {
        return Ok(true);
    }
    // A ferray/numpy MaskedArray is NOT a mask.
    if m.extract::<PyMaskedArray>().is_ok() {
        return Ok(false);
    }
    let np = py.import("numpy")?;
    let is_ndarray = m.is_instance(&np.getattr("ndarray")?)?;
    if !is_ndarray {
        return Ok(false);
    }
    let dt: String = m.getattr("dtype")?.getattr("name")?.extract()?;
    Ok(dt == "bool")
}

/// `numpy.ma.isMaskedArray(a)` / `isMA` / `isarray` — true iff `a` is a
/// masked array.
#[pyfunction]
#[pyo3(name = "isMaskedArray")]
pub fn is_masked_array<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<bool> {
    if a.extract::<PyMaskedArray>().is_ok() {
        return Ok(true);
    }
    // A genuine numpy.ma.MaskedArray passed through also counts.
    let np_ma = py.import("numpy.ma")?;
    a.is_instance(&np_ma.getattr("MaskedArray")?)
}

/// `numpy.ma.set_fill_value(a, fill_value)` — set the masked array's fill
/// value in place. The given `fill_value` is cast to the array's native dtype
/// (numpy stores the fill in `self.dtype`).
#[pyfunction]
pub fn set_fill_value(
    py: Python<'_>,
    a: &mut PyMaskedArray,
    fill_value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let dt = a.inner.dtype_name();
    match_ma!(&mut a.inner, m, T => {
        let fv = coerce_scalar::<T>(py, fill_value, dt)?;
        m.set_fill_value(fv);
        Ok(())
    })
}

/// `numpy.ma.default_fill_value(obj)` — the default fill value used for
/// masked positions of a float64 array (`1e20`).
#[pyfunction]
pub fn default_fill_value(_obj: &Bound<'_, PyAny>) -> f64 {
    fma::default_fill_value_f64()
}

// ---------------------------------------------------------------------------
// Stateful mask: harden/soften/put/putmask (refs #835 #842 #843 #844 #845)
//
// numpy's `harden_mask`/`soften_mask`/`put`/`putmask` MUTATE the array in
// place. The module-level functions take the pyclass by `Py<PyMaskedArray>`
// (a reference to the SAME Python object the caller holds), borrow it mutably,
// and mutate the wrapped `MaskedArray` in the pyclass cell — NOT a clone. The
// `harden_mask`/`soften_mask` functions return that same object (numpy returns
// the mutated array); `put`/`putmask` return `None` like numpy.
// ---------------------------------------------------------------------------

/// `numpy.ma.harden_mask(a)` — harden `a`'s mask in place and return `a`.
///
/// numpy exposes this via a `_frommethod` wrapper that calls
/// `a.harden_mask()` and returns the (same, mutated) array
/// (`numpy/ma/core.py:3620`). Taking `a: Py<PyMaskedArray>` and returning a
/// new reference to the same object preserves numpy's "returns the mutated
/// self" contract: `b = ma.harden_mask(a)` leaves `a` and `b` aliasing one
/// hardened array.
#[pyfunction]
pub fn harden_mask(py: Python<'_>, a: Py<PyMaskedArray>) -> PyResult<Py<PyMaskedArray>> {
    a.borrow_mut(py).harden_mask()?;
    Ok(a)
}

/// `numpy.ma.soften_mask(a)` — soften `a`'s mask in place and return `a`
/// (`numpy/ma/core.py:3638`).
#[pyfunction]
pub fn soften_mask(py: Python<'_>, a: Py<PyMaskedArray>) -> PyResult<Py<PyMaskedArray>> {
    a.borrow_mut(py).soften_mask()?;
    Ok(a)
}

/// `numpy.ma.put(a, indices, values, mode='raise')` — pure delegation to
/// `a.put(...)` (`numpy/ma/core.py:7496` `return a.put(indices, values,
/// mode=mode)`). Mutates `a` in place and returns `None`.
#[pyfunction]
#[pyo3(signature = (a, indices, values, mode = "raise"))]
pub fn put<'py>(
    py: Python<'py>,
    a: Py<PyMaskedArray>,
    indices: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    mode: &str,
) -> PyResult<()> {
    a.borrow_mut(py).put(py, indices, values, mode)
}

/// `numpy.ma.putmask(a, mask, values)` — conditional in-place assignment
/// where `mask` is true (`numpy/ma/core.py:7537`). Mutates `a` in place and
/// returns `None`.
#[pyfunction]
pub fn putmask<'py>(
    py: Python<'py>,
    a: Py<PyMaskedArray>,
    mask: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
) -> PyResult<()> {
    a.borrow_mut(py).putmask(py, mask, values)
}

// ===========================================================================
// numpy.ma specialized algorithms (refs #835): bindings over the existing
// ferray-ma library functions (`MaskedArray::sort`/`sort_axis`/`argsort`/
// `take`/`trace`, `ma_dot_flat`, `ma_unique`, `ma_vander`, `ma_isin`/
// `ma_in1d`). Each numpy.ma contract was verified live against numpy 2.4.5;
// the pytest oracle (tests/test_expansion_ma_specialized.py) constructs every
// expected value from `numpy.ma.*` directly (R-CHAR-3).
//
// Masked-sort convention (numpy.ma.sort / argsort): unmasked values sort
// ascending, masked values trail at the end of each lane
// (`numpy/ma/core.py` `MaskedArray.sort`: "Place the masked values at the
// end"). Default `axis=-1` (last axis), `axis=None` flattens — matching
// numpy.ma's documented default for `sort`.
// ===========================================================================

/// Build the `f64` IxDyn data + bool IxDyn mask of a masked array as flat
/// row-major `Vec`s plus its shape — the common entry point for the typed
/// (Ix1 / Ix2) ferray-ma algorithms below.
fn ma_parts(m: &RustMa<f64, IxDyn>) -> PyResult<(Vec<f64>, Vec<bool>, Vec<usize>)> {
    let data: ArrayD<f64> = fma::getdata(m).map_err(ferr_to_pyerr)?;
    let mask: ArrayD<bool> = fma::getmaskarray(m).map_err(ferr_to_pyerr)?;
    let shape = data.shape().to_vec();
    let data_v: Vec<f64> = data.iter().copied().collect();
    let mask_v: Vec<bool> = mask.iter().copied().collect();
    Ok((data_v, mask_v, shape))
}

/// Reinterpret a masked array as a 1-D `MaskedArray<f64, Ix1>` (flattening
/// in row-major order). Used by the algorithms ferray-ma exposes only for
/// the 1-D case (`vander`, the flat `dot`, `in1d`).
fn ma_as_ix1(m: &RustMa<f64, IxDyn>) -> PyResult<RustMa<f64, Ix1>> {
    let (data_v, mask_v, _shape) = ma_parts(m)?;
    let n = data_v.len();
    let data_arr = Array::<f64, Ix1>::from_vec(Ix1::new([n]), data_v).map_err(ferr_to_pyerr)?;
    let mask_arr = Array::<bool, Ix1>::from_vec(Ix1::new([n]), mask_v).map_err(ferr_to_pyerr)?;
    RustMa::new(data_arr, mask_arr).map_err(ferr_to_pyerr)
}

/// Reinterpret a masked array as a 2-D `MaskedArray<f64, Ix2>`. Errors with
/// a `ValueError` if the array is not 2-D (matching the dimensionality
/// numpy.ma.trace requires for a square 2-D trace).
fn ma_as_ix2(m: &RustMa<f64, IxDyn>) -> PyResult<RustMa<f64, Ix2>> {
    let (data_v, mask_v, shape) = ma_parts(m)?;
    if shape.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "ma operation requires a 2-D masked array, got {}-D",
            shape.len()
        )));
    }
    let dim = Ix2::new([shape[0], shape[1]]);
    let data_arr = Array::<f64, Ix2>::from_vec(dim.clone(), data_v).map_err(ferr_to_pyerr)?;
    let mask_arr = Array::<bool, Ix2>::from_vec(dim, mask_v).map_err(ferr_to_pyerr)?;
    RustMa::new(data_arr, mask_arr).map_err(ferr_to_pyerr)
}

/// `numpy.ma.sort(a, axis=-1)` — sort with masked values pushed to the end
/// of each lane. `axis=None` flattens first (numpy.ma collapses to 1-D),
/// any other `axis` sorts each 1-D slice independently via the ferray-ma
/// `sort_axis`. Default `axis=-1` mirrors numpy.ma.sort's documented
/// default (`numpy/ma/core.py` `def sort(self, axis=-1, ...)`).
#[pyfunction]
#[pyo3(signature = (a, axis = Some(-1)))]
pub fn sort<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<PyMaskedArray> {
    let m = coerce_to_ma(py, a)?;
    match axis {
        None => {
            // Flatten and sort (masked values last).
            let m1 = ma_as_ix1(&m)?;
            let out = m1.sort().map_err(ferr_to_pyerr)?;
            Ok(PyMaskedArray::from_inner(ix1_ma_to_dyn(out)?))
        }
        Some(ax) => {
            let ndim = m.ndim();
            let axis_u = crate::conv::normalize_axis(py, ax, ndim)?;
            let out = m.sort_axis(axis_u).map_err(ferr_to_pyerr)?;
            Ok(PyMaskedArray::from_inner(out))
        }
    }
}

/// `numpy.ma.argsort(a, axis=None)` — indices that would sort a masked array,
/// with masked positions trailing each lane
/// (`numpy/ma/core.py` `MaskedArray.argsort`). With `axis=None` (or a 1-D
/// array) ferray-ma flattens and returns a 1-D index array; with an explicit
/// `axis` on a multi-dimensional array, `argsort_axis` sorts each lane
/// independently (masked entries fill-to-max so they trail, `endwith=True`),
/// returning an index array of the input shape. Results are numpy
/// `intp`-style `uint64` arrays.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn argsort<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = coerce_to_ma(py, a)?;
    match axis {
        // Flat contract: numpy.ma collapses axis=None (or a 1-D input) to a
        // single flattened argsort.
        None => {
            let m1 = ma_as_ix1(&m)?;
            let idx: Array1<u64> = m1.argsort().map_err(ferr_to_pyerr)?;
            Ok(idx.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
        }
        Some(_) if m.ndim() <= 1 => {
            let m1 = ma_as_ix1(&m)?;
            let idx: Array1<u64> = m1.argsort().map_err(ferr_to_pyerr)?;
            Ok(idx.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
        }
        Some(ax) => {
            let ndim = m.ndim();
            let axis_u = crate::conv::normalize_axis(py, ax, ndim)?;
            let idx: ArrayD<u64> = m.argsort_axis(axis_u).map_err(ferr_to_pyerr)?;
            Ok(idx.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
        }
    }
}

/// `numpy.ma.take(a, indices)` — gather elements at flat `indices`,
/// carrying the mask of each picked position
/// (`numpy/ma/core.py` `def take(...)`). Returns a 1-D masked array.
#[pyfunction]
pub fn take<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
) -> PyResult<PyMaskedArray> {
    let m = coerce_to_ma(py, a)?;
    let idx_arr = extract_data(py, indices)?;
    let idx: Vec<usize> = idx_arr
        .iter()
        .map(|&v| {
            if v < 0.0 || v.fract() != 0.0 {
                Err(PyValueError::new_err(
                    "ma.take: indices must be non-negative integers",
                ))
            } else {
                Ok(v as usize)
            }
        })
        .collect::<PyResult<Vec<usize>>>()?;
    let out = m.take(&idx).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(ix1_ma_to_dyn(out)?))
}

/// `numpy.ma.trace(a, offset=0)` — sum of the (offset) diagonal of a 2-D
/// masked array, masked diagonal entries contributing zero
/// (`numpy/ma/core.py` `def trace(...)`). Returns a Python float.
#[pyfunction]
#[pyo3(signature = (a, offset = 0))]
pub fn trace<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, offset: i64) -> PyResult<f64> {
    let m = coerce_to_ma(py, a)?;
    let m2 = ma_as_ix2(&m)?;
    m2.trace(offset as isize).map_err(ferr_to_pyerr)
}

/// `numpy.ma.dot(a, b)` — masked dot product (`numpy/ma/core.py:8214`,
/// non-strict default). Two shapes are supported:
///
/// - **1-D × 1-D** returns the scalar inner product as a Python float; masked
///   positions on either operand contribute zero (`ma_dot_flat`).
/// - **2-D × 2-D** returns a `MaskedArray` matrix product (`ma_dot_2d`): the
///   data is `dot(filled(a,0), filled(b,0))` and `out[i,j]` is masked iff
///   every contributing `k` has `a[i,k]` OR `b[k,j]` masked
///   (`m = ~dot(~mask_a, ~mask_b)`).
///
/// Mixed 1-D/2-D operands remain a ferray-ma library gap and raise a clear
/// `ValueError`.
#[pyfunction]
pub fn dot<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let ma = coerce_to_ma(py, a)?;
    let mb = coerce_to_ma(py, b)?;
    match (ma.ndim(), mb.ndim()) {
        (d, e) if d <= 1 && e <= 1 => {
            let m1a = ma_as_ix1(&ma)?;
            let m1b = ma_as_ix1(&mb)?;
            let s = m1a.ma_dot_flat(&m1b).map_err(ferr_to_pyerr)?;
            Ok(s.into_pyobject(py)?.into_any())
        }
        (2, 2) => {
            let m2a = ma_as_ix2(&ma)?;
            let m2b = ma_as_ix2(&mb)?;
            let out = m2a.ma_dot_2d(&m2b).map_err(ferr_to_pyerr)?;
            let py_ma = ix2_to_py(out)?;
            Ok(Py::new(py, py_ma)?.into_bound(py).into_any())
        }
        (d, e) => Err(PyValueError::new_err(format!(
            "ma.dot supports 1-D inner products and 2-D matrix products; mixed \
             {d}-D × {e}-D operands are a ferray-ma library gap (#835)"
        ))),
    }
}

/// `numpy.ma.unique(a)` — sorted unique unmasked values, with a single
/// trailing masked entry iff the input contained any masked element
/// (`numpy/ma/core.py` `def unique(...)`: masked values collapse to one
/// `masked` slot at the end). ferray-ma's `ma_unique` yields the sorted
/// unique *unmasked* values; the binding appends the one masked entry to
/// match numpy's observable mask.
#[pyfunction]
pub fn unique<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    let m = coerce_to_ma(py, a)?;
    let uniq: Array1<f64> = fma::ma_unique(&m).map_err(ferr_to_pyerr)?;
    let mut data: Vec<f64> = uniq.iter().copied().collect();
    let mut mask: Vec<bool> = vec![false; data.len()];

    // Mirror numpy.ma.unique: a single masked slot trails iff any input
    // element was masked. Its data value is not observable (masked), so we
    // reuse the first masked input value as the placeholder.
    let (in_data, in_mask, _shape) = ma_parts(&m)?;
    if let Some(pos) = in_mask.iter().position(|&b| b) {
        data.push(in_data[pos]);
        mask.push(true);
    }

    let n = data.len();
    let data_arr = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[n]), data).map_err(ferr_to_pyerr)?;
    let mask_arr = Array::<bool, IxDyn>::from_vec(IxDyn::new(&[n]), mask).map_err(ferr_to_pyerr)?;
    let inner = RustMa::new(data_arr, mask_arr).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(inner))
}

// ===========================================================================
// numpy.ma specialized masked algorithms (refs #835): where/choose/diff/
// ediff1d/nonzero, plus the now-corrected vander/isin/in1d. Each binds the
// matching ferray-ma library function; every numpy.ma contract was verified
// live against numpy 2.4.5 and the pytest oracle constructs expected values
// from `numpy.ma.*` directly (R-CHAR-3).
// ===========================================================================

/// Coerce a Python object into a 1-/N-D bool `MaskedArray<bool, IxDyn>`.
///
/// A `ferray.ma.MaskedArray` is reinterpreted by thresholding its float data
/// at `!= 0.0` (numpy treats any non-zero as `True`), preserving its mask; any
/// other array-like is routed through `numpy.asarray(..., bool)` with a
/// `nomask` mask.
fn coerce_to_bool_ma<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
) -> PyResult<RustMa<bool, IxDyn>> {
    if let Ok(m) = obj.extract::<PyMaskedArray>() {
        let (data, mask, shape) = ma_parts(&m.to_f64_ma()?)?;
        let bdata: Vec<bool> = data.iter().map(|&v| v != 0.0).collect();
        let data_arr =
            ArrayD::<bool>::from_vec(IxDyn::new(&shape), bdata).map_err(ferr_to_pyerr)?;
        let mask_arr = ArrayD::<bool>::from_vec(IxDyn::new(&shape), mask).map_err(ferr_to_pyerr)?;
        return RustMa::new(data_arr, mask_arr).map_err(ferr_to_pyerr);
    }
    let data = extract_bool_array(py, obj)?;
    let n = data.size();
    let shape = data.shape().to_vec();
    let mask =
        ArrayD::<bool>::from_vec(IxDyn::new(&shape), vec![false; n]).map_err(ferr_to_pyerr)?;
    RustMa::new(data, mask).map_err(ferr_to_pyerr)
}

/// `numpy.ma.where(condition, x, y)` — masked elementwise select
/// (`numpy/ma/core.py:7915`). Result masked where the chosen source is masked
/// OR the condition is masked; a masked condition picks the `y` branch.
/// Bound as `where_` to avoid colliding with the Rust keyword; registered
/// under the Python name `where`.
#[pyfunction]
#[pyo3(name = "where")]
pub fn where_<'py>(
    py: Python<'py>,
    condition: &Bound<'py, PyAny>,
    x: &Bound<'py, PyAny>,
    y: &Bound<'py, PyAny>,
) -> PyResult<PyMaskedArray> {
    let cond = coerce_to_bool_ma(py, condition)?;
    let mx = coerce_to_ma(py, x)?;
    let my = coerce_to_ma(py, y)?;
    let out = fma::ma_where(&cond, &mx, &my).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(out))
}

/// `numpy.ma.choose(indices, choices)` — masked choose
/// (`numpy/ma/core.py:8007`). A masked index is filled with 0; the result is
/// masked where the chosen source OR the index is masked.
#[pyfunction]
pub fn choose<'py>(
    py: Python<'py>,
    indices: &Bound<'py, PyAny>,
    choices: &Bound<'py, PyAny>,
) -> PyResult<PyMaskedArray> {
    let idx = coerce_to_ma(py, indices)?;
    let seq = choices.try_iter()?;
    let mut chs: Vec<RustMa<f64, IxDyn>> = Vec::new();
    for item in seq {
        chs.push(coerce_to_ma(py, &item?)?);
    }
    let out = fma::ma_choose(&idx, &chs).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(out))
}

/// `numpy.ma.diff(a, n=1, axis=-1)` — masked adjacent difference
/// (`numpy/ma/core.py:7774`); result masked iff either operand is masked.
#[pyfunction]
#[pyo3(signature = (a, n = 1, axis = -1))]
pub fn diff<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    n: usize,
    axis: isize,
) -> PyResult<PyMaskedArray> {
    let m = coerce_to_ma(py, a)?;
    let out = fma::ma_diff(&m, n, axis).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(out))
}

/// `numpy.ma.ediff1d(ary, to_end=None, to_begin=None)` — flattened first
/// difference (`numpy/ma/extras.py:1229`); `to_begin`/`to_end` always unmasked.
#[pyfunction]
#[pyo3(signature = (ary, to_end = None, to_begin = None))]
pub fn ediff1d<'py>(
    py: Python<'py>,
    ary: &Bound<'py, PyAny>,
    to_end: Option<&Bound<'py, PyAny>>,
    to_begin: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyMaskedArray> {
    let m = coerce_to_ma(py, ary)?;
    let begin: Option<Vec<f64>> = match to_begin {
        Some(o) => Some(extract_data(py, o)?.iter().copied().collect()),
        None => None,
    };
    let end: Option<Vec<f64>> = match to_end {
        Some(o) => Some(extract_data(py, o)?.iter().copied().collect()),
        None => None,
    };
    let out = fma::ma_ediff1d(&m, begin.as_deref(), end.as_deref()).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(ix1_ma_to_dyn(out)?))
}

/// `numpy.ma.nonzero(a)` — tuple of per-dimension index arrays of the non-zero
/// UNMASKED elements (masked treated as zero; `numpy/ma/core.py:5049`). Returns
/// a Python tuple of `int64` numpy arrays, matching numpy's layout.
#[pyfunction]
pub fn nonzero<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let m = coerce_to_ma(py, a)?;
    let coords = fma::ma_nonzero(&m).map_err(ferr_to_pyerr)?;
    let mut arrays: Vec<Bound<'py, PyAny>> = Vec::with_capacity(coords.len());
    for axis_coords in coords {
        arrays.push(
            axis_coords
                .into_pyarray(py)
                .map_err(ferr_to_pyerr)?
                .into_any(),
        );
    }
    Ok(pyo3::types::PyTuple::new(py, arrays)?.into_any())
}

// ===========================================================================
// numpy.ma mask-structure run-length analysis (refs #835): clump_masked /
// clump_unmasked / notmasked_contiguous / notmasked_edges /
// flatnotmasked_contiguous / flatnotmasked_edges. The ferray-ma library returns
// half-open `(start, stop)` index pairs; the binding materializes Python
// `slice` objects to match numpy's observable `slice(start, stop, None)` output.
// Every numpy.ma contract was verified live against numpy 2.4.5 (R-CHAR-3).
// ===========================================================================

/// Build a Python `slice(start, stop, None)` from a half-open index pair.
///
/// numpy's `_ezclump` yields `slice(start, stop)` with `step=None`; the
/// `slice` builtin produces exactly that two-argument form (PyO3's
/// `PySlice::new` always materializes an explicit step, which would diverge
/// from numpy's observable `step is None`).
fn pyslice<'py>(py: Python<'py>, (start, stop): (usize, usize)) -> PyResult<Bound<'py, PyAny>> {
    let builtins = py.import("builtins")?;
    builtins.getattr("slice")?.call1((start, stop))
}

/// Convert a `Vec<(usize, usize)>` into a Python list of `slice` objects.
fn slice_list<'py>(py: Python<'py>, runs: &[(usize, usize)]) -> PyResult<Bound<'py, PyAny>> {
    let items: Vec<Bound<'py, PyAny>> = runs
        .iter()
        .map(|&p| pyslice(py, p))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(pyo3::types::PyList::new(py, items)?.into_any())
}

/// `numpy.ma.clump_masked(a)` (`numpy/ma/extras.py:2189`) — list of `slice`
/// objects, one per contiguous run of masked elements in a 1-D masked array.
#[pyfunction]
pub fn clump_masked<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let m = coerce_to_ma(py, a)?;
    let runs = fma::clump_masked(&m);
    slice_list(py, &runs)
}

/// `numpy.ma.clump_unmasked(a)` (`numpy/ma/extras.py:2235`) — list of `slice`
/// objects, one per contiguous run of unmasked elements.
#[pyfunction]
pub fn clump_unmasked<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let m = coerce_to_ma(py, a)?;
    let runs = fma::clump_unmasked(&m);
    slice_list(py, &runs)
}

/// `numpy.ma.flatnotmasked_contiguous(a)` (`numpy/ma/extras.py:1818`) — list of
/// `slice` objects of contiguous unmasked regions of the flattened array.
#[pyfunction]
pub fn flatnotmasked_contiguous<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = coerce_to_ma(py, a)?;
    let runs = fma::flatnotmasked_contiguous(&m);
    slice_list(py, &runs)
}

/// `numpy.ma.flatnotmasked_edges(a)` (`numpy/ma/extras.py:1762`) — `[first,
/// last]` unmasked indices as an `int64` numpy array, or `None` if every
/// element is masked.
#[pyfunction]
pub fn flatnotmasked_edges<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = coerce_to_ma(py, a)?;
    match fma::flatnotmasked_edges(&m) {
        None => Ok(py.None().into_bound(py)),
        Some([first, last]) => {
            let arr = Array1::<i64>::from_vec(Ix1::new([2]), vec![first as i64, last as i64])
                .map_err(ferr_to_pyerr)?;
            Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
        }
    }
}

/// `numpy.ma.notmasked_edges(a, axis=None)` (`numpy/ma/extras.py:1878`) — for
/// `axis=None` or a 1-D array, the flat `[first, last]` unmasked indices (or
/// `None` if all masked). The multi-axis coordinate-tuple form is a deferred
/// ferray-ma extension; an explicit `axis` on a multi-dimensional array raises
/// a clear `ValueError`.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn notmasked_edges<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = coerce_to_ma(py, a)?;
    if axis.is_some() && m.ndim() > 1 {
        return Err(PyValueError::new_err(
            "ma.notmasked_edges with an explicit axis on a multi-dimensional array is a \
             ferray-ma library gap (#835); pass axis=None",
        ));
    }
    match fma::notmasked_edges(&m) {
        None => Ok(py.None().into_bound(py)),
        Some([first, last]) => {
            let arr = Array1::<i64>::from_vec(Ix1::new([2]), vec![first as i64, last as i64])
                .map_err(ferr_to_pyerr)?;
            Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
        }
    }
}

/// `numpy.ma.notmasked_contiguous(a, axis=None)` (`numpy/ma/extras.py:1936`).
///
/// `axis=None` (or a 1-D array) returns a flat list of `slice` objects of the
/// unmasked runs (identical to `flatnotmasked_contiguous`). For a 2-D array
/// with an explicit `axis`, returns a list-of-lists — one slice list per lane
/// along the orthogonal axis.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn notmasked_contiguous<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = coerce_to_ma(py, a)?;
    match axis {
        None => {
            let runs = fma::flatnotmasked_contiguous(&m);
            slice_list(py, &runs)
        }
        Some(_) if m.ndim() <= 1 => {
            let runs = fma::flatnotmasked_contiguous(&m);
            slice_list(py, &runs)
        }
        Some(ax) => {
            let m2 = ma_as_ix2(&m)?;
            let axis_u = crate::conv::normalize_axis(py, ax, 2)?;
            let lanes = fma::notmasked_contiguous_axis(&m2, axis_u).map_err(ferr_to_pyerr)?;
            let lists: Vec<Bound<'py, PyAny>> = lanes
                .iter()
                .map(|runs| slice_list(py, runs))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(pyo3::types::PyList::new(py, lists)?.into_any())
        }
    }
}

/// `numpy.ma.vander(x, n=None)` — masked Vandermonde
/// (`numpy/ma/extras.py:2216`). Masked input rows become all-zeros and the
/// result carries no mask (nomask).
#[pyfunction]
#[pyo3(signature = (x, n = None))]
pub fn vander<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    n: Option<usize>,
) -> PyResult<PyMaskedArray> {
    let m = coerce_to_ma(py, x)?;
    let m1 = ma_as_ix1(&m)?;
    let out = fma::ma_vander(&m1, n).map_err(ferr_to_pyerr)?;
    let data = fma::getdata(&out).map_err(ferr_to_pyerr)?.into_dyn();
    let mask = fma::getmaskarray(&out).map_err(ferr_to_pyerr)?.into_dyn();
    let inner = RustMa::new(data, mask).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(inner))
}

/// `numpy.ma.isin(element, test_elements)` — masked membership
/// (`numpy/ma/extras.py:1434`). Carries the input mask; masked positions are
/// reported `False` (a masked value is never a member).
#[pyfunction]
pub fn isin<'py>(
    py: Python<'py>,
    element: &Bound<'py, PyAny>,
    test_elements: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = coerce_to_ma(py, element)?;
    let tests: Vec<f64> = extract_data(py, test_elements)?.iter().copied().collect();
    let out = fma::ma_isin(&m, &tests).map_err(ferr_to_pyerr)?;
    // Return a plain numpy bool ndarray of the underlying membership data,
    // matching numpy.ma.isin's observable (the boolean result shape of
    // `element`); masked positions read `False`.
    let data = fma::getdata(&out).map_err(ferr_to_pyerr)?;
    Ok(data.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.ma.in1d(ar1, ar2)` — flattened masked membership
/// (`numpy/ma/extras.py:1387`). 1-D `isin`.
#[pyfunction]
pub fn in1d<'py>(
    py: Python<'py>,
    ar1: &Bound<'py, PyAny>,
    ar2: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = coerce_to_ma(py, ar1)?;
    let m1 = ma_as_ix1(&m)?;
    let tests: Vec<f64> = extract_data(py, ar2)?.iter().copied().collect();
    let out = fma::ma_in1d(&m1, &tests).map_err(ferr_to_pyerr)?;
    let data = fma::getdata(&out).map_err(ferr_to_pyerr)?;
    Ok(data.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

// ===========================================================================
// numpy.ma set operations, row/col suppression, and masked cov/corrcoef
// (refs #835). Each binds the matching ferray-ma library function added this
// iteration; every numpy.ma contract was verified live against numpy 2.4.5 and
// the pytest oracle (tests/test_expansion_ma_lib2.py) constructs every expected
// value from `numpy.ma.*` directly (R-CHAR-3).
// ===========================================================================

/// Promote a 1-D `MaskedArray<f64, Ix1>` produced by the masked set ops into
/// the dynamic-dimension wrapper the `PyMaskedArray` class holds.
fn ix1_to_py(m: RustMa<f64, Ix1>) -> PyResult<PyMaskedArray> {
    Ok(PyMaskedArray::from_inner(ix1_ma_to_dyn(m)?))
}

/// Promote a 2-D `MaskedArray<f64, Ix2>` into the dynamic-dimension wrapper.
fn ix2_to_py(m: RustMa<f64, Ix2>) -> PyResult<PyMaskedArray> {
    let data = fma::getdata(&m).map_err(ferr_to_pyerr)?.into_dyn();
    let mask = fma::getmaskarray(&m).map_err(ferr_to_pyerr)?.into_dyn();
    let inner = RustMa::new(data, mask).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(inner))
}

/// `numpy.ma.intersect1d(ar1, ar2)` — sorted unique common values, with a
/// trailing masked slot iff both inputs were masked
/// (`numpy/ma/extras.py:1317`).
#[pyfunction]
pub fn intersect1d<'py>(
    py: Python<'py>,
    ar1: &Bound<'py, PyAny>,
    ar2: &Bound<'py, PyAny>,
) -> PyResult<PyMaskedArray> {
    let a = ma_as_ix1(&coerce_to_ma(py, ar1)?)?;
    let b = ma_as_ix1(&coerce_to_ma(py, ar2)?)?;
    ix1_to_py(fma::ma_intersect1d(&a, &b).map_err(ferr_to_pyerr)?)
}

/// `numpy.ma.union1d(ar1, ar2)` — sorted unique union, masked slot iff either
/// input was masked (`numpy/ma/extras.py:1463`).
#[pyfunction]
pub fn union1d<'py>(
    py: Python<'py>,
    ar1: &Bound<'py, PyAny>,
    ar2: &Bound<'py, PyAny>,
) -> PyResult<PyMaskedArray> {
    let a = ma_as_ix1(&coerce_to_ma(py, ar1)?)?;
    let b = ma_as_ix1(&coerce_to_ma(py, ar2)?)?;
    ix1_to_py(fma::ma_union1d(&a, &b).map_err(ferr_to_pyerr)?)
}

/// `numpy.ma.setdiff1d(ar1, ar2)` — sorted unique values of `ar1` not in `ar2`
/// (`numpy/ma/extras.py:1485`).
#[pyfunction]
pub fn setdiff1d<'py>(
    py: Python<'py>,
    ar1: &Bound<'py, PyAny>,
    ar2: &Bound<'py, PyAny>,
) -> PyResult<PyMaskedArray> {
    let a = ma_as_ix1(&coerce_to_ma(py, ar1)?)?;
    let b = ma_as_ix1(&coerce_to_ma(py, ar2)?)?;
    ix1_to_py(fma::ma_setdiff1d(&a, &b).map_err(ferr_to_pyerr)?)
}

/// `numpy.ma.setxor1d(ar1, ar2)` — symmetric difference of the unique values
/// (`numpy/ma/extras.py:1350`).
#[pyfunction]
pub fn setxor1d<'py>(
    py: Python<'py>,
    ar1: &Bound<'py, PyAny>,
    ar2: &Bound<'py, PyAny>,
) -> PyResult<PyMaskedArray> {
    let a = ma_as_ix1(&coerce_to_ma(py, ar1)?)?;
    let b = ma_as_ix1(&coerce_to_ma(py, ar2)?)?;
    ix1_to_py(fma::ma_setxor1d(&a, &b).map_err(ferr_to_pyerr)?)
}

/// `numpy.ma.compress_rowcols(x, axis)` — suppress whole rows/cols containing
/// masked values; returns a plain ndarray (`numpy/ma/extras.py:920`). 2-D only.
#[pyfunction]
#[pyo3(signature = (x, axis = None))]
pub fn compress_rowcols<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = ma_as_ix2(&coerce_to_ma(py, x)?)?;
    let ax = match axis {
        None => None,
        Some(0) => Some(0usize),
        Some(1) | Some(-1) => Some(1usize),
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "ma.compress_rowcols: axis must be None, 0, 1 or -1, got {other}"
            )));
        }
    };
    let out = fma::ma_compress_rowcols(&m, ax).map_err(ferr_to_pyerr)?;
    Ok(out.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.ma.compress_rows(a)` — suppress whole rows containing masked values
/// (`numpy/ma/extras.py:953`).
#[pyfunction]
pub fn compress_rows<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let m = ma_as_ix2(&coerce_to_ma(py, a)?)?;
    let out = fma::ma_compress_rows(&m).map_err(ferr_to_pyerr)?;
    Ok(out.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.ma.compress_cols(a)` — suppress whole columns containing masked
/// values (`numpy/ma/extras.py:991`).
#[pyfunction]
pub fn compress_cols<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let m = ma_as_ix2(&coerce_to_ma(py, a)?)?;
    let out = fma::ma_compress_cols(&m).map_err(ferr_to_pyerr)?;
    Ok(out.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.ma.mask_rowcols(a, axis)` — mask whole rows/cols containing a masked
/// value (`numpy/ma/extras.py:830`). 2-D only; returns a masked array.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn mask_rowcols<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<PyMaskedArray> {
    let m = ma_as_ix2(&coerce_to_ma(py, a)?)?;
    let ax = match axis {
        None => None,
        Some(0) => Some(0usize),
        Some(1) | Some(-1) => Some(1usize),
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "ma.mask_rowcols: axis must be None, 0, 1 or -1, got {other}"
            )));
        }
    };
    let out = fma::ma_mask_rowcols(&m, ax).map_err(ferr_to_pyerr)?;
    ix2_to_py(out)
}

/// `numpy.ma.cov(x, y=None, rowvar=True, bias=False, ddof=None)` — masked
/// covariance over the unmasked observation pairs (`numpy/ma/extras.py:1580`).
/// A second variable set `y` is stacked with `x` per numpy's `_covhelper`
/// (`numpy/ma/extras.py:1521`) before the masked covariance.
#[pyfunction]
#[pyo3(signature = (x, y = None, rowvar = true, bias = false, allow_masked = true, ddof = None))]
#[allow(
    clippy::fn_params_excessive_bools,
    reason = "mirrors numpy.ma.cov's keyword surface (rowvar/bias/allow_masked)"
)]
pub fn cov<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    y: Option<&Bound<'py, PyAny>>,
    rowvar: bool,
    bias: bool,
    allow_masked: bool,
    ddof: Option<usize>,
) -> PyResult<PyMaskedArray> {
    let _ = allow_masked; // ferray-ma always processes the masked pairs.
    match y {
        None => {
            let m = coerce_to_ma_npmask(py, x)?;
            let out = fma::ma_cov(&m, rowvar, bias, ddof).map_err(ferr_to_pyerr)?;
            ix2_to_py(out)
        }
        Some(yv) => {
            // numpy stacks x and y into one variable matrix (`_covhelper`),
            // then runs the masked covariance with variables in rows.
            let stacked = stack_xy(py, x, yv, rowvar)?;
            let out = fma::ma_cov(&stacked, true, bias, ddof).map_err(ferr_to_pyerr)?;
            ix2_to_py(out)
        }
    }
}

/// `numpy.ma.corrcoef(x, y=None, rowvar=True)` — masked Pearson correlation
/// (`numpy/ma/extras.py:1672`): the masked covariance normalized by the outer
/// product of the per-variable standard deviations. A second variable set `y`
/// is stacked with `x` per numpy's `_covhelper` (`numpy/ma/extras.py:1521`).
#[pyfunction]
#[pyo3(signature = (x, y = None, rowvar = true, allow_masked = true))]
pub fn corrcoef<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    y: Option<&Bound<'py, PyAny>>,
    rowvar: bool,
    allow_masked: bool,
) -> PyResult<PyMaskedArray> {
    let _ = allow_masked;
    match y {
        None => {
            let m = coerce_to_ma_npmask(py, x)?;
            let out = fma::ma_corrcoef(&m, rowvar).map_err(ferr_to_pyerr)?;
            ix2_to_py(out)
        }
        Some(yv) => {
            let stacked = stack_xy(py, x, yv, rowvar)?;
            let out = fma::ma_corrcoef(&stacked, true).map_err(ferr_to_pyerr)?;
            ix2_to_py(out)
        }
    }
}

// ===========================================================================
// numpy.ma residual (refs #837): polyfit / convolve / correlate composed at
// the binding over the existing ferray crates, plus the two-variable `y` form
// of cov / corrcoef. Every contract was verified live against numpy 2.4.5.
// ===========================================================================

/// Mirror numpy.ma's `_covhelper` stacking (`numpy/ma/extras.py:1521`) for the
/// two-variable `cov`/`corrcoef` form: promote `x` and `y` to a single row each
/// (numpy `ndmin=2`; a 1-D input becomes one variable), apply the pairwise
/// common mask when both have the same shape, then concatenate the variables
/// into one `(nvars x nobs)` masked array on which the existing `ma_cov` /
/// `ma_corrcoef` operate with `rowvar=True`.
///
/// numpy forces `rowvar=True` whenever the promoted `x` has a single row
/// (`numpy/ma/extras.py:1532` `if x.shape[0] == 1: rowvar = True`); for two
/// 1-D inputs that is always the case, so the stacked array always has one row
/// per input variable and `rowvar` only selects the concatenation axis for
/// genuinely 2-D inputs.
fn stack_xy<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    y: &Bound<'py, PyAny>,
    rowvar: bool,
) -> PyResult<RustMa<f64, IxDyn>> {
    let (xd, xm, xs) = ma_parts(&coerce_to_ma_npmask(py, x)?)?;
    let (yd, ym, ys) = ma_parts(&coerce_to_ma_npmask(py, y)?)?;

    // Promote each input to a (rows x cols) variable matrix (ndmin=2: a 1-D
    // input is a single variable row). numpy forces rowvar=True for a single
    // row, so we keep variables in rows and observations in columns.
    let promote =
        |shape: &[usize], data: Vec<f64>, mask: Vec<bool>| -> (usize, usize, Vec<f64>, Vec<bool>) {
            match shape.len() {
                0 | 1 => (1, data.len(), data, mask),
                _ => {
                    let (r, c) = (shape[0], shape[1]);
                    if rowvar {
                        (r, c, data, mask)
                    } else {
                        // rowvar=False: variables are columns; transpose to rows.
                        let mut td = vec![0.0f64; r * c];
                        let mut tm = vec![false; r * c];
                        for i in 0..r {
                            for j in 0..c {
                                td[j * r + i] = data[i * c + j];
                                tm[j * r + i] = mask[i * c + j];
                            }
                        }
                        (c, r, td, tm)
                    }
                }
            }
        };

    let (xr, xc, xdata, mut xmask) = promote(&xs, xd, xm);
    let (yr, yc, ydata, mut ymask) = promote(&ys, yd, ym);

    if xc != yc {
        return Err(PyValueError::new_err(format!(
            "ma.cov/corrcoef: x and y must have the same number of observations, got {xc} and {yc}"
        )));
    }
    let nobs = xc;

    // numpy applies a pairwise common mask only when the two promoted arrays
    // have identical shape (`numpy/ma/extras.py:1557`): if either is masked at
    // a position, both become masked there.
    if xr == yr && xc == yc && (xmask.iter().any(|&b| b) || ymask.iter().any(|&b| b)) {
        for k in 0..xmask.len() {
            let common = xmask[k] || ymask[k];
            xmask[k] = common;
            ymask[k] = common;
        }
    }

    let nvars = xr + yr;
    let mut data = vec![0.0f64; nvars * nobs];
    let mut mask = vec![false; nvars * nobs];
    data[..xr * nobs].copy_from_slice(&xdata);
    mask[..xr * nobs].copy_from_slice(&xmask);
    data[xr * nobs..].copy_from_slice(&ydata);
    mask[xr * nobs..].copy_from_slice(&ymask);

    let data_arr =
        Array::<f64, IxDyn>::from_vec(IxDyn::new(&[nvars, nobs]), data).map_err(ferr_to_pyerr)?;
    let mask_arr =
        Array::<bool, IxDyn>::from_vec(IxDyn::new(&[nvars, nobs]), mask).map_err(ferr_to_pyerr)?;
    RustMa::new(data_arr, mask_arr).map_err(ferr_to_pyerr)
}

/// `numpy.ma.polyfit(x, y, deg)` — least-squares polynomial fit ignoring any
/// `(x, y)` pair where either coordinate is masked
/// (`numpy/ma/extras.py:2231`). numpy unions the `x` and `y` masks, drops the
/// masked rows, then delegates to `numpy.polyfit`; the binding composes the
/// same drop over [`coerce_to_ma`] and the existing `Poly::fit`, returning the
/// highest-degree-first coefficients `numpy.polyfit` produces.
///
/// Only a 1-D `y` is supported (the common masked-fit case); a 2-D `y` raises
/// a clear `ValueError`, mirroring numpy's `TypeError` for non-1D/2D `y` in
/// spirit while the multi-column path stays a follow-up.
#[pyfunction]
pub fn polyfit<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    y: &Bound<'py, PyAny>,
    deg: usize,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_polynomial::traits::Poly;

    let (xd, xm, xs) = ma_parts(&coerce_to_ma_npmask(py, x)?)?;
    let (yd, ym, ys) = ma_parts(&coerce_to_ma_npmask(py, y)?)?;
    if xs.len() > 1 || ys.len() > 1 {
        return Err(PyValueError::new_err(
            "ma.polyfit: only 1-D x and y are supported in this binding",
        ));
    }
    if xd.len() != yd.len() {
        return Err(PyValueError::new_err(format!(
            "ma.polyfit: x and y must have the same length, got {} and {}",
            xd.len(),
            yd.len()
        )));
    }

    // Union the per-pair masks and keep only the unmasked coordinates.
    let mut xs_kept = Vec::with_capacity(xd.len());
    let mut ys_kept = Vec::with_capacity(yd.len());
    for k in 0..xd.len() {
        if !(xm[k] || ym[k]) {
            xs_kept.push(xd[k]);
            ys_kept.push(yd[k]);
        }
    }

    let fitted =
        <fp_poly::Polynomial as Poly>::fit(&xs_kept, &ys_kept, deg).map_err(ferr_to_pyerr)?;
    let mut coeffs = fitted.coeffs().to_vec();
    coeffs.reverse(); // lowest-first -> highest-first (numpy.polyfit order)
    let arr =
        Array::<f64, Ix1>::from_vec(Ix1::new([coeffs.len()]), coeffs).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// Apply a 1-D `f(a, v, mode)` over the data and over the boolean masks to
/// realise numpy.ma's `_convolve_or_correlate` (`numpy/ma/core.py:8337`).
///
/// With `propagate_mask=True` a result element is masked if *any* masked input
/// element contributes to it: the mask is the union of `f(mask(a), ones(v))`
/// and `f(ones(a), mask(v))` (a position is masked iff that convolution/
/// correlation sum is non-zero). The data is `f(data(a), data(v))` — masked
/// output positions are overwritten by the mask, so the stored data there is
/// immaterial.
///
/// With `propagate_mask=False` a result element is masked only when *no* valid
/// pair contributes: `mask = ~f(~mask(a), ~mask(v))` and the data is computed
/// from the zero-filled inputs.
fn masked_convolve_or_correlate<F>(
    a_data: &[f64],
    a_mask: &[bool],
    v_data: &[f64],
    v_mask: &[bool],
    propagate_mask: bool,
    f: F,
) -> PyResult<(Vec<f64>, Vec<bool>)>
where
    F: Fn(&[f64], &[f64]) -> PyResult<Vec<f64>>,
{
    let ones_a = vec![1.0f64; a_data.len()];
    let ones_v = vec![1.0f64; v_data.len()];

    if propagate_mask {
        let am: Vec<f64> = a_mask.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        let vm: Vec<f64> = v_mask.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        let left = f(&am, &ones_v)?;
        let right = f(&ones_a, &vm)?;
        let mask: Vec<bool> = left
            .iter()
            .zip(right.iter())
            .map(|(&l, &r)| l != 0.0 || r != 0.0)
            .collect();
        let data = f(a_data, v_data)?;
        Ok((data, mask))
    } else {
        let not_am: Vec<f64> = a_mask.iter().map(|&b| if b { 0.0 } else { 1.0 }).collect();
        let not_vm: Vec<f64> = v_mask.iter().map(|&b| if b { 0.0 } else { 1.0 }).collect();
        let overlap = f(&not_am, &not_vm)?;
        let mask: Vec<bool> = overlap.iter().map(|&v| v == 0.0).collect();
        let a_filled: Vec<f64> = a_data
            .iter()
            .zip(a_mask.iter())
            .map(|(&d, &m)| if m { 0.0 } else { d })
            .collect();
        let v_filled: Vec<f64> = v_data
            .iter()
            .zip(v_mask.iter())
            .map(|(&d, &m)| if m { 0.0 } else { d })
            .collect();
        let data = f(&a_filled, &v_filled)?;
        Ok((data, mask))
    }
}

/// Parse a convolution/correlation `mode` string into a closure-friendly tag.
fn parse_mode(mode: &str) -> PyResult<u8> {
    match mode {
        "full" => Ok(0),
        "same" => Ok(1),
        "valid" => Ok(2),
        other => Err(PyValueError::new_err(format!(
            "mode must be one of 'full', 'same', 'valid', got '{other}'"
        ))),
    }
}

/// Extract the flat 1-D data + mask of an operand, erroring on non-1-D input.
fn ma_flat_1d<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    who: &str,
) -> PyResult<(Vec<f64>, Vec<bool>)> {
    let (d, m, s) = ma_parts(&coerce_to_ma_npmask(py, obj)?)?;
    if s.len() > 1 {
        return Err(PyValueError::new_err(format!(
            "ma.{who}: inputs must be 1-D"
        )));
    }
    Ok((d, m))
}

/// Build a 1-D `PyMaskedArray` from a flat data + mask pair.
fn ma_from_flat(data: Vec<f64>, mask: Vec<bool>) -> PyResult<PyMaskedArray> {
    let n = data.len();
    let data_arr = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[n]), data).map_err(ferr_to_pyerr)?;
    let mask_arr = Array::<bool, IxDyn>::from_vec(IxDyn::new(&[n]), mask).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(
        RustMa::new(data_arr, mask_arr).map_err(ferr_to_pyerr)?,
    ))
}

// ===========================================================================
// numpy.ma manipulation / elementwise / aliases / fill-value helpers
// (refs #835 #818): the COMPOSABLE residual surface — each function applies
// the matching `numpy.<name>` (or top-level ferray) op to the masked array's
// DATA and BOOL MASK independently, then rebuilds the masked array, mirroring
// numpy.ma's own `_frommethod` / mask-propagation wrappers
// (`numpy/ma/core.py`, `numpy/ma/extras.py`). Every contract was verified live
// against numpy 2.4.5; the pytest oracle (tests/test_expansion_ma_batch2.py)
// constructs every expected value from `numpy.ma.*` directly (R-CHAR-3).
// ===========================================================================

/// Apply `numpy.<func>(data, *args)` and `numpy.<func>(mask, *args)`
/// independently, then rebuild a `PyMaskedArray`.
///
/// This is the generalization of the existing `diag` binding: numpy.ma's
/// shape-only manipulation functions (`atleast_*`, `column_stack`, `dstack`,
/// `diagonal`, `diagflat`, `swapaxes`, `resize`, `compress`) act on the data
/// and the mask with the *identical* structural transform, so applying the
/// same `numpy.<func>` to each buffer and recombining reproduces numpy.ma's
/// observable `(data, mask)` exactly. The mask buffer is carried as `bool`,
/// so `numpy.<func>` preserves the all-False (`nomask`) case as all-False.
fn np_data_mask_op<'py>(
    py: Python<'py>,
    func: &str,
    m: &RustMa<f64, IxDyn>,
    extra: &Bound<'py, pyo3::types::PyTuple>,
) -> PyResult<PyMaskedArray> {
    use pyo3::types::PyTuple;
    let np = py.import("numpy")?;
    let data: ArrayD<f64> = fma::getdata(m).map_err(ferr_to_pyerr)?;
    let mask: ArrayD<bool> = fma::getmaskarray(m).map_err(ferr_to_pyerr)?;
    let data_py = data.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    let mask_py = mask.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    let func_obj = np.getattr(func)?;
    // Build `(array, *extra)` argument tuples for the data and mask calls.
    let mut data_items: Vec<Bound<'py, PyAny>> = vec![data_py];
    data_items.extend(extra.iter());
    let mut mask_items: Vec<Bound<'py, PyAny>> = vec![mask_py];
    mask_items.extend(extra.iter());
    let data_args = PyTuple::new(py, data_items)?;
    let mask_args = PyTuple::new(py, mask_items)?;
    let data_res = coerce_dtype(py, &func_obj.call1(data_args)?, "float64")?;
    let mask_res = coerce_dtype(py, &func_obj.call1(mask_args)?, "bool")?;
    let data_fa: ArrayD<f64> = data_res
        .extract::<PyReadonlyArrayDyn<f64>>()?
        .as_ferray()
        .map_err(ferr_to_pyerr)?;
    let mask_fa: ArrayD<bool> = mask_res
        .extract::<PyReadonlyArrayDyn<bool>>()?
        .as_ferray()
        .map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(
        RustMa::new(data_fa, mask_fa).map_err(ferr_to_pyerr)?,
    ))
}

/// `numpy.ma.atleast_1d(a)` — view `a` with at least one dimension, mask
/// reshaped identically (`numpy/ma/extras.py` `atleast_1d = _fromnxfunction_*`).
#[pyfunction]
pub fn atleast_1d<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    let extra = pyo3::types::PyTuple::empty(py);
    np_data_mask_op(py, "atleast_1d", &coerce_to_ma(py, a)?, &extra)
}

/// `numpy.ma.atleast_2d(a)` — at least two dimensions, mask reshaped alongside.
#[pyfunction]
pub fn atleast_2d<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    let extra = pyo3::types::PyTuple::empty(py);
    np_data_mask_op(py, "atleast_2d", &coerce_to_ma(py, a)?, &extra)
}

/// `numpy.ma.atleast_3d(a)` — at least three dimensions, mask reshaped alongside.
#[pyfunction]
pub fn atleast_3d<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    let extra = pyo3::types::PyTuple::empty(py);
    np_data_mask_op(py, "atleast_3d", &coerce_to_ma(py, a)?, &extra)
}

/// `numpy.ma.diagflat(a)` — 2-D array with the flattened `a` on its diagonal,
/// mask placed on the same diagonal (off-diagonal fills are unmasked zeros).
#[pyfunction]
#[pyo3(signature = (a, k = 0))]
pub fn diagflat<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, k: i64) -> PyResult<PyMaskedArray> {
    let extra = pyo3::types::PyTuple::new(py, [k])?;
    np_data_mask_op(py, "diagflat", &coerce_to_ma(py, a)?, &extra)
}

/// `numpy.ma.diagonal(a, offset=0, ...)` — the requested diagonal of `a`, mask
/// carried element-wise (`numpy/ma/core.py` `diagonal = _frommethod`).
#[pyfunction]
#[pyo3(signature = (a, offset = 0))]
pub fn diagonal<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    offset: i64,
) -> PyResult<PyMaskedArray> {
    let extra = pyo3::types::PyTuple::new(py, [offset])?;
    np_data_mask_op(py, "diagonal", &coerce_to_ma(py, a)?, &extra)
}

/// `numpy.ma.swapaxes(a, axis1, axis2)` — swap two axes of data + mask.
#[pyfunction]
pub fn swapaxes<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis1: isize,
    axis2: isize,
) -> PyResult<PyMaskedArray> {
    let extra = pyo3::types::PyTuple::new(py, [axis1, axis2])?;
    np_data_mask_op(py, "swapaxes", &coerce_to_ma(py, a)?, &extra)
}

/// `numpy.ma.resize(a, new_shape)` — resize data + mask to `new_shape`
/// (repeating data as numpy does), the mask resized in lockstep.
#[pyfunction]
pub fn resize<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    new_shape: &Bound<'py, PyAny>,
) -> PyResult<PyMaskedArray> {
    let shape = crate::conv::extract_shape(new_shape)?;
    let shape_t = pyo3::types::PyTuple::new(py, shape)?;
    let extra = pyo3::types::PyTuple::new(py, [shape_t])?;
    np_data_mask_op(py, "resize", &coerce_to_ma(py, a)?, &extra)
}

/// `numpy.ma.empty_like(a)` — an uninitialized masked array shaped like `a`,
/// PRESERVING `a`'s mask (`numpy/ma/core.py` `empty_like` carries the input
/// mask). numpy returns arbitrary data; ferray returns the deterministic zero
/// buffer, a valid `empty_like` for the data (R-DEV-7).
#[pyfunction]
pub fn empty_like<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    like_const(py, a, 0.0)
}

/// `numpy.ma.ones_like(a)` / `zeros_like(a)` share the same shape-only path;
/// build a constant-filled masked array shaped like `a`, PRESERVING `a`'s mask
/// (numpy.ma's `*_like` functions carry the input mask through).
fn like_const<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, fill: f64) -> PyResult<PyMaskedArray> {
    let m = coerce_to_ma(py, a)?;
    let data: ArrayD<f64> = fma::getdata(&m).map_err(ferr_to_pyerr)?;
    let mask: ArrayD<bool> = fma::getmaskarray(&m).map_err(ferr_to_pyerr)?;
    let n = data.size();
    let shape = data.shape().to_vec();
    let buf = ArrayD::<f64>::from_vec(IxDyn::new(&shape), vec![fill; n]).map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(
        RustMa::new(buf, mask).map_err(ferr_to_pyerr)?,
    ))
}

/// `numpy.ma.ones_like(a)` — masked array of ones shaped like `a` (nomask).
#[pyfunction]
pub fn ones_like<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    like_const(py, a, 1.0)
}

/// `numpy.ma.zeros_like(a)` — masked array of zeros shaped like `a` (nomask).
#[pyfunction]
pub fn zeros_like<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    like_const(py, a, 0.0)
}

/// `numpy.ma.column_stack(tup)` — stack 1-D/2-D masked arrays as columns,
/// stacking their masks alongside (`numpy/ma/extras.py` `column_stack`).
#[pyfunction]
pub fn column_stack<'py>(py: Python<'py>, tup: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    stack_family(py, "column_stack", tup)
}

/// `numpy.ma.dstack(tup)` — stack masked arrays depth-wise (3rd axis), masks
/// stacked alongside (`numpy/ma/extras.py` `dstack`).
#[pyfunction]
pub fn dstack<'py>(py: Python<'py>, tup: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    stack_family(py, "dstack", tup)
}

/// `numpy.ma.vstack(tup)` / `row_stack` — stack masked arrays row-wise.
#[pyfunction]
pub fn vstack<'py>(py: Python<'py>, tup: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    stack_family(py, "vstack", tup)
}

/// `numpy.ma.hstack(tup)` — stack masked arrays column-wise (horizontally).
#[pyfunction]
pub fn hstack<'py>(py: Python<'py>, tup: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    stack_family(py, "hstack", tup)
}

/// `numpy.ma.stack(tup)` — join masked arrays along a new leading axis.
#[pyfunction]
pub fn stack<'py>(py: Python<'py>, tup: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    stack_family(py, "stack", tup)
}

/// Apply a numpy *sequence* stack op (`column_stack`/`dstack`) to a list of
/// masked arrays: collect each operand's data and mask, call `numpy.<func>` on
/// the data list and the mask list independently, recombine.
fn stack_family<'py>(
    py: Python<'py>,
    func: &str,
    tup: &Bound<'py, PyAny>,
) -> PyResult<PyMaskedArray> {
    let seq: Vec<Bound<'py, PyAny>> = tup.extract()?;
    if seq.is_empty() {
        return Err(PyValueError::new_err(format!(
            "ma.{func}: need at least one array"
        )));
    }
    let np = py.import("numpy")?;
    let mut datas: Vec<Bound<'py, PyAny>> = Vec::with_capacity(seq.len());
    let mut masks: Vec<Bound<'py, PyAny>> = Vec::with_capacity(seq.len());
    for item in &seq {
        let m = coerce_to_ma(py, item)?;
        let d: ArrayD<f64> = fma::getdata(&m).map_err(ferr_to_pyerr)?;
        let k: ArrayD<bool> = fma::getmaskarray(&m).map_err(ferr_to_pyerr)?;
        datas.push(d.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any());
        masks.push(k.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any());
    }
    let data_list = pyo3::types::PyList::new(py, datas)?;
    let mask_list = pyo3::types::PyList::new(py, masks)?;
    let data_res = coerce_dtype(py, &np.getattr(func)?.call1((data_list,))?, "float64")?;
    let mask_res = coerce_dtype(py, &np.getattr(func)?.call1((mask_list,))?, "bool")?;
    let data_fa: ArrayD<f64> = data_res
        .extract::<PyReadonlyArrayDyn<f64>>()?
        .as_ferray()
        .map_err(ferr_to_pyerr)?;
    let mask_fa: ArrayD<bool> = mask_res
        .extract::<PyReadonlyArrayDyn<bool>>()?
        .as_ferray()
        .map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(
        RustMa::new(data_fa, mask_fa).map_err(ferr_to_pyerr)?,
    ))
}

/// `numpy.ma.append(a, b, axis=None)` — append `b` to `a`, concatenating the
/// masks alongside the data (`numpy/ma/core.py:8470`). `axis=None` flattens.
#[pyfunction]
#[pyo3(signature = (a, b, axis = None))]
pub fn append<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<PyMaskedArray> {
    let np = py.import("numpy")?;
    let ma = coerce_to_ma(py, a)?;
    let mb = coerce_to_ma(py, b)?;
    let (ad, am) = (
        fma::getdata(&ma).map_err(ferr_to_pyerr)?,
        fma::getmaskarray(&ma).map_err(ferr_to_pyerr)?,
    );
    let (bd, bm) = (
        fma::getdata(&mb).map_err(ferr_to_pyerr)?,
        fma::getmaskarray(&mb).map_err(ferr_to_pyerr)?,
    );
    let ad_py = ad.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    let am_py = am.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    let bd_py = bd.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    let bm_py = bm.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    let (data_res, mask_res) = match axis {
        None => (
            np.call_method1("append", (ad_py, bd_py))?,
            np.call_method1("append", (am_py, bm_py))?,
        ),
        Some(ax) => (
            np.call_method1("append", (ad_py, bd_py, ax))?,
            np.call_method1("append", (am_py, bm_py, ax))?,
        ),
    };
    let data_res = coerce_dtype(py, &data_res, "float64")?;
    let mask_res = coerce_dtype(py, &mask_res, "bool")?;
    let data_fa: ArrayD<f64> = data_res
        .extract::<PyReadonlyArrayDyn<f64>>()?
        .as_ferray()
        .map_err(ferr_to_pyerr)?;
    let mask_fa: ArrayD<bool> = mask_res
        .extract::<PyReadonlyArrayDyn<bool>>()?
        .as_ferray()
        .map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(
        RustMa::new(data_fa, mask_fa).map_err(ferr_to_pyerr)?,
    ))
}

/// `numpy.ma.compress(condition, a, axis=None)` — select entries of `a` where
/// `condition` is true, the mask compressed alongside the data
/// (`numpy/ma/core.py` `compress`). Mirrors `numpy.compress` on data + mask.
#[pyfunction]
#[pyo3(signature = (condition, a, axis = None))]
pub fn compress<'py>(
    py: Python<'py>,
    condition: &Bound<'py, PyAny>,
    a: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<PyMaskedArray> {
    let np = py.import("numpy")?;
    let m = coerce_to_ma(py, a)?;
    let data: ArrayD<f64> = fma::getdata(&m).map_err(ferr_to_pyerr)?;
    let mask: ArrayD<bool> = fma::getmaskarray(&m).map_err(ferr_to_pyerr)?;
    let data_py = data.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    let mask_py = mask.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    let (data_res, mask_res) = match axis {
        None => (
            np.call_method1("compress", (condition, &data_py))?,
            np.call_method1("compress", (condition, &mask_py))?,
        ),
        Some(ax) => (
            np.call_method1("compress", (condition, &data_py, ax))?,
            np.call_method1("compress", (condition, &mask_py, ax))?,
        ),
    };
    let data_res = coerce_dtype(py, &data_res, "float64")?;
    let mask_res = coerce_dtype(py, &mask_res, "bool")?;
    let data_fa: ArrayD<f64> = data_res
        .extract::<PyReadonlyArrayDyn<f64>>()?
        .as_ferray()
        .map_err(ferr_to_pyerr)?;
    let mask_fa: ArrayD<bool> = mask_res
        .extract::<PyReadonlyArrayDyn<bool>>()?
        .as_ferray()
        .map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(
        RustMa::new(data_fa, mask_fa).map_err(ferr_to_pyerr)?,
    ))
}

// ---------------------------------------------------------------------------
// Masked cumulative ops (mask → reduction identity, then accumulate)
// ---------------------------------------------------------------------------

/// `numpy.ma.cumsum(a)` / `cumprod(a)`: numpy.ma fills the masked positions
/// with the reduction identity (0 for sum, 1 for prod) BEFORE the cumulative
/// op, runs `numpy.<func>` on that filled data, and keeps the original mask
/// (`numpy/ma/core.py:5294` `result = self.filled(0).cumsum(...)`). The result
/// dtype is numpy's accumulator (`ReduceAcc`: int8→int64, uint8→uint64,
/// bool→int64, float unchanged), NOT a float64 collapse (#858, R-CODE-4): the
/// masked data is filled with the identity in its NATIVE dtype and handed to
/// `numpy.<func>`, which yields the accumulator dtype; we rebuild a
/// dtype-preserving `DynMa` from that result and re-apply the original mask.
fn ma_cumulative<'py>(
    py: Python<'py>,
    func: &str,
    product: bool,
    a: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<PyMaskedArray> {
    let np = py.import("numpy")?;
    let dynma = coerce_to_dynma(py, a)?;
    let mask_fa = dynma.mask_bits()?;
    // Fill masked slots with the reduction identity in the NATIVE dtype, egress
    // to a numpy ndarray, then run `numpy.<func>` — numpy promotes the dtype to
    // its reduction accumulator exactly as `filled(0).cumsum()` does.
    let filled_py: Bound<'py, PyAny> = match_ma!(&dynma, m, T => {
        let ident: T = if product { <T as MaIdentity>::ONE } else { <T as MaIdentity>::ZERO };
        let filled: ArrayD<T> = m.filled(ident).map_err(ferr_to_pyerr)?;
        filled.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    });
    let mask_py = mask_fa
        .clone()
        .into_pyarray(py)
        .map_err(ferr_to_pyerr)?
        .into_any();
    let (data_res, mask_res) = match axis {
        None => (
            np.call_method1(func, (filled_py,))?,
            // axis=None flattens the data; flatten the mask to match.
            np.call_method1("ravel", (mask_py,))?,
        ),
        Some(ax) => (np.call_method1(func, (filled_py, ax))?, mask_py),
    };
    // Rebuild a dtype-preserving DynMa from numpy's accumulator-typed result.
    let mask_res = coerce_dtype(py, &mask_res, "bool")?;
    let result_mask: ArrayD<bool> = mask_res
        .extract::<PyReadonlyArrayDyn<bool>>()?
        .as_ferray()
        .map_err(ferr_to_pyerr)?;
    let inner = build_dynma(&data_res, Some(result_mask))?;
    Ok(PyMaskedArray::from_dynma(inner))
}

/// `numpy.ma.cumsum(a, axis=None)` — cumulative sum, masked slots contribute 0
/// and stay masked in the output.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn cumsum<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<PyMaskedArray> {
    ma_cumulative(py, "cumsum", false, a, axis)
}

/// `numpy.ma.cumprod(a, axis=None)` — cumulative product, masked slots
/// contribute 1 and stay masked in the output.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn cumprod<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<PyMaskedArray> {
    ma_cumulative(py, "cumprod", true, a, axis)
}

// ---------------------------------------------------------------------------
// Masked rounding (numpy.round on data, mask preserved)
// ---------------------------------------------------------------------------

/// `numpy.ma.round(a, decimals=0)` / `round_` — round unmasked data to
/// `decimals` places, mask preserved (`numpy/ma/core.py` `round`).
#[pyfunction]
#[pyo3(signature = (a, decimals = 0))]
pub fn round<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    decimals: i64,
) -> PyResult<PyMaskedArray> {
    let np = py.import("numpy")?;
    let m = coerce_to_ma(py, a)?;
    let data: ArrayD<f64> = fma::getdata(&m).map_err(ferr_to_pyerr)?;
    let mask: ArrayD<bool> = fma::getmaskarray(&m).map_err(ferr_to_pyerr)?;
    let data_py = data.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    let rounded = coerce_dtype(
        py,
        &np.call_method1("round", (data_py, decimals))?,
        "float64",
    )?;
    let data_fa: ArrayD<f64> = rounded
        .extract::<PyReadonlyArrayDyn<f64>>()?
        .as_ferray()
        .map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(
        RustMa::new(data_fa, mask).map_err(ferr_to_pyerr)?,
    ))
}

// ---------------------------------------------------------------------------
// Masked comparisons (data = numpy comparison, mask = OR of operand masks)
// ---------------------------------------------------------------------------

/// Apply a numpy comparison ufunc over two masked operands: the result data is
/// `numpy.<func>(adata, bdata)`, the result mask is the elementwise OR of the
/// (broadcast) operand masks — matching numpy.ma's comparison wrappers
/// (`numpy/ma/core.py` `_extrema_operation` / `_DomainedBinaryOperation`,
/// "the result is masked wherever either operand is masked").
fn ma_compare<'py>(
    py: Python<'py>,
    func: &str,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<PyMaskedArray> {
    let np = py.import("numpy")?;
    let ma = coerce_to_ma(py, a)?;
    let mb = coerce_to_ma(py, b)?;
    let ad = fma::getdata(&ma).map_err(ferr_to_pyerr)?;
    let am = fma::getmaskarray(&ma).map_err(ferr_to_pyerr)?;
    let bd = fma::getdata(&mb).map_err(ferr_to_pyerr)?;
    let bm = fma::getmaskarray(&mb).map_err(ferr_to_pyerr)?;
    let ad_py = ad.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    let am_py = am.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    let bd_py = bd.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    let bm_py = bm.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    // Comparison result data (bool) → stored as float64 (0.0/1.0) in the
    // f64-only masked wrapper, matching numpy.ma.<cmp>'s boolean data values.
    let data_res = coerce_dtype(py, &np.getattr(func)?.call1((ad_py, bd_py))?, "float64")?;
    let mask_res = coerce_dtype(py, &np.call_method1("logical_or", (am_py, bm_py))?, "bool")?;
    let data_fa: ArrayD<f64> = data_res
        .extract::<PyReadonlyArrayDyn<f64>>()?
        .as_ferray()
        .map_err(ferr_to_pyerr)?;
    let mask_fa: ArrayD<bool> = mask_res
        .extract::<PyReadonlyArrayDyn<bool>>()?
        .as_ferray()
        .map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(
        RustMa::new(data_fa, mask_fa).map_err(ferr_to_pyerr)?,
    ))
}

macro_rules! ma_cmp {
    ($name:ident, $np:literal, $doc:literal) => {
        #[doc = $doc]
        #[pyfunction]
        pub fn $name<'py>(
            py: Python<'py>,
            a: &Bound<'py, PyAny>,
            b: &Bound<'py, PyAny>,
        ) -> PyResult<PyMaskedArray> {
            ma_compare(py, $np, a, b)
        }
    };
}

ma_cmp!(
    equal,
    "equal",
    "`numpy.ma.equal(a, b)` — masked elementwise `==`."
);
ma_cmp!(
    not_equal,
    "not_equal",
    "`numpy.ma.not_equal(a, b)` — masked elementwise `!=`."
);
ma_cmp!(
    greater,
    "greater",
    "`numpy.ma.greater(a, b)` — masked elementwise `>`."
);
ma_cmp!(
    greater_equal,
    "greater_equal",
    "`numpy.ma.greater_equal(a, b)` — masked elementwise `>=`."
);
ma_cmp!(
    less,
    "less",
    "`numpy.ma.less(a, b)` — masked elementwise `<`."
);
ma_cmp!(
    less_equal,
    "less_equal",
    "`numpy.ma.less_equal(a, b)` — masked elementwise `<=`."
);

// ---------------------------------------------------------------------------
// Masked angle (numpy.angle on data, mask preserved)
// ---------------------------------------------------------------------------

/// `numpy.ma.angle(z, deg=False)` — angle of the (real, f64) masked data, mask
/// preserved. The f64-only wrapper carries real data, so this is `numpy.angle`
/// of the real buffer (0 for positives, π for negatives); genuine complex
/// masked angle needs the complex `PyMaskedArray` wrapper tracked under #835.
#[pyfunction]
#[pyo3(signature = (z, deg = false))]
pub fn angle<'py>(py: Python<'py>, z: &Bound<'py, PyAny>, deg: bool) -> PyResult<PyMaskedArray> {
    let np = py.import("numpy")?;
    let m = coerce_to_ma(py, z)?;
    let data: ArrayD<f64> = fma::getdata(&m).map_err(ferr_to_pyerr)?;
    let mask: ArrayD<bool> = fma::getmaskarray(&m).map_err(ferr_to_pyerr)?;
    let data_py = data.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("deg", deg)?;
    let res = coerce_dtype(
        py,
        &np.getattr("angle")?.call((data_py,), Some(&kwargs))?,
        "float64",
    )?;
    let data_fa: ArrayD<f64> = res
        .extract::<PyReadonlyArrayDyn<f64>>()?
        .as_ferray()
        .map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(
        RustMa::new(data_fa, mask).map_err(ferr_to_pyerr)?,
    ))
}

// ---------------------------------------------------------------------------
// Reduction free-functions + aliases mirroring the MaskedArray methods
// ---------------------------------------------------------------------------

/// `numpy.ma.max(a)` / `amax(a)` — maximum unmasked element. All-masked
/// reduces to the `numpy.ma.masked` singleton.
#[pyfunction]
pub fn max<'py>(py: Python<'py>, a: &PyMaskedArray) -> PyResult<Bound<'py, PyAny>> {
    a.max(py)
}

/// `numpy.ma.min(a)` / `amin(a)` — minimum unmasked element.
#[pyfunction]
pub fn min<'py>(py: Python<'py>, a: &PyMaskedArray) -> PyResult<Bound<'py, PyAny>> {
    a.min(py)
}

/// `numpy.ma.sum(a)` — sum of unmasked elements.
#[pyfunction]
pub fn sum<'py>(py: Python<'py>, a: &PyMaskedArray) -> PyResult<Bound<'py, PyAny>> {
    a.sum(py)
}

/// `numpy.ma.mean(a)` — mean of unmasked elements.
#[pyfunction]
pub fn mean<'py>(py: Python<'py>, a: &PyMaskedArray) -> PyResult<Bound<'py, PyAny>> {
    a.mean(py)
}

/// `numpy.ma.std(a)` — standard deviation of unmasked elements.
#[pyfunction]
pub fn std<'py>(py: Python<'py>, a: &PyMaskedArray) -> PyResult<Bound<'py, PyAny>> {
    a.std(py)
}

/// `numpy.ma.var(a)` — variance of unmasked elements.
#[pyfunction]
pub fn var<'py>(py: Python<'py>, a: &PyMaskedArray) -> PyResult<Bound<'py, PyAny>> {
    a.var(py)
}

/// `numpy.ma.anomalies(a)` — alias of [`anom`] (deviation from the unmasked
/// mean; `numpy/ma/core.py`: `anomalies = anom`).
#[pyfunction]
pub fn anomalies(a: &PyMaskedArray) -> PyResult<PyMaskedArray> {
    anom(a)
}

// ---------------------------------------------------------------------------
// Predicates / fill-value helpers
// ---------------------------------------------------------------------------

/// `numpy.ma.allequal(a, b, fill_value=True)` — true iff `a` and `b` have
/// equal UNMASKED elements (`numpy/ma/core.py` `allequal`). With the default
/// `fill_value=True`, masked positions in either operand are treated as equal.
#[pyfunction]
#[pyo3(signature = (a, b, fill_value = true))]
pub fn allequal<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    fill_value: bool,
) -> PyResult<bool> {
    let ma = coerce_to_ma(py, a)?;
    let mb = coerce_to_ma(py, b)?;
    let ad = fma::getdata(&ma).map_err(ferr_to_pyerr)?;
    let am = fma::getmaskarray(&ma).map_err(ferr_to_pyerr)?;
    let bd = fma::getdata(&mb).map_err(ferr_to_pyerr)?;
    let bm = fma::getmaskarray(&mb).map_err(ferr_to_pyerr)?;
    if ad.shape() != bd.shape() {
        return Ok(false);
    }
    let zipped = ad.iter().zip(am.iter()).zip(bd.iter().zip(bm.iter()));
    for ((&av, &amk), (&bv, &bmk)) in zipped {
        let either_masked = amk || bmk;
        if either_masked {
            // With fill_value=True a masked pair counts as equal; with
            // fill_value=False any masked position makes the arrays unequal.
            if !fill_value {
                return Ok(false);
            }
        } else if av != bv {
            return Ok(false);
        }
    }
    Ok(true)
}

/// `numpy.ma.allclose(a, b, masked_equal=True, rtol=1e-5, atol=1e-8)` — true
/// iff the UNMASKED elements of `a` and `b` are close within tolerance
/// (`numpy/ma/core.py` `allclose`). Masked positions are ignored when
/// `masked_equal` is true (the default).
#[pyfunction]
#[pyo3(signature = (a, b, masked_equal = true, rtol = 1e-5, atol = 1e-8))]
pub fn allclose<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    masked_equal: bool,
    rtol: f64,
    atol: f64,
) -> PyResult<bool> {
    let ma = coerce_to_ma(py, a)?;
    let mb = coerce_to_ma(py, b)?;
    let ad = fma::getdata(&ma).map_err(ferr_to_pyerr)?;
    let am = fma::getmaskarray(&ma).map_err(ferr_to_pyerr)?;
    let bd = fma::getdata(&mb).map_err(ferr_to_pyerr)?;
    let bm = fma::getmaskarray(&mb).map_err(ferr_to_pyerr)?;
    if ad.shape() != bd.shape() {
        return Ok(false);
    }
    let zipped = ad.iter().zip(am.iter()).zip(bd.iter().zip(bm.iter()));
    for ((&av, &amk), (&bv, &bmk)) in zipped {
        let either_masked = amk || bmk;
        if either_masked {
            if !masked_equal {
                return Ok(false);
            }
        } else {
            // numpy.allclose criterion: |a - b| <= atol + rtol*|b|.
            let close = (av - bv).abs() <= atol + rtol * bv.abs();
            if !close {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// `numpy.ma.maximum_fill_value(obj)` — the value used to fill masked slots so
/// they never win a maximum reduction: `-inf` for float64
/// (`numpy/ma/core.py` `maximum_fill_value`).
#[pyfunction]
pub fn maximum_fill_value(_obj: &Bound<'_, PyAny>) -> f64 {
    f64::NEG_INFINITY
}

/// `numpy.ma.minimum_fill_value(obj)` — the value used to fill masked slots so
/// they never win a minimum reduction: `+inf` for float64
/// (`numpy/ma/core.py` `minimum_fill_value`).
#[pyfunction]
pub fn minimum_fill_value(_obj: &Bound<'_, PyAny>) -> f64 {
    f64::INFINITY
}

/// `numpy.ma.common_fill_value(a, b)` — the shared fill value of `a` and `b`
/// if they agree, else `None` (`numpy/ma/core.py` `common_fill_value`).
#[pyfunction]
pub fn common_fill_value<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let fa = coerce_to_ma(py, a)?.fill_value();
    let fb = coerce_to_ma(py, b)?.fill_value();
    if fa == fb {
        Ok(fa.into_pyobject(py)?.into_any())
    } else {
        Ok(py.None().into_bound(py))
    }
}

// ---------------------------------------------------------------------------
// Shape inspectors (numpy.ma.ndim / shape / size — free-function form)
// ---------------------------------------------------------------------------

/// `numpy.ma.ndim(a)` — number of dimensions of `a`.
#[pyfunction]
pub fn ndim<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<usize> {
    Ok(coerce_to_ma(py, a)?.ndim())
}

/// `numpy.ma.shape(a)` — shape tuple of `a`.
#[pyfunction]
pub fn shape<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let m = coerce_to_ma(py, a)?;
    let s: Vec<usize> = m.shape().to_vec();
    Ok(pyo3::types::PyTuple::new(py, s)?.into_any())
}

/// `numpy.ma.size(a)` — total number of elements of `a`.
#[pyfunction]
pub fn size<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<usize> {
    Ok(coerce_to_ma(py, a)?.size())
}

// ---------------------------------------------------------------------------
// Masked logical ops + bit shifts (data = numpy op, mask = OR of masks)
// ---------------------------------------------------------------------------

macro_rules! ma_binop_orm {
    ($name:ident, $np:literal, $doc:literal) => {
        #[doc = $doc]
        #[pyfunction]
        pub fn $name<'py>(
            py: Python<'py>,
            a: &Bound<'py, PyAny>,
            b: &Bound<'py, PyAny>,
        ) -> PyResult<PyMaskedArray> {
            ma_compare(py, $np, a, b)
        }
    };
}

ma_binop_orm!(
    logical_and,
    "logical_and",
    "`numpy.ma.logical_and(a, b)` — masked elementwise logical AND."
);
ma_binop_orm!(
    logical_or,
    "logical_or",
    "`numpy.ma.logical_or(a, b)` — masked elementwise logical OR."
);
ma_binop_orm!(
    logical_xor,
    "logical_xor",
    "`numpy.ma.logical_xor(a, b)` — masked elementwise logical XOR."
);

/// `numpy.ma.logical_not(a)` — masked elementwise logical NOT, mask preserved.
#[pyfunction]
pub fn logical_not<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<PyMaskedArray> {
    let np = py.import("numpy")?;
    let m = coerce_to_ma(py, a)?;
    let data: ArrayD<f64> = fma::getdata(&m).map_err(ferr_to_pyerr)?;
    let mask: ArrayD<bool> = fma::getmaskarray(&m).map_err(ferr_to_pyerr)?;
    let data_py = data.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    let res = coerce_dtype(py, &np.call_method1("logical_not", (data_py,))?, "float64")?;
    let data_fa: ArrayD<f64> = res
        .extract::<PyReadonlyArrayDyn<f64>>()?
        .as_ferray()
        .map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(
        RustMa::new(data_fa, mask).map_err(ferr_to_pyerr)?,
    ))
}

// ---------------------------------------------------------------------------
// Masked inner / outer products (data op + mask propagation)
// ---------------------------------------------------------------------------

/// `numpy.ma.outer(a, b)` / `outerproduct` — outer product of the flattened
/// operands; the result is masked wherever either contributing operand element
/// is masked (`numpy/ma/extras.py` `outer`).
#[pyfunction]
pub fn outer<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<PyMaskedArray> {
    let np = py.import("numpy")?;
    let ma = coerce_to_ma(py, a)?;
    let mb = coerce_to_ma(py, b)?;
    let ad = fma::getdata(&ma).map_err(ferr_to_pyerr)?;
    let am = fma::getmaskarray(&ma).map_err(ferr_to_pyerr)?;
    let bd = fma::getdata(&mb).map_err(ferr_to_pyerr)?;
    let bm = fma::getmaskarray(&mb).map_err(ferr_to_pyerr)?;
    let ad_py = ad.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    let am_py = am.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    let bd_py = bd.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    let bm_py = bm.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    let data_res = coerce_dtype(py, &np.call_method1("outer", (ad_py, bd_py))?, "float64")?;
    // mask = outer(am, ones) OR outer(ones, bm): masked iff either source slot
    // is masked. numpy.ma builds this via `mask = logical_or.outer(am, bm)`.
    let mask_res = coerce_dtype(
        py,
        &np.getattr("logical_or")?
            .getattr("outer")?
            .call1((am_py, bm_py))?,
        "bool",
    )?;
    let data_fa: ArrayD<f64> = data_res
        .extract::<PyReadonlyArrayDyn<f64>>()?
        .as_ferray()
        .map_err(ferr_to_pyerr)?;
    let mask_fa: ArrayD<bool> = mask_res
        .extract::<PyReadonlyArrayDyn<bool>>()?
        .as_ferray()
        .map_err(ferr_to_pyerr)?;
    Ok(PyMaskedArray::from_inner(
        RustMa::new(data_fa, mask_fa).map_err(ferr_to_pyerr)?,
    ))
}

/// `numpy.ma.inner(a, b)` / `innerproduct` — inner product over the unmasked
/// elements of the flattened operands (`numpy/ma/extras.py` `inner`: masked
/// slots are filled with 0 before the dot). Returns a scalar.
#[pyfunction]
pub fn inner<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let ma = coerce_to_ma(py, a)?;
    let mb = coerce_to_ma(py, b)?;
    let ad = fma::getdata(&ma).map_err(ferr_to_pyerr)?;
    let am = fma::getmaskarray(&ma).map_err(ferr_to_pyerr)?;
    let bd = fma::getdata(&mb).map_err(ferr_to_pyerr)?;
    let bm = fma::getmaskarray(&mb).map_err(ferr_to_pyerr)?;
    let mut acc = 0.0f64;
    for (((&av, &amk), &bv), &bmk) in ad.iter().zip(am.iter()).zip(bd.iter()).zip(bm.iter()) {
        let a_eff = if amk { 0.0 } else { av };
        let b_eff = if bmk { 0.0 } else { bv };
        acc += a_eff * b_eff;
    }
    Ok(acc.into_pyobject(py)?.into_any())
}

/// `numpy.ma.convolve(a, v, mode='full', propagate_mask=True)` — masked
/// convolution (`numpy/ma/core.py:8415`). Composes the existing
/// `ferray_ufunc::convolve` over both the data and the boolean masks per
/// numpy's `_convolve_or_correlate` semantics.
#[pyfunction]
#[pyo3(signature = (a, v, mode = "full", propagate_mask = true))]
pub fn convolve<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    v: &Bound<'py, PyAny>,
    mode: &str,
    propagate_mask: bool,
) -> PyResult<PyMaskedArray> {
    let mode_tag = parse_mode(mode)?;
    let (ad, am) = ma_flat_1d(py, a, "convolve")?;
    let (vd, vm) = ma_flat_1d(py, v, "convolve")?;
    let conv = |x: &[f64], y: &[f64]| -> PyResult<Vec<f64>> {
        let xa =
            Array::<f64, Ix1>::from_vec(Ix1::new([x.len()]), x.to_vec()).map_err(ferr_to_pyerr)?;
        let ya =
            Array::<f64, Ix1>::from_vec(Ix1::new([y.len()]), y.to_vec()).map_err(ferr_to_pyerr)?;
        let m = match mode_tag {
            0 => ConvolveMode::Full,
            1 => ConvolveMode::Same,
            _ => ConvolveMode::Valid,
        };
        let r = ufunc_convolve(&xa, &ya, m).map_err(ferr_to_pyerr)?;
        Ok(r.iter().copied().collect())
    };
    let (data, mask) = masked_convolve_or_correlate(&ad, &am, &vd, &vm, propagate_mask, conv)?;
    ma_from_flat(data, mask)
}

/// `numpy.ma.correlate(a, v, mode='valid', propagate_mask=True)` — masked
/// cross-correlation (`numpy/ma/core.py:8356`; note the default mode is
/// `'valid'`, unlike convolve's `'full'`). Composes the existing
/// `ferray_stats::correlate` over both the data and the boolean masks.
#[pyfunction]
#[pyo3(signature = (a, v, mode = "valid", propagate_mask = true))]
pub fn correlate<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    v: &Bound<'py, PyAny>,
    mode: &str,
    propagate_mask: bool,
) -> PyResult<PyMaskedArray> {
    let mode_tag = parse_mode(mode)?;
    let (ad, am) = ma_flat_1d(py, a, "correlate")?;
    let (vd, vm) = ma_flat_1d(py, v, "correlate")?;
    let corr = |x: &[f64], y: &[f64]| -> PyResult<Vec<f64>> {
        let xa =
            Array::<f64, Ix1>::from_vec(Ix1::new([x.len()]), x.to_vec()).map_err(ferr_to_pyerr)?;
        let ya =
            Array::<f64, Ix1>::from_vec(Ix1::new([y.len()]), y.to_vec()).map_err(ferr_to_pyerr)?;
        let m = match mode_tag {
            0 => CorrelateMode::Full,
            1 => CorrelateMode::Same,
            _ => CorrelateMode::Valid,
        };
        let r = stats_correlate(&xa, &ya, m).map_err(ferr_to_pyerr)?;
        Ok(r.iter().copied().collect())
    };
    let (data, mask) = masked_convolve_or_correlate(&ad, &am, &vd, &vm, propagate_mask, corr)?;
    ma_from_flat(data, mask)
}

// ===========================================================================
// numpy.ma composable batch 3 (refs #835 #818): the remaining COMPOSABLE
// numpy.ma surface — array-of-masked constructors (`masked_all`,
// `masked_all_like`, `fromfunction`, `indices`), masked iteration
// (`apply_along_axis`, `apply_over_axes`, `ndenumerate`), masked split
// (`hsplit`), whole-row/col masking (`mask_rows`, `mask_cols` → ferray-ma's
// `ma_mask_rowcols`), multi-axis compress (`compress_nd`), the `bool_`
// dtype-scalar re-export, and `ids`.
//
// Several of these produce a non-float64 MaskedArray (`masked_all((n,),
// dtype=int)`, `indices` int grids, `apply_along_axis` preserving the input
// integer dtype) or a plain ndarray (`compress_nd`). The f64-only
// `PyMaskedArray` wrapper cannot hold those dtypes without a lossy f64 cast
// (R-CODE-4), so those functions COMPOSE over numpy.ma's own algorithm — they
// build a `numpy.ma.MaskedArray` from the input, call the canonical
// `numpy.ma.<func>`, and return numpy's result verbatim, preserving dtype +
// mask exactly. `mask_rows` / `mask_cols` instead delegate to ferray-ma's
// `ma_mask_rowcols` (the #835 algorithm) and return a `PyMaskedArray`.
//
// Every contract was verified live against numpy 2.4.5; the pytest oracle
// (tests/test_expansion_ma_batch3.py) constructs every expected value from
// `numpy.ma.*` directly (R-CHAR-3).
// ===========================================================================

/// Build a `numpy.ma.MaskedArray` (Python object) from any masked input.
///
/// A `ferray.ma.MaskedArray` contributes its data + mask; any other array-like
/// is coerced via [`coerce_to_ma`] (nomask). The result is a genuine numpy.ma
/// array so the canonical `numpy.ma.<func>` algorithms operate on it directly
/// and preserve numpy's `(data, mask)` contract.
fn build_npma<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let m = coerce_to_ma(py, obj)?;
    let data: ArrayD<f64> = fma::getdata(&m).map_err(ferr_to_pyerr)?;
    let mask: ArrayD<bool> = fma::getmaskarray(&m).map_err(ferr_to_pyerr)?;
    let data_py = data.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    let mask_py = mask.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    let np_ma = py.import("numpy.ma")?;
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("mask", mask_py)?;
    np_ma.call_method("array", (data_py,), Some(&kwargs))
}

/// `numpy.ma.masked_all(shape, dtype=float)` — an all-masked masked array of
/// the given shape/dtype (`numpy/ma/extras.py` `def masked_all`). Composed via
/// numpy.ma so any `dtype` (int/float/…) is honoured exactly.
#[pyfunction]
#[pyo3(signature = (shape, dtype = None))]
pub fn masked_all<'py>(
    py: Python<'py>,
    shape: &Bound<'py, PyAny>,
    dtype: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let np_ma = py.import("numpy.ma")?;
    let kwargs = pyo3::types::PyDict::new(py);
    if let Some(dt) = dtype {
        kwargs.set_item("dtype", dt)?;
    }
    np_ma.call_method("masked_all", (shape,), Some(&kwargs))
}

/// `numpy.ma.masked_all_like(arr)` — an all-masked masked array matching the
/// shape and dtype of `arr` (`numpy/ma/extras.py` `def masked_all_like`).
#[pyfunction]
pub fn masked_all_like<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let np_ma = py.import("numpy.ma")?;
    // `masked_all_like` reads `arr.shape`/`arr.dtype`; a `ferray.ma.MaskedArray`
    // is routed through its numpy.ma form so the int/float dtype is preserved.
    let target = if arr.extract::<PyMaskedArray>().is_ok() {
        build_npma(py, arr)?
    } else {
        py.import("numpy")?.call_method1("asarray", (arr,))?
    };
    np_ma.call_method1("masked_all_like", (target,))
}

/// `numpy.ma.fromfunction(function, shape, **kwargs)` — evaluate `function`
/// over the index grid of `shape`, returning a `MaskedArray`
/// (`numpy/ma/core.py` `fromfunction`).
#[pyfunction]
#[pyo3(signature = (function, shape, **kwargs))]
pub fn fromfunction<'py>(
    py: Python<'py>,
    function: &Bound<'py, PyAny>,
    shape: &Bound<'py, PyAny>,
    kwargs: Option<&Bound<'py, pyo3::types::PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let np_ma = py.import("numpy.ma")?;
    np_ma.call_method("fromfunction", (function, shape), kwargs)
}

/// `numpy.ma.indices(dimensions, dtype=int)` — the index grid for `dimensions`
/// as a `MaskedArray` (`numpy/ma/extras.py` `def indices`). The integer index
/// grid is preserved verbatim from numpy.ma.
#[pyfunction]
#[pyo3(signature = (dimensions, dtype = None))]
pub fn indices<'py>(
    py: Python<'py>,
    dimensions: &Bound<'py, PyAny>,
    dtype: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let np_ma = py.import("numpy.ma")?;
    let kwargs = pyo3::types::PyDict::new(py);
    if let Some(dt) = dtype {
        kwargs.set_item("dtype", dt)?;
    }
    np_ma.call_method("indices", (dimensions,), Some(&kwargs))
}

/// `numpy.ma.apply_along_axis(func1d, axis, arr, *args, **kwargs)` — apply
/// `func1d` to 1-D masked slices of `arr` along `axis`
/// (`numpy/ma/extras.py` `def apply_along_axis`).
#[pyfunction]
#[pyo3(signature = (func1d, axis, arr, *args, **kwargs))]
pub fn apply_along_axis<'py>(
    py: Python<'py>,
    func1d: &Bound<'py, PyAny>,
    axis: isize,
    arr: &Bound<'py, PyAny>,
    args: &Bound<'py, pyo3::types::PyTuple>,
    kwargs: Option<&Bound<'py, pyo3::types::PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let np_ma = py.import("numpy.ma")?;
    let npma = build_npma(py, arr)?;
    let mut items: Vec<Bound<'py, PyAny>> =
        vec![func1d.clone(), axis.into_pyobject(py)?.into_any(), npma];
    items.extend(args.iter());
    let call_args = pyo3::types::PyTuple::new(py, items)?;
    np_ma.call_method("apply_along_axis", call_args, kwargs)
}

/// `numpy.ma.apply_over_axes(func, a, axes)` — apply `func` over each axis in
/// `axes` successively (`numpy/ma/extras.py` `def apply_over_axes`).
#[pyfunction]
pub fn apply_over_axes<'py>(
    py: Python<'py>,
    func: &Bound<'py, PyAny>,
    a: &Bound<'py, PyAny>,
    axes: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let np_ma = py.import("numpy.ma")?;
    let npma = build_npma(py, a)?;
    np_ma.call_method1("apply_over_axes", (func, npma, axes))
}

/// `numpy.ma.ndenumerate(a, compressed=True)` — yield `(index, value)` pairs.
/// With the default `compressed=True` masked positions are skipped; with
/// `compressed=False` a masked position yields the `numpy.ma.masked` constant
/// (`numpy/ma/extras.py` `def ndenumerate`). Returns a Python list of pairs
/// (numpy.ma's generator materialized at the boundary).
#[pyfunction]
#[pyo3(signature = (a, compressed = true))]
pub fn ndenumerate<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    compressed: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let np_ma = py.import("numpy.ma")?;
    let npma = build_npma(py, a)?;
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("compressed", compressed)?;
    let iter = np_ma.call_method("ndenumerate", (npma,), Some(&kwargs))?;
    // Materialize the generator into a list so the Python caller can iterate /
    // index it without holding the GIL-bound generator object.
    py.import("builtins")?.call_method1("list", (iter,))
}

/// `numpy.ma.hsplit(ary, indices_or_sections)` — split a masked array
/// horizontally (along axis 1; axis 0 for 1-D), preserving the mask per piece
/// (`numpy/ma/core.py` `hsplit`). Returns a Python list of masked arrays.
#[pyfunction]
pub fn hsplit<'py>(
    py: Python<'py>,
    ary: &Bound<'py, PyAny>,
    indices_or_sections: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let np_ma = py.import("numpy.ma")?;
    let npma = build_npma(py, ary)?;
    np_ma.call_method1("hsplit", (npma, indices_or_sections))
}

/// `numpy.ma.mask_rows(a, axis=None)` — mask whole rows containing any masked
/// value (`numpy/ma/extras.py` `def mask_rows`, `== mask_rowcols(a, 0)`).
/// Wired to ferray-ma's `ma_mask_rowcols` (#835).
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn mask_rows<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<PyMaskedArray> {
    // numpy.ma.mask_rows fixes axis=0; the optional `axis` arg is accepted for
    // signature parity but only `None`/`0` are meaningful (numpy ignores it and
    // always masks rows). Reject a non-row axis to surface caller error early.
    if let Some(ax) = axis {
        if ax != 0 {
            return Err(PyValueError::new_err(format!(
                "ma.mask_rows masks rows (axis 0); got axis={ax}"
            )));
        }
    }
    let m = ma_as_ix2(&coerce_to_ma(py, a)?)?;
    let out = fma::ma_mask_rowcols(&m, Some(0)).map_err(ferr_to_pyerr)?;
    ix2_to_py(out)
}

/// `numpy.ma.mask_cols(a, axis=None)` — mask whole columns containing any
/// masked value (`numpy/ma/extras.py` `def mask_cols`, `== mask_rowcols(a, 1)`).
/// Wired to ferray-ma's `ma_mask_rowcols` (#835).
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn mask_cols<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<PyMaskedArray> {
    if let Some(ax) = axis {
        if ax != 1 && ax != -1 {
            return Err(PyValueError::new_err(format!(
                "ma.mask_cols masks columns (axis 1); got axis={ax}"
            )));
        }
    }
    let m = ma_as_ix2(&coerce_to_ma(py, a)?)?;
    let out = fma::ma_mask_rowcols(&m, Some(1)).map_err(ferr_to_pyerr)?;
    ix2_to_py(out)
}

/// `numpy.ma.compress_nd(x, axis=None)` — suppress slices along `axis`
/// (all axes when `None`) that contain any masked value, returning a plain
/// `numpy.ndarray` (`numpy/ma/extras.py` `def compress_nd`).
#[pyfunction]
#[pyo3(signature = (x, axis = None))]
pub fn compress_nd<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    axis: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let np_ma = py.import("numpy.ma")?;
    let npma = build_npma(py, x)?;
    let kwargs = pyo3::types::PyDict::new(py);
    if let Some(ax) = axis {
        kwargs.set_item("axis", ax)?;
    }
    np_ma.call_method("compress_nd", (npma,), Some(&kwargs))
}

/// `numpy.ma.MaskedArray.ids` analog — `numpy.ma.ids(a)` returns the
/// `(id(data), id(mask))` pair (`numpy/ma/core.py` `def ids`). The ferray
/// data + mask buffers are surfaced as fresh numpy arrays whose object ids are
/// returned (matching numpy's documented `(data_id, mask_id)` tuple shape).
#[pyfunction]
pub fn ids<'py>(py: Python<'py>, a: &PyMaskedArray) -> PyResult<Bound<'py, PyAny>> {
    let data_py = a.data(py)?;
    let mask_py = a.mask(py)?;
    let builtins = py.import("builtins")?;
    let data_id = builtins.call_method1("id", (data_py,))?;
    let mask_id = builtins.call_method1("id", (mask_py,))?;
    Ok(pyo3::types::PyTuple::new(py, [data_id, mask_id])?.into_any())
}

// ===========================================================================
// numpy.ma structured-mask + integer-bitwise batch 4 (refs #835 #818): the
// residual non-stateful numpy.ma surface — structured-dtype mask vocabulary
// (`make_mask_descr`, `flatten_mask`, `flatten_structured_array`, `fromflex`,
// `mvoid`), the buffer-protocol constructor (`frombuffer`), and the
// integer-dtype masked bitwise shifts (`left_shift`, `right_shift`).
//
// These all touch numpy's SHARED boundary vocabulary (structured dtypes,
// record/void scalars, the buffer protocol, dtype-preserving integer results)
// that the f64-only `PyMaskedArray` wrapper cannot represent without a lossy
// f64 cast (R-CODE-4). They therefore delegate to numpy.ma's canonical
// algorithm and return numpy's dtype-preserving object verbatim — exactly the
// established shared-vocabulary delegation pattern of `masked_all` / `indices`
// (batch 3) and the `mvoid` / `bool_` scalar re-exports.
//
// Every contract was verified live against numpy 2.4.5; the pytest oracle
// (tests/test_expansion_ma_batch4.py) constructs every expected value from
// `numpy.ma.*` directly (R-CHAR-3).
// ===========================================================================

/// `numpy.ma.make_mask_descr(ndtype)` — the all-bool mask dtype mirroring a
/// (possibly structured) dtype (`numpy/ma/core.py` `def make_mask_descr`).
///
/// Pure structured-dtype vocabulary: a plain dtype maps to `bool`, a structured
/// dtype maps to the same field layout with every field's type replaced by
/// `bool` (recursing into nested sub-dtypes). The result is numpy's own
/// `numpy.dtype` object, preserved verbatim across the boundary.
#[pyfunction]
pub fn make_mask_descr<'py>(
    py: Python<'py>,
    ndtype: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let np_ma = py.import("numpy.ma")?;
    np_ma.call_method1("make_mask_descr", (ndtype,))
}

/// `numpy.ma.flatten_mask(mask)` — flatten a (possibly nested / structured)
/// mask to a flat bool `numpy.ndarray` (`numpy/ma/core.py` `def flatten_mask`).
///
/// Structured-mask vocabulary: a structured-dtype mask is recursively flattened
/// into a 1-D `bool` array, so a `[('a', bool), ('b', [('ba', bool), ('bb',
/// bool)])]` mask of length N becomes a flat `bool` array of length `3*N`.
#[pyfunction]
pub fn flatten_mask<'py>(py: Python<'py>, mask: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let np_ma = py.import("numpy.ma")?;
    np_ma.call_method1("flatten_mask", (mask,))
}

/// `numpy.ma.flatten_structured_array(a)` — flatten a structured array to a
/// (2-D for multi-field rows, else 1-D) array
/// (`numpy/ma/core.py` `def flatten_structured_array`).
///
/// Structured-dtype vocabulary: each structured record is flattened into a row
/// of its (recursively-expanded) scalar fields, yielding a plain numeric array
/// with numpy's promoted dtype preserved verbatim.
#[pyfunction]
pub fn flatten_structured_array<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let np_ma = py.import("numpy.ma")?;
    np_ma.call_method1("flatten_structured_array", (a,))
}

/// `numpy.ma.fromflex(fxarray)` — rebuild a `MaskedArray` from a flexible
/// structured array carrying `_data` / `_mask` fields (the inverse of
/// `MaskedArray.toflex` / `torecords`) (`numpy/ma/core.py` `def fromflex`).
///
/// Structured-record vocabulary: `fxarray` is a structured array whose `_data`
/// field carries the values and whose `_mask` field carries the mask; numpy
/// reassembles them into a dtype-preserving `numpy.ma.MaskedArray`, returned
/// verbatim. fr.ma does not expose `toflex`, so this is a standalone
/// inverse-constructor (no fr.ma round-trip to wire).
#[pyfunction]
pub fn fromflex<'py>(py: Python<'py>, fxarray: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let np_ma = py.import("numpy.ma")?;
    np_ma.call_method1("fromflex", (fxarray,))
}

/// `numpy.ma.frombuffer(buffer, dtype=float, count=-1, offset=0)` — a
/// `MaskedArray` (no mask) over a buffer-protocol object
/// (`numpy/ma/core.py` `frombuffer`).
///
/// Buffer-protocol shared vocabulary: numpy interprets `buffer`'s bytes as a
/// flat array of `dtype` (honouring `count` / `offset`) and wraps it as a
/// `numpy.ma.MaskedArray` with `nomask`; the dtype-preserving result is
/// returned verbatim.
#[pyfunction]
#[pyo3(signature = (buffer, dtype = None, count = -1, offset = 0))]
pub fn frombuffer<'py>(
    py: Python<'py>,
    buffer: &Bound<'py, PyAny>,
    dtype: Option<&Bound<'py, PyAny>>,
    count: isize,
    offset: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let np_ma = py.import("numpy.ma")?;
    let kwargs = pyo3::types::PyDict::new(py);
    match dtype {
        Some(dt) => kwargs.set_item("dtype", dt)?,
        // numpy.ma.frombuffer's default dtype is `float` (Python builtin).
        None => kwargs.set_item("dtype", py.import("builtins")?.getattr("float")?)?,
    }
    kwargs.set_item("count", count)?;
    kwargs.set_item("offset", offset)?;
    np_ma.call_method("frombuffer", (buffer,), Some(&kwargs))
}

/// Build a *dtype-preserving* `numpy.ma.MaskedArray` (Python object) from any
/// masked input, WITHOUT the f64 coercion `build_npma` performs.
///
/// A `ferray.ma.MaskedArray` is f64-only, so it routes through the f64
/// [`build_npma`]; any other array-like (a plain `numpy.ndarray`, a
/// `numpy.ma.MaskedArray`, a list, …) is passed to `numpy.ma.asarray`, which
/// preserves its native dtype and mask. This is required for the integer-dtype
/// bitwise shifts, which numpy refuses to evaluate on f64 data (R-CODE-4: no
/// lossy int→f64 round-trip).
fn build_npma_native<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if obj.extract::<PyMaskedArray>().is_ok() {
        return build_npma(py, obj);
    }
    let np_ma = py.import("numpy.ma")?;
    np_ma.call_method1("asarray", (obj,))
}

/// `numpy.ma.left_shift(a, n)` — masked elementwise bitwise left shift on
/// integer data, preserving the mask (`numpy/ma/core.py` `def left_shift`).
///
/// fr.ma's `PyMaskedArray` is f64-only and cannot hold int data losslessly
/// (R-CODE-4), so this delegates via [`build_npma_native`] to
/// `numpy.ma.left_shift`, returning numpy's dtype-preserving masked result
/// (same pattern as `masked_all` / `indices`).
#[pyfunction]
pub fn left_shift<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    n: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let np_ma = py.import("numpy.ma")?;
    let npma = build_npma_native(py, a)?;
    np_ma.call_method1("left_shift", (npma, n))
}

/// `numpy.ma.right_shift(a, n)` — masked elementwise bitwise right shift on
/// integer data, preserving the mask (`numpy/ma/core.py` `def right_shift`).
///
/// Delegates via [`build_npma_native`] to `numpy.ma.right_shift` (see
/// [`left_shift`]).
#[pyfunction]
pub fn right_shift<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    n: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let np_ma = py.import("numpy.ma")?;
    let npma = build_npma_native(py, a)?;
    np_ma.call_method1("right_shift", (npma, n))
}
