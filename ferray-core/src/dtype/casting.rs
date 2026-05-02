// ferray-core: Type casting and inspection (REQ-25, REQ-26)
//
// Provides:
// - CastKind enum (No, Equiv, Safe, SameKind, Unsafe) mirroring NumPy's casting rules
// - can_cast(from, to, casting) — check if a cast is allowed
// - promote_types(a, b) — runtime promotion (delegates to promotion::result_type)
// - common_type(a, b) — alias for promote_types
// - min_scalar_type(dtype) — smallest type that can hold the range of dtype
// - issubdtype(child, parent) — type hierarchy check
// - isrealobj / iscomplexobj — type inspection predicates
// - astype() / view_cast() — methods on Array for explicit casting

#[cfg(feature = "std")]
use crate::dimension::Dimension;
use crate::dtype::{DType, Element};
#[cfg(feature = "std")]
use crate::error::FerrayError;
use crate::error::FerrayResult;

#[cfg(feature = "std")]
use super::promotion::PromoteTo;

// ---------------------------------------------------------------------------
// CastKind — mirrors NumPy's casting parameter
// ---------------------------------------------------------------------------

/// Describes the safety level of a type cast.
///
/// Mirrors `NumPy`'s `casting` parameter for `np.can_cast`, `np.result_type`, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CastKind {
    /// No casting allowed — types must be identical.
    No,
    /// Types must have the same byte-level representation (e.g., byte order only).
    /// In ferray (native endian only), this is the same as `No`.
    Equiv,
    /// Safe cast: no information loss (e.g., i32 -> i64, f32 -> f64).
    Safe,
    /// Same kind: both are the same category (int->int, float->float) regardless
    /// of size (e.g., i64 -> i32 is allowed but may truncate).
    SameKind,
    /// Any cast is allowed (may lose information, e.g., f64 -> i32).
    Unsafe,
}

// ---------------------------------------------------------------------------
// can_cast — check if a cast between DTypes is allowed
// ---------------------------------------------------------------------------

/// Check whether a cast from `from` to `to` is allowed under the given `casting` rule.
///
/// # Errors
/// Returns `FerrayError::InvalidDtype` if the combination is not recognized.
pub fn can_cast(from: DType, to: DType, casting: CastKind) -> FerrayResult<bool> {
    Ok(match casting {
        CastKind::No | CastKind::Equiv => from == to,
        CastKind::Safe => is_safe_cast(from, to),
        CastKind::SameKind => is_same_kind(from, to),
        CastKind::Unsafe => true, // anything goes
    })
}

/// Check if `from` can be safely cast to `to` without information loss.
fn is_safe_cast(from: DType, to: DType) -> bool {
    if from == to {
        return true;
    }
    // A safe cast exists when promoting from + to yields `to`.
    // That means `to` can represent all values of `from`.
    match super::promotion::result_type(from, to) {
        Ok(promoted) => promoted == to,
        Err(_) => false,
    }
}

/// Check if `from` and `to` are the same "kind" (integer, float, complex).
fn is_same_kind(from: DType, to: DType) -> bool {
    if from == to {
        return true;
    }
    dtype_kind(from) == dtype_kind(to)
}

/// Return a "kind" category for a `DType`.
fn dtype_kind(dt: DType) -> u8 {
    if dt == DType::Bool {
        0 // bool is its own kind but also considered integer-compatible
    } else if dt.is_integer() {
        1
    } else if dt.is_float() {
        2
    } else if dt.is_complex() {
        3
    } else {
        255
    }
}

// ---------------------------------------------------------------------------
// promote_types — runtime type promotion
// ---------------------------------------------------------------------------

/// Determine the promoted type for two dtypes (runtime version).
///
/// This is the runtime equivalent of the `promoted_type!()` compile-time macro.
/// Returns the smallest type that can represent both `a` and `b` without
/// precision loss.
///
/// # Errors
/// Returns `FerrayError::InvalidDtype` if promotion fails.
pub fn promote_types(a: DType, b: DType) -> FerrayResult<DType> {
    super::promotion::result_type(a, b)
}

// ---------------------------------------------------------------------------
// common_type — alias for promote_types
// ---------------------------------------------------------------------------

/// Determine the common type for two dtypes. This is an alias for [`promote_types`].
///
/// # Errors
/// Returns `FerrayError::InvalidDtype` if promotion fails.
pub fn common_type(a: DType, b: DType) -> FerrayResult<DType> {
    promote_types(a, b)
}

// ---------------------------------------------------------------------------
// min_scalar_type — smallest type for a given dtype
// ---------------------------------------------------------------------------

/// Return the smallest scalar type that can hold the full range of `dt`.
///
/// This is analogous to `NumPy`'s `np.min_scalar_type`. For example:
/// - `min_scalar_type(DType::I64)` returns `DType::I8` (the smallest signed int)
/// - `min_scalar_type(DType::F64)` returns `DType::F32` (smallest float that is lossless for the kind)
///
/// Note: without a concrete value, we return the smallest type of the same kind.
#[must_use]
pub const fn min_scalar_type(dt: DType) -> DType {
    #[cfg(feature = "bf16")]
    use DType::BF16;
    #[cfg(feature = "f16")]
    use DType::F16;
    use DType::{
        Bool, Complex32, Complex64, F32, F64, I8, I16, I32, I64, I128, I256, U8, U16, U32, U64,
        U128,
    };
    match dt {
        Bool => Bool,
        U8 | U16 | U32 | U64 | U128 => U8,
        I8 | I16 | I32 | I64 | I128 | I256 => I8,
        F32 | F64 => F32,
        Complex32 | Complex64 => Complex32,
        #[cfg(feature = "f16")]
        F16 => F16,
        #[cfg(feature = "bf16")]
        BF16 => BF16,
        // datetime64 and timedelta64 are i64-backed and don't promote
        // to a smaller integer family — keep them at their canonical
        // ns-unit form. Real reduction across units is a separate
        // arithmetic-promotion concern, not min-scalar-type.
        DType::DateTime64(_) => DType::DateTime64(super::TimeUnit::Ns),
        DType::Timedelta64(_) => DType::Timedelta64(super::TimeUnit::Ns),
        // Struct dtypes don't participate in scalar-type promotion;
        // structured arrays are atomic in this layer (#342). Return
        // the input dtype unchanged.
        DType::Struct(_) => dt,
        // Fixed-width string / void dtypes are byte-payload types
        // (#741) — they don't promote to anything narrower.
        DType::FixedAscii(_) | DType::FixedUnicode(_) | DType::RawBytes(_) => dt,
    }
}

// ---------------------------------------------------------------------------
// issubdtype — type hierarchy check (REQ-26)
// ---------------------------------------------------------------------------

/// Category for dtype hierarchy checks, matching `NumPy`'s abstract type categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DTypeCategory {
    /// Any numeric type (integer, float, or complex).
    Number,
    /// Integer types (signed or unsigned, including bool).
    Integer,
    /// Signed integer types.
    SignedInteger,
    /// Unsigned integer types (including bool).
    UnsignedInteger,
    /// Floating-point types.
    Floating,
    /// Complex floating-point types.
    ComplexFloating,
}

/// Check if a dtype is a sub-type of a given category.
///
/// Analogous to `NumPy`'s `np.issubdtype(dtype, category)`.
///
/// # Examples
/// ```
/// use ferray_core::dtype::DType;
/// use ferray_core::dtype::casting::{issubdtype, DTypeCategory};
///
/// assert!(issubdtype(DType::I32, DTypeCategory::Integer));
/// assert!(issubdtype(DType::I32, DTypeCategory::SignedInteger));
/// assert!(issubdtype(DType::I32, DTypeCategory::Number));
/// assert!(!issubdtype(DType::I32, DTypeCategory::Floating));
/// ```
#[must_use]
pub const fn issubdtype(dt: DType, category: DTypeCategory) -> bool {
    match category {
        DTypeCategory::Number => true, // all our dtypes are numeric
        DTypeCategory::Integer => dt.is_integer(),
        DTypeCategory::SignedInteger => dt.is_signed() && dt.is_integer(),
        DTypeCategory::UnsignedInteger => dt.is_integer() && !dt.is_signed(),
        DTypeCategory::Floating => dt.is_float(),
        DTypeCategory::ComplexFloating => dt.is_complex(),
    }
}

// ---------------------------------------------------------------------------
// isrealobj / iscomplexobj — type inspection predicates (REQ-26)
// ---------------------------------------------------------------------------

/// Check if an array's element type is a real (non-complex) type.
///
/// Analogous to `NumPy`'s `np.isrealobj`.
#[must_use]
pub fn isrealobj<T: Element>() -> bool {
    !T::dtype().is_complex()
}

/// Check if an array's element type is a complex type.
///
/// Analogous to `NumPy`'s `np.iscomplexobj`.
#[must_use]
pub fn iscomplexobj<T: Element>() -> bool {
    T::dtype().is_complex()
}

// ===========================================================================
// Everything below requires std (depends on Array / ndarray)
// ===========================================================================

// ---------------------------------------------------------------------------
// astype() — explicit type casting on Array (REQ-25)
// ---------------------------------------------------------------------------

/// Trait providing the `astype()` method for arrays.
///
/// This is intentionally a separate trait so that mixed-type binary operations
/// do NOT compile implicitly (REQ-24). Users must call `.astype::<T>()` or use
/// `add_promoted()` etc.
#[cfg(feature = "std")]
pub trait AsType<D: Dimension> {
    /// Cast all elements to type `U`, returning a new array.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidDtype` if the cast is not possible.
    fn astype<U: Element>(&self) -> FerrayResult<crate::array::owned::Array<U, D>>
    where
        Self: AsTypeInner<U, D>;
}

/// Internal trait that performs the actual casting. Implemented for specific
/// (T, U) pairs where `T: PromoteTo<U>` or where unsafe casting is valid.
#[cfg(feature = "std")]
pub trait AsTypeInner<U: Element, D: Dimension> {
    /// Perform the cast.
    fn astype_inner(&self) -> FerrayResult<crate::array::owned::Array<U, D>>;
}

#[cfg(feature = "std")]
impl<T: Element, D: Dimension> AsType<D> for crate::array::owned::Array<T, D> {
    fn astype<U: Element>(&self) -> FerrayResult<crate::array::owned::Array<U, D>>
    where
        Self: AsTypeInner<U, D>,
    {
        self.astype_inner()
    }
}

/// Blanket implementation: any T that can `PromoteTo`<U> can astype.
#[cfg(feature = "std")]
impl<T, U, D> AsTypeInner<U, D> for crate::array::owned::Array<T, D>
where
    T: Element + PromoteTo<U>,
    U: Element,
    D: Dimension,
{
    fn astype_inner(&self) -> FerrayResult<crate::array::owned::Array<U, D>> {
        let mapped = self.inner.mapv(super::promotion::PromoteTo::promote);
        Ok(crate::array::owned::Array::from_ndarray(mapped))
    }
}

// ---------------------------------------------------------------------------
// cast() — explicit type casting with safety check (issue #361)
//
// `astype<U>` above is restricted to safe (PromoteTo) conversions. NumPy
// users want to be able to cast lossily as well: `arr.astype(np.int8)` works
// even when the source is f64. To support that, we provide `cast::<U>(kind)`
// which uses the `CastTo` trait (covers every Element pair) and validates
// the cast against the chosen safety level via `can_cast`.
// ---------------------------------------------------------------------------

#[cfg(feature = "std")]
impl<T, D> crate::array::owned::Array<T, D>
where
    T: Element,
    D: Dimension,
{
    /// Cast this array to element type `U` with the given safety check.
    ///
    /// This is the general-purpose casting method matching `NumPy`'s
    /// `arr.astype(dtype, casting=...)`. Unlike [`AsType::astype`] (which is
    /// restricted to safe widening), this method permits any cast as long
    /// as `can_cast(T::dtype(), U::dtype(), casting)` returns `true`.
    ///
    /// # Arguments
    /// * `casting` — controls which casts are permitted:
    ///   - [`CastKind::No`] / [`CastKind::Equiv`]: only `T == U` allowed
    ///   - [`CastKind::Safe`]: information-preserving widening only
    ///   - [`CastKind::SameKind`]: int↔int, float↔float, etc. (may narrow)
    ///   - [`CastKind::Unsafe`]: any cast (the `NumPy` default)
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidDtype` if the requested cast is not
    /// permitted at the chosen safety level.
    ///
    /// # Examples
    /// ```
    /// # use ferray_core::Array;
    /// # use ferray_core::dimension::Ix1;
    /// # use ferray_core::dtype::casting::CastKind;
    /// let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.5, 2.7, -3.9]).unwrap();
    /// // f64 -> i32 truncates per `as` semantics
    /// let b = a.cast::<i32>(CastKind::Unsafe).unwrap();
    /// assert_eq!(b.as_slice().unwrap(), &[1, 2, -3]);
    /// ```
    pub fn cast<U: Element>(
        &self,
        casting: CastKind,
    ) -> FerrayResult<crate::array::owned::Array<U, D>>
    where
        T: super::unsafe_cast::CastTo<U>,
    {
        if !can_cast(T::dtype(), U::dtype(), casting)? {
            return Err(FerrayError::invalid_dtype(format!(
                "cannot cast {} -> {} with casting={:?}",
                T::dtype(),
                U::dtype(),
                casting
            )));
        }
        let mapped = self
            .inner
            .mapv(|x| <T as super::unsafe_cast::CastTo<U>>::cast_to(x));
        Ok(crate::array::owned::Array::from_ndarray(mapped))
    }
}

// ---------------------------------------------------------------------------
// view_cast() — reinterpret cast (zero-copy where possible)
// ---------------------------------------------------------------------------

/// Reinterpret the raw bytes of an array as a different element type.
///
/// This is analogous to `NumPy`'s `.view(dtype)`. It requires that the element
/// sizes match (or the last dimension is adjusted accordingly).
///
/// # Safety
/// This is a reinterpret cast: the bit patterns are preserved but interpreted
/// as a different type. The caller must ensure this is meaningful.
///
/// # Errors
/// Returns `FerrayError::InvalidDtype` if the element sizes are incompatible.
#[cfg(feature = "std")]
pub fn view_cast<T: Element, U: Element, D: Dimension>(
    arr: &crate::array::owned::Array<T, D>,
) -> FerrayResult<crate::array::owned::Array<U, D>> {
    let t_size = core::mem::size_of::<T>();
    let u_size = core::mem::size_of::<U>();

    if t_size != u_size {
        return Err(FerrayError::invalid_dtype(format!(
            "view cast requires equal element sizes: {} ({} bytes) vs {} ({} bytes)",
            T::dtype(),
            t_size,
            U::dtype(),
            u_size,
        )));
    }

    let t_align = core::mem::align_of::<T>();
    let u_align = core::mem::align_of::<U>();
    if u_align > t_align {
        return Err(FerrayError::invalid_dtype(format!(
            "view cast requires compatible alignment: {} (align {}) -> {} (align {})",
            T::dtype(),
            t_align,
            U::dtype(),
            u_align,
        )));
    }

    // Both types have the same size and compatible alignment
    let data: Vec<T> = arr.inner.iter().cloned().collect();
    let len = data.len();
    let cap = data.capacity();

    // SAFETY: T and U have the same size and U's alignment <= T's alignment.
    // The Vec<T> allocation satisfies U's alignment requirement.
    let reinterpreted: Vec<U> = unsafe {
        let mut data = core::mem::ManuallyDrop::new(data);
        Vec::from_raw_parts(data.as_mut_ptr().cast::<U>(), len, cap)
    };

    crate::array::owned::Array::from_vec(arr.dim().clone(), reinterpreted)
}

// ---------------------------------------------------------------------------
// add_promoted / mul_promoted etc. — promoted binary operations (REQ-24)
// ---------------------------------------------------------------------------

#[cfg(feature = "std")]
impl<T: Element, D: Dimension> crate::array::owned::Array<T, D> {
    /// Add two arrays after promoting both to their common type.
    ///
    /// This is the explicit way to perform mixed-type addition (REQ-24).
    /// Implicit mixed-type `+` does not compile.
    pub fn add_promoted<U>(
        &self,
        other: &crate::array::owned::Array<U, D>,
    ) -> FerrayResult<crate::array::owned::Array<<T as super::promotion::Promoted<U>>::Output, D>>
    where
        U: Element + PromoteTo<<T as super::promotion::Promoted<U>>::Output>,
        T: super::promotion::Promoted<U> + PromoteTo<<T as super::promotion::Promoted<U>>::Output>,
        <T as super::promotion::Promoted<U>>::Output:
            Element + core::ops::Add<Output = <T as super::promotion::Promoted<U>>::Output>,
    {
        type Out<A, B> = <A as super::promotion::Promoted<B>>::Output;

        let a_promoted: ndarray::Array<Out<T, U>, D::NdarrayDim> =
            self.inner.mapv(super::promotion::PromoteTo::promote);
        let b_promoted: ndarray::Array<Out<T, U>, D::NdarrayDim> =
            other.inner.mapv(super::promotion::PromoteTo::promote);

        let result = a_promoted + b_promoted;
        Ok(crate::array::owned::Array::from_ndarray(result))
    }

    /// Subtract two arrays after promoting both to their common type.
    pub fn sub_promoted<U>(
        &self,
        other: &crate::array::owned::Array<U, D>,
    ) -> FerrayResult<crate::array::owned::Array<<T as super::promotion::Promoted<U>>::Output, D>>
    where
        U: Element + PromoteTo<<T as super::promotion::Promoted<U>>::Output>,
        T: super::promotion::Promoted<U> + PromoteTo<<T as super::promotion::Promoted<U>>::Output>,
        <T as super::promotion::Promoted<U>>::Output:
            Element + core::ops::Sub<Output = <T as super::promotion::Promoted<U>>::Output>,
    {
        type Out<A, B> = <A as super::promotion::Promoted<B>>::Output;

        let a_promoted: ndarray::Array<Out<T, U>, D::NdarrayDim> =
            self.inner.mapv(super::promotion::PromoteTo::promote);
        let b_promoted: ndarray::Array<Out<T, U>, D::NdarrayDim> =
            other.inner.mapv(super::promotion::PromoteTo::promote);

        let result = a_promoted - b_promoted;
        Ok(crate::array::owned::Array::from_ndarray(result))
    }

    /// Multiply two arrays after promoting both to their common type.
    pub fn mul_promoted<U>(
        &self,
        other: &crate::array::owned::Array<U, D>,
    ) -> FerrayResult<crate::array::owned::Array<<T as super::promotion::Promoted<U>>::Output, D>>
    where
        U: Element + PromoteTo<<T as super::promotion::Promoted<U>>::Output>,
        T: super::promotion::Promoted<U> + PromoteTo<<T as super::promotion::Promoted<U>>::Output>,
        <T as super::promotion::Promoted<U>>::Output:
            Element + core::ops::Mul<Output = <T as super::promotion::Promoted<U>>::Output>,
    {
        type Out<A, B> = <A as super::promotion::Promoted<B>>::Output;

        let a_promoted: ndarray::Array<Out<T, U>, D::NdarrayDim> =
            self.inner.mapv(super::promotion::PromoteTo::promote);
        let b_promoted: ndarray::Array<Out<T, U>, D::NdarrayDim> =
            other.inner.mapv(super::promotion::PromoteTo::promote);

        let result = a_promoted * b_promoted;
        Ok(crate::array::owned::Array::from_ndarray(result))
    }

    /// Divide two arrays after promoting both to their common type.
    pub fn div_promoted<U>(
        &self,
        other: &crate::array::owned::Array<U, D>,
    ) -> FerrayResult<crate::array::owned::Array<<T as super::promotion::Promoted<U>>::Output, D>>
    where
        U: Element + PromoteTo<<T as super::promotion::Promoted<U>>::Output>,
        T: super::promotion::Promoted<U> + PromoteTo<<T as super::promotion::Promoted<U>>::Output>,
        <T as super::promotion::Promoted<U>>::Output:
            Element + core::ops::Div<Output = <T as super::promotion::Promoted<U>>::Output>,
    {
        type Out<A, B> = <A as super::promotion::Promoted<B>>::Output;

        let a_promoted: ndarray::Array<Out<T, U>, D::NdarrayDim> =
            self.inner.mapv(super::promotion::PromoteTo::promote);
        let b_promoted: ndarray::Array<Out<T, U>, D::NdarrayDim> =
            other.inner.mapv(super::promotion::PromoteTo::promote);

        let result = a_promoted / b_promoted;
        Ok(crate::array::owned::Array::from_ndarray(result))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::Ix1;
    use num_complex::Complex;

    #[test]
    fn can_cast_no() {
        assert!(can_cast(DType::F64, DType::F64, CastKind::No).unwrap());
        assert!(!can_cast(DType::F32, DType::F64, CastKind::No).unwrap());
    }

    #[test]
    fn can_cast_safe() {
        assert!(can_cast(DType::F32, DType::F64, CastKind::Safe).unwrap());
        assert!(can_cast(DType::I32, DType::I64, CastKind::Safe).unwrap());
        assert!(can_cast(DType::U8, DType::I16, CastKind::Safe).unwrap());
        // Downcast is not safe
        assert!(!can_cast(DType::F64, DType::F32, CastKind::Safe).unwrap());
        assert!(!can_cast(DType::I64, DType::I32, CastKind::Safe).unwrap());
    }

    #[test]
    fn can_cast_same_kind() {
        // int -> int (different size) is same_kind
        assert!(can_cast(DType::I64, DType::I32, CastKind::SameKind).unwrap());
        assert!(can_cast(DType::F64, DType::F32, CastKind::SameKind).unwrap());
        // float -> int is not same_kind
        assert!(!can_cast(DType::F64, DType::I32, CastKind::SameKind).unwrap());
    }

    #[test]
    fn can_cast_unsafe_allows_all() {
        assert!(can_cast(DType::F64, DType::I8, CastKind::Unsafe).unwrap());
        assert!(can_cast(DType::Complex64, DType::Bool, CastKind::Unsafe).unwrap());
    }

    #[test]
    fn promote_types_basic() {
        assert_eq!(promote_types(DType::F32, DType::F64).unwrap(), DType::F64);
        assert_eq!(promote_types(DType::I32, DType::F32).unwrap(), DType::F64);
        assert_eq!(promote_types(DType::U8, DType::I8).unwrap(), DType::I16);
    }

    #[test]
    fn common_type_is_promote_types() {
        assert_eq!(
            common_type(DType::I32, DType::F64).unwrap(),
            promote_types(DType::I32, DType::F64).unwrap()
        );
    }

    #[test]
    fn min_scalar_type_basics() {
        assert_eq!(min_scalar_type(DType::I64), DType::I8);
        assert_eq!(min_scalar_type(DType::F64), DType::F32);
        assert_eq!(min_scalar_type(DType::Complex64), DType::Complex32);
        assert_eq!(min_scalar_type(DType::Bool), DType::Bool);
        assert_eq!(min_scalar_type(DType::U32), DType::U8);
    }

    #[test]
    fn issubdtype_checks() {
        assert!(issubdtype(DType::I32, DTypeCategory::Integer));
        assert!(issubdtype(DType::I32, DTypeCategory::SignedInteger));
        assert!(issubdtype(DType::I32, DTypeCategory::Number));
        assert!(!issubdtype(DType::I32, DTypeCategory::Floating));
        assert!(!issubdtype(DType::I32, DTypeCategory::ComplexFloating));

        assert!(issubdtype(DType::U16, DTypeCategory::UnsignedInteger));
        assert!(!issubdtype(DType::U16, DTypeCategory::SignedInteger));

        assert!(issubdtype(DType::F64, DTypeCategory::Floating));
        assert!(issubdtype(DType::F64, DTypeCategory::Number));
        assert!(!issubdtype(DType::F64, DTypeCategory::Integer));

        assert!(issubdtype(DType::Complex64, DTypeCategory::ComplexFloating));
        assert!(issubdtype(DType::Complex64, DTypeCategory::Number));
    }

    #[test]
    fn isrealobj_iscomplexobj() {
        assert!(isrealobj::<f64>());
        assert!(isrealobj::<i32>());
        assert!(!isrealobj::<Complex<f64>>());

        assert!(iscomplexobj::<Complex<f64>>());
        assert!(iscomplexobj::<Complex<f32>>());
        assert!(!iscomplexobj::<f64>());
    }

    #[test]
    fn astype_widen() {
        let arr =
            crate::array::owned::Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![1, 2, 3]).unwrap();
        let result = arr.astype::<f64>().unwrap();
        assert_eq!(result.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(result.dtype(), DType::F64);
    }

    #[test]
    fn astype_same_type() {
        let arr = crate::array::owned::Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![1.5, 2.5])
            .unwrap();
        let result = arr.astype::<f64>().unwrap();
        assert_eq!(result.as_slice().unwrap(), &[1.5, 2.5]);
    }

    #[test]
    fn view_cast_same_size() {
        // f32 and i32 are both 4 bytes
        let arr = crate::array::owned::Array::<f32, Ix1>::from_vec(Ix1::new([2]), vec![1.0, 2.0])
            .unwrap();
        let result = view_cast::<f32, i32, Ix1>(&arr);
        assert!(result.is_ok());
        let casted = result.unwrap();
        assert_eq!(casted.shape(), &[2]);
    }

    #[test]
    fn view_cast_different_size_fails() {
        let arr = crate::array::owned::Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![1.0, 2.0])
            .unwrap();
        let result = view_cast::<f64, f32, Ix1>(&arr);
        assert!(result.is_err());
    }

    #[test]
    fn add_promoted_i32_f64() {
        let a =
            crate::array::owned::Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![1, 2, 3]).unwrap();
        let b =
            crate::array::owned::Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![0.5, 1.5, 2.5])
                .unwrap();
        let result = a.add_promoted(&b).unwrap();
        assert_eq!(result.dtype(), DType::F64);
        assert_eq!(result.as_slice().unwrap(), &[1.5, 3.5, 5.5]);
    }

    #[test]
    fn mul_promoted_u8_i16() {
        let a =
            crate::array::owned::Array::<u8, Ix1>::from_vec(Ix1::new([3]), vec![2, 3, 4]).unwrap();
        let b = crate::array::owned::Array::<i16, Ix1>::from_vec(Ix1::new([3]), vec![10, 20, 30])
            .unwrap();
        let result = a.mul_promoted(&b).unwrap();
        // u8 + i16 => i16
        assert_eq!(result.dtype(), DType::I16);
        assert_eq!(result.as_slice().unwrap(), &[20i16, 60, 120]);
    }

    #[test]
    fn sub_promoted_f32_f64() {
        let a = crate::array::owned::Array::<f32, Ix1>::from_vec(Ix1::new([2]), vec![10.0, 20.0])
            .unwrap();
        let b = crate::array::owned::Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![1.0, 2.0])
            .unwrap();
        let result = a.sub_promoted(&b).unwrap();
        assert_eq!(result.dtype(), DType::F64);
        assert_eq!(result.as_slice().unwrap(), &[9.0, 18.0]);
    }

    // -----------------------------------------------------------------------
    // Array::cast<U>(casting) — issue #361
    // -----------------------------------------------------------------------

    #[test]
    fn cast_unsafe_f64_to_i32_truncates() {
        let a = crate::array::owned::Array::<f64, Ix1>::from_vec(
            Ix1::new([4]),
            vec![1.5, 2.7, -3.9, 4.2],
        )
        .unwrap();
        let b = a.cast::<i32>(CastKind::Unsafe).unwrap();
        assert_eq!(b.as_slice().unwrap(), &[1, 2, -3, 4]);
        assert_eq!(b.dtype(), DType::I32);
    }

    #[test]
    fn cast_safe_widening_succeeds() {
        let a =
            crate::array::owned::Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![1, 2, 3]).unwrap();
        let b = a.cast::<i64>(CastKind::Safe).unwrap();
        assert_eq!(b.as_slice().unwrap(), &[1i64, 2, 3]);
    }

    #[test]
    fn cast_safe_narrowing_errors() {
        let a =
            crate::array::owned::Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0])
                .unwrap();
        let result = a.cast::<i32>(CastKind::Safe);
        assert!(result.is_err());
    }

    #[test]
    fn cast_same_kind_int_narrowing_succeeds() {
        let a =
            crate::array::owned::Array::<i64, Ix1>::from_vec(Ix1::new([3]), vec![1, 2, 3]).unwrap();
        let b = a.cast::<i32>(CastKind::SameKind).unwrap();
        assert_eq!(b.as_slice().unwrap(), &[1, 2, 3]);
    }

    #[test]
    fn cast_same_kind_float_to_int_errors() {
        let a = crate::array::owned::Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![1.0, 2.0])
            .unwrap();
        let result = a.cast::<i32>(CastKind::SameKind);
        assert!(result.is_err());
    }

    #[test]
    fn cast_no_requires_identity() {
        let a = crate::array::owned::Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![1.0, 2.0])
            .unwrap();
        // Same type allowed
        assert!(a.cast::<f64>(CastKind::No).is_ok());
        // Different type rejected
        assert!(a.cast::<f32>(CastKind::No).is_err());
    }

    #[test]
    fn cast_complex_to_real_unsafe() {
        let a = crate::array::owned::Array::<Complex<f64>, Ix1>::from_vec(
            Ix1::new([2]),
            vec![Complex::new(1.5, 2.0), Complex::new(3.5, -1.0)],
        )
        .unwrap();
        let b = a.cast::<f64>(CastKind::Unsafe).unwrap();
        assert_eq!(b.as_slice().unwrap(), &[1.5, 3.5]);
    }

    #[test]
    fn cast_real_to_complex_safe() {
        let a = crate::array::owned::Array::<f32, Ix1>::from_vec(Ix1::new([2]), vec![1.0, 2.0])
            .unwrap();
        let b = a.cast::<Complex<f64>>(CastKind::Safe).unwrap();
        assert_eq!(
            b.as_slice().unwrap(),
            &[Complex::new(1.0_f64, 0.0), Complex::new(2.0, 0.0)]
        );
    }

    #[test]
    fn cast_int_to_bool_unsafe() {
        let a = crate::array::owned::Array::<i32, Ix1>::from_vec(Ix1::new([4]), vec![0, 1, -3, 0])
            .unwrap();
        let b = a.cast::<bool>(CastKind::Unsafe).unwrap();
        assert_eq!(b.as_slice().unwrap(), &[false, true, true, false]);
    }
}
