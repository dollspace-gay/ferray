// ferray-core: datetime64 / timedelta64 element types and TimeUnit
//
// Mirrors NumPy's datetime64 and timedelta64 dtypes — both are int64
// counts of `TimeUnit` ticks since the Unix epoch (datetime) or zero
// (timedelta). The NaT ("Not a Time") sentinel is `i64::MIN`, matching
// NumPy's wire format and NPY descriptor encoding (`<M8[ns]` / `<m8[ns]`).
//
// These types are deliberately newtype wrappers around `i64` so they
// can be distinguished by the trait-dispatch system from raw integer
// arrays. The TimeUnit lives at the [`DType`](super::DType) level —
// `DType::DateTime64(TimeUnit)` and `DType::Timedelta64(TimeUnit)` —
// so a single `DateTime64` value can be reinterpreted across units by
// changing only the runtime descriptor.

use core::fmt;

use super::private;
use super::{DType, Element};

/// Time unit for [`DateTime64`] / [`Timedelta64`] arrays.
///
/// Mirrors the unit suffix of NumPy descriptors like `"<M8[ns]"`. Each
/// variant denotes the meaning of one i64 tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimeUnit {
    /// Nanoseconds (1 tick = 1ns). NumPy default.
    Ns,
    /// Microseconds (1 tick = 1µs).
    Us,
    /// Milliseconds (1 tick = 1ms).
    Ms,
    /// Seconds (1 tick = 1s).
    S,
    /// Minutes (1 tick = 60s).
    M,
    /// Hours (1 tick = 3600s).
    H,
    /// Days (1 tick = 86400s).
    D,
}

impl TimeUnit {
    /// NumPy descriptor suffix (e.g. `"ns"`, `"us"`, `"D"`).
    #[inline]
    #[must_use]
    pub const fn descr_suffix(self) -> &'static str {
        match self {
            Self::Ns => "ns",
            Self::Us => "us",
            Self::Ms => "ms",
            Self::S => "s",
            Self::M => "m",
            Self::H => "h",
            Self::D => "D",
        }
    }

    /// Parse a NumPy descriptor unit suffix into a `TimeUnit`. Returns
    /// `None` for an unknown / unsupported suffix.
    #[inline]
    #[must_use]
    pub fn from_descr_suffix(s: &str) -> Option<Self> {
        match s {
            "ns" => Some(Self::Ns),
            "us" | "μs" => Some(Self::Us),
            "ms" => Some(Self::Ms),
            "s" => Some(Self::S),
            "m" => Some(Self::M),
            "h" => Some(Self::H),
            "D" => Some(Self::D),
            _ => None,
        }
    }

    /// Number of nanoseconds in one tick of this unit.
    ///
    /// Ns = 1, Us = 1_000, Ms = 1_000_000, S = 10^9, M = 60·10^9, etc.
    /// Used by [`TimeUnit::finer`] / [`TimeUnit::scale_to`] to compute
    /// promotion factors. Always finite; the largest value is `D` =
    /// `86_400 * 10^9 = 8.64e13` which fits comfortably in i64.
    #[inline]
    #[must_use]
    pub const fn ns_per_tick(self) -> i64 {
        match self {
            Self::Ns => 1,
            Self::Us => 1_000,
            Self::Ms => 1_000_000,
            Self::S => 1_000_000_000,
            Self::M => 60 * 1_000_000_000,
            Self::H => 3_600 * 1_000_000_000,
            Self::D => 86_400 * 1_000_000_000,
        }
    }

    /// The finer (smaller-tick) of two units. NumPy's promotion rule for
    /// arithmetic between `datetime64[Ns]` and `datetime64[Us]` is to
    /// rescale both to `Ns` before subtracting; this helper picks that
    /// target unit.
    #[inline]
    #[must_use]
    pub const fn finer(self, other: Self) -> Self {
        if self.ns_per_tick() <= other.ns_per_tick() {
            self
        } else {
            other
        }
    }

    /// Scale factor to convert from `self` ticks to `target` ticks.
    ///
    /// Returns `Some(factor)` when `target` is the same as or finer than
    /// `self` (a value of `n` self-ticks corresponds to `n * factor`
    /// target-ticks). Returns `None` when `target` is coarser than
    /// `self`, since a strict integer rescale would lose precision —
    /// NumPy's promotion always goes finer, never coarser, so the API
    /// matches.
    #[inline]
    #[must_use]
    pub const fn scale_to(self, target: Self) -> Option<i64> {
        let src = self.ns_per_tick();
        let dst = target.ns_per_tick();
        if src % dst == 0 {
            Some(src / dst)
        } else {
            None
        }
    }
}

impl fmt::Display for TimeUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.descr_suffix())
    }
}

/// NaT ("Not a Time") sentinel value — the minimum i64.
///
/// Both [`DateTime64`] and [`Timedelta64`] use this exact bit pattern
/// to represent "no value", matching NumPy's wire convention.
pub const NAT: i64 = i64::MIN;

// ---------------------------------------------------------------------------
// DateTime64
// ---------------------------------------------------------------------------

/// Calendar instant stored as an i64 count of [`TimeUnit`] ticks since
/// the Unix epoch (1970-01-01T00:00:00Z).
///
/// The unit itself lives in [`DType::DateTime64`](super::DType::DateTime64);
/// a `DateTime64` value alone is unit-less, just an i64 count.
///
/// # NaT semantics
/// The sentinel value [`NAT`] (`i64::MIN`) represents "Not a Time". Per
/// NumPy convention, `DateTime64(NAT) != DateTime64(NAT)` — NaT is
/// never equal to anything, even itself. Use [`DateTime64::is_nat`] to
/// check for the sentinel.
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct DateTime64(pub i64);

impl DateTime64 {
    /// Construct a NaT (`Not a Time`) value.
    #[inline]
    #[must_use]
    pub const fn nat() -> Self {
        Self(NAT)
    }

    /// Whether this value is the NaT sentinel.
    #[inline]
    #[must_use]
    pub const fn is_nat(self) -> bool {
        self.0 == NAT
    }
}

impl PartialEq for DateTime64 {
    /// NaT is never equal to anything, including another NaT — matching
    /// NumPy's behavior on datetime64.
    fn eq(&self, other: &Self) -> bool {
        if self.is_nat() || other.is_nat() {
            false
        } else {
            self.0 == other.0
        }
    }
}

impl fmt::Display for DateTime64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_nat() {
            f.write_str("NaT")
        } else {
            write!(f, "{}", self.0)
        }
    }
}

// ---------------------------------------------------------------------------
// Timedelta64
// ---------------------------------------------------------------------------

/// Duration stored as an i64 count of [`TimeUnit`] ticks.
///
/// Same NaT semantics as [`DateTime64`]: the sentinel value [`NAT`]
/// represents "Not a Time", and `NaT != NaT`.
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct Timedelta64(pub i64);

impl Timedelta64 {
    /// Construct a NaT value.
    #[inline]
    #[must_use]
    pub const fn nat() -> Self {
        Self(NAT)
    }

    /// Whether this value is the NaT sentinel.
    #[inline]
    #[must_use]
    pub const fn is_nat(self) -> bool {
        self.0 == NAT
    }
}

impl PartialEq for Timedelta64 {
    fn eq(&self, other: &Self) -> bool {
        if self.is_nat() || other.is_nat() {
            false
        } else {
            self.0 == other.0
        }
    }
}

impl fmt::Display for Timedelta64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_nat() {
            f.write_str("NaT")
        } else {
            write!(f, "{}", self.0)
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar arithmetic operator overloads
// ---------------------------------------------------------------------------
//
// All four operations preserve the NaT-as-NaN rule: any NaT operand
// makes the result NaT. The arithmetic uses `wrapping_*` to match the
// kernel implementations in ferray-ufunc::ops::datetime so that scalar
// loops behave identically to vectorized ones.

impl core::ops::Sub<DateTime64> for DateTime64 {
    type Output = Timedelta64;

    /// `datetime - datetime → timedelta`. NaT propagates.
    fn sub(self, rhs: DateTime64) -> Timedelta64 {
        if self.is_nat() || rhs.is_nat() {
            Timedelta64::nat()
        } else {
            Timedelta64(self.0.wrapping_sub(rhs.0))
        }
    }
}

impl core::ops::Add<Timedelta64> for DateTime64 {
    type Output = DateTime64;

    /// `datetime + timedelta → datetime`. NaT propagates.
    fn add(self, rhs: Timedelta64) -> DateTime64 {
        if self.is_nat() || rhs.is_nat() {
            DateTime64::nat()
        } else {
            DateTime64(self.0.wrapping_add(rhs.0))
        }
    }
}

impl core::ops::Sub<Timedelta64> for DateTime64 {
    type Output = DateTime64;

    /// `datetime - timedelta → datetime`. NaT propagates.
    fn sub(self, rhs: Timedelta64) -> DateTime64 {
        if self.is_nat() || rhs.is_nat() {
            DateTime64::nat()
        } else {
            DateTime64(self.0.wrapping_sub(rhs.0))
        }
    }
}

impl core::ops::Add<Timedelta64> for Timedelta64 {
    type Output = Timedelta64;

    /// `timedelta + timedelta → timedelta`. NaT propagates.
    fn add(self, rhs: Timedelta64) -> Timedelta64 {
        if self.is_nat() || rhs.is_nat() {
            Timedelta64::nat()
        } else {
            Timedelta64(self.0.wrapping_add(rhs.0))
        }
    }
}

impl core::ops::Sub<Timedelta64> for Timedelta64 {
    type Output = Timedelta64;

    /// `timedelta - timedelta → timedelta`. NaT propagates.
    fn sub(self, rhs: Timedelta64) -> Timedelta64 {
        if self.is_nat() || rhs.is_nat() {
            Timedelta64::nat()
        } else {
            Timedelta64(self.0.wrapping_sub(rhs.0))
        }
    }
}

impl core::ops::Neg for Timedelta64 {
    type Output = Timedelta64;

    /// Negate a duration. NaT stays NaT.
    fn neg(self) -> Timedelta64 {
        if self.is_nat() {
            Timedelta64::nat()
        } else {
            // wrapping_neg handles i64::MIN+1..= cleanly; the i64::MIN
            // case is reserved for NaT and is filtered out above.
            Timedelta64(self.0.wrapping_neg())
        }
    }
}

// ---------------------------------------------------------------------------
// Array-level operator overloads (feature = "std")
// ---------------------------------------------------------------------------
//
// These mirror NumPy's datetime arithmetic on arrays:
//   &Array<DateTime64, D>  - &Array<DateTime64, D>  → Result<Array<Timedelta64, D>>
//   &Array<DateTime64, D>  + &Array<Timedelta64, D> → Result<Array<DateTime64, D>>
//   &Array<DateTime64, D>  - &Array<Timedelta64, D> → Result<Array<DateTime64, D>>
//   &Array<Timedelta64, D> ± &Array<Timedelta64, D> → Result<Array<Timedelta64, D>>
//
// All impls walk the element pairs in lockstep and reuse the scalar
// `Add`/`Sub` impls above (which carry the NaT-propagation logic).
// They live in ferray-core because the orphan rule requires the type
// or the trait to be local — `Array` lives here, so this is the only
// place these impls can land.
//
// Std-only because they construct fresh arrays; the no_std build of
// ferray-core doesn't expose `Array` in the first place.

#[cfg(feature = "std")]
mod array_ops {
    use super::{DateTime64, Timedelta64};
    use crate::array::owned::Array;
    use crate::dimension::Dimension;
    use crate::error::FerrayResult;

    fn check_same_shape<A, B, D>(a: &Array<A, D>, b: &Array<B, D>, op: &str) -> FerrayResult<()>
    where
        A: crate::dtype::Element,
        B: crate::dtype::Element,
        D: Dimension,
    {
        if a.shape() != b.shape() {
            return Err(crate::error::FerrayError::shape_mismatch(format!(
                "{op}: shapes {:?} and {:?} differ",
                a.shape(),
                b.shape()
            )));
        }
        Ok(())
    }

    impl<D: Dimension> core::ops::Sub<&Array<DateTime64, D>> for &Array<DateTime64, D> {
        type Output = FerrayResult<Array<Timedelta64, D>>;

        fn sub(self, rhs: &Array<DateTime64, D>) -> Self::Output {
            check_same_shape(self, rhs, "Array<DateTime64> - Array<DateTime64>")?;
            let data: Vec<Timedelta64> =
                self.iter().zip(rhs.iter()).map(|(a, b)| *a - *b).collect();
            Array::from_vec(self.dim().clone(), data)
        }
    }

    impl<D: Dimension> core::ops::Add<&Array<Timedelta64, D>> for &Array<DateTime64, D> {
        type Output = FerrayResult<Array<DateTime64, D>>;

        fn add(self, rhs: &Array<Timedelta64, D>) -> Self::Output {
            check_same_shape(self, rhs, "Array<DateTime64> + Array<Timedelta64>")?;
            let data: Vec<DateTime64> = self.iter().zip(rhs.iter()).map(|(a, b)| *a + *b).collect();
            Array::from_vec(self.dim().clone(), data)
        }
    }

    impl<D: Dimension> core::ops::Sub<&Array<Timedelta64, D>> for &Array<DateTime64, D> {
        type Output = FerrayResult<Array<DateTime64, D>>;

        fn sub(self, rhs: &Array<Timedelta64, D>) -> Self::Output {
            check_same_shape(self, rhs, "Array<DateTime64> - Array<Timedelta64>")?;
            let data: Vec<DateTime64> = self.iter().zip(rhs.iter()).map(|(a, b)| *a - *b).collect();
            Array::from_vec(self.dim().clone(), data)
        }
    }

    // `&Array<Timedelta64, D> + &Array<Timedelta64, D>` and the matching
    // `-` flow are covered by the generic `impl_binary_op!` blanket in
    // `crate::ops` — that blanket dispatches through the scalar
    // `Timedelta64: Add<Output = Timedelta64>` / `Sub<Output = Timedelta64>`
    // impls above, which already carry NaT propagation.
}

// ---------------------------------------------------------------------------
// chrono interop (feature = "chrono")
// ---------------------------------------------------------------------------

#[cfg(feature = "chrono")]
mod chrono_interop {
    use super::{DateTime64, NAT, TimeUnit, Timedelta64};
    use chrono::{
        DateTime, Duration, FixedOffset, NaiveDate, NaiveDateTime, NaiveTime, TimeZone, Utc,
    };

    impl DateTime64 {
        /// Convert this `DateTime64` (stored in `unit` ticks since the Unix
        /// epoch) to a `chrono::DateTime<Utc>`.
        ///
        /// Returns `None` if the value is NaT or if the resulting time
        /// falls outside chrono's representable range.
        ///
        /// Requires the `chrono` cargo feature.
        #[must_use]
        pub fn to_chrono(self, unit: TimeUnit) -> Option<DateTime<Utc>> {
            if self.is_nat() {
                return None;
            }
            let ticks = self.0;
            let (secs, nanos) = match unit {
                TimeUnit::Ns => {
                    let s = ticks.div_euclid(1_000_000_000);
                    let n = ticks.rem_euclid(1_000_000_000);
                    (s, n as u32)
                }
                TimeUnit::Us => {
                    let s = ticks.div_euclid(1_000_000);
                    let n = ticks.rem_euclid(1_000_000) * 1_000;
                    (s, n as u32)
                }
                TimeUnit::Ms => {
                    let s = ticks.div_euclid(1_000);
                    let n = ticks.rem_euclid(1_000) * 1_000_000;
                    (s, n as u32)
                }
                TimeUnit::S => (ticks, 0),
                TimeUnit::M => (ticks.checked_mul(60)?, 0),
                TimeUnit::H => (ticks.checked_mul(3600)?, 0),
                TimeUnit::D => (ticks.checked_mul(86_400)?, 0),
            };
            Utc.timestamp_opt(secs, nanos).single()
        }

        /// Construct a `DateTime64` from a `chrono::DateTime<Utc>` in the
        /// requested time unit.
        ///
        /// Returns `None` if the moment cannot be represented in `unit`
        /// ticks within `i64` bounds.
        ///
        /// Requires the `chrono` cargo feature.
        #[must_use]
        pub fn from_chrono(dt: DateTime<Utc>, unit: TimeUnit) -> Option<Self> {
            let secs = dt.timestamp();
            let nanos = dt.timestamp_subsec_nanos() as i64;
            let ticks: i64 = match unit {
                TimeUnit::Ns => secs.checked_mul(1_000_000_000)?.checked_add(nanos)?,
                TimeUnit::Us => secs
                    .checked_mul(1_000_000)?
                    .checked_add(nanos.div_euclid(1_000))?,
                TimeUnit::Ms => secs
                    .checked_mul(1_000)?
                    .checked_add(nanos.div_euclid(1_000_000))?,
                TimeUnit::S => secs,
                TimeUnit::M => secs.div_euclid(60),
                TimeUnit::H => secs.div_euclid(3_600),
                TimeUnit::D => secs.div_euclid(86_400),
            };
            // i64::MIN is reserved for NaT — refuse round-tripping a
            // chrono moment that lands on the sentinel.
            if ticks == NAT {
                return None;
            }
            Some(Self(ticks))
        }
    }

    impl DateTime64 {
        /// Construct a `DateTime64` from a `chrono::NaiveDate` (a
        /// timezone-naive date) at the start of the day in the requested
        /// unit.
        ///
        /// Internally treats the date as UTC midnight; in NumPy
        /// `datetime64[D]` is also TZ-naive, so this matches the parity
        /// expectation. Returns `None` if the resulting timestamp falls
        /// outside i64 range or lands on the NaT sentinel.
        ///
        /// Requires the `chrono` cargo feature.
        #[must_use]
        pub fn from_naive_date(date: NaiveDate, unit: TimeUnit) -> Option<Self> {
            let dt = date.and_time(NaiveTime::from_hms_opt(0, 0, 0)?);
            Self::from_naive_datetime(dt, unit)
        }

        /// Convert this `DateTime64` to a `chrono::NaiveDate` at the
        /// `unit` precision (date is the calendar day containing the
        /// timestamp interpreted as UTC).
        ///
        /// Returns `None` for NaT or out-of-chrono-range values.
        #[must_use]
        pub fn to_naive_date(self, unit: TimeUnit) -> Option<NaiveDate> {
            self.to_chrono(unit).map(|dt| dt.date_naive())
        }

        /// Construct a `DateTime64` from a `chrono::NaiveDateTime` (a
        /// timezone-naive instant) interpreted as UTC. Useful when the
        /// caller has a wall-clock timestamp without timezone metadata.
        ///
        /// Returns `None` on overflow or if the result lands on the NaT
        /// sentinel.
        ///
        /// Requires the `chrono` cargo feature.
        #[must_use]
        pub fn from_naive_datetime(dt: NaiveDateTime, unit: TimeUnit) -> Option<Self> {
            Self::from_chrono(dt.and_utc(), unit)
        }

        /// Convert to a `chrono::NaiveDateTime` (drops the implied UTC
        /// timezone). Returns `None` for NaT or out-of-range.
        #[must_use]
        pub fn to_naive_datetime(self, unit: TimeUnit) -> Option<NaiveDateTime> {
            self.to_chrono(unit).map(|dt| dt.naive_utc())
        }

        /// Construct a `DateTime64` from a `chrono::DateTime<FixedOffset>`.
        /// The instant is converted to the equivalent UTC time before
        /// being stored — ferray's datetime64 dtype is timezone-naive
        /// (matching NumPy), so the offset is folded into the underlying
        /// instant.
        ///
        /// Requires the `chrono` cargo feature.
        #[must_use]
        pub fn from_fixed_offset(dt: DateTime<FixedOffset>, unit: TimeUnit) -> Option<Self> {
            Self::from_chrono(dt.with_timezone(&Utc), unit)
        }

        /// Convert to a `chrono::DateTime<FixedOffset>` with the given
        /// offset. The stored instant is in UTC; the conversion shifts
        /// the wall-clock representation but preserves the moment.
        ///
        /// Returns `None` for NaT or out-of-range.
        #[must_use]
        pub fn to_fixed_offset(
            self,
            unit: TimeUnit,
            offset: FixedOffset,
        ) -> Option<DateTime<FixedOffset>> {
            self.to_chrono(unit).map(|utc| utc.with_timezone(&offset))
        }
    }

    impl Timedelta64 {
        /// Convert this `Timedelta64` to a `chrono::Duration`. Returns
        /// `None` for NaT or if the duration overflows chrono's range.
        ///
        /// Requires the `chrono` cargo feature.
        #[must_use]
        pub fn to_chrono(self, unit: TimeUnit) -> Option<Duration> {
            if self.is_nat() {
                return None;
            }
            let ticks = self.0;
            match unit {
                TimeUnit::Ns => Some(Duration::nanoseconds(ticks)),
                TimeUnit::Us => Some(Duration::microseconds(ticks)),
                TimeUnit::Ms => Some(Duration::milliseconds(ticks)),
                TimeUnit::S => Some(Duration::seconds(ticks)),
                TimeUnit::M => ticks.checked_mul(60).map(Duration::seconds),
                TimeUnit::H => ticks.checked_mul(3_600).map(Duration::seconds),
                TimeUnit::D => Some(Duration::days(ticks)),
            }
        }

        /// Construct a `Timedelta64` from a `chrono::Duration`.
        ///
        /// Requires the `chrono` cargo feature.
        #[must_use]
        pub fn from_chrono(d: Duration, unit: TimeUnit) -> Option<Self> {
            let ticks: i64 = match unit {
                TimeUnit::Ns => d.num_nanoseconds()?,
                TimeUnit::Us => d.num_microseconds()?,
                TimeUnit::Ms => d.num_milliseconds(),
                TimeUnit::S => d.num_seconds(),
                TimeUnit::M => d.num_minutes(),
                TimeUnit::H => d.num_hours(),
                TimeUnit::D => d.num_days(),
            };
            if ticks == NAT {
                return None;
            }
            Some(Self(ticks))
        }
    }
}

// ---------------------------------------------------------------------------
// Element impls
// ---------------------------------------------------------------------------
//
// DateTime64 and Timedelta64 are seal-implemented Element types. The
// `dtype()` impls return `DateTime64(TimeUnit::Ns)` and
// `Timedelta64(TimeUnit::Ns)` respectively — Ns is NumPy's default unit
// when none is specified, and the unit on a typed Array<DateTime64, D>
// is fixed at construction time. To work with a non-Ns unit at the
// type-erased boundary, use [`crate::DynArray`] with a
// `DType::DateTime64(other_unit)` tag.

impl private::Sealed for DateTime64 {}
impl private::Sealed for Timedelta64 {}

impl Element for DateTime64 {
    #[inline]
    fn dtype() -> DType {
        DType::DateTime64(TimeUnit::Ns)
    }

    #[inline]
    fn zero() -> Self {
        Self(0)
    }

    #[inline]
    fn one() -> Self {
        Self(1)
    }
}

impl Element for Timedelta64 {
    #[inline]
    fn dtype() -> DType {
        DType::Timedelta64(TimeUnit::Ns)
    }

    #[inline]
    fn zero() -> Self {
        Self(0)
    }

    #[inline]
    fn one() -> Self {
        Self(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeunit_descr_roundtrip() {
        for u in [
            TimeUnit::Ns,
            TimeUnit::Us,
            TimeUnit::Ms,
            TimeUnit::S,
            TimeUnit::M,
            TimeUnit::H,
            TimeUnit::D,
        ] {
            let s = u.descr_suffix();
            let parsed = TimeUnit::from_descr_suffix(s).unwrap();
            assert_eq!(parsed, u);
        }
    }

    #[test]
    fn datetime64_nat_neq_nat() {
        let a = DateTime64::nat();
        let b = DateTime64::nat();
        assert_ne!(a, b, "NaT must not equal NaT");
        assert!(a.is_nat());
        assert!(b.is_nat());
    }

    #[test]
    fn datetime64_normal_equality() {
        let a = DateTime64(1234);
        let b = DateTime64(1234);
        let c = DateTime64(5678);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn timedelta64_nat_neq_nat() {
        let a = Timedelta64::nat();
        let b = Timedelta64::nat();
        assert_ne!(a, b);
        assert!(a.is_nat());
    }

    #[test]
    fn datetime64_display_nat_and_normal() {
        assert_eq!(DateTime64::nat().to_string(), "NaT");
        assert_eq!(DateTime64(42).to_string(), "42");
    }

    #[test]
    fn element_impl_returns_ns_unit() {
        assert_eq!(DateTime64::dtype(), DType::DateTime64(TimeUnit::Ns));
        assert_eq!(Timedelta64::dtype(), DType::Timedelta64(TimeUnit::Ns));
    }

    #[test]
    fn nat_value_is_min_i64() {
        assert_eq!(NAT, i64::MIN);
        assert_eq!(DateTime64::nat().0, i64::MIN);
    }

    // ---- Scalar operator overloads ----

    #[test]
    fn scalar_datetime_sub_yields_timedelta() {
        let a = DateTime64(1_000);
        let b = DateTime64(400);
        let d: Timedelta64 = a - b;
        assert_eq!(d.0, 600);
    }

    #[test]
    fn scalar_datetime_plus_minus_timedelta() {
        let dt = DateTime64(1_000);
        let td = Timedelta64(150);
        assert_eq!((dt + td).0, 1_150);
        assert_eq!((dt - td).0, 850);
    }

    #[test]
    fn scalar_timedelta_arith() {
        let a = Timedelta64(50);
        let b = Timedelta64(20);
        assert_eq!((a + b).0, 70);
        assert_eq!((a - b).0, 30);
        let n = -a;
        assert_eq!(n.0, -50);
    }

    #[test]
    fn scalar_nat_propagates_through_ops() {
        let nat_dt = DateTime64::nat();
        let dt = DateTime64(100);
        let td = Timedelta64(10);
        let nat_td = Timedelta64::nat();

        assert!((nat_dt - dt).is_nat());
        assert!((dt - nat_dt).is_nat());
        assert!((nat_dt + td).is_nat());
        assert!((dt + nat_td).is_nat());
        assert!((dt - nat_td).is_nat());
        assert!((nat_td + td).is_nat());
        assert!((nat_td - td).is_nat());
        assert!((-nat_td).is_nat());
    }

    #[cfg(feature = "chrono")]
    mod chrono_tests {
        use super::super::{DateTime64, TimeUnit, Timedelta64};
        use chrono::{Duration, TimeZone, Utc};

        #[test]
        fn chrono_roundtrip_ns() {
            // 1 second after epoch in ns ticks.
            let dt64 = DateTime64(1_000_000_000);
            let chrono_dt = dt64.to_chrono(TimeUnit::Ns).unwrap();
            assert_eq!(chrono_dt, Utc.timestamp_opt(1, 0).single().unwrap());

            let back = DateTime64::from_chrono(chrono_dt, TimeUnit::Ns).unwrap();
            assert_eq!(back, dt64);
        }

        #[test]
        fn chrono_roundtrip_seconds() {
            let chrono_dt = Utc.timestamp_opt(1_700_000_000, 0).single().unwrap();
            let dt64 = DateTime64::from_chrono(chrono_dt, TimeUnit::S).unwrap();
            assert_eq!(dt64.0, 1_700_000_000);
            let back = dt64.to_chrono(TimeUnit::S).unwrap();
            assert_eq!(back, chrono_dt);
        }

        #[test]
        fn chrono_nat_returns_none() {
            assert!(DateTime64::nat().to_chrono(TimeUnit::Ns).is_none());
            assert!(Timedelta64::nat().to_chrono(TimeUnit::Ns).is_none());
        }

        #[test]
        fn timedelta64_chrono_roundtrip() {
            let td = Timedelta64(3_600_000_000_000);
            let chrono_d = td.to_chrono(TimeUnit::Ns).unwrap();
            // 3.6e12 ns = 1 hour
            assert_eq!(chrono_d, Duration::hours(1));
            let back = Timedelta64::from_chrono(chrono_d, TimeUnit::Ns).unwrap();
            assert_eq!(back.0, td.0);
        }

        #[test]
        fn from_chrono_micros_truncates_subus() {
            // 1.5 µs → 1 µs at us-unit precision (truncating toward zero).
            let chrono_dt = Utc.timestamp_opt(0, 1_500).single().unwrap();
            let dt64 = DateTime64::from_chrono(chrono_dt, TimeUnit::Us).unwrap();
            assert_eq!(dt64.0, 1);
        }

        #[test]
        fn naive_date_roundtrip() {
            use chrono::NaiveDate;
            let d = NaiveDate::from_ymd_opt(2025, 4, 30).unwrap();
            let dt64 = DateTime64::from_naive_date(d, TimeUnit::S).unwrap();
            let back = dt64.to_naive_date(TimeUnit::S).unwrap();
            assert_eq!(back, d);
        }

        #[test]
        fn naive_datetime_roundtrip() {
            use chrono::{NaiveDate, NaiveDateTime, NaiveTime};
            let date = NaiveDate::from_ymd_opt(2025, 4, 30).unwrap();
            let time = NaiveTime::from_hms_opt(12, 34, 56).unwrap();
            let ndt: NaiveDateTime = date.and_time(time);
            let dt64 = DateTime64::from_naive_datetime(ndt, TimeUnit::S).unwrap();
            let back = dt64.to_naive_datetime(TimeUnit::S).unwrap();
            assert_eq!(back, ndt);
        }

        #[test]
        fn fixed_offset_preserves_instant() {
            use chrono::FixedOffset;
            // 2025-04-30T12:00:00+05:00 represents the same moment as
            // 2025-04-30T07:00:00Z. After round-tripping through
            // DateTime64<S> (which stores UTC) and converting back to
            // +05:00, the wall-clock representation should match.
            let plus5 = FixedOffset::east_opt(5 * 3600).unwrap();
            let original = plus5
                .with_ymd_and_hms(2025, 4, 30, 12, 0, 0)
                .single()
                .unwrap();
            let dt64 = DateTime64::from_fixed_offset(original, TimeUnit::S).unwrap();
            // Stored as UTC: 07:00:00 → epoch seconds.
            let utc = dt64.to_chrono(TimeUnit::S).unwrap();
            assert_eq!(
                utc.format("%Y-%m-%dT%H:%M:%SZ").to_string(),
                "2025-04-30T07:00:00Z"
            );
            // Round back to +05:00:
            let back = dt64.to_fixed_offset(TimeUnit::S, plus5).unwrap();
            assert_eq!(back, original);
        }

        #[test]
        fn nat_date_returns_none() {
            assert!(DateTime64::nat().to_naive_date(TimeUnit::S).is_none());
            assert!(DateTime64::nat().to_naive_datetime(TimeUnit::Ns).is_none());
            use chrono::FixedOffset;
            let plus0 = FixedOffset::east_opt(0).unwrap();
            assert!(
                DateTime64::nat()
                    .to_fixed_offset(TimeUnit::S, plus0)
                    .is_none()
            );
        }
    }
}
