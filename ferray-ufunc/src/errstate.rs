//! Floating-point error state tracking (#386).
//!
//! Mirrors numpy's `np.errstate` / `np.seterr` API: a thread-local set
//! of policies for divide-by-zero, overflow, underflow, and invalid
//! (NaN-producing) operations. Each policy is one of:
//!
//! - [`FpErrorState::Ignore`] — silently allow (numpy default).
//! - [`FpErrorState::Warn`]   — record the event for later inspection.
//! - [`FpErrorState::Raise`]  — surface as a `FerrayError::FpException`
//!   on the next [`check_fp_errors`] call.
//!
//! Use [`with_errstate`] for scoped policy changes. The state is
//! per-thread, restored on guard drop.
//!
//! # Current ufunc integration
//!
//! Numerical ufunc kernels record divide-by-zero, overflow, underflow,
//! and invalid-domain events after evaluating a batch. The detection is
//! array-level, so a single ufunc call records each observed class at most
//! once and then lets the active policy decide whether to ignore, log, or
//! queue the event for [`check_fp_errors`].

use std::cell::RefCell;

use std::any::TypeId;

use ferray_core::Array;
use ferray_core::dimension::Dimension;
use ferray_core::dtype::Element;
use ferray_core::error::FerrayError;
use num_traits::Float;

/// Classes of floating-point events numpy distinguishes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FpErrorClass {
    /// Division by zero (1.0 / 0.0 → Inf).
    DivideByZero,
    /// Result overflows the dtype range (e.g. f32 above ~3.4e38 → +Inf).
    Overflow,
    /// Result underflows to zero or denormal.
    Underflow,
    /// NaN-producing operation (e.g. 0.0 / 0.0, sqrt(-1)).
    Invalid,
}

/// Policy for what to do when a given [`FpErrorClass`] event occurs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FpErrorState {
    /// Allow silently (numpy default for all classes except `invalid`,
    /// which numpy defaults to `Warn`).
    Ignore,
    /// Record the event in the per-thread log for inspection via
    /// [`take_fp_events`]. Equivalent to numpy's `'warn'` mode but
    /// without writing to stderr.
    Warn,
    /// Cause the next [`check_fp_errors`] call to return an
    /// `Err(FerrayError::InvalidValue)` describing the event.
    Raise,
}

/// Per-thread mapping from event class to current policy.
#[derive(Debug, Clone, Copy)]
struct Policies {
    divide_by_zero: FpErrorState,
    overflow: FpErrorState,
    underflow: FpErrorState,
    invalid: FpErrorState,
}

impl Default for Policies {
    fn default() -> Self {
        // Match numpy's Python-level default:
        // {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}.
        Self {
            divide_by_zero: FpErrorState::Warn,
            overflow: FpErrorState::Warn,
            underflow: FpErrorState::Ignore,
            invalid: FpErrorState::Warn,
        }
    }
}

thread_local! {
    static POLICIES: RefCell<Policies> = RefCell::new(Policies::default());
    static EVENT_LOG: RefCell<Vec<FpErrorClass>> = const { RefCell::new(Vec::new()) };
    static PENDING_RAISES: RefCell<Vec<FpErrorClass>> = const { RefCell::new(Vec::new()) };
}

/// RAII guard that restores the previous policy state on drop.
///
/// Constructed via [`with_errstate`]. Don't try to construct directly.
#[must_use]
pub struct ErrstateGuard {
    previous: Policies,
}

impl Drop for ErrstateGuard {
    fn drop(&mut self) {
        POLICIES.with(|p| *p.borrow_mut() = self.previous);
    }
}

/// Set the policy for a single error class. Returns the previous policy.
///
/// Mirrors numpy's `np.seterr(divide='raise')` etc. Note: callers that
/// want a scoped override should prefer [`with_errstate`] which restores
/// on drop instead of leaving the policy permanently changed.
pub fn seterr(class: FpErrorClass, state: FpErrorState) -> FpErrorState {
    POLICIES.with(|p| {
        let mut policies = p.borrow_mut();
        let previous = match class {
            FpErrorClass::DivideByZero => policies.divide_by_zero,
            FpErrorClass::Overflow => policies.overflow,
            FpErrorClass::Underflow => policies.underflow,
            FpErrorClass::Invalid => policies.invalid,
        };
        match class {
            FpErrorClass::DivideByZero => policies.divide_by_zero = state,
            FpErrorClass::Overflow => policies.overflow = state,
            FpErrorClass::Underflow => policies.underflow = state,
            FpErrorClass::Invalid => policies.invalid = state,
        }
        previous
    })
}

/// Get the current policy for an error class.
#[must_use]
pub fn geterr(class: FpErrorClass) -> FpErrorState {
    POLICIES.with(|p| {
        let policies = p.borrow();
        match class {
            FpErrorClass::DivideByZero => policies.divide_by_zero,
            FpErrorClass::Overflow => policies.overflow,
            FpErrorClass::Underflow => policies.underflow,
            FpErrorClass::Invalid => policies.invalid,
        }
    })
}

/// Run `f` with the given per-class policies, restoring the previous
/// policies on return (even if `f` panics).
///
/// Mirrors numpy's `with errstate(divide='raise', invalid='ignore'): ...`.
pub fn with_errstate<F, R>(overrides: &[(FpErrorClass, FpErrorState)], f: F) -> R
where
    F: FnOnce() -> R,
{
    let previous = POLICIES.with(|p| *p.borrow());
    let _guard = ErrstateGuard { previous };
    for &(class, state) in overrides {
        seterr(class, state);
    }
    f()
}

/// Record that an FP event occurred. Called by ufunc kernels when they
/// detect divide-by-zero, overflow, etc.
///
/// Behavior depends on the current policy for `class`:
/// - `Ignore`: no-op.
/// - `Warn`:  appended to the per-thread event log
///   (drained via [`take_fp_events`]).
/// - `Raise`: queued as a pending error; the next [`check_fp_errors`]
///   call returns Err.
pub fn record_fp_event(class: FpErrorClass) {
    let policy = geterr(class);
    match policy {
        FpErrorState::Ignore => {}
        FpErrorState::Warn => {
            EVENT_LOG.with(|log| log.borrow_mut().push(class));
        }
        FpErrorState::Raise => {
            PENDING_RAISES.with(|q| q.borrow_mut().push(class));
        }
    }
}

#[derive(Debug, Default)]
struct FpEventFlags {
    divide_by_zero: bool,
    overflow: bool,
    underflow: bool,
    invalid: bool,
}

impl FpEventFlags {
    fn record(self) {
        if self.divide_by_zero {
            record_fp_event(FpErrorClass::DivideByZero);
        }
        if self.overflow {
            record_fp_event(FpErrorClass::Overflow);
        }
        if self.underflow {
            record_fp_event(FpErrorClass::Underflow);
        }
        if self.invalid {
            record_fp_event(FpErrorClass::Invalid);
        }
    }
}

#[inline]
fn zero<T: Float>() -> T {
    T::zero()
}

#[inline]
fn finite<T: Float>(x: T) -> bool {
    x.is_finite()
}

#[inline]
fn finite_nonzero<T: Float>(x: T) -> bool {
    finite(x) && x != zero()
}

#[inline]
fn all_finite<T: Float>(xs: &[T]) -> bool {
    xs.iter().all(|&x| finite(x))
}

#[inline]
fn any_nan<T: Float>(xs: &[T]) -> bool {
    xs.iter().any(|&x| x.is_nan())
}

#[inline]
fn underflowed<T: Float>(out: T, finite_nonzero_result_candidate: bool) -> bool {
    finite_nonzero_result_candidate
        && finite(out)
        && (out == zero() || (out != zero() && out.abs() < T::min_positive_value()))
}

#[inline]
fn overflowed_from_finite<T: Float>(out: T, inputs: &[T]) -> bool {
    all_finite(inputs) && out.is_infinite()
}

#[inline]
fn invalid_from_non_nan_inputs<T: Float>(out: T, inputs: &[T]) -> bool {
    out.is_nan() && !any_nan(inputs)
}

fn scan_unary<T, I, O>(input: I, output: O, mut classify: impl FnMut(T, T, &mut FpEventFlags))
where
    T: Float,
    I: IntoIterator<Item = T>,
    O: IntoIterator<Item = T>,
{
    let mut flags = FpEventFlags::default();
    for (x, out) in input.into_iter().zip(output) {
        classify(x, out, &mut flags);
    }
    flags.record();
}

fn scan_binary<T, IA, IB, O>(
    a: IA,
    b: IB,
    output: O,
    mut classify: impl FnMut(T, T, T, &mut FpEventFlags),
) where
    T: Float,
    IA: IntoIterator<Item = T>,
    IB: IntoIterator<Item = T>,
    O: IntoIterator<Item = T>,
{
    let mut flags = FpEventFlags::default();
    for ((x, y), out) in a.into_iter().zip(b).zip(output) {
        classify(x, y, out, &mut flags);
    }
    flags.record();
}

pub(crate) fn record_addlike_events<T, IA, IB, O>(a: IA, b: IB, output: O)
where
    T: Float,
    IA: IntoIterator<Item = T>,
    IB: IntoIterator<Item = T>,
    O: IntoIterator<Item = T>,
{
    scan_binary(a, b, output, |x, y, out, flags| {
        flags.overflow |= overflowed_from_finite(out, &[x, y]);
        flags.invalid |= invalid_from_non_nan_inputs(out, &[x, y]);
    });
}

pub(crate) fn record_multiply_events<T, IA, IB, O>(a: IA, b: IB, output: O)
where
    T: Float,
    IA: IntoIterator<Item = T>,
    IB: IntoIterator<Item = T>,
    O: IntoIterator<Item = T>,
{
    scan_binary(a, b, output, |x, y, out, flags| {
        flags.overflow |= overflowed_from_finite(out, &[x, y]);
        flags.underflow |= underflowed(out, finite_nonzero(x) && finite_nonzero(y));
        flags.invalid |= invalid_from_non_nan_inputs(out, &[x, y]);
    });
}

pub(crate) fn record_divide_events<T, IA, IB, O>(a: IA, b: IB, output: O)
where
    T: Float,
    IA: IntoIterator<Item = T>,
    IB: IntoIterator<Item = T>,
    O: IntoIterator<Item = T>,
{
    scan_binary(a, b, output, |x, y, out, flags| {
        if y == zero() && !x.is_nan() {
            if x == zero() {
                flags.invalid = true;
            } else {
                flags.divide_by_zero = true;
            }
            return;
        }
        flags.overflow |= overflowed_from_finite(out, &[x, y]);
        flags.underflow |= underflowed(out, finite_nonzero(x) && finite_nonzero(y));
        flags.invalid |= invalid_from_non_nan_inputs(out, &[x, y]);
    });
}

pub(crate) fn record_power_events<T, IA, IB, O>(a: IA, b: IB, output: O)
where
    T: Float,
    IA: IntoIterator<Item = T>,
    IB: IntoIterator<Item = T>,
    O: IntoIterator<Item = T>,
{
    scan_binary(a, b, output, |x, y, out, flags| {
        if x == zero() && y < zero() && !y.is_nan() {
            flags.divide_by_zero = true;
        }
        flags.overflow |= overflowed_from_finite(out, &[x, y]);
        flags.underflow |= underflowed(out, finite_nonzero(x) && finite(y));
        flags.invalid |= invalid_from_non_nan_inputs(out, &[x, y]);
    });
}

pub(crate) fn record_reciprocal_events<T, I, O>(input: I, output: O)
where
    T: Float,
    I: IntoIterator<Item = T>,
    O: IntoIterator<Item = T>,
{
    scan_unary(input, output, |x, out, flags| {
        if x == zero() {
            flags.divide_by_zero = true;
            return;
        }
        flags.overflow |= overflowed_from_finite(out, &[x]);
        flags.underflow |= underflowed(out, finite_nonzero(x));
        flags.invalid |= invalid_from_non_nan_inputs(out, &[x]);
    });
}

pub(crate) fn record_sqrt_events<T, I, O>(input: I, output: O)
where
    T: Float,
    I: IntoIterator<Item = T>,
    O: IntoIterator<Item = T>,
{
    scan_unary(input, output, |x, out, flags| {
        flags.invalid |= x < zero() || invalid_from_non_nan_inputs(out, &[x]);
        flags.underflow |= underflowed(out, finite_nonzero(x));
    });
}

pub(crate) fn record_log_events<T, I, O>(input: I, output: O)
where
    T: Float,
    I: IntoIterator<Item = T>,
    O: IntoIterator<Item = T>,
{
    scan_unary(input, output, |x, out, flags| {
        if x == zero() {
            flags.divide_by_zero = true;
        } else if x < zero() {
            flags.invalid = true;
        }
        flags.invalid |= invalid_from_non_nan_inputs(out, &[x]);
    });
}

pub(crate) fn record_log1p_events<T, I, O>(input: I, output: O)
where
    T: Float,
    I: IntoIterator<Item = T>,
    O: IntoIterator<Item = T>,
{
    scan_unary(input, output, |x, out, flags| {
        let neg_one = -T::one();
        if x == neg_one {
            flags.divide_by_zero = true;
        } else if x < neg_one {
            flags.invalid = true;
        }
        flags.invalid |= invalid_from_non_nan_inputs(out, &[x]);
    });
}

pub(crate) fn record_exp_events<T, I, O>(input: I, output: O)
where
    T: Float,
    I: IntoIterator<Item = T>,
    O: IntoIterator<Item = T>,
{
    scan_unary(input, output, |x, out, flags| {
        flags.overflow |= overflowed_from_finite(out, &[x]);
        flags.underflow |= underflowed(out, finite(x));
        flags.invalid |= invalid_from_non_nan_inputs(out, &[x]);
    });
}

pub(crate) fn record_square_events<T, I, O>(input: I, output: O)
where
    T: Float,
    I: IntoIterator<Item = T>,
    O: IntoIterator<Item = T>,
{
    scan_unary(input, output, |x, out, flags| {
        flags.overflow |= overflowed_from_finite(out, &[x]);
        flags.underflow |= underflowed(out, finite_nonzero(x));
        flags.invalid |= invalid_from_non_nan_inputs(out, &[x]);
    });
}

pub(crate) fn record_unit_domain_events<T, I, O>(input: I, output: O)
where
    T: Float,
    I: IntoIterator<Item = T>,
    O: IntoIterator<Item = T>,
{
    scan_unary(input, output, |x, out, flags| {
        let one = T::one();
        flags.invalid |= x < -one || x > one || invalid_from_non_nan_inputs(out, &[x]);
    });
}

pub(crate) fn record_arccosh_events<T, I, O>(input: I, output: O)
where
    T: Float,
    I: IntoIterator<Item = T>,
    O: IntoIterator<Item = T>,
{
    scan_unary(input, output, |x, out, flags| {
        flags.invalid |= x < T::one() || invalid_from_non_nan_inputs(out, &[x]);
    });
}

pub(crate) fn record_arctanh_events<T, I, O>(input: I, output: O)
where
    T: Float,
    I: IntoIterator<Item = T>,
    O: IntoIterator<Item = T>,
{
    scan_unary(input, output, |x, out, flags| {
        let one = T::one();
        if x == one || x == -one {
            flags.divide_by_zero = true;
        } else if x < -one || x > one {
            flags.invalid = true;
        }
        flags.invalid |= invalid_from_non_nan_inputs(out, &[x]);
    });
}

pub(crate) fn record_overflow_unary_events<T, I, O>(input: I, output: O)
where
    T: Float,
    I: IntoIterator<Item = T>,
    O: IntoIterator<Item = T>,
{
    scan_unary(input, output, |x, out, flags| {
        flags.overflow |= overflowed_from_finite(out, &[x]);
        flags.invalid |= invalid_from_non_nan_inputs(out, &[x]);
    });
}

pub(crate) fn record_invalid_unary_events<T, I, O>(input: I, output: O)
where
    T: Float,
    I: IntoIterator<Item = T>,
    O: IntoIterator<Item = T>,
{
    scan_unary(input, output, |x, out, flags| {
        flags.invalid |= invalid_from_non_nan_inputs(out, &[x]);
    });
}

fn collect_broadcast_f64<T, D>(array: &Array<T, D>, shape: &[usize]) -> Option<Vec<f64>>
where
    T: Element + Copy + 'static,
    D: Dimension,
{
    if TypeId::of::<T>() != TypeId::of::<f64>() {
        return None;
    }
    let view = array.broadcast_to(shape).ok()?;
    Some(
        view.iter()
            .map(|&x| {
                // SAFETY: guarded by the TypeId equality above.
                unsafe { *(&raw const x).cast::<f64>() }
            })
            .collect(),
    )
}

fn collect_broadcast_f32<T, D>(array: &Array<T, D>, shape: &[usize]) -> Option<Vec<f32>>
where
    T: Element + Copy + 'static,
    D: Dimension,
{
    if TypeId::of::<T>() != TypeId::of::<f32>() {
        return None;
    }
    let view = array.broadcast_to(shape).ok()?;
    Some(
        view.iter()
            .map(|&x| {
                // SAFETY: guarded by the TypeId equality above.
                unsafe { *(&raw const x).cast::<f32>() }
            })
            .collect(),
    )
}

fn collect_f64<T, D>(array: &Array<T, D>) -> Option<Vec<f64>>
where
    T: Element + Copy + 'static,
    D: Dimension,
{
    collect_broadcast_f64(array, array.shape())
}

fn collect_f32<T, D>(array: &Array<T, D>) -> Option<Vec<f32>>
where
    T: Element + Copy + 'static,
    D: Dimension,
{
    collect_broadcast_f32(array, array.shape())
}

macro_rules! real_binary_array_recorder {
    ($name:ident, $scanner:ident) => {
        pub(crate) fn $name<T, D1, D2, DO>(
            a: &Array<T, D1>,
            b: &Array<T, D2>,
            output: &Array<T, DO>,
        ) where
            T: Element + Copy + 'static,
            D1: Dimension,
            D2: Dimension,
            DO: Dimension,
        {
            let shape = output.shape();
            if let (Some(a), Some(b), Some(out)) = (
                collect_broadcast_f64(a, shape),
                collect_broadcast_f64(b, shape),
                collect_f64(output),
            ) {
                $scanner::<f64, _, _, _>(a, b, out);
            } else if let (Some(a), Some(b), Some(out)) = (
                collect_broadcast_f32(a, shape),
                collect_broadcast_f32(b, shape),
                collect_f32(output),
            ) {
                $scanner::<f32, _, _, _>(a, b, out);
            }
        }
    };
}

real_binary_array_recorder!(record_addlike_array_events, record_addlike_events);
real_binary_array_recorder!(record_multiply_array_events, record_multiply_events);
real_binary_array_recorder!(record_divide_array_events, record_divide_events);
real_binary_array_recorder!(record_power_array_events, record_power_events);

pub(crate) fn record_divide_mixed_array_events<A, B, O, DA, DB, DO>(
    a: &Array<A, DA>,
    b: &Array<B, DB>,
    output: &Array<O, DO>,
) where
    A: Element + Copy + 'static,
    B: Element + Copy + 'static,
    O: Element + Copy + 'static,
    DA: Dimension,
    DB: Dimension,
    DO: Dimension,
{
    let shape = output.shape();
    if let (Some(a), Some(b), Some(out)) = (
        collect_broadcast_f64(a, shape),
        collect_broadcast_f64(b, shape),
        collect_f64(output),
    ) {
        record_divide_events::<f64, _, _, _>(a, b, out);
    } else if let (Some(a), Some(b), Some(out)) = (
        collect_broadcast_f32(a, shape),
        collect_broadcast_f32(b, shape),
        collect_f32(output),
    ) {
        record_divide_events::<f32, _, _, _>(a, b, out);
    }
}

macro_rules! real_unary_array_recorder {
    ($name:ident, $scanner:ident) => {
        pub(crate) fn $name<T, D>(input: &Array<T, D>, output: &Array<T, D>)
        where
            T: Element + Copy + 'static,
            D: Dimension,
        {
            if let (Some(input), Some(out)) = (collect_f64(input), collect_f64(output)) {
                $scanner::<f64, _, _>(input, out);
            } else if let (Some(input), Some(out)) = (collect_f32(input), collect_f32(output)) {
                $scanner::<f32, _, _>(input, out);
            }
        }
    };
}

real_unary_array_recorder!(record_reciprocal_array_events, record_reciprocal_events);
real_unary_array_recorder!(record_sqrt_array_events, record_sqrt_events);
real_unary_array_recorder!(record_log_array_events, record_log_events);
real_unary_array_recorder!(record_log1p_array_events, record_log1p_events);
real_unary_array_recorder!(record_exp_array_events, record_exp_events);
real_unary_array_recorder!(record_square_array_events, record_square_events);
real_unary_array_recorder!(record_unit_domain_array_events, record_unit_domain_events);
real_unary_array_recorder!(record_arccosh_array_events, record_arccosh_events);
real_unary_array_recorder!(record_arctanh_array_events, record_arctanh_events);
real_unary_array_recorder!(
    record_overflow_unary_array_events,
    record_overflow_unary_events
);
real_unary_array_recorder!(
    record_invalid_unary_array_events,
    record_invalid_unary_events
);

/// Drain the per-thread event log, returning all `Warn`-policy events
/// recorded since the last call.
pub fn take_fp_events() -> Vec<FpErrorClass> {
    EVENT_LOG.with(|log| std::mem::take(&mut *log.borrow_mut()))
}

/// If any `Raise`-policy events have been recorded since the last call,
/// drain them and return an `Err` describing them.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` describing the queued events.
pub fn check_fp_errors() -> Result<(), FerrayError> {
    let pending = PENDING_RAISES.with(|q| std::mem::take(&mut *q.borrow_mut()));
    if pending.is_empty() {
        return Ok(());
    }
    Err(FerrayError::invalid_value(format!(
        "floating-point exception(s) raised: {pending:?}"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy_invalid_is_warn() {
        // Reset to defaults for this test in case earlier tests
        // changed the thread-local state.
        seterr(FpErrorClass::DivideByZero, FpErrorState::Warn);
        seterr(FpErrorClass::Overflow, FpErrorState::Warn);
        seterr(FpErrorClass::Underflow, FpErrorState::Ignore);
        seterr(FpErrorClass::Invalid, FpErrorState::Warn);
        assert_eq!(geterr(FpErrorClass::Invalid), FpErrorState::Warn);
        assert_eq!(geterr(FpErrorClass::DivideByZero), FpErrorState::Warn);
        assert_eq!(geterr(FpErrorClass::Overflow), FpErrorState::Warn);
    }

    #[test]
    fn seterr_changes_and_returns_previous() {
        let _ = take_fp_events();
        let prev = seterr(FpErrorClass::DivideByZero, FpErrorState::Raise);
        assert_eq!(geterr(FpErrorClass::DivideByZero), FpErrorState::Raise);
        // Restore.
        seterr(FpErrorClass::DivideByZero, prev);
        assert_eq!(geterr(FpErrorClass::DivideByZero), prev);
    }

    #[test]
    fn record_fp_event_warn_appends_to_log() {
        // Drain any leftover events from prior tests.
        let _ = take_fp_events();
        seterr(FpErrorClass::Overflow, FpErrorState::Warn);
        record_fp_event(FpErrorClass::Overflow);
        record_fp_event(FpErrorClass::Overflow);
        let events = take_fp_events();
        assert_eq!(events, vec![FpErrorClass::Overflow, FpErrorClass::Overflow]);
        // Drained: subsequent take returns empty.
        assert!(take_fp_events().is_empty());
        seterr(FpErrorClass::Overflow, FpErrorState::Ignore);
    }

    #[test]
    fn record_fp_event_ignore_is_noop() {
        let _ = take_fp_events();
        seterr(FpErrorClass::Underflow, FpErrorState::Ignore);
        record_fp_event(FpErrorClass::Underflow);
        assert!(take_fp_events().is_empty());
    }

    #[test]
    fn record_fp_event_raise_queues_for_check() {
        let _ = check_fp_errors();
        seterr(FpErrorClass::Invalid, FpErrorState::Raise);
        record_fp_event(FpErrorClass::Invalid);
        let err = check_fp_errors().unwrap_err();
        assert!(err.to_string().contains("Invalid"));
        // Drained: subsequent check is Ok.
        assert!(check_fp_errors().is_ok());
        seterr(FpErrorClass::Invalid, FpErrorState::Warn);
    }

    #[test]
    fn with_errstate_scopes_overrides() {
        let _ = take_fp_events();
        seterr(FpErrorClass::DivideByZero, FpErrorState::Ignore);
        let inner_state =
            with_errstate(&[(FpErrorClass::DivideByZero, FpErrorState::Warn)], || {
                record_fp_event(FpErrorClass::DivideByZero);
                geterr(FpErrorClass::DivideByZero)
            });
        assert_eq!(inner_state, FpErrorState::Warn);
        // After the scope, the policy is restored.
        assert_eq!(geterr(FpErrorClass::DivideByZero), FpErrorState::Ignore);
        // The event recorded inside the scope is still in the log.
        let events = take_fp_events();
        assert_eq!(events, vec![FpErrorClass::DivideByZero]);
    }

    #[test]
    fn with_errstate_restores_on_panic() {
        seterr(FpErrorClass::Overflow, FpErrorState::Ignore);
        let result = std::panic::catch_unwind(|| {
            with_errstate(&[(FpErrorClass::Overflow, FpErrorState::Raise)], || {
                panic!("forced panic")
            })
        });
        assert!(result.is_err());
        // Drop ran during unwind; policy restored.
        assert_eq!(geterr(FpErrorClass::Overflow), FpErrorState::Ignore);
    }
}
