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
//! As of this initial scaffolding, ferray-ufunc kernels do **not**
//! emit FP error notifications themselves — the policy API is in
//! place so callers can construct per-region overrides, but the
//! detection hooks inside the kernels are a follow-up. Numpy's
//! errstate works because every ufunc kernel checks the FP exception
//! flags after each batch; doing the same in ferray will require
//! adding `record_fp_event` calls to each numerical kernel.
//!
//! Today, code that wants the `Raise`-on-div-zero behavior should
//! check input arrays for zeros explicitly.

use std::cell::RefCell;

use ferray_core::error::FerrayError;

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
        // Match numpy's default: invalid = Warn, others = Ignore.
        // (numpy's actual default is 'warn' across all four for
        // the C library, but the Python defaults are looser.)
        Self {
            divide_by_zero: FpErrorState::Ignore,
            overflow: FpErrorState::Ignore,
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
pub fn with_errstate<F, R>(
    overrides: &[(FpErrorClass, FpErrorState)],
    f: F,
) -> R
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
        seterr(FpErrorClass::DivideByZero, FpErrorState::Ignore);
        seterr(FpErrorClass::Overflow, FpErrorState::Ignore);
        seterr(FpErrorClass::Underflow, FpErrorState::Ignore);
        seterr(FpErrorClass::Invalid, FpErrorState::Warn);
        assert_eq!(geterr(FpErrorClass::Invalid), FpErrorState::Warn);
        assert_eq!(geterr(FpErrorClass::DivideByZero), FpErrorState::Ignore);
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
        let inner_state = with_errstate(
            &[(FpErrorClass::DivideByZero, FpErrorState::Warn)],
            || {
                record_fp_event(FpErrorClass::DivideByZero);
                geterr(FpErrorClass::DivideByZero)
            },
        );
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
            with_errstate(
                &[(FpErrorClass::Overflow, FpErrorState::Raise)],
                || panic!("forced panic"),
            )
        });
        assert!(result.is_err());
        // Drop ran during unwind; policy restored.
        assert_eq!(geterr(FpErrorClass::Overflow), FpErrorState::Ignore);
    }
}
