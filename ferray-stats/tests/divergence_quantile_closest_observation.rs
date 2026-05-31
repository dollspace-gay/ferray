//! Divergence pin: `ferray_stats::QuantileMethod::ClosestObservation`
//! diverges from `numpy.quantile(..., method='closest_observation')`.
//!
//! ## Root cause
//! NumPy's `closest_observation` builds its virtual index as
//!   `index = (n * q) - 1 - 0.5  =  n*q - 1.5`
//! then rounds toward `floor(index) + 1` (i.e. `next`), EXCEPT when
//! `gamma == 0` AND `floor(index)` is odd, in which case it keeps
//! `previous = floor(index)`; the result is clipped to `>= 0`.
//! (numpy/lib/_function_base_impl.py:4637-4642 `_closest_observation`,
//! delegating to :4623-4634 `_discrete_interpolation_to_boundaries`:
//!   `_closest_observation`:
//!     `return _discrete_interpolation_to_boundaries((n * quantiles) - 1 - 0.5, ...)`
//!   `_discrete_interpolation_to_boundaries`:
//!     `previous = np.floor(index); next = previous + 1; gamma = index - previous`
//!     default `next`, conditioned `previous` where `(gamma==0)&(floor(index)%2==1)`,
//!     then `res[res < 0] = 0`.)
//!
//! ferray instead computes `k = round_half_to_even(n*q - 0.5)`
//! (ferray-stats/src/reductions/quantile.rs:250-261, the
//! `QuantileMethod::ClosestObservation` arm). The offset is `-0.5`
//! where NumPy uses `-1.5`, so ferray's index is systematically one
//! position too high. This is NOT within-tolerance noise (S8): the
//! returned *order statistic* is a different array element.
//!
//! All expected values are live `numpy 2.4.4` oracle results (R-CHAR-3):
//!   np.quantile(np.array([1.,2.,3.,4.]),       0.3,  method='closest_observation') -> 1.0
//!   np.quantile(np.array([1.,2.,3.,4.]),       0.5,  method='closest_observation') -> 2.0
//!   np.quantile(np.array([1.,2.,3.,4.,5.]),    0.25, method='closest_observation') -> 1.0
//!   np.quantile(np.array([1.,2.,3.,4.,5.]),    0.5,  method='closest_observation') -> 2.0
//!   np.quantile(np.array([1.,2.,3.,4.,5.]),    0.9,  method='closest_observation') -> 4.0
//!   np.quantile(np.array([1.,2.,3.,4.,5.,6.]), 0.7,  method='closest_observation') -> 4.0
//!   np.quantile(np.array([1.,2.,3.,4.,5.,6.]), 0.75, method='closest_observation') -> 4.0
//!   np.quantile(np.array([1.,2.,3.,4.,5.,6.]), 0.9,  method='closest_observation') -> 5.0
//!
//! Tracking: see crosslink blocker issue referenced in the audit report.

use ferray_core::{Array, Ix1};
use ferray_stats::{quantile_with_method, QuantileMethod};

fn arr(v: Vec<f64>) -> Array<f64, Ix1> {
    Array::<f64, Ix1>::from_vec(Ix1::new([v.len()]), v).unwrap()
}

fn closest(a: &Array<f64, Ix1>, q: f64) -> f64 {
    quantile_with_method(a, q, None, QuantileMethod::ClosestObservation)
        .unwrap()
        .iter()
        .next()
        .copied()
        .unwrap()
}

/// n=4, q=0.5 — NumPy index = 4*0.5 - 1.5 = 0.5, gamma=0.5 != 0 ->
/// `next` = floor(0.5)+1 = 1 -> sorted[1] = 2.0.
/// ferray: round_half_to_even(4*0.5 - 0.5) = round(1.5) = 2 -> sorted[2] = 3.0.
#[test]
fn divergence_closest_observation_n4_q_half() {
    // numpy 2.4.4: closest_observation -> 2.0
    assert_eq!(closest(&arr(vec![1., 2., 3., 4.]), 0.5), 2.0);
}

/// n=4, q=0.3 — NumPy index = 1.2 - 1.5 = -0.3 -> next = 0 (clipped) ->
/// sorted[0] = 1.0. ferray returns sorted[3] = ... (off-by-formula) -> 3.0.
#[test]
fn divergence_closest_observation_n4_q_0_3() {
    // numpy 2.4.4: closest_observation -> 1.0
    assert_eq!(closest(&arr(vec![1., 2., 3., 4.]), 0.3), 1.0);
}

/// n=5, q=0.25 — NumPy index = 1.25 - 1.5 = -0.25 -> next = 0 (clip) ->
/// sorted[0] = 1.0. ferray -> 2.0.
#[test]
fn divergence_closest_observation_n5_q_quarter() {
    // numpy 2.4.4: closest_observation -> 1.0
    assert_eq!(closest(&arr(vec![1., 2., 3., 4., 5.]), 0.25), 1.0);
}

/// n=5, q=0.5 — NumPy index = 2.5 - 1.5 = 1.0, gamma=0, floor=1 (odd) ->
/// keep `previous` = 1 -> sorted[1] = 2.0. ferray -> 3.0.
/// This is the even-order-statistic correction the ferray impl drops.
#[test]
fn divergence_closest_observation_n5_q_half_even_correction() {
    // numpy 2.4.4: closest_observation -> 2.0
    assert_eq!(closest(&arr(vec![1., 2., 3., 4., 5.]), 0.5), 2.0);
}

/// n=5, q=0.9 — NumPy -> 4.0. ferray -> 5.0.
#[test]
fn divergence_closest_observation_n5_q_0_9() {
    // numpy 2.4.4: closest_observation -> 4.0
    assert_eq!(closest(&arr(vec![1., 2., 3., 4., 5.]), 0.9), 4.0);
}

/// n=6, q=0.7 — NumPy -> 4.0. ferray -> 5.0.
#[test]
fn divergence_closest_observation_n6_q_0_7() {
    // numpy 2.4.4: closest_observation -> 4.0
    assert_eq!(closest(&arr(vec![1., 2., 3., 4., 5., 6.]), 0.7), 4.0);
}

/// n=6, q=0.75 — NumPy index = 4.5 - 1.5 = 3.0, gamma=0, floor=3 (odd) ->
/// keep `previous` = 3 -> sorted[3] = 4.0. ferray -> 5.0.
#[test]
fn divergence_closest_observation_n6_q_0_75_even_correction() {
    // numpy 2.4.4: closest_observation -> 4.0
    assert_eq!(closest(&arr(vec![1., 2., 3., 4., 5., 6.]), 0.75), 4.0);
}

/// n=6, q=0.9 — NumPy -> 5.0. ferray -> 6.0.
#[test]
fn divergence_closest_observation_n6_q_0_9() {
    // numpy 2.4.4: closest_observation -> 5.0
    assert_eq!(closest(&arr(vec![1., 2., 3., 4., 5., 6.]), 0.9), 5.0);
}
