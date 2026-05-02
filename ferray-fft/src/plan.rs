// ferray-fft: FftPlan type and global plan cache (REQ-12, REQ-13)

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, RwLock};

use num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::{Fft, FftPlanner};

use ferray_core::dimension::{Dimension, IxDyn};
use ferray_core::error::{FerrayError, FerrayResult};
use ferray_core::{Array, Ix1};

use crate::norm::{FftDirection, FftNorm};

/// Key for the global FFT plan cache: (transform size, `is_inverse`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CacheKey {
    size: usize,
    inverse: bool,
}

// ---------------------------------------------------------------------------
// Complex FFT plan caches (f32 and f64)
//
// The cache splits into two locks (#431):
//
//   * `RwLock<HashMap<key, Arc<plan>>>` — held in *read* mode for the
//     overwhelmingly common cache-hit lookup. Many concurrent FFTs of
//     already-cached sizes can read in parallel without serializing.
//   * `Mutex<FftPlanner<T>>` — held only on the (rare) cache miss path,
//     because `FftPlanner::plan_fft_forward` requires `&mut self`.
//
// The previous design wrapped both fields in a single `Mutex`, so even
// the lookup path serialized every concurrent caller; under heavy
// multi-threaded load that produced false sharing on what should be a
// read-mostly path.
// ---------------------------------------------------------------------------

// Aliases keep clippy's `type_complexity` lint quiet on the `LazyLock`
// statics below.
type ComplexPlanMap<T> = RwLock<HashMap<CacheKey, Arc<dyn Fft<T>>>>;

static F64_PLANS: LazyLock<ComplexPlanMap<f64>> = LazyLock::new(|| RwLock::new(HashMap::new()));
static F64_PLANNER: LazyLock<Mutex<FftPlanner<f64>>> =
    LazyLock::new(|| Mutex::new(FftPlanner::new()));
static F32_PLANS: LazyLock<ComplexPlanMap<f32>> = LazyLock::new(|| RwLock::new(HashMap::new()));
static F32_PLANNER: LazyLock<Mutex<FftPlanner<f32>>> =
    LazyLock::new(|| Mutex::new(FftPlanner::new()));

/// Obtain a cached f64 FFT plan for the given size and direction.
///
/// This is the primary internal entry point used by all f64 FFT functions.
/// Plans are cached globally so repeated transforms of the same size
/// reuse the same plan.
pub(crate) fn get_cached_plan_f64(size: usize, inverse: bool) -> Arc<dyn Fft<f64>> {
    let key = CacheKey { size, inverse };
    // Fast path: read lock the plan map and return an existing plan.
    {
        let plans = F64_PLANS.read().expect("f64 FFT plan cache poisoned");
        if let Some(plan) = plans.get(&key) {
            return plan.clone();
        }
    }
    // Cache miss: build a new plan with the planner Mutex, then insert
    // it under the write lock. Two concurrent misses for the same key
    // are handled by the second one finding the entry already populated.
    let plan: Arc<dyn Fft<f64>> = {
        let mut planner = F64_PLANNER.lock().expect("f64 FFT planner poisoned");
        if inverse {
            planner.plan_fft_inverse(size)
        } else {
            planner.plan_fft_forward(size)
        }
    };
    let mut plans = F64_PLANS.write().expect("f64 FFT plan cache poisoned");
    plans.entry(key).or_insert(plan).clone()
}

/// Obtain a cached f32 FFT plan for the given size and direction.
pub(crate) fn get_cached_plan_f32(size: usize, inverse: bool) -> Arc<dyn Fft<f32>> {
    let key = CacheKey { size, inverse };
    {
        let plans = F32_PLANS.read().expect("f32 FFT plan cache poisoned");
        if let Some(plan) = plans.get(&key) {
            return plan.clone();
        }
    }
    let plan: Arc<dyn Fft<f32>> = {
        let mut planner = F32_PLANNER.lock().expect("f32 FFT planner poisoned");
        if inverse {
            planner.plan_fft_inverse(size)
        } else {
            planner.plan_fft_forward(size)
        }
    };
    let mut plans = F32_PLANS.write().expect("f32 FFT plan cache poisoned");
    plans.entry(key).or_insert(plan).clone()
}

/// Legacy alias for f64 plan lookup — preserved for tests and any call
/// sites that haven't been generified yet. New code should go through
/// [`crate::float::FftFloat::cached_plan`].
pub(crate) fn get_cached_plan(size: usize, inverse: bool) -> Arc<dyn Fft<f64>> {
    get_cached_plan_f64(size, inverse)
}

// ---------------------------------------------------------------------------
// Real-FFT plan caches (issues #432, #426, #431)
//
// Same RwLock-plus-Mutex split as the complex caches above. realfft
// uses a separate planner type from rustfft because real-to-complex
// (RealToComplex<T>) and complex-to-real (ComplexToReal<T>) are
// distinct trait objects, so each direction gets its own plan map.
// ---------------------------------------------------------------------------

type RealForwardMap<T> = RwLock<HashMap<usize, Arc<dyn RealToComplex<T>>>>;
type RealInverseMap<T> = RwLock<HashMap<usize, Arc<dyn ComplexToReal<T>>>>;

static REAL_F64_FORWARD: LazyLock<RealForwardMap<f64>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));
static REAL_F64_INVERSE: LazyLock<RealInverseMap<f64>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));
static REAL_F64_PLANNER: LazyLock<Mutex<RealFftPlanner<f64>>> =
    LazyLock::new(|| Mutex::new(RealFftPlanner::new()));
static REAL_F32_FORWARD: LazyLock<RealForwardMap<f32>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));
static REAL_F32_INVERSE: LazyLock<RealInverseMap<f32>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));
static REAL_F32_PLANNER: LazyLock<Mutex<RealFftPlanner<f32>>> =
    LazyLock::new(|| Mutex::new(RealFftPlanner::new()));

/// Obtain a cached f64 real-to-complex FFT plan for the given size.
///
/// Returns a plan that consumes a real input of length `size` and
/// produces `size/2 + 1` complex values.
pub(crate) fn get_cached_real_forward_f64(size: usize) -> Arc<dyn RealToComplex<f64>> {
    {
        let plans = REAL_F64_FORWARD
            .read()
            .expect("f64 real-forward plan cache poisoned");
        if let Some(plan) = plans.get(&size) {
            return plan.clone();
        }
    }
    let plan: Arc<dyn RealToComplex<f64>> = {
        let mut planner = REAL_F64_PLANNER
            .lock()
            .expect("f64 real FFT planner poisoned");
        planner.plan_fft_forward(size)
    };
    let mut plans = REAL_F64_FORWARD
        .write()
        .expect("f64 real-forward plan cache poisoned");
    plans.entry(size).or_insert(plan).clone()
}

/// Obtain a cached f64 complex-to-real FFT plan for the given output size.
///
/// The argument `size` is the **real-output** length (not the complex
/// input length, which is `size/2 + 1`).
pub(crate) fn get_cached_real_inverse_f64(size: usize) -> Arc<dyn ComplexToReal<f64>> {
    {
        let plans = REAL_F64_INVERSE
            .read()
            .expect("f64 real-inverse plan cache poisoned");
        if let Some(plan) = plans.get(&size) {
            return plan.clone();
        }
    }
    let plan: Arc<dyn ComplexToReal<f64>> = {
        let mut planner = REAL_F64_PLANNER
            .lock()
            .expect("f64 real FFT planner poisoned");
        planner.plan_fft_inverse(size)
    };
    let mut plans = REAL_F64_INVERSE
        .write()
        .expect("f64 real-inverse plan cache poisoned");
    plans.entry(size).or_insert(plan).clone()
}

/// Obtain a cached f32 real-to-complex FFT plan for the given size.
pub(crate) fn get_cached_real_forward_f32(size: usize) -> Arc<dyn RealToComplex<f32>> {
    {
        let plans = REAL_F32_FORWARD
            .read()
            .expect("f32 real-forward plan cache poisoned");
        if let Some(plan) = plans.get(&size) {
            return plan.clone();
        }
    }
    let plan: Arc<dyn RealToComplex<f32>> = {
        let mut planner = REAL_F32_PLANNER
            .lock()
            .expect("f32 real FFT planner poisoned");
        planner.plan_fft_forward(size)
    };
    let mut plans = REAL_F32_FORWARD
        .write()
        .expect("f32 real-forward plan cache poisoned");
    plans.entry(size).or_insert(plan).clone()
}

/// Obtain a cached f32 complex-to-real FFT plan for the given output size.
pub(crate) fn get_cached_real_inverse_f32(size: usize) -> Arc<dyn ComplexToReal<f32>> {
    {
        let plans = REAL_F32_INVERSE
            .read()
            .expect("f32 real-inverse plan cache poisoned");
        if let Some(plan) = plans.get(&size) {
            return plan.clone();
        }
    }
    let plan: Arc<dyn ComplexToReal<f32>> = {
        let mut planner = REAL_F32_PLANNER
            .lock()
            .expect("f32 real FFT planner poisoned");
        planner.plan_fft_inverse(size)
    };
    let mut plans = REAL_F32_INVERSE
        .write()
        .expect("f32 real-inverse plan cache poisoned");
    plans.entry(size).or_insert(plan).clone()
}

/// A reusable FFT plan for a specific transform size.
///
/// `FftPlan` caches the internal FFT algorithm for a given size,
/// enabling efficient repeated transforms. Plans are `Send + Sync`
/// and can be shared across threads.
///
/// # Example
/// ```
/// use ferray_fft::FftPlan;
/// use ferray_core::{Array, Ix1};
/// use num_complex::Complex;
///
/// let plan = FftPlan::new(8).unwrap();
/// let signal = Array::<Complex<f64>, Ix1>::from_vec(
///     Ix1::new([8]),
///     vec![Complex::new(1.0, 0.0); 8],
/// ).unwrap();
/// let result = plan.execute(&signal).unwrap();
/// assert_eq!(result.shape(), &[8]);
/// ```
pub struct FftPlan {
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
    size: usize,
}

// FftPlan is Send + Sync because Arc<dyn Fft<f64>> is Send + Sync
// (rustfft plans are thread-safe). No manual unsafe impl needed.

impl FftPlan {
    /// Create a new FFT plan for the given transform size.
    ///
    /// The plan pre-computes the internal FFT algorithm so that
    /// subsequent calls to [`execute`](Self::execute) and
    /// [`execute_inverse`](Self::execute_inverse) are fast.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `size` is 0.
    pub fn new(size: usize) -> FerrayResult<Self> {
        if size == 0 {
            return Err(FerrayError::invalid_value("FFT plan size must be > 0"));
        }
        let forward = get_cached_plan(size, false);
        let inverse = get_cached_plan(size, true);
        Ok(Self {
            forward,
            inverse,
            size,
        })
    }

    /// Return the transform size this plan was created for.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Execute a forward FFT on the given signal.
    ///
    /// The input array must have exactly `self.size()` elements.
    /// Uses `FftNorm::Backward` (no scaling on forward).
    ///
    /// # Errors
    /// Returns `FerrayError::ShapeMismatch` if the input length
    /// does not match the plan size.
    pub fn execute(
        &self,
        signal: &Array<Complex<f64>, Ix1>,
    ) -> FerrayResult<Array<Complex<f64>, Ix1>> {
        self.execute_with_norm(signal, FftNorm::Backward)
    }

    /// Execute a forward FFT with the specified normalization.
    ///
    /// # Errors
    /// Returns `FerrayError::ShapeMismatch` if the input length
    /// does not match the plan size.
    pub fn execute_with_norm(
        &self,
        signal: &Array<Complex<f64>, Ix1>,
        norm: FftNorm,
    ) -> FerrayResult<Array<Complex<f64>, Ix1>> {
        if signal.size() != self.size {
            return Err(FerrayError::shape_mismatch(format!(
                "signal length {} does not match plan size {}",
                signal.size(),
                self.size,
            )));
        }
        let mut buffer: Vec<Complex<f64>> = signal.iter().copied().collect();
        let mut scratch = vec![Complex::new(0.0, 0.0); self.forward.get_inplace_scratch_len()];
        self.forward.process_with_scratch(&mut buffer, &mut scratch);

        let scale = norm.scale_factor(self.size, FftDirection::Forward);
        if (scale - 1.0).abs() > f64::EPSILON {
            for c in &mut buffer {
                *c *= scale;
            }
        }

        Array::from_vec(Ix1::new([self.size]), buffer)
    }

    /// Execute an inverse FFT on the given spectrum.
    ///
    /// Uses `FftNorm::Backward` (divides by `n` on inverse).
    ///
    /// # Errors
    /// Returns `FerrayError::ShapeMismatch` if the input length
    /// does not match the plan size.
    pub fn execute_inverse(
        &self,
        spectrum: &Array<Complex<f64>, Ix1>,
    ) -> FerrayResult<Array<Complex<f64>, Ix1>> {
        self.execute_inverse_with_norm(spectrum, FftNorm::Backward)
    }

    /// Execute an inverse FFT with the specified normalization.
    ///
    /// # Errors
    /// Returns `FerrayError::ShapeMismatch` if the input length
    /// does not match the plan size.
    pub fn execute_inverse_with_norm(
        &self,
        spectrum: &Array<Complex<f64>, Ix1>,
        norm: FftNorm,
    ) -> FerrayResult<Array<Complex<f64>, Ix1>> {
        if spectrum.size() != self.size {
            return Err(FerrayError::shape_mismatch(format!(
                "spectrum length {} does not match plan size {}",
                spectrum.size(),
                self.size,
            )));
        }
        let mut buffer: Vec<Complex<f64>> = spectrum.iter().copied().collect();
        let mut scratch = vec![Complex::new(0.0, 0.0); self.inverse.get_inplace_scratch_len()];
        self.inverse.process_with_scratch(&mut buffer, &mut scratch);

        let scale = norm.scale_factor(self.size, FftDirection::Inverse);
        if (scale - 1.0).abs() > f64::EPSILON {
            for c in &mut buffer {
                *c *= scale;
            }
        }

        Array::from_vec(Ix1::new([self.size]), buffer)
    }
}

impl std::fmt::Debug for FftPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The cached `Arc<dyn Fft<f64>>` is intentionally elided: it is an
        // opaque rustfft handle whose Debug repr is noise, not signal.
        f.debug_struct("FftPlan")
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// N-D plan object (#436)
//
// `FftPlanND` mirrors `FftPlan` for the multi-dimensional case: it locks
// in a transform shape + axis list at construction, pre-warms the
// per-axis f64 plan cache (so the first `execute()` call doesn't pay the
// planner Mutex), and validates input arrays against its shape on each
// call. Repeated 2-D / N-D transforms of the same shape — e.g. a video
// pipeline running fft2 on every frame — get the same speedup pattern
// the 1-D `FftPlan` already provided.
// ---------------------------------------------------------------------------

/// A reusable plan for repeated multi-dimensional FFTs of a fixed shape.
///
/// Pre-computes the per-axis 1-D FFT plans (forward and inverse) at
/// construction. Calling [`execute`](Self::execute) /
/// [`execute_inverse`](Self::execute_inverse) on an array of the
/// matching shape skips planner construction entirely.
///
/// `FftPlanND` is the multi-dim equivalent of [`FftPlan`]. For 1-D
/// transforms prefer [`FftPlan`] (lower overhead, simpler API). The
/// 2-D `fft2` / `ifft2` cases are covered by passing `axes = None`
/// with a 2-D shape, or by passing `axes = Some(&[-2, -1])` for
/// higher-dim arrays.
///
/// # Example
/// ```
/// use ferray_fft::{FftPlanND, FftNorm};
/// use ferray_core::{Array, dimension::Ix2};
/// use num_complex::Complex;
///
/// // Reusable plan for any 4×4 complex frame.
/// let plan = FftPlanND::new(&[4, 4], None).unwrap();
///
/// let frame = Array::<Complex<f64>, Ix2>::from_vec(
///     Ix2::new([4, 4]),
///     vec![Complex::new(1.0, 0.0); 16],
/// ).unwrap();
/// let spectrum = plan.execute(&frame, FftNorm::Backward).unwrap();
/// assert_eq!(spectrum.shape(), &[4, 4]);
/// ```
pub struct FftPlanND {
    shape: Vec<usize>,
    axes: Vec<usize>,
    /// Per-axis forward plans, in the order `axes` lists them.
    /// Held in `Arc` so the cache + plan share the same allocation.
    forward_plans: Vec<Arc<dyn Fft<f64>>>,
    /// Per-axis inverse plans, in the same order.
    inverse_plans: Vec<Arc<dyn Fft<f64>>>,
}

// `Arc<dyn Fft<f64>>` is `Send + Sync` (rustfft plans are thread-safe),
// so `FftPlanND` is too.

impl FftPlanND {
    /// Build a plan for a fixed-shape multi-dimensional transform.
    ///
    /// `shape` must match the shape of arrays later passed to
    /// [`execute`](Self::execute) / [`execute_inverse`](Self::execute_inverse).
    /// `axes` selects which dimensions to transform; `None` means "all
    /// of them" (matching `numpy.fft.fftn` defaults).
    ///
    /// Negative axis indices are accepted (`-1` == last axis), matching
    /// the conventions of [`crate::fftn`].
    ///
    /// # Errors
    /// - `FerrayError::InvalidValue` if `shape` is empty or contains a 0.
    /// - `FerrayError::AxisOutOfBounds` if any entry of `axes` is out of
    ///   range for `shape.len()`.
    pub fn new(shape: &[usize], axes: Option<&[isize]>) -> FerrayResult<Self> {
        if shape.is_empty() {
            return Err(FerrayError::invalid_value(
                "FftPlanND requires a non-empty shape",
            ));
        }
        if shape.contains(&0) {
            return Err(FerrayError::invalid_value(
                "FftPlanND shape dimensions must all be > 0",
            ));
        }
        let resolved = crate::axes::resolve_axes(shape.len(), axes)?;
        let mut forward_plans = Vec::with_capacity(resolved.len());
        let mut inverse_plans = Vec::with_capacity(resolved.len());
        for &ax in &resolved {
            let axis_len = shape[ax];
            forward_plans.push(get_cached_plan_f64(axis_len, false));
            inverse_plans.push(get_cached_plan_f64(axis_len, true));
        }
        Ok(Self {
            shape: shape.to_vec(),
            axes: resolved,
            forward_plans,
            inverse_plans,
        })
    }

    /// The shape this plan was built for.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The axes this plan transforms over (canonical, non-negative).
    #[must_use]
    pub fn axes(&self) -> &[usize] {
        &self.axes
    }

    /// Number of pre-computed per-axis 1-D plans (one forward + one
    /// inverse for each axis in [`axes`](Self::axes)).
    #[must_use]
    pub fn num_axis_plans(&self) -> usize {
        self.forward_plans.len()
    }

    /// Borrow the cached forward 1-D FFT plan used for the axis at
    /// position `i` in [`axes`](Self::axes). Useful for callers that
    /// want to drive the plan directly via `rustfft`.
    ///
    /// Returns `None` if `i` is out of range.
    #[must_use]
    pub fn forward_axis_plan(&self, i: usize) -> Option<&Arc<dyn Fft<f64>>> {
        self.forward_plans.get(i)
    }

    /// Borrow the cached inverse 1-D FFT plan used for the axis at
    /// position `i` in [`axes`](Self::axes).
    ///
    /// Returns `None` if `i` is out of range.
    #[must_use]
    pub fn inverse_axis_plan(&self, i: usize) -> Option<&Arc<dyn Fft<f64>>> {
        self.inverse_plans.get(i)
    }

    /// Run the forward N-D FFT on `a`.
    ///
    /// `a` must have the same shape this plan was built for.
    ///
    /// # Errors
    /// - `FerrayError::ShapeMismatch` if `a.shape() != self.shape()`.
    /// - Anything `crate::fftn` may surface for malformed inputs.
    pub fn execute<D: Dimension>(
        &self,
        a: &Array<Complex<f64>, D>,
        norm: FftNorm,
    ) -> FerrayResult<Array<Complex<f64>, IxDyn>>
    where
        Complex<f64>: ferray_core::Element,
    {
        self.check_shape(a.shape())?;
        let isize_axes: Vec<isize> = self.axes.iter().map(|&a| a as isize).collect();
        crate::fftn(a, None, Some(&isize_axes), norm)
    }

    /// Run the inverse N-D FFT on `a`.
    ///
    /// `a` must have the same shape this plan was built for.
    ///
    /// # Errors
    /// Same as [`execute`](Self::execute).
    pub fn execute_inverse<D: Dimension>(
        &self,
        a: &Array<Complex<f64>, D>,
        norm: FftNorm,
    ) -> FerrayResult<Array<Complex<f64>, IxDyn>>
    where
        Complex<f64>: ferray_core::Element,
    {
        self.check_shape(a.shape())?;
        let isize_axes: Vec<isize> = self.axes.iter().map(|&a| a as isize).collect();
        crate::ifftn(a, None, Some(&isize_axes), norm)
    }

    fn check_shape(&self, got: &[usize]) -> FerrayResult<()> {
        if got != self.shape.as_slice() {
            return Err(FerrayError::shape_mismatch(format!(
                "FftPlanND was built for shape {:?}, got {:?}",
                self.shape, got
            )));
        }
        Ok(())
    }
}

impl std::fmt::Debug for FftPlanND {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FftPlanND")
            .field("shape", &self.shape)
            .field("axes", &self.axes)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_new_valid() {
        let plan = FftPlan::new(8).unwrap();
        assert_eq!(plan.size(), 8);
    }

    #[test]
    fn plan_new_zero_errors() {
        assert!(FftPlan::new(0).is_err());
    }

    #[test]
    fn plan_execute_roundtrip() {
        let plan = FftPlan::new(4).unwrap();
        let data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        let signal = Array::<Complex<f64>, Ix1>::from_vec(Ix1::new([4]), data.clone()).unwrap();

        let spectrum = plan.execute(&signal).unwrap();
        let recovered = plan.execute_inverse(&spectrum).unwrap();

        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert!((orig.re - rec.re).abs() < 1e-12);
            assert!((orig.im - rec.im).abs() < 1e-12);
        }
    }

    #[test]
    fn plan_size_mismatch() {
        let plan = FftPlan::new(8).unwrap();
        let signal =
            Array::<Complex<f64>, Ix1>::from_vec(Ix1::new([4]), vec![Complex::new(0.0, 0.0); 4])
                .unwrap();
        assert!(plan.execute(&signal).is_err());
    }

    #[test]
    fn cached_plan_reuse() {
        // Getting the same plan twice should return the same Arc
        let p1 = get_cached_plan(16, false);
        let p2 = get_cached_plan(16, false);
        assert!(Arc::ptr_eq(&p1, &p2));
    }

    #[test]
    fn plan_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<FftPlan>();
    }

    // ---- FftPlanND (#436) ----------------------------------------------

    #[test]
    fn plan_nd_default_axes_match_fftn() {
        use ferray_core::dimension::Ix2;
        let data: Vec<Complex<f64>> = (0..16).map(|i| Complex::new(f64::from(i), 0.0)).collect();
        let a = Array::<Complex<f64>, Ix2>::from_vec(ferray_core::Ix2::new([4, 4]), data).unwrap();

        let plan = FftPlanND::new(&[4, 4], None).unwrap();
        let from_plan = plan.execute(&a, FftNorm::Backward).unwrap();
        let from_fn = crate::fftn(&a, None, None, FftNorm::Backward).unwrap();
        for (p, f) in from_plan.iter().zip(from_fn.iter()) {
            assert!((p.re - f.re).abs() < 1e-12);
            assert!((p.im - f.im).abs() < 1e-12);
        }
    }

    #[test]
    fn plan_nd_roundtrip() {
        use ferray_core::dimension::Ix2;
        let original: Vec<Complex<f64>> = (0..12)
            .map(|i| Complex::new(f64::from(i), -f64::from(i)))
            .collect();
        let a =
            Array::<Complex<f64>, Ix2>::from_vec(ferray_core::Ix2::new([3, 4]), original.clone())
                .unwrap();

        let plan = FftPlanND::new(&[3, 4], None).unwrap();
        let spectrum = plan.execute(&a, FftNorm::Backward).unwrap();
        let recovered = plan.execute_inverse(&spectrum, FftNorm::Backward).unwrap();
        for (orig, rec) in original.iter().zip(recovered.iter()) {
            assert!((orig.re - rec.re).abs() < 1e-10);
            assert!((orig.im - rec.im).abs() < 1e-10);
        }
    }

    #[test]
    fn plan_nd_explicit_axes_subset() {
        use ferray_core::dimension::Ix3;
        // Transform only the last axis of a 3-D array; the plan should
        // pre-compute exactly one per-axis 1-D plan.
        let data: Vec<Complex<f64>> = (0..24).map(|i| Complex::new(f64::from(i), 0.0)).collect();
        let a =
            Array::<Complex<f64>, Ix3>::from_vec(ferray_core::dimension::Ix3::new([2, 3, 4]), data)
                .unwrap();

        let plan = FftPlanND::new(&[2, 3, 4], Some(&[-1])).unwrap();
        assert_eq!(plan.axes(), &[2]);
        assert_eq!(plan.num_axis_plans(), 1);
        let spectrum = plan.execute(&a, FftNorm::Backward).unwrap();
        let reference = crate::fftn(&a, None, Some(&[-1][..]), FftNorm::Backward).unwrap();
        for (p, r) in spectrum.iter().zip(reference.iter()) {
            assert!((p.re - r.re).abs() < 1e-12);
            assert!((p.im - r.im).abs() < 1e-12);
        }
    }

    #[test]
    fn plan_nd_shape_mismatch_errors() {
        use ferray_core::dimension::Ix2;
        let plan = FftPlanND::new(&[4, 4], None).unwrap();
        let wrong = Array::<Complex<f64>, Ix2>::from_vec(
            ferray_core::Ix2::new([3, 4]),
            vec![Complex::new(0.0, 0.0); 12],
        )
        .unwrap();
        let err = plan.execute(&wrong, FftNorm::Backward).unwrap_err();
        assert!(err.to_string().contains("FftPlanND was built for shape"));
    }

    #[test]
    fn plan_nd_rejects_zero_dimension_and_empty_shape() {
        assert!(FftPlanND::new(&[], None).is_err());
        assert!(FftPlanND::new(&[4, 0], None).is_err());
    }

    #[test]
    fn plan_nd_axis_accessors_expose_cached_plans() {
        let plan = FftPlanND::new(&[8, 16], None).unwrap();
        assert_eq!(plan.num_axis_plans(), 2);
        assert!(plan.forward_axis_plan(0).is_some());
        assert!(plan.inverse_axis_plan(1).is_some());
        assert!(plan.forward_axis_plan(2).is_none());
        // The forward plan returned should be Arc-equal to a fresh
        // cache lookup for the same size.
        let cached = get_cached_plan_f64(8, false);
        assert!(Arc::ptr_eq(plan.forward_axis_plan(0).unwrap(), &cached));
    }

    #[test]
    fn plan_nd_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<FftPlanND>();
    }
}
