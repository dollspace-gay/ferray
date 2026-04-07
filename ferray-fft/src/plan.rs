// ferray-fft: FftPlan type and global plan cache (REQ-12, REQ-13)

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

use num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::{Fft, FftPlanner};

use ferray_core::error::{FerrayError, FerrayResult};
use ferray_core::{Array, Ix1};

use crate::norm::{FftDirection, FftNorm};

/// Key for the global FFT plan cache: (transform size, is_inverse).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CacheKey {
    size: usize,
    inverse: bool,
}

// ---------------------------------------------------------------------------
// Complex FFT plan caches (f32 and f64)
//
// Each precision gets its own LazyLock-wrapped Mutex<PlanCache<T>>. The
// caches are lazy — the f32 cache is only constructed if f32 FFTs are
// actually used, so pure f64 users pay nothing.
// ---------------------------------------------------------------------------

static F64_CACHE: LazyLock<Mutex<PlanCacheF64>> =
    LazyLock::new(|| Mutex::new(PlanCacheF64::new()));
static F32_CACHE: LazyLock<Mutex<PlanCacheF32>> =
    LazyLock::new(|| Mutex::new(PlanCacheF32::new()));

struct PlanCacheF64 {
    planner: FftPlanner<f64>,
    plans: HashMap<CacheKey, Arc<dyn Fft<f64>>>,
}

impl PlanCacheF64 {
    fn new() -> Self {
        Self {
            planner: FftPlanner::new(),
            plans: HashMap::new(),
        }
    }

    fn get_plan(&mut self, size: usize, inverse: bool) -> Arc<dyn Fft<f64>> {
        let key = CacheKey { size, inverse };
        self.plans
            .entry(key)
            .or_insert_with(|| {
                if inverse {
                    self.planner.plan_fft_inverse(size)
                } else {
                    self.planner.plan_fft_forward(size)
                }
            })
            .clone()
    }
}

struct PlanCacheF32 {
    planner: FftPlanner<f32>,
    plans: HashMap<CacheKey, Arc<dyn Fft<f32>>>,
}

impl PlanCacheF32 {
    fn new() -> Self {
        Self {
            planner: FftPlanner::new(),
            plans: HashMap::new(),
        }
    }

    fn get_plan(&mut self, size: usize, inverse: bool) -> Arc<dyn Fft<f32>> {
        let key = CacheKey { size, inverse };
        self.plans
            .entry(key)
            .or_insert_with(|| {
                if inverse {
                    self.planner.plan_fft_inverse(size)
                } else {
                    self.planner.plan_fft_forward(size)
                }
            })
            .clone()
    }
}

/// Obtain a cached f64 FFT plan for the given size and direction.
///
/// This is the primary internal entry point used by all f64 FFT functions.
/// Plans are cached globally so repeated transforms of the same size
/// reuse the same plan.
pub(crate) fn get_cached_plan_f64(size: usize, inverse: bool) -> Arc<dyn Fft<f64>> {
    let mut cache = F64_CACHE.lock().expect("f64 FFT plan cache lock poisoned");
    cache.get_plan(size, inverse)
}

/// Obtain a cached f32 FFT plan for the given size and direction.
pub(crate) fn get_cached_plan_f32(size: usize, inverse: bool) -> Arc<dyn Fft<f32>> {
    let mut cache = F32_CACHE.lock().expect("f32 FFT plan cache lock poisoned");
    cache.get_plan(size, inverse)
}

/// Legacy alias for f64 plan lookup — preserved for tests and any call
/// sites that haven't been generified yet. New code should go through
/// [`crate::float::FftFloat::cached_plan`].
pub(crate) fn get_cached_plan(size: usize, inverse: bool) -> Arc<dyn Fft<f64>> {
    get_cached_plan_f64(size, inverse)
}

// ---------------------------------------------------------------------------
// Real-FFT plan caches (issues #432, #426)
//
// realfft uses a separate planner type from rustfft because real-to-complex
// (RealToComplex<T>) and complex-to-real (ComplexToReal<T>) are distinct
// trait objects. We give each precision its own LazyLock so the complex
// caches are untouched, and the real planners are only constructed if
// rfft/irfft is used.
// ---------------------------------------------------------------------------

static REAL_CACHE_F64: LazyLock<Mutex<RealPlanCacheF64>> =
    LazyLock::new(|| Mutex::new(RealPlanCacheF64::new()));
static REAL_CACHE_F32: LazyLock<Mutex<RealPlanCacheF32>> =
    LazyLock::new(|| Mutex::new(RealPlanCacheF32::new()));

struct RealPlanCacheF64 {
    planner: RealFftPlanner<f64>,
    forward_plans: HashMap<usize, Arc<dyn RealToComplex<f64>>>,
    inverse_plans: HashMap<usize, Arc<dyn ComplexToReal<f64>>>,
}

impl RealPlanCacheF64 {
    fn new() -> Self {
        Self {
            planner: RealFftPlanner::new(),
            forward_plans: HashMap::new(),
            inverse_plans: HashMap::new(),
        }
    }

    fn forward(&mut self, size: usize) -> Arc<dyn RealToComplex<f64>> {
        let planner = &mut self.planner;
        self.forward_plans
            .entry(size)
            .or_insert_with(|| planner.plan_fft_forward(size))
            .clone()
    }

    fn inverse(&mut self, size: usize) -> Arc<dyn ComplexToReal<f64>> {
        let planner = &mut self.planner;
        self.inverse_plans
            .entry(size)
            .or_insert_with(|| planner.plan_fft_inverse(size))
            .clone()
    }
}

struct RealPlanCacheF32 {
    planner: RealFftPlanner<f32>,
    forward_plans: HashMap<usize, Arc<dyn RealToComplex<f32>>>,
    inverse_plans: HashMap<usize, Arc<dyn ComplexToReal<f32>>>,
}

impl RealPlanCacheF32 {
    fn new() -> Self {
        Self {
            planner: RealFftPlanner::new(),
            forward_plans: HashMap::new(),
            inverse_plans: HashMap::new(),
        }
    }

    fn forward(&mut self, size: usize) -> Arc<dyn RealToComplex<f32>> {
        let planner = &mut self.planner;
        self.forward_plans
            .entry(size)
            .or_insert_with(|| planner.plan_fft_forward(size))
            .clone()
    }

    fn inverse(&mut self, size: usize) -> Arc<dyn ComplexToReal<f32>> {
        let planner = &mut self.planner;
        self.inverse_plans
            .entry(size)
            .or_insert_with(|| planner.plan_fft_inverse(size))
            .clone()
    }
}

/// Obtain a cached f64 real-to-complex FFT plan for the given size.
///
/// Returns a plan that consumes a real input of length `size` and
/// produces `size/2 + 1` complex values.
pub(crate) fn get_cached_real_forward_f64(size: usize) -> Arc<dyn RealToComplex<f64>> {
    let mut cache = REAL_CACHE_F64
        .lock()
        .expect("f64 real FFT plan cache lock poisoned");
    cache.forward(size)
}

/// Obtain a cached f64 complex-to-real FFT plan for the given output size.
///
/// The argument `size` is the **real-output** length (not the complex
/// input length, which is `size/2 + 1`).
pub(crate) fn get_cached_real_inverse_f64(size: usize) -> Arc<dyn ComplexToReal<f64>> {
    let mut cache = REAL_CACHE_F64
        .lock()
        .expect("f64 real FFT plan cache lock poisoned");
    cache.inverse(size)
}

/// Obtain a cached f32 real-to-complex FFT plan for the given size.
pub(crate) fn get_cached_real_forward_f32(size: usize) -> Arc<dyn RealToComplex<f32>> {
    let mut cache = REAL_CACHE_F32
        .lock()
        .expect("f32 real FFT plan cache lock poisoned");
    cache.forward(size)
}

/// Obtain a cached f32 complex-to-real FFT plan for the given output size.
pub(crate) fn get_cached_real_inverse_f32(size: usize) -> Arc<dyn ComplexToReal<f32>> {
    let mut cache = REAL_CACHE_F32
        .lock()
        .expect("f32 real FFT plan cache lock poisoned");
    cache.inverse(size)
}

/// Legacy f64 alias for the real-forward cache.
#[allow(dead_code)]
pub(crate) fn get_cached_real_forward(size: usize) -> Arc<dyn RealToComplex<f64>> {
    get_cached_real_forward_f64(size)
}

/// Legacy f64 alias for the real-inverse cache.
#[allow(dead_code)]
pub(crate) fn get_cached_real_inverse(size: usize) -> Arc<dyn ComplexToReal<f64>> {
    get_cached_real_inverse_f64(size)
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
    pub fn size(&self) -> usize {
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
        f.debug_struct("FftPlan").field("size", &self.size).finish()
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
}
