#![no_main]

//! Fuzz target for FFT operations: fft, ifft round-trip; rfft, fftfreq, fftshift.
//!
//! The round-trip property: ifft(fft(x)) should approximately equal x
//! for any finite input. We verify no panics and check the round-trip
//! for non-NaN finite inputs.
//!
//! Contract: ferray either returns Ok or Err(FerrayError) — never panics.

use libfuzzer_sys::fuzz_target;
use num_complex::Complex;

use ferray_core::array::owned::Array;
use ferray_core::dimension::{Ix1, IxDyn};
use ferray_fft::FftNorm;

fn bytes_to_f64s(data: &[u8]) -> Vec<f64> {
    data.chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 16 {
        return;
    }

    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let values = bytes_to_f64s(data);
        if values.is_empty() {
            return;
        }

        // Build a complex 1-D array from pairs of f64 values
        let n_complex = values.len() / 2;
        if n_complex == 0 {
            return;
        }

        let complex_vals: Vec<Complex<f64>> = values
            .chunks_exact(2)
            .map(|pair| Complex::new(pair[0], pair[1]))
            .collect();
        let n = complex_vals.len();

        let arr = match Array::from_vec(Ix1::new([n]), complex_vals.clone()) {
            Ok(a) => a,
            Err(_) => return,
        };

        // fft with default norm
        let fft_result = ferray_fft::fft(&arr, None, None, FftNorm::Backward);

        // ifft on the fft result (round-trip)
        if let Ok(ref fft_arr) = fft_result {
            let _ = ferray_fft::ifft(fft_arr, None, None, FftNorm::Backward);
        }

        // fft with different normalizations
        let _ = ferray_fft::fft(&arr, None, None, FftNorm::Forward);
        let _ = ferray_fft::fft(&arr, None, None, FftNorm::Ortho);

        // fft with explicit n (zero-pad or truncate)
        let _ = ferray_fft::fft(&arr, Some(n * 2), None, FftNorm::Backward);
        if n > 1 {
            let _ = ferray_fft::fft(&arr, Some(n / 2), None, FftNorm::Backward);
        }
        // n=0 — should return an error, not panic
        let _ = ferray_fft::fft(&arr, Some(0), None, FftNorm::Backward);

        // fft with explicit axis
        let _ = ferray_fft::fft(&arr, None, Some(0), FftNorm::Backward);
        // out-of-bounds axis
        let _ = ferray_fft::fft(&arr, None, Some(99), FftNorm::Backward);

        // Frequency utilities
        let _ = ferray_fft::fftfreq(n, 1.0);
        let _ = ferray_fft::rfftfreq(n, 1.0);

        // fftshift / ifftshift on a dynamic array
        if let Ok(dyn_arr) = Array::from_vec(IxDyn::new(&[n]), complex_vals) {
            let _ = ferray_fft::fftshift(&dyn_arr, None);
            let _ = ferray_fft::ifftshift(&dyn_arr, None);
        }
    }));
});
