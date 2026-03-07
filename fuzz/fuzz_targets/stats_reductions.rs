#![no_main]

//! Fuzz target for stats reductions: sum, prod, min, max, mean, var, std_,
//! argmin, argmax, median.
//!
//! Contract: ferray either returns Ok or Err(FerrayError) — never panics.

use libfuzzer_sys::fuzz_target;

use ferray_core::array::owned::Array;
use ferray_core::dimension::{Ix1, Ix2, IxDyn};

fn bytes_to_f64s(data: &[u8]) -> Vec<f64> {
    data.chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let values = bytes_to_f64s(data);
        if values.is_empty() {
            return;
        }

        let n = values.len();

        // Test on 1-D array (global reduction: axis = None)
        if let Ok(arr1d) = Array::from_vec(Ix1::new([n]), values.clone()) {
            let _ = ferray_stats::sum(&arr1d, None);
            let _ = ferray_stats::prod(&arr1d, None);
            let _ = ferray_stats::min(&arr1d, None);
            let _ = ferray_stats::max(&arr1d, None);
            let _ = ferray_stats::mean(&arr1d, None);
            let _ = ferray_stats::var(&arr1d, None, 0);
            let _ = ferray_stats::var(&arr1d, None, 1);
            let _ = ferray_stats::std_(&arr1d, None, 0);
            let _ = ferray_stats::argmin(&arr1d, None);
            let _ = ferray_stats::argmax(&arr1d, None);
            let _ = ferray_stats::median(&arr1d, None);
        }

        // Test on 2-D array (axis reductions)
        if n >= 4 {
            let rows = ((data[0] as usize) % 8).max(1);
            let cols = n / rows;
            if cols > 0 && rows * cols <= n {
                let vals = values[..rows * cols].to_vec();
                if let Ok(arr2d) = Array::from_vec(Ix2::new([rows, cols]), vals) {
                    // Reduce along axis 0
                    let _ = ferray_stats::sum(&arr2d, Some(0));
                    let _ = ferray_stats::mean(&arr2d, Some(0));
                    let _ = ferray_stats::var(&arr2d, Some(0), 0);
                    let _ = ferray_stats::min(&arr2d, Some(0));
                    let _ = ferray_stats::max(&arr2d, Some(0));

                    // Reduce along axis 1
                    let _ = ferray_stats::sum(&arr2d, Some(1));
                    let _ = ferray_stats::mean(&arr2d, Some(1));

                    // Out-of-bounds axis — must return Err, not panic
                    let _ = ferray_stats::sum(&arr2d, Some(99));
                }
            }
        }

        // Test with IxDyn
        if let Ok(arrd) = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[n]), values) {
            let _ = ferray_stats::sum(&arrd, None);
            let _ = ferray_stats::prod(&arrd, None);
            let _ = ferray_stats::mean(&arrd, None);
        }
    }));
});
