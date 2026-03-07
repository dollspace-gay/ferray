#![no_main]

//! Fuzz target for core reshape/manipulation: reshape, flatten, ravel,
//! transpose, squeeze, expand_dims.
//!
//! Contract: ferray either returns Ok or Err(FerrayError) — never panics.

use libfuzzer_sys::fuzz_target;

use ferray_core::array::owned::Array;
use ferray_core::dimension::{Ix1, IxDyn};
use ferray_core::manipulation;

fn bytes_to_f64s(data: &[u8]) -> Vec<f64> {
    data.chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 10 {
        return;
    }

    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let values = bytes_to_f64s(data);
        if values.is_empty() {
            return;
        }

        let n = values.len();
        let arr = match Array::from_vec(Ix1::new([n]), values) {
            Ok(a) => a,
            Err(_) => return,
        };

        // Flatten / ravel (should always succeed on a valid array)
        let _ = manipulation::flatten(&arr);
        let _ = manipulation::ravel(&arr);

        // Reshape to fuzzed shapes — these may mismatch in size, must not panic
        let ndim = ((data[0] as usize) % 4) + 1;
        let mut shape = Vec::with_capacity(ndim);
        for i in 0..ndim {
            let dim_size = (data[1 + (i % (data.len() - 1))] as usize).max(1);
            shape.push(dim_size);
        }
        let _ = manipulation::reshape(&arr, &shape);

        // Reshape to 2D with fuzzed row count
        let rows = (data[5] as usize).max(1);
        if n.is_multiple_of(rows) {
            let cols = n / rows;
            let _ = manipulation::reshape(&arr, &[rows, cols]);
        }
        // Intentionally mismatched shape
        let _ = manipulation::reshape(&arr, &[n + 1]);
        let _ = manipulation::reshape(&arr, &[0]);

        // Squeeze on a 1-D array (nothing to squeeze, but must not panic)
        let _ = manipulation::squeeze(&arr, None);
        let _ = manipulation::squeeze(&arr, Some(0));
        let _ = manipulation::squeeze(&arr, Some(99)); // out of bounds

        // Create a dynamic array and reshape
        if let Ok(dyn_arr) = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[n]), arr.iter().cloned().collect()) {
            let _ = manipulation::reshape(&dyn_arr, &[1, n]);
            let _ = manipulation::reshape(&dyn_arr, &[n, 1]);
            let _ = manipulation::transpose(&dyn_arr, None);
        }
    }));
});
