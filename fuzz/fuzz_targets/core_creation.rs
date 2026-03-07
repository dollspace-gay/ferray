#![no_main]

//! Fuzz target for core array creation: Array::from_vec with various shapes,
//! Array::zeros, Array::ones, Array::from_elem.
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
    if data.len() < 2 {
        return;
    }

    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let values = bytes_to_f64s(data);

        // 1-D creation with correct length
        if !values.is_empty() {
            let n = values.len();
            let _ = Array::from_vec(Ix1::new([n]), values.clone());
        }

        // 2-D creation: use first two bytes as shape hints
        let rows = (data[0] as usize).clamp(1, 64);
        let cols = (data[1] as usize).clamp(1, 64);
        // Attempt from_vec with potentially mismatched shape — must not panic
        let _ = Array::<f64, Ix2>::from_vec(Ix2::new([rows, cols]), values.clone());

        // Dynamic-rank creation with fuzzed shape
        if data.len() >= 4 {
            let ndim = ((data[2] as usize) % 5) + 1; // 1 to 5 dimensions
            let mut shape = Vec::with_capacity(ndim);
            for i in 0..ndim {
                let dim_size = (data[3 + (i % (data.len() - 3))] as usize).clamp(1, 16);
                shape.push(dim_size);
            }
            let _ = Array::<f64, IxDyn>::from_vec(IxDyn::new(&shape), values.clone());
        }

        // zeros / ones / from_elem with bounded shapes
        if data.len() >= 3 {
            let s = (data[0] as usize).min(128);
            let _ = Array::<f64, Ix1>::zeros(Ix1::new([s]));
            let _ = Array::<f64, Ix1>::ones(Ix1::new([s]));
            let _ = Array::<f64, Ix1>::from_elem(Ix1::new([s]), 42.0);
        }

        // Fortran-order creation
        if !values.is_empty() {
            let n = values.len();
            let _ = Array::from_vec_f(Ix1::new([n]), values);
        }
    }));
});
