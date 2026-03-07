#![no_main]

//! Fuzz target for rounding ufuncs: round, floor, ceil, trunc, rint, fix.
//!
//! Contract: ferray either returns Ok or Err(FerrayError) — never panics.

use libfuzzer_sys::fuzz_target;

use ferray_core::array::owned::Array;
use ferray_core::dimension::Ix1;

fn bytes_to_f64s(data: &[u8]) -> Vec<f64> {
    data.chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fuzz_target!(|data: &[u8]| {
    let values = bytes_to_f64s(data);
    if values.is_empty() {
        return;
    }

    let n = values.len();
    let arr = match Array::from_vec(Ix1::new([n]), values) {
        Ok(a) => a,
        Err(_) => return,
    };

    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = ferray_ufunc::round(&arr);
        let _ = ferray_ufunc::floor(&arr);
        let _ = ferray_ufunc::ceil(&arr);
        let _ = ferray_ufunc::trunc(&arr);
        let _ = ferray_ufunc::rint(&arr);
        let _ = ferray_ufunc::fix(&arr);
    }));
});
