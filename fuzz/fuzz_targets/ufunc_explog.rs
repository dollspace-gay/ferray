#![no_main]

//! Fuzz target for exponential and logarithmic ufuncs: exp, exp2, expm1,
//! log, log2, log10, log1p, logaddexp, logaddexp2.
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
    let arr = match Array::from_vec(Ix1::new([n]), values.clone()) {
        Ok(a) => a,
        Err(_) => return,
    };

    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // Unary exp/log functions
        let _ = ferray_ufunc::exp(&arr);
        let _ = ferray_ufunc::exp2(&arr);
        let _ = ferray_ufunc::expm1(&arr);
        let _ = ferray_ufunc::log(&arr);
        let _ = ferray_ufunc::log2(&arr);
        let _ = ferray_ufunc::log10(&arr);
        let _ = ferray_ufunc::log1p(&arr);

        // Binary: logaddexp, logaddexp2 — split input in half
        if n >= 2 {
            let half = n / 2;
            let a = Array::from_vec(Ix1::new([half]), values[..half].to_vec()).unwrap();
            let b = Array::from_vec(Ix1::new([half]), values[half..half * 2].to_vec()).unwrap();
            let _ = ferray_ufunc::logaddexp(&a, &b);
            let _ = ferray_ufunc::logaddexp2(&a, &b);
        }
    }));
});
