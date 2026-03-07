#![no_main]

//! Fuzz target for linalg operations: dot, matmul, solve, inv, det, norm.
//!
//! All linalg functions take f64 arrays. We construct small matrices from
//! fuzzed bytes and verify no panics occur.
//!
//! Contract: ferray either returns Ok or Err(FerrayError) — never panics.

use libfuzzer_sys::fuzz_target;

use ferray_core::array::owned::Array;
use ferray_core::dimension::{Ix2, IxDyn};

fn bytes_to_f64s(data: &[u8]) -> Vec<f64> {
    data.chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 24 {
        return;
    }

    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let values = bytes_to_f64s(data);
        if values.len() < 4 {
            return;
        }

        // Build small matrices for linalg ops.
        // Use first byte to determine matrix size (bounded to keep computation fast).
        let dim = ((data[0] as usize) % 6).max(1); // 1x1 to 6x6
        let needed = dim * dim;

        if values.len() < needed {
            return;
        }

        // Square matrix A
        let a_vals: Vec<f64> = values[..needed].to_vec();
        let a_ix2 = match Array::from_vec(Ix2::new([dim, dim]), a_vals.clone()) {
            Ok(a) => a,
            Err(_) => return,
        };
        let a_dyn = match Array::from_vec(IxDyn::new(&[dim, dim]), a_vals) {
            Ok(a) => a,
            Err(_) => return,
        };

        // matmul(A, A)
        let _ = ferray_linalg::matmul(&a_dyn, &a_dyn);

        // dot(A, A)
        let _ = ferray_linalg::dot(&a_dyn, &a_dyn);

        // inv(A)
        let _ = ferray_linalg::inv(&a_ix2);

        // solve(A, b) — b is a 1-D column vector promoted to IxDyn
        if values.len() >= needed + dim {
            let b_vals: Vec<f64> = values[needed..needed + dim].to_vec();
            if let Ok(b_dyn) = Array::from_vec(IxDyn::new(&[dim]), b_vals) {
                let _ = ferray_linalg::solve(&a_ix2, &b_dyn);
            }
        }

        // det(A)
        let _ = ferray_linalg::det(&a_ix2);

        // norm(A) — Frobenius norm
        let _ = ferray_linalg::norm(&a_dyn, ferray_linalg::NormOrder::Fro);

        // trace(A)
        let _ = ferray_linalg::trace(&a_ix2);

        // cholesky — may fail for non-PD matrices, must not panic
        let _ = ferray_linalg::cholesky(&a_ix2);

        // svd
        let _ = ferray_linalg::svd(&a_ix2, true);

        // 1-D dot product
        if values.len() >= 2 * dim {
            let v1 = values[..dim].to_vec();
            let v2 = values[dim..2 * dim].to_vec();
            if let (Ok(v1d), Ok(v2d)) = (
                Array::from_vec(IxDyn::new(&[dim]), v1),
                Array::from_vec(IxDyn::new(&[dim]), v2),
            ) {
                let _ = ferray_linalg::dot(&v1d, &v2d);
            }
        }

        // Non-square matmul: A(dim x dim) * B(dim x k) where k is fuzzed
        let k = ((data[1] as usize) % 6).max(1);
        let needed_b = dim * k;
        if values.len() >= needed + needed_b {
            let b_vals = values[needed..needed + needed_b].to_vec();
            if let Ok(b_rect) = Array::from_vec(IxDyn::new(&[dim, k]), b_vals) {
                let _ = ferray_linalg::matmul(&a_dyn, &b_rect);
            }
        }
    }));
});
