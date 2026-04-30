// End-to-end tests for the integer GEMM entry points: gemm_i16,
// gemm_i8_signed, and quantized_matmul. Each is checked against a
// naive scalar reference at sizes that exercise both the kernel
// path (m, n, k all >= MR/NR/KB) and the edge-tile + K-tail handling.
//
// Plus mixed-precision tests for gemm_bf16_f32 / gemm_f16_f32 (when
// the corresponding cargo feature is on).

#[cfg(feature = "bf16")]
use super::gemm_bf16_f32;
#[cfg(feature = "f16")]
use super::gemm_f16_f32;
use super::{gemm_i8_signed, gemm_i16, quantized_matmul};

fn naive_i16_matmul(m: usize, n: usize, k: usize, a: &[i16], b: &[i16], c: &mut [i32]) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0_i32;
            for p in 0..k {
                acc = acc.wrapping_add((a[i * k + p] as i32) * (b[p * n + j] as i32));
            }
            c[i * n + j] = acc;
        }
    }
}

fn naive_i8_signed_matmul(m: usize, n: usize, k: usize, a: &[i8], b: &[i8], c: &mut [i32]) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0_i32;
            for p in 0..k {
                acc = acc.wrapping_add((a[i * k + p] as i32) * (b[p * n + j] as i32));
            }
            c[i * n + j] = acc;
        }
    }
}

fn naive_quantized(
    m: usize,
    n: usize,
    k: usize,
    a: &[u8],
    a_zp: u8,
    scale_a: f32,
    b: &[i8],
    b_zp: i8,
    scale_b: f32,
    c: &mut [f32],
) {
    let scale = scale_a * scale_b;
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0_i32;
            for p in 0..k {
                let av = (a[i * k + p] as i32) - (a_zp as i32);
                let bv = (b[p * n + j] as i32) - (b_zp as i32);
                acc = acc.wrapping_add(av.wrapping_mul(bv));
            }
            c[i * n + j] = scale * (acc as f32);
        }
    }
}

#[test]
fn gemm_i16_matches_naive_at_kernel_sizes() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let m = 12;
    let n = 24;
    let k = 16;
    let a: Vec<i16> = (0..m * k).map(|i| ((i as i32) * 17 - 200) as i16).collect();
    let b: Vec<i16> = (0..k * n).map(|i| ((i as i32) * 13 - 100) as i16).collect();
    let mut c_ours = vec![0_i32; m * n];
    let mut c_ref = vec![0_i32; m * n];

    let ok = unsafe { gemm_i16(m, n, k, a.as_ptr(), b.as_ptr(), c_ours.as_mut_ptr(), false) };
    assert!(ok);
    naive_i16_matmul(m, n, k, &a, &b, &mut c_ref);

    for i in 0..m * n {
        assert_eq!(c_ours[i], c_ref[i], "mismatch at idx {i}");
    }
}

#[test]
fn gemm_i16_handles_odd_dims_and_k() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    // m, n, k all NOT multiples of MR/NR/KB to hit edge tiles + K-tail.
    let m = 7;
    let n = 11;
    let k = 9;
    let a: Vec<i16> = (0..m * k).map(|i| ((i as i32) * 23 + 5) as i16).collect();
    let b: Vec<i16> = (0..k * n).map(|i| ((i as i32) * 19 - 60) as i16).collect();
    let mut c_ours = vec![0_i32; m * n];
    let mut c_ref = vec![0_i32; m * n];

    let ok = unsafe { gemm_i16(m, n, k, a.as_ptr(), b.as_ptr(), c_ours.as_mut_ptr(), false) };
    assert!(ok);
    naive_i16_matmul(m, n, k, &a, &b, &mut c_ref);

    for i in 0..m * n {
        assert_eq!(c_ours[i], c_ref[i], "edge mismatch at idx {i}");
    }
}

#[test]
fn gemm_i16_accumulate_preserves_existing_c() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let m = 8;
    let n = 8;
    let k = 8;
    let a: Vec<i16> = (0..m * k).map(|i| (i as i16) * 2).collect();
    let b: Vec<i16> = (0..k * n).map(|i| (i as i16) * 3).collect();
    let mut c_ours: Vec<i32> = (0..m * n).map(|i| (i as i32) * 1000).collect();
    let mut c_ref = c_ours.clone();

    let ok = unsafe { gemm_i16(m, n, k, a.as_ptr(), b.as_ptr(), c_ours.as_mut_ptr(), true) };
    assert!(ok);
    // naive ref: c[i] += sum...
    let mut delta = vec![0_i32; m * n];
    naive_i16_matmul(m, n, k, &a, &b, &mut delta);
    for i in 0..m * n {
        c_ref[i] = c_ref[i].wrapping_add(delta[i]);
    }

    for i in 0..m * n {
        assert_eq!(c_ours[i], c_ref[i], "accumulate mismatch at {i}");
    }
}

#[test]
fn gemm_i8_signed_matches_naive() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let m = 12;
    let n = 16;
    let k = 20;
    // i8 inputs in full signed range (incl. negatives).
    let a: Vec<i8> = (0..m * k).map(|i| ((i as i32) * 7 - 60) as i8).collect();
    let b: Vec<i8> = (0..k * n).map(|i| ((i as i32) * 11 - 80) as i8).collect();
    let mut c_ours = vec![0_i32; m * n];
    let mut c_ref = vec![0_i32; m * n];

    let ok = unsafe { gemm_i8_signed(m, n, k, a.as_ptr(), b.as_ptr(), c_ours.as_mut_ptr(), false) };
    assert!(ok);
    naive_i8_signed_matmul(m, n, k, &a, &b, &mut c_ref);

    for i in 0..m * n {
        assert_eq!(c_ours[i], c_ref[i], "i8-signed mismatch at idx {i}");
    }
}

#[test]
fn gemm_i8_signed_extreme_values_through_entry() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    // i8::MIN (-128) and i8::MAX (127) sprinkled at non-multiple dims
    // exercise the dedicated kernel's sign-extension path through the
    // full top-level entry (pack + parallel + edge tiles).
    let m = 13;
    let n = 17;
    let k = 19;
    let a: Vec<i8> = (0..m * k)
        .map(|i| match i % 6 {
            0 => i8::MIN,
            1 => i8::MAX,
            2 => -127,
            3 => 1,
            4 => -1,
            _ => ((i as i32) * 5 - 80) as i8,
        })
        .collect();
    let b: Vec<i8> = (0..k * n)
        .map(|i| match i % 5 {
            0 => i8::MAX,
            1 => i8::MIN,
            2 => 0,
            _ => ((i as i32) * 3 - 50) as i8,
        })
        .collect();
    let mut c_ours = vec![0_i32; m * n];
    let mut c_ref = vec![0_i32; m * n];

    let ok = unsafe { gemm_i8_signed(m, n, k, a.as_ptr(), b.as_ptr(), c_ours.as_mut_ptr(), false) };
    assert!(ok);
    naive_i8_signed_matmul(m, n, k, &a, &b, &mut c_ref);

    for i in 0..m * n {
        assert_eq!(c_ours[i], c_ref[i], "extreme-value mismatch at idx {i}");
    }
}

#[test]
fn gemm_i8_signed_accumulate_through_entry() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let m = 8;
    let n = 8;
    let k = 12;
    let a: Vec<i8> = (0..m * k).map(|i| ((i as i32) * 3 - 30) as i8).collect();
    let b: Vec<i8> = (0..k * n).map(|i| ((i as i32) * 5 - 40) as i8).collect();
    let mut c_ours: Vec<i32> = (0..m * n).map(|i| (i as i32) * 100).collect();
    let mut c_ref = c_ours.clone();

    let ok = unsafe { gemm_i8_signed(m, n, k, a.as_ptr(), b.as_ptr(), c_ours.as_mut_ptr(), true) };
    assert!(ok);
    let mut delta = vec![0_i32; m * n];
    naive_i8_signed_matmul(m, n, k, &a, &b, &mut delta);
    for i in 0..m * n {
        c_ref[i] = c_ref[i].wrapping_add(delta[i]);
    }
    for i in 0..m * n {
        assert_eq!(c_ours[i], c_ref[i], "i8s accumulate mismatch at {i}");
    }
}

#[test]
fn gemm_i8_signed_negative_inputs() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let m = 9;
    let n = 13;
    let k = 11;
    // Mix of positive and negative i8 values.
    let a: Vec<i8> = (0..m * k)
        .map(|i| {
            if i % 2 == 0 {
                -((i % 100) as i8)
            } else {
                (i % 100) as i8
            }
        })
        .collect();
    let b: Vec<i8> = (0..k * n)
        .map(|i| {
            if i % 3 == 0 {
                -((i % 80) as i8)
            } else {
                (i % 80) as i8
            }
        })
        .collect();
    let mut c_ours = vec![0_i32; m * n];
    let mut c_ref = vec![0_i32; m * n];

    let ok = unsafe { gemm_i8_signed(m, n, k, a.as_ptr(), b.as_ptr(), c_ours.as_mut_ptr(), false) };
    assert!(ok);
    naive_i8_signed_matmul(m, n, k, &a, &b, &mut c_ref);

    for i in 0..m * n {
        assert_eq!(c_ours[i], c_ref[i], "negative-input mismatch at {i}");
    }
}

#[test]
fn quantized_matmul_zero_zero_points_matches_scaled_gemm_i8() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    // With a_zp = b_zp = 0 the formula reduces to scale * sum(a*b).
    let m = 8;
    let n = 12;
    let k = 16;
    let a: Vec<u8> = (0..m * k).map(|i| ((i * 7 + 3) as u8) & 0x7f).collect();
    let b: Vec<i8> = (0..k * n).map(|i| ((i as i32) * 11 - 50) as i8).collect();
    let scale_a = 0.01_f32;
    let scale_b = 0.02_f32;
    let mut c_ours = vec![0.0_f32; m * n];
    let mut c_ref = vec![0.0_f32; m * n];

    let ok = unsafe {
        quantized_matmul(
            m,
            n,
            k,
            a.as_ptr(),
            0,
            scale_a,
            b.as_ptr(),
            0,
            scale_b,
            c_ours.as_mut_ptr(),
            false,
        )
    };
    assert!(ok);
    naive_quantized(m, n, k, &a, 0, scale_a, &b, 0, scale_b, &mut c_ref);

    for i in 0..m * n {
        // f32 tolerance — scale * i32 may have small rounding.
        let diff = (c_ours[i] - c_ref[i]).abs();
        let scale = c_ref[i].abs().max(1.0);
        assert!(
            diff / scale < 1e-5,
            "qmm zero-zp mismatch at {i}: ours={} ref={}",
            c_ours[i],
            c_ref[i]
        );
    }
}

#[test]
fn quantized_matmul_with_nonzero_zero_points() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let m = 10;
    let n = 14;
    let k = 18;
    // Constrain u8 to [0, 127] — the AVX2 vpmaddubsw chain saturates i16
    // intermediates when full u8 × i8 products of consistent sign sum
    // beyond ±32767. Production asymmetric-quant flows either limit u8
    // to 7-bit on AVX2-only hardware or dispatch to vpdpbusd (AVX-VNNI/
    // AVX-512 VNNI) which accumulates directly into i32 with no
    // saturation. See the doc comment on `gemm_i8`.
    let a: Vec<u8> = (0..m * k).map(|i| ((i * 13) as u8) & 0x7f).collect();
    let b: Vec<i8> = (0..k * n).map(|i| ((i as i32) * 7 - 40) as i8).collect();
    let a_zp: u8 = 64; // mid-point for the 7-bit u8 sub-range
    let b_zp: i8 = -3;
    let scale_a = 0.0078_f32;
    let scale_b = 0.015_f32;
    let mut c_ours = vec![0.0_f32; m * n];
    let mut c_ref = vec![0.0_f32; m * n];

    let ok = unsafe {
        quantized_matmul(
            m,
            n,
            k,
            a.as_ptr(),
            a_zp,
            scale_a,
            b.as_ptr(),
            b_zp,
            scale_b,
            c_ours.as_mut_ptr(),
            false,
        )
    };
    assert!(ok);
    naive_quantized(m, n, k, &a, a_zp, scale_a, &b, b_zp, scale_b, &mut c_ref);

    for i in 0..m * n {
        let diff = (c_ours[i] - c_ref[i]).abs();
        let scale = c_ref[i].abs().max(1.0);
        assert!(
            diff / scale < 1e-4,
            "qmm asymmetric mismatch at {i}: ours={} ref={}",
            c_ours[i],
            c_ref[i]
        );
    }
}

#[test]
fn quantized_matmul_accumulate() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let m = 6;
    let n = 10;
    let k = 12;
    let a: Vec<u8> = (0..m * k).map(|i| ((i * 5) as u8) & 0x3f).collect();
    let b: Vec<i8> = (0..k * n).map(|i| ((i as i32) - 30) as i8).collect();
    let a_zp: u8 = 64;
    let b_zp: i8 = 0;
    let scale_a = 0.05_f32;
    let scale_b = 0.05_f32;
    let mut c_ours: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.1).collect();
    let mut c_ref = c_ours.clone();

    let ok = unsafe {
        quantized_matmul(
            m,
            n,
            k,
            a.as_ptr(),
            a_zp,
            scale_a,
            b.as_ptr(),
            b_zp,
            scale_b,
            c_ours.as_mut_ptr(),
            true,
        )
    };
    assert!(ok);
    let mut delta = vec![0.0_f32; m * n];
    naive_quantized(m, n, k, &a, a_zp, scale_a, &b, b_zp, scale_b, &mut delta);
    for i in 0..m * n {
        c_ref[i] += delta[i];
    }

    for i in 0..m * n {
        let diff = (c_ours[i] - c_ref[i]).abs();
        let scale = c_ref[i].abs().max(1.0);
        assert!(
            diff / scale < 1e-4,
            "qmm accumulate mismatch at {i}: ours={} ref={}",
            c_ours[i],
            c_ref[i]
        );
    }
}

#[cfg(feature = "bf16")]
fn naive_bf16_matmul(
    m: usize,
    n: usize,
    k: usize,
    a: &[half::bf16],
    b: &[half::bf16],
    c: &mut [f32],
) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for p in 0..k {
                acc += a[i * k + p].to_f32() * b[p * n + j].to_f32();
            }
            c[i * n + j] = acc;
        }
    }
}

#[cfg(feature = "bf16")]
#[test]
fn gemm_bf16_f32_matches_naive() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let m = 12;
    let n = 24;
    let k = 16;
    let a: Vec<half::bf16> = (0..m * k)
        .map(|i| half::bf16::from_f32((i as f32) * 0.013 - 0.5))
        .collect();
    let b: Vec<half::bf16> = (0..k * n)
        .map(|i| half::bf16::from_f32((i as f32) * 0.011 - 0.4))
        .collect();
    let mut c_ours = vec![0.0_f32; m * n];
    let mut c_ref = vec![0.0_f32; m * n];

    let ok = unsafe { gemm_bf16_f32(m, n, k, a.as_ptr(), b.as_ptr(), c_ours.as_mut_ptr(), false) };
    assert!(ok);
    naive_bf16_matmul(m, n, k, &a, &b, &mut c_ref);

    for i in 0..m * n {
        let diff = (c_ours[i] - c_ref[i]).abs();
        let scale = c_ref[i].abs().max(1.0);
        assert!(
            diff / scale < 1e-5,
            "bf16-f32 mismatch at {i}: ours={} ref={}",
            c_ours[i],
            c_ref[i]
        );
    }
}

#[cfg(feature = "bf16")]
#[test]
fn gemm_bf16_f32_odd_dims() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let m = 9;
    let n = 13;
    let k = 11;
    let a: Vec<half::bf16> = (0..m * k)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.7).sin()))
        .collect();
    let b: Vec<half::bf16> = (0..k * n)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.5).cos()))
        .collect();
    let mut c_ours = vec![0.0_f32; m * n];
    let mut c_ref = vec![0.0_f32; m * n];

    let ok = unsafe { gemm_bf16_f32(m, n, k, a.as_ptr(), b.as_ptr(), c_ours.as_mut_ptr(), false) };
    assert!(ok);
    naive_bf16_matmul(m, n, k, &a, &b, &mut c_ref);

    for i in 0..m * n {
        let diff = (c_ours[i] - c_ref[i]).abs();
        let scale = c_ref[i].abs().max(1.0);
        assert!(diff / scale < 1e-5, "bf16-f32 odd-dim mismatch at {i}");
    }
}

#[cfg(feature = "bf16")]
#[test]
fn gemm_bf16_f32_accumulate() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let m = 8;
    let n = 8;
    let k = 12;
    let a: Vec<half::bf16> = (0..m * k)
        .map(|i| half::bf16::from_f32((i as f32) * 0.01))
        .collect();
    let b: Vec<half::bf16> = (0..k * n)
        .map(|i| half::bf16::from_f32((i as f32) * 0.02))
        .collect();
    let mut c_ours: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.5).collect();
    let mut c_ref = c_ours.clone();

    let ok = unsafe { gemm_bf16_f32(m, n, k, a.as_ptr(), b.as_ptr(), c_ours.as_mut_ptr(), true) };
    assert!(ok);
    let mut delta = vec![0.0_f32; m * n];
    naive_bf16_matmul(m, n, k, &a, &b, &mut delta);
    for i in 0..m * n {
        c_ref[i] += delta[i];
    }
    for i in 0..m * n {
        let diff = (c_ours[i] - c_ref[i]).abs();
        let scale = c_ref[i].abs().max(1.0);
        assert!(diff / scale < 1e-5, "bf16-f32 accumulate mismatch at {i}");
    }
}

#[cfg(feature = "f16")]
fn naive_f16_matmul(m: usize, n: usize, k: usize, a: &[half::f16], b: &[half::f16], c: &mut [f32]) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for p in 0..k {
                acc += a[i * k + p].to_f32() * b[p * n + j].to_f32();
            }
            c[i * n + j] = acc;
        }
    }
}

#[cfg(feature = "f16")]
#[test]
fn gemm_f16_f32_matches_naive() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let m = 12;
    let n = 24;
    let k = 16;
    let a: Vec<half::f16> = (0..m * k)
        .map(|i| half::f16::from_f32((i as f32) * 0.013 - 0.5))
        .collect();
    let b: Vec<half::f16> = (0..k * n)
        .map(|i| half::f16::from_f32((i as f32) * 0.011 - 0.4))
        .collect();
    let mut c_ours = vec![0.0_f32; m * n];
    let mut c_ref = vec![0.0_f32; m * n];

    let ok = unsafe { gemm_f16_f32(m, n, k, a.as_ptr(), b.as_ptr(), c_ours.as_mut_ptr(), false) };
    assert!(ok);
    naive_f16_matmul(m, n, k, &a, &b, &mut c_ref);

    for i in 0..m * n {
        let diff = (c_ours[i] - c_ref[i]).abs();
        let scale = c_ref[i].abs().max(1.0);
        assert!(
            diff / scale < 1e-5,
            "f16-f32 mismatch at {i}: ours={} ref={}",
            c_ours[i],
            c_ref[i]
        );
    }
}

#[cfg(feature = "f16")]
#[test]
fn gemm_f16_f32_accumulate() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let m = 8;
    let n = 8;
    let k = 12;
    let a: Vec<half::f16> = (0..m * k)
        .map(|i| half::f16::from_f32((i as f32) * 0.01))
        .collect();
    let b: Vec<half::f16> = (0..k * n)
        .map(|i| half::f16::from_f32((i as f32) * 0.02))
        .collect();
    let mut c_ours: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.5).collect();
    let mut c_ref = c_ours.clone();

    let ok = unsafe { gemm_f16_f32(m, n, k, a.as_ptr(), b.as_ptr(), c_ours.as_mut_ptr(), true) };
    assert!(ok);
    let mut delta = vec![0.0_f32; m * n];
    naive_f16_matmul(m, n, k, &a, &b, &mut delta);
    for i in 0..m * n {
        c_ref[i] += delta[i];
    }
    for i in 0..m * n {
        let diff = (c_ours[i] - c_ref[i]).abs();
        let scale = c_ref[i].abs().max(1.0);
        assert!(diff / scale < 1e-5, "f16-f32 accumulate mismatch at {i}");
    }
}
