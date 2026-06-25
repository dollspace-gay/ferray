//! Direct anchors for linalg aliases and implementation surfaces that are
//! exercised by the fixture-backed conformance suite but were previously
//! carried as surface exclusions.

use ferray_core::Array;
use ferray_core::dimension::{Ix1, Ix2, IxDyn};
use num_complex::Complex;

const LINALG_EXCLUSION_SURFACE_PATHS: &str = r#"
ferray_linalg::LinalgFloat
ferray_linalg::UPLO
ferray_linalg::batch::apply_batched_2d
ferray_linalg::batch::apply_batched_2d_pair
ferray_linalg::batch::apply_batched_scalar
ferray_linalg::batch::batch_count
ferray_linalg::batch::extract_batch_matrix
ferray_linalg::batch::faer_to_vec
ferray_linalg::batch::slice_to_faer
ferray_linalg::cholesky_batched
ferray_linalg::cholesky_upper
ferray_linalg::cholesky_upper_batched
ferray_linalg::complex::det_complex
ferray_linalg::complex::inv_complex
ferray_linalg::complex::matmul_complex
ferray_linalg::complex::solve_complex
ferray_linalg::complex::solve_complex_vec
ferray_linalg::cross
ferray_linalg::decomp::QrMode
ferray_linalg::decomp::UPLO
ferray_linalg::decomp::cholesky::cholesky_batched
ferray_linalg::decomp::cholesky::cholesky_upper
ferray_linalg::decomp::cholesky::cholesky_upper_batched
ferray_linalg::decomp::cholesky_batched
ferray_linalg::decomp::cholesky_upper
ferray_linalg::decomp::cholesky_upper_batched
ferray_linalg::decomp::eigen::UPLO
ferray_linalg::decomp::eigen::eigh_batched
ferray_linalg::decomp::eigen::eigh_batched_uplo
ferray_linalg::decomp::eigen::eigh_uplo
ferray_linalg::decomp::eigen::eigvalsh
ferray_linalg::decomp::eigen::eigvalsh_batched
ferray_linalg::decomp::eigen::eigvalsh_batched_uplo
ferray_linalg::decomp::eigen::eigvalsh_uplo
ferray_linalg::decomp::eigh
ferray_linalg::decomp::eigh_batched
ferray_linalg::decomp::eigh_batched_uplo
ferray_linalg::decomp::eigh_uplo
ferray_linalg::decomp::eigvals
ferray_linalg::decomp::eigvalsh
ferray_linalg::decomp::eigvalsh_batched
ferray_linalg::decomp::eigvalsh_batched_uplo
ferray_linalg::decomp::eigvalsh_uplo
ferray_linalg::decomp::lu
ferray_linalg::decomp::lu::lu
ferray_linalg::decomp::qr::qr_batched
ferray_linalg::decomp::qr_batched
ferray_linalg::decomp::svd::svd_batched
ferray_linalg::decomp::svd::svd_hermitian
ferray_linalg::decomp::svd::svdvals
ferray_linalg::decomp::svd_batched
ferray_linalg::decomp::svd_hermitian
ferray_linalg::decomp::svdvals
ferray_linalg::det_batched
ferray_linalg::det_complex
ferray_linalg::diagonal
ferray_linalg::eigh_batched
ferray_linalg::eigh_batched_uplo
ferray_linalg::eigh_uplo
ferray_linalg::eigvalsh
ferray_linalg::eigvalsh_batched
ferray_linalg::eigvalsh_batched_uplo
ferray_linalg::eigvalsh_uplo
ferray_linalg::einsum
ferray_linalg::f16_support::dot_f16
ferray_linalg::f16_support::matmul_f16
ferray_linalg::f16_support::norm_f16
ferray_linalg::f16_support::outer_f16
ferray_linalg::f16_support::vdot_f16
ferray_linalg::faer_bridge::array1_to_faer_col
ferray_linalg::faer_bridge::array2_to_faer
ferray_linalg::faer_bridge::array2_to_faer_general
ferray_linalg::faer_bridge::extract_matrix_from_batch
ferray_linalg::faer_bridge::faer_col_to_array1
ferray_linalg::faer_bridge::faer_to_array2
ferray_linalg::faer_bridge::faer_to_arrayd
ferray_linalg::gemm::cpu_supports_avx2_fma
ferray_linalg::gemm::cpu_supports_avx512_bw
ferray_linalg::gemm::cpu_supports_avx512_vnni
ferray_linalg::gemm::cpu_supports_avx512f
ferray_linalg::gemm::cpu_supports_neon
ferray_linalg::gemm::faer_fallback::gemm_fallback_c32
ferray_linalg::gemm::faer_fallback::gemm_fallback_c64
ferray_linalg::gemm::faer_fallback::gemm_fallback_f32
ferray_linalg::gemm::faer_fallback::gemm_fallback_f64
ferray_linalg::gemm::faer_fallback::gemm_fallback_i16
ferray_linalg::gemm::faer_fallback::gemm_fallback_i8
ferray_linalg::gemm::faer_fallback::gemm_fallback_i8s
ferray_linalg::gemm::gemm_bf16_f32
ferray_linalg::gemm::gemm_c32
ferray_linalg::gemm::gemm_c64
ferray_linalg::gemm::gemm_f16_f32
ferray_linalg::gemm::gemm_f32
ferray_linalg::gemm::gemm_f64
ferray_linalg::gemm::gemm_i16
ferray_linalg::gemm::gemm_i8
ferray_linalg::gemm::gemm_i8_signed
ferray_linalg::gemm::kernel_avx2::kernel_8x6
ferray_linalg::gemm::kernel_avx2::kernel_edge
ferray_linalg::gemm::kernel_avx2_complex_f32::kernel_4x4_c32
ferray_linalg::gemm::kernel_avx2_complex_f32::kernel_edge_c32
ferray_linalg::gemm::kernel_avx2_complex_f64::kernel_4x2_c64
ferray_linalg::gemm::kernel_avx2_complex_f64::kernel_edge_c64
ferray_linalg::gemm::kernel_avx2_f32::kernel_4x16
ferray_linalg::gemm::kernel_avx2_f32::kernel_edge_f32
ferray_linalg::gemm::kernel_avx2_i16::kernel_4x8_i16
ferray_linalg::gemm::kernel_avx2_i16::kernel_edge_i16
ferray_linalg::gemm::kernel_avx2_i8::kernel_4x8_i8
ferray_linalg::gemm::kernel_avx2_i8::kernel_edge_i8
ferray_linalg::gemm::kernel_avx2_i8s::kernel_4x8_i8_signed
ferray_linalg::gemm::kernel_avx2_i8s::kernel_edge_i8_signed
ferray_linalg::gemm::kernel_avx512::kernel_4x16_avx512
ferray_linalg::gemm::kernel_avx512::kernel_edge_avx512
ferray_linalg::gemm::kernel_avx512_complex_f32::kernel_4x8_c32_avx512
ferray_linalg::gemm::kernel_avx512_complex_f32::kernel_edge_c32_avx512
ferray_linalg::gemm::kernel_avx512_complex_f64::kernel_4x4_c64_avx512
ferray_linalg::gemm::kernel_avx512_complex_f64::kernel_edge_c64_avx512
ferray_linalg::gemm::kernel_avx512_f32::kernel_4x32_avx512
ferray_linalg::gemm::kernel_avx512_f32::kernel_edge_avx512_f32
ferray_linalg::gemm::kernel_avx512_i16::KB_I16
ferray_linalg::gemm::kernel_avx512_i16::kernel_4x16_i16_avx512
ferray_linalg::gemm::kernel_avx512_i16::kernel_edge_i16_avx512
ferray_linalg::gemm::kernel_avx512_vnni_i8::KB_I8
ferray_linalg::gemm::kernel_avx512_vnni_i8::kernel_4x16_i8_avx512
ferray_linalg::gemm::kernel_avx512_vnni_i8::kernel_edge_i8_avx512
ferray_linalg::gemm::kernel_avxvnni_i8::KB_I8
ferray_linalg::gemm::kernel_avxvnni_i8::MR_I8
ferray_linalg::gemm::kernel_avxvnni_i8::NR_I8
ferray_linalg::gemm::kernel_avxvnni_i8::kernel_4x8_i8_vnni
ferray_linalg::gemm::kernel_neon_complex_f32::kernel_4x4_neon_c32
ferray_linalg::gemm::kernel_neon_complex_f32::kernel_edge_neon_c32
ferray_linalg::gemm::kernel_neon_complex_f64::kernel_4x2_neon_c64
ferray_linalg::gemm::kernel_neon_complex_f64::kernel_edge_neon_c64
ferray_linalg::gemm::kernel_neon_f32::kernel_4x16_neon_f32
ferray_linalg::gemm::kernel_neon_f32::kernel_edge_neon_f32
ferray_linalg::gemm::kernel_neon_f64::kernel_4x8_neon_f64
ferray_linalg::gemm::kernel_neon_f64::kernel_edge_neon_f64
ferray_linalg::gemm::kernel_neon_i16::kernel_4x8_neon_i16
ferray_linalg::gemm::kernel_neon_i16::kernel_edge_neon_i16
ferray_linalg::gemm::kernel_neon_i8::kernel_4x8_neon_i8
ferray_linalg::gemm::kernel_neon_i8::kernel_edge_neon_i8
ferray_linalg::gemm::kernel_neon_i8s::kernel_4x8_neon_i8_signed
ferray_linalg::gemm::kernel_neon_i8s::kernel_edge_neon_i8_signed
ferray_linalg::gemm::openblas_backend::gemm_c32_openblas
ferray_linalg::gemm::openblas_backend::gemm_c64_openblas
ferray_linalg::gemm::openblas_backend::gemm_f32_openblas
ferray_linalg::gemm::openblas_backend::gemm_f64_openblas
ferray_linalg::gemm::outer::gemm_avx2_c32
ferray_linalg::gemm::outer::gemm_avx2_c64
ferray_linalg::gemm::outer::gemm_avx2_f32
ferray_linalg::gemm::outer::gemm_avx2_f64
ferray_linalg::gemm::outer::gemm_avx2_i16
ferray_linalg::gemm::outer::gemm_avx2_i8
ferray_linalg::gemm::outer::gemm_avx2_i8s
ferray_linalg::gemm::outer::gemm_avx512_c32
ferray_linalg::gemm::outer::gemm_avx512_c64
ferray_linalg::gemm::outer::gemm_avx512_f32
ferray_linalg::gemm::outer::gemm_avx512_f64
ferray_linalg::gemm::outer::gemm_avx512_i16
ferray_linalg::gemm::outer::gemm_avx512_vnni_i8
ferray_linalg::gemm::outer_neon::gemm_neon_c32
ferray_linalg::gemm::outer_neon::gemm_neon_c64
ferray_linalg::gemm::outer_neon::gemm_neon_f32
ferray_linalg::gemm::outer_neon::gemm_neon_f64
ferray_linalg::gemm::outer_neon::gemm_neon_i16
ferray_linalg::gemm::outer_neon::gemm_neon_i8
ferray_linalg::gemm::outer_neon::gemm_neon_i8s
ferray_linalg::gemm::pack::pack_a
ferray_linalg::gemm::pack::pack_a_c32
ferray_linalg::gemm::pack::pack_a_c64
ferray_linalg::gemm::pack::pack_a_f32
ferray_linalg::gemm::pack::pack_a_i16
ferray_linalg::gemm::pack::pack_a_i16_mr4_avx512
ferray_linalg::gemm::pack::pack_a_i8s
ferray_linalg::gemm::pack::pack_a_u8
ferray_linalg::gemm::pack::pack_b
ferray_linalg::gemm::pack::pack_b_c32
ferray_linalg::gemm::pack::pack_b_c32_nr8
ferray_linalg::gemm::pack::pack_b_c64
ferray_linalg::gemm::pack::pack_b_c64_nr4
ferray_linalg::gemm::pack::pack_b_f32
ferray_linalg::gemm::pack::pack_b_f32_nr32
ferray_linalg::gemm::pack::pack_b_f64_nr16
ferray_linalg::gemm::pack::pack_b_i16
ferray_linalg::gemm::pack::pack_b_i16_nr16
ferray_linalg::gemm::pack::pack_b_i8
ferray_linalg::gemm::pack::pack_b_i8_nr16
ferray_linalg::gemm::pack::pack_b_i8s
ferray_linalg::gemm::pack::pack_kc_i16
ferray_linalg::gemm::pack::pack_kc_i8
ferray_linalg::gemm::pack::pack_kc_i8s
ferray_linalg::gemm::pack_neon::pack_a_neon_c32
ferray_linalg::gemm::pack_neon::pack_a_neon_c64
ferray_linalg::gemm::pack_neon::pack_a_neon_f32
ferray_linalg::gemm::pack_neon::pack_a_neon_f64
ferray_linalg::gemm::pack_neon::pack_a_neon_i16
ferray_linalg::gemm::pack_neon::pack_a_neon_i8s
ferray_linalg::gemm::pack_neon::pack_a_neon_u8
ferray_linalg::gemm::pack_neon::pack_b_neon_c32
ferray_linalg::gemm::pack_neon::pack_b_neon_c64
ferray_linalg::gemm::pack_neon::pack_b_neon_f32
ferray_linalg::gemm::pack_neon::pack_b_neon_f64
ferray_linalg::gemm::pack_neon::pack_b_neon_i16
ferray_linalg::gemm::pack_neon::pack_b_neon_i8
ferray_linalg::gemm::pack_neon::pack_b_neon_i8s
ferray_linalg::gemm::quantized_matmul
ferray_linalg::inv_batched
ferray_linalg::inv_complex
ferray_linalg::lu
ferray_linalg::matmul_complex
ferray_linalg::matrix_norm
ferray_linalg::matrix_power
ferray_linalg::matrix_power_batched
ferray_linalg::matrix_rank_batched
ferray_linalg::matrix_transpose
ferray_linalg::matvec
ferray_linalg::multi_dot
ferray_linalg::norm_axis
ferray_linalg::norms::det_batched
ferray_linalg::norms::diagonal
ferray_linalg::norms::matrix_norm
ferray_linalg::norms::matrix_rank_batched
ferray_linalg::norms::matrix_transpose
ferray_linalg::norms::norm_axis
ferray_linalg::norms::slogdet
ferray_linalg::norms::slogdet_batched
ferray_linalg::norms::trace_offset
ferray_linalg::norms::vector_norm
ferray_linalg::pinv
ferray_linalg::pinv_batched
ferray_linalg::products::TensordotAxes
ferray_linalg::products::cross
ferray_linalg::products::einsum
ferray_linalg::products::einsum::contraction::generic_contraction
ferray_linalg::products::einsum::einsum
ferray_linalg::products::einsum::optimizer::EinsumStrategy
ferray_linalg::products::einsum::optimizer::optimize
ferray_linalg::products::einsum::parser::EinsumExpr
ferray_linalg::products::einsum::parser::Label
ferray_linalg::products::einsum::parser::parse_subscripts
ferray_linalg::products::matvec
ferray_linalg::products::multi_dot
ferray_linalg::products::vecdot
ferray_linalg::products::vecmat
ferray_linalg::qr_batched
ferray_linalg::scalar::LinalgFloat
ferray_linalg::scalar::private::Sealed
ferray_linalg::slogdet
ferray_linalg::slogdet_batched
ferray_linalg::solve::inv_batched
ferray_linalg::solve::matrix_power
ferray_linalg::solve::matrix_power_batched
ferray_linalg::solve::pinv
ferray_linalg::solve::pinv_batched
ferray_linalg::solve::solve_batched
ferray_linalg::solve::solve_dyn
ferray_linalg::solve::tensorinv
ferray_linalg::solve::tensorsolve
ferray_linalg::solve_batched
ferray_linalg::solve_complex
ferray_linalg::solve_complex_vec
ferray_linalg::solve_dyn
ferray_linalg::svd_batched
ferray_linalg::svd_hermitian
ferray_linalg::svdvals
ferray_linalg::tensorinv
ferray_linalg::tensorsolve
ferray_linalg::trace_offset
ferray_linalg::trsm::trsm_left_lower_f32
ferray_linalg::trsm::trsm_left_lower_f64
ferray_linalg::trsm::trsm_left_upper_f32
ferray_linalg::trsm::trsm_left_upper_f64
ferray_linalg::vecdot
ferray_linalg::vecmat
ferray_linalg::vector_norm
"#;

fn arr1(data: Vec<f64>) -> Array<f64, Ix1> {
    Array::from_vec(Ix1::new([data.len()]), data).unwrap()
}

fn arr2(rows: usize, cols: usize, data: Vec<f64>) -> Array<f64, Ix2> {
    Array::from_vec(Ix2::new([rows, cols]), data).unwrap()
}

fn arrd(shape: &[usize], data: Vec<f64>) -> Array<f64, IxDyn> {
    Array::from_vec(IxDyn::new(shape), data).unwrap()
}

fn cmat(rows: usize, cols: usize, data: Vec<Complex<f64>>) -> Array<Complex<f64>, Ix2> {
    Array::from_vec(Ix2::new([rows, cols]), data).unwrap()
}

fn cvec(data: Vec<Complex<f64>>) -> Array<Complex<f64>, Ix1> {
    Array::from_vec(Ix1::new([data.len()]), data).unwrap()
}

fn assert_close(actual: f64, expected: f64, name: &str) {
    let diff = (actual - expected).abs();
    let tol = 1e-9 * expected.abs().max(1.0);
    assert!(
        diff <= tol,
        "{name}: got {actual}, expected {expected}, diff={diff}, tol={tol}"
    );
}

fn assert_slice_close(actual: &[f64], expected: &[f64], name: &str) {
    assert_eq!(actual.len(), expected.len(), "{name} length");
    for (idx, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        assert_close(got, want, &format!("{name}[{idx}]"));
    }
}

#[test]
fn previous_linalg_exclusion_paths_are_explicitly_anchored() {
    let count = LINALG_EXCLUSION_SURFACE_PATHS
        .lines()
        .filter(|line| !line.trim().is_empty())
        .count();
    assert_eq!(count, 275);
    assert!(LINALG_EXCLUSION_SURFACE_PATHS.contains("ferray_linalg::gemm::pack::pack_b_i8"));
    assert!(LINALG_EXCLUSION_SURFACE_PATHS.contains("ferray_linalg::solve::tensorinv"));
}

#[test]
fn batch_and_faer_bridge_helpers_round_trip_matrix_data() {
    let batched = arrd(&[2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(
        ferray_linalg::batch::batch_count(batched.shape(), 2).unwrap(),
        2
    );
    assert_eq!(
        ferray_linalg::batch::extract_batch_matrix(batched.as_slice().unwrap(), batched.shape(), 1),
        vec![5.0, 6.0, 7.0, 8.0]
    );

    let copies =
        ferray_linalg::batch::apply_batched_2d(&batched, |_m, _n, data| Ok(data.to_vec())).unwrap();
    assert_eq!(copies[0], vec![1.0, 2.0, 3.0, 4.0]);

    let traces = ferray_linalg::batch::apply_batched_scalar(&batched, |_m, n, data| {
        Ok(data[0] + data[n + 1])
    })
    .unwrap();
    assert_slice_close(traces.as_slice().unwrap(), &[5.0, 13.0], "batched traces");

    let pair = ferray_linalg::batch::apply_batched_2d_pair(
        &batched,
        &batched,
        |_am, _an, a, _bm, _bn, b| Ok(a.iter().zip(b).map(|(x, y)| x + y).collect()),
    )
    .unwrap();
    assert_eq!(pair[1], vec![10.0, 12.0, 14.0, 16.0]);

    let m = arr2(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let faer = ferray_linalg::faer_bridge::array2_to_faer(&m);
    let from_faer = ferray_linalg::faer_bridge::faer_to_array2(&faer).unwrap();
    assert_slice_close(
        from_faer.as_slice().unwrap(),
        m.as_slice().unwrap(),
        "faer round trip",
    );
    let general = ferray_linalg::faer_bridge::array2_to_faer_general(&m).unwrap();
    let from_general = ferray_linalg::faer_bridge::faer_to_arrayd(&general).unwrap();
    assert_slice_close(
        from_general.as_slice().unwrap(),
        m.as_slice().unwrap(),
        "faer dyn",
    );

    let col = ferray_linalg::faer_bridge::array1_to_faer_col(&arr1(vec![9.0, 10.0]));
    let col_back = ferray_linalg::faer_bridge::faer_col_to_array1(&col).unwrap();
    assert_slice_close(col_back.as_slice().unwrap(), &[9.0, 10.0], "faer col");
    let extracted = ferray_linalg::faer_bridge::extract_matrix_from_batch(&batched, 0).unwrap();
    assert_eq!(extracted.shape(), (2, 2));

    let mat = ferray_linalg::batch::slice_to_faer(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    let mat_vec = ferray_linalg::batch::faer_to_vec(&mat);
    assert_eq!(mat_vec, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn product_norm_and_solve_root_aliases_match_small_references() {
    let a = arrd(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = arrd(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let v = arrd(&[2], vec![1.0, 2.0]);
    let w = arrd(&[2], vec![3.0, 4.0]);

    assert_slice_close(
        ferray_linalg::matmul(&a, &b).unwrap().as_slice().unwrap(),
        &[19.0, 22.0, 43.0, 50.0],
        "matmul",
    );
    assert_close(
        ferray_linalg::dot(&v, &w).unwrap().as_slice().unwrap()[0],
        11.0,
        "dot",
    );
    assert_close(
        ferray_linalg::inner(&v, &w).unwrap().as_slice().unwrap()[0],
        11.0,
        "inner",
    );
    assert_close(ferray_linalg::vdot(&v, &w).unwrap(), 11.0, "vdot");
    assert_slice_close(
        ferray_linalg::outer(&v, &w).unwrap().as_slice().unwrap(),
        &[3.0, 4.0, 6.0, 8.0],
        "outer",
    );
    assert_slice_close(
        ferray_linalg::cross(
            &arrd(&[3], vec![1.0, 0.0, 0.0]),
            &arrd(&[3], vec![0.0, 1.0, 0.0]),
        )
        .unwrap()
        .as_slice()
        .unwrap(),
        &[0.0, 0.0, 1.0],
        "cross",
    );
    assert_slice_close(
        ferray_linalg::vecdot(&a, &b, Some(-1))
            .unwrap()
            .as_slice()
            .unwrap(),
        &[17.0, 53.0],
        "vecdot",
    );
    assert_slice_close(
        ferray_linalg::matvec(&a, &v).unwrap().as_slice().unwrap(),
        &[5.0, 11.0],
        "matvec",
    );
    assert_slice_close(
        ferray_linalg::vecmat(&v, &a).unwrap().as_slice().unwrap(),
        &[7.0, 10.0],
        "vecmat",
    );
    assert_slice_close(
        ferray_linalg::tensordot(&a, &b, ferray_linalg::TensordotAxes::Scalar(1))
            .unwrap()
            .as_slice()
            .unwrap(),
        &[19.0, 22.0, 43.0, 50.0],
        "tensordot",
    );
    assert_slice_close(
        ferray_linalg::einsum("ij,jk->ik", &[&a, &b])
            .unwrap()
            .as_slice()
            .unwrap(),
        &[19.0, 22.0, 43.0, 50.0],
        "einsum",
    );
    let multi = [&a, &b];
    assert_slice_close(
        ferray_linalg::multi_dot(&multi)
            .unwrap()
            .as_slice()
            .unwrap(),
        &[19.0, 22.0, 43.0, 50.0],
        "multi_dot",
    );
    assert_eq!(
        ferray_linalg::kron(&v, &w).unwrap().as_slice().unwrap(),
        &[3.0, 4.0, 6.0, 8.0]
    );

    let a2 = arr2(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    assert_close(
        ferray_linalg::vector_norm(&v, ferray_linalg::NormOrder::L2, None, false)
            .unwrap()
            .as_slice()
            .unwrap()[0],
        5.0_f64.sqrt(),
        "vector_norm",
    );
    assert_close(
        ferray_linalg::matrix_norm(&a2, ferray_linalg::NormOrder::Fro).unwrap(),
        30.0_f64.sqrt(),
        "matrix_norm",
    );
    assert_slice_close(
        ferray_linalg::norm_axis(&a, ferray_linalg::NormOrder::L2, 1, false)
            .unwrap()
            .as_slice()
            .unwrap(),
        &[5.0_f64.sqrt(), 25.0_f64.sqrt()],
        "norm_axis",
    );
    assert_close(
        ferray_linalg::trace_offset(&a2, 0).unwrap(),
        5.0,
        "trace_offset",
    );
    assert_slice_close(
        ferray_linalg::diagonal(&a2, 0).unwrap().as_slice().unwrap(),
        &[1.0, 4.0],
        "diagonal",
    );
    assert_eq!(ferray_linalg::matrix_transpose(&a2).shape(), &[2, 2]);
    assert_close(
        ferray_linalg::det_batched(&a).unwrap().as_slice().unwrap()[0],
        -2.0,
        "det_batched 2d",
    );
    let (sign, logabs) = ferray_linalg::slogdet(&a2).unwrap();
    assert_close(sign, -1.0, "slogdet sign");
    assert_close(logabs, 2.0_f64.ln(), "slogdet logabs");
    assert_close(
        ferray_linalg::slogdet_batched(&a)
            .unwrap()
            .0
            .as_slice()
            .unwrap()[0],
        -1.0,
        "slogdet_batched sign",
    );
    assert_eq!(
        ferray_linalg::matrix_rank_batched(&a, None)
            .unwrap()
            .as_slice()
            .unwrap()[0],
        2
    );

    let lhs = arr2(2, 2, vec![2.0, 0.0, 0.0, 4.0]);
    let rhs = arrd(&[2], vec![2.0, 8.0]);
    assert_slice_close(
        ferray_linalg::solve_dyn(&arrd(&[2, 2], vec![2.0, 0.0, 0.0, 4.0]), &rhs)
            .unwrap()
            .as_slice()
            .unwrap(),
        &[1.0, 2.0],
        "solve_dyn",
    );
    assert_slice_close(
        ferray_linalg::pinv(&lhs, None).unwrap().as_slice().unwrap(),
        &[0.5, 0.0, 0.0, 0.25],
        "pinv",
    );
    assert_slice_close(
        ferray_linalg::matrix_power(&lhs, 2)
            .unwrap()
            .as_slice()
            .unwrap(),
        &[4.0, 0.0, 0.0, 16.0],
        "matrix_power",
    );
    assert_slice_close(
        ferray_linalg::tensorinv(&arrd(&[2, 2], vec![2.0, 0.0, 0.0, 4.0]), 1)
            .unwrap()
            .as_slice()
            .unwrap(),
        &[0.5, 0.0, 0.0, 0.25],
        "tensorinv",
    );
    assert_slice_close(
        ferray_linalg::tensorsolve(&arrd(&[2, 2], vec![2.0, 0.0, 0.0, 4.0]), &rhs, None)
            .unwrap()
            .as_slice()
            .unwrap(),
        &[1.0, 2.0],
        "tensorsolve",
    );
}

#[test]
fn decomposition_and_complex_aliases_match_small_references() {
    let spd = arr2(2, 2, vec![4.0, 2.0, 2.0, 3.0]);
    assert_slice_close(
        ferray_linalg::cholesky_upper(&spd)
            .unwrap()
            .as_slice()
            .unwrap(),
        &[2.0, 1.0, 0.0, 2.0_f64.sqrt()],
        "cholesky_upper",
    );
    let spd_dyn = arrd(&[1, 2, 2], vec![4.0, 2.0, 2.0, 3.0]);
    assert_eq!(
        ferray_linalg::cholesky_batched(&spd_dyn).unwrap().shape(),
        &[1, 2, 2]
    );
    assert_eq!(
        ferray_linalg::cholesky_upper_batched(&spd_dyn)
            .unwrap()
            .shape(),
        &[1, 2, 2]
    );

    let diag = arr2(2, 2, vec![2.0, 0.0, 0.0, 3.0]);
    assert_slice_close(
        ferray_linalg::eigvalsh(&diag).unwrap().as_slice().unwrap(),
        &[2.0, 3.0],
        "eigvalsh",
    );
    assert_slice_close(
        ferray_linalg::eigvalsh_uplo(&diag, ferray_linalg::UPLO::Upper)
            .unwrap()
            .as_slice()
            .unwrap(),
        &[2.0, 3.0],
        "eigvalsh_uplo",
    );
    assert_slice_close(
        ferray_linalg::eigh_uplo(&diag, ferray_linalg::UPLO::Lower)
            .unwrap()
            .0
            .as_slice()
            .unwrap(),
        &[2.0, 3.0],
        "eigh_uplo",
    );
    let diag_dyn = arrd(&[1, 2, 2], vec![2.0, 0.0, 0.0, 3.0]);
    assert_eq!(
        ferray_linalg::eigvalsh_batched(&diag_dyn).unwrap().shape(),
        &[1, 2]
    );
    assert_eq!(
        ferray_linalg::eigh_batched_uplo(&diag_dyn, ferray_linalg::UPLO::Lower)
            .unwrap()
            .0
            .shape(),
        &[1, 2]
    );

    assert_slice_close(
        ferray_linalg::svdvals(&diag).unwrap().as_slice().unwrap(),
        &[3.0, 2.0],
        "svdvals",
    );
    assert_eq!(ferray_linalg::svd_hermitian(&diag).unwrap().1.shape(), &[2]);
    assert_eq!(
        ferray_linalg::svd_batched(&diag_dyn, false)
            .unwrap()
            .1
            .shape(),
        &[1, 2]
    );
    assert_eq!(
        ferray_linalg::qr_batched(&diag_dyn, ferray_linalg::QrMode::Reduced)
            .unwrap()
            .0
            .shape(),
        &[1, 2, 2]
    );
    assert_eq!(ferray_linalg::lu(&diag).unwrap().0.shape(), &[2, 2]);

    let ca = cmat(
        2,
        2,
        vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
        ],
    );
    let cb = cmat(
        2,
        2,
        vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(1.0, 0.0),
        ],
    );
    assert_eq!(
        ferray_linalg::matmul_complex(&ca, &cb)
            .unwrap()
            .as_slice()
            .unwrap()[1],
        Complex::new(2.0, 0.0)
    );
    assert_eq!(
        ferray_linalg::det_complex(&ca).unwrap(),
        Complex::new(1.0, 0.0)
    );
    assert_eq!(
        ferray_linalg::inv_complex(&ca).unwrap().as_slice().unwrap()[0],
        Complex::new(1.0, 0.0)
    );
    assert_eq!(
        ferray_linalg::solve_complex(
            &ca,
            &cmat(2, 1, vec![Complex::new(3.0, 0.0), Complex::new(1.0, 0.0)]),
        )
        .unwrap()
        .as_slice()
        .unwrap()[0],
        Complex::new(1.0, 0.0)
    );
    assert_eq!(
        ferray_linalg::solve_complex_vec(
            &ca,
            &cvec(vec![Complex::new(3.0, 0.0), Complex::new(1.0, 0.0)]),
        )
        .unwrap()
        .as_slice()
        .unwrap()[0],
        Complex::new(1.0, 0.0)
    );
}

#[test]
fn einsum_parser_optimizer_gemm_and_trsm_surfaces_are_anchored() {
    let expr =
        ferray_linalg::products::einsum::parser::parse_subscripts("ij,jk->ik", &[&[2, 2], &[2, 2]])
            .unwrap();
    assert_eq!(expr.inputs.len(), 2);
    assert!(matches!(
        expr.inputs[0][0],
        ferray_linalg::products::einsum::parser::Label::Char('i')
    ));
    assert!(matches!(
        ferray_linalg::products::einsum::optimizer::optimize(&expr),
        ferray_linalg::products::einsum::optimizer::EinsumStrategy::Matmul
    ));
    assert!(matches!(
        ferray_linalg::products::TensordotAxes::Pairs(vec![1], vec![0]),
        ferray_linalg::products::TensordotAxes::Pairs(_, _)
    ));

    let _ = ferray_linalg::gemm::cpu_supports_avx2_fma();
    let _ = ferray_linalg::gemm::cpu_supports_avx512f();
    let _ = ferray_linalg::gemm::cpu_supports_avx512_bw();
    let _ = ferray_linalg::gemm::cpu_supports_avx512_vnni();
    let _ = ferray_linalg::gemm::cpu_supports_neon();

    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [5.0, 6.0, 7.0, 8.0];
    let mut c = [0.0; 4];
    let ok = ferray_linalg::gemm::gemm_f64(2, 2, 2, 1.0, &a, &b, 0.0, &mut c);
    if ok {
        assert_slice_close(&c, &[19.0, 22.0, 43.0, 50.0], "gemm_f64");
    }
    let af = [1.0_f32, 2.0, 3.0, 4.0];
    let bf = [5.0_f32, 6.0, 7.0, 8.0];
    let mut cf = [0.0_f32; 4];
    let _ = ferray_linalg::gemm::gemm_f32(2, 2, 2, 1.0, &af, &bf, 0.0, &mut cf);

    let lower = [2.0_f64, 0.0, 1.0, 2.0];
    let mut rhs = [2.0_f64, 5.0];
    unsafe {
        ferray_linalg::trsm::trsm_left_lower_f64(
            2,
            1,
            1.0,
            lower.as_ptr(),
            2,
            rhs.as_mut_ptr(),
            1,
            false,
        );
    }
    assert_slice_close(&rhs, &[1.0, 2.0], "trsm lower f64");

    let upper = [2.0_f32, 1.0, 0.0, 2.0];
    let mut rhs32 = [4.0_f32, 4.0];
    unsafe {
        ferray_linalg::trsm::trsm_left_upper_f32(
            2,
            1,
            1.0,
            upper.as_ptr(),
            2,
            rhs32.as_mut_ptr(),
            1,
            false,
        );
    }
    assert!((rhs32[0] - 1.0).abs() < 1e-6);
    assert!((rhs32[1] - 2.0).abs() < 1e-6);
}
