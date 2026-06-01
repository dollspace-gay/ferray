# ferray-test-oracle

Oracle-fixture loading and ULP/tolerance comparison helpers for the ferray conformance test suites.

Part of the [ferray](../README.md) workspace — internal test infrastructure (not a NumPy-surface crate; `publish = false`).

## Overview

`ferray-test-oracle` is the shared harness the workspace's conformance tests use to
validate ferray output against NumPy-generated fixtures. It loads JSON fixtures
(produced by `scripts/generate_fixtures.py`), parses the array data into ferray
`Array<T, IxDyn>` values, and compares results against NumPy's expected output using
ULP (Units in the Last Place) tolerances with a near-zero noise floor.

- **Fixture loading** — `load_fixture(path)` parses a JSON file into a `FixtureSuite`
  (`function`, `ferray_function`, `test_cases`); each `TestCase` carries `inputs`,
  `expected`, and a per-case `tolerance_ulps`. `fixtures_dir()` locates the workspace
  `fixtures/` directory by walking up from `CARGO_MANIFEST_DIR`.
- **JSON data parsing** — `parse_f64_data`, `parse_f32_data`, `parse_complex_data`,
  `parse_bool_data`, `parse_string_data`, `parse_usize_data`, `parse_shape`, and the
  scalar `parse_f64_value` (handling `"NaN"`/`"Inf"`/`"-Inf"` string encodings).
- **Array construction** — `make_f64_array`, `make_f32_array`, `make_complex_array`,
  `make_bool_array`, `make_f32_array_from_any`, plus `get_dtype` / `should_skip_f64` /
  `should_skip_f32` helpers.
- **ULP comparison** — `ulp_distance_f64` / `ulp_distance_f32` (sign-magnitude,
  cross-sign-through-zero), the assertion helpers `assert_f64_ulp`, `assert_f32_ulp`,
  `assert_f64_slice_ulp`, `assert_f32_slice_ulp`, `assert_complex_slice_ulp`, and the
  relative+absolute variants `assert_close_f64` / `assert_close_f32` (and their `_slice`
  forms). NaN matches NaN; Inf signs must agree.
- **Tolerance constants** — `MIN_ULP_TOLERANCE` (10), `F64_NOISE_FLOOR` /
  `F32_NOISE_FLOOR`, and the Stage 1 tolerance table (`TOL_TRANSCENDENTAL_*`,
  `TOL_REDUCTION_*`, `TOL_LINALG_*`, `TOL_FFT_*`, `TOL_POLYNOMIAL_*`).
- **Generic test runners** — `run_unary_f64_oracle`, `run_unary_f32_oracle`,
  `run_binary_f64_oracle`, `run_unary_bool_oracle`, `run_scalar_f64_oracle`,
  `run_reduction_f64_oracle`, `run_real_to_complex_oracle`,
  `run_complex_to_complex_oracle`, and `run_matrix_scalar_oracle`.

## Usage

`ferray-test-oracle` is a `dev-dependency` of the library crates and is invoked from
their `tests/conformance_*.rs` files. The generic runners cover the common cases; for
bespoke comparisons, load a fixture and assert directly:

```rust
use ferray_test_oracle::{
    fixtures_dir, load_fixture, make_f64_array, parse_f64_data, assert_f64_slice_ulp,
    MIN_ULP_TOLERANCE,
};

#[test]
fn conformance_sin() {
    // One-liner via the generic runner:
    let path = fixtures_dir().join("ufunc/sin.json");
    ferray_test_oracle::run_unary_f64_oracle(&path, |x| ferray_ufunc::sin(x));

    // Or load + compare by hand:
    let suite = load_fixture(&path);
    for case in &suite.test_cases {
        let input = make_f64_array(&case.inputs["x"]);
        let actual = ferray_ufunc::sin(&input).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            actual.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}
```

`serde_json` is re-exported as `ferray_test_oracle::serde_json` so test files can index
`Value`s without depending on `serde_json` directly.

## Feature flags

None. The crate exposes no `[features]`.

## MSRV & edition

- Edition 2024, MSRV 1.88
- License: MIT OR Apache-2.0
