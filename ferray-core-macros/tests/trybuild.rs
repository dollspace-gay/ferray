//! Compile-fail tests for the `ferray-core-macros` procedural macros.
//!
//! Each `tests/compile_fail/*.rs` file is a tiny crate that exercises one
//! error path in `FerrayRecord`, `s!`, or `promoted_type!`. The `.stderr`
//! companion file pins the diagnostic so that changes to the error
//! messages are surfaced in review (regenerate by running
//! `TRYBUILD=overwrite cargo test -p ferray-core-macros --test trybuild`).
//!
//! Only compiled on nightly-or-stable with the normal toolchain; trybuild
//! shells out to `cargo build` and parses the compiler's diagnostic output.

#[test]
fn ferray_record_compile_fail() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/ferray_record_missing_repr_c.rs");
    t.compile_fail("tests/compile_fail/ferray_record_on_enum.rs");
    t.compile_fail("tests/compile_fail/ferray_record_on_tuple_struct.rs");
    t.compile_fail("tests/compile_fail/ferray_record_on_unit_struct.rs");
}

#[test]
fn promoted_type_compile_fail() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/promoted_type_wrong_arg_count.rs");
    t.compile_fail("tests/compile_fail/promoted_type_unknown_type.rs");
}
