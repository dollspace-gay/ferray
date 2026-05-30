//! Divergence pins: ferray-strings LIBRARY vs numpy.strings (numpy 2.4.5).
//!
//! Each test's expected value comes from a live numpy call or a numpy
//! upstream `file:line` symbolic constant (R-CHAR-3) — never copied from
//! the ferray side. These tests FAIL against the current ferray-strings
//! implementation and pin genuine LIBRARY divergences (not ferray-python
//! marshalling concerns).

use std::any::type_name_of_val;

use ferray_strings::{array, count, str_len};

/// Divergence: `ferray_strings::str_len` counts BYTES, numpy counts code points.
///
/// numpy `string_ufuncs.cpp:118`:
///   `*(npy_intp *)out = buf.num_codepoints();`
/// ferray `str_ops.rs` `str_len`: `s.len() as u64` (byte length).
///
/// For `"héllo"` (UTF-8: 6 bytes, 5 code points):
///   numpy `np.strings.str_len(['héllo'])` -> 5   (verified live, numpy 2.4.5)
///   ferray returns 6.
#[test]
fn divergence_str_len_counts_codepoints_not_bytes() {
    let a = array(&["héllo", "naïve"]).unwrap();
    let r = str_len(&a).unwrap();
    // numpy live: np.strings.str_len(np.array(['héllo','naïve'])) -> [5 5]
    let got: Vec<i64> = r.as_slice().unwrap().to_vec();
    assert_eq!(
        got,
        vec![5i64, 5i64],
        "str_len must count Unicode code points (numpy num_codepoints), got byte lengths"
    );
}

/// Divergence: `ferray_strings::str_len` RETURN DTYPE is `u64` (unsigned),
/// numpy `str_len` returns signed `int64` (npy_intp / NPY_DEFAULT_INT).
///
/// numpy `string_ufuncs.cpp:1537`: `dtypes[1] = NPY_DEFAULT_INT;`
/// numpy live: `np.strings.str_len(np.array(['a'])).dtype` -> int64
///
/// ferray `str_ops.rs`: `pub fn str_len<D>(...) -> FerrayResult<Array<u64, D>>`.
/// The element type witnesses the divergence: numpy's signed contract means
/// the result element type must be `i64`, ferray ships `u64`.
#[test]
fn divergence_str_len_returns_signed_int64() {
    let a = array(&["a"]).unwrap();
    let r = str_len(&a).unwrap();
    let elem = r.as_slice().unwrap()[0];
    let tn = type_name_of_val(&elem);
    // numpy str_len result dtype is signed int64; the Rust analog element
    // type must be `i64`. ferray currently yields `u64`.
    assert_eq!(
        tn, "i64",
        "str_len result element type must be signed i64 to mirror numpy NPY_DEFAULT_INT (int64); got {tn}"
    );
}

/// Divergence: `ferray_strings::count` RETURN DTYPE is `u64` (unsigned),
/// numpy `count` returns signed `int64`.
///
/// numpy `generate_umath.py:1281` registers `count` Ufunc; its loop writes
/// `npy_intp` and numpy live `np.strings.count(np.array(['a']),'a').dtype`
/// -> int64.
///
/// ferray `search.rs`: `pub fn count<D>(...) -> FerrayResult<Array<u64, D>>`.
#[test]
fn divergence_count_returns_signed_int64() {
    let a = array(&["a"]).unwrap();
    let r = count(&a, "a").unwrap();
    let elem = r.as_slice().unwrap()[0];
    let tn = type_name_of_val(&elem);
    assert_eq!(
        tn, "i64",
        "count result element type must be signed i64 to mirror numpy int64; got {tn}"
    );
}
