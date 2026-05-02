//! Zero-length StringArray coverage (#282).
//!
//! The existing tests construct `StringArray::empty(Ix1::new([4]))` —
//! a length-4 array of empty strings. Actual zero-length arrays
//! (shape [0]) were not exercised; these tests pin that every
//! ferray-strings op accepts a zero-length input and produces a
//! zero-length output without panicking or returning an error.

use ferray_core::dimension::{Ix1, Ix2};
use ferray_strings::{
    StringArray1, StringArray2, add, capitalize, classify, count, endswith, find, lower, multiply,
    replace, rfind, search, startswith, title, upper,
};

fn empty_1d() -> StringArray1 {
    StringArray1::from_vec(Ix1::new([0]), Vec::new()).unwrap()
}

#[test]
fn zero_length_construction() {
    let a = empty_1d();
    assert_eq!(a.shape(), &[0]);
    assert_eq!(a.len(), 0);
    assert!(a.is_empty());
    assert_eq!(a.as_slice(), &[] as &[String]);
}

#[test]
fn zero_length_2d() {
    // Both shape variants of a zero-element 2D: (0, 3) and (3, 0).
    let a = StringArray2::from_vec(Ix2::new([0, 3]), Vec::new()).unwrap();
    assert_eq!(a.shape(), &[0, 3]);
    let b = StringArray2::from_vec(Ix2::new([3, 0]), Vec::new()).unwrap();
    assert_eq!(b.shape(), &[3, 0]);
}

// ---- Case ops ----

#[test]
fn zero_length_upper_lower() {
    let a = empty_1d();
    let u = upper(&a).unwrap();
    let l = lower(&a).unwrap();
    assert_eq!(u.shape(), &[0]);
    assert_eq!(l.shape(), &[0]);
}

#[test]
fn zero_length_capitalize_title() {
    let a = empty_1d();
    let c = capitalize(&a).unwrap();
    let t = title(&a).unwrap();
    assert_eq!(c.shape(), &[0]);
    assert_eq!(t.shape(), &[0]);
}

// ---- Search ops ----

#[test]
fn zero_length_find_count_startswith_endswith() {
    let a = empty_1d();
    let f = find(&a, "x").unwrap();
    assert_eq!(f.shape(), &[0]);
    let c = count(&a, "x").unwrap();
    assert_eq!(c.shape(), &[0]);
    let s = startswith(&a, "x").unwrap();
    assert_eq!(s.shape(), &[0]);
    let e = endswith(&a, "x").unwrap();
    assert_eq!(e.shape(), &[0]);
    let r = rfind(&a, "x").unwrap();
    assert_eq!(r.shape(), &[0]);
}

#[test]
fn zero_length_replace() {
    let a = empty_1d();
    let r = replace(&a, "x", "y", None).unwrap();
    assert_eq!(r.shape(), &[0]);
}

// ---- Classify ----

#[test]
fn zero_length_classify_isalpha() {
    let a = empty_1d();
    let r = classify::isalpha(&a).unwrap();
    assert_eq!(r.shape(), &[0]);
    assert_eq!(r.as_slice().unwrap(), &[] as &[bool]);
}

#[test]
fn zero_length_classify_isdigit_isspace() {
    let a = empty_1d();
    let d = classify::isdigit(&a).unwrap();
    let s = classify::isspace(&a).unwrap();
    assert_eq!(d.shape(), &[0]);
    assert_eq!(s.shape(), &[0]);
}

// ---- Concat / multiply ----

#[test]
fn zero_length_add_two_empty() {
    let a = empty_1d();
    let b = empty_1d();
    let c = add(&a, &b).unwrap();
    assert_eq!(c.shape(), &[0]);
}

#[test]
fn zero_length_multiply() {
    let a = empty_1d();
    let r = multiply(&a, 3).unwrap();
    assert_eq!(r.shape(), &[0]);
}

// ---- search re-exports (avoid the namespace collision with case::title) ----

#[test]
fn zero_length_search_module_count_matches() {
    let a = empty_1d();
    let c = search::count(&a, "x").unwrap();
    assert_eq!(c.shape(), &[0]);
}
