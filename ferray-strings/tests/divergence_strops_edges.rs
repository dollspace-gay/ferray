// ACToR critic — operation-kernel divergence pins for ferray-strings.
//
// Each test mirrors the CPython `str` semantics that `numpy.char`/
// `numpy.strings` delegate to (the oracle the design doc binds to,
// `.design/ferray-strings.md`; comments in `case.rs`/`strip.rs`/
// `split_join.rs` explicitly cite `str.capitalize`/`str.title`/`str.strip`/
// `str.splitlines`). Expected values are taken LIVE from the CPython 3.13
// `str` oracle (R-CHAR-3), never copied from the ferray side.
//
// A failing test here pins a real divergence the existing conformance tests
// MISS. The generator — not the critic — fixes the production code.

use ferray_strings::align::center;
use ferray_strings::string_array::array;

/// Divergence: `ferray_strings::align::center` diverges from CPython
/// `str.center` (`ferray-strings/src/align.rs:27`,
/// `let left_pad = total_pad / 2;`) when the padding amount AND the field
/// width are both odd.
///
/// CPython `str.center` (Objects/unicodeobject.c `unicode_center_impl`,
/// `left = marg / 2 + (marg & width & 1)`) biases the EXTRA pad column to the
/// LEFT when `marg` and `width` are both odd; ferray always biases it to the
/// right (`right_pad = total_pad - left_pad`, so left gets `floor(marg/2)`).
///
/// Oracle (CPython 3.13 `str.center`, R-CHAR-3):
///   "hi".center(3, '*')  == "*hi"     (ferray -> "hi*")
///   "ab".center(5, '*')  == "**ab*"   (ferray -> "*ab**")
///   "x".center(7, '*')   == "***x***" (matches; marg even)
/// Upstream returns the left-biased form; ferray returns the right-biased one.
///
/// The existing `align.rs::tests::test_center` uses width 6 (even) only, so
/// the parity quirk is never exercised — this pins the gap.
/// Tracking: see crosslink blocker filed alongside this test.
#[test]
fn divergence_center_odd_width_odd_margin_biases_left() {
    // marg = 1, width = 3 -> both odd -> extra pad on the LEFT.
    let a = array(&["hi"]).unwrap();
    let b = center(&a, 3, '*').unwrap();
    assert_eq!(b.as_slice(), &["*hi"]);
}

/// Companion case: marg = 3, width = 5 (both odd). CPython:
/// `"ab".center(5, '*') == "**ab*"`. ferray yields `"*ab**"`.
#[test]
fn divergence_center_marg3_width5_biases_left() {
    let a = array(&["ab"]).unwrap();
    let b = center(&a, 5, '*').unwrap();
    assert_eq!(b.as_slice(), &["**ab*"]);
}

/// Guard / control: when `width` is even the bias rule collapses to
/// `floor(marg/2)` on the left and ferray already matches CPython.
/// `"x".center(6, '*') == "**x***"`. This asserts the PASSING case so the
/// divergence is pinned to the odd-width parity, not to `center` wholesale.
#[test]
fn center_even_width_matches_oracle() {
    let a = array(&["x"]).unwrap();
    let b = center(&a, 6, '*').unwrap();
    // CPython: marg=5, left = 5/2 + (5 & 6 & 1) = 2 + 0 = 2 -> "**x***".
    assert_eq!(b.as_slice(), &["**x***"]);
}
