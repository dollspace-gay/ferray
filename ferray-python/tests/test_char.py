"""Tests for #715 — numpy.char namespace bindings."""

import numpy as np
import pytest

import ferray


# ---------------------------------------------------------------------------
# Case operations
# ---------------------------------------------------------------------------


def test_lower_matches_numpy():
    arr = np.array(["Hello", "WORLD", "MixedCase"])
    np.testing.assert_array_equal(ferray.char.lower(arr), np.char.lower(arr))


def test_upper_matches_numpy():
    arr = np.array(["hello", "world"])
    np.testing.assert_array_equal(ferray.char.upper(arr), np.char.upper(arr))


def test_capitalize_matches_numpy():
    arr = np.array(["hello world", "FOO BAR"])
    np.testing.assert_array_equal(
        ferray.char.capitalize(arr), np.char.capitalize(arr)
    )


def test_title_matches_numpy():
    arr = np.array(["hello world", "foo bar baz"])
    np.testing.assert_array_equal(ferray.char.title(arr), np.char.title(arr))


def test_swapcase_matches_numpy():
    arr = np.array(["Hello", "WoRlD"])
    np.testing.assert_array_equal(ferray.char.swapcase(arr), np.char.swapcase(arr))


def test_lower_preserves_2d_shape():
    arr = np.array([["Hi", "There"], ["Foo", "Bar"]])
    out = ferray.char.lower(arr)
    assert out.shape == arr.shape


# ---------------------------------------------------------------------------
# Strip family
# ---------------------------------------------------------------------------


def test_strip_default_whitespace():
    arr = np.array(["  hi  ", "\thello\n", "world"])
    np.testing.assert_array_equal(ferray.char.strip(arr), np.char.strip(arr))


def test_strip_specific_chars():
    arr = np.array(["xxhix", "xworldxx"])
    np.testing.assert_array_equal(
        ferray.char.strip(arr, chars="x"),
        np.char.strip(arr, chars="x"),
    )


def test_lstrip_matches_numpy():
    arr = np.array(["  hi  ", " hello"])
    np.testing.assert_array_equal(ferray.char.lstrip(arr), np.char.lstrip(arr))


def test_rstrip_matches_numpy():
    arr = np.array(["  hi  ", "hello "])
    np.testing.assert_array_equal(ferray.char.rstrip(arr), np.char.rstrip(arr))


# ---------------------------------------------------------------------------
# Search / query
# ---------------------------------------------------------------------------


def test_count_matches_numpy():
    arr = np.array(["hello", "foo", "barbaz"])
    np.testing.assert_array_equal(
        ferray.char.count(arr, "o"), np.char.count(arr, "o")
    )


def test_count_no_matches():
    arr = np.array(["abc", "def"])
    np.testing.assert_array_equal(
        ferray.char.count(arr, "z"), np.char.count(arr, "z")
    )


def test_find_matches_numpy():
    arr = np.array(["hello", "world", "foo"])
    np.testing.assert_array_equal(
        ferray.char.find(arr, "o"), np.char.find(arr, "o")
    )


def test_find_returns_minus_one_when_not_found():
    arr = np.array(["abc"])
    assert int(ferray.char.find(arr, "z")[0]) == -1


def test_startswith_matches_numpy():
    arr = np.array(["hello", "help", "world"])
    np.testing.assert_array_equal(
        ferray.char.startswith(arr, "hel"),
        np.char.startswith(arr, "hel"),
    )


def test_endswith_matches_numpy():
    arr = np.array(["hello", "world", "echo"])
    np.testing.assert_array_equal(
        ferray.char.endswith(arr, "o"),
        np.char.endswith(arr, "o"),
    )


def test_str_len_matches_numpy():
    arr = np.array(["a", "ab", "abc", "abcd"])
    np.testing.assert_array_equal(ferray.char.str_len(arr), np.char.str_len(arr))


# ---------------------------------------------------------------------------
# Replace / concat / multiply
# ---------------------------------------------------------------------------


def test_replace_matches_numpy():
    arr = np.array(["hello", "world", "foo"])
    np.testing.assert_array_equal(
        ferray.char.replace(arr, "o", "0"),
        np.char.replace(arr, "o", "0"),
    )


def test_replace_with_count():
    arr = np.array(["foofoofoo"])
    out = ferray.char.replace(arr, "foo", "bar", count=2)
    expected = np.char.replace(arr, "foo", "bar", count=2)
    np.testing.assert_array_equal(out, expected)


def test_add_concatenates_elementwise():
    a = np.array(["a", "b", "c"])
    b = np.array(["1", "2", "3"])
    np.testing.assert_array_equal(ferray.char.add(a, b), np.char.add(a, b))


def test_multiply_repeats():
    arr = np.array(["ab", "cd"])
    np.testing.assert_array_equal(
        ferray.char.multiply(arr, 3), np.char.multiply(arr, 3)
    )


# ---------------------------------------------------------------------------
# Pairwise comparison
# ---------------------------------------------------------------------------


def test_equal_matches_numpy():
    a = np.array(["hello", "world"])
    b = np.array(["hello", "WORLD"])
    np.testing.assert_array_equal(
        ferray.char.equal(a, b), np.char.equal(a, b)
    )


def test_less_matches_numpy():
    a = np.array(["abc", "xyz"])
    b = np.array(["abd", "abc"])
    np.testing.assert_array_equal(
        ferray.char.less(a, b), np.char.less(a, b)
    )


@pytest.mark.parametrize("name", ["not_equal", "less", "less_equal", "greater", "greater_equal"])
def test_string_compare_matches_numpy(name):
    a = np.array(["aa", "bb", "cc"])
    b = np.array(["aa", "ba", "cd"])
    fr_fn = getattr(ferray.char, name)
    np_fn = getattr(np.char, name)
    np.testing.assert_array_equal(fr_fn(a, b), np_fn(a, b))
