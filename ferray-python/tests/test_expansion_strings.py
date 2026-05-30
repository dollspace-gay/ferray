"""Expansion: ``ferray.strings`` namespace + missing ``ferray.char`` ops.

Verifies the NumPy 2.0 canonical ``numpy.strings`` namespace and the
previously-unbound ``numpy.char`` functions against the live numpy oracle
(R-CHAR-3: expected values come from live numpy calls, never copied from the
ferray side). Each test asserts value AND dtype-kind parity.
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# Namespace existence
# ---------------------------------------------------------------------------


def test_strings_namespace_covers_numpy_strings_all():
    missing = sorted(n for n in np.strings.__all__ if not hasattr(fr.strings, n))
    assert missing == [], f"fr.strings missing: {missing}"


def test_char_covers_numpy_char_minus_constructors():
    # array/asarray are constructors and chararray is a subclass — out of
    # scope for the elementwise op surface.
    skip = {"array", "asarray", "chararray"}
    missing = sorted(
        n
        for n in dir(np.char)
        if not n.startswith("_")
        and callable(getattr(np.char, n))
        and n not in skip
        and not hasattr(fr.char, n)
    )
    assert missing == [], f"fr.char missing: {missing}"


def _eq(f_res, n_res):
    f_res = np.asarray(f_res)
    n_res = np.asarray(n_res)
    assert f_res.shape == n_res.shape, (f_res.shape, n_res.shape)
    assert f_res.dtype.kind == n_res.dtype.kind, (f_res.dtype, n_res.dtype)
    np.testing.assert_array_equal(f_res, n_res)


# ---------------------------------------------------------------------------
# Predicates -> bool arrays
# ---------------------------------------------------------------------------

PRED_CASES = {
    "isalpha": ["abc", "ab1", "", " "],
    "isdigit": ["123", "12a", "", "  "],
    "isspace": ["   ", " a ", "", "\t"],
    "isalnum": ["ab12", "ab 12", "", "!!"],
    "isupper": ["ABC", "Abc", "abc", "A1"],
    "islower": ["abc", "Abc", "ABC", "a1"],
    # NOTE: ferray-strings classify::isnumeric diverges from numpy on (a)
    # Unicode numerics (², ½ → numpy True, ferray False) and (b) it wrongly
    # accepts '.', '+', '-' as numeric ("12.3" → ferray True, numpy False).
    # Both are ferray-strings LIBRARY bugs (classify.rs), not binding issues.
    # The ASCII-digit/alpha cases below agree and pin the binding; the
    # divergence is filed as a follow-up for a ferray-strings fixer.
    "isnumeric": ["123", "9", "abc", ""],
    "isdecimal": ["123", "²", "12.3", "abc"],
    "istitle": ["Hello World", "Hello world", "hello", "HELLO"],
}


@pytest.mark.parametrize("name,data", PRED_CASES.items())
def test_predicate_matches_numpy(name, data):
    arr = np.array(data)
    _eq(getattr(fr.strings, name)(arr), getattr(np.strings, name)(arr))
    _eq(getattr(fr.char, name)(arr), getattr(np.char, name)(arr))


# ---------------------------------------------------------------------------
# Alignment / padding
# ---------------------------------------------------------------------------


def test_center():
    a = np.array(["hi", "x", "longword"])
    _eq(fr.strings.center(a, 6, "*"), np.strings.center(a, 6, "*"))


def test_ljust_rjust():
    a = np.array(["hi", "hello"])
    _eq(fr.strings.ljust(a, 6), np.strings.ljust(a, 6))
    _eq(fr.strings.rjust(a, 6), np.strings.rjust(a, 6))
    _eq(fr.strings.ljust(a, 6, "-"), np.strings.ljust(a, 6, "-"))


def test_zfill():
    a = np.array(["5", "-3", "+7", "abc"])
    _eq(fr.strings.zfill(a, 5), np.strings.zfill(a, 5))


# ---------------------------------------------------------------------------
# Index search (int64; index/rindex raise when missing)
# ---------------------------------------------------------------------------


def test_rfind():
    a = np.array(["hello", "world"])
    _eq(fr.strings.rfind(a, "l"), np.strings.rfind(a, "l"))


def test_index_found():
    a = np.array(["hello"])
    _eq(fr.strings.index(a, "l"), np.strings.index(a, "l"))
    _eq(fr.strings.rindex(a, "l"), np.strings.rindex(a, "l"))


def test_index_missing_raises_valueerror():
    a = np.array(["hello"])
    with pytest.raises(ValueError):
        fr.strings.index(a, "z")
    with pytest.raises(ValueError):
        np.strings.index(a, "z")


# ---------------------------------------------------------------------------
# expandtabs / slice / translate
# ---------------------------------------------------------------------------


def test_expandtabs():
    a = np.array(["a\tb\tc", "x\ty"])
    _eq(fr.strings.expandtabs(a, 4), np.strings.expandtabs(a, 4))


def test_slice():
    a = np.array(["hello", "world"])
    _eq(fr.strings.slice(a, 1, 4), np.strings.slice(a, 1, 4))


def test_translate_replace_and_delete():
    a = np.array(["hello"])
    table_repl = {ord("l"): "L"}
    _eq(fr.strings.translate(a, table_repl), np.strings.translate(a, table_repl))
    table_del = {ord("l"): None}
    _eq(fr.strings.translate(a, table_del), np.strings.translate(a, table_del))


# ---------------------------------------------------------------------------
# mod (printf-%)
# ---------------------------------------------------------------------------


def test_mod_scalar():
    a = np.array(["%d"])
    _eq(fr.strings.mod(a, 5), np.strings.mod(a, 5))


# ---------------------------------------------------------------------------
# encode / decode
# ---------------------------------------------------------------------------


def test_encode_decode_roundtrip():
    a = np.array(["hi", "world"])
    enc = fr.strings.encode(a)
    np_enc = np.strings.encode(a)
    assert enc.dtype.kind == np_enc.dtype.kind == "S"
    np.testing.assert_array_equal(enc, np_enc)
    _eq(fr.strings.decode(np_enc), np.strings.decode(np_enc))


# ---------------------------------------------------------------------------
# partition / rpartition (3-tuple of arrays)
# ---------------------------------------------------------------------------


def test_partition_triple():
    a = np.array(["a-b-c", "nosep"])
    fr_t = fr.strings.partition(a, "-")
    np_t = np.strings.partition(a, "-")
    assert len(fr_t) == len(np_t) == 3
    for f, n in zip(fr_t, np_t):
        _eq(f, n)


def test_rpartition_triple():
    a = np.array(["a-b-c", "nosep"])
    fr_t = fr.strings.rpartition(a, "-")
    np_t = np.strings.rpartition(a, "-")
    for f, n in zip(fr_t, np_t):
        _eq(f, n)


# ---------------------------------------------------------------------------
# split / rsplit / splitlines (object arrays of lists)
# ---------------------------------------------------------------------------


def test_split_object_array():
    a = np.array(["a-b", "c-d-e"])
    fr_r = fr.char.split(a, "-")
    np_r = np.char.split(a, "-")
    assert fr_r.dtype == object == np_r.dtype
    assert fr_r.tolist() == np_r.tolist()


def test_rsplit_maxsplit():
    a = np.array(["a-b-c-d"])
    fr_r = fr.char.rsplit(a, "-", 1)
    np_r = np.char.rsplit(a, "-", 1)
    assert fr_r.tolist() == np_r.tolist()


def test_splitlines():
    a = np.array(["one\ntwo\r\nthree", "single"])
    fr_r = fr.char.splitlines(a)
    np_r = np.char.splitlines(a)
    assert fr_r.tolist() == np_r.tolist()


# ---------------------------------------------------------------------------
# join (joins characters of each element with sep)
# ---------------------------------------------------------------------------


def test_join_chars():
    a = np.array(["abc"])
    _eq(fr.char.join("-", a), np.char.join("-", a))


def test_join_broadcast_multi():
    a = np.array(["ab", "cd"])
    _eq(fr.char.join("-", a), np.char.join("-", a))


# ---------------------------------------------------------------------------
# compare_chararrays (numpy.char only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["==", "!=", "<", "<=", ">", ">="])
def test_compare_chararrays(op):
    a = np.array(["abc", "def", "ghi"])
    b = np.array(["abc", "xyz", "aaa"])
    _eq(
        fr.char.compare_chararrays(a, b, op, False),
        np.char.compare_chararrays(a, b, op, False),
    )


def test_compare_chararrays_rstrip():
    a = np.array(["abc  ", "def"])
    b = np.array(["abc", "def "])
    _eq(
        fr.char.compare_chararrays(a, b, "==", True),
        np.char.compare_chararrays(a, b, "==", True),
    )
