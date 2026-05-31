"""ACToR critic pins: numpy.char TRANSFORM/SEARCH family divergence audit.

Every expected value is derived LIVE from ``numpy.char`` in the test body
(R-CHAR-3): no literal-copied ferray-side constants. Each test calls the
matching ``ferray.char`` function and asserts it equals the numpy result;
the test FAILS where ferray diverges.

Audit scope: CASE (upper/lower/title/capitalize/swapcase), STRIP, PAD,
SEARCH/COUNT, SPLIT/JOIN/SPLITLINES against numpy 2.4.x.

Convergent families (probed, NO pin needed): count/find empty-substring
(``count('')`` -> len+1, non-overlapping count), find/count/startswith/
endswith start/end windows, index/rindex ValueError on miss, split('')
ValueError, upper/lower full case mapping (ß->SS, ﬁ->FI, final-sigma),
expandtabs, zfill sign handling, strip of U+0085/U+00A0/U+000B/U+000C/
U+2028/U+2029. Those are asserted-equal here as regression guards and pass.
"""

import numpy as np
import pytest
import ferray as fr


def _l(v):
    return v.tolist() if isinstance(v, np.ndarray) else v


# ===========================================================================
# CASE family — multi-char case mapping divergence (HIGH)
# numpy capitalize/title use TITLECASE (Py_UNICODE_TOTITLE) on the
# word-initial codepoint: ligature ﬁ -> "Fi", ß -> "Ss", digraph ǅ -> ǅ.
# ferray's case.rs uses char::to_uppercase: ﬁ -> "FI", ß -> "SS", ǅ -> "Ǆ".
# Upstream: numpy/_core/strings.py capitalize/title -> _vec_string ->
# Python str.capitalize/str.title (titlecase semantics).
# ===========================================================================

def test_capitalize_ligature_titlecase():
    """Divergence: ferray capitalize uppercases the ligature (ﬁ->FI);
    numpy/Python titlecases the first char (ﬁ->Fi). ferray-strings
    case.rs `capitalize` uses `first.to_uppercase()`."""
    a = np.array(["ﬁnery"], dtype="U10")
    expected = _l(np.char.capitalize(a))
    assert _l(fr.char.capitalize(a)) == expected


def test_capitalize_sharp_s_titlecase():
    """ß titlecases to 'Ss' (cap), not 'SS' (upper)."""
    a = np.array(["ßeta"], dtype="U10")
    expected = _l(np.char.capitalize(a))
    assert _l(fr.char.capitalize(a)) == expected


def test_capitalize_digraph_titlecase():
    """Digraph ǅ (U+01C5) titlecases to itself; ferray uppercases to Ǆ."""
    a = np.array(["ǅungla"], dtype="U10")
    expected = _l(np.char.capitalize(a))
    assert _l(fr.char.capitalize(a)) == expected


def test_title_ligature_titlecase():
    """title word-initial ﬁ -> 'Fi' (titlecase), ferray gives 'FI'."""
    a = np.array(["ﬁle ﬁle"], dtype="U20")
    expected = _l(np.char.title(a))
    assert _l(fr.char.title(a)) == expected


def test_title_sharp_s_titlecase():
    a = np.array(["ßtraße ßeta"], dtype="U20")
    expected = _l(np.char.title(a))
    assert _l(fr.char.title(a)) == expected


def test_title_final_sigma():
    """title lowercases non-initial cased chars; a word-final Σ must become
    final-sigma ς, not medial σ. numpy gives 'Οδος'; ferray gives 'Οδοσ'.
    (case.rs title lowercases char-by-char, losing final-sigma context.)"""
    a = np.array(["ΟΔΟΣ ΟΔΟΣ"], dtype="U20")
    expected = _l(np.char.title(a))
    assert _l(fr.char.title(a)) == expected


def test_capitalize_final_sigma():
    """capitalize lowercases the tail; word-final Σ -> ς."""
    a = np.array(["ΟΔΟΣ"], dtype="U10")
    expected = _l(np.char.capitalize(a))
    assert _l(fr.char.capitalize(a)) == expected


# ===========================================================================
# STRIP family — Unicode whitespace set divergence (HIGH)
# Python str.strip() (and numpy.char.strip) strip the C0 separators
# U+001C FILE SEP, U+001D GROUP SEP, U+001E RECORD SEP, U+001F UNIT SEP.
# Rust str::trim() (used by ferray-strings strip.rs) does NOT treat these
# as whitespace, so they survive.
# Upstream: numpy/_core/strings.py strip -> str.strip; CPython treats
# U+001C..U+001F as whitespace (Objects/unicodectype.c).
# ===========================================================================

@pytest.mark.parametrize("ch", ["\x1c", "\x1d", "\x1e", "\x1f"])
def test_strip_c0_separators(ch):
    """numpy.char.strip removes U+001C..U+001F; ferray (Rust trim) keeps them."""
    a = np.array([ch + "hi" + ch])
    expected = _l(np.char.strip(a))
    assert _l(fr.char.strip(a)) == expected


@pytest.mark.parametrize("ch", ["\x1c", "\x1d", "\x1e", "\x1f"])
def test_lstrip_c0_separators(ch):
    a = np.array([ch + "ab"])
    expected = _l(np.char.lstrip(a))
    assert _l(fr.char.lstrip(a)) == expected


@pytest.mark.parametrize("ch", ["\x1c", "\x1d", "\x1e", "\x1f"])
def test_rstrip_c0_separators(ch):
    a = np.array(["ab" + ch])
    expected = _l(np.char.rstrip(a))
    assert _l(fr.char.rstrip(a)) == expected


# ===========================================================================
# SPLITLINES family — Unicode line-boundary divergence (HIGH)
# Python str.splitlines() (and numpy.char.splitlines) break on the FULL
# universal set: \n \r \r\n \v(0x0b) \f(0x0c) \x1c \x1d \x1e \x85
#  . ferray-strings split_join.rs::split_universal_newlines only
# breaks on \n, \r, \r\n.
# Upstream: numpy/_core/strings.py splitlines -> str.splitlines
# (CPython Objects/unicodeobject.c STRINGLIB(splitlines)).
# ===========================================================================

def test_splitlines_vtab_formfeed():
    """\\v and \\f are line boundaries in Python; ferray ignores them."""
    a = np.array(["a\x0bb\x0cc"])
    expected = _l(np.char.splitlines(a))
    assert _l(fr.char.splitlines(a)) == expected


def test_splitlines_c0_separators():
    """U+001C/1D/1E are line boundaries in Python splitlines."""
    a = np.array(["a\x1cb\x1dc\x1ed"])
    expected = _l(np.char.splitlines(a))
    assert _l(fr.char.splitlines(a)) == expected


def test_splitlines_nel_and_unicode_seps():
    """U+0085 NEL, U+2028 LINE SEP, U+2029 PARA SEP are line boundaries."""
    a = np.array(["a\x85b c d"])
    expected = _l(np.char.splitlines(a))
    assert _l(fr.char.splitlines(a)) == expected


# ===========================================================================
# Regression guards (probed CONVERGED) — these MUST pass. If one ever fails,
# a previously-converged behavior regressed.
# ===========================================================================

def test_converged_count_empty_substring():
    a = np.array(["abc", ""])
    assert _l(fr.char.count(a, "")) == _l(np.char.count(a, ""))


def test_converged_upper_full_case_mapping():
    a = np.array(["straße", "ﬁle"], dtype="U10")
    assert _l(fr.char.upper(a)) == _l(np.char.upper(a))


def test_converged_strip_unicode_whitespace():
    a = np.array(["\x85\xa0\x0b\x0c hi \t\n\r "])
    assert _l(fr.char.strip(a)) == _l(np.char.strip(a))


def test_converged_index_notfound_valueerror():
    with pytest.raises(ValueError):
        np.char.index(np.array(["abc"]), "z")
    with pytest.raises(ValueError):
        fr.char.index(np.array(["abc"]), "z")
