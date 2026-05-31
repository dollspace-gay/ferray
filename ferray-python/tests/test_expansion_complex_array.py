"""Main-array complex input-coercion + unary-transcendental tests (epic #919,
blockers #920/#921/#922) for ferray-python vs numpy 2.4.5.

Oracle is the LIVE `numpy` package (R-CHAR-3): every expected value is produced
by calling numpy in-process (`np.sqrt(...)`, `np.exp(...)`, ...), never
literal-copied from the ferray side.

Three divergence classes are pinned here:

  R-1 / #920 — input coercion accepts complex: `fr.array([1+2j])` /
    `fr.asarray(np_complex)` must build a complex ndarray, not raise
    `TypeError: unsupported dtype: "complex128"`.

  R-2 / #921 — STOP the silent imaginary-discard (R-CODE-4): `fr.sqrt(z)` on a
    complex input must NOT emit a real-only result + `ComplexWarning`. It must
    compute the complex value (numpy's behavior).

  R-3 / #922 — the named complex transcendentals compute complex via the
    `ferray_ufunc::*_complex` family.

Upstream numpy registration (read-only oracle tree):
  numpy/_core/code_generators/generate_umath.py:966-973  `sqrt` registers
    `'fdg' + cmplx` (a complex loop); `exp`/`log`/`sin`/`cos`/... likewise.
    `cbrt`/`fabs`/`floor` register NO complex loop -> numpy raises TypeError.
"""

import warnings

import numpy as np
import pytest

import ferray as fr


# Transcendentals that numpy computes elementwise-complex (each has a
# ferray_ufunc::*_complex op, except exp2 which the binding composes via
# num_complex). Name -> (ferray fn, numpy fn).
COMPLEX_TRANSCENDENTALS = [
    "sqrt",
    "exp",
    "exp2",
    "expm1",
    "log",
    "log2",
    "log10",
    "log1p",
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "arcsin",
    "arccos",
    "arctan",
    "arcsinh",
    "arccosh",
    "arctanh",
]

# ufuncs numpy registers with NO complex loop -> TypeError on complex input.
COMPLEX_RAISERS = ["cbrt", "fabs", "degrees", "radians", "deg2rad", "rad2deg"]

# Interior (off-real-axis) sample points: every one of the 20 transcendentals
# computes these correctly. The real-axis branch-cut points (`im == +0`,
# `|re| > 1`) are excluded here because three ops (arcsin/arccos/arctanh) pick
# the opposite branch-cut SIGN to numpy at those points — a num_complex vs C99
# library divergence in `ferray_ufunc::*_complex` (NOT this binding), pinned
# separately in `test_arc_branch_cut_real_axis_known_divergence` below.
SAMPLE = [1 + 2j, -3 + 1j, 0.5 - 0.25j, -2 + 3j, 4 - 1j]

# Ops whose num_complex branch cut matches numpy on the real axis too — safe
# to assert at `|re| > 1, im == +0` points.
REAL_AXIS_SAFE = [
    "sqrt",
    "exp",
    "exp2",
    "expm1",
    "log",
    "log2",
    "log10",
    "log1p",
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "arctan",
    "arcsinh",
    "arccosh",
]


# ---------------------------------------------------------------------------
# R-1 / #920 — input coercion accepts complex
# ---------------------------------------------------------------------------


def test_array_python_complex_list_builds_complex128():
    r = fr.array([1 + 2j, 3 - 1j])
    assert np.asarray(r).dtype == np.dtype("complex128")
    assert np.array_equal(np.asarray(r), np.array([1 + 2j, 3 - 1j]))


def test_array_explicit_complex64_dtype_preserves_width():
    r = fr.array([1 + 2j], dtype="complex64")
    assert np.asarray(r).dtype == np.dtype("complex64")
    assert np.allclose(np.asarray(r), np.array([1 + 2j], dtype="complex64"))


def test_asarray_complex128_ndarray_round_trips():
    src = np.array([1 + 2j, -4 + 0.5j])
    r = fr.asarray(src)
    assert np.asarray(r).dtype == np.dtype("complex128")
    assert np.array_equal(np.asarray(r), src)


def test_asarray_complex64_preserves_dtype():
    src = np.array([1 + 2j, -4 + 0.5j], dtype="complex64")
    r = fr.asarray(src)
    assert np.asarray(r).dtype == np.dtype("complex64")
    assert np.array_equal(np.asarray(r), src)


def test_copy_complex_round_trips():
    src = np.array([[1 + 1j, 2 - 2j], [3 + 0j, -1 + 4j]])
    r = fr.copy(src)
    assert np.asarray(r).dtype == np.dtype("complex128")
    assert np.array_equal(np.asarray(r), src)


# ---------------------------------------------------------------------------
# R-2 / #921 — STOP the silent imaginary-discard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", COMPLEX_TRANSCENDENTALS)
def test_transcendental_emits_no_complex_warning(name):
    """numpy computes complex; ferray must too. The OLD path coerced complex
    -> float32 (dropping imag) and emitted ComplexWarning. Assert no warning
    AND a complex result (the R-CODE-4 imag-drop guard)."""
    z = fr.array([1 + 2j, -3 + 1j])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any ComplexWarning becomes an error
        r = getattr(fr, name)(z)
    assert np.asarray(r).dtype == np.dtype("complex128"), name


def test_sqrt_complex_no_imag_discard_matches_numpy():
    z_in = np.array([1 + 2j])
    r = fr.sqrt(fr.array(z_in))
    assert np.allclose(np.asarray(r), np.sqrt(z_in))
    # The imaginary part must be non-zero (the corruption produced [1+0j]).
    assert np.asarray(r).imag[0] != 0.0


# ---------------------------------------------------------------------------
# R-3 / #922 — complex transcendentals compute complex (oracle: live numpy)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", COMPLEX_TRANSCENDENTALS)
def test_transcendental_complex128_matches_numpy(name):
    z = np.array(SAMPLE, dtype="complex128")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # numpy itself may warn on branch cuts
        expected = getattr(np, name)(z)
        got = np.asarray(getattr(fr, name)(fr.array(z)))
    assert got.dtype == np.dtype("complex128"), name
    assert np.allclose(got, expected, equal_nan=True), (name, got, expected)


@pytest.mark.parametrize("name", COMPLEX_TRANSCENDENTALS)
def test_transcendental_complex64_preserves_width_matches_numpy(name):
    z = np.array(SAMPLE, dtype="complex64")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = getattr(np, name)(z)
        got = np.asarray(getattr(fr, name)(fr.array(z, dtype="complex64")))
    assert got.dtype == np.dtype("complex64"), name
    assert np.allclose(got, expected, equal_nan=True, atol=1e-5), name


@pytest.mark.parametrize("name", REAL_AXIS_SAFE)
def test_transcendental_real_axis_branch_matches_numpy(name):
    """For the ops whose branch cut agrees with numpy, also assert the
    real-axis points (`im == +0`, `|re| > 1`) — the place where sqrt's complex
    branch (np.sqrt([-1+0j]) == 1j) and log's branch matter."""
    z = np.array([-1 + 0j, 2 + 0j, -4 + 0j, 3 + 0j], dtype="complex128")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = getattr(np, name)(z)
        got = np.asarray(getattr(fr, name)(fr.array(z)))
    assert np.allclose(got, expected, equal_nan=True), (name, got, expected)


@pytest.mark.parametrize("name", ["arcsin", "arccos", "arctanh"])
@pytest.mark.parametrize("re", [2.0, -2.0])
@pytest.mark.parametrize("im_sign", [0.0, -0.0])
def test_arc_branch_cut_real_axis_matches_numpy_signed_zero(name, re, im_sign):
    """C99 signed-zero branch cut (fixed in ferray-ufunc/src/ops/complex.rs,
    closes #926). At a real argument with `|re| > 1` and a `±0` imaginary part,
    numpy follows C99: the result's imaginary sign is fixed by the sign of the
    input's imaginary signed-zero (arcsin/arctanh) or its negation (arccos).
    num_complex ignored the input zero sign — ferray now corrects it. Both the
    magnitude AND the sign must match numpy (regression coverage, not just a
    documented divergence). Expected from a LIVE numpy call (R-CHAR-3)."""
    z = np.array([complex(re, im_sign)], dtype="complex128")
    got = np.asarray(getattr(fr, name)(fr.array(z)))[0]
    expected = getattr(np, name)(z)[0]
    assert np.isclose(got.real, expected.real), (name, re, im_sign, got, expected)
    assert np.isclose(got.imag, expected.imag), (name, re, im_sign, got, expected)
    # exact signed-zero / sign agreement at the cut
    assert np.signbit(got.imag) == np.signbit(expected.imag), (
        name,
        re,
        im_sign,
        got,
        expected,
    )


def test_sqrt_negative_real_complex_takes_complex_branch():
    """sqrt of a negative real *complex* value uses the complex branch:
    np.sqrt([-1+0j]) == [0+1j] (NOT nan, which the real loop gives)."""
    z = np.array([-1 + 0j, -4 + 0j])
    got = np.asarray(fr.sqrt(fr.array(z)))
    assert np.allclose(got, np.sqrt(z))
    assert np.allclose(got, np.array([1j, 2j]))


def test_log10_complex_matches_numpy():
    z = np.array([1 + 2j, -1 + 0j])
    got = np.asarray(fr.log10(fr.array(z)))
    assert np.allclose(got, np.log10(z))


# ---------------------------------------------------------------------------
# Real arrays UNCHANGED — the complex branch must not perturb the real path.
# ---------------------------------------------------------------------------


def test_sqrt_real_array_unchanged_no_complex():
    got = np.asarray(fr.sqrt(fr.array([4.0, 9.0, 16.0])))
    assert got.dtype == np.dtype("float64")
    assert np.array_equal(got, np.array([2.0, 3.0, 4.0]))


def test_sqrt_negative_real_float_is_nan_not_complex():
    """A negative REAL (float) input stays on the real loop: np.sqrt(-1.0) is
    nan, NOT 1j (numpy only goes complex when the *input* is complex)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = np.asarray(fr.sqrt(fr.array([-1.0])))
    assert got.dtype == np.dtype("float64")
    assert np.isnan(got[0])


def test_exp_int_array_promotes_to_float_not_complex():
    got = np.asarray(fr.exp(fr.array([0, 1, 2])))
    assert got.dtype == np.dtype("float64")
    assert np.allclose(got, np.exp(np.array([0, 1, 2])))


# ---------------------------------------------------------------------------
# Raisers — numpy has NO complex loop, so these must TypeError (not corrupt).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", COMPLEX_RAISERS)
def test_no_complex_loop_raises_typeerror(name):
    z = fr.array([1 + 2j])
    # numpy raises TypeError; ferray must raise (NEVER silently drop imag).
    with pytest.raises(TypeError):
        getattr(fr, name)(z)
