"""Divergence pins — re-audit of acto-builder commit 469e7816 (#920/#921/#922).

The commit added complex-input coercion for the main array path and complex
dispatch for unary transcendentals, routing ops with NO ferray-ufunc complex
loop through `reject_complex_unary` (TypeError). The audit found two ops that
the builder routed to RAISE on complex input, but for which **numpy COMPUTES a
complex result**:

  - ``sinc``  — numpy defines it as ``sin(pi*x)/(pi*x)`` (works for complex);
    ``np.sinc([1+2j])`` returns ``[-34.0903387-17.04516935j]``.
    (upstream: numpy/lib/_function_base_impl.py:3743 ``def sinc`` -> ``sin(y)/y``)
  - ``rint``  — numpy registers a complex loop:
    ``TD('fdg' + cmplx, f='rint')`` so ``np.rint([1.4+2.6j])`` -> ``[1.+3.j]``.
    (upstream: numpy/_core/code_generators/generate_umath.py:1027)

In both cases ferray raises ``TypeError``. Per R-DEV-2 (match numpy's
exception-vs-value contract) and R-DEV-1 (numerical contract) this is a
divergence: ferray must compute, not raise.

Expected values are taken from a LIVE numpy call (R-CHAR-3) — never copied from
the ferray side. These tests are EXPECTED TO FAIL against commit 469e7816.

Tracking: see crosslink blockers filed alongside this file.
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# Divergence 1: sinc(complex) — numpy computes, ferray raises TypeError.
# upstream: numpy/lib/_function_base_impl.py:3743  (sin(pi*x)/(pi*x))
# ---------------------------------------------------------------------------
def test_divergence_sinc_complex_numpy_computes_ferray_raises():
    """numpy COMPUTES sinc for complex input; ferray raises TypeError.

    Divergence: ferray's `sinc` is routed to reject_complex_unary in
    ferray-python/src/ufunc.rs (`bind_unary_float!(sinc, ...)`, no complex
    form), but numpy's sinc is `sin(pi*x)/(pi*x)`, valid for complex.
    numpy returns a complex128 value; ferray raises.
    """
    z = np.array([1 + 2j])
    expected = np.sinc(z)  # live numpy oracle (R-CHAR-3)
    assert expected.dtype == np.complex128

    result = np.asarray(fr.sinc(fr.array([1 + 2j])))
    assert result.dtype == expected.dtype, (
        f"sinc complex dtype: numpy {expected.dtype}, ferray {result.dtype}"
    )
    assert np.allclose(result, expected, rtol=1e-10), (
        f"sinc complex value: numpy {expected}, ferray {result}"
    )


def test_divergence_sinc_complex64_width_preserved():
    """numpy preserves complex64 width for sinc; ferray raises."""
    z = np.array([1 + 2j], dtype=np.complex64)
    expected = np.sinc(z)  # live numpy oracle
    assert expected.dtype == np.complex64

    result = np.asarray(fr.sinc(fr.asarray(z)))
    assert result.dtype == np.complex64, (
        f"sinc c64 width: numpy {expected.dtype}, ferray {result.dtype}"
    )
    assert np.allclose(result, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# Divergence 2: rint(complex) — numpy computes (complex loop), ferray raises.
# upstream: numpy/_core/code_generators/generate_umath.py:1027
#           TD('fdg' + cmplx, f='rint')
# ---------------------------------------------------------------------------
def test_divergence_rint_complex_numpy_computes_ferray_raises():
    """numpy registers a complex `rint` loop; ferray raises TypeError.

    Divergence: ferray's `rint` is routed to reject_complex_unary
    (`bind_unary_float!(rint, ...)`, no complex form). numpy rounds the real
    and imaginary parts independently and returns complex128.
    """
    z = np.array([1.4 + 2.6j])
    expected = np.rint(z)  # live numpy oracle (R-CHAR-3)
    assert expected.dtype == np.complex128

    result = np.asarray(fr.rint(fr.array([1.4 + 2.6j])))
    assert result.dtype == expected.dtype, (
        f"rint complex dtype: numpy {expected.dtype}, ferray {result.dtype}"
    )
    assert np.array_equal(result, expected), (
        f"rint complex value: numpy {expected}, ferray {result}"
    )


def test_divergence_rint_complex64_width_preserved():
    """numpy preserves complex64 width for rint; ferray raises."""
    z = np.array([1.4 + 2.6j], dtype=np.complex64)
    expected = np.rint(z)  # live numpy oracle
    assert expected.dtype == np.complex64

    result = np.asarray(fr.rint(fr.asarray(z)))
    assert result.dtype == np.complex64, (
        f"rint c64 width: numpy {expected.dtype}, ferray {result.dtype}"
    )
    assert np.array_equal(result, expected)


# ---------------------------------------------------------------------------
# Guard: the ops that genuinely have NO numpy complex loop must keep raising.
# This is NOT a divergence — it documents the correct half of the split so a
# future fix for sinc/rint does not over-correct cbrt/fabs/i0/degrees.
# Verified live: numpy raises TypeError for each of these on complex input.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["cbrt", "fabs", "i0", "degrees", "radians",
                                  "deg2rad", "rad2deg", "spacing"])
def test_reject_ops_match_numpy_raise(name):
    """These ops correctly raise on complex in BOTH numpy and ferray."""
    z = np.array([1 + 2j])
    np_fn = getattr(np, name)
    with pytest.raises(TypeError):
        np_fn(z)  # confirm numpy itself raises (oracle)
    fr_fn = getattr(fr, name)
    with pytest.raises(TypeError):
        fr_fn(fr.array([1 + 2j]))
