"""Seed range validation parity (#914).

numpy applies *different* integer bounds to a scalar seed depending on the
entry point:

  - legacy ``numpy.random.seed`` / ``RandomState`` -> ``RandomState.seed`` is
    ``uint32``-bounded: an int outside ``[0, 2**32 - 1]`` raises
    ``ValueError("Seed must be between 0 and 2**32 - 1")``.
  - new-style ``default_rng`` / ``Generator`` route a scalar int through
    ``SeedSequence``: any non-negative int (arbitrary size, incl. ``> 2**64``)
    is accepted; a negative int raises
    ``ValueError("expected non-negative integer")``.

A non-int (float / str) keeps numpy's ``TypeError`` in both cases.

Expected values come from the live numpy oracle (R-CHAR-3), asserted below
against ``import ferray as fr``.
"""

import ferray as fr
import numpy as np
import pytest


# --- legacy seed bound [0, 2**32 - 1] -------------------------------------

@pytest.mark.parametrize("bad", [-1, 2**32, 2**40])
def test_legacy_seed_out_of_range_valueerror(bad):
    # numpy oracle: legacy seed raises ValueError outside [0, 2**32-1].
    with pytest.raises(ValueError):
        np.random.seed(bad)
    with pytest.raises(ValueError):
        fr.random.seed(bad)


@pytest.mark.parametrize("ok", [0, 1, 2**32 - 1])
def test_legacy_seed_in_range_ok(ok):
    np.random.seed(ok)  # numpy: no error
    fr.random.seed(ok)  # ferray: no error


def test_legacy_randomstate_out_of_range_valueerror():
    with pytest.raises(ValueError):
        np.random.RandomState(-1)
    with pytest.raises(ValueError):
        fr.random.RandomState(-1)
    with pytest.raises(ValueError):
        fr.random.RandomState(2**32)


# --- default_rng / Generator: non-negative (arbitrary size) ----------------

def test_default_rng_negative_valueerror():
    with pytest.raises(ValueError):
        np.random.default_rng(-1)
    with pytest.raises(ValueError):
        fr.random.default_rng(-1)
    with pytest.raises(ValueError):
        fr.random.default_rng(-5)


@pytest.mark.parametrize("ok", [0, 1, 2**32, 2**64, 2**64 + 5])
def test_default_rng_non_negative_ok(ok):
    np.random.default_rng(ok)  # numpy: accepts arbitrary non-negative int
    fr.random.default_rng(ok)  # ferray: accepts (big ints folded)


def test_generator_negative_valueerror():
    with pytest.raises(ValueError):
        fr.random.Generator(-1)


def test_default_rng_big_int_reproducible():
    # A big (>2**64) non-negative seed must fold deterministically.
    a = fr.random.default_rng(2**64 + 5).random(4)
    b = fr.random.default_rng(2**64 + 5).random(4)
    assert (a == b).all()


# --- non-int seed -> TypeError (both numpy and ferray) ---------------------

@pytest.mark.parametrize("bad", ["x", 1.5])
def test_non_int_seed_typeerror(bad):
    with pytest.raises(TypeError):
        np.random.seed(bad)
    with pytest.raises(TypeError):
        fr.random.seed(bad)
    with pytest.raises(TypeError):
        np.random.default_rng(bad)
    with pytest.raises(TypeError):
        fr.random.default_rng(bad)


# --- non-regression: int reproducibility (#913) ----------------------------

def test_int_seed_reproducible_unregressed():
    fr.random.seed(42)
    x = fr.random.random(5)
    fr.random.seed(42)
    y = fr.random.random(5)
    assert (x == y).all()
