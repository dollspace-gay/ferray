"""Adversarial divergence suite for ferray.random vs numpy 2.4 (oracle).

These tests pin CONTRACT divergences (shape / dtype / scalar-vs-array /
exception TYPE / missing symbols / negative-dim handling), NOT bit-exact
RNG output — ferray uses Xoshiro256** while numpy uses PCG64/MT19937, so
value-stream equality is out of scope and never asserted here.

Each test FAILS on current ferray and documents the numpy contract it
violates with a numpy file:line citation. None of these are fixes.
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# Missing legacy aliases (numpy/random/mtrand.pyi:975,991-993 — random_sample,
# sample, ranf are all exposed as module-level legacy aliases of random_sample;
# __init__.py:159-162 lists ranf/random_sample/sample in __all__).
# ---------------------------------------------------------------------------


def test_random_sample_alias_exists():
    """numpy/random/__init__.py:158 + mtrand.pyi:975 — `random_sample` is a
    legacy module-level function returning float64 of given shape."""
    a = fr.random.random_sample(5)
    assert a.shape == (5,)
    assert a.dtype == np.float64


def test_sample_alias_exists():
    """numpy/random/mtrand.pyi:992 — `sample = _rand.random_sample`."""
    a = fr.random.sample(5)
    assert a.shape == (5,)


def test_ranf_alias_exists():
    """numpy/random/mtrand.pyi:993 — `ranf = _rand.random_sample`."""
    a = fr.random.ranf(5)
    assert a.shape == (5,)


def test_random_integers_exists():
    """numpy/random/__init__.py:157 — legacy `random_integers` (inclusive
    high). ferray omits it entirely."""
    assert hasattr(fr.random, "random_integers")


def test_bytes_exists():
    """numpy/random/__init__.py:128 + docstring line 'bytes' — random bytes
    generator. ferray omits it entirely."""
    assert hasattr(fr.random, "bytes")


# ---------------------------------------------------------------------------
# choice(a, size=None) must return a SCALAR, not a length-1 array.
# numpy/random/mtrand.pyi:379-385 — choice(a:int, size:None) -> int;
# mtrand.pyi:394-401 — choice(a:ArrayLike, size:None) -> Any (a scalar element).
# ---------------------------------------------------------------------------


def test_choice_int_no_size_returns_scalar():
    """mtrand.pyi:385 — choice(int, size=None) returns a scalar int, not a
    1-D array. numpy: np.isscalar(np.random.choice(5)) is True."""
    fr.random.seed(0)
    v = fr.random.choice(5)
    assert np.ndim(v) == 0, f"expected 0-d scalar, got ndim={np.ndim(v)}"


def test_choice_array_no_size_returns_scalar():
    """mtrand.pyi:401 — choice(array, size=None) returns a single element
    (0-d / scalar) of the array's dtype, not a length-1 array."""
    fr.random.seed(0)
    src = np.array([10, 20, 30])
    v = fr.random.choice(src)
    assert np.ndim(v) == 0, f"expected scalar element, got shape {getattr(v,'shape',None)}"


def test_choice_size_tuple_supported():
    """mtrand.pyi:387-393 — choice accepts size as a _ShapeLike (tuple),
    producing a multi-dim result. ferray types size as `usize` and rejects
    tuples with TypeError."""
    fr.random.seed(0)
    out = fr.random.choice(5, size=(2, 3))
    assert out.shape == (2, 3)


# ---------------------------------------------------------------------------
# shuffle is documented (numpy/random/__init__.py:46) and typed
# (mtrand.pyi:933) as IN-PLACE returning None. ferray returns a copy and
# leaves the input untouched — opposite contract.
# ---------------------------------------------------------------------------


def test_shuffle_is_in_place_returns_none():
    """mtrand.pyi:933 — `def shuffle(self, x) -> None`. numpy mutates x in
    place and returns None. ferray returns a shuffled copy and does not
    mutate the input."""
    fr.random.seed(0)
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    original = x.copy()
    ret = fr.random.shuffle(x)
    assert ret is None, "shuffle must return None (in-place contract)"
    # The input must have been permuted in place (overwhelmingly likely for n=8).
    assert not np.array_equal(x, original), "shuffle must mutate input in place"


# ---------------------------------------------------------------------------
# permutation must accept N-D arrays (permute along axis 0) and negative
# int. numpy/random/mtrand.pyi:937-939 — permutation(int)->array,
# permutation(ArrayLike)->array. ferray only handles 1-D arrays.
# ---------------------------------------------------------------------------


def test_permutation_2d_array():
    """numpy permutes a 2-D array along axis 0 and preserves shape
    (numpy/random/mtrand.pyi:939, ArrayLike overload). ferray's binding
    extracts PyReadonlyArray1 and raises TypeError on 2-D input."""
    fr.random.seed(0)
    arr2d = np.arange(12).reshape(4, 3)
    out = fr.random.permutation(arr2d)
    assert out.shape == (4, 3)


def test_permutation_negative_int_empty():
    """numpy: np.random.permutation(-3) -> empty int64 array. ferray's int
    extraction fails on negatives and falls through to the array path,
    raising TypeError instead of returning an empty array."""
    fr.random.seed(0)
    out = fr.random.permutation(-3)
    assert isinstance(out, np.ndarray)
    assert out.shape == (0,)


# ---------------------------------------------------------------------------
# Negative-dimension handling — numpy raises ValueError("negative dimensions
# are not allowed") consistently. ferray raises TypeError / OverflowError.
# ---------------------------------------------------------------------------


def test_random_negative_size_valueerror():
    """numpy: np.random.random(-1) raises ValueError. ferray raises
    TypeError ('size must be an int, a tuple of ints, or None')."""
    with pytest.raises(ValueError):
        fr.random.random(-1)


def test_rand_negative_dim_valueerror():
    """numpy: np.random.rand(-1) raises ValueError. ferray raises
    OverflowError when extracting the varargs into usize."""
    with pytest.raises(ValueError):
        fr.random.rand(-1)


def test_random_negative_tuple_dim_valueerror():
    """numpy: np.random.random((2, -1)) raises ValueError. ferray raises
    TypeError because the tuple-of-usize extraction fails on a negative."""
    with pytest.raises(ValueError):
        fr.random.random((2, -1))


# ---------------------------------------------------------------------------
# binomial(n<0) — numpy raises ValueError("n < 0"). ferray binds n as u64
# and raises OverflowError at the PyO3 boundary before any check.
# ---------------------------------------------------------------------------


def test_binomial_negative_n_valueerror():
    """numpy: np.random.binomial(-1, 0.5) raises ValueError('n < 0').
    ferray binds n: u64 so a negative raises OverflowError instead."""
    with pytest.raises(ValueError):
        fr.random.binomial(-1, 0.5, size=3)
