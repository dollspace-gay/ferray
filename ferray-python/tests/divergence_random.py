"""Adversarial divergence suite for ferray.random vs numpy 2.4 (oracle).

These tests pin CONTRACT divergences (shape / dtype / scalar-vs-array /
exception TYPE / missing symbols / negative-dim handling), NOT bit-exact
RNG output — ferray uses Xoshiro256** while numpy uses PCG64/MT19937, so
value-stream equality is out of scope and never asserted here.

Every expected value is obtained from a LIVE numpy call in-test (the
`np` oracle) or from a numpy `file:line` symbolic contract — never
literal-copied from the ferray side (R-CHAR-3).

Each test FAILS on current ferray and documents the numpy contract it
violates with a numpy file:line citation. None of these are fixes.

Re-audited by acto-critic; supersedes the prior stand-in pins.
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# Missing legacy module-level aliases. numpy/random/__init__.py:126-177 lists
# these in `__all__`; they are live attributes of `numpy.random`.
# ---------------------------------------------------------------------------


def test_random_sample_alias_exists():
    """numpy/random/__init__.py:159 — `random_sample` is in __all__ and is a
    live module attribute (the canonical name `random` aliases)."""
    assert hasattr(np.random, "random_sample")  # oracle: numpy exposes it
    assert hasattr(fr.random, "random_sample")


def test_sample_alias_exists():
    """numpy/random/__init__.py:162 — `sample` is in __all__ (legacy alias of
    random_sample)."""
    assert hasattr(np.random, "sample")  # oracle
    assert hasattr(fr.random, "sample")


def test_ranf_alias_exists():
    """numpy/random/__init__.py:160 — `ranf` is in __all__ (legacy alias)."""
    assert hasattr(np.random, "ranf")  # oracle
    assert hasattr(fr.random, "ranf")


def test_random_integers_exists():
    """numpy/random/__init__.py:158 — legacy `random_integers` (inclusive
    high). ferray omits it entirely."""
    assert hasattr(np.random, "random_integers")  # oracle
    assert hasattr(fr.random, "random_integers")


def test_bytes_exists():
    """numpy/random/__init__.py:129 — `bytes` is in __all__ (random bytes
    generator). ferray omits it entirely."""
    assert hasattr(np.random, "bytes")  # oracle
    assert hasattr(fr.random, "bytes")


# ---------------------------------------------------------------------------
# choice(a, size=None) must return a SCALAR (ndim 0), not a length-1 array.
# numpy/random/mtrand.pyi choice overloads: choice(a, size=None) -> scalar.
# ferray types `size: Option<usize>` defaulting to 1, so size=None yields a
# length-1 array.
# ---------------------------------------------------------------------------


def test_choice_int_no_size_returns_scalar():
    """choice(int, size=None) returns a 0-d scalar in numpy."""
    oracle_ndim = np.ndim(np.random.choice(5))  # oracle: 0
    fr.random.seed(0)
    v = fr.random.choice(5)
    assert np.ndim(v) == oracle_ndim, f"expected ndim {oracle_ndim}, got {np.ndim(v)}"


def test_choice_array_no_size_returns_scalar():
    """choice(array, size=None) returns a single 0-d element in numpy."""
    src = np.array([10, 20, 30])
    oracle_ndim = np.ndim(np.random.choice(src))  # oracle: 0
    fr.random.seed(0)
    v = fr.random.choice(src)
    assert np.ndim(v) == oracle_ndim, f"expected ndim {oracle_ndim}, got {np.ndim(v)}"


def test_choice_size_tuple_supported():
    """choice accepts size as a shape tuple, producing a multi-dim result.
    ferray types size as `usize` and rejects tuples with TypeError."""
    oracle_shape = np.random.choice(5, size=(2, 3)).shape  # oracle: (2, 3)
    fr.random.seed(0)
    out = fr.random.choice(5, size=(2, 3))
    assert out.shape == oracle_shape


# ---------------------------------------------------------------------------
# shuffle is documented (numpy/random/__init__.py:45) as IN-PLACE returning
# None. ferray returns a copy and leaves the input untouched — opposite
# contract.
# ---------------------------------------------------------------------------


def test_shuffle_is_in_place_returns_none():
    """numpy/random/__init__.py:45 — `shuffle: Randomly permute a sequence in
    place.` numpy mutates x and returns None."""
    oracle_x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    oracle_ret = np.random.shuffle(oracle_x)  # oracle: None, oracle_x mutated
    assert oracle_ret is None  # confirm oracle contract

    fr.random.seed(0)
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    original = x.copy()
    ret = fr.random.shuffle(x)
    assert ret is None, "shuffle must return None (in-place contract)"
    assert not np.array_equal(x, original), "shuffle must mutate input in place"


# ---------------------------------------------------------------------------
# permutation must accept N-D arrays (permute along axis 0) and negative int.
# numpy/random/__init__.py:44 — `permutation: Randomly permute a sequence`.
# ferray only handles 1-D arrays and positive ints.
# ---------------------------------------------------------------------------


def test_permutation_2d_array():
    """numpy permutes a 2-D array along axis 0 and preserves shape. ferray
    extracts PyReadonlyArray1 and raises TypeError on 2-D input."""
    src = np.arange(12).reshape(4, 3)
    oracle_shape = np.random.permutation(src).shape  # oracle: (4, 3)
    fr.random.seed(0)
    out = fr.random.permutation(np.arange(12).reshape(4, 3))
    assert out.shape == oracle_shape


def test_permutation_negative_int_empty():
    """numpy: permutation(-3) -> empty int64 array. ferray's usize extraction
    fails on negatives, falls through to the array path, and raises
    TypeError."""
    oracle = np.random.permutation(-3)  # oracle: array([], dtype=int64)
    assert oracle.shape == (0,)
    fr.random.seed(0)
    out = fr.random.permutation(-3)
    assert isinstance(out, np.ndarray)
    assert out.shape == (0,)


# ---------------------------------------------------------------------------
# Negative-dimension handling — numpy raises ValueError("negative dimensions
# are not allowed") consistently. ferray raises TypeError / OverflowError.
# ---------------------------------------------------------------------------


def _numpy_raises(fn):
    """Return the exception type numpy raises for `fn`, else None."""
    try:
        fn()
        return None
    except Exception as e:  # noqa: BLE001
        return type(e)


def test_random_negative_size_valueerror():
    """numpy: random(-1) raises ValueError. ferray raises TypeError."""
    assert _numpy_raises(lambda: np.random.random(-1)) is ValueError  # oracle
    with pytest.raises(ValueError):
        fr.random.random(-1)


def test_rand_negative_dim_valueerror():
    """numpy: rand(-1) raises ValueError. ferray raises OverflowError when
    extracting the varargs into usize."""
    assert _numpy_raises(lambda: np.random.rand(-1)) is ValueError  # oracle
    with pytest.raises(ValueError):
        fr.random.rand(-1)


def test_random_negative_tuple_dim_valueerror():
    """numpy: random((2, -1)) raises ValueError. ferray raises TypeError
    because tuple-of-usize extraction fails on a negative."""
    assert _numpy_raises(lambda: np.random.random((2, -1))) is ValueError  # oracle
    with pytest.raises(ValueError):
        fr.random.random((2, -1))


# ---------------------------------------------------------------------------
# binomial(n<0) — numpy raises ValueError("n < 0"). ferray binds n as u64 and
# raises OverflowError at the PyO3 boundary before any check.
# ---------------------------------------------------------------------------


def test_binomial_negative_n_valueerror():
    """numpy: binomial(-1, 0.5) raises ValueError('n < 0'). ferray binds
    n: u64 so a negative raises OverflowError."""
    assert _numpy_raises(lambda: np.random.binomial(-1, 0.5)) is ValueError  # oracle
    with pytest.raises(ValueError):
        fr.random.binomial(-1, 0.5, size=3)


# ---------------------------------------------------------------------------
# NEW: missing module-level univariate distributions. numpy/random/__init__.py
# __all__ lists these (lines 126-177); ferray binds ~15 distributions but
# omits this representative set. The underlying ferray_random::Generator
# implements them (design doc REQ-8/REQ-9) — it is a binding-surface gap.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "logistic",           # __init__.py:142
        "power",              # __init__.py:153
        "standard_t",         # __init__.py:170 (np alias standard_t)
        "f",                  # __init__.py:134
        "vonmises",           # __init__.py:173
        "wald",               # __init__.py:174
        "standard_cauchy",    # __init__.py:166
        "standard_exponential",  # __init__.py:167
        "standard_gamma",     # __init__.py:168
        "negative_binomial",  # __init__.py:146
        "hypergeometric",     # __init__.py:139
    ],
)
def test_module_level_distribution_present(name):
    """numpy/random/__init__.py:126-177 — `<name>` is a module-level random
    function. ferray binds its siblings (normal, gamma, …) but omits this
    one entirely."""
    assert hasattr(np.random, name)  # oracle: numpy exposes it
    assert hasattr(fr.random, name), f"ferray.random is missing `{name}`"


# ---------------------------------------------------------------------------
# NEW: modern Generator class (numpy/random/__init__.py:6 — "Use default_rng()
# to create a Generator") is the canonical API. ferray's PyGenerator omits
# choice / permutation / shuffle and most distributions present at module
# level.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["choice", "permutation", "shuffle"])
def test_generator_sampling_methods_present(name):
    """numpy Generator exposes choice/permutation/shuffle. ferray's
    PyGenerator omits them (they exist only as module-level functions)."""
    assert hasattr(np.random.default_rng(0), name)  # oracle
    g = fr.random.default_rng(0)
    assert hasattr(g, name), f"ferray Generator is missing `{name}`"


@pytest.mark.parametrize(
    "name",
    ["laplace", "gumbel", "triangular", "rayleigh", "weibull", "lognormal", "geometric"],
)
def test_generator_distribution_methods_present(name):
    """numpy Generator exposes these distribution methods; ferray binds them
    at module level but not on the Generator class."""
    assert hasattr(np.random.default_rng(0), name)  # oracle
    g = fr.random.default_rng(0)
    assert hasattr(g, name), f"ferray Generator is missing `{name}`"


# ---------------------------------------------------------------------------
# NEW: default_rng must accept a sequence / SeedSequence-compatible seed.
# numpy/random/__init__.py:181 imports default_rng; its signature accepts
# `seed: None | int | array_like[int] | SeedSequence | BitGenerator`.
# ferray types seed as `Option<u64>` and rejects a list with TypeError.
# ---------------------------------------------------------------------------


def test_default_rng_accepts_sequence_seed():
    """numpy: default_rng([1, 2, 3]) succeeds (array_like seed). ferray binds
    seed: u64 and raises TypeError on a list."""
    np.random.default_rng([1, 2, 3])  # oracle: succeeds, no exception
    fr.random.default_rng([1, 2, 3])  # ferray: raises TypeError today
