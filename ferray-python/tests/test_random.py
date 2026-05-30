"""Phase-3 parity tests for ferray.random.

These don't compare to numpy's exact RNG output (different bit
generators) — they verify shape, dtype, range invariants, and
seed determinism.
"""

import numpy as np
import pytest

import ferray


# ---------------------------------------------------------------------------
# Seed determinism
# ---------------------------------------------------------------------------


def test_seed_makes_random_deterministic():
    ferray.random.seed(42)
    a = ferray.random.random(size=10)
    ferray.random.seed(42)
    b = ferray.random.random(size=10)
    np.testing.assert_array_equal(a, b)


def test_seed_makes_normal_deterministic():
    ferray.random.seed(42)
    a = ferray.random.normal(size=10)
    ferray.random.seed(42)
    b = ferray.random.normal(size=10)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_produce_different_output():
    ferray.random.seed(1)
    a = ferray.random.random(size=10)
    ferray.random.seed(2)
    b = ferray.random.random(size=10)
    assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Distributions: shape and dtype
# ---------------------------------------------------------------------------


def test_random_returns_float64():
    a = ferray.random.random(size=10)
    assert a.dtype == np.float64
    assert a.shape == (10,)


def test_random_2d_shape():
    a = ferray.random.random(size=(3, 4))
    assert a.shape == (3, 4)


def test_random_in_unit_interval():
    ferray.random.seed(42)
    a = ferray.random.random(size=1000)
    assert (a >= 0).all() and (a < 1).all()


def test_random_no_size_returns_python_float():
    ferray.random.seed(42)
    v = ferray.random.random()
    assert isinstance(v, float)
    assert 0.0 <= v < 1.0


def test_rand_varargs():
    ferray.random.seed(42)
    a = ferray.random.rand(3, 4)
    assert a.shape == (3, 4)


def test_standard_normal_shape():
    a = ferray.random.standard_normal(size=(5, 5))
    assert a.shape == (5, 5)


def test_randn_varargs():
    ferray.random.seed(42)
    a = ferray.random.randn(3, 4)
    assert a.shape == (3, 4)


def test_normal_loc_scale():
    ferray.random.seed(42)
    a = ferray.random.normal(loc=10.0, scale=0.001, size=10000)
    assert abs(a.mean() - 10.0) < 0.01


def test_uniform_range():
    ferray.random.seed(42)
    a = ferray.random.uniform(low=-1.0, high=1.0, size=1000)
    assert (a >= -1.0).all() and (a < 1.0).all()


# ---------------------------------------------------------------------------
# integers / randint
# ---------------------------------------------------------------------------


def test_integers_range():
    ferray.random.seed(42)
    a = ferray.random.integers(0, 10, size=100)
    assert a.dtype == np.int64
    assert (a >= 0).all() and (a < 10).all()


def test_integers_one_arg_is_high():
    ferray.random.seed(42)
    a = ferray.random.integers(5, size=100)
    assert (a >= 0).all() and (a < 5).all()


def test_randint_alias_for_integers():
    ferray.random.seed(42)
    a = ferray.random.randint(0, 10, size=20)
    ferray.random.seed(42)
    b = ferray.random.integers(0, 10, size=20)
    np.testing.assert_array_equal(a, b)


def test_integers_no_size_returns_python_int():
    ferray.random.seed(42)
    v = ferray.random.integers(0, 100)
    assert isinstance(v, int)
    assert 0 <= v < 100


# ---------------------------------------------------------------------------
# Permutation / shuffle / choice
# ---------------------------------------------------------------------------


def test_permutation_int_arg():
    ferray.random.seed(42)
    p = ferray.random.permutation(10)
    assert p.shape == (10,)
    assert sorted(p.tolist()) == list(range(10))


def test_permutation_array_arg():
    ferray.random.seed(42)
    src = np.array([1, 2, 3, 4, 5])
    p = ferray.random.permutation(src)
    assert sorted(p.tolist()) == [1, 2, 3, 4, 5]


def test_shuffle_permutes_in_place():
    # numpy/random/__init__.py:45 — shuffle permutes a sequence IN PLACE and
    # returns None. The array passed in is mutated; no copy is returned.
    ferray.random.seed(42)
    src = np.array([1, 2, 3, 4, 5])
    ret = ferray.random.shuffle(src)
    assert ret is None
    # Same elements, permuted in place.
    assert sorted(src.tolist()) == [1, 2, 3, 4, 5]


def test_choice_with_replacement():
    ferray.random.seed(42)
    src = np.array([1, 2, 3, 4, 5])
    out = ferray.random.choice(src, size=10, replace=True)
    assert out.shape == (10,)
    # All values come from src.
    assert all(v in src for v in out.tolist())


def test_choice_without_replacement():
    ferray.random.seed(42)
    src = np.array([1, 2, 3, 4, 5])
    out = ferray.random.choice(src, size=3, replace=False)
    # No duplicates.
    assert len(set(out.tolist())) == 3


def test_choice_int_arg_means_arange():
    ferray.random.seed(42)
    out = ferray.random.choice(10, size=5)
    assert out.shape == (5,)
    assert all(0 <= v < 10 for v in out.tolist())
