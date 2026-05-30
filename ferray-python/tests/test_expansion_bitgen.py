"""API-compatibility tests for the numpy.random BitGenerator / RandomState /
SeedSequence class surface (refs #834 #818).

These verify the numpy *API contract* — class existence, constructibility,
output shape/dtype, acceptance by ``default_rng``, legacy ``RandomState``
methods, and seeded reproducibility of ferray's own Xoshiro256** stream.

They deliberately do NOT compare against numpy's exact bit-stream: ferray
backs every BitGenerator with a single Xoshiro256** PRNG, so PCG64 /
MT19937 / Philox / SFC64 are API-compatible wrappers, not numpy-bit-identical
streams (documented limitation in ``src/random.rs``). Expected shapes/dtypes
come from live ``numpy`` calls (R-CHAR-3).
"""

import numpy as np
import pytest

import ferray as fr


BITGENS = ["MT19937", "PCG64", "PCG64DXSM", "Philox", "SFC64"]


# ---------------------------------------------------------------------------
# Class existence + construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", BITGENS + ["BitGenerator", "RandomState", "SeedSequence"])
def test_class_exists(name):
    assert hasattr(fr.random, name), f"fr.random.{name} missing"
    # numpy exposes the same name.
    assert hasattr(np.random, name)


@pytest.mark.parametrize("name", BITGENS)
def test_bitgen_constructs_with_seed(name):
    cls = getattr(fr.random, name)
    bg = cls(42)
    assert bg is not None
    # numpy BitGenerators carry a `.state` dict with a `bit_generator` key.
    np_keys = set(getattr(np.random, name)(42).state.keys())
    assert "bit_generator" in np_keys  # confirm the contract we mirror
    assert isinstance(bg.state, dict)
    assert "bit_generator" in bg.state
    assert bg.state["bit_generator"] == name


@pytest.mark.parametrize("name", BITGENS)
def test_bitgen_constructs_default(name):
    cls = getattr(fr.random, name)
    bg = cls()  # seed=None path
    assert isinstance(bg.state, dict)


@pytest.mark.parametrize("name", BITGENS)
def test_bitgen_is_bitgenerator_subclass(name):
    cls = getattr(fr.random, name)
    assert issubclass(cls, fr.random.BitGenerator)


@pytest.mark.parametrize("name", BITGENS)
def test_bitgen_seed_seq(name):
    bg = getattr(fr.random, name)(7)
    assert isinstance(bg.seed_seq, fr.random.SeedSequence)


# ---------------------------------------------------------------------------
# default_rng(BitGenerator) — accepts an instance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", BITGENS)
def test_default_rng_accepts_bitgenerator(name):
    bg = getattr(fr.random, name)(42)
    g = fr.random.default_rng(bg)
    out = g.random(3)
    # numpy contract: Generator name + shape (3,) float64.
    ref = np.random.default_rng(getattr(np.random, name)(42)).random(3)
    assert out.shape == ref.shape == (3,)
    assert out.dtype == ref.dtype == np.float64


def test_default_rng_still_accepts_int_and_none():
    assert fr.random.default_rng(0).random(2).shape == (2,)
    assert fr.random.default_rng().random(2).shape == (2,)


def test_default_rng_accepts_seedsequence():
    ss = fr.random.SeedSequence(123)
    g = fr.random.default_rng(ss)
    assert g.random(4).shape == (4,)


def test_default_rng_bitgen_reproducible():
    # Same seed via a BitGenerator -> same ferray stream (reproducibility, NOT
    # numpy-bit-identity).
    a = fr.random.default_rng(fr.random.PCG64(99)).random(5)
    b = fr.random.default_rng(fr.random.PCG64(99)).random(5)
    np.testing.assert_array_equal(a, b)


def test_bitgen_seed_matches_int_seed():
    # default_rng(PCG64(7)) and default_rng(7) re-seed the same Xoshiro stream.
    a = fr.random.default_rng(fr.random.MT19937(7)).random(5)
    b = fr.random.default_rng(7).random(5)
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# RandomState — legacy API surface
# ---------------------------------------------------------------------------


def test_randomstate_constructs():
    rs = fr.random.RandomState(0)
    assert rs is not None


def test_randomstate_rand_shape():
    rs = fr.random.RandomState(0)
    out = rs.rand(3)
    ref = np.random.RandomState(0).rand(3)
    assert out.shape == ref.shape == (3,)
    assert out.dtype == ref.dtype == np.float64
    # scalar form
    assert np.isscalar(fr.random.RandomState(0).rand())


def test_randomstate_randn_shape():
    out = fr.random.RandomState(0).randn(2, 3)
    ref = np.random.RandomState(0).randn(2, 3)
    assert out.shape == ref.shape == (2, 3)
    assert out.dtype == ref.dtype


def test_randomstate_randint_shape_dtype():
    out = fr.random.RandomState(0).randint(0, 10, 5)
    ref = np.random.RandomState(0).randint(0, 10, 5)
    assert out.shape == ref.shape == (5,)
    assert np.issubdtype(out.dtype, np.integer)
    assert (out >= 0).all() and (out < 10).all()


def test_randomstate_random_sample_and_random():
    rs = fr.random.RandomState(1)
    assert rs.random_sample(4).shape == (4,)
    assert rs.random(4).shape == (4,)


def test_randomstate_standard_normal_normal_uniform():
    rs = fr.random.RandomState(2)
    assert rs.standard_normal(3).shape == (3,)
    assert rs.normal(0.0, 1.0, 3).shape == (3,)
    assert rs.uniform(0.0, 1.0, 3).shape == (3,)


def test_randomstate_choice():
    rs = fr.random.RandomState(3)
    out = rs.choice(5, size=4)
    assert out.shape == (4,)
    assert ((out >= 0) & (out < 5)).all()
    # scalar form
    assert np.isscalar(fr.random.RandomState(3).choice(5))


def test_randomstate_shuffle_inplace():
    rs = fr.random.RandomState(4)
    a = np.arange(10)
    rs.shuffle(a)
    assert sorted(a.tolist()) == list(range(10))


def test_randomstate_permutation():
    rs = fr.random.RandomState(5)
    p = rs.permutation(10)
    assert sorted(p.tolist()) == list(range(10))
    arr = rs.permutation(np.arange(6))
    assert sorted(arr.tolist()) == list(range(6))


def test_randomstate_distributions():
    rs = fr.random.RandomState(6)
    assert rs.exponential(1.0, 3).shape == (3,)
    assert rs.poisson(2.0, 3).shape == (3,)
    assert rs.binomial(10, 0.5, 3).shape == (3,)


def test_randomstate_seed_reseed_reproducible():
    rs = fr.random.RandomState(7)
    a = rs.rand(5)
    rs.seed(7)
    b = rs.rand(5)
    np.testing.assert_array_equal(a, b)


def test_randomstate_seed_constructor_reproducible():
    a = fr.random.RandomState(123).rand(8)
    b = fr.random.RandomState(123).rand(8)
    np.testing.assert_array_equal(a, b)


def test_randomstate_get_set_state_roundtrip():
    rs = fr.random.RandomState(11)
    rs.rand(3)  # advance
    st = rs.get_state()
    a = rs.rand(5)
    rs.set_state(st)
    b = rs.rand(5)
    np.testing.assert_array_equal(a, b)


def test_randomstate_binomial_negative_raises():
    with pytest.raises(ValueError):
        fr.random.RandomState(0).binomial(-1, 0.5)


# ---------------------------------------------------------------------------
# SeedSequence
# ---------------------------------------------------------------------------


def test_seedsequence_entropy():
    ss = fr.random.SeedSequence(42)
    assert ss.entropy == 42
    # numpy contract: .entropy echoes the constructor entropy.
    assert np.random.SeedSequence(42).entropy == 42


def test_seedsequence_generate_state_shape_dtype():
    ss = fr.random.SeedSequence(42)
    out = ss.generate_state(4)
    ref = np.random.SeedSequence(42).generate_state(4)
    assert out.shape == ref.shape == (4,)
    assert out.dtype == ref.dtype == np.uint32


def test_seedsequence_generate_state_deterministic():
    a = fr.random.SeedSequence(42).generate_state(8)
    b = fr.random.SeedSequence(42).generate_state(8)
    np.testing.assert_array_equal(a, b)


def test_seedsequence_different_entropy_differs():
    a = fr.random.SeedSequence(42).generate_state(8)
    b = fr.random.SeedSequence(43).generate_state(8)
    assert not np.array_equal(a, b)


def test_seedsequence_spawn():
    ss = fr.random.SeedSequence(42)
    children = ss.spawn(3)
    assert len(children) == 3
    assert all(isinstance(c, fr.random.SeedSequence) for c in children)
    # children produce distinct states.
    states = [tuple(c.generate_state(4).tolist()) for c in children]
    assert len(set(states)) == 3


def test_seedsequence_default_entropy_present():
    ss = fr.random.SeedSequence()
    # numpy assigns random entropy when none is given; ferray does too.
    assert ss.entropy is not None


# ---------------------------------------------------------------------------
# module-level legacy get_state / set_state
# ---------------------------------------------------------------------------


def test_module_get_set_state_roundtrip():
    fr.random.seed(21)
    fr.random.random(3)  # advance
    st = fr.random.get_state()
    a = fr.random.random(5)
    fr.random.set_state(st)
    b = fr.random.random(5)
    np.testing.assert_array_equal(a, b)


def test_module_get_state_tuple_shape():
    st = fr.random.get_state()
    ref = np.random.get_state()
    # numpy returns a 5-tuple (str, ndarray, int, int, float).
    assert len(st) == len(ref) == 5
    assert isinstance(st[0], str)
