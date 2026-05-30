"""Contract tests for the multivariate / extra numpy.random distributions
bound over ferray-random (refs #834 #818).

ferray uses a different bit-generator (Xoshiro256**) than numpy's PCG64, so
these tests verify the OBSERVABLE CONTRACT against numpy — output shape,
dtype, value-range / support, row-sum invariants, and seed-reproducibility
within ferray — NEVER bit-exact agreement with numpy's stream.

Each distribution is exercised both module-level (`fr.random.<name>`) and as
a `Generator` method (`fr.random.default_rng(0).<name>`).
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# Shape / dtype contracts derived live from numpy (R-CHAR-3): the expected
# shapes and dtypes are read from a live numpy call, never literal-copied
# from ferray.
# ---------------------------------------------------------------------------


def _np_rng():
    return np.random.default_rng(0)


def test_dirichlet_shape_dtype_and_sum():
    nr = _np_rng()
    # size=None -> 1-D (k,)
    expected_1d = nr.dirichlet([1.0, 2.0, 3.0])
    out = fr.random.dirichlet([1.0, 2.0, 3.0])
    assert out.shape == expected_1d.shape == (3,)
    assert out.dtype == expected_1d.dtype == np.float64
    np.testing.assert_allclose(out.sum(), 1.0, atol=1e-9)
    assert np.all(out >= 0.0)

    # size=int -> (n, k)
    expected_2d = nr.dirichlet([1.0, 1.0], size=5)
    out2 = fr.random.dirichlet([1.0, 1.0], size=5)
    assert out2.shape == expected_2d.shape == (5, 2)
    np.testing.assert_allclose(out2.sum(axis=1), np.ones(5), atol=1e-9)

    # size=tuple -> size + (k,)
    expected_t = nr.dirichlet([1.0, 1.0, 1.0], size=(2, 3))
    out_t = fr.random.dirichlet([1.0, 1.0, 1.0], size=(2, 3))
    assert out_t.shape == expected_t.shape == (2, 3, 3)
    np.testing.assert_allclose(out_t.sum(axis=-1), np.ones((2, 3)), atol=1e-9)


def test_dirichlet_negative_alpha_raises():
    with pytest.raises(ValueError):
        fr.random.dirichlet([-1.0, 1.0])


def test_multinomial_shape_dtype_and_rowsum():
    nr = _np_rng()
    expected_1d = nr.multinomial(10, [0.2, 0.8])
    out = fr.random.multinomial(10, [0.2, 0.8])
    assert out.shape == expected_1d.shape == (2,)
    assert out.dtype == expected_1d.dtype == np.int64
    assert out.sum() == 10

    expected_2d = nr.multinomial(10, [0.2, 0.3, 0.5], size=5)
    out2 = fr.random.multinomial(10, [0.2, 0.3, 0.5], size=5)
    assert out2.shape == expected_2d.shape == (5, 3)
    assert np.all(out2.sum(axis=1) == 10)
    assert np.all(out2 >= 0)

    expected_t = nr.multinomial(7, [0.5, 0.5], size=(2, 3))
    out_t = fr.random.multinomial(7, [0.5, 0.5], size=(2, 3))
    assert out_t.shape == expected_t.shape == (2, 3, 2)
    assert np.all(out_t.sum(axis=-1) == 7)


def test_multinomial_negative_n_raises():
    with pytest.raises(ValueError):
        fr.random.multinomial(-1, [0.5, 0.5])


def test_multivariate_normal_shape_dtype():
    nr = _np_rng()
    mean = [0.0, 0.0]
    cov = [[1.0, 0.0], [0.0, 1.0]]
    expected_1d = nr.multivariate_normal(mean, cov)
    out = fr.random.multivariate_normal(mean, cov)
    assert out.shape == expected_1d.shape == (2,)
    assert out.dtype == expected_1d.dtype == np.float64

    expected_2d = nr.multivariate_normal(mean, cov, size=3)
    out2 = fr.random.multivariate_normal(mean, cov, size=3)
    assert out2.shape == expected_2d.shape == (3, 2)

    expected_t = nr.multivariate_normal(mean, cov, size=(2, 3))
    out_t = fr.random.multivariate_normal(mean, cov, size=(2, 3))
    assert out_t.shape == expected_t.shape == (2, 3, 2)


def test_multivariate_hypergeometric_shape_dtype_and_rowsum():
    nr = _np_rng()
    colors = [3, 4, 5]
    nsample = 6
    expected_1d = nr.multivariate_hypergeometric(colors, nsample)
    out = fr.random.multivariate_hypergeometric(colors, nsample)
    assert out.shape == expected_1d.shape == (3,)
    assert out.dtype == expected_1d.dtype == np.int64
    assert out.sum() == nsample
    assert np.all(out >= 0)
    # each color count cannot exceed the available items of that color
    assert np.all(out <= np.array(colors))

    expected_2d = nr.multivariate_hypergeometric(colors, nsample, size=4)
    out2 = fr.random.multivariate_hypergeometric(colors, nsample, size=4)
    assert out2.shape == expected_2d.shape == (4, 3)
    assert np.all(out2.sum(axis=1) == nsample)


def test_multivariate_hypergeometric_negative_raises():
    with pytest.raises(ValueError):
        fr.random.multivariate_hypergeometric([3, -1], 2)


def test_logseries_shape_dtype_and_support():
    nr = _np_rng()
    expected = nr.logseries(0.5, size=4)
    out = fr.random.logseries(0.5, size=4)
    assert out.shape == expected.shape == (4,)
    assert out.dtype == expected.dtype == np.int64
    # support is {1, 2, 3, ...}
    assert np.all(out >= 1)


def test_logseries_bad_p_raises():
    with pytest.raises(ValueError):
        fr.random.logseries(1.5)


def test_noncentral_chisquare_shape_dtype_and_range():
    nr = _np_rng()
    expected = nr.noncentral_chisquare(3.0, 2.0, size=2)
    out = fr.random.noncentral_chisquare(3.0, 2.0, size=2)
    assert out.shape == expected.shape == (2,)
    assert out.dtype == expected.dtype == np.float64
    assert np.all(out >= 0.0)


def test_noncentral_chisquare_bad_df_raises():
    with pytest.raises(ValueError):
        fr.random.noncentral_chisquare(-1.0, 2.0)


def test_noncentral_f_shape_dtype_and_range():
    nr = _np_rng()
    expected = nr.noncentral_f(3.0, 4.0, 2.0, size=2)
    out = fr.random.noncentral_f(3.0, 4.0, 2.0, size=2)
    assert out.shape == expected.shape == (2,)
    assert out.dtype == expected.dtype == np.float64
    assert np.all(out >= 0.0)


def test_zipf_shape_dtype_and_support():
    nr = _np_rng()
    expected = nr.zipf(2.0, size=3)
    out = fr.random.zipf(2.0, size=3)
    assert out.shape == expected.shape == (3,)
    assert out.dtype == expected.dtype == np.int64
    assert np.all(out >= 1)


def test_zipf_bad_a_raises():
    with pytest.raises(ValueError):
        fr.random.zipf(0.5)


# ---------------------------------------------------------------------------
# Generator-method surface: fr.random.default_rng(seed).<name>
# ---------------------------------------------------------------------------


def test_generator_methods_exist_and_match_shape():
    g = fr.random.default_rng(0)
    nr = _np_rng()

    assert g.dirichlet([1.0, 2.0]).shape == nr.dirichlet([1.0, 2.0]).shape
    assert (
        g.multinomial(5, [0.5, 0.5], size=3).shape
        == nr.multinomial(5, [0.5, 0.5], size=3).shape
    )
    cov = [[1.0, 0.0], [0.0, 1.0]]
    assert (
        g.multivariate_normal([0.0, 0.0], cov, size=4).shape
        == nr.multivariate_normal([0.0, 0.0], cov, size=4).shape
    )
    assert (
        g.multivariate_hypergeometric([2, 3], 4, size=2).shape
        == nr.multivariate_hypergeometric([2, 3], 4, size=2).shape
    )
    assert g.logseries(0.5, size=4).shape == nr.logseries(0.5, size=4).shape
    assert (
        g.noncentral_chisquare(3.0, 2.0, size=2).shape
        == nr.noncentral_chisquare(3.0, 2.0, size=2).shape
    )
    assert (
        g.noncentral_f(3.0, 4.0, 2.0, size=2).shape
        == nr.noncentral_f(3.0, 4.0, 2.0, size=2).shape
    )
    assert g.zipf(2.0, size=3).shape == nr.zipf(2.0, size=3).shape


def test_generator_dirichlet_rowsum():
    g = fr.random.default_rng(7)
    out = g.dirichlet([1.0, 2.0, 3.0], size=10)
    assert out.shape == (10, 3)
    np.testing.assert_allclose(out.sum(axis=1), np.ones(10), atol=1e-9)


def test_generator_multinomial_rowsum():
    g = fr.random.default_rng(7)
    out = g.multinomial(12, [0.1, 0.2, 0.7], size=6)
    assert np.all(out.sum(axis=1) == 12)


# ---------------------------------------------------------------------------
# Seed reproducibility within ferray (same seed -> same sequence). This is a
# valid contract check and is INTERNAL to ferray (not compared to numpy).
# ---------------------------------------------------------------------------


def test_module_seed_reproducible_dirichlet():
    fr.random.seed(123)
    a = fr.random.dirichlet([1.0, 2.0, 3.0], size=4)
    fr.random.seed(123)
    b = fr.random.dirichlet([1.0, 2.0, 3.0], size=4)
    np.testing.assert_array_equal(a, b)


def test_module_seed_reproducible_zipf():
    fr.random.seed(99)
    a = fr.random.zipf(2.5, size=8)
    fr.random.seed(99)
    b = fr.random.zipf(2.5, size=8)
    np.testing.assert_array_equal(a, b)


def test_generator_seed_reproducible():
    a = fr.random.default_rng(2024).multinomial(20, [0.25, 0.25, 0.5], size=5)
    b = fr.random.default_rng(2024).multinomial(20, [0.25, 0.25, 0.5], size=5)
    np.testing.assert_array_equal(a, b)


def test_generator_noncentral_f_reproducible():
    a = fr.random.default_rng(55).noncentral_f(3.0, 4.0, 2.0, size=6)
    b = fr.random.default_rng(55).noncentral_f(3.0, 4.0, 2.0, size=6)
    np.testing.assert_array_equal(a, b)
