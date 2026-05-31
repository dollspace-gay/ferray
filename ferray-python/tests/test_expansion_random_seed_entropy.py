"""Divergence pin #913 — `fr.random.seed()` / `seed(None)` entropy reseed.

numpy's legacy `numpy.random.seed` accepts the tri-state
`None | int | array_like[int]`:

  - `seed()` / `seed(None)`  -> reseed the global generator from OS entropy;
    returns ``None``; subsequent draws are fresh and NOT reproducible across
    two `seed()` calls.
  - `seed(int)`              -> reseed deterministically (reproducible).
  - `seed(array_like[int])`  -> folded via SeedSequence (reproducible).

Before the fix, ferray's `seed` signature was `seed(s: u64)` (mandatory int),
so `fr.random.seed()` / `fr.random.seed(None)` raised ``TypeError`` instead of
reseeding. Expected values below are taken from the live numpy oracle
(numpy 2.4.x), never literal-copied from the ferray side (R-CHAR-3).
"""

import numpy as np
import pytest

import ferray as fr


def test_seed_no_arg_returns_none_like_numpy():
    # numpy contract: seed() returns None.
    assert np.random.seed() is None
    assert fr.random.seed() is None


def test_seed_none_returns_none_like_numpy():
    assert np.random.seed(None) is None
    assert fr.random.seed(None) is None


def test_seed_no_arg_reseeds_from_entropy_nonreproducible():
    # numpy: two seed() calls produce different streams (fresh entropy).
    np.random.seed()
    na = np.random.random()
    np.random.seed()
    nb = np.random.random()
    assert na != nb  # numpy oracle: non-reproducible

    fr.random.seed()
    fa = fr.random.random()
    fr.random.seed()
    fb = fr.random.random()
    assert fa != fb  # ferray must match the entropy-reseed contract


def test_seed_none_reseeds_from_entropy_nonreproducible():
    np.random.seed(None)
    na = np.random.random()
    np.random.seed(None)
    nb = np.random.random()
    assert na != nb

    fr.random.seed(None)
    fa = fr.random.random()
    fr.random.seed(None)
    fb = fr.random.random()
    assert fa != fb


def test_seed_int_is_deterministic_and_reproducible():
    # numpy oracle: seed(42) -> identical stream across calls.
    np.random.seed(42)
    n1 = np.random.random()
    np.random.seed(42)
    n2 = np.random.random()
    assert n1 == n2

    fr.random.seed(42)
    f1 = fr.random.random()
    fr.random.seed(42)
    f2 = fr.random.random()
    assert f1 == f2  # int-seed reproducibility preserved (#910 round-trip)


def test_seed_array_like_accepted_and_reproducible():
    # numpy accepts an array seed (folded via SeedSequence) and it is
    # reproducible across two calls. ferray folds to a single u64 — the
    # CONTRACT is "accepted + reproducible", not bit-identical to PCG64.
    np.random.seed([1, 2, 3])
    n1 = np.random.random()
    np.random.seed([1, 2, 3])
    n2 = np.random.random()
    assert n1 == n2

    fr.random.seed([1, 2, 3])
    f1 = fr.random.random()
    fr.random.seed([1, 2, 3])
    f2 = fr.random.random()
    assert f1 == f2


def test_seed_int_differs_from_other_int_stream():
    # Distinct deterministic seeds yield distinct streams (both sides).
    fr.random.seed(1)
    a = fr.random.random()
    fr.random.seed(2)
    b = fr.random.random()
    assert a != b


@pytest.mark.parametrize("bad", ["foo", 1.5])
def test_seed_non_int_non_array_raises_typeerror_like_numpy(bad):
    # numpy raises TypeError for a str / float seed.
    with pytest.raises(TypeError):
        np.random.seed(bad)
    with pytest.raises(TypeError):
        fr.random.seed(bad)
