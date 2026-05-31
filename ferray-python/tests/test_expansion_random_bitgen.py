"""Coverage for the last numpy public-callable gap (closes #910):
``np.random.get_bit_generator`` / ``np.random.set_bit_generator`` and the
``Generator.bit_generator`` attribute.

Expectations are derived from the LIVE numpy contract (R-CHAR-3): each test
asserts the *observable* contract numpy exposes — the functions exist, accept
and return BitGenerator objects, ``set_bit_generator`` returns ``None``, and
the global stream round-trips/reproduces deterministically. Bit-exact
agreement with numpy's MT19937/PCG64 stream is NOT asserted: ferray's
fr.random is Xoshiro256**-backed per its documented contract
(``ferray-python/src/random.rs`` BitGenerator family docs), so the requirement
is API + round-trip + reproducibility, not numpy-bit-identity.
"""

import numpy as np
import ferray as fr

# The ferray BitGenerator classes — the return/accept type set.
FR_BITGEN_CLASSES = (
    fr.random.BitGenerator,
    fr.random.MT19937,
    fr.random.PCG64,
    fr.random.PCG64DXSM,
    fr.random.Philox,
    fr.random.SFC64,
)


# --------------------------------------------------------------------------
# numpy contract probes (the oracle): establish what numpy does so the ferray
# assertions below trace to live numpy behaviour, not literal-copied values.
# --------------------------------------------------------------------------

def test_numpy_contract_shapes():
    # get_bit_generator returns a BitGenerator object; set returns None.
    bg = np.random.get_bit_generator()
    assert isinstance(bg, np.random.BitGenerator)
    assert np.random.set_bit_generator(bg) is None
    # Generator.bit_generator is a BitGenerator object.
    assert isinstance(np.random.default_rng(42).bit_generator,
                      np.random.BitGenerator)


# --------------------------------------------------------------------------
# ferray: existence + type contract (mirrors numpy's observable contract).
# --------------------------------------------------------------------------

def test_get_bit_generator_returns_bitgenerator_object():
    bg = fr.random.get_bit_generator()
    assert isinstance(bg, FR_BITGEN_CLASSES)
    assert isinstance(bg, fr.random.BitGenerator)


def test_set_bit_generator_returns_none():
    # numpy: set_bit_generator(...) -> None.
    bg = fr.random.get_bit_generator()
    assert fr.random.set_bit_generator(bg) is None


def test_set_bit_generator_rejects_non_bitgenerator():
    # A non-BitGenerator argument is a TypeError at the boundary.
    import pytest
    with pytest.raises(TypeError):
        fr.random.set_bit_generator(object())
    with pytest.raises(TypeError):
        fr.random.set_bit_generator(123)


def test_generator_bit_generator_attribute():
    g = fr.random.default_rng(42)
    bg = g.bit_generator
    assert isinstance(bg, fr.random.BitGenerator)


def test_randomstate_underscore_bit_generator():
    # numpy RandomState exposes ._bit_generator (mtrand.pyi:86), an MT19937.
    rs = fr.random.RandomState(7)
    assert isinstance(rs._bit_generator, fr.random.BitGenerator)


# --------------------------------------------------------------------------
# ferray: round-trip + reproducibility (the load-bearing contract).
# --------------------------------------------------------------------------

def test_get_set_round_trip_is_reproducible():
    # set_bit_generator(get_bit_generator()) reinstalls the SAME stream:
    # capturing the bitgen, then re-installing it twice and drawing must
    # yield the identical sequence each time.
    bg = fr.random.get_bit_generator()
    fr.random.set_bit_generator(bg)
    a = fr.random.random(5)
    fr.random.set_bit_generator(bg)
    b = fr.random.random(5)
    np.testing.assert_array_equal(a, b)


def test_set_bit_generator_installs_reproducible_stream():
    # After set_bit_generator(MT19937(7)) the global stream is deterministic
    # and reproducible across two identical installs (matches numpy's
    # reproducible-after-set contract).
    fr.random.set_bit_generator(fr.random.MT19937(7))
    a = fr.random.random(4)
    fr.random.set_bit_generator(fr.random.MT19937(7))
    b = fr.random.random(4)
    np.testing.assert_array_equal(a, b)


def test_set_bit_generator_changes_stream_for_distinct_seeds():
    # Distinct seeds installed via set_bit_generator give distinct streams
    # (the installed generator actually drives top-level draws).
    fr.random.set_bit_generator(fr.random.MT19937(1))
    a = fr.random.random(8)
    fr.random.set_bit_generator(fr.random.MT19937(2))
    b = fr.random.random(8)
    assert not np.array_equal(a, b)


def test_generator_bit_generator_round_trips_into_global():
    # A Generator's bit_generator can be installed as the global, and the
    # global then reproduces deterministically when re-installed.
    g = fr.random.default_rng(123)
    fr.random.set_bit_generator(g.bit_generator)
    a = fr.random.random(3)
    fr.random.set_bit_generator(g.bit_generator)
    b = fr.random.random(3)
    np.testing.assert_array_equal(a, b)


def test_coverage_gap_closed():
    # The whole point of #910: get/set_bit_generator are the last two numpy
    # public callables ferray lacked. After this, the only numpy.random
    # public callable absent from fr.random is 'test' (pytest infra).
    nn = set(
        n for n in dir(np.random)
        if not n.startswith("_") and callable(getattr(np.random, n, None))
    )
    missing = sorted(nn - set(dir(fr.random)))
    assert missing == ["test"], f"unexpected missing numpy.random callables: {missing}"
