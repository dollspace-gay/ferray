"""ACToR critic CONVERGENCE-GATE pins: ferray.ma view fill-discriminator vs numpy.ma.

Final-convergence audit of the ferray.ma ``MaskedArray`` VIEW overlay system
after commit 508a8a5e (#882). The full overlay-interaction probe matrix
(own_fill+own_mask together on nomask and real-mask bases, own_hard+own_mask
hard-protection, chained-view fill tracking, view-as-operand reductions
(sum/count/compressed/getmaskarray with own_mask), array/slice/fancy setitem
write-through, own_mask surviving a getitem-subview, 3-level writeback) all MATCH
numpy.ma. The campaign converges on ONE remaining view divergence:

  Divergence (fill discriminator — value-compare vs numpy identity/None):
    numpy's ``MaskedArray._fill_value`` is ``None`` until an explicit
    ``fill_value`` assignment MATERIALIZES it (``None -> 1e20``), regardless of
    whether the assigned value EQUALS the per-dtype default. Once materialized,
    ``__array_finalize__`` shares the base's 0-d ``_fill_value`` array by
    reference, so a view created afterward TRACKS the base's later changes.

    ferray's ``Storage::View { base_fill_is_default }`` is computed in
    ``__getitem__`` by comparing the base's effective ``fill_value`` VALUE to
    the per-dtype default (``base_fill_now.eq(&default_fill)``). When a base is
    explicitly set to a value that HAPPENS to equal the default (float ``1e20``,
    int ``999999``), ferray's value-compare yields ``true`` and treats the view
    as INDEPENDENT (returns the per-dtype default forever, ignoring the base's
    later change). numpy treats the base as MATERIALIZED (identity test
    ``_fill_value is not None``) and the view TRACKS it.

    Live numpy oracle (numpy 2.4.x): ``a = np.ma.array([1.,2.,3.])`` has
    ``a._fill_value is None``; ``a.fill_value = 1e20`` flips it to ``1e20``
    (materialized). A view ``b = a[1:3]`` taken AFTER materialization then sees
    ``a.fill_value = 5.0`` and reports ``b.fill_value == 5.0``. ferray reports
    the default ``1e20`` (and ``999999`` for int).

Every expected value below is computed by running the SAME operation against
``numpy.ma`` (the live oracle), never literal-copied from the ferray side
(R-CHAR-3 / goal.md). Each test asserts the numpy value and FAILS against the
current ferray implementation.

Tracking issue filed from repo root via crosslink (see report).
"""

import numpy as np
import numpy.ma as nma
import pytest

import ferray as fr


def _np_set_fill(a, v):
    a.fill_value = v


def _fr_set_fill(a, v):
    fr.ma.set_fill_value(a, v)


def test_view_of_explicit_default_float_base_tracks_later_change():
    """Float base explicitly set to the DEFAULT value (1e20) is materialized in
    numpy; a view taken after tracks the base's later ``fill_value`` change.

    numpy: ``a._fill_value`` flips ``None -> 1e20`` on the explicit assignment
    (identity-materialized), so the post-assignment view shares it by reference
    and reports the base's CURRENT value (5.0). ferray's value-compare sees
    ``1e20 == default`` -> treats the view as independent -> returns 1e20.
    """
    # numpy oracle
    a = nma.array([1.0, 2.0, 3.0])
    a.fill_value = 1e20  # explicit set to a value == per-dtype default
    b = a[1:3]
    a.fill_value = 5.0
    expected = float(b.fill_value)  # numpy: 5.0 (materialized base, tracked)

    # ferray under test
    fa = fr.ma.array([1.0, 2.0, 3.0])
    _fr_set_fill(fa, 1e20)
    fb = fa[1:3]
    _fr_set_fill(fa, 5.0)
    actual = float(fb.fill_value)

    assert actual == expected, (
        f"view of explicitly-default-set base must track base's later "
        f"fill_value: numpy={expected}, ferray={actual}"
    )


def test_view_of_explicit_default_int_base_tracks_later_change():
    """Int base explicitly set to the DEFAULT value (999999) is materialized in
    numpy; the post-assignment view tracks the base's later change to 5.

    Same root cause as the float case; the int per-dtype default is 999999.
    """
    a = nma.array([1, 2, 3], dtype=np.int64)
    a.fill_value = 999999  # explicit set == int per-dtype default
    b = a[1:3]
    a.fill_value = 5
    expected = int(b.fill_value)  # numpy: 5

    fa = fr.ma.array([1, 2, 3], dtype="int64")
    _fr_set_fill(fa, 999999)
    fb = fa[1:3]
    _fr_set_fill(fa, 5)
    actual = int(fb.fill_value)

    assert actual == expected, (
        f"int view of explicitly-default-set base must track base's later "
        f"fill_value: numpy={expected}, ferray={actual}"
    )


def test_chained_view_of_explicit_default_base_tracks_later_change():
    """Chained view (c = b[0:1], b = a[1:3]) of an explicitly-default-set base
    tracks the base's later change at BOTH levels.

    numpy: the materialized base is shared by reference through every view in
    the chain, so both b and c report 5.0. ferray's value-compare propagates
    its wrong "independent" verdict through the chain (both report 1e20).
    """
    a = nma.array([1.0, 2.0, 3.0, 4.0])
    a.fill_value = 1e20
    b = a[1:3]
    c = b[0:1]
    a.fill_value = 5.0
    expected = (float(b.fill_value), float(c.fill_value))  # numpy: (5.0, 5.0)

    fa = fr.ma.array([1.0, 2.0, 3.0, 4.0])
    _fr_set_fill(fa, 1e20)
    fb = fa[1:3]
    fc = fb[0:1]
    _fr_set_fill(fa, 5.0)
    actual = (float(fb.fill_value), float(fc.fill_value))

    assert actual == expected, (
        f"chained view of explicitly-default-set base must track base's later "
        f"fill_value at both levels: numpy={expected}, ferray={actual}"
    )


@pytest.mark.parametrize(
    "label,probe_np,probe_fr",
    [
        # own_fill + own_mask together (nomask base): filled uses own_fill (7)
        # and own_mask masks position 0.
        (
            "own_fill+own_mask nomask base",
            lambda: (
                lambda a, b: (
                    _np_set_fill(b, 7),
                    b.__setitem__(0, nma.masked),
                    list(b.filled()),
                )[-1]
            )(
                _a := nma.array([10, 20, 30], dtype=np.int64),
                _a[1:3],
            ),
            lambda: (
                lambda a, b: (
                    _fr_set_fill(b, 7),
                    b.__setitem__(0, fr.ma.masked),
                    list(b.filled()),
                )[-1]
            )(
                _a := fr.ma.array([10, 20, 30], dtype="int64"),
                _a[1:3],
            ),
        ),
    ],
)
def test_overlay_interaction_matrix_regression_guard(label, probe_np, probe_fr):
    """Regression guard for the CONVERGED overlay interactions (these MATCH
    numpy today; this test pins them green so a future fill-discriminator fix
    cannot regress them)."""
    expected = probe_np()
    actual = probe_fr()
    assert [int(x) for x in actual] == [int(x) for x in expected], (
        f"{label}: numpy={expected}, ferray={actual}"
    )
