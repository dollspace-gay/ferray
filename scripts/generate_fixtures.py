#!/usr/bin/env python3
"""
Generate NumPy oracle fixtures for the ferray test suite.

Produces JSON files under fixtures/ organized by subcrate.
Each fixture contains inputs, expected outputs, and tolerance information.

Usage:
    python3 scripts/generate_fixtures.py
"""

import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def _serialize_value(v):
    """Convert a single value to JSON-safe representation."""
    if isinstance(v, (np.complexfloating, complex)):
        return {"re": _serialize_value(v.real), "im": _serialize_value(v.imag)}
    if isinstance(v, float) or isinstance(v, np.floating):
        if np.isnan(v):
            return "NaN"
        if np.isposinf(v):
            return "Inf"
        if np.isneginf(v):
            return "-Inf"
        return float(repr_float(v))
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, str):
        return v
    return float(v)


def repr_float(x):
    """Return a high-precision float string."""
    return float(np.format_float_positional(x, unique=True, trim="-"))


def array_to_dict(arr, dtype_str=None):
    """Convert a numpy array to a JSON-serializable dict."""
    if np.isscalar(arr) or (isinstance(arr, np.ndarray) and arr.ndim == 0):
        val = arr.item() if isinstance(arr, np.ndarray) else arr
        dt = dtype_str or (str(arr.dtype) if isinstance(arr, np.ndarray) else type(val).__name__)
        return {"data": _serialize_value(val), "shape": [], "dtype": dt}

    flat = arr.ravel()
    data = [_serialize_value(x) for x in flat]
    dt = dtype_str or str(arr.dtype)
    return {"data": data, "shape": list(arr.shape), "dtype": dt}


def save_fixture(subdir, filename, fixture):
    """Write a fixture dict to JSON."""
    path = FIXTURES_DIR / subdir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    # Inject schema v2 metadata at write time so all generators benefit.
    # The two fields go to the FRONT of the dict (insertion-order preserved).
    if "fixture_schema_version" not in fixture:
        meta = {"numpy_version": np.__version__, "fixture_schema_version": 2}
        meta.update(fixture)
        fixture = meta
    with open(path, "w") as f:
        json.dump(fixture, f, indent=2)
        f.write("\n")  # trailing newline matches retrofit script's normalized form
    print(f"  -> {path.relative_to(FIXTURES_DIR.parent)}")


def make_fixture(np_func_name, ferray_func_name, test_cases):
    """Build a fixture dict."""
    return {
        "function": np_func_name,
        "ferray_function": ferray_func_name,
        "test_cases": test_cases,
    }


def case(name, inputs, expected, tolerance_ulps=4):
    """Build a single test case dict."""
    return {
        "name": name,
        "inputs": inputs,
        "expected": expected,
        "tolerance_ulps": tolerance_ulps,
    }


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------

def generate_core_fixtures():
    print("Generating core fixtures...")

    # --- arange ---
    cases = []
    for start, stop, step, dt in [
        (0, 10, 1, "float64"),
        (0, 10, 2, "float64"),
        (1.5, 5.5, 0.5, "float64"),
        (0, 5, 1, "int32"),
        (10, 0, -2, "float64"),
        (0, 0, 1, "float64"),
    ]:
        arr = np.arange(start, stop, step, dtype=dt)
        cases.append(case(
            f"arange_{start}_{stop}_{step}_{dt}",
            {"start": start, "stop": stop, "step": step, "dtype": dt},
            array_to_dict(arr, dt),
            tolerance_ulps=0,
        ))
    save_fixture("core", "arange.json", make_fixture("numpy.arange", "ferray_core::arange", cases))

    # --- linspace ---
    cases = []
    for start, stop, num, dt in [
        (0.0, 1.0, 5, "float64"),
        (0.0, 1.0, 11, "float64"),
        (-1.0, 1.0, 3, "float64"),
        (0.0, 10.0, 1, "float64"),
        (0.0, 0.0, 5, "float64"),
        (0.0, 1.0, 50, "float32"),
    ]:
        arr = np.linspace(start, stop, num, dtype=dt)
        cases.append(case(
            f"linspace_{start}_{stop}_{num}_{dt}",
            {"start": start, "stop": stop, "num": num, "dtype": dt},
            array_to_dict(arr, dt),
            tolerance_ulps=1,
        ))
    save_fixture("core", "linspace.json", make_fixture("numpy.linspace", "ferray_core::linspace", cases))

    # --- zeros ---
    cases = []
    for shape, dt in [
        ([5], "float64"),
        ([3, 4], "float64"),
        ([2, 3, 4], "float64"),
        ([0], "float64"),
        ([5], "float32"),
        ([5], "int32"),
    ]:
        arr = np.zeros(shape, dtype=dt)
        cases.append(case(
            f"zeros_{'x'.join(map(str, shape))}_{dt}",
            {"shape": shape, "dtype": dt},
            array_to_dict(arr, dt),
            tolerance_ulps=0,
        ))
    save_fixture("core", "zeros.json", make_fixture("numpy.zeros", "ferray_core::zeros", cases))

    # --- ones ---
    cases = []
    for shape, dt in [
        ([5], "float64"),
        ([3, 4], "float64"),
        ([2, 3, 4], "float32"),
        ([0], "float64"),
        ([1], "int64"),
    ]:
        arr = np.ones(shape, dtype=dt)
        cases.append(case(
            f"ones_{'x'.join(map(str, shape))}_{dt}",
            {"shape": shape, "dtype": dt},
            array_to_dict(arr, dt),
            tolerance_ulps=0,
        ))
    save_fixture("core", "ones.json", make_fixture("numpy.ones", "ferray_core::ones", cases))

    # --- eye ---
    cases = []
    for n, m, k, dt in [
        (3, 3, 0, "float64"),
        (4, 4, 0, "float64"),
        (3, 4, 0, "float64"),
        (3, 3, 1, "float64"),
        (3, 3, -1, "float64"),
        (4, 4, 0, "int32"),
    ]:
        arr = np.eye(n, m, k, dtype=dt)
        cases.append(case(
            f"eye_{n}_{m}_k{k}_{dt}",
            {"n": n, "m": m, "k": k, "dtype": dt},
            array_to_dict(arr, dt),
            tolerance_ulps=0,
        ))
    save_fixture("core", "eye.json", make_fixture("numpy.eye", "ferray_core::eye", cases))

    # --- full ---
    cases = []
    for shape, fill, dt in [
        ([5], 3.14, "float64"),
        ([3, 4], 0.0, "float64"),
        ([2, 3], -1.0, "float32"),
        ([0], 1.0, "float64"),
        ([2, 2], 42, "int32"),
    ]:
        arr = np.full(shape, fill, dtype=dt)
        cases.append(case(
            f"full_{'x'.join(map(str, shape))}_{fill}_{dt}",
            {"shape": shape, "fill_value": fill, "dtype": dt},
            array_to_dict(arr, dt),
            tolerance_ulps=0,
        ))
    save_fixture("core", "full.json", make_fixture("numpy.full", "ferray_core::full", cases))

    # --- reshape ---
    cases = []
    src = np.arange(12, dtype="float64")
    for new_shape in [[3, 4], [4, 3], [2, 6], [2, 2, 3], [12], [1, 12]]:
        arr = src.reshape(new_shape)
        cases.append(case(
            f"reshape_{'x'.join(map(str, new_shape))}",
            {"x": array_to_dict(src), "new_shape": new_shape},
            array_to_dict(arr),
            tolerance_ulps=0,
        ))
    save_fixture("core", "reshape.json", make_fixture("numpy.reshape", "ferray_core::reshape", cases))

    # --- transpose ---
    cases = []
    a2d = np.arange(12, dtype="float64").reshape(3, 4)
    cases.append(case("transpose_2d", {"x": array_to_dict(a2d)}, array_to_dict(a2d.T), tolerance_ulps=0))
    a3d = np.arange(24, dtype="float64").reshape(2, 3, 4)
    cases.append(case("transpose_3d", {"x": array_to_dict(a3d)}, array_to_dict(a3d.transpose()), tolerance_ulps=0))
    save_fixture("core", "transpose.json", make_fixture("numpy.transpose", "ferray_core::transpose", cases))

    # --- flatten ---
    cases = []
    a2d = np.arange(12, dtype="float64").reshape(3, 4)
    cases.append(case("flatten_2d", {"x": array_to_dict(a2d)}, array_to_dict(a2d.flatten()), tolerance_ulps=0))
    a3d = np.arange(24, dtype="float64").reshape(2, 3, 4)
    cases.append(case("flatten_3d", {"x": array_to_dict(a3d)}, array_to_dict(a3d.flatten()), tolerance_ulps=0))
    save_fixture("core", "flatten.json", make_fixture("numpy.ndarray.flatten", "ferray_core::flatten", cases))

    # --- squeeze ---
    cases = []
    a = np.arange(6, dtype="float64").reshape(1, 6, 1)
    cases.append(case("squeeze_1x6x1", {"x": array_to_dict(a)}, array_to_dict(a.squeeze()), tolerance_ulps=0))
    a2 = np.arange(4, dtype="float64").reshape(1, 1, 4)
    cases.append(case("squeeze_1x1x4", {"x": array_to_dict(a2)}, array_to_dict(a2.squeeze()), tolerance_ulps=0))
    save_fixture("core", "squeeze.json", make_fixture("numpy.squeeze", "ferray_core::squeeze", cases))

    # --- expand_dims ---
    cases = []
    a = np.arange(5, dtype="float64")
    for axis in [0, 1]:
        r = np.expand_dims(a, axis)
        cases.append(case(f"expand_dims_axis{axis}", {"x": array_to_dict(a), "axis": axis}, array_to_dict(r), tolerance_ulps=0))
    save_fixture("core", "expand_dims.json", make_fixture("numpy.expand_dims", "ferray_core::expand_dims", cases))

    # --- broadcast_shapes ---
    cases = []
    test_pairs = [
        ([4, 3], [3]),
        ([4, 1], [4, 3]),
        ([2, 1, 4], [3, 4]),
        ([5, 1], [1, 6]),
        ([1], [5, 4]),
        ([3, 1, 5], [1, 4, 5]),
    ]
    for s1, s2 in test_pairs:
        result = list(np.broadcast_shapes(tuple(s1), tuple(s2)))
        cases.append(case(
            f"broadcast_{'x'.join(map(str, s1))}_{'x'.join(map(str, s2))}",
            {"shape1": s1, "shape2": s2},
            {"shape": result},
            tolerance_ulps=0,
        ))
    save_fixture("core", "broadcast_shapes.json", make_fixture("numpy.broadcast_shapes", "ferray_core::broadcast_shapes", cases))


# ---------------------------------------------------------------------------
# Ufunc fixtures
# ---------------------------------------------------------------------------

def _standard_float_inputs():
    """Return a dict of standard test arrays for ufuncs."""
    return {
        "small_f64": np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype="float64"),
        "neg_f64": np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype="float64"),
        "wide_f64": np.array([-3.14159265, -1.0, 0.0, 1.0, 3.14159265], dtype="float64"),
        "large_f64": np.array([1e-7, 1e-3, 1.0, 1e3, 1e6], dtype="float64"),
        "special_f64": np.array([float("nan"), float("inf"), float("-inf"), 0.0, -0.0], dtype="float64"),
    }


def _generate_unary_ufunc(np_func, np_name, ferray_name, subdir, filename,
                          inputs_override=None, tolerance=4, extra_cases=None):
    """Generate fixture for a unary ufunc."""
    cases = []

    if inputs_override is not None:
        test_inputs = inputs_override
    else:
        test_inputs = _standard_float_inputs()

    for label, arr in test_inputs.items():
        with np.errstate(all="ignore"):
            result = np_func(arr)
        dt = str(arr.dtype)
        cases.append(case(
            label,
            {"x": array_to_dict(arr, dt)},
            array_to_dict(result, str(result.dtype)),
            tolerance_ulps=tolerance,
        ))

    # f32 variant
    arr32 = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype="float32")
    with np.errstate(all="ignore"):
        r32 = np_func(arr32)
    cases.append(case("standard_f32", {"x": array_to_dict(arr32, "float32")},
                       array_to_dict(r32, "float32"), tolerance_ulps=tolerance))

    # 2D test
    arr2d = np.linspace(-1, 1, 12, dtype="float64").reshape(3, 4)
    with np.errstate(all="ignore"):
        r2d = np_func(arr2d)
    cases.append(case("2d_f64", {"x": array_to_dict(arr2d)}, array_to_dict(r2d), tolerance_ulps=tolerance))

    # 0-D test
    scalar = np.float64(0.5)
    with np.errstate(all="ignore"):
        rscalar = np_func(scalar)
    cases.append(case("scalar_f64", {"x": array_to_dict(np.array(scalar))},
                       array_to_dict(np.array(rscalar)), tolerance_ulps=tolerance))

    # empty array
    empty = np.array([], dtype="float64")
    with np.errstate(all="ignore"):
        rempty = np_func(empty)
    cases.append(case("empty", {"x": array_to_dict(empty)}, array_to_dict(rempty), tolerance_ulps=0))

    # single element
    single = np.array([0.7], dtype="float64")
    with np.errstate(all="ignore"):
        rsingle = np_func(single)
    cases.append(case("single_element", {"x": array_to_dict(single)}, array_to_dict(rsingle), tolerance_ulps=tolerance))

    if extra_cases:
        cases.extend(extra_cases)

    save_fixture(subdir, filename, make_fixture(np_name, ferray_name, cases))


def _generate_binary_ufunc(np_func, np_name, ferray_name, subdir, filename,
                           inputs_override=None, tolerance=4):
    """Generate fixture for a binary ufunc."""
    cases = []

    if inputs_override is not None:
        for label, (a, b) in inputs_override.items():
            with np.errstate(all="ignore"):
                result = np_func(a, b)
            cases.append(case(
                label,
                {"a": array_to_dict(a), "b": array_to_dict(b)},
                array_to_dict(result),
                tolerance_ulps=tolerance,
            ))
    else:
        # Standard pairs
        a1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype="float64")
        b1 = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype="float64")
        with np.errstate(all="ignore"):
            r1 = np_func(a1, b1)
        cases.append(case("standard_f64", {"a": array_to_dict(a1), "b": array_to_dict(b1)},
                          array_to_dict(r1), tolerance_ulps=tolerance))

        # With negatives
        a2 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype="float64")
        b2 = np.array([3.0, 3.0, 3.0, 3.0, 3.0], dtype="float64")
        with np.errstate(all="ignore"):
            r2 = np_func(a2, b2)
        cases.append(case("with_negatives", {"a": array_to_dict(a2), "b": array_to_dict(b2)},
                          array_to_dict(r2), tolerance_ulps=tolerance))

        # Broadcasting scalar
        a3 = np.array([1.0, 2.0, 3.0], dtype="float64")
        b3 = np.array([2.0], dtype="float64")
        with np.errstate(all="ignore"):
            r3 = np_func(a3, b3)
        cases.append(case("broadcast_scalar", {"a": array_to_dict(a3), "b": array_to_dict(b3)},
                          array_to_dict(r3), tolerance_ulps=tolerance))

        # 2D
        a4 = np.arange(12, dtype="float64").reshape(3, 4)
        b4 = np.arange(4, dtype="float64")
        with np.errstate(all="ignore"):
            r4 = np_func(a4, b4)
        cases.append(case("broadcast_2d", {"a": array_to_dict(a4), "b": array_to_dict(b4)},
                          array_to_dict(r4), tolerance_ulps=tolerance))

        # Special values
        a5 = np.array([float("nan"), float("inf"), 0.0, 1.0, -1.0], dtype="float64")
        b5 = np.array([1.0, 1.0, 0.0, float("nan"), float("inf")], dtype="float64")
        with np.errstate(all="ignore"):
            r5 = np_func(a5, b5)
        cases.append(case("special_values", {"a": array_to_dict(a5), "b": array_to_dict(b5)},
                          array_to_dict(r5), tolerance_ulps=tolerance))

        # f32
        a6 = np.array([1.0, 2.0, 3.0], dtype="float32")
        b6 = np.array([4.0, 5.0, 6.0], dtype="float32")
        with np.errstate(all="ignore"):
            r6 = np_func(a6, b6)
        cases.append(case("standard_f32", {"a": array_to_dict(a6, "float32"), "b": array_to_dict(b6, "float32")},
                          array_to_dict(r6, "float32"), tolerance_ulps=tolerance))

        # Empty
        ae = np.array([], dtype="float64")
        be = np.array([], dtype="float64")
        with np.errstate(all="ignore"):
            re = np_func(ae, be)
        cases.append(case("empty", {"a": array_to_dict(ae), "b": array_to_dict(be)},
                          array_to_dict(re), tolerance_ulps=0))

    save_fixture(subdir, filename, make_fixture(np_name, ferray_name, cases))


def generate_ufunc_fixtures():
    print("Generating ufunc fixtures...")

    # --- Trigonometric ---
    trig_input = {
        "standard_f64": np.array([0.0, 0.5235987755982988, 0.7853981633974483,
                                   1.0471975511965976, 1.5707963267948966], dtype="float64"),
        "full_circle": np.array([0.0, 1.5707963267948966, 3.141592653589793,
                                  4.71238898038469, 6.283185307179586], dtype="float64"),
        "negative": np.array([-3.14159265, -1.57079633, 0.0, 1.57079633, 3.14159265], dtype="float64"),
        "large": np.array([100.0, 1000.0, 1e6, -100.0, -1e6], dtype="float64"),
        "special_f64": np.array([float("nan"), float("inf"), float("-inf"), 0.0, -0.0], dtype="float64"),
    }

    _generate_unary_ufunc(np.sin, "numpy.sin", "ferray_ufunc::sin", "ufunc", "sin.json",
                          inputs_override=trig_input)
    _generate_unary_ufunc(np.cos, "numpy.cos", "ferray_ufunc::cos", "ufunc", "cos.json",
                          inputs_override=trig_input)
    _generate_unary_ufunc(np.tan, "numpy.tan", "ferray_ufunc::tan", "ufunc", "tan.json",
                          inputs_override=trig_input)

    # arcsin/arccos need [-1, 1]
    arc_input = {
        "standard_f64": np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype="float64"),
        "fine_f64": np.array([-0.99, -0.1, 0.0, 0.1, 0.99], dtype="float64"),
        "special_f64": np.array([float("nan"), 0.0, -0.0], dtype="float64"),
    }
    _generate_unary_ufunc(np.arcsin, "numpy.arcsin", "ferray_ufunc::arcsin", "ufunc", "arcsin.json",
                          inputs_override=arc_input)
    _generate_unary_ufunc(np.arccos, "numpy.arccos", "ferray_ufunc::arccos", "ufunc", "arccos.json",
                          inputs_override=arc_input)

    arctan_input = {
        "standard_f64": np.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype="float64"),
        "large_f64": np.array([-1e6, -1e3, 0.0, 1e3, 1e6], dtype="float64"),
        "special_f64": np.array([float("nan"), float("inf"), float("-inf"), 0.0, -0.0], dtype="float64"),
    }
    _generate_unary_ufunc(np.arctan, "numpy.arctan", "ferray_ufunc::arctan", "ufunc", "arctan.json",
                          inputs_override=arctan_input)

    # arctan2
    _generate_binary_ufunc(np.arctan2, "numpy.arctan2", "ferray_ufunc::arctan2", "ufunc", "arctan2.json",
                           inputs_override={
                               "standard": (np.array([0.0, 1.0, -1.0, 1.0], dtype="float64"),
                                            np.array([1.0, 0.0, 0.0, -1.0], dtype="float64")),
                               "zero_zero": (np.array([0.0, 0.0, -0.0, -0.0], dtype="float64"),
                                             np.array([0.0, -0.0, 0.0, -0.0], dtype="float64")),
                               "inf": (np.array([float("inf"), float("-inf"), 1.0, -1.0], dtype="float64"),
                                       np.array([1.0, 1.0, float("inf"), float("-inf")], dtype="float64")),
                           })

    # Hyperbolic
    hyp_input = {
        "standard_f64": np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype="float64"),
        "large_f64": np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype="float64"),
        "special_f64": np.array([float("nan"), float("inf"), float("-inf"), 0.0, -0.0], dtype="float64"),
    }
    _generate_unary_ufunc(np.sinh, "numpy.sinh", "ferray_ufunc::sinh", "ufunc", "sinh.json",
                          inputs_override=hyp_input)
    _generate_unary_ufunc(np.cosh, "numpy.cosh", "ferray_ufunc::cosh", "ufunc", "cosh.json",
                          inputs_override=hyp_input)
    _generate_unary_ufunc(np.tanh, "numpy.tanh", "ferray_ufunc::tanh", "ufunc", "tanh.json",
                          inputs_override=hyp_input)

    # Inverse hyperbolic
    arcsinh_input = {
        "standard_f64": np.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype="float64"),
        "special_f64": np.array([float("nan"), float("inf"), float("-inf"), 0.0], dtype="float64"),
    }
    _generate_unary_ufunc(np.arcsinh, "numpy.arcsinh", "ferray_ufunc::arcsinh", "ufunc", "arcsinh.json",
                          inputs_override=arcsinh_input)

    arccosh_input = {
        "standard_f64": np.array([1.0, 1.5, 2.0, 5.0, 10.0], dtype="float64"),
        "special_f64": np.array([float("nan"), float("inf"), 1.0], dtype="float64"),
    }
    _generate_unary_ufunc(np.arccosh, "numpy.arccosh", "ferray_ufunc::arccosh", "ufunc", "arccosh.json",
                          inputs_override=arccosh_input)

    arctanh_input = {
        "standard_f64": np.array([-0.99, -0.5, 0.0, 0.5, 0.99], dtype="float64"),
        "special_f64": np.array([float("nan"), 0.0, -0.0], dtype="float64"),
    }
    _generate_unary_ufunc(np.arctanh, "numpy.arctanh", "ferray_ufunc::arctanh", "ufunc", "arctanh.json",
                          inputs_override=arctanh_input)

    # --- Exp/log ---
    exp_input = {
        "standard_f64": np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype="float64"),
        "large_f64": np.array([-100.0, -10.0, 0.0, 10.0, 100.0], dtype="float64"),
        "small_f64": np.array([1e-15, 1e-10, 1e-7, 1e-3, 0.5], dtype="float64"),
        "special_f64": np.array([float("nan"), float("inf"), float("-inf"), 0.0, -0.0], dtype="float64"),
    }
    _generate_unary_ufunc(np.exp, "numpy.exp", "ferray_ufunc::exp", "ufunc", "exp.json",
                          inputs_override=exp_input)
    _generate_unary_ufunc(np.exp2, "numpy.exp2", "ferray_ufunc::exp2", "ufunc", "exp2.json",
                          inputs_override=exp_input)
    _generate_unary_ufunc(np.expm1, "numpy.expm1", "ferray_ufunc::expm1", "ufunc", "expm1.json",
                          inputs_override=exp_input)

    log_input = {
        "standard_f64": np.array([0.01, 0.1, 1.0, 10.0, 100.0], dtype="float64"),
        "one_f64": np.array([1.0, 2.718281828459045, 7.389056098930650], dtype="float64"),
        "small_f64": np.array([1e-15, 1e-10, 1e-7, 1e-3], dtype="float64"),
        "special_f64": np.array([float("nan"), float("inf"), 0.0, -1.0], dtype="float64"),
    }
    _generate_unary_ufunc(np.log, "numpy.log", "ferray_ufunc::log", "ufunc", "log.json",
                          inputs_override=log_input)
    _generate_unary_ufunc(np.log2, "numpy.log2", "ferray_ufunc::log2", "ufunc", "log2.json",
                          inputs_override=log_input)
    _generate_unary_ufunc(np.log10, "numpy.log10", "ferray_ufunc::log10", "ufunc", "log10.json",
                          inputs_override=log_input)

    log1p_input = {
        "standard_f64": np.array([-0.5, 0.0, 0.5, 1.0, 10.0], dtype="float64"),
        "tiny_f64": np.array([1e-15, 1e-10, 1e-7, 1e-3], dtype="float64"),
        "special_f64": np.array([float("nan"), float("inf"), -1.0, 0.0], dtype="float64"),
    }
    _generate_unary_ufunc(np.log1p, "numpy.log1p", "ferray_ufunc::log1p", "ufunc", "log1p.json",
                          inputs_override=log1p_input)

    # --- Rounding ---
    round_input = {
        "standard_f64": np.array([-2.7, -2.5, -2.3, -0.5, 0.0, 0.5, 2.3, 2.5, 2.7], dtype="float64"),
        "halves_f64": np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5], dtype="float64"),
        "tiny_f64": np.array([1e-10, -1e-10, 0.4999999999999999, 0.5000000000000001], dtype="float64"),
        "special_f64": np.array([float("nan"), float("inf"), float("-inf"), 0.0, -0.0], dtype="float64"),
    }
    _generate_unary_ufunc(np.round, "numpy.round", "ferray_ufunc::round", "ufunc", "round.json",
                          inputs_override=round_input, tolerance=0)
    _generate_unary_ufunc(np.floor, "numpy.floor", "ferray_ufunc::floor", "ufunc", "floor.json",
                          inputs_override=round_input, tolerance=0)
    _generate_unary_ufunc(np.ceil, "numpy.ceil", "ferray_ufunc::ceil", "ufunc", "ceil.json",
                          inputs_override=round_input, tolerance=0)
    _generate_unary_ufunc(np.trunc, "numpy.trunc", "ferray_ufunc::trunc", "ufunc", "trunc.json",
                          inputs_override=round_input, tolerance=0)
    _generate_unary_ufunc(np.fix, "numpy.fix", "ferray_ufunc::fix", "ufunc", "fix.json",
                          inputs_override=round_input, tolerance=0)
    _generate_unary_ufunc(np.rint, "numpy.rint", "ferray_ufunc::rint", "ufunc", "rint.json",
                          inputs_override=round_input, tolerance=0)

    # --- Arithmetic (binary) ---
    _generate_binary_ufunc(np.add, "numpy.add", "ferray_ufunc::add", "ufunc", "add.json", tolerance=0)
    _generate_binary_ufunc(np.subtract, "numpy.subtract", "ferray_ufunc::subtract", "ufunc", "subtract.json", tolerance=0)
    _generate_binary_ufunc(np.multiply, "numpy.multiply", "ferray_ufunc::multiply", "ufunc", "multiply.json", tolerance=0)
    _generate_binary_ufunc(np.divide, "numpy.divide", "ferray_ufunc::divide", "ufunc", "divide.json")
    _generate_binary_ufunc(np.power, "numpy.power", "ferray_ufunc::power", "ufunc", "power.json",
                           inputs_override={
                               "standard": (np.array([1.0, 2.0, 3.0, 4.0], dtype="float64"),
                                            np.array([0.0, 1.0, 2.0, 3.0], dtype="float64")),
                               "fractional": (np.array([4.0, 9.0, 16.0, 25.0], dtype="float64"),
                                              np.array([0.5, 0.5, 0.5, 0.5], dtype="float64")),
                               "negative_exp": (np.array([2.0, 3.0, 4.0], dtype="float64"),
                                                np.array([-1.0, -2.0, -1.0], dtype="float64")),
                               "special": (np.array([0.0, 1.0, float("inf"), float("nan")], dtype="float64"),
                                           np.array([0.0, float("inf"), 0.0, 1.0], dtype="float64")),
                           })
    _generate_binary_ufunc(np.remainder, "numpy.remainder", "ferray_ufunc::remainder", "ufunc", "remainder.json",
                           inputs_override={
                               "standard": (np.array([7.0, 8.0, 9.0, 10.0], dtype="float64"),
                                            np.array([3.0, 3.0, 3.0, 3.0], dtype="float64")),
                               "negative": (np.array([-7.0, -8.0, 7.0, 8.0], dtype="float64"),
                                            np.array([3.0, -3.0, -3.0, 3.0], dtype="float64")),
                               "float": (np.array([5.5, 6.7, 8.1], dtype="float64"),
                                         np.array([2.3, 2.3, 2.3], dtype="float64")),
                           })
    _generate_binary_ufunc(np.mod, "numpy.mod", "ferray_ufunc::mod_", "ufunc", "mod.json",
                           inputs_override={
                               "standard": (np.array([7.0, 8.0, 9.0], dtype="float64"),
                                            np.array([3.0, 3.0, 3.0], dtype="float64")),
                           })

    # --- Arithmetic (unary) ---
    abs_input = {
        "standard_f64": np.array([-3.0, -1.5, 0.0, 1.5, 3.0], dtype="float64"),
        "special_f64": np.array([float("nan"), float("inf"), float("-inf"), 0.0, -0.0], dtype="float64"),
    }
    _generate_unary_ufunc(np.absolute, "numpy.absolute", "ferray_ufunc::absolute", "ufunc", "absolute.json",
                          inputs_override=abs_input, tolerance=0)
    _generate_unary_ufunc(np.negative, "numpy.negative", "ferray_ufunc::negative", "ufunc", "negative.json",
                          tolerance=0)
    _generate_unary_ufunc(np.square, "numpy.square", "ferray_ufunc::square", "ufunc", "square.json",
                          tolerance=0)

    sqrt_input = {
        "standard_f64": np.array([0.0, 1.0, 4.0, 9.0, 16.0, 100.0], dtype="float64"),
        "small_f64": np.array([1e-14, 1e-7, 0.01, 0.5], dtype="float64"),
        "special_f64": np.array([float("nan"), float("inf"), 0.0], dtype="float64"),
    }
    _generate_unary_ufunc(np.sqrt, "numpy.sqrt", "ferray_ufunc::sqrt", "ufunc", "sqrt.json",
                          inputs_override=sqrt_input)
    _generate_unary_ufunc(np.cbrt, "numpy.cbrt", "ferray_ufunc::cbrt", "ufunc", "cbrt.json")

    recip_input = {
        "standard_f64": np.array([0.5, 1.0, 2.0, 4.0, 10.0], dtype="float64"),
        "negative_f64": np.array([-0.5, -1.0, -2.0], dtype="float64"),
        "special_f64": np.array([float("nan"), float("inf"), 0.0, -0.0], dtype="float64"),
    }
    _generate_unary_ufunc(np.reciprocal, "numpy.reciprocal", "ferray_ufunc::reciprocal", "ufunc", "reciprocal.json",
                          inputs_override=recip_input)

    # --- Special ---
    sinc_input = {
        "standard_f64": np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype="float64"),
        "small_f64": np.array([1e-10, 1e-7, 0.001, 0.01], dtype="float64"),
        "special_f64": np.array([float("nan"), float("inf"), float("-inf")], dtype="float64"),
    }
    _generate_unary_ufunc(np.sinc, "numpy.sinc", "ferray_ufunc::sinc", "ufunc", "sinc.json",
                          inputs_override=sinc_input)

    # heaviside
    _generate_binary_ufunc(np.heaviside, "numpy.heaviside", "ferray_ufunc::heaviside", "ufunc", "heaviside.json",
                           inputs_override={
                               "standard": (np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype="float64"),
                                            np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype="float64")),
                               "zero_val": (np.array([-1.0, 0.0, 1.0], dtype="float64"),
                                            np.array([0.0, 0.0, 0.0], dtype="float64")),
                               "one_val": (np.array([-1.0, 0.0, 1.0], dtype="float64"),
                                           np.array([1.0, 1.0, 1.0], dtype="float64")),
                               "special": (np.array([float("nan"), float("inf"), float("-inf")], dtype="float64"),
                                           np.array([0.5, 0.5, 0.5], dtype="float64")),
                           })

    # --- Float intrinsics ---
    # clip
    cases = []
    arr = np.array([-3.0, -1.0, 0.0, 1.0, 3.0, 5.0, float("nan")], dtype="float64")
    r = np.clip(arr, -2.0, 4.0)
    cases.append(case("standard_f64",
                      {"x": array_to_dict(arr), "a_min": -2.0, "a_max": 4.0},
                      array_to_dict(r), tolerance_ulps=0))
    arr2 = np.arange(12, dtype="float64").reshape(3, 4)
    r2 = np.clip(arr2, 2.0, 8.0)
    cases.append(case("2d_f64",
                      {"x": array_to_dict(arr2), "a_min": 2.0, "a_max": 8.0},
                      array_to_dict(r2), tolerance_ulps=0))
    save_fixture("ufunc", "clip.json", make_fixture("numpy.clip", "ferray_ufunc::clip", cases))

    # nan_to_num
    cases = []
    arr = np.array([float("nan"), float("inf"), float("-inf"), 0.0, 1.0, -1.0], dtype="float64")
    r = np.nan_to_num(arr, nan=0.0, posinf=1e308, neginf=-1e308)
    cases.append(case("standard_f64",
                      {"x": array_to_dict(arr), "nan": 0.0, "posinf": 1e308, "neginf": -1e308},
                      array_to_dict(r), tolerance_ulps=0))
    r2 = np.nan_to_num(arr)
    cases.append(case("defaults",
                      {"x": array_to_dict(arr)},
                      array_to_dict(r2), tolerance_ulps=0))
    save_fixture("ufunc", "nan_to_num.json", make_fixture("numpy.nan_to_num", "ferray_ufunc::nan_to_num", cases))

    # isnan, isinf, isfinite
    bool_input = np.array([float("nan"), float("inf"), float("-inf"), 0.0, 1.0, -1.0], dtype="float64")
    for func_name, np_func in [("isnan", np.isnan), ("isinf", np.isinf), ("isfinite", np.isfinite)]:
        r = np_func(bool_input)
        save_fixture("ufunc", f"{func_name}.json",
                     make_fixture(f"numpy.{func_name}", f"ferray_ufunc::{func_name}",
                                  [case("standard", {"x": array_to_dict(bool_input)},
                                        array_to_dict(r), tolerance_ulps=0)]))

    # maximum, minimum, fmax, fmin
    for func_name, np_func in [("maximum", np.maximum), ("minimum", np.minimum),
                                ("fmax", np.fmax), ("fmin", np.fmin)]:
        _generate_binary_ufunc(np_func, f"numpy.{func_name}", f"ferray_ufunc::{func_name}",
                               "ufunc", f"{func_name}.json", tolerance=0)


# ---------------------------------------------------------------------------
# Stats fixtures
# ---------------------------------------------------------------------------

def generate_stats_fixtures():
    print("Generating stats fixtures...")

    # --- Reduction functions ---
    arr1d = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], dtype="float64")
    arr2d = np.arange(12, dtype="float64").reshape(3, 4)
    arr_nan = np.array([1.0, float("nan"), 3.0, 4.0, float("nan"), 6.0], dtype="float64")

    for func_name, np_func in [("sum", np.sum), ("prod", np.prod),
                                ("min", np.min), ("max", np.max)]:
        cases = []
        cases.append(case("1d_f64", {"x": array_to_dict(arr1d)},
                          array_to_dict(np.array(np_func(arr1d))), tolerance_ulps=0))
        cases.append(case("2d_axis0", {"x": array_to_dict(arr2d), "axis": 0},
                          array_to_dict(np_func(arr2d, axis=0)), tolerance_ulps=0))
        cases.append(case("2d_axis1", {"x": array_to_dict(arr2d), "axis": 1},
                          array_to_dict(np_func(arr2d, axis=1)), tolerance_ulps=0))
        cases.append(case("2d_none", {"x": array_to_dict(arr2d)},
                          array_to_dict(np.array(np_func(arr2d))), tolerance_ulps=0))
        # empty
        emp = np.array([], dtype="float64")
        try:
            r = np_func(emp)
            cases.append(case("empty", {"x": array_to_dict(emp)},
                              array_to_dict(np.array(r)), tolerance_ulps=0))
        except ValueError:
            pass  # min/max on empty raises
        # single
        single = np.array([42.0], dtype="float64")
        cases.append(case("single", {"x": array_to_dict(single)},
                          array_to_dict(np.array(np_func(single))), tolerance_ulps=0))
        # f32
        arr32 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype="float32")
        cases.append(case("f32", {"x": array_to_dict(arr32, "float32")},
                          array_to_dict(np.array(np_func(arr32)), "float32"), tolerance_ulps=0))
        save_fixture("stats", f"{func_name}.json",
                     make_fixture(f"numpy.{func_name}", f"ferray_stats::{func_name}", cases))

    # --- argmin, argmax ---
    for func_name, np_func in [("argmin", np.argmin), ("argmax", np.argmax)]:
        cases = []
        cases.append(case("1d_f64", {"x": array_to_dict(arr1d)},
                          {"data": int(np_func(arr1d)), "shape": [], "dtype": "int64"}, tolerance_ulps=0))
        cases.append(case("2d_axis0", {"x": array_to_dict(arr2d), "axis": 0},
                          array_to_dict(np_func(arr2d, axis=0).astype("int64"), "int64"), tolerance_ulps=0))
        cases.append(case("2d_axis1", {"x": array_to_dict(arr2d), "axis": 1},
                          array_to_dict(np_func(arr2d, axis=1).astype("int64"), "int64"), tolerance_ulps=0))
        save_fixture("stats", f"{func_name}.json",
                     make_fixture(f"numpy.{func_name}", f"ferray_stats::{func_name}", cases))

    # --- mean ---
    cases = []
    cases.append(case("1d_f64", {"x": array_to_dict(arr1d)},
                      array_to_dict(np.array(np.mean(arr1d))), tolerance_ulps=4))
    cases.append(case("2d_axis0", {"x": array_to_dict(arr2d), "axis": 0},
                      array_to_dict(np.mean(arr2d, axis=0)), tolerance_ulps=4))
    cases.append(case("2d_axis1", {"x": array_to_dict(arr2d), "axis": 1},
                      array_to_dict(np.mean(arr2d, axis=1)), tolerance_ulps=4))
    cases.append(case("2d_none", {"x": array_to_dict(arr2d)},
                      array_to_dict(np.array(np.mean(arr2d))), tolerance_ulps=4))
    cases.append(case("single", {"x": array_to_dict(np.array([42.0]))},
                      array_to_dict(np.array(42.0)), tolerance_ulps=0))
    arr_large = np.array([1e15, 1e15, 1e15, 1.0], dtype="float64")
    cases.append(case("large_values", {"x": array_to_dict(arr_large)},
                      array_to_dict(np.array(np.mean(arr_large))), tolerance_ulps=4))
    save_fixture("stats", "mean.json", make_fixture("numpy.mean", "ferray_stats::mean", cases))

    # --- var ---
    cases = []
    cases.append(case("ddof0_1d", {"x": array_to_dict(arr1d), "ddof": 0},
                      array_to_dict(np.array(np.var(arr1d, ddof=0))), tolerance_ulps=4))
    cases.append(case("ddof1_1d", {"x": array_to_dict(arr1d), "ddof": 1},
                      array_to_dict(np.array(np.var(arr1d, ddof=1))), tolerance_ulps=4))
    cases.append(case("constant", {"x": array_to_dict(np.array([5.0, 5.0, 5.0, 5.0])), "ddof": 0},
                      array_to_dict(np.array(0.0)), tolerance_ulps=0))
    cases.append(case("2d_axis0", {"x": array_to_dict(arr2d), "axis": 0, "ddof": 0},
                      array_to_dict(np.var(arr2d, axis=0, ddof=0)), tolerance_ulps=4))
    save_fixture("stats", "var.json", make_fixture("numpy.var", "ferray_stats::var", cases))

    # --- std ---
    cases = []
    cases.append(case("ddof0_1d", {"x": array_to_dict(arr1d), "ddof": 0},
                      array_to_dict(np.array(np.std(arr1d, ddof=0))), tolerance_ulps=4))
    cases.append(case("ddof1_1d", {"x": array_to_dict(arr1d), "ddof": 1},
                      array_to_dict(np.array(np.std(arr1d, ddof=1))), tolerance_ulps=4))
    cases.append(case("constant", {"x": array_to_dict(np.array([5.0, 5.0, 5.0, 5.0])), "ddof": 0},
                      array_to_dict(np.array(0.0)), tolerance_ulps=0))
    save_fixture("stats", "std.json", make_fixture("numpy.std", "ferray_stats::std", cases))

    # --- median ---
    cases = []
    cases.append(case("odd_len", {"x": array_to_dict(np.array([3.0, 1.0, 4.0, 1.0, 5.0]))},
                      array_to_dict(np.array(np.median(np.array([3.0, 1.0, 4.0, 1.0, 5.0])))), tolerance_ulps=0))
    cases.append(case("even_len", {"x": array_to_dict(np.array([3.0, 1.0, 4.0, 2.0]))},
                      array_to_dict(np.array(np.median(np.array([3.0, 1.0, 4.0, 2.0])))), tolerance_ulps=4))
    cases.append(case("single", {"x": array_to_dict(np.array([7.0]))},
                      array_to_dict(np.array(7.0)), tolerance_ulps=0))
    cases.append(case("2d_axis0", {"x": array_to_dict(arr2d), "axis": 0},
                      array_to_dict(np.median(arr2d, axis=0)), tolerance_ulps=4))
    save_fixture("stats", "median.json", make_fixture("numpy.median", "ferray_stats::median", cases))

    # --- percentile / quantile ---
    arr_pct = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype="float64")
    cases = []
    for q in [0, 25, 50, 75, 100]:
        r = np.percentile(arr_pct, q)
        cases.append(case(f"p{q}", {"x": array_to_dict(arr_pct), "q": q},
                          array_to_dict(np.array(r)), tolerance_ulps=4))
    save_fixture("stats", "percentile.json", make_fixture("numpy.percentile", "ferray_stats::percentile", cases))

    cases = []
    for q in [0.0, 0.25, 0.5, 0.75, 1.0]:
        r = np.quantile(arr_pct, q)
        cases.append(case(f"q{q}", {"x": array_to_dict(arr_pct), "q": q},
                          array_to_dict(np.array(r)), tolerance_ulps=4))
    save_fixture("stats", "quantile.json", make_fixture("numpy.quantile", "ferray_stats::quantile", cases))

    # --- cumsum / cumprod ---
    cs_arr = np.array([1.0, 2.0, 3.0, 4.0], dtype="float64")
    cases = []
    cases.append(case("1d", {"x": array_to_dict(cs_arr)},
                      array_to_dict(np.cumsum(cs_arr)), tolerance_ulps=0))
    cases.append(case("2d_axis0", {"x": array_to_dict(arr2d), "axis": 0},
                      array_to_dict(np.cumsum(arr2d, axis=0)), tolerance_ulps=0))
    cases.append(case("2d_axis1", {"x": array_to_dict(arr2d), "axis": 1},
                      array_to_dict(np.cumsum(arr2d, axis=1)), tolerance_ulps=0))
    save_fixture("stats", "cumsum.json", make_fixture("numpy.cumsum", "ferray_stats::cumsum", cases))

    cases = []
    cp_arr = np.array([1.0, 2.0, 3.0, 4.0], dtype="float64")
    cases.append(case("1d", {"x": array_to_dict(cp_arr)},
                      array_to_dict(np.cumprod(cp_arr)), tolerance_ulps=0))
    cases.append(case("2d_axis0", {"x": array_to_dict(arr2d), "axis": 0},
                      array_to_dict(np.cumprod(arr2d + 1, axis=0)), tolerance_ulps=0))
    save_fixture("stats", "cumprod.json", make_fixture("numpy.cumprod", "ferray_stats::cumprod", cases))

    # --- histogram ---
    cases = []
    hist_arr = np.array([1.0, 2.0, 1.5, 2.5, 3.0, 3.5, 4.0, 2.0, 1.0], dtype="float64")
    counts, edges = np.histogram(hist_arr, bins=5)
    cases.append(case("5_bins",
                      {"x": array_to_dict(hist_arr), "bins": 5},
                      {"counts": array_to_dict(counts.astype("int64"), "int64"),
                       "bin_edges": array_to_dict(edges)},
                      tolerance_ulps=4))
    counts2, edges2 = np.histogram(hist_arr, bins=[1.0, 2.0, 3.0, 4.0])
    cases.append(case("explicit_bins",
                      {"x": array_to_dict(hist_arr),
                       "bins": array_to_dict(np.array([1.0, 2.0, 3.0, 4.0]))},
                      {"counts": array_to_dict(counts2.astype("int64"), "int64"),
                       "bin_edges": array_to_dict(edges2)},
                      tolerance_ulps=4))
    save_fixture("stats", "histogram.json", make_fixture("numpy.histogram", "ferray_stats::histogram", cases))

    # --- histogram_bin_edges ---
    cases = []
    edges = np.histogram_bin_edges(hist_arr, bins=5)
    cases.append(case("5_bins", {"x": array_to_dict(hist_arr), "bins": 5},
                      array_to_dict(edges), tolerance_ulps=4))
    save_fixture("stats", "histogram_bin_edges.json",
                 make_fixture("numpy.histogram_bin_edges", "ferray_stats::histogram_bin_edges", cases))

    # --- sort ---
    sort_arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], dtype="float64")
    cases = []
    cases.append(case("1d", {"x": array_to_dict(sort_arr)},
                      array_to_dict(np.sort(sort_arr)), tolerance_ulps=0))
    sort_2d = np.array([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]], dtype="float64")
    cases.append(case("2d_axis0", {"x": array_to_dict(sort_2d), "axis": 0},
                      array_to_dict(np.sort(sort_2d, axis=0)), tolerance_ulps=0))
    cases.append(case("2d_axis1", {"x": array_to_dict(sort_2d), "axis": 1},
                      array_to_dict(np.sort(sort_2d, axis=1)), tolerance_ulps=0))
    save_fixture("stats", "sort.json", make_fixture("numpy.sort", "ferray_stats::sort", cases))

    # --- argsort ---
    cases = []
    cases.append(case("1d", {"x": array_to_dict(sort_arr)},
                      array_to_dict(np.argsort(sort_arr).astype("int64"), "int64"), tolerance_ulps=0))
    save_fixture("stats", "argsort.json", make_fixture("numpy.argsort", "ferray_stats::argsort", cases))

    # --- unique ---
    uniq_arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0], dtype="float64")
    cases = []
    u = np.unique(uniq_arr)
    cases.append(case("basic", {"x": array_to_dict(uniq_arr)},
                      array_to_dict(u), tolerance_ulps=0))
    u_idx, u_cnt = np.unique(uniq_arr, return_index=True, return_counts=True)[1:]
    u_vals = np.unique(uniq_arr)
    cases.append(case("with_counts", {"x": array_to_dict(uniq_arr), "return_counts": True},
                      {"values": array_to_dict(u_vals),
                       "counts": array_to_dict(u_cnt.astype("int64"), "int64")},
                      tolerance_ulps=0))
    save_fixture("stats", "unique.json", make_fixture("numpy.unique", "ferray_stats::unique", cases))


# ---------------------------------------------------------------------------
# Linalg fixtures
# ---------------------------------------------------------------------------

def generate_linalg_fixtures():
    print("Generating linalg fixtures...")

    # --- dot / matmul ---
    # 1D dot
    a1 = np.array([1.0, 2.0, 3.0], dtype="float64")
    b1 = np.array([4.0, 5.0, 6.0], dtype="float64")

    cases = []
    cases.append(case("1d_inner", {"a": array_to_dict(a1), "b": array_to_dict(b1)},
                      array_to_dict(np.array(np.dot(a1, b1))), tolerance_ulps=4))
    # 2D matmul
    a2 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype="float64")  # 3x2
    b2 = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype="float64")  # 2x3
    cases.append(case("2d_matmul", {"a": array_to_dict(a2), "b": array_to_dict(b2)},
                      array_to_dict(np.dot(a2, b2)), tolerance_ulps=4))
    # f32
    a32 = a1.astype("float32")
    b32 = b1.astype("float32")
    cases.append(case("1d_f32", {"a": array_to_dict(a32, "float32"), "b": array_to_dict(b32, "float32")},
                      array_to_dict(np.array(np.dot(a32, b32)), "float32"), tolerance_ulps=4))
    save_fixture("linalg", "dot.json", make_fixture("numpy.dot", "ferray_linalg::dot", cases))

    # --- matmul ---
    cases = []
    cases.append(case("2d", {"a": array_to_dict(a2), "b": array_to_dict(b2)},
                      array_to_dict(np.matmul(a2, b2)), tolerance_ulps=4))
    # Square
    sq = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float64")
    cases.append(case("square", {"a": array_to_dict(sq), "b": array_to_dict(sq)},
                      array_to_dict(np.matmul(sq, sq)), tolerance_ulps=4))
    # Identity
    eye2 = np.eye(3, dtype="float64")
    m3 = np.arange(9, dtype="float64").reshape(3, 3)
    cases.append(case("identity", {"a": array_to_dict(eye2), "b": array_to_dict(m3)},
                      array_to_dict(np.matmul(eye2, m3)), tolerance_ulps=0))
    save_fixture("linalg", "matmul.json", make_fixture("numpy.matmul", "ferray_linalg::matmul", cases))

    # --- inner ---
    cases = []
    cases.append(case("1d", {"a": array_to_dict(a1), "b": array_to_dict(b1)},
                      array_to_dict(np.array(np.inner(a1, b1))), tolerance_ulps=4))
    save_fixture("linalg", "inner.json", make_fixture("numpy.inner", "ferray_linalg::inner", cases))

    # --- outer ---
    cases = []
    cases.append(case("1d", {"a": array_to_dict(a1), "b": array_to_dict(b1)},
                      array_to_dict(np.outer(a1, b1)), tolerance_ulps=0))
    save_fixture("linalg", "outer.json", make_fixture("numpy.outer", "ferray_linalg::outer", cases))

    # --- vdot ---
    cases = []
    cases.append(case("1d", {"a": array_to_dict(a1), "b": array_to_dict(b1)},
                      array_to_dict(np.array(np.vdot(a1, b1))), tolerance_ulps=4))
    # complex
    ac = np.array([1+2j, 3+4j], dtype="complex128")
    bc = np.array([5+6j, 7+8j], dtype="complex128")
    r = np.vdot(ac, bc)
    cases.append(case("complex128", {"a": array_to_dict(ac, "complex128"), "b": array_to_dict(bc, "complex128")},
                      array_to_dict(np.array(r), "complex128"), tolerance_ulps=4))
    save_fixture("linalg", "vdot.json", make_fixture("numpy.vdot", "ferray_linalg::vdot", cases))

    # --- inv ---
    inv_m = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float64")
    cases = []
    cases.append(case("2x2", {"a": array_to_dict(inv_m)},
                      array_to_dict(np.linalg.inv(inv_m)), tolerance_ulps=4))
    inv_m3 = np.array([[2.0, 1.0, 1.0], [1.0, 3.0, 2.0], [1.0, 0.0, 0.0]], dtype="float64")
    cases.append(case("3x3", {"a": array_to_dict(inv_m3)},
                      array_to_dict(np.linalg.inv(inv_m3)), tolerance_ulps=4))
    save_fixture("linalg", "inv.json", make_fixture("numpy.linalg.inv", "ferray_linalg::inv", cases))

    # --- solve ---
    A = np.array([[3.0, 1.0], [1.0, 2.0]], dtype="float64")
    b = np.array([9.0, 8.0], dtype="float64")
    x = np.linalg.solve(A, b)
    cases = []
    cases.append(case("2x2", {"a": array_to_dict(A), "b": array_to_dict(b)},
                      array_to_dict(x), tolerance_ulps=4))
    A3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype="float64")
    b3 = np.array([1, 2, 3], dtype="float64")
    x3 = np.linalg.solve(A3, b3)
    cases.append(case("3x3", {"a": array_to_dict(A3), "b": array_to_dict(b3)},
                      array_to_dict(x3), tolerance_ulps=4))
    save_fixture("linalg", "solve.json", make_fixture("numpy.linalg.solve", "ferray_linalg::solve", cases))

    # --- lstsq ---
    A_ls = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], dtype="float64")
    b_ls = np.array([1.0, 2.0, 2.0], dtype="float64")
    x_ls, residuals, rank, sv = np.linalg.lstsq(A_ls, b_ls, rcond=None)
    cases = []
    cases.append(case("overdetermined",
                      {"a": array_to_dict(A_ls), "b": array_to_dict(b_ls)},
                      {"x": array_to_dict(x_ls),
                       "rank": int(rank)},
                      tolerance_ulps=4))
    save_fixture("linalg", "lstsq.json", make_fixture("numpy.linalg.lstsq", "ferray_linalg::lstsq", cases))

    # --- det ---
    cases = []
    cases.append(case("2x2", {"a": array_to_dict(inv_m)},
                      array_to_dict(np.array(np.linalg.det(inv_m))), tolerance_ulps=4))
    cases.append(case("3x3", {"a": array_to_dict(inv_m3)},
                      array_to_dict(np.array(np.linalg.det(inv_m3))), tolerance_ulps=4))
    eye3 = np.eye(3, dtype="float64")
    cases.append(case("identity", {"a": array_to_dict(eye3)},
                      array_to_dict(np.array(np.linalg.det(eye3))), tolerance_ulps=0))
    save_fixture("linalg", "det.json", make_fixture("numpy.linalg.det", "ferray_linalg::det", cases))

    # --- norm ---
    cases = []
    v = np.array([3.0, 4.0], dtype="float64")
    cases.append(case("vector_l2", {"a": array_to_dict(v), "ord": 2},
                      array_to_dict(np.array(np.linalg.norm(v))), tolerance_ulps=4))
    cases.append(case("vector_l1", {"a": array_to_dict(v), "ord": 1},
                      array_to_dict(np.array(np.linalg.norm(v, ord=1))), tolerance_ulps=4))
    cases.append(case("vector_inf", {"a": array_to_dict(v), "ord": "Inf"},
                      array_to_dict(np.array(np.linalg.norm(v, ord=np.inf))), tolerance_ulps=0))
    m_norm = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float64")
    cases.append(case("matrix_fro", {"a": array_to_dict(m_norm), "ord": "fro"},
                      array_to_dict(np.array(np.linalg.norm(m_norm, "fro"))), tolerance_ulps=4))
    save_fixture("linalg", "norm.json", make_fixture("numpy.linalg.norm", "ferray_linalg::norm", cases))

    # --- trace ---
    cases = []
    cases.append(case("3x3", {"a": array_to_dict(m3)},
                      array_to_dict(np.array(np.trace(m3))), tolerance_ulps=0))
    cases.append(case("2x2", {"a": array_to_dict(inv_m)},
                      array_to_dict(np.array(np.trace(inv_m))), tolerance_ulps=0))
    save_fixture("linalg", "trace.json", make_fixture("numpy.trace", "ferray_linalg::trace", cases))

    # --- matrix_rank ---
    cases = []
    rank_m = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype="float64")
    cases.append(case("rank_2", {"a": array_to_dict(rank_m)},
                      {"data": int(np.linalg.matrix_rank(rank_m)), "shape": [], "dtype": "int64"},
                      tolerance_ulps=0))
    cases.append(case("full_rank", {"a": array_to_dict(inv_m3)},
                      {"data": int(np.linalg.matrix_rank(inv_m3)), "shape": [], "dtype": "int64"},
                      tolerance_ulps=0))
    save_fixture("linalg", "matrix_rank.json",
                 make_fixture("numpy.linalg.matrix_rank", "ferray_linalg::matrix_rank", cases))

    # --- cond ---
    cases = []
    cases.append(case("well_conditioned", {"a": array_to_dict(eye3)},
                      array_to_dict(np.array(np.linalg.cond(eye3))), tolerance_ulps=4))
    cases.append(case("2x2", {"a": array_to_dict(inv_m)},
                      array_to_dict(np.array(np.linalg.cond(inv_m))), tolerance_ulps=4))
    save_fixture("linalg", "cond.json", make_fixture("numpy.linalg.cond", "ferray_linalg::cond", cases))

    # --- SVD ---
    svd_m = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype="float64")
    U, S, Vt = np.linalg.svd(svd_m, full_matrices=False)
    cases = []
    cases.append(case("3x2_reduced",
                      {"a": array_to_dict(svd_m), "full_matrices": False},
                      {"U": array_to_dict(U), "S": array_to_dict(S), "Vt": array_to_dict(Vt)},
                      tolerance_ulps=4))
    U_full, S_full, Vt_full = np.linalg.svd(svd_m, full_matrices=True)
    cases.append(case("3x2_full",
                      {"a": array_to_dict(svd_m), "full_matrices": True},
                      {"U": array_to_dict(U_full), "S": array_to_dict(S_full), "Vt": array_to_dict(Vt_full)},
                      tolerance_ulps=4))
    save_fixture("linalg", "svd.json", make_fixture("numpy.linalg.svd", "ferray_linalg::svd", cases))

    # --- QR ---
    qr_m = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype="float64")
    Q, R = np.linalg.qr(qr_m, mode="reduced")
    cases = []
    cases.append(case("3x2_reduced",
                      {"a": array_to_dict(qr_m), "mode": "reduced"},
                      {"Q": array_to_dict(Q), "R": array_to_dict(R)},
                      tolerance_ulps=4))
    Q_c, R_c = np.linalg.qr(qr_m, mode="complete")
    cases.append(case("3x2_complete",
                      {"a": array_to_dict(qr_m), "mode": "complete"},
                      {"Q": array_to_dict(Q_c), "R": array_to_dict(R_c)},
                      tolerance_ulps=4))
    save_fixture("linalg", "qr.json", make_fixture("numpy.linalg.qr", "ferray_linalg::qr", cases))

    # --- Cholesky ---
    chol_m = np.array([[4.0, 2.0], [2.0, 3.0]], dtype="float64")
    L = np.linalg.cholesky(chol_m)
    cases = []
    cases.append(case("2x2_spd", {"a": array_to_dict(chol_m)},
                      array_to_dict(L), tolerance_ulps=4))
    chol3 = np.array([[25, 15, -5], [15, 18, 0], [-5, 0, 11]], dtype="float64")
    L3 = np.linalg.cholesky(chol3)
    cases.append(case("3x3_spd", {"a": array_to_dict(chol3)},
                      array_to_dict(L3), tolerance_ulps=4))
    save_fixture("linalg", "cholesky.json", make_fixture("numpy.linalg.cholesky", "ferray_linalg::cholesky", cases))

    # --- eig ---
    eig_m = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float64")
    vals, vecs = np.linalg.eig(eig_m)
    cases = []
    cases.append(case("2x2",
                      {"a": array_to_dict(eig_m)},
                      {"eigenvalues": array_to_dict(vals), "eigenvectors": array_to_dict(vecs)},
                      tolerance_ulps=4))
    save_fixture("linalg", "eig.json", make_fixture("numpy.linalg.eig", "ferray_linalg::eig", cases))

    # --- eigh ---
    eigh_m = np.array([[2.0, 1.0], [1.0, 3.0]], dtype="float64")
    vals_h, vecs_h = np.linalg.eigh(eigh_m)
    cases = []
    cases.append(case("2x2_symmetric",
                      {"a": array_to_dict(eigh_m)},
                      {"eigenvalues": array_to_dict(vals_h), "eigenvectors": array_to_dict(vecs_h)},
                      tolerance_ulps=4))
    save_fixture("linalg", "eigh.json", make_fixture("numpy.linalg.eigh", "ferray_linalg::eigh", cases))

    # --- eigvals ---
    cases = []
    ev = np.linalg.eigvals(eig_m)
    cases.append(case("2x2", {"a": array_to_dict(eig_m)},
                      array_to_dict(ev), tolerance_ulps=4))
    save_fixture("linalg", "eigvals.json", make_fixture("numpy.linalg.eigvals", "ferray_linalg::eigvals", cases))

    # --- kron ---
    ka = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float64")
    kb = np.array([[0.0, 5.0], [6.0, 7.0]], dtype="float64")
    cases = []
    cases.append(case("2x2", {"a": array_to_dict(ka), "b": array_to_dict(kb)},
                      array_to_dict(np.kron(ka, kb)), tolerance_ulps=0))
    save_fixture("linalg", "kron.json", make_fixture("numpy.kron", "ferray_linalg::kron", cases))

    # --- tensordot ---
    ta = np.arange(60, dtype="float64").reshape(3, 4, 5)
    tb = np.arange(24, dtype="float64").reshape(4, 3, 2)
    r = np.tensordot(ta, tb, axes=([1, 0], [0, 1]))
    cases = []
    cases.append(case("3d_contraction",
                      {"a": array_to_dict(ta), "b": array_to_dict(tb),
                       "axes": [[1, 0], [0, 1]]},
                      array_to_dict(r), tolerance_ulps=4))
    # Simple case
    ta2 = np.arange(6, dtype="float64").reshape(2, 3)
    tb2 = np.arange(6, dtype="float64").reshape(3, 2)
    r2 = np.tensordot(ta2, tb2, axes=1)
    cases.append(case("2d_axes1",
                      {"a": array_to_dict(ta2), "b": array_to_dict(tb2), "axes": 1},
                      array_to_dict(r2), tolerance_ulps=4))
    save_fixture("linalg", "tensordot.json", make_fixture("numpy.tensordot", "ferray_linalg::tensordot", cases))


# ---------------------------------------------------------------------------
# FFT fixtures
# ---------------------------------------------------------------------------

def generate_fft_fixtures():
    print("Generating FFT fixtures...")

    # --- fft ---
    cases = []
    # Power of 2
    sig8 = np.array([1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0], dtype="float64")
    r8 = np.fft.fft(sig8)
    cases.append(case("real_8", {"x": array_to_dict(sig8)},
                      array_to_dict(r8, "complex128"), tolerance_ulps=4))
    # Non-power-of-2
    sig7 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype="float64")
    r7 = np.fft.fft(sig7)
    cases.append(case("real_7", {"x": array_to_dict(sig7)},
                      array_to_dict(r7, "complex128"), tolerance_ulps=4))
    # Complex input
    sigc = np.array([1+0j, 0+1j, -1+0j, 0-1j], dtype="complex128")
    rc = np.fft.fft(sigc)
    cases.append(case("complex_4", {"x": array_to_dict(sigc, "complex128")},
                      array_to_dict(rc, "complex128"), tolerance_ulps=4))
    # Single element
    sig1 = np.array([5.0], dtype="float64")
    r1 = np.fft.fft(sig1)
    cases.append(case("single", {"x": array_to_dict(sig1)},
                      array_to_dict(r1, "complex128"), tolerance_ulps=0))
    # Length 1024
    t = np.linspace(0, 1, 64, endpoint=False)
    sig64 = np.sin(2 * np.pi * 5 * t) + 0.5 * np.cos(2 * np.pi * 10 * t)
    r64 = np.fft.fft(sig64)
    cases.append(case("sine_64", {"x": array_to_dict(sig64)},
                      array_to_dict(r64, "complex128"), tolerance_ulps=4))
    save_fixture("fft", "fft.json", make_fixture("numpy.fft.fft", "ferray_fft::fft", cases))

    # --- ifft ---
    cases = []
    # Round-trip
    cases.append(case("roundtrip_8",
                      {"x": array_to_dict(r8, "complex128")},
                      array_to_dict(np.fft.ifft(r8), "complex128"), tolerance_ulps=4))
    cases.append(case("roundtrip_7",
                      {"x": array_to_dict(r7, "complex128")},
                      array_to_dict(np.fft.ifft(r7), "complex128"), tolerance_ulps=4))
    save_fixture("fft", "ifft.json", make_fixture("numpy.fft.ifft", "ferray_fft::ifft", cases))

    # --- rfft ---
    cases = []
    rr8 = np.fft.rfft(sig8)
    cases.append(case("real_8", {"x": array_to_dict(sig8)},
                      array_to_dict(rr8, "complex128"), tolerance_ulps=4))
    rr7 = np.fft.rfft(sig7)
    cases.append(case("real_7", {"x": array_to_dict(sig7)},
                      array_to_dict(rr7, "complex128"), tolerance_ulps=4))
    save_fixture("fft", "rfft.json", make_fixture("numpy.fft.rfft", "ferray_fft::rfft", cases))

    # --- irfft ---
    cases = []
    irr8 = np.fft.irfft(rr8)
    cases.append(case("roundtrip_8", {"x": array_to_dict(rr8, "complex128")},
                      array_to_dict(irr8), tolerance_ulps=4))
    save_fixture("fft", "irfft.json", make_fixture("numpy.fft.irfft", "ferray_fft::irfft", cases))

    # --- fft2 ---
    cases = []
    m2d = np.arange(16, dtype="float64").reshape(4, 4)
    r2d = np.fft.fft2(m2d)
    cases.append(case("4x4", {"x": array_to_dict(m2d)},
                      array_to_dict(r2d, "complex128"), tolerance_ulps=4))
    save_fixture("fft", "fft2.json", make_fixture("numpy.fft.fft2", "ferray_fft::fft2", cases))

    # --- ifft2 ---
    cases = []
    ir2d = np.fft.ifft2(r2d)
    cases.append(case("roundtrip_4x4", {"x": array_to_dict(r2d, "complex128")},
                      array_to_dict(ir2d, "complex128"), tolerance_ulps=4))
    save_fixture("fft", "ifft2.json", make_fixture("numpy.fft.ifft2", "ferray_fft::ifft2", cases))

    # --- fftfreq ---
    cases = []
    for n, d in [(8, 1.0), (8, 0.5), (7, 1.0), (16, 1.0)]:
        freq = np.fft.fftfreq(n, d)
        cases.append(case(f"n{n}_d{d}", {"n": n, "d": d},
                          array_to_dict(freq), tolerance_ulps=0))
    save_fixture("fft", "fftfreq.json", make_fixture("numpy.fft.fftfreq", "ferray_fft::fftfreq", cases))

    # --- rfftfreq ---
    cases = []
    for n, d in [(8, 1.0), (7, 1.0), (16, 1.0)]:
        freq = np.fft.rfftfreq(n, d)
        cases.append(case(f"n{n}_d{d}", {"n": n, "d": d},
                          array_to_dict(freq), tolerance_ulps=0))
    save_fixture("fft", "rfftfreq.json", make_fixture("numpy.fft.rfftfreq", "ferray_fft::rfftfreq", cases))

    # --- fftshift / ifftshift ---
    cases = []
    x = np.array([0.0, 1.0, 2.0, 3.0, -4.0, -3.0, -2.0, -1.0], dtype="float64")
    shifted = np.fft.fftshift(x)
    cases.append(case("1d_8", {"x": array_to_dict(x)},
                      array_to_dict(shifted), tolerance_ulps=0))
    x2d = np.arange(9, dtype="float64").reshape(3, 3)
    shifted2d = np.fft.fftshift(x2d)
    cases.append(case("2d_3x3", {"x": array_to_dict(x2d)},
                      array_to_dict(shifted2d), tolerance_ulps=0))
    save_fixture("fft", "fftshift.json", make_fixture("numpy.fft.fftshift", "ferray_fft::fftshift", cases))

    cases = []
    unshifted = np.fft.ifftshift(shifted)
    cases.append(case("roundtrip_1d", {"x": array_to_dict(shifted)},
                      array_to_dict(unshifted), tolerance_ulps=0))
    save_fixture("fft", "ifftshift.json", make_fixture("numpy.fft.ifftshift", "ferray_fft::ifftshift", cases))


# ---------------------------------------------------------------------------
# Random fixtures (moments only, not values)
# ---------------------------------------------------------------------------

def generate_random_fixtures():
    print("Generating random fixtures...")

    cases = []
    # We test that distribution moments are in expected ranges.
    # Not comparing exact values since RNG output is non-deterministic across implementations.
    rng = np.random.default_rng(42)

    # Normal distribution moments
    samples = rng.standard_normal(100000)
    cases.append(case("standard_normal_moments",
                      {"distribution": "standard_normal", "size": 100000, "seed": 42},
                      {"expected_mean": 0.0, "expected_var": 1.0,
                       "mean_tolerance": 0.02, "var_tolerance": 0.02},
                      tolerance_ulps=0))

    # Uniform distribution moments
    rng2 = np.random.default_rng(42)
    samples_u = rng2.uniform(0, 1, 100000)
    cases.append(case("uniform_moments",
                      {"distribution": "uniform", "low": 0.0, "high": 1.0, "size": 100000, "seed": 42},
                      {"expected_mean": 0.5, "expected_var": 1.0/12.0,
                       "mean_tolerance": 0.02, "var_tolerance": 0.02},
                      tolerance_ulps=0))

    # Exponential
    rng3 = np.random.default_rng(42)
    samples_e = rng3.exponential(2.0, 100000)
    cases.append(case("exponential_moments",
                      {"distribution": "exponential", "scale": 2.0, "size": 100000, "seed": 42},
                      {"expected_mean": 2.0, "expected_var": 4.0,
                       "mean_tolerance": 0.05, "var_tolerance": 0.1},
                      tolerance_ulps=0))

    # Gamma
    rng4 = np.random.default_rng(42)
    shape_param, scale_param = 2.0, 3.0
    cases.append(case("gamma_moments",
                      {"distribution": "gamma", "shape": shape_param, "scale": scale_param,
                       "size": 100000, "seed": 42},
                      {"expected_mean": shape_param * scale_param,
                       "expected_var": shape_param * scale_param**2,
                       "mean_tolerance": 0.1, "var_tolerance": 0.5},
                      tolerance_ulps=0))

    # Beta
    a_beta, b_beta = 2.0, 5.0
    expected_mean = a_beta / (a_beta + b_beta)
    expected_var = (a_beta * b_beta) / ((a_beta + b_beta)**2 * (a_beta + b_beta + 1))
    cases.append(case("beta_moments",
                      {"distribution": "beta", "a": a_beta, "b": b_beta,
                       "size": 100000, "seed": 42},
                      {"expected_mean": expected_mean,
                       "expected_var": expected_var,
                       "mean_tolerance": 0.02, "var_tolerance": 0.01},
                      tolerance_ulps=0))

    # Poisson
    lam = 5.0
    cases.append(case("poisson_moments",
                      {"distribution": "poisson", "lam": lam, "size": 100000, "seed": 42},
                      {"expected_mean": lam, "expected_var": lam,
                       "mean_tolerance": 0.05, "var_tolerance": 0.1},
                      tolerance_ulps=0))

    # Binomial
    n_binom, p_binom = 10, 0.3
    cases.append(case("binomial_moments",
                      {"distribution": "binomial", "n": n_binom, "p": p_binom,
                       "size": 100000, "seed": 42},
                      {"expected_mean": n_binom * p_binom,
                       "expected_var": n_binom * p_binom * (1 - p_binom),
                       "mean_tolerance": 0.05, "var_tolerance": 0.1},
                      tolerance_ulps=0))

    save_fixture("random", "distribution_moments.json",
                 make_fixture("numpy.random.Generator", "ferray_random::Generator", cases))


# ---------------------------------------------------------------------------
# IO fixtures
# ---------------------------------------------------------------------------

def generate_io_fixtures():
    print("Generating IO fixtures...")

    cases = []
    # Document dtype strings for round-trip verification
    dtypes = ["float32", "float64", "int32", "int64", "uint8",
              "complex64", "complex128", "bool"]
    for dt in dtypes:
        if dt == "bool":
            arr = np.array([True, False, True, False, True], dtype=dt)
        elif "complex" in dt:
            arr = np.array([1+2j, 3+4j, 5+6j], dtype=dt)
        elif "int" in dt or "uint" in dt:
            arr = np.array([1, 2, 3, 4, 5], dtype=dt)
        else:
            arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dt)

        npy_descr = arr.dtype.str
        cases.append(case(
            f"dtype_{dt}",
            {"data": array_to_dict(arr, dt), "numpy_dtype_str": npy_descr},
            {"shape": list(arr.shape), "dtype": dt, "numpy_dtype_str": npy_descr},
            tolerance_ulps=0,
        ))

    # 2D array
    arr2d = np.arange(12, dtype="float64").reshape(3, 4)
    cases.append(case("2d_f64",
                      {"data": array_to_dict(arr2d), "numpy_dtype_str": arr2d.dtype.str},
                      {"shape": [3, 4], "dtype": "float64", "numpy_dtype_str": arr2d.dtype.str},
                      tolerance_ulps=0))

    # Fortran order
    arr_f = np.asfortranarray(arr2d)
    cases.append(case("2d_fortran_order",
                      {"data": array_to_dict(arr_f), "fortran_order": True,
                       "numpy_dtype_str": arr_f.dtype.str},
                      {"shape": [3, 4], "dtype": "float64", "fortran_order": True},
                      tolerance_ulps=0))

    save_fixture("io", "npy_dtypes.json",
                 make_fixture("numpy.save/load", "ferray_io::save/load", cases))


# ---------------------------------------------------------------------------
# Polynomial fixtures
# ---------------------------------------------------------------------------

def generate_polynomial_fixtures():
    print("Generating polynomial fixtures...")

    # --- polyval ---
    cases = []
    # p(x) = 1 + 2x + 3x^2  -> numpy polyval uses highest-degree-first: [3, 2, 1]
    coeffs_np = np.array([3.0, 2.0, 1.0], dtype="float64")  # numpy convention
    coeffs_ferray = np.array([1.0, 2.0, 3.0], dtype="float64")  # ferray convention (low-to-high)
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype="float64")
    r = np.polyval(coeffs_np, x)
    cases.append(case("quadratic",
                      {"coefficients": array_to_dict(coeffs_ferray),
                       "coefficients_numpy_order": array_to_dict(coeffs_np),
                       "x": array_to_dict(x)},
                      array_to_dict(r), tolerance_ulps=4))

    # Linear: p(x) = 2 + 3x -> numpy: [3, 2]
    coeffs_np2 = np.array([3.0, 2.0], dtype="float64")
    coeffs_f2 = np.array([2.0, 3.0], dtype="float64")
    r2 = np.polyval(coeffs_np2, x)
    cases.append(case("linear",
                      {"coefficients": array_to_dict(coeffs_f2),
                       "coefficients_numpy_order": array_to_dict(coeffs_np2),
                       "x": array_to_dict(x)},
                      array_to_dict(r2), tolerance_ulps=4))

    # Constant
    coeffs_np3 = np.array([5.0], dtype="float64")
    r3 = np.polyval(coeffs_np3, x)
    cases.append(case("constant",
                      {"coefficients": array_to_dict(coeffs_np3),
                       "coefficients_numpy_order": array_to_dict(coeffs_np3),
                       "x": array_to_dict(x)},
                      array_to_dict(r3), tolerance_ulps=0))

    # Higher degree: x^4 - 3x^2 + 2 -> numpy: [1, 0, -3, 0, 2]
    coeffs_np4 = np.array([1.0, 0.0, -3.0, 0.0, 2.0], dtype="float64")
    coeffs_f4 = np.array([2.0, 0.0, -3.0, 0.0, 1.0], dtype="float64")
    r4 = np.polyval(coeffs_np4, x)
    cases.append(case("quartic",
                      {"coefficients": array_to_dict(coeffs_f4),
                       "coefficients_numpy_order": array_to_dict(coeffs_np4),
                       "x": array_to_dict(x)},
                      array_to_dict(r4), tolerance_ulps=4))

    save_fixture("polynomial", "polyval.json",
                 make_fixture("numpy.polyval", "ferray_polynomial::Polynomial::eval", cases))

    # --- polyfit ---
    cases = []
    xfit = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype="float64")
    yfit = np.array([1.0, 3.0, 7.0, 13.0, 21.0], dtype="float64")  # roughly 1 + x + x^2
    for deg in [1, 2, 3]:
        coeffs = np.polyfit(xfit, yfit, deg)
        cases.append(case(f"degree_{deg}",
                          {"x": array_to_dict(xfit), "y": array_to_dict(yfit), "degree": deg},
                          {"coefficients_numpy_order": array_to_dict(coeffs)},
                          tolerance_ulps=4))
    save_fixture("polynomial", "polyfit.json",
                 make_fixture("numpy.polyfit", "ferray_polynomial::Polynomial::fit", cases))

    # --- roots ---
    cases = []
    # x^2 - 3x + 2 = (x-1)(x-2) -> numpy: [1, -3, 2], roots: [2, 1]
    coeffs_r = np.array([1.0, -3.0, 2.0], dtype="float64")
    roots = np.roots(coeffs_r)
    roots_sorted = np.sort(roots.real)
    cases.append(case("quadratic_real",
                      {"coefficients_numpy_order": array_to_dict(coeffs_r)},
                      {"roots_real_sorted": array_to_dict(roots_sorted)},
                      tolerance_ulps=4))

    # x^2 + 1 = 0 -> roots: +-i
    coeffs_r2 = np.array([1.0, 0.0, 1.0], dtype="float64")
    roots2 = np.roots(coeffs_r2)
    cases.append(case("quadratic_complex",
                      {"coefficients_numpy_order": array_to_dict(coeffs_r2)},
                      {"roots": array_to_dict(roots2, "complex128")},
                      tolerance_ulps=4))

    # Cubic: x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
    coeffs_r3 = np.array([1.0, -6.0, 11.0, -6.0], dtype="float64")
    roots3 = np.roots(coeffs_r3)
    roots3_sorted = np.sort(roots3.real)
    cases.append(case("cubic_real",
                      {"coefficients_numpy_order": array_to_dict(coeffs_r3)},
                      {"roots_real_sorted": array_to_dict(roots3_sorted)},
                      tolerance_ulps=4))

    save_fixture("polynomial", "roots.json",
                 make_fixture("numpy.roots", "ferray_polynomial::Polynomial::roots", cases))


# ---------------------------------------------------------------------------
# Strings fixtures
# ---------------------------------------------------------------------------

def generate_strings_fixtures():
    print("Generating strings fixtures...")

    arr = np.array(["hello", "WORLD", "FoO BaR", "  spaces  ", "123abc"], dtype="U20")

    # --- Case operations ---
    for op_name, np_func in [
        ("upper", np.strings.upper),
        ("lower", np.strings.lower),
        ("title", np.strings.title),
        ("capitalize", np.strings.capitalize),
        ("swapcase", np.strings.swapcase),
    ]:
        result = np_func(arr)
        cases = [case("standard",
                       {"x": {"data": arr.tolist(), "shape": list(arr.shape), "dtype": "str"}},
                       {"data": result.tolist(), "shape": list(result.shape), "dtype": "str"},
                       tolerance_ulps=0)]
        # Single element
        single = np.array(["tEsT"], dtype="U10")
        r = np_func(single)
        cases.append(case("single",
                          {"x": {"data": single.tolist(), "shape": [1], "dtype": "str"}},
                          {"data": r.tolist(), "shape": [1], "dtype": "str"},
                          tolerance_ulps=0))
        # Empty string
        emp = np.array(["", "a", ""], dtype="U10")
        r_e = np_func(emp)
        cases.append(case("with_empty",
                          {"x": {"data": emp.tolist(), "shape": [3], "dtype": "str"}},
                          {"data": r_e.tolist(), "shape": [3], "dtype": "str"},
                          tolerance_ulps=0))
        save_fixture("strings", f"{op_name}.json",
                     make_fixture(f"numpy.strings.{op_name}", f"ferray_strings::{op_name}", cases))

    # --- Strip operations ---
    strip_arr = np.array(["  hello  ", "\tworld\t", "  foo", "bar  ", "none"], dtype="U20")
    for op_name, np_func in [
        ("strip", np.strings.strip),
        ("lstrip", np.strings.lstrip),
        ("rstrip", np.strings.rstrip),
    ]:
        result = np_func(strip_arr)
        cases = [case("whitespace",
                       {"x": {"data": strip_arr.tolist(), "shape": list(strip_arr.shape), "dtype": "str"}},
                       {"data": result.tolist(), "shape": list(result.shape), "dtype": "str"},
                       tolerance_ulps=0)]
        # With chars
        char_arr = np.array(["xxhelloxx", "xxworldxx"], dtype="U20")
        r_c = np_func(char_arr, "x")
        cases.append(case("with_chars",
                          {"x": {"data": char_arr.tolist(), "shape": list(char_arr.shape), "dtype": "str"},
                           "chars": "x"},
                          {"data": r_c.tolist(), "shape": list(r_c.shape), "dtype": "str"},
                          tolerance_ulps=0))
        save_fixture("strings", f"{op_name}.json",
                     make_fixture(f"numpy.strings.{op_name}", f"ferray_strings::{op_name}", cases))

    # --- startswith / endswith ---
    sw_arr = np.array(["hello", "help", "world", "helm"], dtype="U10")
    for op_name, np_func in [("startswith", np.strings.startswith), ("endswith", np.strings.endswith)]:
        prefix = "hel" if op_name == "startswith" else "ld"
        result = np_func(sw_arr, prefix)
        cases = [case("standard",
                       {"x": {"data": sw_arr.tolist(), "shape": list(sw_arr.shape), "dtype": "str"},
                        "substr": prefix},
                       array_to_dict(result, "bool"),
                       tolerance_ulps=0)]
        save_fixture("strings", f"{op_name}.json",
                     make_fixture(f"numpy.strings.{op_name}", f"ferray_strings::{op_name}", cases))

    # --- find ---
    find_arr = np.array(["hello world", "foobar", "hello", "hi"], dtype="U20")
    result = np.strings.find(find_arr, "lo")
    cases = [case("standard",
                   {"x": {"data": find_arr.tolist(), "shape": list(find_arr.shape), "dtype": "str"},
                    "substr": "lo"},
                   array_to_dict(result.astype("int64"), "int64"),
                   tolerance_ulps=0)]
    save_fixture("strings", "find.json",
                 make_fixture("numpy.strings.find", "ferray_strings::find", cases))

    # --- count ---
    count_arr = np.array(["aabaa", "abab", "cccc", ""], dtype="U10")
    result = np.strings.count(count_arr, "a")
    cases = [case("standard",
                   {"x": {"data": count_arr.tolist(), "shape": list(count_arr.shape), "dtype": "str"},
                    "substr": "a"},
                   array_to_dict(result.astype("int64"), "int64"),
                   tolerance_ulps=0)]
    save_fixture("strings", "count.json",
                 make_fixture("numpy.strings.count", "ferray_strings::count", cases))

    # --- replace ---
    repl_arr = np.array(["hello world", "foo bar foo", "aaa"], dtype="U30")
    result = np.strings.replace(repl_arr, "o", "0")
    cases = [case("standard",
                   {"x": {"data": repl_arr.tolist(), "shape": list(repl_arr.shape), "dtype": "str"},
                    "old": "o", "new": "0"},
                   {"data": result.tolist(), "shape": list(result.shape), "dtype": "str"},
                   tolerance_ulps=0)]
    # With count limit
    result2 = np.strings.replace(repl_arr, "o", "0", 1)
    cases.append(case("with_count",
                      {"x": {"data": repl_arr.tolist(), "shape": list(repl_arr.shape), "dtype": "str"},
                       "old": "o", "new": "0", "count": 1},
                      {"data": result2.tolist(), "shape": list(result2.shape), "dtype": "str"},
                      tolerance_ulps=0))
    save_fixture("strings", "replace.json",
                 make_fixture("numpy.strings.replace", "ferray_strings::replace", cases))

    # --- contains (isin for strings - use find >= 0) ---
    contains_arr = np.array(["hello", "world", "help", "foo"], dtype="U10")
    result = np.strings.find(contains_arr, "hel") >= 0
    cases = [case("standard",
                   {"x": {"data": contains_arr.tolist(), "shape": list(contains_arr.shape), "dtype": "str"},
                    "substr": "hel"},
                   array_to_dict(result, "bool"),
                   tolerance_ulps=0)]
    save_fixture("strings", "contains.json",
                 make_fixture("numpy.strings (find>=0)", "ferray_strings::contains", cases))


# ---------------------------------------------------------------------------
# Masked array fixtures
# ---------------------------------------------------------------------------

def generate_ma_fixtures():
    print("Generating ma fixtures...")

    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype="float64")
    mask = np.array([False, False, True, False, False])

    # --- creation + filled ---
    ma_arr = np.ma.array(data, mask=mask)
    filled = ma_arr.filled(0.0)
    cases = []
    cases.append(case("basic_creation",
                      {"data": array_to_dict(data), "mask": array_to_dict(mask, "bool")},
                      {"filled_0": array_to_dict(filled)},
                      tolerance_ulps=0))
    filled_99 = ma_arr.filled(99.0)
    cases.append(case("filled_custom",
                      {"data": array_to_dict(data), "mask": array_to_dict(mask, "bool"), "fill_value": 99.0},
                      {"filled": array_to_dict(filled_99)},
                      tolerance_ulps=0))
    save_fixture("ma", "creation_filled.json",
                 make_fixture("numpy.ma.array + filled", "ferray_ma::MaskedArray", cases))

    # --- compressed ---
    compressed = ma_arr.compressed()
    cases = []
    cases.append(case("basic",
                      {"data": array_to_dict(data), "mask": array_to_dict(mask, "bool")},
                      array_to_dict(compressed),
                      tolerance_ulps=0))
    # All masked
    all_mask = np.array([True, True, True], dtype="bool")
    all_data = np.array([1.0, 2.0, 3.0], dtype="float64")
    ma_all = np.ma.array(all_data, mask=all_mask)
    comp_all = ma_all.compressed()
    cases.append(case("all_masked",
                      {"data": array_to_dict(all_data), "mask": array_to_dict(all_mask, "bool")},
                      array_to_dict(comp_all),
                      tolerance_ulps=0))
    # No mask
    no_mask = np.array([False, False, False], dtype="bool")
    ma_none = np.ma.array(all_data, mask=no_mask)
    comp_none = ma_none.compressed()
    cases.append(case("no_mask",
                      {"data": array_to_dict(all_data), "mask": array_to_dict(no_mask, "bool")},
                      array_to_dict(comp_none),
                      tolerance_ulps=0))
    save_fixture("ma", "compressed.json",
                 make_fixture("numpy.ma.compressed", "ferray_ma::MaskedArray::compressed", cases))

    # --- Masked reductions ---
    for func_name in ["mean", "sum", "min", "max"]:
        cases = []
        np_func = getattr(np.ma, func_name if func_name != "min" else "min")
        ma_arr = np.ma.array(data, mask=mask)
        result = getattr(ma_arr, func_name)()
        cases.append(case("basic",
                          {"data": array_to_dict(data), "mask": array_to_dict(mask, "bool")},
                          array_to_dict(np.array(float(result))),
                          tolerance_ulps=4))

        # 2D with axis
        data2d = np.arange(12, dtype="float64").reshape(3, 4)
        mask2d = np.array([[False, True, False, False],
                           [False, False, True, False],
                           [True, False, False, False]], dtype="bool")
        ma_2d = np.ma.array(data2d, mask=mask2d)
        r_ax0 = getattr(ma_2d, func_name)(axis=0)
        r_ax0_data = np.array(r_ax0.data, dtype="float64") if hasattr(r_ax0, 'data') else np.array(r_ax0)
        cases.append(case("2d_axis0",
                          {"data": array_to_dict(data2d),
                           "mask": array_to_dict(mask2d, "bool"),
                           "axis": 0},
                          array_to_dict(r_ax0_data),
                          tolerance_ulps=4))

        save_fixture("ma", f"masked_{func_name}.json",
                     make_fixture(f"numpy.ma.{func_name}", f"ferray_ma::MaskedArray::{func_name}", cases))

    # --- getmask / getdata ---
    cases = []
    cases.append(case("basic",
                      {"data": array_to_dict(data), "mask": array_to_dict(mask, "bool")},
                      {"mask": array_to_dict(mask, "bool"), "data": array_to_dict(data)},
                      tolerance_ulps=0))
    save_fixture("ma", "getmask_getdata.json",
                 make_fixture("numpy.ma.getmask/getdata", "ferray_ma::getmask/getdata", cases))

    # --- masked_invalid ---
    invalid_data = np.array([1.0, float("nan"), 3.0, float("inf"), 5.0, float("-inf")], dtype="float64")
    ma_inv = np.ma.masked_invalid(invalid_data)
    cases = []
    cases.append(case("nan_inf",
                      {"data": array_to_dict(invalid_data)},
                      {"mask": array_to_dict(np.array(ma_inv.mask, dtype="bool"), "bool")},
                      tolerance_ulps=0))
    save_fixture("ma", "masked_invalid.json",
                 make_fixture("numpy.ma.masked_invalid", "ferray_ma::masked_invalid", cases))


# ---------------------------------------------------------------------------
# Window fixtures (Stage 3 — ferray-window conformance suite)
# ---------------------------------------------------------------------------

def _make_window_fixture(scipy_func_name: str, ferray_func_name: str,
                         test_cases: list, ref_lib: str = "scipy.signal") -> dict:
    """Build a window fixture dict with the v2 schema + reference_library."""
    import scipy
    return {
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "fixture_schema_version": 2,
        "reference_library": ref_lib,
        "function": scipy_func_name,
        "ferray_function": ferray_func_name,
        "test_cases": test_cases,
    }


def _window_case(name: str, inputs: dict, m: int,
                 scipy_func, tolerance_ulps: int = 4) -> dict:
    """Generate a single window test case by calling scipy_func(m, **inputs)."""
    w = scipy_func(**inputs)
    expected = {
        "data": [float(x) for x in w.tolist()],
        "shape": [m],
        "dtype": "float64",
    }
    return {
        "name": name,
        "inputs": inputs,
        "expected": expected,
        "tolerance_ulps": tolerance_ulps,
    }


def generate_window_fixtures() -> None:
    """Generate fixtures/window/<name>.json for each ferray-window public function.

    All window functions are pure (no RNG), so output is fully deterministic.
    The reference is scipy.signal.windows for scipy-origin windows, and
    numpy for the handful that exist in numpy directly (hanning, hamming,
    bartlett, blackman, kaiser).
    """
    from scipy.signal import windows as spwin
    import scipy

    print("Generating window fixtures...")
    (FIXTURES_DIR / "window").mkdir(parents=True, exist_ok=True)

    scipy_ver = scipy.__version__
    numpy_ver = np.__version__

    # ------------------------------------------------------------------
    # Helper: build numpy-origin fixture (reference_library = "numpy")
    # ------------------------------------------------------------------
    def _np_window_fixture(np_func_name, ferray_func_name, test_cases):
        return {
            "numpy_version": numpy_ver,
            "scipy_version": scipy_ver,
            "fixture_schema_version": 2,
            "reference_library": "numpy",
            "function": np_func_name,
            "ferray_function": ferray_func_name,
            "test_cases": test_cases,
        }

    # ------------------------------------------------------------------
    # Helper: build scipy-origin fixture (reference_library = "scipy.signal")
    # ------------------------------------------------------------------
    def _sp_window_fixture(sp_func_name, ferray_func_name, test_cases):
        return {
            "numpy_version": numpy_ver,
            "scipy_version": scipy_ver,
            "fixture_schema_version": 2,
            "reference_library": "scipy.signal",
            "function": sp_func_name,
            "ferray_function": ferray_func_name,
            "test_cases": test_cases,
        }

    # ------------------------------------------------------------------
    # Shared helper: standard sizes to test per window
    # Sizes: small (16), medium (64), minimum edge case (2)
    # ------------------------------------------------------------------
    SIZES = [2, 16, 64]

    def _simple_cases(scipy_fn, fname_prefix):
        """Cases for a window taking only M (no extra params)."""
        cases = []
        for m in SIZES:
            w = scipy_fn(m)
            cases.append({
                "name": f"{fname_prefix}_m{m}",
                "inputs": {"m": m},
                "expected": {
                    "data": [float(x) for x in w.tolist()],
                    "shape": [m],
                    "dtype": "float64",
                },
                "tolerance_ulps": 4,
            })
        # Edge: M=1
        w1 = scipy_fn(1)
        cases.append({
            "name": f"{fname_prefix}_m1",
            "inputs": {"m": 1},
            "expected": {
                "data": [float(x) for x in w1.tolist()],
                "shape": [1],
                "dtype": "float64",
            },
            "tolerance_ulps": 0,
        })
        # Edge: M=0
        w0 = scipy_fn(0)
        cases.append({
            "name": f"{fname_prefix}_m0",
            "inputs": {"m": 0},
            "expected": {
                "data": [],
                "shape": [0],
                "dtype": "float64",
            },
            "tolerance_ulps": 0,
        })
        return cases

    # ------------------------------------------------------------------
    # bartlett — numpy.bartlett (also in scipy as scipy.signal.windows.bartlett)
    # ferray uses numpy naming; both agree for symmetric windows.
    # ------------------------------------------------------------------
    bartlett_cases = _simple_cases(np.bartlett, "bartlett")
    save_fixture("window", "bartlett.json",
                 _np_window_fixture("numpy.bartlett",
                                    "ferray_window::bartlett",
                                    bartlett_cases))

    # ------------------------------------------------------------------
    # blackman — numpy.blackman
    # ------------------------------------------------------------------
    blackman_cases = _simple_cases(np.blackman, "blackman")
    save_fixture("window", "blackman.json",
                 _np_window_fixture("numpy.blackman",
                                    "ferray_window::blackman",
                                    blackman_cases))

    # ------------------------------------------------------------------
    # hamming — numpy.hamming
    # ------------------------------------------------------------------
    hamming_cases = _simple_cases(np.hamming, "hamming")
    save_fixture("window", "hamming.json",
                 _np_window_fixture("numpy.hamming",
                                    "ferray_window::hamming",
                                    hamming_cases))

    # ------------------------------------------------------------------
    # hanning — numpy.hanning  (scipy calls it 'hann' but numpy uses 'hanning')
    # ------------------------------------------------------------------
    hanning_cases = _simple_cases(np.hanning, "hanning")
    save_fixture("window", "hanning.json",
                 _np_window_fixture("numpy.hanning",
                                    "ferray_window::hanning",
                                    hanning_cases))

    # ------------------------------------------------------------------
    # kaiser — numpy.kaiser (parametric: vary beta)
    # ------------------------------------------------------------------
    kaiser_cases = []
    for m in SIZES:
        for beta in [0.0, 5.0, 14.0]:
            w = np.kaiser(m, beta)
            kaiser_cases.append({
                "name": f"kaiser_m{m}_beta{beta}",
                "inputs": {"m": m, "beta": beta},
                "expected": {
                    "data": [float(x) for x in w.tolist()],
                    "shape": [m],
                    "dtype": "float64",
                },
                "tolerance_ulps": 4,
            })
    # Edge M=1
    w1 = np.kaiser(1, 5.0)
    kaiser_cases.append({
        "name": "kaiser_m1_beta5",
        "inputs": {"m": 1, "beta": 5.0},
        "expected": {"data": [float(x) for x in w1.tolist()], "shape": [1], "dtype": "float64"},
        "tolerance_ulps": 0,
    })
    save_fixture("window", "kaiser.json",
                 _np_window_fixture("numpy.kaiser",
                                    "ferray_window::kaiser",
                                    kaiser_cases))

    # ------------------------------------------------------------------
    # cosine — scipy.signal.windows.cosine
    # ------------------------------------------------------------------
    cosine_cases = _simple_cases(spwin.cosine, "cosine")
    save_fixture("window", "cosine.json",
                 _sp_window_fixture("scipy.signal.windows.cosine",
                                    "ferray_window::cosine",
                                    cosine_cases))

    # ------------------------------------------------------------------
    # exponential — scipy.signal.windows.exponential(m, center=None, tau=tau)
    # ------------------------------------------------------------------
    exp_cases = []
    for m in SIZES:
        for tau in [1.0, 5.0, 10.0]:
            w = spwin.exponential(m, tau=tau)
            exp_cases.append({
                "name": f"exponential_m{m}_tau{tau}",
                "inputs": {"m": m, "tau": tau},
                "expected": {
                    "data": [float(x) for x in w.tolist()],
                    "shape": [m],
                    "dtype": "float64",
                },
                "tolerance_ulps": 4,
            })
    # With explicit center (sym=False required when center != default)
    w_center = spwin.exponential(16, center=4.0, tau=5.0, sym=False)
    exp_cases.append({
        "name": "exponential_m16_center4_tau5_asymmetric",
        "inputs": {"m": 16, "center": 4.0, "tau": 5.0},
        "expected": {
            "data": [float(x) for x in w_center.tolist()],
            "shape": [16],
            "dtype": "float64",
        },
        "tolerance_ulps": 4,
    })
    save_fixture("window", "exponential.json",
                 _sp_window_fixture("scipy.signal.windows.exponential",
                                    "ferray_window::exponential",
                                    exp_cases))

    # ------------------------------------------------------------------
    # gaussian — scipy.signal.windows.gaussian(m, std)
    # ------------------------------------------------------------------
    gauss_cases = []
    for m in SIZES:
        for std in [1.0, 3.0, 7.0]:
            w = spwin.gaussian(m, std=std)
            gauss_cases.append({
                "name": f"gaussian_m{m}_std{std}",
                "inputs": {"m": m, "std": std},
                "expected": {
                    "data": [float(x) for x in w.tolist()],
                    "shape": [m],
                    "dtype": "float64",
                },
                "tolerance_ulps": 4,
            })
    save_fixture("window", "gaussian.json",
                 _sp_window_fixture("scipy.signal.windows.gaussian",
                                    "ferray_window::gaussian",
                                    gauss_cases))

    # ------------------------------------------------------------------
    # general_cosine — scipy.signal.windows.general_cosine(m, a)
    # ------------------------------------------------------------------
    gc_cases = []
    for m in SIZES:
        for coeffs, cname in [
            ([0.5, 0.5], "hann_coeffs"),
            ([0.42, 0.5, 0.08], "blackman_coeffs"),
            ([0.3635819, 0.4891775, 0.1365995, 0.0106411], "nuttall_coeffs"),
        ]:
            w = spwin.general_cosine(m, coeffs, sym=True)
            gc_cases.append({
                "name": f"general_cosine_m{m}_{cname}",
                "inputs": {"m": m, "coeffs": coeffs},
                "expected": {
                    "data": [float(x) for x in w.tolist()],
                    "shape": [m],
                    "dtype": "float64",
                },
                "tolerance_ulps": 4,
            })
    save_fixture("window", "general_cosine.json",
                 _sp_window_fixture("scipy.signal.windows.general_cosine",
                                    "ferray_window::general_cosine",
                                    gc_cases))

    # ------------------------------------------------------------------
    # general_hamming — scipy.signal.windows.general_hamming(m, alpha)
    # ------------------------------------------------------------------
    gh_cases = []
    for m in SIZES:
        for alpha in [0.5, 0.54, 25.0 / 46.0]:
            w = spwin.general_hamming(m, alpha, sym=True)
            gh_cases.append({
                "name": f"general_hamming_m{m}_alpha{alpha:.4f}",
                "inputs": {"m": m, "alpha": alpha},
                "expected": {
                    "data": [float(x) for x in w.tolist()],
                    "shape": [m],
                    "dtype": "float64",
                },
                "tolerance_ulps": 4,
            })
    save_fixture("window", "general_hamming.json",
                 _sp_window_fixture("scipy.signal.windows.general_hamming",
                                    "ferray_window::general_hamming",
                                    gh_cases))

    # ------------------------------------------------------------------
    # nuttall — scipy.signal.windows.nuttall
    # ------------------------------------------------------------------
    nuttall_cases = _simple_cases(lambda m: spwin.nuttall(m, sym=True), "nuttall")
    save_fixture("window", "nuttall.json",
                 _sp_window_fixture("scipy.signal.windows.nuttall",
                                    "ferray_window::nuttall",
                                    nuttall_cases))

    # ------------------------------------------------------------------
    # parzen — scipy.signal.windows.parzen
    # ------------------------------------------------------------------
    parzen_cases = _simple_cases(lambda m: spwin.parzen(m, sym=True), "parzen")
    save_fixture("window", "parzen.json",
                 _sp_window_fixture("scipy.signal.windows.parzen",
                                    "ferray_window::parzen",
                                    parzen_cases))

    # ------------------------------------------------------------------
    # taylor — scipy.signal.windows.taylor — DIVERGENCE documented in
    # _divergences.toml: ferray uses analytic peak normalisation,
    # scipy uses secant midpoint for even M.
    # Tests use scipy as the reference for the shape/magnitude, but
    # the divergence entry documents why values differ for even M with norm=True.
    # We generate cases with norm=False (no divergence) plus norm=True
    # with odd M (no divergence) as the primary conformance fixtures.
    # ------------------------------------------------------------------
    taylor_cases = []
    # norm=False cases: no normalisation divergence
    for m in [8, 16, 64]:
        for nbar, sll in [(4, 30.0), (6, 50.0)]:
            w = spwin.taylor(m, nbar=nbar, sll=sll, norm=False)
            taylor_cases.append({
                "name": f"taylor_m{m}_nbar{nbar}_sll{sll}_nonorm",
                "inputs": {"m": m, "nbar": nbar, "sll": sll, "norm": False},
                "expected": {
                    "data": [float(x) for x in w.tolist()],
                    "shape": [m],
                    "dtype": "float64",
                },
                "tolerance_ulps": 4,
            })
    # norm=True, odd M: no divergence (odd M has a sample exactly at centre)
    for m in [15, 63]:
        for nbar, sll in [(4, 30.0), (6, 50.0)]:
            w = spwin.taylor(m, nbar=nbar, sll=sll, norm=True)
            taylor_cases.append({
                "name": f"taylor_m{m}_nbar{nbar}_sll{sll}_norm",
                "inputs": {"m": m, "nbar": nbar, "sll": sll, "norm": True},
                "expected": {
                    "data": [float(x) for x in w.tolist()],
                    "shape": [m],
                    "dtype": "float64",
                },
                "tolerance_ulps": 4,
            })
    # Edge cases
    taylor_cases.append({
        "name": "taylor_m1",
        "inputs": {"m": 1, "nbar": 4, "sll": 30.0, "norm": False},
        "expected": {"data": [1.0], "shape": [1], "dtype": "float64"},
        "tolerance_ulps": 0,
    })
    taylor_cases.append({
        "name": "taylor_m0",
        "inputs": {"m": 0, "nbar": 4, "sll": 30.0, "norm": False},
        "expected": {"data": [], "shape": [0], "dtype": "float64"},
        "tolerance_ulps": 0,
    })
    save_fixture("window", "taylor.json",
                 _sp_window_fixture("scipy.signal.windows.taylor",
                                    "ferray_window::taylor",
                                    taylor_cases))

    # ------------------------------------------------------------------
    # tukey — scipy.signal.windows.tukey(m, alpha)
    # ------------------------------------------------------------------
    tukey_cases = []
    for m in SIZES:
        for alpha in [0.0, 0.5, 1.0]:
            w = spwin.tukey(m, alpha=alpha, sym=True)
            tukey_cases.append({
                "name": f"tukey_m{m}_alpha{alpha}",
                "inputs": {"m": m, "alpha": alpha},
                "expected": {
                    "data": [float(x) for x in w.tolist()],
                    "shape": [m],
                    "dtype": "float64",
                },
                "tolerance_ulps": 4,
            })
    save_fixture("window", "tukey.json",
                 _sp_window_fixture("scipy.signal.windows.tukey",
                                    "ferray_window::tukey",
                                    tukey_cases))

    # ------------------------------------------------------------------
    # chebwin — scipy.signal.windows.chebwin(m, at)
    # ------------------------------------------------------------------
    cheb_cases = []
    for m in [8, 16, 64]:
        for at in [40.0, 60.0, 100.0]:
            w = spwin.chebwin(m, at=at)
            cheb_cases.append({
                "name": f"chebwin_m{m}_at{at}",
                "inputs": {"m": m, "at": at},
                "expected": {
                    "data": [float(x) for x in w.tolist()],
                    "shape": [m],
                    "dtype": "float64",
                },
                "tolerance_ulps": 100,  # DFT-based; some accumulation drift expected
            })
    save_fixture("window", "chebwin.json",
                 _sp_window_fixture("scipy.signal.windows.chebwin",
                                    "ferray_window::chebwin",
                                    cheb_cases))

    # ------------------------------------------------------------------
    # dpss — scipy.signal.windows.dpss(m, NW) — single window (Kmax default)
    # Wider tolerance: power iteration vs scipy's eig solver differ slightly.
    # ------------------------------------------------------------------
    dpss_cases = []
    for m in [16, 64]:
        for nw in [2.0, 3.0, 4.0]:
            w_arr = spwin.dpss(m, NW=nw)
            # scipy returns (windows, ratios) when Kmax is given explicitly,
            # or a 1-D array when Kmax=1 (default). Normalise to 1-D.
            w = w_arr if w_arr.ndim == 1 else w_arr[0]
            # scipy default norm='approximate': max-normalise
            peak = float(np.max(np.abs(w)))
            if peak > 0:
                w = w / peak
            dpss_cases.append({
                "name": f"dpss_m{m}_nw{nw}",
                "inputs": {"m": m, "nw": nw},
                "expected": {
                    "data": [float(x) for x in w.tolist()],
                    "shape": [m],
                    "dtype": "float64",
                },
                "tolerance_ulps": 1000,  # power-iteration vs eig; wider envelope
            })
    save_fixture("window", "dpss.json",
                 _sp_window_fixture("scipy.signal.windows.dpss",
                                    "ferray_window::dpss",
                                    dpss_cases))

    # ------------------------------------------------------------------
    # boxcar — scipy.signal.windows.boxcar
    # ------------------------------------------------------------------
    boxcar_cases = _simple_cases(lambda m: spwin.boxcar(m, sym=True), "boxcar")
    save_fixture("window", "boxcar.json",
                 _sp_window_fixture("scipy.signal.windows.boxcar",
                                    "ferray_window::boxcar",
                                    boxcar_cases))

    # ------------------------------------------------------------------
    # triang — scipy.signal.windows.triang
    # ------------------------------------------------------------------
    triang_cases = _simple_cases(lambda m: spwin.triang(m, sym=True), "triang")
    save_fixture("window", "triang.json",
                 _sp_window_fixture("scipy.signal.windows.triang",
                                    "ferray_window::triang",
                                    triang_cases))

    # ------------------------------------------------------------------
    # bohman — scipy.signal.windows.bohman
    # ------------------------------------------------------------------
    bohman_cases = _simple_cases(lambda m: spwin.bohman(m, sym=True), "bohman")
    save_fixture("window", "bohman.json",
                 _sp_window_fixture("scipy.signal.windows.bohman",
                                    "ferray_window::bohman",
                                    bohman_cases))

    # ------------------------------------------------------------------
    # flattop — scipy.signal.windows.flattop
    # ------------------------------------------------------------------
    flattop_cases = _simple_cases(lambda m: spwin.flattop(m, sym=True), "flattop")
    save_fixture("window", "flattop.json",
                 _sp_window_fixture("scipy.signal.windows.flattop",
                                    "ferray_window::flattop",
                                    flattop_cases))

    # ------------------------------------------------------------------
    # lanczos — scipy.signal.windows.lanczos
    # ------------------------------------------------------------------
    lanczos_cases = _simple_cases(lambda m: spwin.lanczos(m, sym=True), "lanczos")
    save_fixture("window", "lanczos.json",
                 _sp_window_fixture("scipy.signal.windows.lanczos",
                                    "ferray_window::lanczos",
                                    lanczos_cases))

    # ------------------------------------------------------------------
    # bohman already done; now add functional items.
    # Functional items (apply_along_axis, apply_over_axes, piecewise,
    # sum_axis_keepdims, vectorize) have NO scipy/numpy equivalent fixture
    # — they are utility functions whose behaviour is tested by the Rust
    # conformance tests directly against spec, not against scipy output.
    # They are excluded in _surface_exclusions.toml with this reason.
    # ------------------------------------------------------------------

    print(f"  Window fixtures complete — {len(list((FIXTURES_DIR / 'window').glob('*.json')))} files.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"NumPy version: {np.__version__}")
    print(f"Output directory: {FIXTURES_DIR}")
    print()

    # Ensure directories exist
    for subdir in ["core", "ufunc", "stats", "linalg", "fft", "random", "io", "polynomial", "strings", "ma", "window"]:
        (FIXTURES_DIR / subdir).mkdir(parents=True, exist_ok=True)

    generators = [
        ("core", generate_core_fixtures),
        ("ufunc", generate_ufunc_fixtures),
        ("stats", generate_stats_fixtures),
        ("linalg", generate_linalg_fixtures),
        ("fft", generate_fft_fixtures),
        ("random", generate_random_fixtures),
        ("io", generate_io_fixtures),
        ("polynomial", generate_polynomial_fixtures),
        ("strings", generate_strings_fixtures),
        ("ma", generate_ma_fixtures),
        ("window", generate_window_fixtures),
    ]

    errors = []
    for name, gen_func in generators:
        try:
            gen_func()
        except Exception as e:
            errors.append((name, e))
            traceback.print_exc()
            print(f"ERROR generating {name} fixtures: {e}")

    print()
    print("=" * 60)
    total = sum(1 for f in FIXTURES_DIR.rglob("*.json"))
    print(f"Generated {total} fixture files total.")
    if errors:
        print(f"ERRORS in {len(errors)} generators:")
        for name, e in errors:
            print(f"  - {name}: {e}")
        sys.exit(1)
    else:
        print("All generators completed successfully.")


if __name__ == "__main__":
    main()
