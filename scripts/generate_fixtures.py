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
import warnings
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
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
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

    generate_ufunc_gap_fixtures()


def generate_ufunc_gap_fixtures():
    """Generate direct fixtures for implemented ufunc surfaces still audited as gaps."""
    # Aliases with NumPy names distinct from the representative fixture.
    _generate_unary_ufunc(np.around, "numpy.around", "ferray_ufunc::around",
                          "ufunc", "around.json", inputs_override={
                              "standard_f64": np.array(
                                  [-2.7, -2.5, -2.3, -0.5, 0.0, 0.5, 2.3, 2.5, 2.7],
                                  dtype="float64",
                              ),
                              "halves_f64": np.array(
                                  [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5],
                                  dtype="float64",
                              ),
                              "special_f64": np.array(
                                  [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
                                  dtype="float64",
                              ),
                          }, tolerance=0)
    _generate_unary_ufunc(np.fabs, "numpy.fabs", "ferray_ufunc::fabs",
                          "ufunc", "fabs.json", inputs_override={
                              "standard_f64": np.array([-3.0, -1.5, 0.0, 1.5, 3.0],
                                                       dtype="float64"),
                              "special_f64": np.array(
                                  [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
                                  dtype="float64",
                              ),
                          }, tolerance=0)
    _generate_unary_ufunc(np.positive, "numpy.positive", "ferray_ufunc::positive",
                          "ufunc", "positive.json", inputs_override={
                              "standard_f64": np.array([-3.0, -0.0, 0.0, 1.5, 3.0],
                                                       dtype="float64"),
                              "special_f64": np.array(
                                  [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
                                  dtype="float64",
                              ),
                          }, tolerance=0)
    _generate_unary_ufunc(np.sign, "numpy.sign", "ferray_ufunc::sign",
                          "ufunc", "sign.json", inputs_override={
                              "standard_f64": np.array([-3.0, -0.0, 0.0, 1.5, 3.0],
                                                       dtype="float64"),
                              "special_f64": np.array(
                                  [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
                                  dtype="float64",
                              ),
                          }, tolerance=0)

    # Implemented arithmetic/trig/explog functions that previously had no
    # committed direct fixture file.
    safe_mod_pairs = {
        "standard": (np.array([7.0, 8.0, 9.0, 10.0], dtype="float64"),
                     np.array([3.0, 3.0, 3.0, 3.0], dtype="float64")),
        "negative": (np.array([-7.0, -8.0, 7.0, 8.0], dtype="float64"),
                     np.array([3.0, -3.0, -3.0, 3.0], dtype="float64")),
        "float": (np.array([5.5, 6.7, 8.1], dtype="float64"),
                  np.array([2.3, 2.3, 2.3], dtype="float64")),
        "broadcast_2d": (np.arange(12, dtype="float64").reshape(3, 4),
                         np.array([2.5, 3.5, 4.5, 5.5], dtype="float64")),
    }
    _generate_binary_ufunc(np.floor_divide, "numpy.floor_divide",
                           "ferray_ufunc::floor_divide", "ufunc",
                           "floor_divide.json", inputs_override=safe_mod_pairs,
                           tolerance=0)
    _generate_binary_ufunc(np.fmod, "numpy.fmod", "ferray_ufunc::fmod",
                           "ufunc", "fmod.json", inputs_override=safe_mod_pairs,
                           tolerance=0)
    _generate_binary_ufunc(np.true_divide, "numpy.true_divide",
                           "ferray_ufunc::true_divide", "ufunc",
                           "true_divide.json", inputs_override={
                               "standard": (np.array([10.0, 20.0, -7.5], dtype="float64"),
                                            np.array([2.0, 4.0, 2.5], dtype="float64")),
                               "signed_zero_divisor": (
                                   np.array([1.0, -1.0, 0.0, -0.0], dtype="float64"),
                                   np.array([0.0, -0.0, 0.0, -0.0], dtype="float64"),
                               ),
                               "broadcast_2d": (np.arange(12, dtype="float64").reshape(3, 4),
                                                np.array([2.0, 4.0, 5.0, 10.0],
                                                         dtype="float64")),
                               "special": (np.array([float("nan"), float("inf"), 1.0],
                                                    dtype="float64"),
                                           np.array([1.0, 2.0, float("inf")],
                                                    dtype="float64")),
                           })

    divmod_cases = []
    for name, a, b in [
        ("standard", np.array([7.0, -7.0, 8.5, -8.5], dtype="float64"),
         np.array([3.0, 3.0, -2.0, -2.0], dtype="float64")),
        ("broadcast_2d", np.arange(12, dtype="float64").reshape(3, 4),
         np.array([2.0, 3.0, 4.0, 5.0], dtype="float64")),
        ("empty", np.array([], dtype="float64"), np.array([], dtype="float64")),
    ]:
        with np.errstate(all="ignore"):
            quotient, remainder = np.divmod(a, b)
        divmod_cases.append(case(name, {"a": array_to_dict(a), "b": array_to_dict(b)}, {
            "quotient": array_to_dict(quotient),
            "remainder": array_to_dict(remainder),
        }, tolerance_ulps=0))
    save_fixture("ufunc", "divmod.json",
                 make_fixture("numpy.divmod", "ferray_ufunc::divmod", divmod_cases))

    for func_name, np_func in [("gcd", np.gcd), ("lcm", np.lcm)]:
        error_cases = []
        for case_name, a, b in [
            ("float_1d_rejected",
             np.array([12.0, 15.0, 0.0, -18.0], dtype="float64"),
             np.array([8.0, 20.0, 7.0, 6.0], dtype="float64")),
            ("float_2d_rejected",
             np.array([[4.0, 6.0], [8.0, 10.0]], dtype="float64"),
             np.array([[6.0, 8.0], [12.0, 15.0]], dtype="float64")),
        ]:
            try:
                np_func(a, b)
            except TypeError as exc:
                expected = {"error_type": type(exc).__name__}
            else:
                raise AssertionError(f"numpy.{func_name} unexpectedly accepted float input")
            error_cases.append(case(case_name, {
                "a": array_to_dict(a),
                "b": array_to_dict(b),
            }, expected, tolerance_ulps=0))
        save_fixture("ufunc", f"{func_name}.json",
                     make_fixture(f"numpy.{func_name}", f"ferray_ufunc::{func_name}",
                                  error_cases))

        int_cases = []
        for case_name, a, b in [
            ("signed_i64",
             np.array([12, 15, 0, -18, -128], dtype="int64"),
             np.array([8, 20, 7, 6, 0], dtype="int64")),
            ("signed_i64_2d",
             np.array([[4, 6], [8, 10]], dtype="int64"),
             np.array([[6, 8], [12, 15]], dtype="int64")),
        ]:
            result = np_func(a, b)
            int_cases.append(case(case_name, {
                "a": array_to_dict(a),
                "b": array_to_dict(b),
            }, array_to_dict(result), tolerance_ulps=0))
        save_fixture("ufunc", f"{func_name}_int.json",
                     make_fixture(f"numpy.{func_name}", f"ferray_ufunc::{func_name}_int",
                                  int_cases))

    _generate_unary_ufunc(np.i0, "numpy.i0", "ferray_ufunc::i0",
                          "ufunc", "i0.json", inputs_override={
                              "standard_f64": np.array(
                                  [-12.0, -8.0, -3.0, -1.0, 0.0, 1.0, 3.0, 8.0, 12.0],
                                  dtype="float64",
                              ),
                              "matrix_f64": np.array(
                                  [[0.0, 0.5, 2.0], [4.0, 6.0, 10.0]],
                                  dtype="float64",
                              ),
                          }, tolerance=64)

    ufunc_method_cases = []
    matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float64")
    cube = np.arange(24, dtype="float64").reshape(2, 3, 4)
    vector = np.array([10.0, 3.0, 2.0, 1.0], dtype="float64")
    a_vec = np.array([1.0, 2.0, 3.0], dtype="float64")
    b_vec = np.array([10.0, 20.0], dtype="float64")

    for method, op_name, np_func, arr, kwargs in [
        ("reduce_axis", "add", np.add.reduce, matrix, {"axis": 1}),
        ("reduce_axis_keepdims", "add", np.add.reduce, matrix, {"axis": 1, "keepdims": True}),
        ("reduce_axes", "add", np.add.reduce, cube, {"axis": (0, 2)}),
        ("reduce_axes_keepdims", "add", np.add.reduce, cube, {"axis": (0, 2), "keepdims": True}),
        ("accumulate_axis", "add", np.add.accumulate, matrix, {"axis": 1}),
        ("accumulate_axis", "subtract", np.subtract.accumulate, matrix, {"axis": 1}),
    ]:
        result = np_func(arr, **kwargs)
        inputs = {
            "method": method,
            "op": op_name,
            "x": array_to_dict(arr),
        }
        if "axis" in kwargs:
            axis = kwargs["axis"]
            inputs["axis"] = list(axis) if isinstance(axis, tuple) else axis
        if "keepdims" in kwargs:
            inputs["keepdims"] = kwargs["keepdims"]
        ufunc_method_cases.append(case(
            f"{op_name}_{method}",
            inputs,
            array_to_dict(result),
            tolerance_ulps=0,
        ))

    reduce_all_result = np.add.reduce(matrix, axis=None)
    ufunc_method_cases.append(case(
        "add_reduce_all",
        {"method": "reduce_all", "op": "add", "x": array_to_dict(matrix)},
        array_to_dict(np.array(reduce_all_result)),
        tolerance_ulps=0,
    ))

    max_reduce_result = np.maximum.reduce(matrix, axis=1)
    ufunc_method_cases.append(case(
        "custom_max_reduce",
        {
            "method": "custom_reduce",
            "op": "maximum",
            "x": array_to_dict(matrix),
            "axis": 1,
        },
        array_to_dict(max_reduce_result),
        tolerance_ulps=0,
    ))

    for op_name, np_func in [
        ("add", np.add.outer),
        ("multiply", np.multiply.outer),
    ]:
        result = np_func(a_vec, b_vec)
        ufunc_method_cases.append(case(
            f"{op_name}_outer",
            {
                "method": "outer",
                "op": op_name,
                "a": array_to_dict(a_vec),
                "b": array_to_dict(b_vec),
            },
            array_to_dict(result),
            tolerance_ulps=0,
        ))

    add_at_target = np.array([0.0, 0.0, 0.0], dtype="float64")
    add_at_indices = np.array([0, 0, 1, 2], dtype="int64")
    add_at_values = np.array([1.0, 2.0, 5.0, 10.0], dtype="float64")
    np.add.at(add_at_target, add_at_indices, add_at_values)
    ufunc_method_cases.append(case(
        "add_at",
        {
            "method": "at",
            "op": "add",
            "x": array_to_dict(np.array([0.0, 0.0, 0.0], dtype="float64")),
            "indices": array_to_dict(add_at_indices),
            "values": array_to_dict(add_at_values),
        },
        array_to_dict(add_at_target),
        tolerance_ulps=0,
    ))

    for op_name, np_func, lhs, rhs in [
        ("add", np.add, np.array([1.0, 2.0, 3.0], dtype="float64"),
         np.array([10.0, 20.0, 30.0], dtype="float64")),
        ("subtract", np.subtract, np.array([10.0, 20.0, 30.0], dtype="float64"),
         np.array([1.0, 2.0, 3.0], dtype="float64")),
        ("multiply", np.multiply, np.array([2.0, 3.0, 4.0], dtype="float64"),
         np.array([5.0, 6.0, 7.0], dtype="float64")),
        ("divide", np.divide, np.array([10.0, 20.0, 30.0], dtype="float64"),
         np.array([2.0, 4.0, 5.0], dtype="float64")),
    ]:
        result = np_func(lhs, rhs)
        ufunc_method_cases.append(case(
            f"{op_name}_call",
            {
                "method": "call",
                "op": op_name,
                "a": array_to_dict(lhs),
                "b": array_to_dict(rhs),
            },
            array_to_dict(result),
            tolerance_ulps=0,
        ))

    for op_name, np_ufunc in [("add", np.add), ("multiply", np.multiply)]:
        ufunc_method_cases.append(case(
            f"{op_name}_metadata",
            {"method": "metadata", "op": op_name},
            {"name": np_ufunc.__name__, "identity": _serialize_value(np_ufunc.identity)},
            tolerance_ulps=0,
        ))

    save_fixture("ufunc", "ufunc_methods.json",
                 make_fixture("numpy ufunc methods", "ferray_ufunc::Ufunc",
                              ufunc_method_cases))

    interp_cases = []
    for case_name, x, xp, fp in [
        ("linear_midpoints",
         np.array([1.5, 2.5], dtype="float64"),
         np.array([1.0, 2.0, 3.0], dtype="float64"),
         np.array([0.0, 10.0, 20.0], dtype="float64")),
        ("boundary_clipping",
         np.array([0.0, 1.0, 4.0], dtype="float64"),
         np.array([1.0, 2.0, 3.0], dtype="float64"),
         np.array([10.0, 20.0, 30.0], dtype="float64")),
        ("nonuniform_xp",
         np.array([-2.0, -0.5, 0.5, 3.0], dtype="float64"),
         np.array([-1.0, 0.0, 2.0, 5.0], dtype="float64"),
         np.array([2.0, -2.0, 4.0, 10.0], dtype="float64")),
    ]:
        result = np.interp(x, xp, fp)
        interp_cases.append(case(
            case_name,
            {"x": array_to_dict(x), "xp": array_to_dict(xp), "fp": array_to_dict(fp)},
            array_to_dict(result),
            tolerance_ulps=4,
        ))
    save_fixture("ufunc", "interp.json",
                 make_fixture("numpy.interp", "ferray_ufunc::interp", interp_cases))

    interp_one_cases = []
    for case_name, xi, xp, fp in [
        ("scalar_midpoint", 2.5,
         np.array([1.0, 2.0, 3.0], dtype="float64"),
         np.array([3.0, 2.0, 0.0], dtype="float64")),
        ("scalar_left_clip", -2.0,
         np.array([-1.0, 0.0, 3.0], dtype="float64"),
         np.array([8.0, 4.0, 12.0], dtype="float64")),
        ("scalar_right_clip", 5.0,
         np.array([-1.0, 0.0, 3.0], dtype="float64"),
         np.array([8.0, 4.0, 12.0], dtype="float64")),
    ]:
        result = np.interp(np.array([xi], dtype="float64"), xp, fp)[0]
        interp_one_cases.append(case(
            case_name,
            {"xi": xi, "xp": array_to_dict(xp), "fp": array_to_dict(fp)},
            array_to_dict(np.array(result)),
            tolerance_ulps=4,
        ))
    save_fixture("ufunc", "interp_one.json",
                 make_fixture("numpy.interp", "ferray_ufunc::interp_one",
                              interp_one_cases))

    convolve_cases = []
    for case_name, a, v in [
        ("standard", np.array([1.0, 2.0, 3.0], dtype="float64"),
         np.array([0.0, 1.0, 0.5], dtype="float64")),
        ("left_shorter", np.array([2.0, -1.0], dtype="float64"),
         np.array([3.0, 4.0, 5.0, 6.0], dtype="float64")),
        ("signed_values", np.array([-2.0, 0.0, 4.0], dtype="float64"),
         np.array([1.5, -0.5], dtype="float64")),
    ]:
        for mode in ["full", "same", "valid"]:
            result = np.convolve(a, v, mode=mode)
            convolve_cases.append(case(
                f"{case_name}_{mode}",
                {"a": array_to_dict(a), "v": array_to_dict(v), "mode": mode},
                array_to_dict(result),
                tolerance_ulps=4,
            ))
    save_fixture("ufunc", "convolve.json",
                 make_fixture("numpy.convolve", "ferray_ufunc::convolve", convolve_cases))

    try:
        import scipy
        from scipy import signal as scipy_signal
    except ImportError as exc:
        raise RuntimeError(
            "generate_ufunc_gap_fixtures requires scipy for "
            "scipy.signal.fftconvolve fixtures"
        ) from exc

    fftconvolve_cases = []
    for case_name, a, v in [
        ("standard", np.array([1.0, 2.0, 3.0], dtype="float64"),
         np.array([0.0, 1.0, 0.5], dtype="float64")),
        ("left_shorter", np.array([2.0, -1.0], dtype="float64"),
         np.array([3.0, 4.0, 5.0, 6.0], dtype="float64")),
        ("signed_values", np.array([-2.0, 0.0, 4.0], dtype="float64"),
         np.array([1.5, -0.5], dtype="float64")),
        ("longer_signal",
         np.linspace(-1.25, 1.75, 17, dtype="float64"),
         np.array([0.25, -1.0, 0.5, 2.0, -0.75], dtype="float64")),
    ]:
        for mode in ["full", "same", "valid"]:
            result = scipy_signal.fftconvolve(a, v, mode=mode)
            fftconvolve_cases.append(case(
                f"{case_name}_{mode}",
                {"a": array_to_dict(a), "v": array_to_dict(v), "mode": mode},
                array_to_dict(result),
                tolerance_ulps=4096,
            ))
    fftconvolve_fixture = make_fixture("scipy.signal.fftconvolve",
                                       "ferray_ufunc::fftconvolve",
                                       fftconvolve_cases)
    fftconvolve_fixture["reference_library"] = "scipy.signal"
    fftconvolve_fixture["scipy_version"] = scipy.__version__
    save_fixture("ufunc", "fftconvolve.json", fftconvolve_fixture)

    unwrap_cases = []
    for case_name, arr, kwargs in [
        ("default_period_jump",
         np.array([0.0, np.pi / 2.0, np.pi, -3.0 * np.pi / 4.0],
                  dtype="float64"),
         {}),
        ("custom_discont",
         np.array([0.0, 1.0, 2.0, 8.0, 9.0], dtype="float64"),
         {"discont": 2.5}),
        ("empty",
         np.array([], dtype="float64"),
         {}),
    ]:
        result = np.unwrap(arr, **kwargs)
        inputs = {"x": array_to_dict(arr)}
        if "discont" in kwargs:
            inputs["discont"] = kwargs["discont"]
        unwrap_cases.append(case(case_name, inputs, array_to_dict(result),
                                 tolerance_ulps=4))
    save_fixture("ufunc", "unwrap.json",
                 make_fixture("numpy.unwrap", "ferray_ufunc::unwrap", unwrap_cases))

    exp_fast_cases = []
    for case_name, arr in [
        ("standard_f64", np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 3.0],
                                  dtype="float64")),
        ("special_values", np.array([np.nan, np.inf, -np.inf], dtype="float64")),
        ("matrix_f64", np.array([[-4.0, -0.25], [0.25, 4.0]], dtype="float64")),
    ]:
        with np.errstate(all="ignore"):
            result = np.exp(arr)
        exp_fast_cases.append(case(case_name, {"x": array_to_dict(arr)},
                                   array_to_dict(result), tolerance_ulps=4))
    save_fixture("ufunc", "exp_fast.json",
                 make_fixture("numpy.exp", "ferray_ufunc::exp_fast", exp_fast_cases))

    errstate_default = np.geterr()
    with np.errstate(over="raise", invalid="ignore"):
        errstate_scoped_inner = np.geterr()
    errstate_cases = [
        case("default", {}, errstate_default, tolerance_ulps=0),
        case("scoped_over_raise_invalid_ignore", {
            "over": "raise",
            "invalid": "ignore",
        }, {
            "inner": errstate_scoped_inner,
            "outer": np.geterr(),
        }, tolerance_ulps=0),
        case("policy_meanings", {}, {
            "warn": "record",
            "ignore": "drop",
            "raise": "defer_error",
        }, tolerance_ulps=0),
    ]
    save_fixture("ufunc", "errstate.json",
                 make_fixture("numpy.errstate", "ferray_ufunc::errstate",
                              errstate_cases))

    f16_unary_inputs = {
        "standard": {
            "standard_f16": np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float16),
            "matrix_f16": np.array([[-1.5, 0.0], [1.5, 3.0]], dtype=np.float16),
        },
        "nonnegative": {
            "standard_f16": np.array([0.0, 0.25, 1.0, 4.0, 16.0], dtype=np.float16),
            "matrix_f16": np.array([[0.0, 0.5], [2.0, 9.0]], dtype=np.float16),
        },
        "positive": {
            "standard_f16": np.array([0.25, 0.5, 1.0, 2.0, 8.0], dtype=np.float16),
            "matrix_f16": np.array([[0.125, 1.0], [4.0, 16.0]], dtype=np.float16),
        },
        "log1p": {
            "standard_f16": np.array([-0.75, -0.5, 0.0, 0.5, 2.0], dtype=np.float16),
            "matrix_f16": np.array([[-0.25, 0.0], [1.0, 3.0]], dtype=np.float16),
        },
        "exp": {
            "standard_f16": np.array([-4.0, -1.0, 0.0, 1.0, 4.0], dtype=np.float16),
            "matrix_f16": np.array([[-2.0, -0.5], [0.5, 2.0]], dtype=np.float16),
        },
        "trig": {
            "standard_f16": np.array(
                [-np.pi / 2, -np.pi / 4, 0.0, np.pi / 4, np.pi / 2],
                dtype=np.float16,
            ),
            "matrix_f16": np.array([[-1.0, 0.0], [0.5, 1.0]], dtype=np.float16),
        },
        "asin_acos": {
            "standard_f16": np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float16),
            "matrix_f16": np.array([[-0.75, 0.0], [0.25, 0.75]], dtype=np.float16),
        },
        "acosh": {
            "standard_f16": np.array([1.0, 1.25, 2.0, 4.0, 8.0], dtype=np.float16),
            "matrix_f16": np.array([[1.0, 1.5], [3.0, 6.0]], dtype=np.float16),
        },
        "atanh": {
            "standard_f16": np.array([-0.75, -0.25, 0.0, 0.25, 0.75], dtype=np.float16),
            "matrix_f16": np.array([[-0.5, 0.0], [0.125, 0.5]], dtype=np.float16),
        },
        "rounding": {
            "standard_f16": np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float16),
            "matrix_f16": np.array([[-1.75, -1.25], [1.25, 1.75]], dtype=np.float16),
        },
        "classify": {
            "special_f16": np.array(
                [np.nan, -np.inf, -1.0, 0.0, 1.0, np.inf],
                dtype=np.float16,
            ),
            "matrix_f16": np.array([[np.nan, np.inf], [-np.inf, 2.0]], dtype=np.float16),
        },
    }
    f16_unary_specs = [
        ("absolute_f16", np.absolute, "ferray_ufunc::absolute_f16", "standard", 0),
        ("negative_f16", np.negative, "ferray_ufunc::negative_f16", "standard", 0),
        ("sqrt_f16", np.sqrt, "ferray_ufunc::sqrt_f16", "nonnegative", 1),
        ("cbrt_f16", np.cbrt, "ferray_ufunc::cbrt_f16", "standard", 1),
        ("square_f16", np.square, "ferray_ufunc::square_f16", "standard", 0),
        ("reciprocal_f16", np.reciprocal, "ferray_ufunc::reciprocal_f16", "positive", 1),
        ("sign_f16", np.sign, "ferray_ufunc::sign_f16", "standard", 0),
        ("exp_f16", np.exp, "ferray_ufunc::exp_f16", "exp", 4),
        ("exp2_f16", np.exp2, "ferray_ufunc::exp2_f16", "exp", 4),
        ("expm1_f16", np.expm1, "ferray_ufunc::expm1_f16", "exp", 4),
        ("log_f16", np.log, "ferray_ufunc::log_f16", "positive", 4),
        ("log2_f16", np.log2, "ferray_ufunc::log2_f16", "positive", 4),
        ("log10_f16", np.log10, "ferray_ufunc::log10_f16", "positive", 4),
        ("log1p_f16", np.log1p, "ferray_ufunc::log1p_f16", "log1p", 4),
        ("floor_f16", np.floor, "ferray_ufunc::floor_f16", "rounding", 0),
        ("ceil_f16", np.ceil, "ferray_ufunc::ceil_f16", "rounding", 0),
        ("trunc_f16", np.trunc, "ferray_ufunc::trunc_f16", "rounding", 0),
        ("round_f16", np.round, "ferray_ufunc::round_f16", "rounding", 0),
        ("sin_f16", np.sin, "ferray_ufunc::sin_f16", "trig", 4),
        ("cos_f16", np.cos, "ferray_ufunc::cos_f16", "trig", 4),
        ("tan_f16", np.tan, "ferray_ufunc::tan_f16", "trig", 4),
        ("arcsin_f16", np.arcsin, "ferray_ufunc::arcsin_f16", "asin_acos", 4),
        ("arccos_f16", np.arccos, "ferray_ufunc::arccos_f16", "asin_acos", 4),
        ("arctan_f16", np.arctan, "ferray_ufunc::arctan_f16", "standard", 4),
        ("sinh_f16", np.sinh, "ferray_ufunc::sinh_f16", "trig", 4),
        ("cosh_f16", np.cosh, "ferray_ufunc::cosh_f16", "trig", 4),
        ("tanh_f16", np.tanh, "ferray_ufunc::tanh_f16", "trig", 4),
        ("arcsinh_f16", np.arcsinh, "ferray_ufunc::arcsinh_f16", "standard", 4),
        ("arccosh_f16", np.arccosh, "ferray_ufunc::arccosh_f16", "acosh", 4),
        ("arctanh_f16", np.arctanh, "ferray_ufunc::arctanh_f16", "atanh", 4),
        ("degrees_f16", np.degrees, "ferray_ufunc::degrees_f16", "trig", 1),
        ("radians_f16", np.radians, "ferray_ufunc::radians_f16", "rounding", 1),
        ("sinc_f16", np.sinc, "ferray_ufunc::sinc_f16", "standard", 4),
        ("isnan_f16", np.isnan, "ferray_ufunc::isnan_f16", "classify", 0),
        ("isinf_f16", np.isinf, "ferray_ufunc::isinf_f16", "classify", 0),
        ("isfinite_f16", np.isfinite, "ferray_ufunc::isfinite_f16", "classify", 0),
    ]
    for func_name, np_func, ferray_name, input_key, tolerance in f16_unary_specs:
        cases = []
        for case_name, arr in f16_unary_inputs[input_key].items():
            with np.errstate(all="ignore"):
                result = np_func(arr)
            cases.append(case(case_name, {"x": array_to_dict(arr)},
                              array_to_dict(result), tolerance_ulps=tolerance))
        save_fixture("ufunc", f"{func_name}.json",
                     make_fixture(f"numpy.{np_func.__name__}", ferray_name, cases))

    f16_binary_inputs = {
        "standard": [
            ("standard_f16",
             np.array([3.0, 4.0, -3.5, 3.5], dtype=np.float16),
             np.array([4.0, 3.0, 2.0, 2.0], dtype=np.float16)),
            ("matrix_f16",
             np.array([[1.5, -4.0], [7.0, -7.0]], dtype=np.float16),
             np.array([[2.0, 3.0], [2.0, 2.0]], dtype=np.float16)),
        ],
        "positive": [
            ("standard_f16",
             np.array([0.25, 1.5, 3.0, 8.0], dtype=np.float16),
             np.array([2.0, 0.5, 2.0, 0.25], dtype=np.float16)),
            ("matrix_f16",
             np.array([[1.25, 4.0], [9.0, 16.0]], dtype=np.float16),
             np.array([[2.0, 0.5], [0.5, 2.0]], dtype=np.float16)),
        ],
        "atan2": [
            ("quadrants_f16",
             np.array([0.0, 1.0, 1.0, -1.0, -1.0], dtype=np.float16),
             np.array([1.0, 0.0, -1.0, -1.0, 1.0], dtype=np.float16)),
            ("matrix_f16",
             np.array([[0.5, -0.5], [2.0, -2.0]], dtype=np.float16),
             np.array([[1.0, 1.0], [-1.0, -1.0]], dtype=np.float16)),
        ],
    }
    f16_binary_specs = [
        ("add_f16", np.add, "ferray_ufunc::add_f16", "standard", 0),
        ("subtract_f16", np.subtract, "ferray_ufunc::subtract_f16", "standard", 0),
        ("multiply_f16", np.multiply, "ferray_ufunc::multiply_f16", "standard", 0),
        ("divide_f16", np.divide, "ferray_ufunc::divide_f16", "standard", 1),
        ("power_f16", np.power, "ferray_ufunc::power_f16", "positive", 2),
        ("floor_divide_f16", np.floor_divide, "ferray_ufunc::floor_divide_f16", "standard", 0),
        ("remainder_f16", np.remainder, "ferray_ufunc::remainder_f16", "standard", 0),
        ("arctan2_f16", np.arctan2, "ferray_ufunc::arctan2_f16", "atan2", 4),
        ("hypot_f16", np.hypot, "ferray_ufunc::hypot_f16", "standard", 2),
        ("maximum_f16", np.maximum, "ferray_ufunc::maximum_f16", "standard", 0),
        ("minimum_f16", np.minimum, "ferray_ufunc::minimum_f16", "standard", 0),
    ]
    for func_name, np_func, ferray_name, input_key, tolerance in f16_binary_specs:
        cases = []
        for case_name, a, b in f16_binary_inputs[input_key]:
            with np.errstate(all="ignore"):
                result = np_func(a, b)
            cases.append(case(case_name, {"a": array_to_dict(a), "b": array_to_dict(b)},
                              array_to_dict(result), tolerance_ulps=tolerance))
        save_fixture("ufunc", f"{func_name}.json",
                     make_fixture(f"numpy.{np_func.__name__}", ferray_name, cases))

    clip_f16_cases = []
    for case_name, arr in f16_unary_inputs["standard"].items():
        amin = np.float16(-0.75)
        amax = np.float16(1.25)
        result = np.clip(arr, amin, amax)
        clip_f16_cases.append(case(
            case_name,
            {
                "x": array_to_dict(arr),
                "a_min": array_to_dict(np.array(amin, dtype=np.float16)),
                "a_max": array_to_dict(np.array(amax, dtype=np.float16)),
            },
            array_to_dict(result),
            tolerance_ulps=0,
        ))
    save_fixture("ufunc", "clip_f16.json",
                 make_fixture("numpy.clip", "ferray_ufunc::clip_f16", clip_f16_cases))

    nan_to_num_f16_cases = []
    for case_name, arr, kwargs in [
        (
            "default_specials_f16",
            np.array([np.nan, -np.inf, -1.0, 0.0, 1.0, np.inf], dtype=np.float16),
            {},
        ),
        (
            "custom_specials_f16",
            np.array([[np.nan, np.inf], [-np.inf, 2.0]], dtype=np.float16),
            {
                "nan": np.float16(9.0),
                "posinf": np.float16(11.0),
                "neginf": np.float16(-11.0),
            },
        ),
    ]:
        inputs = {"x": array_to_dict(arr)}
        for key, value in kwargs.items():
            inputs[key] = array_to_dict(np.array(value, dtype=np.float16))
        with np.errstate(all="ignore"):
            result = np.nan_to_num(arr, **kwargs)
        nan_to_num_f16_cases.append(case(
            case_name,
            inputs,
            array_to_dict(result),
            tolerance_ulps=0,
        ))
    save_fixture("ufunc", "nan_to_num_f16.json",
                 make_fixture("numpy.nan_to_num", "ferray_ufunc::nan_to_num_f16",
                              nan_to_num_f16_cases))

    def time_unit(arr):
        dtype = str(arr.dtype)
        return dtype[dtype.index("[") + 1:dtype.index("]")]

    def time_array_to_dict(arr):
        return {
            "data": [_serialize_value(x) for x in arr.view("int64").ravel()],
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "unit": time_unit(arr),
        }

    dt_d = np.array(["2020-01-01", "NaT", "2020-01-10"], dtype="datetime64[D]")
    dt_d_rhs = np.array(["2019-12-31", "2020-01-03", "NaT"], dtype="datetime64[D]")
    td_d = np.array([2, "NaT", -3], dtype="timedelta64[D]")
    td_d_rhs = np.array([5, 7, "NaT"], dtype="timedelta64[D]")

    time_cases = []
    result = np.isnat(dt_d)
    time_cases.append(case("datetime_days", {"x": time_array_to_dict(dt_d)},
                           array_to_dict(result), tolerance_ulps=0))
    save_fixture("ufunc", "isnat_datetime.json",
                 make_fixture("numpy.isnat", "ferray_ufunc::isnat_datetime",
                              time_cases))

    time_cases = []
    result = np.isnat(td_d)
    time_cases.append(case("timedelta_days", {"x": time_array_to_dict(td_d)},
                           array_to_dict(result), tolerance_ulps=0))
    save_fixture("ufunc", "isnat_timedelta.json",
                 make_fixture("numpy.isnat", "ferray_ufunc::isnat_timedelta",
                              time_cases))

    for fixture_name, ferray_name, lhs, rhs, np_func, lhs_key, rhs_key in [
        ("sub_datetime", "ferray_ufunc::sub_datetime", dt_d, dt_d_rhs,
         np.subtract, "a", "b"),
        ("add_datetime_timedelta", "ferray_ufunc::add_datetime_timedelta",
         dt_d, td_d, np.add, "a", "b"),
        ("sub_datetime_timedelta", "ferray_ufunc::sub_datetime_timedelta",
         dt_d, td_d, np.subtract, "a", "b"),
        ("add_timedelta", "ferray_ufunc::add_timedelta", td_d, td_d_rhs,
         np.add, "a", "b"),
        ("sub_timedelta", "ferray_ufunc::sub_timedelta", td_d, td_d_rhs,
         np.subtract, "a", "b"),
    ]:
        with np.errstate(all="ignore"):
            result = np_func(lhs, rhs)
        save_fixture("ufunc", f"{fixture_name}.json",
                     make_fixture(f"numpy.{np_func.__name__}", ferray_name, [
                         case("same_unit_days", {
                             lhs_key: time_array_to_dict(lhs),
                             rhs_key: time_array_to_dict(rhs),
                         }, time_array_to_dict(result), tolerance_ulps=0)
                     ]))

    dt_ms = np.array(["1970-01-01T00:00:01.500", "NaT",
                      "1970-01-01T00:00:04.250"], dtype="datetime64[ms]")
    dt_s = np.array(["1970-01-01T00:00:01", "1970-01-01T00:00:02",
                     "NaT"], dtype="datetime64[s]")
    td_ms = np.array([1500, "NaT", -250], dtype="timedelta64[ms]")
    td_s = np.array([1, 2, "NaT"], dtype="timedelta64[s]")

    for fixture_name, ferray_name, lhs, rhs, np_func in [
        ("sub_datetime_promoted", "ferray_ufunc::sub_datetime_promoted",
         dt_ms, dt_s, np.subtract),
        ("add_datetime_timedelta_promoted",
         "ferray_ufunc::add_datetime_timedelta_promoted", dt_s, td_ms, np.add),
        ("add_timedelta_promoted", "ferray_ufunc::add_timedelta_promoted",
         td_s, td_ms, np.add),
    ]:
        with np.errstate(all="ignore"):
            result = np_func(lhs, rhs)
        save_fixture("ufunc", f"{fixture_name}.json",
                     make_fixture(f"numpy.{np_func.__name__}", ferray_name, [
                         case("mixed_units", {
                             "a": time_array_to_dict(lhs),
                             "b": time_array_to_dict(rhs),
                         }, time_array_to_dict(result), tolerance_ulps=0)
                     ]))

    nan_matrix = np.array(
        [[1.0, np.nan, 3.0],
         [np.nan, np.nan, np.nan],
         [-2.0, 5.0, np.nan]],
        dtype="float64",
    )
    nan_cube = np.array(
        [[[1.0, np.nan, 3.0], [np.nan, np.nan, np.nan]],
         [[-2.0, 5.0, np.nan], [np.nan, np.nan, np.nan]]],
        dtype="float64",
    )
    nan_vector = np.array([1.0, np.nan, 3.0, -2.0, 5.0, np.nan],
                          dtype="float64")
    all_nan_vector = np.array([np.nan, np.nan], dtype="float64")

    def nan_extreme_result(np_func, arr, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np_func(arr, **kwargs)

    for func_name, np_func, ferray_name in [
        ("nan_max_reduce", np.nanmax, "ferray_ufunc::nan_max_reduce"),
        ("nan_min_reduce", np.nanmin, "ferray_ufunc::nan_min_reduce"),
    ]:
        cases = []
        for case_name, axis, keepdims in [
            ("axis1", 1, False),
            ("axis1_keepdims", 1, True),
            ("axis0", 0, False),
        ]:
            result = nan_extreme_result(np_func, nan_matrix, axis=axis,
                                        keepdims=keepdims)
            cases.append(case(
                case_name,
                {"x": array_to_dict(nan_matrix), "axis": axis, "keepdims": keepdims},
                array_to_dict(result),
                tolerance_ulps=0,
            ))
        save_fixture("ufunc", f"{func_name}.json",
                     make_fixture(f"numpy.{np_func.__name__}", ferray_name, cases))

    for func_name, np_func, ferray_name in [
        ("nan_max_reduce_all", np.nanmax, "ferray_ufunc::nan_max_reduce_all"),
        ("nan_min_reduce_all", np.nanmin, "ferray_ufunc::nan_min_reduce_all"),
    ]:
        cases = []
        for case_name, arr in [
            ("mixed_values", nan_vector),
            ("all_nan", all_nan_vector),
        ]:
            result = nan_extreme_result(np_func, arr)
            cases.append(case(
                case_name,
                {"x": array_to_dict(arr)},
                array_to_dict(np.array(result)),
                tolerance_ulps=0,
            ))
        save_fixture("ufunc", f"{func_name}.json",
                     make_fixture(f"numpy.{np_func.__name__}", ferray_name, cases))

    for func_name, np_func, ferray_name in [
        ("nan_max_reduce_axes", np.nanmax, "ferray_ufunc::nan_max_reduce_axes"),
        ("nan_min_reduce_axes", np.nanmin, "ferray_ufunc::nan_min_reduce_axes"),
    ]:
        cases = []
        for case_name, axes, keepdims in [
            ("axes_0_2", (0, 2), False),
            ("axes_0_2_keepdims", (0, 2), True),
        ]:
            result = nan_extreme_result(np_func, nan_cube, axis=axes,
                                        keepdims=keepdims)
            cases.append(case(
                case_name,
                {"x": array_to_dict(nan_cube), "axis": list(axes),
                 "keepdims": keepdims},
                array_to_dict(result),
                tolerance_ulps=0,
            ))
        save_fixture("ufunc", f"{func_name}.json",
                     make_fixture(f"numpy.{np_func.__name__}", ferray_name, cases))

    cumulative_sources = [
        ("axis0_1d", np.array([1.0, 2.0, 3.0, 4.0], dtype="float64"), 0),
        ("axis0_2d", np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                              dtype="float64"), 0),
        ("axis1_2d", np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                              dtype="float64"), 1),
        ("empty_axis0", np.array([], dtype="float64"), 0),
    ]
    for func_name, np_func, ferray_name in [
        ("cumsum", np.cumsum, "ferray_ufunc::cumsum"),
        ("cumprod", np.cumprod, "ferray_ufunc::cumprod"),
        ("cumulative_sum", np.cumulative_sum, "ferray_ufunc::cumulative_sum"),
        ("cumulative_prod", np.cumulative_prod, "ferray_ufunc::cumulative_prod"),
    ]:
        cases = []
        for name, arr, axis in cumulative_sources:
            with np.errstate(all="ignore"):
                result = np_func(arr, axis=axis)
            cases.append(case(name, {"x": array_to_dict(arr), "axis": axis},
                              array_to_dict(result), tolerance_ulps=0))
        save_fixture("ufunc", f"{func_name}.json",
                     make_fixture(f"numpy.{func_name}", ferray_name, cases))

    nan_cumulative_sources = [
        ("axis0_1d", np.array([1.0, np.nan, 3.0, 4.0], dtype="float64"), 0),
        ("axis1_2d", np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]],
                              dtype="float64"), 1),
        ("empty_axis0", np.array([], dtype="float64"), 0),
    ]
    for func_name, np_func in [("nancumsum", np.nancumsum),
                               ("nancumprod", np.nancumprod)]:
        cases = []
        for name, arr, axis in nan_cumulative_sources:
            with np.errstate(all="ignore"):
                result = np_func(arr, axis=axis)
            cases.append(case(name, {"x": array_to_dict(arr), "axis": axis},
                              array_to_dict(result), tolerance_ulps=0))
        save_fixture("ufunc", f"{func_name}.json",
                     make_fixture(f"numpy.{func_name}", f"ferray_ufunc::{func_name}", cases))

    diff_cases = []
    for name, arr, n in [
        ("n1", np.array([1.0, 3.0, 6.0, 10.0], dtype="float64"), 1),
        ("n2", np.array([1.0, 3.0, 6.0, 10.0], dtype="float64"), 2),
        ("empty", np.array([], dtype="float64"), 1),
    ]:
        result = np.diff(arr, n=n)
        diff_cases.append(case(name, {"x": array_to_dict(arr), "n": n},
                               array_to_dict(result), tolerance_ulps=0))
    save_fixture("ufunc", "diff.json",
                 make_fixture("numpy.diff", "ferray_ufunc::diff", diff_cases))

    ediff1d_cases = []
    for name, arr, to_begin, to_end in [
        ("standard", np.array([1.0, 2.0, 4.0, 7.0], dtype="float64"), None, None),
        ("with_edges", np.array([1.0, 2.0, 4.0, 7.0], dtype="float64"),
         np.array([-1.0], dtype="float64"), np.array([99.0, 100.0], dtype="float64")),
        ("empty_with_edges", np.array([], dtype="float64"),
         np.array([-1.0], dtype="float64"), np.array([1.0], dtype="float64")),
    ]:
        result = np.ediff1d(arr, to_end=to_end, to_begin=to_begin)
        inputs = {"x": array_to_dict(arr)}
        if to_begin is not None:
            inputs["to_begin"] = array_to_dict(to_begin)
        if to_end is not None:
            inputs["to_end"] = array_to_dict(to_end)
        ediff1d_cases.append(case(name, inputs, array_to_dict(result), tolerance_ulps=0))
    save_fixture("ufunc", "ediff1d.json",
                 make_fixture("numpy.ediff1d", "ferray_ufunc::ediff1d", ediff1d_cases))

    gradient_cases = []
    for name, arr, spacing in [
        ("default_spacing", np.array([1.0, 2.0, 4.0, 7.0, 11.0], dtype="float64"), None),
        ("scalar_spacing", np.array([1.0, 4.0, 9.0, 16.0], dtype="float64"), 2.0),
    ]:
        result = np.gradient(arr) if spacing is None else np.gradient(arr, spacing)
        inputs = {"x": array_to_dict(arr)}
        if spacing is not None:
            inputs["spacing"] = spacing
        gradient_cases.append(case(name, inputs, array_to_dict(result), tolerance_ulps=0))
    save_fixture("ufunc", "gradient.json",
                 make_fixture("numpy.gradient", "ferray_ufunc::gradient", gradient_cases))

    cross_cases = []
    for name, a, b in [
        ("basis", np.array([1.0, 0.0, 0.0], dtype="float64"),
         np.array([0.0, 1.0, 0.0], dtype="float64")),
        ("general", np.array([2.0, 3.0, 4.0], dtype="float64"),
         np.array([5.0, 6.0, 7.0], dtype="float64")),
    ]:
        result = np.cross(a, b)
        cross_cases.append(case(name, {"a": array_to_dict(a), "b": array_to_dict(b)},
                                array_to_dict(result), tolerance_ulps=0))
    save_fixture("ufunc", "cross.json",
                 make_fixture("numpy.cross", "ferray_ufunc::cross", cross_cases))

    trapezoid_cases = []
    for name, y, x, dx in [
        ("default_dx", np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype="float64"), None, None),
        ("with_dx", np.array([0.0, 1.0, 4.0], dtype="float64"), None, 0.5),
        ("with_x", np.array([0.0, 1.0, 4.0], dtype="float64"),
         np.array([0.0, 1.0, 2.0], dtype="float64"), None),
        ("single", np.array([42.0], dtype="float64"), None, None),
    ]:
        if x is not None:
            result = np.trapezoid(y, x=x)
        elif dx is not None:
            result = np.trapezoid(y, dx=dx)
        else:
            result = np.trapezoid(y)
        inputs = {"y": array_to_dict(y)}
        if x is not None:
            inputs["x"] = array_to_dict(x)
        if dx is not None:
            inputs["dx"] = dx
        trapezoid_cases.append(case(name, inputs, array_to_dict(np.array(result)),
                                    tolerance_ulps=0))
    save_fixture("ufunc", "trapezoid.json",
                 make_fixture("numpy.trapezoid", "ferray_ufunc::trapezoid",
                              trapezoid_cases))

    promoted_arithmetic_cases = []
    promoted_a_1d = np.array([1, -2, 3, 4], dtype="int32")
    promoted_b_1d = np.array([0.5, 1.5, -2.0, 4.0], dtype="float64")
    promoted_a_2d = np.array([[1, 2, 3], [-4, 5, -6]], dtype="int32")
    promoted_b_2d = np.array([[0.25, -0.5, 1.5], [2.0, -3.5, 4.0]],
                             dtype="float64")
    for op_name, np_func in [
        ("add_promoted", np.add),
        ("subtract_promoted", np.subtract),
        ("multiply_promoted", np.multiply),
        ("divide_promoted", np.divide),
    ]:
        for case_name, a, b in [
            ("mixed_i32_f64_1d", promoted_a_1d, promoted_b_1d),
            ("mixed_i32_f64_2d", promoted_a_2d, promoted_b_2d),
        ]:
            with np.errstate(all="ignore"):
                result = np_func(a, b)
            promoted_arithmetic_cases.append(case(
                f"{op_name}_{case_name}",
                {"op": op_name, "a": array_to_dict(a), "b": array_to_dict(b)},
                array_to_dict(result),
                tolerance_ulps=0,
            ))
    save_fixture("ufunc", "promoted_arithmetic.json",
                 make_fixture("numpy promoted arithmetic",
                              "ferray_ufunc::{add,subtract,multiply,divide}_promoted",
                              promoted_arithmetic_cases))

    nan_reduce_arr = np.array(
        [[1.0, np.nan, 3.0], [np.nan, np.nan, np.nan], [2.0, -4.0, np.nan]],
        dtype="float64",
    )

    def save_nan_reduction_fixture(filename, np_name, ferray_name, np_func):
        cases = []
        for case_name, kwargs in [
            ("axis1", {"axis": 1, "keepdims": False}),
            ("axis1_keepdims", {"axis": 1, "keepdims": True}),
            ("axis0", {"axis": 0, "keepdims": False}),
        ]:
            with np.errstate(all="ignore"):
                result = np_func(nan_reduce_arr, **kwargs)
            cases.append(case(case_name, {
                "x": array_to_dict(nan_reduce_arr),
                "axis": kwargs["axis"],
                "keepdims": kwargs["keepdims"],
            }, array_to_dict(result), tolerance_ulps=0))
        save_fixture("ufunc", filename,
                     make_fixture(np_name, ferray_name, cases))

    save_nan_reduction_fixture("nan_add_reduce.json", "numpy.nansum",
                               "ferray_ufunc::nan_add_reduce", np.nansum)
    save_nan_reduction_fixture("nan_multiply_reduce.json", "numpy.nanprod",
                               "ferray_ufunc::nan_multiply_reduce",
                               np.nanprod)

    def save_nan_reduction_axes_fixture(filename, np_name, ferray_name, np_func):
        cases = []
        for case_name, kwargs in [
            ("axes_0_2", {"axis": (0, 2), "keepdims": False}),
            ("axes_0_2_keepdims", {"axis": (0, 2), "keepdims": True}),
        ]:
            cube = np.array(
                [
                    [[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0]],
                    [[5.0, 6.0], [np.nan, 7.0], [8.0, np.nan]],
                ],
                dtype="float64",
            )
            with np.errstate(all="ignore"):
                result = np_func(cube, **kwargs)
            cases.append(case(case_name, {
                "x": array_to_dict(cube),
                "axes": list(kwargs["axis"]),
                "keepdims": kwargs["keepdims"],
            }, array_to_dict(result), tolerance_ulps=0))
        save_fixture("ufunc", filename,
                     make_fixture(np_name, ferray_name, cases))

    save_nan_reduction_axes_fixture("nan_add_reduce_axes.json", "numpy.nansum",
                                    "ferray_ufunc::nan_add_reduce_axes",
                                    np.nansum)
    save_nan_reduction_axes_fixture("nan_multiply_reduce_axes.json",
                                    "numpy.nanprod",
                                    "ferray_ufunc::nan_multiply_reduce_axes",
                                    np.nanprod)

    for filename, np_name, ferray_name, np_func in [
        ("nan_add_reduce_all.json", "numpy.nansum",
         "ferray_ufunc::nan_add_reduce_all", np.nansum),
        ("nan_multiply_reduce_all.json", "numpy.nanprod",
         "ferray_ufunc::nan_multiply_reduce_all", np.nanprod),
    ]:
        with np.errstate(all="ignore"):
            result = np_func(nan_reduce_arr)
        save_fixture("ufunc", filename,
                     make_fixture(np_name, ferray_name, [
                         case("all", {"x": array_to_dict(nan_reduce_arr)},
                              array_to_dict(np.array(result)), tolerance_ulps=0)
                     ]))

    _generate_binary_ufunc(np.hypot, "numpy.hypot", "ferray_ufunc::hypot",
                           "ufunc", "hypot.json", inputs_override={
                               "pythagorean": (np.array([3.0, 5.0, 8.0], dtype="float64"),
                                               np.array([4.0, 12.0, 15.0], dtype="float64")),
                               "signed": (np.array([-3.0, 3.0, -5.0], dtype="float64"),
                                          np.array([4.0, -4.0, -12.0], dtype="float64")),
                               "special": (np.array([float("nan"), float("inf"), 1.0],
                                                    dtype="float64"),
                                           np.array([1.0, 1.0, float("inf")],
                                                    dtype="float64")),
                           })

    angle_input = {
        "standard_f64": np.array(
            [-2 * np.pi, -np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi, 2 * np.pi],
            dtype="float64",
        ),
        "special_f64": np.array([float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
                                dtype="float64"),
    }
    _generate_unary_ufunc(np.degrees, "numpy.degrees", "ferray_ufunc::degrees",
                          "ufunc", "degrees.json", inputs_override=angle_input)
    _generate_unary_ufunc(np.rad2deg, "numpy.rad2deg", "ferray_ufunc::rad2deg",
                          "ufunc", "rad2deg.json", inputs_override=angle_input)

    degree_input = {
        "standard_f64": np.array([-360.0, -180.0, -90.0, 0.0, 90.0, 180.0, 360.0],
                                 dtype="float64"),
        "special_f64": np.array([float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
                                dtype="float64"),
    }
    _generate_unary_ufunc(np.radians, "numpy.radians", "ferray_ufunc::radians",
                          "ufunc", "radians.json", inputs_override=degree_input)
    _generate_unary_ufunc(np.deg2rad, "numpy.deg2rad", "ferray_ufunc::deg2rad",
                          "ufunc", "deg2rad.json", inputs_override=degree_input)

    _generate_binary_ufunc(np.logaddexp, "numpy.logaddexp",
                           "ferray_ufunc::logaddexp", "ufunc",
                           "logaddexp.json", inputs_override={
                               "standard": (np.array([-2.0, -1.0, 0.0, 1.0, 2.0],
                                                     dtype="float64"),
                                            np.array([2.0, 1.0, 0.0, -1.0, -2.0],
                                                     dtype="float64")),
                               "large_gap": (np.array([-1000.0, 1000.0, -100.0],
                                                      dtype="float64"),
                                             np.array([1000.0, -1000.0, -101.0],
                                                      dtype="float64")),
                               "special": (np.array([float("nan"), float("inf"), float("-inf")],
                                                    dtype="float64"),
                                           np.array([1.0, float("inf"), float("-inf")],
                                                    dtype="float64")),
                           })
    _generate_binary_ufunc(np.logaddexp2, "numpy.logaddexp2",
                           "ferray_ufunc::logaddexp2", "ufunc",
                           "logaddexp2.json", inputs_override={
                               "standard": (np.array([-2.0, -1.0, 0.0, 1.0, 2.0],
                                                     dtype="float64"),
                                            np.array([2.0, 1.0, 0.0, -1.0, -2.0],
                                                     dtype="float64")),
                               "large_gap": (np.array([-1000.0, 1000.0, -100.0],
                                                      dtype="float64"),
                                             np.array([1000.0, -1000.0, -101.0],
                                                      dtype="float64")),
                               "special": (np.array([float("nan"), float("inf"), float("-inf")],
                                                    dtype="float64"),
                                           np.array([1.0, float("inf"), float("-inf")],
                                                    dtype="float64")),
                           })

    _generate_binary_ufunc(np.copysign, "numpy.copysign",
                           "ferray_ufunc::copysign", "ufunc",
                           "copysign.json", inputs_override={
                               "signed_zero": (np.array([1.0, -2.0, 0.0, -0.0],
                                                       dtype="float64"),
                                               np.array([-1.0, 1.0, -0.0, 0.0],
                                                        dtype="float64")),
                               "broadcast_2d": (np.array([[1.0], [-2.0]], dtype="float64"),
                                                np.array([[-1.0, 0.0, 1.0]],
                                                         dtype="float64")),
                               "special": (np.array([float("nan"), float("inf"), 1.0],
                                                    dtype="float64"),
                                           np.array([-1.0, -0.0, float("-inf")],
                                                    dtype="float64")),
                           }, tolerance=0)
    _generate_binary_ufunc(np.float_power, "numpy.float_power",
                           "ferray_ufunc::float_power", "ufunc",
                           "float_power.json", inputs_override={
                               "standard": (np.array([2.0, 3.0, 4.0], dtype="float64"),
                                            np.array([3.0, 2.0, 0.5], dtype="float64")),
                               "negative_integer_exponent": (
                                   np.array([-2.0, -3.0, 4.0], dtype="float64"),
                                   np.array([3.0, 2.0, -2.0], dtype="float64"),
                               ),
                               "special": (np.array([float("nan"), float("inf"), 1.0],
                                                    dtype="float64"),
                                           np.array([1.0, 2.0, float("nan")],
                                                    dtype="float64")),
                           })
    _generate_binary_ufunc(np.nextafter, "numpy.nextafter",
                           "ferray_ufunc::nextafter", "ufunc",
                           "nextafter.json", inputs_override={
                               "toward_positive": (np.array([0.0, 1.0, -1.0],
                                                            dtype="float64"),
                                                   np.array([1.0, 2.0, 0.0],
                                                            dtype="float64")),
                               "toward_negative": (np.array([0.0, 1.0, -1.0],
                                                            dtype="float64"),
                                                   np.array([-1.0, 0.0, -2.0],
                                                            dtype="float64")),
                               "equal_and_nan": (np.array([1.0, float("nan"), 2.0],
                                                          dtype="float64"),
                                                 np.array([1.0, 0.0, float("nan")],
                                                          dtype="float64")),
                               "broadcast_2d": (np.array([[0.0], [1.0]], dtype="float64"),
                                                np.array([[1.0, -1.0]], dtype="float64")),
                           }, tolerance=0)

    _generate_unary_ufunc(np.spacing, "numpy.spacing", "ferray_ufunc::spacing",
                          "ufunc", "spacing.json", inputs_override={
                              "standard_f64": np.array([-2.0, -1.0, -0.0, 0.0, 1.0, 2.0],
                                                       dtype="float64"),
                              "special_f64": np.array(
                                  [float("nan"), float("inf"), float("-inf")],
                                  dtype="float64",
                              ),
                          }, tolerance=0)

    ldexp_cases = []
    for name, x, n in [
        ("standard", np.array([0.5, 1.0, 2.0, -3.0], dtype="float64"),
         np.array([1, 2, -1, 3], dtype="int32")),
        ("broadcast_2d", np.array([[1.0], [3.0]], dtype="float64"),
         np.array([[1, 2, 3]], dtype="int32")),
        ("special", np.array([float("nan"), float("inf"), float("-inf"), 0.0],
                             dtype="float64"),
         np.array([1, -2, 3, 10], dtype="int32")),
        ("empty", np.array([], dtype="float64"), np.array([], dtype="int32")),
    ]:
        with np.errstate(all="ignore"):
            result = np.ldexp(x, n)
        ldexp_cases.append(case(name, {"x": array_to_dict(x), "n": array_to_dict(n)},
                                array_to_dict(result), tolerance_ulps=0))
    save_fixture("ufunc", "ldexp.json",
                 make_fixture("numpy.ldexp", "ferray_ufunc::ldexp", ldexp_cases))

    frexp_cases = []
    for name, arr in {
        "standard": np.array([0.0, -0.0, 1.0, -2.5, 1024.0], dtype="float64"),
        "special": np.array([float("nan"), float("inf"), float("-inf")], dtype="float64"),
        "2d": np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float64"),
        "empty": np.array([], dtype="float64"),
    }.items():
        with np.errstate(all="ignore"):
            mantissa, exponent = np.frexp(arr)
        frexp_cases.append(case(name, {"x": array_to_dict(arr)}, {
            "mantissa": array_to_dict(mantissa),
            "exponent": array_to_dict(exponent, "int32"),
        }, tolerance_ulps=0))
    save_fixture("ufunc", "frexp.json",
                 make_fixture("numpy.frexp", "ferray_ufunc::frexp", frexp_cases))

    modf_cases = []
    for name, arr in {
        "standard": np.array([0.0, -0.0, 1.25, -2.75, 3.0], dtype="float64"),
        "special": np.array([float("nan"), float("inf"), float("-inf")], dtype="float64"),
        "2d": np.array([[1.25, -2.75], [3.5, -4.5]], dtype="float64"),
        "empty": np.array([], dtype="float64"),
    }.items():
        with np.errstate(all="ignore"):
            fractional, integral = np.modf(arr)
        modf_cases.append(case(name, {"x": array_to_dict(arr)}, {
            "fractional": array_to_dict(fractional),
            "integral": array_to_dict(integral),
        }, tolerance_ulps=0))
    save_fixture("ufunc", "modf.json",
                 make_fixture("numpy.modf", "ferray_ufunc::modf", modf_cases))

    clip_ord_cases = []
    for name, arr, a_min, a_max in [
        ("int64", np.array([-10, 0, 5, 100, 255], dtype="int64"), 0, 200),
        ("int64_negative_bounds", np.array([-10, -5, 0, 5, 10], dtype="int64"), -6, 6),
        ("empty_int64", np.array([], dtype="int64"), 0, 10),
    ]:
        result = np.clip(arr, a_min, a_max)
        clip_ord_cases.append(case(name, {
            "x": array_to_dict(arr),
            "a_min": a_min,
            "a_max": a_max,
        }, array_to_dict(result), tolerance_ulps=0))
    save_fixture("ufunc", "clip_ord.json",
                 make_fixture("numpy.clip", "ferray_ufunc::clip_ord", clip_ord_cases))

    for func_name, np_func in [
        ("equal", np.equal),
        ("not_equal", np.not_equal),
        ("less", np.less),
        ("less_equal", np.less_equal),
        ("greater", np.greater),
        ("greater_equal", np.greater_equal),
    ]:
        _generate_binary_ufunc(np_func, f"numpy.{func_name}", f"ferray_ufunc::{func_name}",
                               "ufunc", f"{func_name}.json", tolerance=0)

    # Comparison helpers with scalar bool results or tolerance parameters.
    cases = []
    for name, a, b, rtol, atol, equal_nan in [
        ("standard", np.array([1.0, 2.0, 3.0], dtype="float64"),
         np.array([1.0 + 1e-9, 2.1, 3.0], dtype="float64"), 1e-5, 1e-8, False),
        ("equal_nan_false", np.array([np.nan, 1.0], dtype="float64"),
         np.array([np.nan, 1.0], dtype="float64"), 1e-5, 1e-8, False),
        ("equal_nan_true", np.array([np.nan, 1.0], dtype="float64"),
         np.array([np.nan, 1.0], dtype="float64"), 1e-5, 1e-8, True),
        ("matching_infinities", np.array([np.inf, -np.inf, 1.0], dtype="float64"),
         np.array([np.inf, -np.inf, np.inf], dtype="float64"), 1e-5, 1e-8, False),
        ("broadcast_2d", np.array([[1.0, 2.0, 3.0], [1.0, 2.01, 3.0]], dtype="float64"),
         np.array([1.0, 2.0, 4.0], dtype="float64"), 1e-2, 1e-8, False),
    ]:
        with np.errstate(all="ignore"):
            result = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
        cases.append(case(name, {
            "a": array_to_dict(a),
            "b": array_to_dict(b),
            "rtol": rtol,
            "atol": atol,
            "equal_nan": equal_nan,
        }, array_to_dict(result), tolerance_ulps=0))
    save_fixture("ufunc", "isclose.json", make_fixture("numpy.isclose", "ferray_ufunc::isclose", cases))

    cases = []
    for name, a, b, rtol, atol in [
        ("all_close", np.array([1.0, 2.0, 3.0], dtype="float64"),
         np.array([1.0 + 1e-9, 2.0 + 1e-9, 3.0 + 1e-9], dtype="float64"), 1e-5, 1e-8),
        ("not_all_close", np.array([1.0, 2.0, 3.0], dtype="float64"),
         np.array([1.0, 2.0, 4.0], dtype="float64"), 1e-5, 1e-8),
        ("broadcast_true", np.array([[1.0, 2.0, 3.0], [1.0, 2.01, 3.0]], dtype="float64"),
         np.array([1.0, 2.0, 3.0], dtype="float64"), 1e-2, 1e-8),
        ("matching_infinities", np.array([np.inf, -np.inf], dtype="float64"),
         np.array([np.inf, -np.inf], dtype="float64"), 1e-5, 1e-8),
    ]:
        with np.errstate(all="ignore"):
            result = np.array(np.allclose(a, b, rtol=rtol, atol=atol))
        cases.append(case(name, {
            "a": array_to_dict(a),
            "b": array_to_dict(b),
            "rtol": rtol,
            "atol": atol,
        }, array_to_dict(result), tolerance_ulps=0))
    save_fixture("ufunc", "allclose.json", make_fixture("numpy.allclose", "ferray_ufunc::allclose", cases))

    cases = []
    for name, a, b in [
        ("same_shape_equal", np.array([1.0, 2.0, 3.0], dtype="float64"),
         np.array([1.0, 2.0, 3.0], dtype="float64")),
        ("same_shape_not_equal", np.array([1.0, 2.0, 3.0], dtype="float64"),
         np.array([1.0, 2.0, 4.0], dtype="float64")),
        ("broadcastable_equal_is_false", np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype="float64"),
         np.array([1.0, 2.0, 3.0], dtype="float64")),
        ("nan_default_false", np.array([np.nan], dtype="float64"),
         np.array([np.nan], dtype="float64")),
    ]:
        result = np.array(np.array_equal(a, b))
        cases.append(case(name, {"a": array_to_dict(a), "b": array_to_dict(b)},
                          array_to_dict(result), tolerance_ulps=0))
    save_fixture("ufunc", "array_equal.json",
                 make_fixture("numpy.array_equal", "ferray_ufunc::array_equal", cases))

    cases = []
    for name, a, b in [
        ("same_shape_equal", np.array([1.0, 2.0, 3.0], dtype="float64"),
         np.array([1.0, 2.0, 3.0], dtype="float64")),
        ("same_shape_not_equal", np.array([1.0, 2.0, 3.0], dtype="float64"),
         np.array([1.0, 2.0, 4.0], dtype="float64")),
        ("broadcastable_equal", np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype="float64"),
         np.array([1.0, 2.0, 3.0], dtype="float64")),
        ("broadcastable_not_equal", np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 4.0]], dtype="float64"),
         np.array([1.0, 2.0, 3.0], dtype="float64")),
        ("incompatible_false", np.array([1.0, 2.0], dtype="float64"),
         np.array([1.0, 2.0, 3.0], dtype="float64")),
    ]:
        result = np.array(np.array_equiv(a, b))
        cases.append(case(name, {"a": array_to_dict(a), "b": array_to_dict(b)},
                          array_to_dict(result), tolerance_ulps=0))
    save_fixture("ufunc", "array_equiv.json",
                 make_fixture("numpy.array_equiv", "ferray_ufunc::array_equiv", cases))

    # Bitwise ufuncs. Signed `bitwise_count` intentionally includes the
    # minimum i64 value because NumPy counts set bits in abs(x), not in the
    # two's-complement representation.
    bitwise_pairs = {
        "standard_i64": (np.array([-3, -2, -1, 0, 1, 2, 3], dtype="int64"),
                         np.array([1, 2, 3, 4, -1, -2, -3], dtype="int64")),
        "broadcast_i64": (np.array([[1], [2]], dtype="int64"),
                          np.array([[3, 4, 5]], dtype="int64")),
        "bool": (np.array([True, False, True], dtype="bool"),
                 np.array([False, False, True], dtype="bool")),
        "empty_i64": (np.array([], dtype="int64"), np.array([], dtype="int64")),
    }
    for func_name, np_func in [
        ("bitwise_and", np.bitwise_and),
        ("bitwise_or", np.bitwise_or),
        ("bitwise_xor", np.bitwise_xor),
    ]:
        _generate_binary_ufunc(np_func, f"numpy.{func_name}", f"ferray_ufunc::{func_name}",
                               "ufunc", f"{func_name}.json",
                               inputs_override=bitwise_pairs, tolerance=0)

    bitwise_unary = {
        "signed_i64": np.array([-3, -1, 0, 1, 3], dtype="int64"),
        "bool": np.array([True, False], dtype="bool"),
        "empty_i64": np.array([], dtype="int64"),
    }
    for func_name, np_func in [
        ("bitwise_not", np.bitwise_not),
        ("invert", np.invert),
    ]:
        cases = []
        for name, arr in bitwise_unary.items():
            result = np_func(arr)
            cases.append(case(name, {"x": array_to_dict(arr)}, array_to_dict(result),
                              tolerance_ulps=0))
        save_fixture("ufunc", f"{func_name}.json",
                     make_fixture(f"numpy.{func_name}", f"ferray_ufunc::{func_name}", cases))

    for func_name, np_func in [
        ("isneginf", np.isneginf),
        ("isposinf", np.isposinf),
        ("signbit", np.signbit),
    ]:
        _generate_unary_ufunc(np_func, f"numpy.{func_name}", f"ferray_ufunc::{func_name}",
                              "ufunc", f"{func_name}.json", inputs_override={
                                  "standard_f64": np.array(
                                      [float("-inf"), -1.0, -0.0, 0.0, 1.0, float("inf")],
                                      dtype="float64",
                                  ),
                                  "special_f64": np.array(
                                      [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
                                      dtype="float64",
                                  ),
                              }, tolerance=0)

    shift_pairs = {
        "standard_i64": (np.array([-8, -1, 0, 1, 8], dtype="int64"),
                         np.array([0, 1, 2, 3, 4], dtype="uint32")),
        "broadcast_i64": (np.array([[1], [2]], dtype="int64"),
                          np.array([[0, 1, 2]], dtype="uint32")),
        "empty_i64": (np.array([], dtype="int64"), np.array([], dtype="uint32")),
    }
    for func_name, np_func in [
        ("left_shift", np.left_shift),
        ("right_shift", np.right_shift),
    ]:
        _generate_binary_ufunc(np_func, f"numpy.{func_name}", f"ferray_ufunc::{func_name}",
                               "ufunc", f"{func_name}.json",
                               inputs_override=shift_pairs, tolerance=0)

    cases = []
    for name, arr in {
        "signed_i64": np.array([np.iinfo(np.int64).min, -3, -2, -1, 0, 1, 2, 3,
                                np.iinfo(np.int64).max], dtype="int64"),
        "unsigned_u32": np.array([0, 1, 15, 255, np.iinfo(np.uint32).max],
                                 dtype="uint32"),
        "bool": np.array([True, False, True], dtype="bool"),
        "empty_i64": np.array([], dtype="int64"),
    }.items():
        result = np.bitwise_count(arr)
        cases.append(case(name, {"x": array_to_dict(arr)}, array_to_dict(result),
                          tolerance_ulps=0))
    save_fixture("ufunc", "bitwise_count.json",
                 make_fixture("numpy.bitwise_count", "ferray_ufunc::bitwise_count", cases))

    # Complex accessors and predicates.
    complex_values = np.array([
        1.0 + 2.0j,
        3.0 - 4.0j,
        -1.0 + 0.0j,
        complex(-0.0, -0.0),
    ], dtype="complex128")
    complex_values_2d = np.array([[1.0 + 0.0j, 0.0 + 1.0j],
                                  [-1.0 - 1.0j, 2.0 + 3.0j]], dtype="complex128")
    complex_inputs = {
        "standard_complex128": complex_values,
        "2d_complex128": complex_values_2d,
        "empty_complex128": np.array([], dtype="complex128"),
    }

    for func_name, np_func, ferray_name, filename in [
        ("real", np.real, "ferray_ufunc::real", "real.json"),
        ("imag", np.imag, "ferray_ufunc::imag", "imag.json"),
        ("angle", np.angle, "ferray_ufunc::angle", "angle.json"),
        ("abs", np.abs, "ferray_ufunc::abs", "abs_complex.json"),
    ]:
        cases = []
        for name, arr in complex_inputs.items():
            result = np_func(arr)
            cases.append(case(name, {"x": array_to_dict(arr, "complex128")},
                              array_to_dict(result, str(result.dtype)), tolerance_ulps=4))
        save_fixture("ufunc", filename,
                     make_fixture(f"numpy.{func_name}", ferray_name, cases))

    for func_name, np_func in [("conj", np.conj), ("conjugate", np.conjugate)]:
        cases = []
        for name, arr in complex_inputs.items():
            result = np_func(arr)
            cases.append(case(name, {"x": array_to_dict(arr, "complex128")},
                              array_to_dict(result, "complex128"), tolerance_ulps=0))
        save_fixture("ufunc", f"{func_name}.json",
                     make_fixture(f"numpy.{func_name}", f"ferray_ufunc::{func_name}", cases))

    for func_name, np_func, ferray_name, filename in [
        ("iscomplex", np.iscomplex, "ferray_ufunc::iscomplex", "iscomplex.json"),
        ("isreal", np.isreal, "ferray_ufunc::isreal", "isreal.json"),
    ]:
        cases = []
        for name, arr in complex_inputs.items():
            result = np_func(arr)
            cases.append(case(name, {"x": array_to_dict(arr, "complex128")},
                              array_to_dict(result, "bool"), tolerance_ulps=0))
        save_fixture("ufunc", filename,
                     make_fixture(f"numpy.{func_name}", ferray_name, cases))

    real_values = np.array([1.0, 0.0, -2.5, np.nan], dtype="float64")
    for func_name, np_func, ferray_name, filename in [
        ("iscomplex", np.iscomplex, "ferray_ufunc::iscomplex_real", "iscomplex_real.json"),
        ("isreal", np.isreal, "ferray_ufunc::isreal_real", "isreal_real.json"),
    ]:
        result = np_func(real_values)
        save_fixture("ufunc", filename, make_fixture(
            f"numpy.{func_name}",
            ferray_name,
            [case("real_float64", {"x": array_to_dict(real_values)},
                  array_to_dict(result, "bool"), tolerance_ulps=0)],
        ))

    for func_name, np_func in [("iscomplexobj", np.iscomplexobj),
                               ("isrealobj", np.isrealobj)]:
        cases = []
        for name, arr in [("complex128", complex_values), ("float64", real_values)]:
            result = np.array(np_func(arr))
            cases.append(case(name, {"x": array_to_dict(arr)},
                              array_to_dict(result, "bool"), tolerance_ulps=0))
        save_fixture("ufunc", f"{func_name}.json",
                     make_fixture(f"numpy.{func_name}", f"ferray_ufunc::{func_name}", cases))

    isscalar_cases = []
    for name, arr in [
        ("zero_d_float64", np.array(1.5, dtype="float64")),
        ("one_d_float64", np.array([1.5], dtype="float64")),
        ("zero_d_complex128", np.array(1.0 + 2.0j, dtype="complex128")),
        ("one_d_complex128", np.array([1.0 + 2.0j], dtype="complex128")),
    ]:
        result = np.array(np.isscalar(arr))
        isscalar_cases.append(case(name, {"x": array_to_dict(arr)},
                                   array_to_dict(result, "bool"), tolerance_ulps=0))
    save_fixture("ufunc", "isscalar.json",
                 make_fixture("numpy.isscalar", "ferray_ufunc::isscalar",
                              isscalar_cases))

    complex_trans_inputs = {
        "standard_complex128": np.array([
            0.0 + 0.0j,
            1.0 + 2.0j,
            -1.5 + 0.5j,
            0.25 - 0.75j,
        ], dtype="complex128"),
        "2d_complex128": np.array([[1.0 + 0.0j, 0.0 + 1.0j],
                                  [-1.0 - 1.0j, 2.0 + 3.0j]], dtype="complex128"),
        "empty_complex128": np.array([], dtype="complex128"),
    }
    branch_real_cut = np.array([
        complex(2.0, 0.0),
        complex(2.0, -0.0),
        complex(-2.0, 0.0),
        complex(-2.0, -0.0),
        complex(0.5, 0.0),
    ], dtype="complex128")
    branch_imag_cut = np.array([
        complex(0.0, 2.0),
        complex(-0.0, 2.0),
        complex(0.0, -2.0),
        complex(-0.0, -2.0),
        complex(0.0, 0.5),
    ], dtype="complex128")

    for func_name, np_func, ferray_name, filename, extra_inputs in [
        ("sin", np.sin, "ferray_ufunc::sin_complex", "sin_complex.json", {}),
        ("cos", np.cos, "ferray_ufunc::cos_complex", "cos_complex.json", {}),
        ("tan", np.tan, "ferray_ufunc::tan_complex", "tan_complex.json", {}),
        ("sinh", np.sinh, "ferray_ufunc::sinh_complex", "sinh_complex.json", {}),
        ("cosh", np.cosh, "ferray_ufunc::cosh_complex", "cosh_complex.json", {}),
        ("tanh", np.tanh, "ferray_ufunc::tanh_complex", "tanh_complex.json", {}),
        ("exp", np.exp, "ferray_ufunc::exp_complex", "exp_complex.json", {}),
        ("expm1", np.expm1, "ferray_ufunc::expm1_complex", "expm1_complex.json", {}),
        ("log", np.log, "ferray_ufunc::ln_complex", "ln_complex.json", {}),
        ("log1p", np.log1p, "ferray_ufunc::log1p_complex", "log1p_complex.json", {}),
        ("log2", np.log2, "ferray_ufunc::log2_complex", "log2_complex.json", {}),
        ("log10", np.log10, "ferray_ufunc::log10_complex", "log10_complex.json", {}),
        ("sqrt", np.sqrt, "ferray_ufunc::sqrt_complex", "sqrt_complex.json", {}),
        ("arcsin", np.arcsin, "ferray_ufunc::asin_complex", "asin_complex.json",
         {"real_axis_cut": branch_real_cut}),
        ("arccos", np.arccos, "ferray_ufunc::acos_complex", "acos_complex.json",
         {"real_axis_cut": branch_real_cut}),
        ("arctan", np.arctan, "ferray_ufunc::atan_complex", "atan_complex.json",
         {"imag_axis_cut": branch_imag_cut}),
        ("arcsinh", np.arcsinh, "ferray_ufunc::asinh_complex", "asinh_complex.json",
         {"imag_axis_cut": branch_imag_cut}),
        ("arccosh", np.arccosh, "ferray_ufunc::acosh_complex", "acosh_complex.json",
         {"real_axis_cut": branch_real_cut}),
        ("arctanh", np.arctanh, "ferray_ufunc::atanh_complex", "atanh_complex.json",
         {"real_axis_cut": branch_real_cut}),
    ]:
        cases = []
        all_inputs = dict(complex_trans_inputs)
        all_inputs.update(extra_inputs)
        for name, arr in all_inputs.items():
            with np.errstate(all="ignore"):
                result = np_func(arr)
            cases.append(case(name, {"x": array_to_dict(arr, "complex128")},
                              array_to_dict(result, "complex128"), tolerance_ulps=64))
        save_fixture("ufunc", filename, make_fixture(f"numpy.{func_name}", ferray_name, cases))

    cases = []
    for name, base, exponent in [
        ("square", complex_trans_inputs["standard_complex128"], np.array(2.0 + 0.0j, dtype="complex128")),
        ("complex_exponent", complex_trans_inputs["standard_complex128"], np.array(0.5 + 0.25j, dtype="complex128")),
        ("2d_complex128", complex_trans_inputs["2d_complex128"], np.array(1.5 - 0.5j, dtype="complex128")),
        ("empty_complex128", complex_trans_inputs["empty_complex128"], np.array(2.0 + 0.0j, dtype="complex128")),
    ]:
        with np.errstate(all="ignore"):
            result = np.power(base, exponent)
        cases.append(case(name, {"x": array_to_dict(base, "complex128"),
                                 "exponent": array_to_dict(exponent, "complex128")},
                          array_to_dict(result, "complex128"), tolerance_ulps=64))
    save_fixture("ufunc", "power_complex.json",
                 make_fixture("numpy.power", "ferray_ufunc::power_complex", cases))

    # Logical ufuncs and reductions.
    logical_pairs = {
        "standard_f64": (np.array([0.0, 1.0, -2.0, np.nan, np.inf], dtype="float64"),
                         np.array([1.0, 0.0, 3.0, np.nan, 0.0], dtype="float64")),
        "broadcast_2d": (np.array([[0.0], [2.0]], dtype="float64"),
                         np.array([[0.0, 1.0, np.nan]], dtype="float64")),
        "empty": (np.array([], dtype="float64"), np.array([], dtype="float64")),
    }
    for func_name, np_func in [
        ("logical_and", np.logical_and),
        ("logical_or", np.logical_or),
        ("logical_xor", np.logical_xor),
    ]:
        _generate_binary_ufunc(np_func, f"numpy.{func_name}", f"ferray_ufunc::{func_name}",
                               "ufunc", f"{func_name}.json",
                               inputs_override=logical_pairs, tolerance=0)

    logical_unary = {
        "standard_f64": np.array([0.0, 1.0, -2.0, np.nan, np.inf, -np.inf], dtype="float64"),
        "2d_f64": np.array([[0.0, 1.0], [np.nan, 0.0]], dtype="float64"),
        "scalar_zero": np.array(0.0, dtype="float64"),
        "scalar_nonzero": np.array(-3.0, dtype="float64"),
        "empty": np.array([], dtype="float64"),
    }
    _generate_unary_ufunc(np.logical_not, "numpy.logical_not",
                          "ferray_ufunc::logical_not", "ufunc",
                          "logical_not.json", inputs_override=logical_unary,
                          tolerance=0)

    for func_name, np_func in [("all", np.all), ("any", np.any)]:
        cases = []
        for name, arr in {
            "all_true": np.array([1.0, -2.0, np.nan], dtype="float64"),
            "has_false": np.array([1.0, 0.0, np.nan], dtype="float64"),
            "all_false": np.array([0.0, 0.0], dtype="float64"),
            "empty": np.array([], dtype="float64"),
            "2d": np.array([[1.0, 0.0, np.nan], [1.0, 2.0, 3.0]], dtype="float64"),
        }.items():
            result = np.array(np_func(arr))
            cases.append(case(name, {"x": array_to_dict(arr)}, array_to_dict(result),
                              tolerance_ulps=0))
        save_fixture("ufunc", f"{func_name}.json",
                     make_fixture(f"numpy.{func_name}", f"ferray_ufunc::{func_name}", cases))

    axis_source = np.array([[1.0, 0.0, np.nan], [1.0, 2.0, 3.0]], dtype="float64")
    axis_source_1d = np.array([1.0, np.nan, 0.0], dtype="float64")
    for func_name, np_func in [("all_axis", np.all), ("any_axis", np.any)]:
        cases = []
        for name, arr, axis in [
            ("axis0_2d", axis_source, 0),
            ("axis1_2d", axis_source, 1),
            ("axis0_1d_scalar", axis_source_1d, 0),
        ]:
            result = np_func(arr, axis=axis)
            cases.append(case(name, {"x": array_to_dict(arr), "axis": axis},
                              array_to_dict(result), tolerance_ulps=0))
        save_fixture("ufunc", f"{func_name}.json",
                     make_fixture(f"numpy.{func_name}", f"ferray_ufunc::{func_name}", cases))


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
    generate_fft_gap_fixtures()


def generate_fft_gap_fixtures():
    """Generate FFT fixtures added after the original Stage 1 corpus."""
    print("Generating FFT gap fixtures...")

    # --- rfft2 / irfft2 ---
    m2d = np.arange(16, dtype="float64").reshape(4, 4)
    r2 = np.fft.rfft2(m2d)
    save_fixture("fft", "rfft2.json", make_fixture(
        "numpy.fft.rfft2",
        "ferray_fft::rfft2",
        [case("4x4", {"x": array_to_dict(m2d)}, array_to_dict(r2, "complex128"), tolerance_ulps=4)],
    ))
    save_fixture("fft", "irfft2.json", make_fixture(
        "numpy.fft.irfft2",
        "ferray_fft::irfft2",
        [case("roundtrip_4x4", {"x": array_to_dict(r2, "complex128"), "s": [4, 4]},
              array_to_dict(np.fft.irfft2(r2, s=(4, 4))), tolerance_ulps=4)],
    ))

    # --- rfftn / irfftn ---
    a3d = np.arange(24, dtype="float64").reshape(2, 3, 4)
    rn = np.fft.rfftn(a3d)
    save_fixture("fft", "rfftn.json", make_fixture(
        "numpy.fft.rfftn",
        "ferray_fft::rfftn",
        [case("2x3x4", {"x": array_to_dict(a3d)}, array_to_dict(rn, "complex128"), tolerance_ulps=4)],
    ))
    save_fixture("fft", "irfftn.json", make_fixture(
        "numpy.fft.irfftn",
        "ferray_fft::irfftn",
        [case("roundtrip_2x3x4", {"x": array_to_dict(rn, "complex128"), "s": [2, 3, 4]},
              array_to_dict(np.fft.irfftn(rn, s=(2, 3, 4))), tolerance_ulps=4)],
    ))

    # --- hfft / ihfft ---
    sig = np.array([1.0, 2.0, 3.0, 4.0], dtype="float64")
    ih = np.fft.ihfft(sig)
    save_fixture("fft", "ihfft.json", make_fixture(
        "numpy.fft.ihfft",
        "ferray_fft::ihfft",
        [case("real_4", {"x": array_to_dict(sig)}, array_to_dict(ih, "complex128"), tolerance_ulps=4)],
    ))
    save_fixture("fft", "hfft.json", make_fixture(
        "numpy.fft.hfft",
        "ferray_fft::hfft",
        [case("roundtrip_4", {"x": array_to_dict(ih, "complex128"), "n": 4},
              array_to_dict(np.fft.hfft(ih, n=4)), tolerance_ulps=4)],
    ))


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

    # --- Masking constructors ---
    constructor_data = np.array([[-2.0, -1.0, 0.0],
                                 [1.0, 2.0, 3.0]], dtype="float64")
    constructor_condition = np.array([[False, True, False],
                                      [True, False, True]], dtype="bool")
    cases = []

    def masked_constructor_case(name, op, inputs, result):
        return case(
            name,
            inputs,
            {
                "data": array_to_dict(np.array(result.data, dtype="float64")),
                "mask": array_to_dict(np.ma.getmaskarray(result).astype("bool"), "bool"),
                "fill_value": _serialize_value(result.fill_value),
            },
            tolerance_ulps=0,
        )

    result = np.ma.masked_where(constructor_condition, constructor_data)
    cases.append(masked_constructor_case(
        "masked_where",
        "masked_where",
        {"op": "masked_where",
         "data": array_to_dict(constructor_data),
         "condition": array_to_dict(constructor_condition, "bool")},
        result,
    ))
    for op, np_func, value in [
        ("masked_equal", np.ma.masked_equal, 2.0),
        ("masked_not_equal", np.ma.masked_not_equal, 2.0),
        ("masked_greater", np.ma.masked_greater, 1.0),
        ("masked_greater_equal", np.ma.masked_greater_equal, 1.0),
        ("masked_less", np.ma.masked_less, 0.0),
        ("masked_less_equal", np.ma.masked_less_equal, 0.0),
    ]:
        result = np_func(constructor_data, value)
        cases.append(masked_constructor_case(
            op,
            op,
            {"op": op,
             "data": array_to_dict(constructor_data),
             "value": value},
            result,
        ))
    for op, np_func, v1, v2 in [
        ("masked_inside", np.ma.masked_inside, -1.0, 2.0),
        ("masked_inside_swapped", np.ma.masked_inside, 2.0, -1.0),
        ("masked_outside", np.ma.masked_outside, -1.0, 2.0),
        ("masked_outside_swapped", np.ma.masked_outside, 2.0, -1.0),
    ]:
        result = np_func(constructor_data, v1, v2)
        cases.append(masked_constructor_case(
            op,
            op,
            {"op": op.split("_swapped")[0],
             "data": array_to_dict(constructor_data),
             "v1": v1,
             "v2": v2},
            result,
        ))

    fix_data = np.array([1.0, np.nan, 3.0, np.inf, -np.inf], dtype="float64")
    fix_value = -99.0
    result = np.ma.fix_invalid(fix_data, fill_value=fix_value)
    cases.append(masked_constructor_case(
        "fix_invalid",
        "fix_invalid",
        {"op": "fix_invalid",
         "data": array_to_dict(fix_data),
         "fill_value": fix_value},
        result,
    ))
    save_fixture("ma", "masked_constructors.json",
                 make_fixture("numpy.ma masking constructors",
                              "ferray_ma::{fix_invalid,masked_where,masked_equal,masked_not_equal,masked_greater,masked_greater_equal,masked_less,masked_less_equal,masked_inside,masked_outside}",
                              cases))

    # --- Masked arithmetic ---
    lhs_data = np.array([1.0, 2.0, 0.0, 4.0], dtype="float64")
    lhs_mask = np.array([False, True, False, False], dtype="bool")
    rhs_data = np.array([1.0, 0.0, 0.0, 2.0], dtype="float64")
    rhs_mask = np.array([False, False, False, True], dtype="bool")
    array_rhs = np.array([10.0, 0.0, 0.0, 2.0], dtype="float64")
    lhs = np.ma.array(lhs_data, mask=lhs_mask)
    rhs = np.ma.array(rhs_data, mask=rhs_mask)

    cases = []

    def arithmetic_case(name, op, inputs, result):
        return case(
            name,
            inputs,
            {
                "data": array_to_dict(np.array(result.data, dtype="float64")),
                "mask": array_to_dict(np.ma.getmaskarray(result).astype("bool"), "bool"),
                "fill_value": _serialize_value(result.fill_value),
            },
            tolerance_ulps=4,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for op, result in [
            ("masked_add", lhs + rhs),
            ("masked_sub", lhs - rhs),
            ("masked_mul", lhs * rhs),
            ("masked_div", lhs / rhs),
        ]:
            cases.append(arithmetic_case(
                op,
                op,
                {"op": op,
                 "lhs_data": array_to_dict(lhs_data),
                 "lhs_mask": array_to_dict(lhs_mask, "bool"),
                 "rhs_data": array_to_dict(rhs_data),
                 "rhs_mask": array_to_dict(rhs_mask, "bool")},
                result,
            ))
        for op, result in [
            ("masked_add_array", lhs + array_rhs),
            ("masked_sub_array", lhs - array_rhs),
            ("masked_mul_array", lhs * array_rhs),
            ("masked_div_array", lhs / array_rhs),
        ]:
            cases.append(arithmetic_case(
                op,
                op,
                {"op": op,
                 "lhs_data": array_to_dict(lhs_data),
                 "lhs_mask": array_to_dict(lhs_mask, "bool"),
                 "rhs_data": array_to_dict(array_rhs)},
                result,
            ))
    save_fixture("ma", "masked_arithmetic.json",
                 make_fixture("numpy.ma masked arithmetic",
                              "ferray_ma::{masked_add,masked_sub,masked_mul,masked_div,masked_add_array,masked_sub_array,masked_mul_array,masked_div_array}",
                              cases))

    # --- Masked shape manipulation + flat indexing ---
    manipulation_data = np.arange(6, dtype="float64").reshape(2, 3)
    manipulation_mask = np.array([[False, True, False],
                                  [True, False, False]], dtype="bool")
    manipulation_fill = -7.0
    manipulation_arr = np.ma.array(manipulation_data, mask=manipulation_mask)
    manipulation_arr.set_fill_value(manipulation_fill)

    squeeze_data = np.arange(6, dtype="float64").reshape(1, 2, 3, 1)
    squeeze_mask = np.array([[[[False], [True], [False]],
                              [[True], [False], [False]]]], dtype="bool")
    squeeze_arr = np.ma.array(squeeze_data, mask=squeeze_mask)
    squeeze_arr.set_fill_value(manipulation_fill)

    def masked_result_payload(result):
        return {
            "data": array_to_dict(np.array(result.data, dtype="float64")),
            "mask": array_to_dict(np.ma.getmaskarray(result).astype("bool"), "bool"),
            "fill_value": _serialize_value(result.fill_value),
        }

    def manipulation_case(name, op, source, result, **extra_inputs):
        inputs = {
            "op": op,
            "data": array_to_dict(np.array(source.data, dtype="float64")),
            "mask": array_to_dict(np.ma.getmaskarray(source).astype("bool"), "bool"),
            "fill_value": _serialize_value(source.fill_value),
        }
        inputs.update(extra_inputs)
        return case(name, inputs, masked_result_payload(result), tolerance_ulps=0)

    selector = np.array([[True, False, True],
                         [False, True, False]], dtype="bool")
    take_indices = np.array([0, 5, 2, 1], dtype="uint64")
    cases = [
        manipulation_case(
            "reshape_3x2",
            "reshape",
            manipulation_arr,
            manipulation_arr.reshape(3, 2),
            new_shape=[3, 2],
        ),
        manipulation_case(
            "ravel",
            "ravel",
            manipulation_arr,
            manipulation_arr.ravel(),
        ),
        manipulation_case(
            "flatten",
            "flatten",
            manipulation_arr,
            manipulation_arr.flatten(),
        ),
        manipulation_case(
            "transpose_default",
            "transpose",
            manipulation_arr,
            manipulation_arr.transpose(),
        ),
        manipulation_case(
            "transpose_axes",
            "transpose",
            manipulation_arr,
            manipulation_arr.transpose((1, 0)),
            axes=[1, 0],
        ),
        manipulation_case(
            "t",
            "t",
            manipulation_arr,
            manipulation_arr.T,
        ),
        manipulation_case(
            "squeeze_all",
            "squeeze",
            squeeze_arr,
            np.ma.squeeze(squeeze_arr),
        ),
        manipulation_case(
            "squeeze_axis0",
            "squeeze",
            squeeze_arr,
            np.ma.squeeze(squeeze_arr, axis=0),
            axis=0,
        ),
        manipulation_case(
            "boolean_index",
            "boolean_index",
            manipulation_arr,
            manipulation_arr[selector],
            selector=array_to_dict(selector, "bool"),
        ),
        manipulation_case(
            "take",
            "take",
            manipulation_arr,
            np.ma.take(manipulation_arr, take_indices),
            indices=array_to_dict(take_indices, "uint64"),
        ),
    ]
    for flat_index in [1, 4]:
        flat = manipulation_arr.ravel()
        cases.append(case(
            f"get_flat_{flat_index}",
            {
                "op": "get_flat",
                "data": array_to_dict(np.array(manipulation_arr.data, dtype="float64")),
                "mask": array_to_dict(np.ma.getmaskarray(manipulation_arr).astype("bool"), "bool"),
                "fill_value": _serialize_value(manipulation_arr.fill_value),
                "flat_index": flat_index,
            },
            {
                "value": _serialize_value(flat.data[flat_index]),
                "masked": bool(np.ma.getmaskarray(flat)[flat_index]),
            },
            tolerance_ulps=0,
        ))
    save_fixture("ma", "masked_manipulation.json",
                 make_fixture("numpy.ma shape manipulation and take/get_flat",
                              "ferray_ma::manipulation::MaskedArray::{reshape,ravel,flatten,transpose,t,squeeze,boolean_index,take,get_flat}",
                              cases))

    # --- Masked ufunc support ---
    ufunc_data = np.array([-2.5, -1.0, -0.5, 0.0, 0.5, 1.5, 2.25], dtype="float64")
    ufunc_mask = np.array([False, True, False, False, False, False, False], dtype="bool")
    ufunc_fill = -9.0
    ufunc_arr = np.ma.array(ufunc_data, mask=ufunc_mask)
    ufunc_arr.set_fill_value(ufunc_fill)

    lhs_data = np.array([-2.0, -1.0, 0.5, 1.0, 2.0], dtype="float64")
    lhs_mask = np.array([False, True, False, False, False], dtype="bool")
    rhs_data = np.array([1.0, 0.0, 2.0, 0.0, 3.0], dtype="float64")
    rhs_mask = np.array([False, False, True, False, False], dtype="bool")
    lhs = np.ma.array(lhs_data, mask=lhs_mask)
    rhs = np.ma.array(rhs_data, mask=rhs_mask)
    lhs.set_fill_value(ufunc_fill)
    rhs.set_fill_value(77.0)

    cases = []

    def ufunc_unary_case(name, op, result):
        return case(
            name,
            {
                "op": op,
                "data": array_to_dict(ufunc_data),
                "mask": array_to_dict(ufunc_mask, "bool"),
                "fill_value": ufunc_fill,
            },
            masked_result_payload(result),
            tolerance_ulps=16,
        )

    def ufunc_binary_case(name, op, result):
        return case(
            name,
            {
                "op": op,
                "lhs_data": array_to_dict(lhs_data),
                "lhs_mask": array_to_dict(lhs_mask, "bool"),
                "lhs_fill_value": ufunc_fill,
                "rhs_data": array_to_dict(rhs_data),
                "rhs_mask": array_to_dict(rhs_mask, "bool"),
                "rhs_fill_value": 77.0,
            },
            masked_result_payload(result),
            tolerance_ulps=16,
        )

    def manual_unary_result(source, func, in_domain=None):
        source_data = np.array(source.data, dtype="float64")
        result_mask = np.ma.getmaskarray(source).astype("bool")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            computed = func(source_data)
        if in_domain is not None:
            result_mask = np.logical_or(result_mask, np.logical_not(in_domain(source_data)))
            result_data = np.where(result_mask, source_data, computed)
        else:
            result_data = np.where(result_mask, source_data, computed)
        result = np.ma.array(result_data, mask=result_mask)
        result.set_fill_value(source.fill_value)
        return result

    def manual_binary_result(left, right, func, in_domain=None):
        left_data = np.array(left.data, dtype="float64")
        right_data = np.array(right.data, dtype="float64")
        result_mask = np.logical_or(np.ma.getmaskarray(left), np.ma.getmaskarray(right)).astype("bool")
        if in_domain is not None:
            result_mask = np.logical_or(result_mask, np.logical_not(in_domain(left_data, right_data)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            computed = func(left_data, right_data)
        result_data = np.where(result_mask, left_data, computed)
        result = np.ma.array(result_data, mask=result_mask)
        result.set_fill_value(left.fill_value)
        return result

    unary_ops = [
        "absolute",
        "arccos",
        "arccosh",
        "arcsin",
        "arcsinh",
        "arctan",
        "arctanh",
        "ceil",
        "cos",
        "cosh",
        "exp",
        "exp2",
        "expm1",
        "floor",
        "log",
        "log10",
        "log1p",
        "log2",
        "negative",
        "reciprocal",
        "round",
        "sign",
        "sin",
        "sinh",
        "sqrt",
        "square",
        "tan",
        "tanh",
        "trunc",
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for op in unary_ops:
            np_func = getattr(np.ma, op, None) or getattr(np, op)
            cases.append(ufunc_unary_case(op, op, np_func(ufunc_arr)))

    domain_unary_ops = [
        ("arccos_domain", np.ma.arccos),
        ("arccosh_domain", np.ma.arccosh),
        ("arcsin_domain", np.ma.arcsin),
        ("arctanh_domain", np.ma.arctanh),
        ("log_domain", np.ma.log),
        ("log10_domain", np.ma.log10),
        ("log2_domain", np.ma.log2),
        ("sqrt_domain", np.ma.sqrt),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for op, np_func in domain_unary_ops:
            cases.append(ufunc_unary_case(op, op, np_func(ufunc_arr)))

    generic_unary = manual_unary_result(ufunc_arr, lambda x: x * 10.0 + 1.0)
    cases.append(ufunc_unary_case("masked_unary", "masked_unary", generic_unary))
    generic_unary_domain = manual_unary_result(
        ufunc_arr,
        lambda x: x + 100.0,
        lambda x: x >= 0.0,
    )
    cases.append(ufunc_unary_case(
        "masked_unary_domain",
        "masked_unary_domain",
        generic_unary_domain,
    ))

    binary_ops = [
        ("add", np.ma.add),
        ("subtract", np.ma.subtract),
        ("multiply", np.ma.multiply),
        ("divide", np.ma.divide),
        ("power", np.ma.power),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for op, np_func in binary_ops:
            cases.append(ufunc_binary_case(op, op, np_func(lhs, rhs)))
        cases.append(ufunc_binary_case("divide_domain", "divide_domain", np.ma.divide(lhs, rhs)))

    generic_binary = manual_binary_result(lhs, rhs, lambda x, y: x * 2.0 + y)
    cases.append(ufunc_binary_case("masked_binary", "masked_binary", generic_binary))
    generic_binary_domain = manual_binary_result(
        lhs,
        rhs,
        lambda x, y: x - y,
        lambda _x, y: y > 0.0,
    )
    cases.append(ufunc_binary_case(
        "masked_binary_domain",
        "masked_binary_domain",
        generic_binary_domain,
    ))

    save_fixture("ma", "masked_ufunc_support.json",
                 make_fixture("numpy masked ufunc protocol",
                              "ferray_ma::ufunc_support::{masked_unary,masked_binary,masked_unary_domain,masked_binary_domain,absolute,add,arccos,arccos_domain,arccosh,arccosh_domain,arcsin,arcsin_domain,arcsinh,arctan,arctanh,arctanh_domain,ceil,cos,cosh,divide,divide_domain,exp,exp2,expm1,floor,log,log10,log10_domain,log1p,log2,log2_domain,log_domain,multiply,negative,power,reciprocal,round,sign,sin,sinh,sqrt,sqrt_domain,square,subtract,tan,tanh,trunc}",
                              cases))

    # --- Masked extras helper functions ---
    helper_data = np.array([1.0, 2.0, 3.0], dtype="float64")
    helper_mask = np.array([False, True, False], dtype="bool")
    helper_arr = np.ma.array(helper_data, mask=helper_mask)
    helper_arr.set_fill_value(-5.0)

    cases = []
    cases.append(case(
        "default_fill_values",
        {"op": "default_fill_values"},
        {
            "f64": _serialize_value(np.ma.default_fill_value(np.float64(0.0))),
            "f32": _serialize_value(np.float32(np.ma.default_fill_value(np.float32(0.0)))),
            "bool": _serialize_value(np.ma.default_fill_value(np.bool_(False))),
            "i64": _serialize_value(np.ma.default_fill_value(np.int64(0))),
        },
        tolerance_ulps=0,
    ))
    cases.append(case(
        "extremum_fill_values",
        {"op": "extremum_fill_values"},
        {
            "maximum_f64": _serialize_value(np.ma.maximum_fill_value(np.float64(0.0))),
            "minimum_f64": _serialize_value(np.ma.minimum_fill_value(np.float64(0.0))),
        },
        tolerance_ulps=0,
    ))
    cases.append(case(
        "nomask",
        {"op": "nomask"},
        {"value": _serialize_value(np.ma.nomask)},
        tolerance_ulps=0,
    ))

    mask_lhs = np.array([False, True, False], dtype="bool")
    mask_rhs = np.array([True, False, False], dtype="bool")
    cases.append(case(
        "mask_or",
        {
            "op": "mask_or",
            "lhs_mask": array_to_dict(mask_lhs, "bool"),
            "rhs_mask": array_to_dict(mask_rhs, "bool"),
        },
        array_to_dict(np.ma.mask_or(mask_lhs, mask_rhs), "bool"),
        tolerance_ulps=0,
    ))

    make_mask_values = np.array([False, True, True], dtype="bool")
    cases.append(case(
        "make_mask",
        {"op": "make_mask", "values": array_to_dict(make_mask_values, "bool")},
        array_to_dict(np.ma.make_mask(make_mask_values, dtype=bool, shrink=False), "bool"),
        tolerance_ulps=0,
    ))
    cases.append(case(
        "make_mask_none",
        {"op": "make_mask_none", "shape": [2, 2]},
        array_to_dict(np.ma.make_mask_none((2, 2)), "bool"),
        tolerance_ulps=0,
    ))

    masked_all_result = np.ma.masked_all((2, 3), dtype="float64")
    cases.append(case(
        "masked_all",
        {"op": "masked_all", "shape": [2, 3]},
        {
            "shape": list(masked_all_result.shape),
            "mask": array_to_dict(np.ma.getmaskarray(masked_all_result).astype("bool"), "bool"),
            "fill_value": _serialize_value(masked_all_result.fill_value),
        },
        tolerance_ulps=0,
    ))
    masked_all_like_reference = np.arange(6, dtype="float64").reshape(2, 3)
    masked_all_like_result = np.ma.masked_all_like(masked_all_like_reference)
    cases.append(case(
        "masked_all_like",
        {
            "op": "masked_all_like",
            "reference": array_to_dict(masked_all_like_reference),
        },
        {
            "shape": list(masked_all_like_result.shape),
            "mask": array_to_dict(np.ma.getmaskarray(masked_all_like_result).astype("bool"), "bool"),
            "fill_value": _serialize_value(masked_all_like_result.fill_value),
        },
        tolerance_ulps=0,
    ))

    masked_values_data = np.array([1.0, 1.0001, 2.0], dtype="float64")
    masked_values_result = np.ma.masked_values(masked_values_data, 1.0, rtol=1e-3, atol=0.0)
    cases.append(case(
        "masked_values",
        {
            "op": "masked_values",
            "data": array_to_dict(masked_values_data),
            "value": 1.0,
            "rtol": 1e-3,
            "atol": 0.0,
        },
        masked_result_payload(masked_values_result),
        tolerance_ulps=0,
    ))
    cases.append(case(
        "getmaskarray",
        {
            "op": "getmaskarray",
            "data": array_to_dict(helper_data),
            "mask": array_to_dict(helper_mask, "bool"),
        },
        array_to_dict(np.ma.getmaskarray(helper_arr).astype("bool"), "bool"),
        tolerance_ulps=0,
    ))
    cases.append(case(
        "is_masked_true",
        {
            "op": "is_masked",
            "data": array_to_dict(helper_data),
            "mask": array_to_dict(helper_mask, "bool"),
        },
        {"value": _serialize_value(np.ma.is_masked(helper_arr))},
        tolerance_ulps=0,
    ))
    unmasked_arr = np.ma.array(helper_data, mask=np.zeros_like(helper_mask, dtype="bool"))
    cases.append(case(
        "is_masked_false",
        {
            "op": "is_masked",
            "data": array_to_dict(helper_data),
            "mask": array_to_dict(np.zeros_like(helper_mask, dtype="bool"), "bool"),
        },
        {"value": _serialize_value(np.ma.is_masked(unmasked_arr))},
        tolerance_ulps=0,
    ))
    cases.append(case(
        "is_ma",
        {
            "op": "is_ma",
            "data": array_to_dict(helper_data),
            "mask": array_to_dict(helper_mask, "bool"),
        },
        {"value": _serialize_value(np.ma.isMA(helper_arr))},
        tolerance_ulps=0,
    ))
    cases.append(case(
        "is_masked_array",
        {
            "op": "is_masked_array",
            "data": array_to_dict(helper_data),
            "mask": array_to_dict(helper_mask, "bool"),
        },
        {"value": _serialize_value(np.ma.isMaskedArray(helper_arr))},
        tolerance_ulps=0,
    ))
    cases.append(case(
        "filled_default",
        {
            "op": "filled_default",
            "data": array_to_dict(helper_data),
            "mask": array_to_dict(helper_mask, "bool"),
            "fill_value": -5.0,
        },
        array_to_dict(helper_arr.filled()),
        tolerance_ulps=0,
    ))
    save_fixture("ma", "masked_extras_helpers.json",
                 make_fixture("numpy.ma extras helper functions",
                              "ferray_ma::{NOMASK,default_fill_value_bool,default_fill_value_f32,default_fill_value_f64,default_fill_value_i64,maximum_fill_value,minimum_fill_value,mask_or,make_mask,make_mask_none,masked_all,masked_all_like,masked_values,getmaskarray,is_masked,is_ma,is_masked_array}",
                              cases))

    # --- MaskedArray core surface and mask state ---
    core_data = np.array([1.0, 2.0, 3.0], dtype="float64")
    core_mask = np.array([False, True, False], dtype="bool")
    core_arr = np.ma.array(core_data, mask=core_mask)
    core_arr.set_fill_value(-7.0)

    common_rhs = np.ma.array([4.0, 5.0, 6.0],
                             mask=np.array([False, False, True], dtype="bool"))
    common_rhs.set_fill_value(-7.0)

    hard_arr = np.ma.array(core_data, mask=core_mask.copy())
    hard_arr.harden_mask()
    hard_arr[1] = 20.0
    hard_arr[0] = np.ma.masked
    hard_mask_after_harden = np.ma.getmaskarray(hard_arr).astype("bool")
    hard_arr.soften_mask()
    hard_arr[1] = 20.0
    hard_mask_after_soften = np.ma.getmaskarray(hard_arr).astype("bool")

    cases = []
    cases.append(case(
        "masked_array_accessors",
        {
            "op": "masked_array_accessors",
            "data": array_to_dict(core_data),
            "mask": array_to_dict(core_mask, "bool"),
            "fill_value": _serialize_value(core_arr.fill_value),
        },
        {
            "data": array_to_dict(np.array(core_arr.data, dtype="float64")),
            "mask": array_to_dict(np.ma.getmaskarray(core_arr).astype("bool"), "bool"),
            "shape": list(core_arr.shape),
            "ndim": int(core_arr.ndim),
            "size": int(core_arr.size),
            "fill_value": _serialize_value(core_arr.fill_value),
            "has_real_mask": True,
            "mask_opt": True,
            "is_hard_mask": False,
        },
        tolerance_ulps=0,
    ))
    cases.append(case(
        "masked_array_from_data",
        {
            "op": "masked_array_from_data",
            "data": array_to_dict(core_data),
        },
        {
            "data": array_to_dict(core_data),
            "mask": array_to_dict(np.zeros_like(core_mask, dtype="bool"), "bool"),
            "shape": list(core_data.shape),
            "ndim": int(core_data.ndim),
            "size": int(core_data.size),
            "fill_value": _serialize_value(np.ma.array(core_data).fill_value),
            "has_real_mask": False,
            "mask_opt": False,
        },
        tolerance_ulps=0,
    ))
    cases.append(case(
        "masked_array_mutation",
        {
            "op": "masked_array_mutation",
            "data": array_to_dict(core_data),
            "mask": array_to_dict(core_mask, "bool"),
            "fill_value": _serialize_value(core_arr.fill_value),
            "replacement_fill_value": 11.0,
            "replacement_mask": array_to_dict(np.array([True, False, True], dtype="bool"), "bool"),
        },
        {
            "data_after_data_mut": array_to_dict(np.array([1.0, 20.0, 3.0], dtype="float64")),
            "mask_after_set_mask": array_to_dict(np.array([True, False, True], dtype="bool"), "bool"),
            "fill_value_after_set": 11.0,
        },
        tolerance_ulps=0,
    ))
    cases.append(case(
        "masked_array_hard_soft",
        {
            "op": "masked_array_hard_soft",
            "data": array_to_dict(core_data),
            "mask": array_to_dict(core_mask, "bool"),
        },
        {
            "mask_after_harden": array_to_dict(hard_mask_after_harden, "bool"),
            "mask_after_soften": array_to_dict(hard_mask_after_soften, "bool"),
        },
        tolerance_ulps=0,
    ))
    cases.append(case(
        "masked_array_shares_mask",
        {
            "op": "masked_array_shares_mask",
            "data": array_to_dict(core_data),
            "mask": array_to_dict(core_mask, "bool"),
        },
        {
            "shares_after_clone": True,
            "shares_after_child_mutation": False,
            "parent_mask": array_to_dict(core_mask, "bool"),
            "child_mask": array_to_dict(np.array([True, True, False], dtype="bool"), "bool"),
        },
        tolerance_ulps=0,
    ))
    cases.append(case(
        "common_fill_value_same",
        {
            "op": "common_fill_value_same",
            "data": array_to_dict(core_data),
            "mask": array_to_dict(core_mask, "bool"),
            "fill_value": _serialize_value(core_arr.fill_value),
            "rhs_data": array_to_dict(np.array(common_rhs.data, dtype="float64")),
            "rhs_mask": array_to_dict(np.ma.getmaskarray(common_rhs).astype("bool"), "bool"),
            "rhs_fill_value": _serialize_value(common_rhs.fill_value),
        },
        {"value": _serialize_value(np.ma.common_fill_value(core_arr, common_rhs))},
        tolerance_ulps=0,
    ))
    cases.append(case(
        "ids",
        {
            "op": "ids",
            "data": array_to_dict(core_data),
            "mask": array_to_dict(core_mask, "bool"),
        },
        {"tuple_len": len(core_arr.ids())},
        tolerance_ulps=0,
    ))

    interop_data = np.array([1.0, 4.0, 9.0, 16.0], dtype="float64")
    interop_mask = np.array([False, False, True, False], dtype="bool")
    interop_arr = np.ma.array(interop_data, mask=interop_mask)
    interop_arr.set_fill_value(-1.0)

    unary_to_data = np.array([1.0, -2.0, 3.0, -4.0], dtype="float64")
    unary_to_mask = np.array([False, False, True, False], dtype="bool")
    unary_to_arr = np.ma.array(unary_to_data, mask=unary_to_mask)

    binary_lhs_data = np.array([10.0, 20.0, 30.0], dtype="float64")
    binary_lhs_mask = np.array([False, True, False], dtype="bool")
    binary_rhs_data = np.array([1.0, 2.0, 3.0], dtype="float64")
    binary_rhs_mask = np.array([False, False, True], dtype="bool")
    binary_lhs = np.ma.array(binary_lhs_data, mask=binary_lhs_mask)
    binary_lhs.set_fill_value(-1.0)
    binary_rhs = np.ma.array(binary_rhs_data, mask=binary_rhs_mask)

    cases.append(case(
        "interop_into_data",
        {
            "op": "interop_into_data",
            "data": array_to_dict(core_data),
            "mask": array_to_dict(core_mask, "bool"),
            "fill_value": _serialize_value(core_arr.fill_value),
        },
        array_to_dict(np.array(core_arr.data, dtype="float64")),
        tolerance_ulps=0,
    ))
    cases.append(case(
        "interop_apply_unary",
        {
            "op": "interop_apply_unary",
            "data": array_to_dict(interop_data),
            "mask": array_to_dict(interop_mask, "bool"),
            "fill_value": _serialize_value(interop_arr.fill_value),
        },
        {
            "data": array_to_dict(np.array([1.0, 2.0, interop_arr.fill_value, 4.0],
                                           dtype="float64")),
            "mask": array_to_dict(interop_mask, "bool"),
            "fill_value": _serialize_value(interop_arr.fill_value),
        },
        tolerance_ulps=4,
    ))
    cases.append(case(
        "interop_apply_unary_to",
        {
            "op": "interop_apply_unary_to",
            "data": array_to_dict(unary_to_data),
            "mask": array_to_dict(unary_to_mask, "bool"),
            "default_for_masked": False,
        },
        {
            "data": array_to_dict(np.array([True, False, False, False], dtype="bool"), "bool"),
            "mask": array_to_dict(unary_to_mask, "bool"),
            "fill_value": False,
        },
        tolerance_ulps=0,
    ))
    cases.append(case(
        "interop_apply_binary",
        {
            "op": "interop_apply_binary",
            "data": array_to_dict(binary_lhs_data),
            "mask": array_to_dict(binary_lhs_mask, "bool"),
            "fill_value": _serialize_value(binary_lhs.fill_value),
            "rhs_data": array_to_dict(binary_rhs_data),
            "rhs_mask": array_to_dict(binary_rhs_mask, "bool"),
            "rhs_fill_value": _serialize_value(binary_rhs.fill_value),
        },
        {
            "data": array_to_dict(np.array([11.0, binary_lhs.fill_value, binary_lhs.fill_value],
                                           dtype="float64")),
            "mask": array_to_dict(np.logical_or(binary_lhs_mask, binary_rhs_mask), "bool"),
            "fill_value": _serialize_value(binary_lhs.fill_value),
        },
        tolerance_ulps=0,
    ))
    cases.append(case(
        "ma_apply_unary_masked",
        {
            "op": "ma_apply_unary_masked",
            "data": array_to_dict(core_data),
            "mask": array_to_dict(core_mask, "bool"),
            "fill_value": _serialize_value(core_arr.fill_value),
        },
        {
            "data": array_to_dict(np.array([2.0, core_arr.fill_value, 6.0], dtype="float64")),
            "mask": array_to_dict(core_mask, "bool"),
            "fill_value": _serialize_value(core_arr.fill_value),
        },
        tolerance_ulps=0,
    ))
    cases.append(case(
        "ma_apply_unary_plain",
        {
            "op": "ma_apply_unary_plain",
            "data": array_to_dict(core_data),
        },
        {
            "data": array_to_dict(core_data * 2.0),
            "mask": array_to_dict(np.zeros_like(core_mask, dtype="bool"), "bool"),
            "fill_value": 0.0,
            "has_real_mask": False,
        },
        tolerance_ulps=0,
    ))
    cases.append(case(
        "io_roundtrip",
        {
            "op": "io_roundtrip",
            "data": array_to_dict(core_data),
            "mask": array_to_dict(core_mask, "bool"),
            "fill_value": _serialize_value(core_arr.fill_value),
        },
        {
            "data": array_to_dict(core_data),
            "mask": array_to_dict(core_mask, "bool"),
            "fill_value": _serialize_value(np.ma.array(core_data, mask=core_mask).fill_value),
        },
        tolerance_ulps=0,
    ))

    save_fixture("ma", "masked_core_surface.json",
                 make_fixture("numpy.ma MaskedArray core surface and mask state",
                              "ferray_ma::{MaskedArray,MaskAware,common_fill_value,ids,ma_apply_unary} ferray_ma::masked_array::MaskedArray ferray_ma::mask_ops::MaskedArray::{harden_mask,soften_mask} ferray_ma::interop::{MaskAware,ma_apply_unary} ferray_ma::io::{save_masked,load_masked}",
                              cases))

    # --- Masked extras numeric and shape methods ---
    method_data = np.array([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]], dtype="float64")
    method_mask = np.array([[False, True, False],
                            [True, False, False]], dtype="bool")
    method_arr = np.ma.array(method_data, mask=method_mask)
    method_arr.set_fill_value(-7.0)
    method_weights = np.array([[1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0]], dtype="float64")

    cases = []

    def method_inputs(op, **extra_inputs):
        inputs = {
            "op": op,
            "data": array_to_dict(method_data),
            "mask": array_to_dict(method_mask, "bool"),
            "fill_value": _serialize_value(method_arr.fill_value),
        }
        inputs.update(extra_inputs)
        return inputs

    for op, result in [
        ("prod", method_arr.prod()),
        ("ptp", method_arr.ptp()),
        ("median", np.ma.median(method_arr)),
        ("average", np.ma.average(method_arr)),
        ("trace", method_arr.trace()),
    ]:
        cases.append(case(
            op,
            method_inputs(op),
            {"value": _serialize_value(result)},
            tolerance_ulps=4,
        ))
    for op, result in [
        ("argmin", method_arr.argmin()),
        ("argmax", method_arr.argmax()),
    ]:
        cases.append(case(
            op,
            method_inputs(op),
            {"value": _serialize_value(result)},
            tolerance_ulps=0,
        ))

    avg, sum_weights = np.ma.average(method_arr, returned=True)
    cases.append(case(
        "average_returned",
        method_inputs("average_returned"),
        {
            "average": _serialize_value(avg),
            "sum_weights": _serialize_value(sum_weights),
        },
        tolerance_ulps=4,
    ))
    avg_w, sum_weights_w = np.ma.average(method_arr, weights=method_weights, returned=True)
    cases.append(case(
        "average_weighted_returned",
        method_inputs("average_weighted_returned",
                      weights=array_to_dict(method_weights)),
        {
            "average": _serialize_value(avg_w),
            "sum_weights": _serialize_value(sum_weights_w),
        },
        tolerance_ulps=4,
    ))

    for op, result in [
        ("cumsum_flat", method_arr.cumsum()),
        ("cumprod_flat", method_arr.cumprod()),
        ("anom", np.ma.anom(method_arr)),
        ("clip", method_arr.clip(2.0, 5.0)),
        ("repeat", method_arr.repeat(2)),
        ("diagonal", method_arr.diagonal()),
        ("diagonal_offset1", method_arr.diagonal(1)),
    ]:
        inputs = method_inputs(op)
        if op == "clip":
            inputs.update({"min": 2.0, "max": 5.0})
        elif op == "repeat":
            inputs["repeats"] = 2
        elif op == "diagonal_offset1":
            inputs["offset"] = 1
        cases.append(case(
            op,
            inputs,
            masked_result_payload(result),
            tolerance_ulps=4,
        ))

    atleast_data = np.array([1.0, 2.0, 3.0], dtype="float64")
    atleast_mask = np.array([False, True, False], dtype="bool")
    atleast_arr = np.ma.array(atleast_data, mask=atleast_mask)
    atleast_arr.set_fill_value(-7.0)

    def atleast_inputs(op, **extra_inputs):
        inputs = {
            "op": op,
            "data": array_to_dict(atleast_data),
            "mask": array_to_dict(atleast_mask, "bool"),
            "fill_value": _serialize_value(atleast_arr.fill_value),
        }
        inputs.update(extra_inputs)
        return inputs

    for op, result in [
        ("atleast_1d", np.ma.atleast_1d(atleast_arr)),
        ("atleast_2d", np.ma.atleast_2d(atleast_arr)),
        ("atleast_3d", np.ma.atleast_3d(atleast_arr)),
    ]:
        cases.append(case(
            op,
            atleast_inputs(op),
            masked_result_payload(result),
            tolerance_ulps=0,
        ))

    cases.append(case(
        "expand_dims",
        method_inputs("expand_dims", axis=1),
        masked_result_payload(np.ma.expand_dims(method_arr, 1)),
        tolerance_ulps=0,
    ))

    dot_rhs_data = method_data + 1.0
    dot_rhs_mask = np.array([[False, False, False],
                             [False, True, False]], dtype="bool")
    dot_rhs = np.ma.array(dot_rhs_data, mask=dot_rhs_mask)
    dot_rhs.set_fill_value(-3.0)
    cases.append(case(
        "ma_dot_flat",
        method_inputs("ma_dot_flat",
                      rhs_data=array_to_dict(dot_rhs_data),
                      rhs_mask=array_to_dict(dot_rhs_mask, "bool"),
                      rhs_fill_value=_serialize_value(dot_rhs.fill_value)),
        {"value": _serialize_value(np.ma.dot(method_arr.ravel(), dot_rhs.ravel()))},
        tolerance_ulps=4,
    ))

    save_fixture("ma", "masked_extras_methods.json",
                 make_fixture("numpy.ma extras numeric and shape methods",
                              "ferray_ma::MaskedArray::{prod,cumsum_flat,cumprod_flat,argmin,argmax,ptp,median,average,anom,clip,repeat,atleast_1d,atleast_2d,atleast_3d,expand_dims,diagonal,ma_dot_flat,trace}",
                              cases))

    # --- Masked extras set and logical helpers ---
    set_data = np.array([1.0, 2.0, 3.0, 4.0], dtype="float64")
    set_mask = np.array([False, True, False, False], dtype="bool")
    set_arr = np.ma.array(set_data, mask=set_mask)
    set_arr.set_fill_value(-7.0)

    bool_lhs_data = np.array([True, False, True, False], dtype="bool")
    bool_lhs_mask = np.array([False, True, False, False], dtype="bool")
    bool_rhs_data = np.array([True, True, False, False], dtype="bool")
    bool_rhs_mask = np.array([False, False, True, False], dtype="bool")
    bool_lhs = np.ma.array(bool_lhs_data, mask=bool_lhs_mask)
    bool_lhs.set_fill_value(True)
    bool_rhs = np.ma.array(bool_rhs_data, mask=bool_rhs_mask)
    bool_rhs.set_fill_value(False)

    cmp_lhs_data = np.array([1.0, 0.0, 3.0, 0.0], dtype="float64")
    cmp_lhs_mask = np.array([False, True, False, True], dtype="bool")
    cmp_rhs_data = np.array([1.0, 9.0, 2.0, 5.0], dtype="float64")
    cmp_rhs_mask = np.array([False, False, True, True], dtype="bool")
    cmp_lhs = np.ma.array(cmp_lhs_data, mask=cmp_lhs_mask)
    cmp_lhs.set_fill_value(-7.0)
    cmp_rhs = np.ma.array(cmp_rhs_data, mask=cmp_rhs_mask)
    cmp_rhs.set_fill_value(-3.0)

    isin_data = np.array([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]], dtype="float64")
    isin_mask = np.array([[False, True, False],
                          [True, False, False]], dtype="bool")
    isin_arr = np.ma.array(isin_data, mask=isin_mask)
    isin_arr.set_fill_value(-7.0)
    membership_values = np.array([2.0, 3.0, 5.0], dtype="float64")

    def masked_bool_payload(result):
        return {
            "data": array_to_dict(np.array(result.data, dtype="bool"), "bool"),
            "mask": array_to_dict(np.ma.getmaskarray(result).astype("bool"), "bool"),
            "fill_value": _serialize_value(result.fill_value),
        }

    def masked_bool_data_mask_payload(result):
        return {
            "data": array_to_dict(np.array(result.data, dtype="bool"), "bool"),
            "mask": array_to_dict(np.ma.getmaskarray(result).astype("bool"), "bool"),
        }

    cases = []
    cases.append(case(
        "ma_unique",
        {
            "op": "ma_unique",
            "data": array_to_dict(set_data),
            "mask": array_to_dict(set_mask, "bool"),
            "fill_value": _serialize_value(set_arr.fill_value),
        },
        masked_result_payload(np.ma.unique(set_arr)),
        tolerance_ulps=0,
    ))

    vander_result = np.ma.vander(set_arr, 3)
    cases.append(case(
        "ma_vander",
        {
            "op": "ma_vander",
            "data": array_to_dict(set_data),
            "mask": array_to_dict(set_mask, "bool"),
            "fill_value": _serialize_value(set_arr.fill_value),
            "n": 3,
        },
        {
            "data": array_to_dict(np.array(vander_result, dtype="float64")),
            "mask": array_to_dict(np.zeros_like(vander_result, dtype="bool"), "bool"),
        },
        tolerance_ulps=0,
    ))

    for op, result in [
        ("ma_equal", np.ma.equal(cmp_lhs, cmp_rhs)),
        ("ma_not_equal", np.ma.not_equal(cmp_lhs, cmp_rhs)),
        ("ma_less", np.ma.less(cmp_lhs, cmp_rhs)),
        ("ma_greater", np.ma.greater(cmp_lhs, cmp_rhs)),
        ("ma_less_equal", np.ma.less_equal(cmp_lhs, cmp_rhs)),
        ("ma_greater_equal", np.ma.greater_equal(cmp_lhs, cmp_rhs)),
    ]:
        cases.append(case(
            op,
            {
                "op": op,
                "lhs_data": array_to_dict(cmp_lhs_data),
                "lhs_mask": array_to_dict(cmp_lhs_mask, "bool"),
                "lhs_fill_value": _serialize_value(cmp_lhs.fill_value),
                "rhs_data": array_to_dict(cmp_rhs_data),
                "rhs_mask": array_to_dict(cmp_rhs_mask, "bool"),
                "rhs_fill_value": _serialize_value(cmp_rhs.fill_value),
            },
            masked_bool_data_mask_payload(result),
            tolerance_ulps=0,
        ))

    cases.append(case(
        "ma_in1d",
        {
            "op": "ma_in1d",
            "data": array_to_dict(set_data),
            "mask": array_to_dict(set_mask, "bool"),
            "fill_value": _serialize_value(set_arr.fill_value),
            "test_values": array_to_dict(membership_values),
        },
        masked_bool_payload(np.ma.in1d(set_arr, membership_values)),
        tolerance_ulps=0,
    ))
    cases.append(case(
        "ma_isin",
        {
            "op": "ma_isin",
            "data": array_to_dict(isin_data),
            "mask": array_to_dict(isin_mask, "bool"),
            "fill_value": _serialize_value(isin_arr.fill_value),
            "test_values": array_to_dict(membership_values),
        },
        masked_bool_payload(np.ma.isin(isin_arr, membership_values)),
        tolerance_ulps=0,
    ))

    def logical_inputs(op):
        return {
            "op": op,
            "lhs_data": array_to_dict(bool_lhs_data, "bool"),
            "lhs_mask": array_to_dict(bool_lhs_mask, "bool"),
            "lhs_fill_value": _serialize_value(bool_lhs.fill_value),
            "rhs_data": array_to_dict(bool_rhs_data, "bool"),
            "rhs_mask": array_to_dict(bool_rhs_mask, "bool"),
            "rhs_fill_value": _serialize_value(bool_rhs.fill_value),
        }

    for op, result in [
        ("ma_logical_and", np.ma.logical_and(bool_lhs, bool_rhs)),
        ("ma_logical_or", np.ma.logical_or(bool_lhs, bool_rhs)),
        ("ma_logical_xor", np.ma.logical_xor(bool_lhs, bool_rhs)),
    ]:
        cases.append(case(
            op,
            logical_inputs(op),
            masked_bool_payload(result),
            tolerance_ulps=0,
        ))
    cases.append(case(
        "ma_logical_not",
        {
            "op": "ma_logical_not",
            "data": array_to_dict(bool_lhs_data, "bool"),
            "mask": array_to_dict(bool_lhs_mask, "bool"),
            "fill_value": _serialize_value(bool_lhs.fill_value),
        },
        masked_bool_payload(np.ma.logical_not(bool_lhs)),
        tolerance_ulps=0,
    ))

    save_fixture("ma", "masked_extras_set_logic.json",
                 make_fixture("numpy.ma extras set, membership, comparison, and logical helpers",
                              "ferray_ma::{ma_unique,ma_vander,ma_in1d,ma_isin,ma_equal,ma_not_equal,ma_less,ma_greater,ma_less_equal,ma_greater_equal,ma_logical_and,ma_logical_or,ma_logical_xor,ma_logical_not}",
                              cases))

    # --- Masked extras functional helpers ---
    func_data = np.array([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]], dtype="float64")
    func_mask = np.array([[False, True, False],
                          [True, False, False]], dtype="bool")
    func_arr = np.ma.array(func_data, mask=func_mask)
    func_arr.set_fill_value(-7.0)
    concat_rhs_data = func_data + 10.0
    concat_rhs_mask = np.logical_not(func_mask)
    concat_rhs = np.ma.array(concat_rhs_data, mask=concat_rhs_mask)
    concat_rhs.set_fill_value(-3.0)

    def func_inputs(op, **extra_inputs):
        inputs = {
            "op": op,
            "data": array_to_dict(func_data),
            "mask": array_to_dict(func_mask, "bool"),
            "fill_value": _serialize_value(func_arr.fill_value),
        }
        inputs.update(extra_inputs)
        return inputs

    cases = []
    for axis in [0, 1]:
        result = np.ma.concatenate([func_arr, concat_rhs], axis=axis)
        cases.append(case(
            f"ma_concatenate_axis{axis}",
            func_inputs("ma_concatenate",
                        axis=axis,
                        rhs_data=array_to_dict(concat_rhs_data),
                        rhs_mask=array_to_dict(concat_rhs_mask, "bool"),
                        rhs_fill_value=_serialize_value(concat_rhs.fill_value)),
            masked_result_payload(result),
            tolerance_ulps=0,
        ))
    for axis in [0, 1]:
        result = np.ma.apply_along_axis(np.ma.sum, axis, func_arr)
        cases.append(case(
            f"ma_apply_along_axis{axis}",
            func_inputs("ma_apply_along_axis", axis=axis),
            masked_result_payload(result),
            tolerance_ulps=4,
        ))
    for axes in [[1], [0, 1]]:
        result = np.ma.apply_over_axes(np.ma.sum, func_arr, axes)
        cases.append(case(
            "ma_apply_over_axes_" + "_".join(str(ax) for ax in axes),
            func_inputs("ma_apply_over_axes", axes=axes),
            masked_result_payload(result),
            tolerance_ulps=4,
        ))

    save_fixture("ma", "masked_extras_functional.json",
                 make_fixture("numpy.ma extras functional helpers",
                              "ferray_ma::{ma_concatenate,ma_apply_along_axis,ma_apply_over_axes}",
                              cases))

    # --- Axis-aware masked reductions + count helpers ---
    data2d = np.arange(12, dtype="float64").reshape(3, 4)
    mask2d = np.array([[False, True, True, False],
                       [False, True, True, False],
                       [True, True, True, False]], dtype="bool")
    ma_2d = np.ma.array(data2d, mask=mask2d)

    cases = []
    for func_name in ["sum", "mean", "min", "max"]:
        for axis in [0, 1]:
            result = getattr(ma_2d, func_name)(axis=axis)
            cases.append(case(
                f"{func_name}_axis{axis}",
                {
                    "op": func_name,
                    "data": array_to_dict(data2d),
                    "mask": array_to_dict(mask2d, "bool"),
                    "axis": axis,
                },
                {
                    "data": array_to_dict(np.array(result.data, dtype="float64")),
                    "mask": array_to_dict(np.ma.getmaskarray(result).astype("bool"), "bool"),
                },
                tolerance_ulps=4,
            ))
    save_fixture("ma", "masked_reductions_axis.json",
                 make_fixture("numpy.ma sum/mean/min/max axis",
                              "ferray_ma::MaskedArray::{sum_axis,mean_axis,min_axis,max_axis}",
                              cases))

    cases = []
    count_total = ma_2d.count()
    count_masked_total = np.ma.count_masked(ma_2d)
    cases.append(case("count_total",
                      {"op": "count",
                       "data": array_to_dict(data2d),
                       "mask": array_to_dict(mask2d, "bool")},
                      array_to_dict(np.array(count_total, dtype="uint64"), "uint64"),
                      tolerance_ulps=0))
    cases.append(case("count_masked_total",
                      {"op": "count_masked",
                       "data": array_to_dict(data2d),
                       "mask": array_to_dict(mask2d, "bool")},
                      array_to_dict(np.array(count_masked_total, dtype="uint64"), "uint64"),
                      tolerance_ulps=0))
    for axis in [0, 1]:
        count_axis = ma_2d.count(axis=axis)
        count_masked_axis = np.ma.count_masked(ma_2d, axis=axis)
        cases.append(case(f"count_axis{axis}",
                          {"op": "count_axis",
                           "data": array_to_dict(data2d),
                           "mask": array_to_dict(mask2d, "bool"),
                           "axis": axis},
                          array_to_dict(np.array(count_axis, dtype="uint64"), "uint64"),
                          tolerance_ulps=0))
        cases.append(case(f"count_masked_axis{axis}",
                          {"op": "count_masked_axis",
                           "data": array_to_dict(data2d),
                           "mask": array_to_dict(mask2d, "bool"),
                           "axis": axis},
                          array_to_dict(np.array(count_masked_axis, dtype="uint64"), "uint64"),
                          tolerance_ulps=0))
    save_fixture("ma", "masked_counts.json",
                 make_fixture("numpy.ma.count/count_masked",
                              "ferray_ma::MaskedArray::count/count_axis and ferray_ma::count_masked/count_masked_axis",
                              cases))

    cases = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for func_name in ["var", "std"]:
            for ddof in [0, 1]:
                result = getattr(ma_2d, func_name)(ddof=ddof)
                cases.append(case(
                    f"{func_name}_ddof{ddof}",
                    {
                        "op": func_name,
                        "data": array_to_dict(data2d),
                        "mask": array_to_dict(mask2d, "bool"),
                        "ddof": ddof,
                    },
                    array_to_dict(np.array(float(result), dtype="float64")),
                    tolerance_ulps=16,
                ))
                for axis in [0, 1]:
                    axis_result = getattr(ma_2d, func_name)(axis=axis, ddof=ddof)
                    cases.append(case(
                        f"{func_name}_axis{axis}_ddof{ddof}",
                        {
                            "op": func_name,
                            "data": array_to_dict(data2d),
                            "mask": array_to_dict(mask2d, "bool"),
                            "axis": axis,
                            "ddof": ddof,
                        },
                        {
                            "data": array_to_dict(np.array(axis_result.data, dtype="float64")),
                            "mask": array_to_dict(np.ma.getmaskarray(axis_result).astype("bool"), "bool"),
                        },
                        tolerance_ulps=16,
                    ))
    save_fixture("ma", "masked_var_std.json",
                 make_fixture("numpy.ma.var/std",
                              "ferray_ma::MaskedArray::{var,var_ddof,std,std_ddof,var_axis,var_axis_ddof,std_axis,std_axis_ddof}",
                              cases))

    # --- Masked sorting ---
    sort1d_data = np.array([3.0, 1.0, 2.0, 9.0], dtype="float64")
    sort1d_mask = np.array([False, True, False, True], dtype="bool")
    sort1d = np.ma.array(sort1d_data, mask=sort1d_mask)
    sort2d_data = np.array([[3.0, 1.0, 2.0],
                            [6.0, 5.0, 4.0]], dtype="float64")
    sort2d_mask = np.array([[False, True, False],
                            [True, False, False]], dtype="bool")
    sort2d = np.ma.array(sort2d_data, mask=sort2d_mask)

    cases = []
    flat_sorted = np.ma.sort(sort1d, axis=None)
    cases.append(case(
        "sort_flat",
        {"op": "sort",
         "data": array_to_dict(sort1d_data),
         "mask": array_to_dict(sort1d_mask, "bool")},
        {"data": array_to_dict(np.array(flat_sorted.data, dtype="float64")),
         "mask": array_to_dict(np.ma.getmaskarray(flat_sorted).astype("bool"), "bool")},
        tolerance_ulps=0,
    ))
    flat_argsort = np.ma.argsort(sort2d, axis=None)
    cases.append(case(
        "argsort_flat_2d",
        {"op": "argsort",
         "data": array_to_dict(sort2d_data),
         "mask": array_to_dict(sort2d_mask, "bool")},
        array_to_dict(np.array(flat_argsort, dtype="uint64"), "uint64"),
        tolerance_ulps=0,
    ))
    for axis in [0, 1]:
        sorted_axis = np.ma.sort(sort2d, axis=axis)
        cases.append(case(
            f"sort_axis{axis}",
            {"op": "sort_axis",
             "data": array_to_dict(sort2d_data),
             "mask": array_to_dict(sort2d_mask, "bool"),
             "axis": axis},
            {"data": array_to_dict(np.array(sorted_axis.data, dtype="float64")),
             "mask": array_to_dict(np.ma.getmaskarray(sorted_axis).astype("bool"), "bool")},
            tolerance_ulps=0,
        ))
    save_fixture("ma", "masked_sorting.json",
                 make_fixture("numpy.ma.sort/argsort",
                              "ferray_ma::MaskedArray::{sort,argsort,sort_axis}",
                              cases))


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
# Stride-tricks fixtures
# ---------------------------------------------------------------------------

def generate_stride_tricks_fixtures() -> None:
    """Generate fixtures/stride_tricks/<name>.json for each ferray-stride-tricks
    public view-constructing function.

    Reference: numpy.lib.stride_tricks (as_strided, sliding_window_view) and
    numpy top-level (broadcast_to, broadcast_arrays, broadcast_shapes).

    All operations are pure view-construction (no float arithmetic, no RNG),
    so outputs are bit-exact and `tolerance_ulps = 0` is the only correct
    value for the fixture-strict tolerance policy.
    """
    from numpy.lib.stride_tricks import as_strided, sliding_window_view

    print("Generating stride_tricks fixtures...")
    subdir = FIXTURES_DIR / "stride_tricks"
    subdir.mkdir(parents=True, exist_ok=True)

    numpy_ver = np.__version__

    def _st_fixture(np_func_name, ferray_func_name, test_cases):
        return {
            "numpy_version": numpy_ver,
            "fixture_schema_version": 2,
            "reference_library": "numpy",
            "function": np_func_name,
            "ferray_function": ferray_func_name,
            "test_cases": test_cases,
        }

    # ------------------------------------------------------------------
    # as_strided (numpy.lib.stride_tricks.as_strided)
    #
    # Inputs encoded:
    #   source   = array_to_dict(source_array, dtype=float64)
    #   shape    = list[int]                — view shape
    #   strides  = list[int]                — view strides in ELEMENT units
    #                                         (we test non-overlapping cases
    #                                         so the safe `as_strided` path
    #                                         accepts them)
    # Expected:
    #   array_to_dict of the materialised numpy view (numpy expects byte
    #   strides; we convert from element strides at fixture-build time).
    # ------------------------------------------------------------------
    def _as_strided_case(name, source, shape, strides):
        elem_size = source.dtype.itemsize
        byte_strides = tuple(s * elem_size for s in strides)
        out = as_strided(source, shape=tuple(shape), strides=byte_strides)
        # .copy() materialises the view into a contiguous buffer so the
        # flattened data is well-defined regardless of overlap.
        out_mat = np.array(out, copy=True)
        return {
            "name": name,
            "inputs": {
                "source": array_to_dict(source.astype(np.float64), "float64"),
                "shape": list(shape),
                "strides": list(strides),
            },
            "expected": array_to_dict(out_mat.astype(np.float64), "float64"),
            "tolerance_ulps": 0,
        }

    as_strided_cases = [
        # 1-D source reshaped to 2x3 with row-major strides (non-overlapping).
        _as_strided_case(
            "reshape_1d_to_2x3",
            np.arange(6, dtype=np.float64),
            shape=[2, 3],
            strides=[3, 1],
        ),
        # 1-D source, take every 2nd element (stride 2).
        _as_strided_case(
            "stride_2_subset",
            np.arange(6, dtype=np.float64),
            shape=[3],
            strides=[2],
        ),
        # Larger source, non-trivial 2-D view with strides [10, 3] (skip).
        _as_strided_case(
            "non_overlap_skip_strides",
            np.arange(20, dtype=np.float64),
            shape=[2, 3],
            strides=[10, 3],
        ),
    ]
    save_fixture("stride_tricks", "as_strided.json",
                 _st_fixture("numpy.lib.stride_tricks.as_strided",
                             "ferray_stride_tricks::as_strided",
                             as_strided_cases))

    # ------------------------------------------------------------------
    # broadcast_to (numpy.broadcast_to)
    # ------------------------------------------------------------------
    def _broadcast_to_case(name, source, target_shape):
        out = np.broadcast_to(source, tuple(target_shape))
        out_mat = np.array(out, copy=True)
        return {
            "name": name,
            "inputs": {
                "source": array_to_dict(source.astype(np.float64), "float64"),
                "target_shape": list(target_shape),
            },
            "expected": array_to_dict(out_mat.astype(np.float64), "float64"),
            "tolerance_ulps": 0,
        }

    broadcast_to_cases = [
        # 1-D (3,) -> 2-D (4, 3) — row broadcast.
        _broadcast_to_case(
            "row_to_2d",
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            target_shape=[4, 3],
        ),
        # 2-D (1, 3) -> (4, 3) — explicit unit-axis broadcast.
        _broadcast_to_case(
            "unit_axis_to_2d",
            np.array([[10.0, 20.0, 30.0]], dtype=np.float64),
            target_shape=[4, 3],
        ),
        # 2-D (4, 1) -> (4, 3) — column broadcast.
        _broadcast_to_case(
            "column_to_2d",
            np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float64),
            target_shape=[4, 3],
        ),
    ]
    save_fixture("stride_tricks", "broadcast_to.json",
                 _st_fixture("numpy.broadcast_to",
                             "ferray_stride_tricks::broadcast_to",
                             broadcast_to_cases))

    # ------------------------------------------------------------------
    # broadcast_arrays (numpy.broadcast_arrays)
    #
    # Reference takes N arrays and returns N broadcast views of the common
    # shape. Fixture encodes each input as a separate slot ("a", "b", ...)
    # and the expected output as a list of array_to_dict materialised views.
    # ------------------------------------------------------------------
    def _broadcast_arrays_case(name, arrays):
        outs = np.broadcast_arrays(*arrays)
        return {
            "name": name,
            "inputs": {
                f"arr_{i}": array_to_dict(a.astype(np.float64), "float64")
                for i, a in enumerate(arrays)
            },
            "expected": {
                "arrays": [
                    array_to_dict(np.array(o, copy=True).astype(np.float64), "float64")
                    for o in outs
                ],
            },
            "tolerance_ulps": 0,
        }

    broadcast_arrays_cases = [
        # (4, 1) + (1, 3) -> two views of (4, 3).
        _broadcast_arrays_case(
            "column_and_row",
            [
                np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float64),
                np.array([[10.0, 20.0, 30.0]], dtype=np.float64),
            ],
        ),
        # (3,) + (4, 3) -> two views of (4, 3) — 1-D row broadcast against 2-D.
        _broadcast_arrays_case(
            "row_and_matrix",
            [
                np.array([1.0, 2.0, 3.0], dtype=np.float64),
                np.arange(12, dtype=np.float64).reshape(4, 3),
            ],
        ),
        # Three-way: (2, 1) + (3,) + (1, 3) -> common shape (2, 3).
        _broadcast_arrays_case(
            "three_arrays",
            [
                np.array([[1.0], [2.0]], dtype=np.float64),
                np.array([10.0, 20.0, 30.0], dtype=np.float64),
                np.array([[100.0, 200.0, 300.0]], dtype=np.float64),
            ],
        ),
    ]
    save_fixture("stride_tricks", "broadcast_arrays.json",
                 _st_fixture("numpy.broadcast_arrays",
                             "ferray_stride_tricks::broadcast_arrays",
                             broadcast_arrays_cases))

    # ------------------------------------------------------------------
    # broadcast_shapes (numpy.broadcast_shapes)
    #
    # Pure shape arithmetic — no array data involved. Inputs encode shapes
    # as `[[usize], [usize], ...]`. Expected output is the result shape as
    # a `[usize]` list.
    # ------------------------------------------------------------------
    def _broadcast_shapes_case(name, shapes):
        result = list(np.broadcast_shapes(*[tuple(s) for s in shapes]))
        return {
            "name": name,
            "inputs": {"shapes": [list(s) for s in shapes]},
            "expected": {"shape": result},
            "tolerance_ulps": 0,
        }

    broadcast_shapes_cases = [
        _broadcast_shapes_case("two_2d_unit_axes", [[3, 1], [1, 4]]),
        _broadcast_shapes_case("three_mixed_ndim", [[2, 1], [3], [1, 3]]),
        _broadcast_shapes_case("scalar_with_2d", [[], [4, 5]]),
    ]
    save_fixture("stride_tricks", "broadcast_shapes.json",
                 _st_fixture("numpy.broadcast_shapes",
                             "ferray_stride_tricks::broadcast_shapes",
                             broadcast_shapes_cases))

    # ------------------------------------------------------------------
    # sliding_window_view (numpy.lib.stride_tricks.sliding_window_view)
    # ------------------------------------------------------------------
    def _sliding_window_case(name, source, window_shape):
        out = sliding_window_view(source, window_shape=tuple(window_shape))
        out_mat = np.array(out, copy=True)
        return {
            "name": name,
            "inputs": {
                "source": array_to_dict(source.astype(np.float64), "float64"),
                "window_shape": list(window_shape),
            },
            "expected": array_to_dict(out_mat.astype(np.float64), "float64"),
            "tolerance_ulps": 0,
        }

    sliding_window_cases = [
        # 1-D source length 5, window 3 -> output shape (3, 3).
        _sliding_window_case(
            "1d_window3",
            np.arange(1.0, 6.0, dtype=np.float64),
            window_shape=[3],
        ),
        # 1-D source length 6, window 2 -> output shape (5, 2).
        _sliding_window_case(
            "1d_window2",
            np.arange(1.0, 7.0, dtype=np.float64),
            window_shape=[2],
        ),
        # 2-D source (3, 4), window (2, 2) -> output shape (2, 3, 2, 2).
        _sliding_window_case(
            "2d_window_2x2",
            np.arange(1.0, 13.0, dtype=np.float64).reshape(3, 4),
            window_shape=[2, 2],
        ),
    ]
    save_fixture("stride_tricks", "sliding_window_view.json",
                 _st_fixture("numpy.lib.stride_tricks.sliding_window_view",
                             "ferray_stride_tricks::sliding_window_view",
                             sliding_window_cases))

    print(f"  Stride-tricks fixtures complete — "
          f"{len(list(subdir.glob('*.json')))} files.")


# ---------------------------------------------------------------------------
# Autodiff
# ---------------------------------------------------------------------------

def generate_autodiff_fixtures() -> None:
    """Generate fixtures/autodiff/<name>.json for each ferray-autodiff public
    function (forward-mode dual-number AD).

    The reference values are analytic derivatives computed by hand and
    encoded as NumPy expressions. Each fixture tests value+derivative
    (or gradient / jacobian) for a representative simple function. No
    autograd backward lane: ferray-autodiff is forward-mode, so coverage
    is over eval(f, x) and grad(f, x) only.
    """
    print("Generating autodiff fixtures...")
    subdir = FIXTURES_DIR / "autodiff"
    subdir.mkdir(parents=True, exist_ok=True)

    numpy_ver = np.__version__

    def _ad_fixture(np_func_name, ferray_func_name, test_cases, ref_lib="numpy"):
        return {
            "numpy_version": numpy_ver,
            "fixture_schema_version": 2,
            "reference_library": ref_lib,
            "function": np_func_name,
            "ferray_function": ferray_func_name,
            "test_cases": test_cases,
        }

    # ------------------------------------------------------------------
    # api::derivative — d/dx f(x) for a univariate f.
    # Input: { x: scalar, fn_id: str (selects the closure on the Rust side) }
    # Expected: { value: f(x), deriv: f'(x) }
    # ------------------------------------------------------------------
    derivative_cases = [
        # f(x) = x^2 at x=3 → value=9, deriv=6
        {
            "name": "x_squared_at_3",
            "inputs": {"x": 3.0, "fn_id": "x_squared"},
            "expected": {"value": 9.0, "deriv": 6.0},
            "tolerance_ulps": 4,
        },
        # f(x) = sin(x) at x=π/4 → value=sin(π/4), deriv=cos(π/4)
        {
            "name": "sin_at_pi_over_4",
            "inputs": {"x": float(np.pi / 4.0), "fn_id": "sin"},
            "expected": {
                "value": float(np.sin(np.pi / 4.0)),
                "deriv": float(np.cos(np.pi / 4.0)),
            },
            "tolerance_ulps": 4,
        },
        # f(x) = exp(x) at x=1 → value=e, deriv=e
        {
            "name": "exp_at_1",
            "inputs": {"x": 1.0, "fn_id": "exp"},
            "expected": {
                "value": float(np.exp(1.0)),
                "deriv": float(np.exp(1.0)),
            },
            "tolerance_ulps": 4,
        },
    ]
    save_fixture("autodiff", "derivative.json",
                 _ad_fixture("analytic.d_dx",
                             "ferray_autodiff::api::derivative",
                             derivative_cases))

    # ------------------------------------------------------------------
    # api::gradient — ∇f(x) for f: R^n → R.
    # Input: { point: [f64], fn_id: str }
    # Expected: { gradient: [f64] }
    # ------------------------------------------------------------------
    gradient_cases = [
        # f(x,y) = x^2 + y^2 at (3,4) → ∇f = (2x, 2y) = (6, 8)
        {
            "name": "sum_of_squares_at_3_4",
            "inputs": {"point": [3.0, 4.0], "fn_id": "sum_of_squares"},
            "expected": {"gradient": [6.0, 8.0]},
            "tolerance_ulps": 4,
        },
        # f(x,y,z) = x*y*z at (1,2,3) → ∇f = (yz, xz, xy) = (6, 3, 2)
        {
            "name": "product_of_three_at_1_2_3",
            "inputs": {"point": [1.0, 2.0, 3.0], "fn_id": "product_of_three"},
            "expected": {"gradient": [6.0, 3.0, 2.0]},
            "tolerance_ulps": 4,
        },
    ]
    save_fixture("autodiff", "gradient.json",
                 _ad_fixture("analytic.grad",
                             "ferray_autodiff::api::gradient",
                             gradient_cases))

    # ------------------------------------------------------------------
    # api::jacobian — J of f: R^n → R^m.
    # Input: { point: [f64], fn_id: str }
    # Expected: { jacobian: [f64], m: usize, n: usize }
    # ------------------------------------------------------------------
    jacobian_cases = [
        # f(x,y) = (x+y, x-y) at (3,4) → J = [[1,1],[1,-1]]
        {
            "name": "linear_pair_at_3_4",
            "inputs": {"point": [3.0, 4.0], "fn_id": "linear_pair"},
            "expected": {"jacobian": [1.0, 1.0, 1.0, -1.0], "m": 2, "n": 2},
            "tolerance_ulps": 4,
        },
        # f(x,y) = (x*y, x+y) at (3,4) → J = [[y,x],[1,1]] = [[4,3],[1,1]]
        {
            "name": "product_and_sum_at_3_4",
            "inputs": {"point": [3.0, 4.0], "fn_id": "product_and_sum"},
            "expected": {"jacobian": [4.0, 3.0, 1.0, 1.0], "m": 2, "n": 2},
            "tolerance_ulps": 4,
        },
    ]
    save_fixture("autodiff", "jacobian.json",
                 _ad_fixture("analytic.jacobian",
                             "ferray_autodiff::api::jacobian",
                             jacobian_cases))

    # ------------------------------------------------------------------
    # array_ops::derivative_elementwise — d/dx f(x) over an array.
    # Input: { x: {data, shape}, fn_id: "x_squared" }
    # Expected: { data, shape } of f'(x_i)
    # ------------------------------------------------------------------
    derivative_ew_cases = [
        # f(x) = x^2 over [0,1,2,3] → f' = [0,2,4,6]
        {
            "name": "x_squared_over_1d",
            "inputs": {
                "x": array_to_dict(np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64), "float64"),
                "fn_id": "x_squared",
            },
            "expected": array_to_dict(
                np.array([0.0, 2.0, 4.0, 6.0], dtype=np.float64), "float64"
            ),
            "tolerance_ulps": 4,
        },
    ]
    save_fixture("autodiff", "derivative_elementwise.json",
                 _ad_fixture("analytic.d_dx_elementwise",
                             "ferray_autodiff::array_ops::derivative_elementwise",
                             derivative_ew_cases))

    # ------------------------------------------------------------------
    # array_ops::value_and_derivative_elementwise — (f(x), f'(x))
    # Input: { x: {data, shape}, fn_id }
    # Expected: { values: {data, shape}, derivs: {data, shape} }
    # ------------------------------------------------------------------
    val_and_deriv_cases = [
        # f(x) = x^2 over [1,2,3] → values=[1,4,9], derivs=[2,4,6]
        {
            "name": "x_squared_over_1d",
            "inputs": {
                "x": array_to_dict(np.array([1.0, 2.0, 3.0], dtype=np.float64), "float64"),
                "fn_id": "x_squared",
            },
            "expected": {
                "values": array_to_dict(
                    np.array([1.0, 4.0, 9.0], dtype=np.float64), "float64"
                ),
                "derivs": array_to_dict(
                    np.array([2.0, 4.0, 6.0], dtype=np.float64), "float64"
                ),
            },
            "tolerance_ulps": 4,
        },
    ]
    save_fixture("autodiff", "value_and_derivative_elementwise.json",
                 _ad_fixture("analytic.value_and_d_dx",
                             "ferray_autodiff::array_ops::value_and_derivative_elementwise",
                             val_and_deriv_cases))

    # ------------------------------------------------------------------
    # array_ops::gradient_vector — ∇f as a ferray Array
    # Same input/expected shape as api::gradient but using ferray arrays.
    # ------------------------------------------------------------------
    gradient_vector_cases = [
        # f(x) = x[0]^2 + 2*x[1]^2 + 3*x[2]^2 at [1,2,3]
        # ∇f = [2*x[0], 4*x[1], 6*x[2]] = [2,8,18]
        {
            "name": "weighted_squares_at_1_2_3",
            "inputs": {
                "point": array_to_dict(
                    np.array([1.0, 2.0, 3.0], dtype=np.float64), "float64"
                ),
                "fn_id": "weighted_squares",
            },
            "expected": array_to_dict(
                np.array([2.0, 8.0, 18.0], dtype=np.float64), "float64"
            ),
            "tolerance_ulps": 4,
        },
    ]
    save_fixture("autodiff", "gradient_vector.json",
                 _ad_fixture("analytic.grad_vector",
                             "ferray_autodiff::array_ops::gradient_vector",
                             gradient_vector_cases))

    # ------------------------------------------------------------------
    # array_ops::value_and_gradient — (f(x), ∇f(x)) in one pass
    # ------------------------------------------------------------------
    val_and_grad_cases = [
        # f(x) = x[0] + x[1]*x[2] at [1,2,3] → value=7, ∇f=[1,3,2]
        {
            "name": "linear_plus_product_at_1_2_3",
            "inputs": {
                "point": array_to_dict(
                    np.array([1.0, 2.0, 3.0], dtype=np.float64), "float64"
                ),
                "fn_id": "linear_plus_product",
            },
            "expected": {
                "value": 7.0,
                "gradient": array_to_dict(
                    np.array([1.0, 3.0, 2.0], dtype=np.float64), "float64"
                ),
            },
            "tolerance_ulps": 4,
        },
    ]
    save_fixture("autodiff", "value_and_gradient.json",
                 _ad_fixture("analytic.value_and_grad",
                             "ferray_autodiff::array_ops::value_and_gradient",
                             val_and_grad_cases))

    # ------------------------------------------------------------------
    # array_ops::jacobian_array — J as an (m,n) ferray Array
    # ------------------------------------------------------------------
    jacobian_array_cases = [
        # f(x,y) = (x+y, x-y) at (3,4) → J = [[1,1],[1,-1]]
        {
            "name": "linear_pair_at_3_4",
            "inputs": {
                "point": array_to_dict(
                    np.array([3.0, 4.0], dtype=np.float64), "float64"
                ),
                "fn_id": "linear_pair",
            },
            "expected": array_to_dict(
                np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float64), "float64"
            ),
            "tolerance_ulps": 4,
        },
    ]
    save_fixture("autodiff", "jacobian_array.json",
                 _ad_fixture("analytic.jacobian_array",
                             "ferray_autodiff::array_ops::jacobian_array",
                             jacobian_array_cases))

    # ------------------------------------------------------------------
    # functions::atan2 — free function (y, x) → atan2(y, x) with the
    # joint partials (∂/∂y, ∂/∂x) seeded so the dual value of the
    # output equals (∂atan2/∂y) * y_dual + (∂atan2/∂x) * x_dual.
    #
    # The fixture exercises the canonical case y_dual=1, x_dual=0 so
    # the output dual part is the partial w.r.t. y:
    #     ∂atan2(y, x)/∂y =  x / (x^2 + y^2)
    # ------------------------------------------------------------------
    atan2_cases = [
        # (y=1, x=1) → atan2(1,1)=π/4 ; ∂/∂y = 1/(1+1) = 0.5
        {
            "name": "y1_x1_partial_y",
            "inputs": {"y": 1.0, "x": 1.0, "y_dual": 1.0, "x_dual": 0.0},
            "expected": {
                "value": float(np.arctan2(1.0, 1.0)),
                "deriv": 0.5,
            },
            "tolerance_ulps": 4,
        },
        # (y=1, x=1) → ∂/∂x = -y/(x^2+y^2) = -0.5
        {
            "name": "y1_x1_partial_x",
            "inputs": {"y": 1.0, "x": 1.0, "y_dual": 0.0, "x_dual": 1.0},
            "expected": {
                "value": float(np.arctan2(1.0, 1.0)),
                "deriv": -0.5,
            },
            "tolerance_ulps": 4,
        },
    ]
    save_fixture("autodiff", "atan2.json",
                 _ad_fixture("numpy.arctan2",
                             "ferray_autodiff::functions::atan2",
                             atan2_cases))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"NumPy version: {np.__version__}")
    print(f"Output directory: {FIXTURES_DIR}")
    print()

    # Ensure directories exist
    for subdir in ["core", "ufunc", "stats", "linalg", "fft", "random", "io", "polynomial", "strings", "ma", "window", "stride_tricks", "autodiff"]:
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
        ("stride_tricks", generate_stride_tricks_fixtures),
        ("autodiff", generate_autodiff_fixtures),
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
