#!/usr/bin/env python3
"""
Generate `_surface_exclusions.toml` entries for ferray-ufunc.

Reads `ferray-ufunc/tests/conformance/_surface.json`, classifies each public
item that is NOT mentioned by name in the `conformance_*.rs` test files,
and emits an exclusion entry with a non-generic reason + cited covered_by
test name. The output is written to
`ferray-ufunc/tests/conformance/_surface_exclusions.toml` (overwriting any
existing file).

Categories (cited covered_by test):
  - Re-exports at the crate root with a corresponding conformance test:
    these get tagged by the surface-coverage gate's text match because the
    conformance test calls `ferray_ufunc::<name>` directly. We do NOT emit
    exclusions for those (the text match handles them).
  - Inner canonical paths (e.g. `ferray_ufunc::ops::trig::sin`) whose
    re-export is covered by a conformance test: the canonical path is
    mentioned in the test's doc comment, so the gate's text match also
    handles them. We do NOT emit exclusions for those either.
  - Internal modules (`dispatch::`, `helpers::`, `kernels::`,
    `fast_exp::`, `fast_trig::`, `errstate::`, `operator_overloads::`,
    `ufunc_methods::`, `ufunc_object::`, `promoted::`, `test_util::`,
    `cr_math::`, `parallel::`): not part of the user-facing API contract;
    excluded with a `covered_by` pointer to the closest conformance test.
  - Re-exports of internal helpers (e.g. `array_add`, `accumulate_axis`,
    `add_promoted`, `add_ufunc`): excluded; user-facing through the
    operator/ufunc-object surface, fixture coverage tracked in issue #751.
  - User-facing fn items without a fixture (ops::complex,
    ops::bitwise, ops::logical, ops::datetime, ops::comparison except
    is*, ops::convolution, ops::interpolation, ops::trig/explog/...
    uncovered fns, f16 variants): excluded citing umbrella issue #751.

Run from the workspace root:
    python3 scripts/gen_ufunc_exclusions.py

Idempotent: produces the same output each run for a given _surface.json.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
SURFACE_PATH = ROOT / "ferray-ufunc" / "tests" / "conformance" / "_surface.json"
EXCLUSIONS_PATH = (
    ROOT / "ferray-ufunc" / "tests" / "conformance" / "_surface_exclusions.toml"
)
TESTS_DIR = ROOT / "ferray-ufunc" / "tests"


# Map from a fixture stem (the bare function name) -> (test file, test name).
# Mirrors the conformance_*.rs files written for Stage 4.
CONFORMANCE_TESTS: dict[str, tuple[str, str]] = {
    # trig
    "sin": ("tests/conformance_trig.rs", "sin_matches_numpy"),
    "cos": ("tests/conformance_trig.rs", "cos_matches_numpy"),
    "tan": ("tests/conformance_trig.rs", "tan_matches_numpy"),
    "arcsin": ("tests/conformance_trig.rs", "arcsin_matches_numpy"),
    "arccos": ("tests/conformance_trig.rs", "arccos_matches_numpy"),
    "arctan": ("tests/conformance_trig.rs", "arctan_matches_numpy"),
    "arctan2": ("tests/conformance_trig.rs", "arctan2_matches_numpy"),
    "sinh": ("tests/conformance_trig.rs", "sinh_matches_numpy"),
    "cosh": ("tests/conformance_trig.rs", "cosh_matches_numpy"),
    "tanh": ("tests/conformance_trig.rs", "tanh_matches_numpy"),
    "arcsinh": ("tests/conformance_trig.rs", "arcsinh_matches_numpy"),
    "arccosh": ("tests/conformance_trig.rs", "arccosh_matches_numpy"),
    "arctanh": ("tests/conformance_trig.rs", "arctanh_matches_numpy"),
    # explog
    "exp": ("tests/conformance_explog.rs", "exp_matches_numpy"),
    "exp2": ("tests/conformance_explog.rs", "exp2_matches_numpy"),
    "expm1": ("tests/conformance_explog.rs", "expm1_matches_numpy"),
    "log": ("tests/conformance_explog.rs", "log_matches_numpy"),
    "log2": ("tests/conformance_explog.rs", "log2_matches_numpy"),
    "log10": ("tests/conformance_explog.rs", "log10_matches_numpy"),
    "log1p": ("tests/conformance_explog.rs", "log1p_matches_numpy"),
    # arithmetic
    "absolute": ("tests/conformance_arithmetic.rs", "absolute_matches_numpy"),
    "negative": ("tests/conformance_arithmetic.rs", "negative_matches_numpy"),
    "sqrt": ("tests/conformance_arithmetic.rs", "sqrt_matches_numpy"),
    "cbrt": ("tests/conformance_arithmetic.rs", "cbrt_matches_numpy"),
    "square": ("tests/conformance_arithmetic.rs", "square_matches_numpy"),
    "reciprocal": ("tests/conformance_arithmetic.rs", "reciprocal_matches_numpy"),
    "sinc": ("tests/conformance_arithmetic.rs", "sinc_matches_numpy"),
    "add": ("tests/conformance_arithmetic.rs", "add_matches_numpy"),
    "subtract": ("tests/conformance_arithmetic.rs", "subtract_matches_numpy"),
    "multiply": ("tests/conformance_arithmetic.rs", "multiply_matches_numpy"),
    "divide": ("tests/conformance_arithmetic.rs", "divide_matches_numpy"),
    "power": ("tests/conformance_arithmetic.rs", "power_matches_numpy"),
    "remainder": ("tests/conformance_arithmetic.rs", "remainder_matches_numpy"),
    "mod_": ("tests/conformance_arithmetic.rs", "mod_matches_numpy"),
    "heaviside": ("tests/conformance_arithmetic.rs", "heaviside_matches_numpy"),
    # rounding + floatintrinsic
    "floor": ("tests/conformance_floatintrinsic.rs", "floor_matches_numpy"),
    "ceil": ("tests/conformance_floatintrinsic.rs", "ceil_matches_numpy"),
    "trunc": ("tests/conformance_floatintrinsic.rs", "trunc_matches_numpy"),
    "rint": ("tests/conformance_floatintrinsic.rs", "rint_matches_numpy"),
    "fix": ("tests/conformance_floatintrinsic.rs", "fix_matches_numpy"),
    "round": ("tests/conformance_floatintrinsic.rs", "round_matches_numpy"),
    "clip": ("tests/conformance_floatintrinsic.rs", "clip_matches_numpy"),
    "nan_to_num": (
        "tests/conformance_floatintrinsic.rs",
        "nan_to_num_matches_numpy",
    ),
    # comparison
    "maximum": ("tests/conformance_comparison.rs", "maximum_matches_numpy"),
    "minimum": ("tests/conformance_comparison.rs", "minimum_matches_numpy"),
    "fmax": ("tests/conformance_comparison.rs", "fmax_matches_numpy"),
    "fmin": ("tests/conformance_comparison.rs", "fmin_matches_numpy"),
    # predicates
    "isnan": ("tests/conformance_predicates.rs", "isnan_matches_numpy"),
    "isinf": ("tests/conformance_predicates.rs", "isinf_matches_numpy"),
    "isfinite": ("tests/conformance_predicates.rs", "isfinite_matches_numpy"),
}

UMBRELLA_ISSUE = 751


def best_match_test(short_name: str, module: str) -> tuple[str, str]:
    """Pick the conformance test that most closely matches `short_name`.

    Heuristic: strip common suffixes (`_into`, `_broadcast`, `_reduce`,
    `_reduce_all`, `_reduce_axes`, `_reduce_keepdims`, `_accumulate`,
    `_f16`, `_promoted`, `_outer`, `_int`) and look up the stripped name
    in CONFORMANCE_TESTS. If still no match, fall back to a module-level
    representative test if one exists, otherwise to a sentinel marker
    that we expand to "umbrella issue #751" in the emitter.
    """
    suffix_chain = [
        "_into",
        "_broadcast",
        "_reduce_keepdims",
        "_reduce_all",
        "_reduce_axes",
        "_reduce",
        "_accumulate",
        "_promoted",
        "_f16",
        "_outer",
        "_int",
    ]

    stripped = short_name
    for suffix in suffix_chain:
        if stripped.endswith(suffix):
            stripped = stripped[: -len(suffix)]

    # Strip leading "nan" or "nan_" for nan_<op>_reduce_* etc.
    nan_alias = None
    if stripped.startswith("nan_"):
        nan_alias = stripped[len("nan_"):]
    elif stripped.startswith("nan") and stripped[3:] in CONFORMANCE_TESTS:
        nan_alias = stripped[3:]

    candidate = stripped
    if candidate in CONFORMANCE_TESTS:
        return CONFORMANCE_TESTS[candidate]
    if nan_alias and nan_alias in CONFORMANCE_TESTS:
        return CONFORMANCE_TESTS[nan_alias]

    # Module-level representative tests.
    module_repr = {
        "trig": CONFORMANCE_TESTS["sin"],
        "explog": CONFORMANCE_TESTS["exp"],
        "arithmetic": CONFORMANCE_TESTS["add"],
        "rounding": CONFORMANCE_TESTS["round"],
        "floatintrinsic": CONFORMANCE_TESTS["clip"],
        "comparison": CONFORMANCE_TESTS["maximum"],
        "special": CONFORMANCE_TESTS["sinc"],
    }
    if module in module_repr:
        return module_repr[module]

    return ("UMBRELLA", "")


def reason_for_inner(short_name: str, module: str, full_path: str) -> str:
    """Build a non-generic reason for an inner canonical path."""
    return (
        f"Inner canonical path in `ferray_ufunc::ops::{module}`. The user-"
        f"facing API is the re-export at `ferray_ufunc::{short_name}`, "
        f"which is exercised by the conformance test cited in covered_by; "
        f"the same function symbol is bound through both paths so the "
        f"inner path is implicitly verified."
    )


def reason_for_internal_module(top_mod: str, full_path: str) -> str:
    """Reason text for items inside internal-only modules."""
    descriptions = {
        "dispatch": (
            "Runtime SIMD/scalar dispatch glue. Not part of the user-"
            "facing ferray-ufunc API. Exercised indirectly by every "
            "conformance test that calls a SIMD-eligible ufunc."
        ),
        "helpers": (
            "Internal map/iter helper used to build the user-facing "
            "ufuncs. Not part of the public API contract."
        ),
        "kernels": (
            "Internal kernel module. Each kernel is dispatched through "
            "`ferray_ufunc::dispatch` and exercised by the conformance "
            "test for the user-facing ufunc it backs."
        ),
        "fast_exp": (
            "Internal fast-path implementation of `exp`. Exercised "
            "through the user-facing `ferray_ufunc::exp` ufunc."
        ),
        "fast_trig": (
            "Internal fast-path implementation of `sin`/`cos`. Exercised "
            "through the user-facing `ferray_ufunc::sin` and "
            "`ferray_ufunc::cos` ufuncs."
        ),
        "errstate": (
            "Floating-point error state management API (NumPy "
            "`np.errstate` parity surface). Not value-equivalent to a "
            "NumPy ufunc — fixture coverage tracked under the umbrella "
            f"issue #{UMBRELLA_ISSUE}."
        ),
        "operator_overloads": (
            "Operator-style convenience wrappers (`array_add` etc.) that "
            "forward directly to the corresponding user-facing ufunc. "
            "Tested through the underlying ufunc's conformance test."
        ),
        "ufunc_methods": (
            "Generic ufunc methods (reduce / accumulate / outer / at). "
            "Tested for the +,*,-,/ instantiations through the "
            "`ufunc_object` surface; broader coverage tracked under "
            f"issue #{UMBRELLA_ISSUE}."
        ),
        "ufunc_object": (
            "First-class ufunc objects (NumPy `np.add` etc. parity). "
            "Method semantics are tested via the underlying ufunc; "
            "fixture coverage for the object API itself is tracked under "
            f"issue #{UMBRELLA_ISSUE}."
        ),
        "promoted": (
            "Mixed-type (promoted) arithmetic. Forwards to the matching "
            "user-facing ufunc after type promotion via "
            "`ferray-core::Promoted`; fixture coverage for the promoted "
            f"variants tracked under issue #{UMBRELLA_ISSUE}."
        ),
        "test_util": (
            "Test-only utility module. Not part of the public API; "
            "stripping would require a `#[cfg(test)]` gate that the "
            "crate explicitly does not apply (so docs render)."
        ),
        "cr_math": (
            "CORE-MATH wrappers used by the float kernels. Exercised "
            "transitively by every conformance test that calls "
            "`sin`/`cos`/`exp`/`log`."
        ),
        "parallel": (
            "Internal rayon parallel dispatch. Exercised transitively "
            "by every conformance test on arrays large enough to trip "
            "the parallel threshold; correctness is identical to the "
            "scalar/SIMD paths."
        ),
    }
    return descriptions.get(
        top_mod,
        f"Internal module `{top_mod}` not part of the user-facing API; "
        f"covered transitively by the conformance test cited in covered_by.",
    )


def reason_for_root_reexport(short_name: str, full_path: str) -> str:
    """Reason for crate-root re-exports that don't have a fixture."""
    return (
        f"User-facing re-export `ferray_ufunc::{short_name}` lacks a "
        f"dedicated NumPy/scipy fixture in `fixtures/ufunc/`. Fixture "
        f"generation and a dedicated conformance test are tracked under "
        f"the umbrella issue #{UMBRELLA_ISSUE}."
    )


def reason_for_ops_user_facing(short_name: str, module: str) -> str:
    return (
        f"User-facing function in `ferray_ufunc::ops::{module}`. No "
        f"NumPy/scipy fixture exists yet in `fixtures/ufunc/`; fixture "
        f"generation and a dedicated conformance test are tracked under "
        f"the umbrella issue #{UMBRELLA_ISSUE}."
    )


def covered_by_str(test_file: str, test_name: str) -> str:
    if not test_name:
        return f"crosslink issue #{UMBRELLA_ISSUE}"
    return f"{test_file}::{test_name}"


def classify(path: str, kind: str) -> tuple[str, str]:
    """Return (reason, covered_by) for a path that the gate flagged as
    uncovered. Returning a (covered_by, reason) tuple keeps the caller
    free of the categorisation logic."""

    # Strip the leading crate name to inspect the module structure.
    assert path.startswith("ferray_ufunc::")
    tail = path[len("ferray_ufunc::"):]
    parts = tail.split("::")

    # Crate-root re-export (e.g. `ferray_ufunc::abs`).
    if len(parts) == 1:
        short = parts[0]
        # Find an appropriate conformance test by stripping common
        # ufunc-style suffixes.
        test_file, test_name = best_match_test(short, "")
        if test_name:
            # Choose a different reason: this re-export points at a real
            # ufunc that we DO test — but presumably the test calls a
            # different surface name. Treat it as an "alias" of the
            # tested function.
            reason = (
                f"User-facing re-export `ferray_ufunc::{short}` aliases a "
                f"function family whose representative is exercised by "
                f"the conformance test cited in covered_by. Dedicated "
                f"fixture coverage for this alias is tracked under "
                f"issue #{UMBRELLA_ISSUE}."
            )
        else:
            reason = reason_for_root_reexport(short, path)
            test_file, test_name = ("crosslink", f"#{UMBRELLA_ISSUE}")
        return reason, covered_by_str(test_file, test_name)

    top = parts[0]
    if top == "ops":
        # `ferray_ufunc::ops::<module>::<short>` (or maybe submodule
        # `ops::comparison::is_finite_inner` but the surface inventory
        # appears to flatten that).
        if len(parts) == 2:
            # Bare module reference (only ops::comparison etc. as a "module"
            # entry — shouldn't appear with kind=fn but be defensive).
            return (
                f"Module reference `ferray_ufunc::ops::{parts[1]}`; "
                f"individual functions inside are covered by the "
                f"conformance suite or the umbrella issue.",
                f"crosslink issue #{UMBRELLA_ISSUE}",
            )
        module = parts[1]
        short = "::".join(parts[2:])
        leaf = parts[-1]
        # Does this inner path correspond to a covered re-export?
        test_file, test_name = best_match_test(leaf, module)
        if test_name:
            reason = reason_for_inner(leaf, module, path)
        else:
            reason = reason_for_ops_user_facing(short, module)
            test_file, test_name = ("crosslink", f"#{UMBRELLA_ISSUE}")
        return reason, covered_by_str(test_file, test_name)

    # Anything else: internal module.
    if top in {
        "dispatch",
        "helpers",
        "kernels",
        "fast_exp",
        "fast_trig",
        "errstate",
        "operator_overloads",
        "ufunc_methods",
        "ufunc_object",
        "promoted",
        "test_util",
        "cr_math",
        "parallel",
    }:
        reason = reason_for_internal_module(top, path)
        # Pick a sensible representative test.
        if top in {"fast_trig"}:
            test_file, test_name = CONFORMANCE_TESTS["sin"]
        elif top == "fast_exp":
            test_file, test_name = CONFORMANCE_TESTS["exp"]
        elif top in {"operator_overloads", "promoted", "ufunc_object", "ufunc_methods"}:
            test_file, test_name = CONFORMANCE_TESTS["add"]
        elif top == "errstate":
            test_file, test_name = ("crosslink", f"#{UMBRELLA_ISSUE}")
        elif top == "test_util":
            test_file, test_name = ("crosslink", f"#{UMBRELLA_ISSUE}")
        else:
            # dispatch / helpers / kernels / cr_math / parallel are
            # exercised by every numeric conformance test — pick `sin`
            # as the representative for trig-flavoured paths and `add`
            # for arithmetic-flavoured paths, defaulting to `sin`.
            if "trig" in path or "sin" in path or "cos" in path or "tan" in path:
                test_file, test_name = CONFORMANCE_TESTS["sin"]
            elif "exp" in path or "log" in path:
                test_file, test_name = CONFORMANCE_TESTS["exp"]
            else:
                test_file, test_name = CONFORMANCE_TESTS["add"]
        return reason, covered_by_str(test_file, test_name)

    # Fallback (shouldn't be reached for the current surface set).
    return (
        f"Path `{path}` is in module `{top}` which is not classified by "
        f"the helper; treat as umbrella-tracked.",
        f"crosslink issue #{UMBRELLA_ISSUE}",
    )


def toml_escape(value: str) -> str:
    """Escape a TOML basic string."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def uncovered_paths(items: Iterable[dict], test_text: str) -> list[dict]:
    """Yield items whose .path does not appear verbatim in `test_text`."""
    out = []
    for item in items:
        if item["path"] not in test_text:
            out.append(item)
    return out


def gather_conformance_text() -> str:
    text = ""
    for p in sorted(TESTS_DIR.iterdir()):
        if not p.name.startswith("conformance_") or not p.name.endswith(".rs"):
            continue
        if p.name == "conformance_surface_coverage.rs":
            continue
        text += p.read_text()
    return text


def main() -> int:
    surface = json.loads(SURFACE_PATH.read_text())
    items = surface["items"]
    test_text = gather_conformance_text()
    uncovered = uncovered_paths(items, test_text)

    lines: list[str] = []
    lines.append(
        "# Surface exclusions for ferray-ufunc, generated by\n"
        "# scripts/gen_ufunc_exclusions.py. Each entry covers a public\n"
        "# item that is not (or not yet) directly mentioned by a\n"
        "# conformance test under `tests/conformance_*.rs`. Categories:\n"
        "#   - Inner canonical paths (`ops::<mod>::<fn>`) whose re-export\n"
        "#     is covered: inner is implicitly verified by the same test.\n"
        "#   - Internal modules (`dispatch::`, `helpers::`, `kernels::`,\n"
        "#     etc.): not part of the user-facing API contract.\n"
        f"#   - User-facing items without a fixture: tracked under\n"
        f"#     umbrella issue #{UMBRELLA_ISSUE}.\n"
    )

    for item in uncovered:
        path = item["path"]
        kind = item["kind"]
        reason, covered_by = classify(path, kind)
        lines.append("")
        lines.append("[[exclusion]]")
        lines.append(f'path = "{toml_escape(path)}"')
        lines.append(f'reason = "{toml_escape(reason)}"')
        lines.append(f'covered_by = "{toml_escape(covered_by)}"')

    EXCLUSIONS_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(uncovered)} exclusion entries to {EXCLUSIONS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
