#!/usr/bin/env python3
"""Retrofit existing ferray fixtures to schema version 2.

Walks fixtures/**/*.json and prepends the two new top-level metadata keys
`numpy_version` and `fixture_schema_version` to every fixture that lacks them.

Schema version 2 adds (as the FIRST two keys in the file):
    "numpy_version": "2.4.4"
    "fixture_schema_version": 2

All existing keys (`function`, `ferray_function`, `test_cases`, etc.) are
preserved exactly in their original order. No test_cases values are changed.

Usage:
    python3 scripts/retrofit_fixtures_v2.py

The script is idempotent: running it twice produces zero git diff on the
second run.
"""

import json
import sys
from pathlib import Path

NUMPY_VERSION = "2.4.4"
FIXTURE_SCHEMA_VERSION = 2

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def retrofit_file(path: Path) -> bool:
    """Retrofit a single fixture file.  Returns True if the file was modified."""
    with open(path, encoding="utf-8") as f:
        original_text = f.read()

    data = json.loads(original_text)

    # Already at v2 — skip (idempotency).
    if data.get("fixture_schema_version") == FIXTURE_SCHEMA_VERSION:
        return False

    # Build new dict: new metadata keys first, then existing keys in order.
    new_data: dict = {}
    new_data["numpy_version"] = NUMPY_VERSION
    new_data["fixture_schema_version"] = FIXTURE_SCHEMA_VERSION
    for key, value in data.items():
        if key in ("numpy_version", "fixture_schema_version"):
            # Skip if already present (shouldn't be at v1, but be safe).
            continue
        new_data[key] = value

    new_text = json.dumps(new_data, indent=2) + "\n"

    # Write only if content changed (belt-and-suspenders idempotency).
    if new_text != original_text:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_text)
        return True

    return False


def main() -> None:
    if not FIXTURES_DIR.is_dir():
        print(f"ERROR: fixtures directory not found: {FIXTURES_DIR}", file=sys.stderr)
        sys.exit(1)

    json_files = sorted(FIXTURES_DIR.rglob("*.json"))
    if not json_files:
        print("No fixture JSON files found.", file=sys.stderr)
        sys.exit(1)

    migrated = 0
    already_v2 = 0
    errors = 0

    for path in json_files:
        try:
            changed = retrofit_file(path)
            if changed:
                migrated += 1
                print(f"  migrated: {path.relative_to(FIXTURES_DIR.parent)}")
            else:
                already_v2 += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR processing {path}: {exc}", file=sys.stderr)
            errors += 1

    print(
        f"\nSummary: {migrated} files migrated, "
        f"{already_v2} files already at v2"
        + (f", {errors} errors" if errors else "")
    )
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
