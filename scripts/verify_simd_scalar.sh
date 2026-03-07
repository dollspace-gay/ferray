#!/usr/bin/env bash
set -euo pipefail

echo "=== SIMD vs Scalar Verification ==="
echo ""

# ---------------------------------------------------------------------------
# Phase 1: Run full test suite with SIMD (default)
# ---------------------------------------------------------------------------
echo "--- Running tests with SIMD (default) ---"
if cargo test --workspace 2>&1 | tee /tmp/ferray_simd.txt; then
    SIMD_RESULT=0
else
    SIMD_RESULT=$?
fi

echo ""

# ---------------------------------------------------------------------------
# Phase 2: Run full test suite with scalar fallback
# ---------------------------------------------------------------------------
echo "--- Running tests with FERRAY_FORCE_SCALAR=1 ---"
if FERRAY_FORCE_SCALAR=1 cargo test --workspace 2>&1 | tee /tmp/ferray_scalar.txt; then
    SCALAR_RESULT=0
else
    SCALAR_RESULT=$?
fi

echo ""

# ---------------------------------------------------------------------------
# Phase 3: Compare results
# ---------------------------------------------------------------------------
echo "=== Results ==="

# Extract test counts from "test result:" lines
# Format: "test result: ok. N passed; M failed; I ignored; ..."
SIMD_PASS=$(grep "^test result:" /tmp/ferray_simd.txt | awk '{s+=$4} END {print s+0}')
SIMD_FAIL=$(grep "^test result:" /tmp/ferray_simd.txt | awk '{s+=$6} END {print s+0}')
SCALAR_PASS=$(grep "^test result:" /tmp/ferray_scalar.txt | awk '{s+=$4} END {print s+0}')
SCALAR_FAIL=$(grep "^test result:" /tmp/ferray_scalar.txt | awk '{s+=$6} END {print s+0}')

echo "SIMD:   ${SIMD_PASS} passed, ${SIMD_FAIL} failed (exit: ${SIMD_RESULT})"
echo "Scalar: ${SCALAR_PASS} passed, ${SCALAR_FAIL} failed (exit: ${SCALAR_RESULT})"

if [ "${SIMD_RESULT}" -eq 0 ] && [ "${SCALAR_RESULT}" -eq 0 ]; then
    echo ""
    echo "PASS: SIMD vs Scalar verification PASSED - both paths produce zero failures"
    echo "  SIMD passed ${SIMD_PASS} tests, Scalar passed ${SCALAR_PASS} tests"
    exit 0
else
    echo ""
    echo "FAIL: SIMD vs Scalar verification FAILED"
    if [ "${SIMD_RESULT}" -ne 0 ]; then
        echo "  SIMD run had failures (exit code ${SIMD_RESULT})"
    fi
    if [ "${SCALAR_RESULT}" -ne 0 ]; then
        echo "  Scalar run had failures (exit code ${SCALAR_RESULT})"
    fi
    exit 1
fi
