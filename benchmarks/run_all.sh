#!/bin/bash
# run_all.sh — Run all core-lens benchmark scripts under Scalene
# Usage: ./benchmarks/run_all.sh [--profile | --time-only]
#
# Modes:
#   --profile   (default) Generate Scalene JSON profiles in benchmarks/profiles/
#               View with: uv run scalene view <profile.json>
#   --time-only No Scalene overhead; plain uv run python for timing only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILES_DIR="${SCRIPT_DIR}/profiles"
mkdir -p "${PROFILES_DIR}"

MODE="${1:---profile}"

BENCHMARKS=(
    "bench_aoi"
    "bench_entity"
    "bench_spatial"
    # "bench_paths"
    "bench_view"
    "bench_result"
    "bench_season"
    "bench_schema"
    "bench_polars_utils"
    "bench_export"
)

echo "=================================================="
echo "  core-lens benchmark suite"
echo "  Mode    : ${MODE}"
echo "  Profiles: ${PROFILES_DIR}"
echo "=================================================="

PASS=0
FAIL=0

for bench in "${BENCHMARKS[@]}"; do
    script="${SCRIPT_DIR}/${bench}.py"
    echo ""
    echo "──────────────────────────────────────────────────"
    echo "  Running: ${bench}"
    echo "──────────────────────────────────────────────────"

    if [[ "${MODE}" == "--time-only" ]]; then
        if uv run python "${script}"; then
            echo "  ✓ ${bench} OK"
            ((PASS++)) || true
        else
            echo "  ✗ ${bench} FAILED"
            ((FAIL++)) || true
        fi
    else
        # --profile (default)
        outfile="${PROFILES_DIR}/${bench}_profile.json"
        if uv run scalene run -o "${outfile}" "${script}"; then
            echo "  ✓ ${bench} → ${outfile}"
            ((PASS++)) || true
        else
            echo "  ✗ ${bench} FAILED"
            ((FAIL++)) || true
        fi
    fi
done

echo ""
echo "=================================================="
echo "  Results: ${PASS} passed, ${FAIL} failed"
if [[ "${MODE}" != "--time-only" ]]; then
    echo "  Profiles saved to: ${PROFILES_DIR}/"
    echo "  To view them, use: uv run scalene view ${PROFILES_DIR}/<profile>.json"
fi
echo "=================================================="

[[ ${FAIL} -eq 0 ]]
