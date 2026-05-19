#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/e2e_lib.sh"

GPUVMEM=${1:?}
DIR=${2:?}

COMMON=(
  -i "${DIR}/all_fields.ms"
  -o "${DIR}/residuals.ms"
  -O "${DIR}/mod_out.fits"
  -m "${DIR}/mod_in_0.fits"
  -p "${DIR}/mem/"
  -X 16 -Y 16 -V 256
  -z 0.001
  -Z 0.01,0.0
  -g 1
  -R 2.0
)

ARGS=("${COMMON[@]}" --verbose --print-images)
e2e_append_iter_flag ARGS
e2e_gpuvmem_run "default_natural_verbose" "${GPUVMEM}" "${DIR}" -- "${ARGS[@]}"

ARGS=("${COMMON[@]}" --verbose -W Uniform)
e2e_append_iter_flag ARGS
e2e_gpuvmem_run "uniform_weighting" "${GPUVMEM}" "${DIR}" -- "${ARGS[@]}"

ARGS=("${COMMON[@]}" --verbose --metrics-file "${DIR}/metrics.txt")
e2e_append_iter_flag ARGS
e2e_gpuvmem_run "metrics_file" "${GPUVMEM}" "${DIR}" -- "${ARGS[@]}"

e2e_cleanup_artifacts "${DIR}"
echo "All antennae E2E scenarios passed."
