#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/e2e_lib.sh"

GPUVMEM=${1:?}
DIR=${2:?}

COMMON=(
  -i "${DIR}/hd142_b9cont_self_tav.ms"
  -o "${DIR}/residuals.ms"
  -O "${DIR}/mod_out.fits"
  -m "${DIR}/mod_in_0.fits"
  -p "${DIR}/mem/"
  -X 16 -Y 16 -V 256
  -z 0.001,3.5
  -Z 0.005,0.0
  -g 1
)

ARGS=("${COMMON[@]}" --verbose --print-images --print-errors)
e2e_append_iter_flag ARGS
e2e_gpuvmem_run "default_verbose_errors" "${GPUVMEM}" "${DIR}" -- "${ARGS[@]}"

ARGS=("${COMMON[@]}" --verbose --progress plain --log-interval 5)
e2e_append_iter_flag ARGS
e2e_gpuvmem_run "plain_progress_log_interval" "${GPUVMEM}" "${DIR}" -- "${ARGS[@]}"

ARGS=("${COMMON[@]}" -q)
e2e_append_iter_flag ARGS
e2e_gpuvmem_run "quiet" "${GPUVMEM}" "${DIR}" -- "${ARGS[@]}"

e2e_cleanup_artifacts "${DIR}"
echo "All selfcalband9 E2E scenarios passed."
