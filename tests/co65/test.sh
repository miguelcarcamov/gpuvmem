#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/e2e_lib.sh"

GPUVMEM=${1:?}
DIR=${2:?}

COMMON=(
  -i "${DIR}/co65.ms"
  -o "${DIR}/residuals.ms"
  -O "${DIR}/mod_out.fits"
  -m "${DIR}/mod_in_0.fits"
  -p "${DIR}/mem/"
  -X 16 -Y 16 -V 256
  -z 0.001
  -Z 0.001
  -g 1
)

ARGS=("${COMMON[@]}" --verbose)
e2e_append_iter_flag ARGS
e2e_gpuvmem_run "default_verbose_gridding" "${GPUVMEM}" "${DIR}" -- "${ARGS[@]}"

ARGS=("${COMMON[@]}" -q --progress plain)
e2e_append_iter_flag ARGS
e2e_gpuvmem_run "quiet_plain" "${GPUVMEM}" "${DIR}" -- "${ARGS[@]}"

ARGS=("${COMMON[@]}" --verbose -W Briggs -R 0.0)
e2e_append_iter_flag ARGS
e2e_gpuvmem_run "briggs_weighting" "${GPUVMEM}" "${DIR}" -- "${ARGS[@]}"

e2e_cleanup_artifacts "${DIR}"
echo "All co65 E2E scenarios passed."
