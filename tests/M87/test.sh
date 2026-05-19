#!/usr/bin/env bash
# M87 end-to-end scenarios (multiple CLI / optimization parameter sets).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/e2e_lib.sh
source "${SCRIPT_DIR}/../common/e2e_lib.sh"

GPUVMEM=${1:?gpuvmem binary required}
DIR=${2:?dataset directory required}

COMMON=(
  -i "${DIR}/SR1_M87_2017_101_hi_hops_netcal_StokesI.selfcal.LLRR.ms"
  -o "${DIR}/residuals.ms"
  -O "${DIR}/mod_out.fits"
  -m "${DIR}/mod_in_0.fits"
  -p "${DIR}/mem/"
  -X 16 -Y 16 -V 256
  -z 0.0,0.0
  -Z 0.0,0.001,0.005
  -R -2.0
)

ARGS=("${COMMON[@]}" --verbose --use-radius-mask)
e2e_append_iter_flag ARGS
e2e_gpuvmem_run "default_verbose_radius_mask" "${GPUVMEM}" "${DIR}" -- "${ARGS[@]}"

ARGS=("${COMMON[@]}" -q --progress plain)
e2e_append_iter_flag ARGS
e2e_gpuvmem_run "quiet_plain_progress" "${GPUVMEM}" "${DIR}" -- "${ARGS[@]}"

ARGS=("${COMMON[@]}" --verbose --metrics-file "${DIR}/metrics.txt")
e2e_append_iter_flag ARGS
e2e_gpuvmem_run "verbose_metrics_file" "${GPUVMEM}" "${DIR}" -- "${ARGS[@]}"

ARGS=("${COMMON[@]}" --verbose -J joint)
e2e_append_iter_flag ARGS
e2e_gpuvmem_run "joint_optimization_mode" "${GPUVMEM}" "${DIR}" -- "${ARGS[@]}"

e2e_cleanup_artifacts "${DIR}"
echo "All M87 E2E scenarios passed."
