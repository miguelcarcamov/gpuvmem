# Shared helpers for dataset end-to-end shell tests.
# Usage: source "$(dirname "$0")/../common/e2e_lib.sh"
#
# Environment:
#   GPUVMEM_E2E_MAX_ITER  Max optimizer iterations per scenario (default: 25).

: "${GPUVMEM_E2E_MAX_ITER:=25}"

e2e_fail() {
  echo "E2E ERROR: $*" >&2
  exit 1
}

# e2e_gpuvmem_run <scenario_name> <gpuvmem_bin> <work_dir> -- <gpuvmem args...>
e2e_gpuvmem_run() {
  local scenario_name=$1
  local gpuvmem_bin=$2
  local work_dir=$3
  shift 3
  if [[ "${1:-}" != "--" ]]; then
    e2e_fail "e2e_gpuvmem_run: expected '--' before gpuvmem arguments"
  fi
  shift
  echo "=== E2E [${scenario_name}] max_iter=${GPUVMEM_E2E_MAX_ITER} ==="
  if ! "${gpuvmem_bin}" "$@"; then
    e2e_fail "scenario '${scenario_name}' failed (exit $?)"
  fi
  echo "=== E2E [${scenario_name}] OK ==="
}

e2e_append_iter_flag() {
  local -n _args=$1
  local has_t=0
  for a in "${_args[@]}"; do
    [[ "$a" == "-t" || "$a" == --iterations ]] && has_t=1
  done
  if [[ $has_t -eq 0 ]]; then
    _args+=(-t "${GPUVMEM_E2E_MAX_ITER}")
  fi
}

e2e_cleanup_artifacts() {
  local work_dir=$1
  rm -rf "${work_dir}/residuals.ms"
  rm -rf "${work_dir}/mem/"
  rm -f "${work_dir}/alpha.fits" "${work_dir}/mod_out.fits" "${work_dir}/metrics.txt"
}
