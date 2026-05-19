set(_GPUVMEM_OPTIMIZER_TEST_EXTRA
    ${CMAKE_SOURCE_DIR}/tests/common/objective_cuda_globals.cu
    ${CMAKE_SOURCE_DIR}/src/reduction/reduction_host.cu
    ${CMAKE_SOURCE_DIR}/src/reduction/reduction_kernels.cu)

set(_GPUVMEM_OPTIMIZER_OBJECTIVE_EXTRA
    ${GPUVMEM_TEST_COMMON_CUDA_SOURCES} ${GPUVMEM_TEST_FI_TERM_SOURCES})

gpuvmem_add_cuda_integration_test(
  gpuvmem_integration_optimizer_stack
  SOURCES
  test_optimizer_linesearch_seeder_matrix.cu ${_GPUVMEM_OPTIMIZER_TEST_EXTRA}
  LINK_LIBS gpuvmem_optimizers_static gpuvmem_linesearch_static
            gpuvmem_appcore_static gpuvmem_common_static)
gpuvmem_link_openmp_if_needed(gpuvmem_integration_optimizer_stack)

gpuvmem_add_cuda_integration_test(
  gpuvmem_integration_optimizer_objective
  REQUIRES_GPU
  SOURCES
  test_optimizer_objective_smoke.cu ${_GPUVMEM_OPTIMIZER_OBJECTIVE_EXTRA}
  LINK_LIBS gpuvmem_optimizers_static gpuvmem_linesearch_static
            gpuvmem_appcore_static gpuvmem_common_static)
gpuvmem_link_openmp_if_needed(gpuvmem_integration_optimizer_objective)
gpuvmem_link_test_external_libs(gpuvmem_integration_optimizer_objective)
