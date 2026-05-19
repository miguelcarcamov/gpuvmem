gpuvmem_add_cuda_unit_test(
  gpuvmem_unit_optimizer
  SOURCES
  test_optimizer.cu ${CMAKE_SOURCE_DIR}/tests/common/objective_cuda_globals.cu
  ${CMAKE_SOURCE_DIR}/src/reduction/reduction_host.cu
  ${CMAKE_SOURCE_DIR}/src/reduction/reduction_kernels.cu
  LINK_LIBS gpuvmem_optimizers_static gpuvmem_linesearch_static
            gpuvmem_appcore_static gpuvmem_common_static)
gpuvmem_link_openmp_if_needed(gpuvmem_unit_optimizer)
