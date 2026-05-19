set(_GPUVMEM_OBJECTIVE_INT_COMMON
    ${GPUVMEM_TEST_COMMON_CUDA_SOURCES}
    ${CMAKE_SOURCE_DIR}/tests/common/flag_opt_stub.cu
    ${GPUVMEM_TEST_FI_TERM_SOURCES})
set(_GPUVMEM_OBJECTIVE_INT_LIBS gpuvmem_common_static)

gpuvmem_add_cuda_integration_test(
  gpuvmem_integration_objective_regularizers
  REQUIRES_GPU
  SOURCES test_regularizer_calc_fi.cu ${_GPUVMEM_OBJECTIVE_INT_COMMON}
  LINK_LIBS ${_GPUVMEM_OBJECTIVE_INT_LIBS})
gpuvmem_link_openmp_if_needed(gpuvmem_integration_objective_regularizers)
gpuvmem_link_test_external_libs(gpuvmem_integration_objective_regularizers)

gpuvmem_add_cuda_integration_test(
  gpuvmem_integration_objective_geometry
  REQUIRES_GPU
  SOURCES test_regularizer_geometry.cu ${_GPUVMEM_OBJECTIVE_INT_COMMON}
  LINK_LIBS ${_GPUVMEM_OBJECTIVE_INT_LIBS})
gpuvmem_link_openmp_if_needed(gpuvmem_integration_objective_geometry)
gpuvmem_link_test_external_libs(gpuvmem_integration_objective_geometry)

gpuvmem_add_cuda_integration_test(
  gpuvmem_integration_objective_reg_combos
  REQUIRES_GPU
  SOURCES test_production_regularizer_combinations.cu ${_GPUVMEM_OBJECTIVE_INT_COMMON}
  LINK_LIBS ${_GPUVMEM_OBJECTIVE_INT_LIBS})
gpuvmem_link_openmp_if_needed(gpuvmem_integration_objective_reg_combos)
gpuvmem_link_test_external_libs(gpuvmem_integration_objective_reg_combos)
