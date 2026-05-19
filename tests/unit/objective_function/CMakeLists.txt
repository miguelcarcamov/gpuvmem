gpuvmem_add_cuda_unit_test(
  gpuvmem_unit_objective_function
  SOURCES
  test_objective_function.cu ${CMAKE_SOURCE_DIR}/tests/common/objective_cuda_globals.cu
  ${CMAKE_SOURCE_DIR}/tests/common/flag_opt_stub.cu
  ${CMAKE_SOURCE_DIR}/tests/common/chi2_link_stub.cu
  ${CMAKE_SOURCE_DIR}/src/objective_function/fi_context.cu)
