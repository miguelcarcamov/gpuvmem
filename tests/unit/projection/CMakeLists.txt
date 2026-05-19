gpuvmem_add_cuda_unit_test(
  gpuvmem_unit_projection
  REQUIRES_GPU
  SOURCES
  test_projection.cu
  ${CMAKE_SOURCE_DIR}/src/projection/projection.cu
  LINK_LIBS gpuvmem_common_static)
