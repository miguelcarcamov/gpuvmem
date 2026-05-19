gpuvmem_add_unit_test(
  gpuvmem_unit_math_utils
  SOURCES test_math_utils.cc ${CMAKE_SOURCE_DIR}/src/utils/math_utils.cc)

gpuvmem_add_unit_test(gpuvmem_unit_imaging_geometry SOURCES test_imaging_geometry.cc)
