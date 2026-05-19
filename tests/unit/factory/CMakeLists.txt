# Factory registration tests (one executable per product family).

macro(gpuvmem_factory_unit TARGET SRC LIBS)
  gpuvmem_add_unit_test(${TARGET} SOURCES ${SRC} LINK_LIBS ${LIBS})
  gpuvmem_link_openmp_if_needed(${TARGET})
endmacro()

macro(gpuvmem_factory_cuda_unit TARGET SRC LIBS)
  gpuvmem_add_cuda_unit_test(${TARGET} SOURCES ${SRC} LINK_LIBS ${LIBS})
  gpuvmem_link_openmp_if_needed(${TARGET})
  gpuvmem_link_test_external_libs(${TARGET})
endmacro()

gpuvmem_add_cuda_unit_test(
  gpuvmem_unit_factory_fi
  SOURCES
  test_fi_factory.cu ${CMAKE_SOURCE_DIR}/tests/common/objective_cuda_globals.cu
  ${CMAKE_SOURCE_DIR}/tests/common/flag_opt_stub.cu
  ${CMAKE_SOURCE_DIR}/tests/common/chi2_link_stub.cu ${GPUVMEM_TEST_FI_TERM_SOURCES}
  LINK_LIBS gpuvmem_common_static)
gpuvmem_link_openmp_if_needed(gpuvmem_unit_factory_fi)

gpuvmem_add_cuda_unit_test(
  gpuvmem_unit_factory_optimizer
  SOURCES
  test_optimizer_factory.cu ${CMAKE_SOURCE_DIR}/tests/common/objective_cuda_globals.cu
  ${CMAKE_SOURCE_DIR}/src/reduction/reduction_host.cu
  ${CMAKE_SOURCE_DIR}/src/reduction/reduction_kernels.cu
  LINK_LIBS gpuvmem_optimizers_static gpuvmem_linesearch_static
            gpuvmem_appcore_static gpuvmem_common_static)
gpuvmem_link_openmp_if_needed(gpuvmem_unit_factory_optimizer)

gpuvmem_add_cuda_unit_test(
  gpuvmem_unit_factory_linesearch
  SOURCES
  test_linesearch_factory.cu ${CMAKE_SOURCE_DIR}/tests/common/objective_cuda_globals.cu
  ${CMAKE_SOURCE_DIR}/src/reduction/reduction_host.cu
  ${CMAKE_SOURCE_DIR}/src/reduction/reduction_kernels.cu
  LINK_LIBS gpuvmem_linesearch_static gpuvmem_optimizers_static
            gpuvmem_appcore_static gpuvmem_common_static)
gpuvmem_link_openmp_if_needed(gpuvmem_unit_factory_linesearch)

gpuvmem_add_cuda_unit_test(
  gpuvmem_unit_factory_seeder
  SOURCES
  test_seeder_factory.cu ${CMAKE_SOURCE_DIR}/tests/common/objective_cuda_globals.cu
  ${CMAKE_SOURCE_DIR}/src/reduction/reduction_host.cu
  ${CMAKE_SOURCE_DIR}/src/reduction/reduction_kernels.cu
  LINK_LIBS gpuvmem_linesearch_static gpuvmem_optimizers_static
            gpuvmem_appcore_static gpuvmem_common_static)
gpuvmem_link_openmp_if_needed(gpuvmem_unit_factory_seeder)

set(_WEIGHTING_FACTORY_SOURCES
    ${CMAKE_SOURCE_DIR}/src/weighting/uniformweightingscheme.cu
    ${CMAKE_SOURCE_DIR}/src/weighting/naturalweightingscheme.cu
    ${CMAKE_SOURCE_DIR}/src/weighting/radialweightingscheme.cu
    ${CMAKE_SOURCE_DIR}/src/weighting/briggsweightingscheme.cu)

gpuvmem_add_cuda_unit_test(
  gpuvmem_unit_factory_weighting
  SOURCES
  test_weighting_factory.cu ${CMAKE_SOURCE_DIR}/tests/common/imaging_link_stubs.cu
  ${CMAKE_SOURCE_DIR}/tests/common/legacy_imaging_globals.cu
  ${CMAKE_SOURCE_DIR}/tests/common/objective_cuda_globals.cu
  ${CMAKE_SOURCE_DIR}/tests/common/flag_opt_stub.cu ${_WEIGHTING_FACTORY_SOURCES}
  LINK_LIBS gpuvmem_common_static)
gpuvmem_link_openmp_if_needed(gpuvmem_unit_factory_weighting)

set(_CKERNEL_FACTORY_SOURCES
    ${CMAKE_SOURCE_DIR}/src/kernels/pillBox2D.cu
    ${CMAKE_SOURCE_DIR}/src/kernels/gaussian2D.cu
    ${CMAKE_SOURCE_DIR}/src/kernels/sinc2D.cu
    ${CMAKE_SOURCE_DIR}/src/kernels/gaussianSinc2D.cu
    ${CMAKE_SOURCE_DIR}/src/kernels/pswf_12D.cu)

gpuvmem_add_cuda_unit_test(
  gpuvmem_unit_factory_ckernel
  SOURCES
  test_ckernel_factory.cu ${CMAKE_SOURCE_DIR}/tests/common/imaging_link_stubs.cu
  ${CMAKE_SOURCE_DIR}/tests/common/legacy_imaging_globals.cu
  ${CMAKE_SOURCE_DIR}/tests/common/objective_cuda_globals.cu ${_CKERNEL_FACTORY_SOURCES}
  LINK_LIBS gpuvmem_common_static)
gpuvmem_link_openmp_if_needed(gpuvmem_unit_factory_ckernel)

gpuvmem_factory_cuda_unit(
  gpuvmem_unit_factory_io
  test_io_factory.cu
  "gpuvmem_io_static;gpuvmem_common_static")
gpuvmem_link_test_external_libs(gpuvmem_unit_factory_io)

# MFS factory pulls the full static dependency graph (same as the main binary).
# Synthesizer (MFS) factory: optional heavy target — enable with -DGPUVMEM_TEST_MFS_FACTORY=ON
if(GPUVMEM_TEST_MFS_FACTORY)
  add_executable(
    gpuvmem_unit_factory_synthesizer
    test_synthesizer_factory.cu
    ${CMAKE_SOURCE_DIR}/tests/common/objective_cuda_globals.cu
    ${CMAKE_SOURCE_DIR}/tests/common/imaging_link_stubs.cu)
  gpuvmem_gtest_common_target_props(gpuvmem_unit_factory_synthesizer)
  target_link_libraries(
    gpuvmem_unit_factory_synthesizer
    PRIVATE -Wl,--whole-archive ${GPUVMEM_INTERNAL_TARGETS} -Wl,--no-whole-archive m
            gomp ${CUDA_LIBS})
  gpuvmem_link_test_external_libs(gpuvmem_unit_factory_synthesizer)
  gpuvmem_gtest_discover(gpuvmem_unit_factory_synthesizer unit gpu)
endif()
