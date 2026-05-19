# Google Test helpers for gpuvmem (unit + integration).

set(GPUVMEM_TEST_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/tests/common ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/classes
    ${CASACORE_INCLUDE_DIRS} ${CFITSIO_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDAToolkit_LIBRARY_ROOT}/samples/common/inc)

function(gpuvmem_gtest_fetch)
  if(TARGET GTest::gtest_main AND COMMAND gtest_discover_tests)
    return()
  endif()
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(googletest)
  include(GoogleTest)
endfunction()

function(gpuvmem_gtest_common_target_props TARGET)
  target_include_directories(${TARGET} PRIVATE ${GPUVMEM_TEST_INCLUDE_DIRS})
  target_link_libraries(${TARGET} PRIVATE GTest::gtest_main)
endfunction()

function(gpuvmem_gtest_discover TARGET)
  if(NOT ARGN)
    message(FATAL_ERROR "gpuvmem_gtest_discover(${TARGET}): at least one CTest label required")
  endif()
  add_test(NAME ${TARGET} COMMAND ${TARGET})
  set_property(TEST ${TARGET} PROPERTY LABELS ${ARGN})
endfunction()

# Host-only C++ test (no .cu sources).
function(gpuvmem_add_unit_test TARGET)
  cmake_parse_arguments(ARG "" "" "SOURCES;LINK_LIBS" ${ARGN})
  if(NOT ARG_SOURCES)
    message(FATAL_ERROR "gpuvmem_add_unit_test: SOURCES required")
  endif()
  add_executable(${TARGET} ${ARG_SOURCES})
  gpuvmem_gtest_common_target_props(${TARGET})
  if(ARG_LINK_LIBS)
    target_link_libraries(${TARGET} PRIVATE ${ARG_LINK_LIBS})
  endif()
  gpuvmem_gtest_discover(${TARGET} unit)
endfunction()

function(gpuvmem_add_integration_test TARGET)
  cmake_parse_arguments(ARG "" "" "SOURCES;LINK_LIBS" ${ARGN})
  if(NOT ARG_SOURCES)
    message(FATAL_ERROR "gpuvmem_add_integration_test: SOURCES required")
  endif()
  add_executable(${TARGET} ${ARG_SOURCES})
  gpuvmem_gtest_common_target_props(${TARGET})
  if(ARG_LINK_LIBS)
    target_link_libraries(${TARGET} PRIVATE ${ARG_LINK_LIBS})
  endif()
  gpuvmem_gtest_discover(${TARGET} integration)
endfunction()

# CUDA test executable (.cu and/or .cc).
function(gpuvmem_add_cuda_unit_test TARGET)
  cmake_parse_arguments(ARG "REQUIRES_GPU" "" "SOURCES;LINK_LIBS" ${ARGN})
  if(NOT ARG_SOURCES)
    message(FATAL_ERROR "gpuvmem_add_cuda_unit_test: SOURCES required")
  endif()
  add_executable(${TARGET} ${ARG_SOURCES})
  gpuvmem_gtest_common_target_props(${TARGET})
  target_link_libraries(${TARGET} PRIVATE ${CUDA_LIBS})
  if(ARG_LINK_LIBS)
    target_link_libraries(${TARGET} PRIVATE ${ARG_LINK_LIBS})
  endif()
  if(ARG_REQUIRES_GPU)
    gpuvmem_gtest_discover(${TARGET} unit gpu)
  else()
    gpuvmem_gtest_discover(${TARGET} unit)
  endif()
endfunction()

function(gpuvmem_add_cuda_integration_test TARGET)
  cmake_parse_arguments(ARG "REQUIRES_GPU" "" "SOURCES;LINK_LIBS" ${ARGN})
  if(NOT ARG_SOURCES)
    message(FATAL_ERROR "gpuvmem_add_cuda_integration_test: SOURCES required")
  endif()
  add_executable(${TARGET} ${ARG_SOURCES})
  gpuvmem_gtest_common_target_props(${TARGET})
  target_link_libraries(${TARGET} PRIVATE ${CUDA_LIBS})
  if(ARG_LINK_LIBS)
    target_link_libraries(${TARGET} PRIVATE ${ARG_LINK_LIBS})
  endif()
  if(ARG_REQUIRES_GPU)
    gpuvmem_gtest_discover(${TARGET} integration gpu)
  else()
    gpuvmem_gtest_discover(${TARGET} integration)
  endif()
endfunction()

# Shared CUDA harness (globals + linkAddToDPhi stub + harness).
set(GPUVMEM_TEST_FI_TERM_SOURCES
    ${CMAKE_SOURCE_DIR}/src/objective_function/fi_context.cu
    ${CMAKE_SOURCE_DIR}/src/objective_function/terms/regularizers/entropy.cu
    ${CMAKE_SOURCE_DIR}/src/objective_function/terms/regularizers/l1norm.cu
    ${CMAKE_SOURCE_DIR}/src/objective_function/terms/regularizers/l2constantprior.cu
    ${CMAKE_SOURCE_DIR}/src/objective_function/terms/regularizers/laplacian.cu
    ${CMAKE_SOURCE_DIR}/src/objective_function/terms/regularizers/quadraticpenalization.cu
    ${CMAKE_SOURCE_DIR}/src/objective_function/terms/regularizers/totalsquaredvariation.cu
    ${CMAKE_SOURCE_DIR}/src/objective_function/terms/regularizers/totalvariation.cu
    ${CMAKE_SOURCE_DIR}/src/regularizer_kernels/regularizers_host.cu
    ${CMAKE_SOURCE_DIR}/src/regularizer_kernels/regularizers_kernels.cu
    ${CMAKE_SOURCE_DIR}/src/reduction/reduction_host.cu
    ${CMAKE_SOURCE_DIR}/src/reduction/reduction_kernels.cu)

set(GPUVMEM_TEST_COMMON_CUDA_SOURCES
    ${CMAKE_SOURCE_DIR}/tests/common/objective_cuda_globals.cu
    ${CMAKE_SOURCE_DIR}/tests/common/objective_cuda_harness.cu
    ${CMAKE_SOURCE_DIR}/tests/common/legacy_imaging_globals.cu
    ${CMAKE_SOURCE_DIR}/tests/common/chi2_link_stub.cu
    ${CMAKE_SOURCE_DIR}/src/objective_function/fi_context.cu
    ${CMAKE_SOURCE_DIR}/src/reduction/reduction_host.cu
    ${CMAKE_SOURCE_DIR}/src/reduction/reduction_kernels.cu)

function(gpuvmem_link_openmp_if_needed TARGET)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(${TARGET} PRIVATE OpenMP::OpenMP_CXX)
  endif()
endfunction()

function(gpuvmem_link_test_external_libs TARGET)
  if(CCfits_LIBRARIES OR CFITSIO_LIBRARIES)
    target_link_libraries(${TARGET} PRIVATE ${CCfits_LIBRARIES} ${CFITSIO_LIBRARIES})
  endif()
  if(CASACORE_LIBRARIES)
    target_link_libraries(${TARGET} PRIVATE ${CASACORE_LIBRARIES})
  endif()
  if(Boost_LIBRARIES)
    target_link_libraries(${TARGET} PRIVATE ${Boost_LIBRARIES})
  endif()
endfunction()
