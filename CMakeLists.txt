cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(gpuvmem LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)
include(CMakePrintHelpers) # In case we need to print any variable

# Set a default build type if not passed
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(
    STATUS
      "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE
      "${default_build_type}"
      CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()

if(CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_CONFIGURATION_TYPES Debug Release)
  set(CMAKE_CONFIGURATION_TYPES
      "${CMAKE_CONFIGURATION_TYPES}"
      CACHE STRING "Reset the configurations to what we need" FORCE)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
find_package(Boost REQUIRED)
find_package(CFITSIO REQUIRED)
set(CASACORE_MAKE_REQUIRED_EXTERNALS_OPTIONAL TRUE)
find_package(Casacore REQUIRED COMPONENTS casa ms tables measures meas)
option(MEMORY_DEBUG "This sets the GDB debug for CUDA code")
option(USE_FAST_MATH
       "This option accelerate CUDA math functions decreasing precision")
if(Boost_FOUND)
  add_executable(gpuvmem ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cu)
  file(GLOB SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)
  target_sources(gpuvmem PRIVATE ${SOURCE_FILES})
  target_compile_features(gpuvmem PUBLIC cxx_std_11)

  if(PREFIX)
    set(BINARY_DIR ${PREFIX})
    set_target_properties(gpuvmem PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                             ${BINARY_DIR})
  else()
    set(BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bin)
    set_target_properties(gpuvmem PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                             ${BINARY_DIR})
  endif()
  message(NOTICE
          "We are currently using the latest casacore stable version >=v3.2.1")
  message(
    NOTICE
    "If one these versions are not installed in your computer, please make sure to install it"
  )
  message(NOTICE "For more information please read the README.md file")
  target_include_directories(
    gpuvmem
    PUBLIC ${CASACORE_INCLUDE_DIRS}
           ${CFITSIO_INCLUDE_DIR}
           ${CMAKE_CURRENT_SOURCE_DIR}/include
           ${CMAKE_CURRENT_SOURCE_DIR}/include/classes
           ${Boost_INCLUDE_DIRS}
           ${CUDAToolkit_LIBRARY_ROOT}/samples/common/inc)

  get_target_property(TEMP gpuvmem COMPILE_FLAGS)
  if(TEMP STREQUAL "TEMP-NOTFOUND")
    set(TEMP "") # Set to empty string
  else()
    set(TEMP "${TEMP} ") # A space to cleanly separate from existing content
  endif()

  # Find CUDA architecture
  include(FindCUDA/select_compute_arch)
  cuda_detect_installed_gpus(INSTALLED_GPU_CCS_1)
  string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
  string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
  string(REPLACE "." "" INSTALLED_GPU_CCS_4 "${INSTALLED_GPU_CCS_3}")
  string(REPLACE "+PTX" "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_4}")
  set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
  set_target_properties(gpuvmem PROPERTIES CUDA_ARCHITECTURES
                                           "${CUDA_ARCH_LIST}")

  message(STATUS "CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
  message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
  message(STATUS "CUDA Path: ${CUDAToolkit_LIBRARY_ROOT}")

  if(MEMORY_DEBUG)
    message(NOTICE "MEMORY DEBUG IS ON")
    set(COMPFLAGS "-g -G -D_FORCE_INLINES -w")
    set(TEMP "${TEMP}${COMPFLAGS}")
    set_target_properties(
      gpuvmem
      PROPERTIES COMPILE_FLAGS ${TEMP}
                 CUDA_SEPARABLE_COMPILATION ON
                 CUDA_STANDARD 11)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")
  else()
    message(NOTICE "MEMORY DEBUG IS OFF")
    if(USE_FAST_MATH)
      message(NOTICE "USING FAST_MATH")
      set(COMPFLAGS "-D_FORCE_INLINES -w --use_fast_math -O3")
      set(TEMP "${TEMP}${COMPFLAGS}")
      set_target_properties(
        gpuvmem
        PROPERTIES COMPILE_FLAGS ${TEMP}
                   CUDA_SEPARABLE_COMPILATION ON
                   CUDA_STANDARD 11)
      set(CMAKE_CUDA_FLAGS
          "${CMAKE_CUDA_FLAGS} -Xptxas -O3 -Xcompiler -fopenmp")
    else()
      set(COMPFLAGS "-D_FORCE_INLINES -w -O3")
      set(TEMP "${TEMP}${COMPFLAGS}")
      set_target_properties(
        gpuvmem
        PROPERTIES COMPILE_FLAGS ${TEMP}
                   CUDA_SEPARABLE_COMPILATION ON
                   CUDA_STANDARD 11)
      set(CMAKE_CUDA_FLAGS
          "${CMAKE_CUDA_FLAGS} -Xptxas -O3 -Xcompiler -fopenmp")
    endif()
  endif()

  set(ALL_CUDA_LIBRARIES ${CUDA_cuda_LIBRARY} ${CUDA_cudart_LIBRARY}
                         ${CUDA_cufft_LIBRARY})
  set(ALL_LIBRARIES ${CASACORE_LIBRARIES} ${CFITSIO_LIBRARY} ${Boost_LIBRARIES})

  target_link_libraries(gpuvmem m stdc++ gomp ${ALL_CUDA_LIBRARIES}
                        ${ALL_LIBRARIES})
  # Adding install target
  install(TARGETS gpuvmem DESTINATION bin)
  # Tests - Tests all the testing datasets
  enable_testing()
  set(TEST_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  add_test(antennae bash ${TEST_DIRECTORY}/antennae/test.sh
           ${BINARY_DIR}/gpuvmem ${TEST_DIRECTORY}/antennae)
  add_test(co65 bash ${TEST_DIRECTORY}/co65/test.sh ${BINARY_DIR}/gpuvmem
           ${TEST_DIRECTORY}/co65)
  add_test(freq78 bash ${TEST_DIRECTORY}/FREQ78/test.sh ${BINARY_DIR}/gpuvmem
           ${TEST_DIRECTORY}/FREQ78)
  add_test(m87 bash ${TEST_DIRECTORY}/M87/test.sh ${BINARY_DIR}/gpuvmem
           ${TEST_DIRECTORY}/M87)
  add_test(selfcalband9 bash ${TEST_DIRECTORY}/selfcalband9/test.sh
           ${BINARY_DIR}/gpuvmem ${TEST_DIRECTORY}/selfcalband9)
else()
  message(
    NOTICE
    "Boost is not installed in your system. Please make sure Boost is installed before compiling."
  )
endif()
