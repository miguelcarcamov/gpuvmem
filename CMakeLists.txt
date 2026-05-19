# ============================================================================
# gpuvmem - GPU-Accelerated Radio Astronomy Maximum Entropy Image Synthesis
# ============================================================================
#
# This CMakeLists.txt builds gpuvmem: a CUDA-accelerated C++ framework for
# regularized (RML-style) radio interferometric imaging — data fidelity plus
# modular regularizers — evolved from the original maximum-entropy (MEM) code.
#
# USE CASES:
# ----------
# 1. Standard Build: mkdir build && cd build && cmake .. && make
#
# 1. Custom CUDA Architecture: cmake -DCUDA_ARCH=75 ..  # Specify GPU compute
#   capability
#
# 1. Debug Build: cmake -DMEMORY_DEBUG=ON ..
#
# 1. Fast Math (lower precision, faster): cmake -DUSE_FAST_MATH=ON ..
#
# 1. Custom C++ Standard (for older systems): cmake -DCMAKE_CXX_STANDARD=11 ..
#
# 1. Custom Install Prefix: cmake -DPREFIX=/custom/path ..
#
# 1. Install program only (default): cmake --install . --prefix /usr/local
#    (installs gpuvmem to bindir; static libs not installed unless enabled)
#
# 1. Install static libs + headers for custom mains:
#    cmake .. -DGPUVMEM_INSTALL_STATIC_LIBS=ON -DGPUVMEM_INSTALL_HEADERS=ON
#    cmake --install . --prefix /path
#
# REQUIREMENTS: - CUDA Toolkit 10.0+ (13.0+ requires C++17) - CMake 3.18+ -
# Boost libraries - CCfits (C++ FITS I/O; requires CFITSIO) - Casacore (casa, ms,
# tables, measures, meas components) - NVIDIA GPU with compute capability >= 5.0
#
# ============================================================================

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(gpuvmem LANGUAGES C CXX CUDA)

include(GNUInstallDirs)

# ============================================================================
# CMake Policies
# ============================================================================

# Suppress FindBoost deprecation warning (CMP0167) Modern CMake uses
# BoostConfig.cmake, but we keep FindBoost for compatibility
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 OLD)
endif()

# Suppress nvpl warnings (optional NVIDIA Performance Primitives Library) These
# warnings come from FindBLAS/FindLAPACK and are harmless
set(CMAKE_FIND_PACKAGE_WARNINGS_OVERRIDE "nvpl" OFF)

# ============================================================================
# Build Configuration
# ============================================================================

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE
      "Release"
      CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()

# Limit configuration types for multi-config generators
if(CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_CONFIGURATION_TYPES
      "Debug;Release"
      CACHE STRING "" FORCE)
endif()

# ============================================================================
# Dependencies
# ============================================================================

# Add custom CMake modules (FindCasacore, FindCFITSIO)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(Boost REQUIRED)
find_package(CFITSIO REQUIRED)  # Required by CCfits
find_package(CCfits REQUIRED)  # Mandatory C++ FITS I/O in gpuvmem

# Casacore configuration Make external dependencies optional to avoid build
# failures
set(CASACORE_MAKE_REQUIRED_EXTERNALS_OPTIONAL TRUE)
find_package(Casacore REQUIRED COMPONENTS casa ms tables measures meas)

# ============================================================================
# Build Options
# ============================================================================

option(MEMORY_DEBUG "Enable GDB debugging for CUDA code (adds -g -G flags)" OFF)
option(USE_FAST_MATH
       "Use fast math functions (decreases precision, increases speed)" OFF)

# Installation (cmake --install / make install under CMAKE_INSTALL_PREFIX)
option(GPUVMEM_INSTALL_EXECUTABLE
       "Install the gpuvmem program (built from src/main.cu) to CMAKE_INSTALL_BINDIR"
       ON)
option(GPUVMEM_INSTALL_STATIC_LIBS
       "Install internal static libraries to CMAKE_INSTALL_LIBDIR (for custom mains / other projects)"
       OFF)
option(GPUVMEM_INSTALL_HEADERS
       "Install include/ tree to CMAKE_INSTALL_INCLUDEDIR/gpuvmem (recommended with GPUVMEM_INSTALL_STATIC_LIBS)"
       OFF)
option(GPUVMEM_BUILD_TESTS "Build GoogleTest unit and integration tests" ON)
option(GPUVMEM_TEST_MFS_FACTORY
       "Build heavy MFS synthesizer factory unit test (links all static libs)" OFF)

# ============================================================================
# C++ Standard Configuration
# ============================================================================

# Strategy: Use C++17 by default (CUDA 13+ requires it, but works for all CUDA
# versions) Users can override: cmake -DCMAKE_CXX_STANDARD=11 ..
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD
      17
      CACHE STRING "C++ standard" FORCE)
  set(CMAKE_CUDA_STANDARD
      17
      CACHE STRING "CUDA C++ standard" FORCE)

  if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
    message(STATUS "C++ standard: 17 (required by CUDA ${CUDAToolkit_VERSION})")
  else()
    message(
      STATUS
        "C++ standard: 17 (default, override with -DCMAKE_CXX_STANDARD=11 if needed)"
    )
  endif()
else()
  # User specified standard - ensure CUDA matches
  set(CMAKE_CUDA_STANDARD
      ${CMAKE_CXX_STANDARD}
      CACHE STRING "CUDA C++ standard" FORCE)

  # Warn if CUDA 13+ but user specified < 17
  if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0" AND CMAKE_CXX_STANDARD
                                                          VERSION_LESS "17")
    message(
      WARNING
        "CUDA ${CUDAToolkit_VERSION} requires C++17, but CMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD} was specified.\n"
        "This may cause compilation errors. Consider using C++17.")
  endif()

  message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD} (user specified)")
endif()

# Require the specified standard (fail if compiler doesn't support it)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ============================================================================
# Target Definition
# ============================================================================

# Verify Boost was found
if(NOT Boost_FOUND)
  message(FATAL_ERROR "Boost is required but not found. Please install Boost.")
endif()

# Domain source groups (STATIC libraries)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

file(GLOB APP_CORE_CUDA_SOURCES ${SRC_DIR}/options.cu ${SRC_DIR}/cli/*.cu)
file(GLOB APP_CORE_CXX_SOURCES ${SRC_DIR}/cli/*.cc)
set(APP_CORE_SOURCES ${APP_CORE_CUDA_SOURCES} ${APP_CORE_CXX_SOURCES})
file(GLOB MFS_CUDA_SOURCES ${SRC_DIR}/mfs/*.cu)
file(GLOB MFS_CPP_SOURCES ${SRC_DIR}/mfs/*.cc)
set(MFS_SOURCES ${MFS_CUDA_SOURCES} ${MFS_CPP_SOURCES})
file(GLOB OPTIMIZER_SOURCES CONFIGURE_DEPENDS ${SRC_DIR}/optimizers/*.cu)
file(GLOB LINESEARCH_SOURCES ${SRC_DIR}/linesearch/*.cu ${SRC_DIR}/seeding/*.cu
                              ${SRC_DIR}/projection/*.cu)
file(
  GLOB
  REGULARIZER_SOURCES
  ${SRC_DIR}/regularizer_kernels/*.cu
  ${SRC_DIR}/objective_function/terms/chi2/*.cu
  ${SRC_DIR}/objective_function/terms/regularizers/*.cu
  ${SRC_DIR}/objective_function/fi_context.cu)
file(
  GLOB
  IMAGING_SOURCES
  ${SRC_DIR}/beam/*.cu
  ${SRC_DIR}/chi2/*.cu
  ${SRC_DIR}/errors/*.cu
  ${SRC_DIR}/fft/*.cu
  ${SRC_DIR}/gridding/*.cu
  ${SRC_DIR}/idft/*.cu
  ${SRC_DIR}/image_processing/*.cu
  ${SRC_DIR}/kernels/*.cu
  ${SRC_DIR}/measurement_operator/*.cu
  ${SRC_DIR}/reduction/*.cu
  ${SRC_DIR}/visibility/*.cu
  ${SRC_DIR}/weighting/*.cu)
file(GLOB IO_SOURCES ${SRC_DIR}/fits/*.cu ${SRC_DIR}/io/*.cu)
file(GLOB MS_SOURCES ${SRC_DIR}/ms/*.cu)
file(GLOB COMMON_CUDA_SOURCES ${SRC_DIR}/utils/*.cu)
file(GLOB COMMON_CPP_SOURCES ${SRC_DIR}/utils/*.cc ${SRC_DIR}/rng/*.cc)

add_library(gpuvmem_appcore_static STATIC ${APP_CORE_SOURCES})
foreach(CC_FILE ${APP_CORE_CXX_SOURCES})
  set_source_files_properties(${CC_FILE} PROPERTIES LANGUAGE CXX)
endforeach()
add_library(gpuvmem_mfs_static STATIC ${MFS_SOURCES})
foreach(CC_FILE ${MFS_CPP_SOURCES})
  set_source_files_properties(${CC_FILE} PROPERTIES LANGUAGE CXX)
endforeach()
add_library(gpuvmem_optimizers_static STATIC ${OPTIMIZER_SOURCES})
add_library(gpuvmem_linesearch_static STATIC ${LINESEARCH_SOURCES})
add_library(gpuvmem_regularizers_static STATIC ${REGULARIZER_SOURCES})
add_library(gpuvmem_imaging_static STATIC ${IMAGING_SOURCES})
add_library(gpuvmem_io_static STATIC ${IO_SOURCES})
add_library(gpuvmem_ms_static STATIC ${MS_SOURCES})
add_library(gpuvmem_common_static STATIC ${COMMON_CUDA_SOURCES} ${COMMON_CPP_SOURCES})

foreach(CC_FILE ${COMMON_CPP_SOURCES})
  set_source_files_properties(${CC_FILE} PROPERTIES LANGUAGE CXX)
endforeach()

# Main executable target
add_executable(gpuvmem ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cu)

set(
  GPUVMEM_INTERNAL_TARGETS
  gpuvmem_appcore_static
  gpuvmem_mfs_static
  gpuvmem_optimizers_static
  gpuvmem_linesearch_static
  gpuvmem_regularizers_static
  gpuvmem_imaging_static
  gpuvmem_io_static
  gpuvmem_ms_static
  gpuvmem_common_static)

set(GPUVMEM_ALL_TARGETS gpuvmem ${GPUVMEM_INTERNAL_TARGETS})

foreach(TGT ${GPUVMEM_ALL_TARGETS})
  target_compile_features(${TGT} PUBLIC cxx_std_${CMAKE_CXX_STANDARD})
endforeach()

# Configure output directories (outside build/: project-root bin/ and lib/)
set(BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)
if(PREFIX)
  set(BINARY_DIR ${PREFIX})
endif()
set_target_properties(gpuvmem PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BINARY_DIR})
foreach(_gpuvmem_lib ${GPUVMEM_INTERNAL_TARGETS})
  set_target_properties(
    ${_gpuvmem_lib}
    PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${LIB_DIR}
               # Windows / mixed: keep import libs with archives for static-only targets
               LIBRARY_OUTPUT_DIRECTORY ${LIB_DIR})
endforeach()

# Include directories
target_include_directories(
  gpuvmem
  PUBLIC ${CASACORE_INCLUDE_DIRS} # Casacore headers
         ${CFITSIO_INCLUDE_DIR} # CFITSIO headers
         ${CMAKE_CURRENT_SOURCE_DIR}/include # Project headers
         ${CMAKE_CURRENT_SOURCE_DIR}/include/classes
         ${CMAKE_CURRENT_SOURCE_DIR}/include/ms
         ${CMAKE_CURRENT_SOURCE_DIR}/include/fits
         ${CMAKE_CURRENT_SOURCE_DIR}/include/io
         ${CMAKE_CURRENT_SOURCE_DIR}/include/optimizers
         ${CMAKE_CURRENT_SOURCE_DIR}/include/linesearch
         ${CMAKE_CURRENT_SOURCE_DIR}/include/projection
         ${CMAKE_CURRENT_SOURCE_DIR}/include/regularizer_kernels
         ${CMAKE_CURRENT_SOURCE_DIR}/include/seeding
         ${CMAKE_CURRENT_SOURCE_DIR}/include/objective_function
         ${CMAKE_CURRENT_SOURCE_DIR}/include/beam
         ${CMAKE_CURRENT_SOURCE_DIR}/include/chi2
         ${CMAKE_CURRENT_SOURCE_DIR}/include/errors
         ${CMAKE_CURRENT_SOURCE_DIR}/include/fft
         ${CMAKE_CURRENT_SOURCE_DIR}/include/gridding
         ${CMAKE_CURRENT_SOURCE_DIR}/include/idft
         ${CMAKE_CURRENT_SOURCE_DIR}/include/measurement_operator
         ${CMAKE_CURRENT_SOURCE_DIR}/include/visibility
         ${CMAKE_CURRENT_SOURCE_DIR}/include/reduction
         ${CMAKE_CURRENT_SOURCE_DIR}/include/image_processing
         ${CMAKE_CURRENT_SOURCE_DIR}/include/utils
         ${CMAKE_CURRENT_SOURCE_DIR}/include/kernels
         ${CMAKE_CURRENT_SOURCE_DIR}/include/weighting
         ${CMAKE_CURRENT_SOURCE_DIR}/include/mfs
         ${CMAKE_CURRENT_SOURCE_DIR}/include/rng
         ${Boost_INCLUDE_DIRS} # Boost headers
         ${CUDAToolkit_LIBRARY_ROOT}/samples/common/inc # CUDA samples headers
)
foreach(TGT ${GPUVMEM_INTERNAL_TARGETS})
  target_include_directories(${TGT} PUBLIC ${CASACORE_INCLUDE_DIRS}
                                           ${CFITSIO_INCLUDE_DIR}
                                           ${CMAKE_CURRENT_SOURCE_DIR}/include
                                           ${CMAKE_CURRENT_SOURCE_DIR}/include/classes
                                           ${CMAKE_CURRENT_SOURCE_DIR}/include/ms
                                           ${CMAKE_CURRENT_SOURCE_DIR}/include/fits
                                           ${CMAKE_CURRENT_SOURCE_DIR}/include/io
                                           ${CMAKE_CURRENT_SOURCE_DIR}/include/optimizers
                                           ${CMAKE_CURRENT_SOURCE_DIR}/include/linesearch
                                           ${CMAKE_CURRENT_SOURCE_DIR}/include/projection
                                           ${CMAKE_CURRENT_SOURCE_DIR}/include/regularizer_kernels
                                           ${CMAKE_CURRENT_SOURCE_DIR}/include/seeding
                                           ${CMAKE_CURRENT_SOURCE_DIR}/include/objective_function
                                           ${CMAKE_CURRENT_SOURCE_DIR}/include/beam
                                           ${CMAKE_CURRENT_SOURCE_DIR}/include/chi2
                                           ${CMAKE_CURRENT_SOURCE_DIR}/include/errors
                                           ${CMAKE_CURRENT_SOURCE_DIR}/include/fft
                                           ${CMAKE_CURRENT_SOURCE_DIR}/include/gridding
                                           ${CMAKE_CURRENT_SOURCE_DIR}/include/idft
                                           ${CMAKE_CURRENT_SOURCE_DIR}/include/measurement_operator
                                           ${CMAKE_CURRENT_SOURCE_DIR}/include/visibility
                                           ${CMAKE_CURRENT_SOURCE_DIR}/include/reduction
                                           ${CMAKE_CURRENT_SOURCE_DIR}/include/image_processing
                                           ${CMAKE_CURRENT_SOURCE_DIR}/include/utils
                                           ${CMAKE_CURRENT_SOURCE_DIR}/include/kernels
                                           ${CMAKE_CURRENT_SOURCE_DIR}/include/weighting
                                           ${CMAKE_CURRENT_SOURCE_DIR}/include/mfs
                                           ${CMAKE_CURRENT_SOURCE_DIR}/include/rng
                                           ${Boost_INCLUDE_DIRS}
                                           ${CUDAToolkit_LIBRARY_ROOT}/samples/common/inc)
endforeach()

# ============================================================================
# CUDA Architecture Detection and Configuration
# ============================================================================

# Allow manual override via -DCUDA_ARCH=<number>
if(DEFINED CUDA_ARCH)
  set(CUDA_ARCH_LIST ${CUDA_ARCH})
  message(
    STATUS "Using manually specified CUDA architecture: ${CUDA_ARCH_LIST}")
else()
  # Auto-detect GPU compute capability using nvidia-smi
  find_program(NVIDIA_SMI nvidia-smi)
  set(CUDA_ARCH_LIST "")

  if(NVIDIA_SMI)
    execute_process(
      COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
      OUTPUT_VARIABLE GPU_COMPUTE_CAP
      ERROR_VARIABLE NVIDIA_SMI_ERROR
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE NVIDIA_SMI_RESULT)

    if(NOT NVIDIA_SMI_RESULT EQUAL 0)
      message(
        WARNING
          "nvidia-smi failed (exit code ${NVIDIA_SMI_RESULT}): ${NVIDIA_SMI_ERROR}"
      )
    elseif(NOT GPU_COMPUTE_CAP)
      message(
        WARNING
          "nvidia-smi returned empty output. GPU may not be available or driver not loaded."
      )
    else()
      # Parse "X.Y" format to "XY" (e.g., "6.1" -> "61") First, split by
      # newlines (handles both single-GPU and multi-GPU systems) Single GPU:
      # "8.0\n" -> "8.0;" -> processes "80" and "" (empty skipped) Multi-GPU:
      # "8.0\n8.0\n8.0" -> "8.0;8.0;8.0;" -> processes all "80" values
      string(REPLACE "\n" ";" GPU_COMPUTE_CAP "${GPU_COMPUTE_CAP}")
      # Then replace dots and commas
      string(REPLACE "." "" GPU_COMPUTE_CAP "${GPU_COMPUTE_CAP}")
      string(REPLACE "," ";" GPU_COMPUTE_CAP "${GPU_COMPUTE_CAP}")

      foreach(CAP ${GPU_COMPUTE_CAP})
        string(STRIP "${CAP}" CAP)
        # Skip empty entries (from trailing newlines or blank lines)
        if(CAP
           AND CAP MATCHES "^[0-9]+$"
           AND NOT CAP STREQUAL "53")
          list(APPEND CUDA_ARCH_LIST "${CAP}")
        endif()
      endforeach()

      # Remove duplicates (for multi-GPU systems)
      if(CUDA_ARCH_LIST)
        list(REMOVE_DUPLICATES CUDA_ARCH_LIST)
      else()
        message(
          WARNING
            "nvidia-smi output parsed but no valid architectures found. Raw output: ${GPU_COMPUTE_CAP}"
        )
      endif()
    endif()
  else()
    message(
      WARNING
        "nvidia-smi not found in PATH. Cannot auto-detect CUDA architecture.")
  endif()

  # Fallback if auto-detection failed
  if(NOT CUDA_ARCH_LIST)
    # Check if we're in a CI/CD environment (GitHub Actions, GitLab CI, etc.)
    if(DEFINED ENV{CI}
       OR DEFINED ENV{GITHUB_ACTIONS}
       OR DEFINED ENV{GITLAB_CI})
      message(
        STATUS
          "CI/CD environment detected - using fallback architecture selection")

      # Query nvcc for supported architectures to use as fallback
      execute_process(
        COMMAND nvcc --list-gpu-arch
        OUTPUT_VARIABLE NVCC_FALLBACK_ARCHS
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)

      if(NVCC_FALLBACK_ARCHS)
        # Extract architecture numbers (e.g., "compute_75" -> "75")
        string(REGEX MATCHALL "compute_([0-9]+)" FALLBACK_MATCHES
                     "${NVCC_FALLBACK_ARCHS}")
        foreach(MATCH ${FALLBACK_MATCHES})
          string(REGEX REPLACE "compute_([0-9]+)" "\\1" ARCH_NUM "${MATCH}")
          if(ARCH_NUM AND NOT ARCH_NUM STREQUAL "53")
            list(APPEND CUDA_ARCH_LIST "${ARCH_NUM}")
          endif()
        endforeach()

        if(CUDA_ARCH_LIST)
          list(SORT CUDA_ARCH_LIST)
          list(GET CUDA_ARCH_LIST 0 CUDA_ARCH_LIST) # Use minimum supported
          set(CUDA_ARCH_LIST "${CUDA_ARCH_LIST}") # Ensure it's a list
          message(
            STATUS
              "Using minimum supported CUDA architecture: ${CUDA_ARCH_LIST} (from nvcc)"
          )
        endif()
      endif()

      # Final fallback: use common architecture (75 = Turing, widely supported)
      if(NOT CUDA_ARCH_LIST)
        set(CUDA_ARCH_LIST "75")
        message(
          STATUS "Using default CUDA architecture: 75 (Turing) for CI/CD build")
      endif()
    else()
      # Not in CI - require manual specification
      set(ERROR_MSG "Could not detect CUDA architecture.\n")

      if(NOT NVIDIA_SMI)
        string(APPEND ERROR_MSG "  - nvidia-smi not found in PATH\n")
      else()
        string(APPEND ERROR_MSG "  - nvidia-smi found but detection failed\n")
        if(DEFINED NVIDIA_SMI_ERROR)
          string(APPEND ERROR_MSG "  - Error: ${NVIDIA_SMI_ERROR}\n")
        endif()
        if(DEFINED GPU_COMPUTE_CAP)
          string(APPEND ERROR_MSG "  - Output: '${GPU_COMPUTE_CAP}'\n")
        endif()
      endif()

      string(APPEND ERROR_MSG "\n")
      string(APPEND ERROR_MSG "Solutions:\n")
      string(APPEND ERROR_MSG
             "  1. Specify manually: cmake -DCUDA_ARCH=<number> ..\n")
      string(APPEND ERROR_MSG "  2. Check GPU availability: nvidia-smi\n")
      string(
        APPEND
        ERROR_MSG
        "  3. Check compute capability: nvidia-smi --query-gpu=compute_cap --format=csv\n"
      )
      string(APPEND ERROR_MSG "\n")
      string(APPEND ERROR_MSG "Common architecture values:\n")
      string(APPEND ERROR_MSG
             "  - 60 (Pascal), 75 (Turing), 80 (Ampere), 89 (Ada), 90 (Hopper)")

      message(FATAL_ERROR "${ERROR_MSG}")
    endif()
  endif()
endif()

# Validate detected architectures
set(VALID_ARCH_LIST "")
foreach(ARCH ${CUDA_ARCH_LIST})
  string(STRIP "${ARCH}" ARCH)
  if(ARCH MATCHES "^[0-9]+$" AND NOT ARCH STREQUAL "53")
    list(APPEND VALID_ARCH_LIST "${ARCH}")
  endif()
endforeach()

if(NOT VALID_ARCH_LIST)
  message(FATAL_ERROR "No valid CUDA architectures found in: ${CUDA_ARCH_LIST}")
endif()
set(CUDA_ARCH_LIST "${VALID_ARCH_LIST}")

# Get CUDA version information
string(REGEX MATCH "^([0-9]+)" CUDA_VERSION_MAJOR "${CUDAToolkit_VERSION}")
message(
  STATUS "CUDA Version: ${CUDAToolkit_VERSION} (major: ${CUDA_VERSION_MAJOR})")

# Query nvcc for supported architectures
execute_process(
  COMMAND nvcc --list-gpu-arch
  OUTPUT_VARIABLE NVCC_SUPPORTED_ARCHS
  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)

# Extract supported architecture numbers from nvcc output
set(NVCC_SUPPORTED_ARCH_NUMS "")
if(NVCC_SUPPORTED_ARCHS)
  # Parse "compute_75", "compute_80", etc. to get numbers
  string(REGEX MATCHALL "compute_([0-9]+)" ARCH_MATCHES
               "${NVCC_SUPPORTED_ARCHS}")
  foreach(MATCH ${ARCH_MATCHES})
    string(REGEX REPLACE "compute_([0-9]+)" "\\1" ARCH_NUM "${MATCH}")
    list(APPEND NVCC_SUPPORTED_ARCH_NUMS "${ARCH_NUM}")
  endforeach()
  list(SORT NVCC_SUPPORTED_ARCH_NUMS)
  message(
    STATUS
      "CUDA ${CUDAToolkit_VERSION} supports architectures: ${NVCC_SUPPORTED_ARCH_NUMS}"
  )
endif()

# Auto-adjust architectures based on CUDA version support If detected GPU is
# older than CUDA supports, use minimum supported architecture
if(NVCC_SUPPORTED_ARCH_NUMS)
  set(ADJUSTED_ARCH_LIST "")
  set(UNSUPPORTED_ARCHS "")

  foreach(ARCH ${CUDA_ARCH_LIST})
    # Check if this architecture is directly supported
    list(FIND NVCC_SUPPORTED_ARCH_NUMS "${ARCH}" ARCH_FOUND)

    if(ARCH_FOUND GREATER_EQUAL 0)
      # Architecture is supported, use it
      list(APPEND ADJUSTED_ARCH_LIST "${ARCH}")
    else()
      # Architecture not supported - find minimum supported that's >= detected
      # This enables forward compatibility (e.g., compile for 7.5, run on 6.1
      # GPU)
      set(MIN_SUPPORTED "")
      foreach(SUPPORTED ${NVCC_SUPPORTED_ARCH_NUMS})
        if(SUPPORTED GREATER_EQUAL ARCH)
          if(NOT MIN_SUPPORTED OR SUPPORTED LESS MIN_SUPPORTED)
            set(MIN_SUPPORTED "${SUPPORTED}")
          endif()
        endif()
      endforeach()

      if(MIN_SUPPORTED)
        # Use minimum supported architecture (forward compatibility attempt)
        message(
          WARNING
            "GPU compute capability ${ARCH} is not supported by CUDA ${CUDAToolkit_VERSION}.\n"
            "Using minimum supported architecture ${MIN_SUPPORTED} instead.\n"
            "This may work via forward compatibility, but is not guaranteed.")
        list(APPEND ADJUSTED_ARCH_LIST "${MIN_SUPPORTED}")
      else()
        # GPU is too old - no supported architecture >= detected
        list(APPEND UNSUPPORTED_ARCHS "${ARCH}")
      endif()
    endif()
  endforeach()

  # Remove duplicates
  if(ADJUSTED_ARCH_LIST)
    list(REMOVE_DUPLICATES ADJUSTED_ARCH_LIST)
  endif()

  # Error if GPU is completely unsupported
  if(UNSUPPORTED_ARCHS)
    message(
      FATAL_ERROR
        "GPU compute capability ${UNSUPPORTED_ARCHS} is too old for CUDA ${CUDAToolkit_VERSION}.\n"
        "CUDA ${CUDAToolkit_VERSION} only supports: ${NVCC_SUPPORTED_ARCH_NUMS}\n\n"
        "Solutions:\n"
        "1. Install CUDA 11.x or earlier that supports compute_${UNSUPPORTED_ARCHS}\n"
        "2. Use a newer GPU\n"
        "3. Override with -DCUDA_ARCH=<supported_arch> (e.g., 75)")
  endif()

  # Update with adjusted architectures
  if(ADJUSTED_ARCH_LIST)
    set(CUDA_ARCH_LIST "${ADJUSTED_ARCH_LIST}")
    message(
      STATUS
        "Using architectures: ${CUDA_ARCH_LIST} (adjusted for CUDA ${CUDAToolkit_VERSION} compatibility)"
    )
  endif()
endif()

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
foreach(TGT ${GPUVMEM_ALL_TARGETS})
  set_target_properties(${TGT} PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
endforeach()

# For CUDA 12+, also add direct -arch=sm_X flags (CMake generates compute_X
# which nvcc rejects)
if(CUDA_VERSION_MAJOR GREATER_EQUAL 12)
  foreach(ARCH ${CUDA_ARCH_LIST})
    foreach(TGT ${GPUVMEM_ALL_TARGETS})
      target_compile_options(${TGT}
                             PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_${ARCH}>)
    endforeach()
  endforeach()
  message(
    STATUS "CUDA 12+ detected: Using -arch=sm_X flags (${CUDA_ARCH_LIST})")
else()
  message(STATUS "CUDA architectures: ${CUDA_ARCH_LIST}")
endif()

# ============================================================================
# CUDA Compilation Settings
# ============================================================================

# Set CUDA properties
set_target_properties(
  gpuvmem
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON # Enable separate compilation for
                                           # faster builds
             CUDA_STANDARD ${CMAKE_CUDA_STANDARD})
foreach(TGT ${GPUVMEM_INTERNAL_TARGETS})
  set_target_properties(${TGT} PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                          CUDA_STANDARD ${CMAKE_CUDA_STANDARD})
endforeach()

# Common CUDA flags (OpenMP support)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")

# Build-specific compilation flags
if(MEMORY_DEBUG)
  message(STATUS "Memory debugging: ON")
  foreach(TGT ${GPUVMEM_ALL_TARGETS})
    target_compile_options(${TGT} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G
                                           -D_FORCE_INLINES -w>)
  endforeach()
else()
  if(USE_FAST_MATH)
    message(STATUS "Fast math: ON")
    foreach(TGT ${GPUVMEM_ALL_TARGETS})
      target_compile_options(
        ${TGT} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-D_FORCE_INLINES -w
                        --use_fast_math -O3>)
    endforeach()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -O3")
  else()
    foreach(TGT ${GPUVMEM_ALL_TARGETS})
      target_compile_options(
        ${TGT} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-D_FORCE_INLINES -w -O3>)
    endforeach()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -O3")
  endif()
endif()

# ============================================================================
# Linking
# ============================================================================

# CUDA libraries
set(CUDA_LIBS ${CUDA_cuda_LIBRARY} ${CUDA_cudart_LIBRARY} ${CUDA_cufft_LIBRARY})

# External libraries
target_include_directories(gpuvmem PUBLIC ${CCfits_INCLUDE_DIRS})
foreach(TGT ${GPUVMEM_INTERNAL_TARGETS})
  target_include_directories(${TGT} PUBLIC ${CCfits_INCLUDE_DIRS})
endforeach()
set(EXTERNAL_LIBS ${CASACORE_LIBRARIES} ${CCfits_LIBRARIES} ${CFITSIO_LIBRARIES}
                  ${Boost_LIBRARIES})
message(STATUS "FITS I/O: CCfits (mandatory)")
message(STATUS "CCfits_LIBRARY=${CCfits_LIBRARY}")

# Propagate CCfits / CFITSIO / Casacore / Boost to anything that links our static libs
# (tests link a subset of gpuvmem_*_static without --whole-archive).
add_library(gpuvmem_external_deps INTERFACE)
target_link_libraries(gpuvmem_external_deps INTERFACE ${EXTERNAL_LIBS})
if(CCfits_LIBRARY)
  get_filename_component(CCfits_LIBRARY_DIR "${CCfits_LIBRARY}" DIRECTORY)
  target_link_directories(gpuvmem_external_deps INTERFACE "${CCfits_LIBRARY_DIR}")
  target_link_directories(gpuvmem PRIVATE "${CCfits_LIBRARY_DIR}")
endif()
foreach(TGT ${GPUVMEM_INTERNAL_TARGETS})
  target_link_libraries(${TGT} PUBLIC gpuvmem_external_deps)
endforeach()

# Link all libraries (CCfits and CFITSIO explicitly so host linker sees them when linking CUDA object files)
target_link_libraries(
  gpuvmem
  PRIVATE -Wl,--whole-archive
          ${GPUVMEM_INTERNAL_TARGETS}
          -Wl,--no-whole-archive
          m # Math library
          stdc++ # C++ standard library
          gomp # OpenMP (GNU implementation)
          ${CUDA_LIBS}
          ${EXTERNAL_LIBS})

# ============================================================================
# Installation (cmake --install . --prefix /path or CMAKE_INSTALL_PREFIX)
# ============================================================================
#
# Typical uses:
#   - End user: default ON for executable only → gpuvmem in prefix/bin
#   - Developer: -DGPUVMEM_INSTALL_STATIC_LIBS=ON -DGPUVMEM_INSTALL_HEADERS=ON
#     then link installed .a files and -I prefix/include/gpuvmem
#
if(GPUVMEM_INSTALL_EXECUTABLE)
  install(TARGETS gpuvmem RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(GPUVMEM_INSTALL_STATIC_LIBS)
  install(
    TARGETS ${GPUVMEM_INTERNAL_TARGETS}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

if(GPUVMEM_INSTALL_HEADERS)
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gpuvmem)
endif()

# ============================================================================
# Testing
# ============================================================================

enable_testing()

if(GPUVMEM_BUILD_TESTS)
  add_subdirectory(tests)
endif()

set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests)

# End-to-end dataset tests: each test.sh runs several scenarios (see tests/common/e2e_lib.sh).
# Override iteration budget: GPUVMEM_E2E_MAX_ITER=25 ctest -L e2e
set(TEST_CASES antennae co65 FREQ78 M87 selfcalband9)

foreach(TEST_NAME ${TEST_CASES})
  add_test(
    NAME "e2e_${TEST_NAME}"
    COMMAND bash ${TEST_DIR}/${TEST_NAME}/test.sh ${BINARY_DIR}/gpuvmem
            ${TEST_DIR}/${TEST_NAME})
  set_tests_properties("e2e_${TEST_NAME}" PROPERTIES LABELS "e2e")
endforeach()

# ============================================================================
# Build Summary
# ============================================================================

message(STATUS "")
message(STATUS "=== Build Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA path: ${CUDAToolkit_LIBRARY_ROOT}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Runtime output directory: ${BINARY_DIR}")
message(STATUS "Static library output directory: ${LIB_DIR}")
message(STATUS "Memory debug: ${MEMORY_DEBUG}")
message(STATUS "Fast math: ${USE_FAST_MATH}")
message(STATUS "Install executable: ${GPUVMEM_INSTALL_EXECUTABLE}")
message(STATUS "Install static libraries: ${GPUVMEM_INSTALL_STATIC_LIBS}")
message(STATUS "Install headers: ${GPUVMEM_INSTALL_HEADERS}")
message(STATUS "Build unit tests: ${GPUVMEM_BUILD_TESTS}")
message(STATUS "===========================")
message(STATUS "")
