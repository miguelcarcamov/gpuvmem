# FindCCfits.cmake - Find the CCfits C++ FITS library (HEASARC)
#
# CCfits is a C++ wrapper around CFITSIO. It requires CFITSIO to be found first.
#
# Variables used by this module:
#   CCfits_ROOT_DIR  - root directory of CCfits installation
#   CFITSIO_ROOT_DIR - (optional) if CCfits is under same prefix as CFITSIO
#
# Variables defined by this module:
#   CCfits_FOUND         - true if CCfits was found
#   CCfits_INCLUDE_DIR   - include directory (for #include <CCfits/FITS.h>)
#   CCfits_INCLUDE_DIRS  - same as CCfits_INCLUDE_DIR
#   CCfits_LIBRARY       - the CCfits library
#   CCfits_LIBRARIES     - CCfits + CFITSIO (link with both; CCfits wraps CFITSIO)
#
# Usage: find_package(CFITSIO REQUIRED) before find_package(CCfits REQUIRED)
#        then target_link_libraries(your_target ${CCfits_LIBRARIES})

if(NOT CCfits_FOUND)

  # Look for CCfits/FITS.h (standard install has include/CCfits/FITS.h)
  find_path(CCfits_INCLUDE_DIR CCfits/FITS.h
    HINTS ${CCfits_ROOT_DIR} ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES include)

  # Library name is typically CCfits (libCCfits.so or libCCfits.a)
  find_library(CCfits_LIBRARY CCfits
    HINTS ${CCfits_ROOT_DIR} ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES lib lib64)

  mark_as_advanced(CCfits_INCLUDE_DIR CCfits_LIBRARY)

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(CCfits DEFAULT_MSG CCfits_LIBRARY CCfits_INCLUDE_DIR)

  if(CCfits_FOUND)
    set(CCfits_INCLUDE_DIRS ${CCfits_INCLUDE_DIR})
    # Caller must link CFITSIO as well (CCfits depends on it). We don't add it here
    # so that the project can already have found CFITSIO; we set CCfits_LIBRARIES
    # to CCfits only. CMakeLists should link: target_link_libraries(... CCfits ${CFITSIO_LIBRARIES})
    # or we add CFITSIO to CCfits_LIBRARIES if CFITSIO is found.
    set(CCfits_LIBRARIES ${CCfits_LIBRARY})
  endif()

endif(NOT CCfits_FOUND)
