include(${CMAKE_SOURCE_DIR}/cmake/GpuvmemGTest.cmake)
gpuvmem_gtest_fetch()

add_subdirectory(unit)
add_subdirectory(integration)
