gpuvmem_add_unit_test(
  gpuvmem_unit_cli_observer
  SOURCES
  test_cli_observer.cc
  ${CMAKE_SOURCE_DIR}/src/cli/console_run_observer.cc
  ${CMAKE_SOURCE_DIR}/src/cli/run_observer_factory.cc
  ${CMAKE_SOURCE_DIR}/src/cli/final_metrics_format.cc)

add_executable(gpuvmem_unit_cli_parse test_cli_parse.cc
                                       ${CMAKE_SOURCE_DIR}/src/options.cu)
set_source_files_properties(${CMAKE_SOURCE_DIR}/src/options.cu PROPERTIES LANGUAGE
                                                                         CUDA)
gpuvmem_gtest_common_target_props(gpuvmem_unit_cli_parse)
target_link_libraries(gpuvmem_unit_cli_parse PRIVATE ${CUDA_LIBS})
gpuvmem_gtest_discover(gpuvmem_unit_cli_parse unit)
