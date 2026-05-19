gpuvmem_add_integration_test(
  gpuvmem_integration_cli
  SOURCES
  test_cli_parse_observer.cc
  ${CMAKE_SOURCE_DIR}/src/options.cu
  ${CMAKE_SOURCE_DIR}/src/cli/console_run_observer.cc
  ${CMAKE_SOURCE_DIR}/src/cli/run_observer_factory.cc
  ${CMAKE_SOURCE_DIR}/src/cli/final_metrics_format.cc)
set_source_files_properties(${CMAKE_SOURCE_DIR}/src/options.cu PROPERTIES LANGUAGE
                                                                         CUDA)
target_link_libraries(gpuvmem_integration_cli PRIVATE ${CUDA_LIBS})
