
enable_testing()
find_package(CUDAToolkit REQUIRED)

add_library(testutils STATIC
  utils.cu
)

target_link_libraries(testutils PRIVATE
  inf_core
)

add_custom_target(run_tests ALL
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS test_bf16_gemm
)

add_subdirectory(kernels)
