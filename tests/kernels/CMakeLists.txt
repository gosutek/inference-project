add_executable(test_bf16_gemm
  test_bf16_gemm.cu
)

target_link_libraries(test_bf16_gemm
  PRIVATE
  testutils
  inf_core
  CUDA::cublas
)

target_include_directories(test_bf16_gemm
  PRIVATE
  ${CMAKE_SOURCE_DIR}/tests
)

add_test(NAME bf16_gemm_correctness COMMAND test_bf16_gemm)
