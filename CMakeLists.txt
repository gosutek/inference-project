cmake_minimum_required(VERSION 3.24)

project(inf LANGUAGES C CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)

add_library(inf_core STATIC
  src/allocator.c
  src/tokenizer.c

  external/cJSON.c

  src/cuda/api/cuda_allocator.cu
  src/cuda/api/cuda_mem_wrapper.cu

  src/cuda/kernels/input_embeddings.cu
  src/cuda/kernels/rmsnorm.cu
  src/cuda/kernels/gemm.cu
)

target_include_directories(inf_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/api
  ${CMAKE_CURRENT_SOURCE_DIR}/external
)

target_compile_options(inf_core PUBLIC
  $<$<COMPILE_LANGUAGE:C>:
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wsign-conversion
    -Wpointer-arith
    -Wstrict-prototypes
    -Wmissing-prototypes
    -Wold-style-definition
    -Wshadow
    -Wcast-align
    -Wcast-qual
    -Wwrite-strings
    -Wformat=2
    -Wundef
    -Wswitch-enum
    >
  $<$<COMPILE_LANGUAGE:CUDA>:
    -Xcompiler=-Wall,-Wextra
    --threads 2
  >
)

target_link_libraries(inf_core PUBLIC CUDA::cudart CUDA::cuda_driver)

add_executable(inf
  src/main.cu
)

target_link_libraries(inf PRIVATE inf_core)
