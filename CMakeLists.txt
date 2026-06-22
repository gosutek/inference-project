cmake_minimum_required(VERSION 3.24)
set(CMAKE_CUDA_COMPILER "/opt/cuda/bin/nvcc")

project(inf LANGUAGES C CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(CUDAToolkit REQUIRED)

add_executable(inf
  src/main.cu
  src/allocator.c
  src/tokenizer.c

  external/cJSON.c

  src/cuda/api/cuda_allocator.cu
  src/cuda/api/cuda_mem_wrapper.cu

  src/cuda/kernels/input_embeddings.cu
)

target_compile_options(inf PRIVATE
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
    -gencode arch=compute_89,code=sm_89
    --threads 2
  >
)

target_link_libraries(inf PRIVATE CUDA::cudart CUDA::cuda_driver)

target_include_directories(inf PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/api
  ${CMAKE_CURRENT_SOURCE_DIR}/external
)
