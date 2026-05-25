cmake_minimum_required(VERSION 3.24)
set(CMAKE_CUDA_COMPILER "/opt/cuda/bin/nvcc")

project(scratch-inf LANGUAGES C CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_C_FLAGS_RELEASE "-O2")
set(CMAKE_CUDA_FLAGS_RELEASE "-O2")

set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -DDEBUG")
set(CMAKE_C_FLAGS_DEBUG "-g -DDEBUG")

find_package(CUDAToolkit REQUIRED)

add_executable(scratch-inference
  src/main.c
  external/cJSON.c
)

target_include_directories(scratch-inference PRIVATE src)
target_include_directories(scratch-inference PRIVATE include)
target_include_directories(scratch-inference PRIVATE external)
