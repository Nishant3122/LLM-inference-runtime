# Adaptive LLM Inference Runtime — top-level build.
#
# Verified 2026-08-12 with CMake 4.4.2 + MSVC 19.44 (VS 2022 Build Tools):
#   cmake -S . -B build -G "Visual Studio 17 2022" -A x64
#   cmake --build build --config RelWithDebInfo
#   ctest --test-dir build -C RelWithDebInfo --output-on-failure
# CUDA is off by default and needs the CUDA Toolkit plus an NVIDIA GPU (BUILD_CUDA=ON)
# — not available on this machine yet, see docs/architecture.md §10.
cmake_minimum_required(VERSION 3.18)
project(adaptive_llm_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

option(BUILD_CUDA "Build the CUDA backend (requires CUDA Toolkit + NVIDIA GPU)" OFF)
option(BUILD_TESTS "Build the tests/ tree" ON)
option(BUILD_EXAMPLES "Build the examples/ tree" ON)

if(BUILD_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_subdirectory(runtime)
add_subdirectory(tools/tokenizer)

if(BUILD_CUDA)
    add_subdirectory(cuda)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
