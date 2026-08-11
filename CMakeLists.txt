# Adaptive LLM Inference Runtime — top-level build.
#
# NOTE (recorded docs/architecture.md §10): this file has not been build-tested on the
# primary dev machine, which currently has no CMake and no C++ compiler installed.
# Install CMake >= 3.18 and a C++17/20 toolchain (MSVC Build Tools or MinGW-w64 on
# Windows) before relying on this. CUDA is off by default and needs the CUDA Toolkit
# plus an NVIDIA GPU (BUILD_CUDA=ON).
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
