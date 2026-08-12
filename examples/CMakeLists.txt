add_executable(generate_example generate.cpp)
target_link_libraries(generate_example PRIVATE runtime tokenizer)

add_executable(benchmark_example benchmark.cpp)
target_link_libraries(benchmark_example PRIVATE runtime)

if(BUILD_CUDA)
    add_executable(benchmark_cuda benchmark_cuda.cu)
    target_link_libraries(benchmark_cuda PRIVATE runtime cuda_kernels)
endif()
