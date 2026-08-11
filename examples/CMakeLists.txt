add_executable(generate_example generate.cpp)
target_link_libraries(generate_example PRIVATE runtime tokenizer)

add_executable(benchmark_example benchmark.cpp)
target_link_libraries(benchmark_example PRIVATE runtime)
