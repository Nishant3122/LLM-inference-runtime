# Each test file is its own executable with a plain main() (see tensor_test.cpp for
# why there's no external test framework dependency). Add a new add_executable/add_test
# pair here as each test file in docs/architecture.md's phase table gets written.
add_executable(tensor_test tensor_test.cpp)
target_link_libraries(tensor_test PRIVATE runtime)
add_test(NAME tensor_test COMMAND tensor_test)

add_executable(tokenizer_test tokenizer_test.cpp)
target_link_libraries(tokenizer_test PRIVATE tokenizer)
add_test(NAME tokenizer_test COMMAND tokenizer_test ${CMAKE_SOURCE_DIR}/models/vocab.bin)

add_executable(model_test model_test.cpp)
target_link_libraries(model_test PRIVATE runtime)
add_test(NAME model_test COMMAND model_test ${CMAKE_SOURCE_DIR}/models/model.bin ${CMAKE_SOURCE_DIR}/tests/golden)

add_executable(kv_cache_test kv_cache_test.cpp)
target_link_libraries(kv_cache_test PRIVATE runtime)
add_test(NAME kv_cache_test COMMAND kv_cache_test ${CMAKE_SOURCE_DIR}/models/model.bin)

if(BUILD_CUDA)
    add_executable(cuda_ops_test cuda_ops_test.cu)
    target_link_libraries(cuda_ops_test PRIVATE runtime cuda_kernels)
    add_test(NAME cuda_ops_test COMMAND cuda_ops_test ${CMAKE_SOURCE_DIR}/models/model.bin)
endif()
