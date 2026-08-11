# Each test file is its own executable with a plain main() (see tensor_test.cpp for
# why there's no external test framework dependency). Add a new add_executable/add_test
# pair here as each test file in docs/architecture.md's phase table gets written.
add_executable(tensor_test tensor_test.cpp)
target_link_libraries(tensor_test PRIVATE runtime_core)
add_test(NAME tensor_test COMMAND tensor_test)
