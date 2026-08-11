add_library(tokenizer STATIC tokenizer.h tokenizer.cpp)
target_include_directories(tokenizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
