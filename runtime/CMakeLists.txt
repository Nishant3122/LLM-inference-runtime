# Only `core/` has real code as of Phase 0. The other subdirectories (model/,
# transformer/, execution/, cache/, quantization/, scheduler/, sampling/) exist as
# placeholders — see the README.md in each — and get their own CMake targets as they
# gain real sources in Phase 1+. Deliberately not pre-declaring empty library targets
# for them now (Engineering Principle 4: avoid premature abstraction).
add_library(runtime_core STATIC
    core/tensor.h
    core/shape.h
    core/types.h
    core/version.h
    core/version.cpp
)
target_include_directories(runtime_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
