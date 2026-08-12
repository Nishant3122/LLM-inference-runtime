# Phase 1: renamed from `runtime_core` (Phase 0) to `runtime` now that this target
# covers core + model + transformer + execution + sampling, not just the Tensor
# abstraction. Phase 2 adds cache/. Phase 3 conditionally adds execution/cuda_backend
# (below, when BUILD_CUDA=ON). quantization/, scheduler/ still have no sources
# (Phase 5/6+) — see the README.md in each.
add_library(runtime STATIC
    core/tensor.h
    core/shape.h
    core/types.h
    core/version.h
    core/version.cpp

    cache/kv_cache.h
    cache/kv_cache.cpp
    cache/cache_manager.h
    cache/cache_manager.cpp

    model/model_config.h
    model/model.h
    model/model_loader.h
    model/model_loader.cpp

    transformer/ops.h
    transformer/ops.cpp
    transformer/normalization.h
    transformer/normalization.cpp
    transformer/embedding.h
    transformer/embedding.cpp
    transformer/attention.h
    transformer/attention.cpp
    transformer/mlp.h
    transformer/mlp.cpp
    transformer/transformer_block.h
    transformer/transformer_block.cpp

    execution/cpu_backend.h
    execution/cpu_backend.cpp

    sampling/greedy.h
    sampling/greedy.cpp
    sampling/temperature.h
    sampling/temperature.cpp
    sampling/top_k.h
    sampling/top_k.cpp
)
target_include_directories(runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(BUILD_CUDA)
    target_sources(runtime PRIVATE execution/cuda_backend.h execution/cuda_backend.cu)
    target_link_libraries(runtime PUBLIC cuda_kernels)
endif()
