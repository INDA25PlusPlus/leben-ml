cmake_minimum_required(VERSION 4.0)

# options
option(ENABLE_CUDA_FEATURES "Uses the CUDA pipeline to replace some processing-intensive parts with GPU processing")
if (ENABLE_CUDA_FEATURES)
    add_compile_definitions(ENABLE_CUDA_FEATURES=true)
else()
    add_compile_definitions(ENABLE_CUDA_FEATURES=false)
endif()

# languages
list(APPEND languages CXX)
set(CMAKE_CXX_STANDARD 20)
if(ENABLE_CUDA_FEATURES)
    list(APPEND languages CUDA)
    set(CMAKE_CUDA_STANDARD 20)
endif()

project(leben_ml LANGUAGES ${languages})

# sources
file(GLOB_RECURSE common_sources src/common/*)
file(GLOB_RECURSE cpu_only_sources src/cpu_only/*)
file(GLOB_RECURSE cuda_sources src/cuda/*)

# targets
if(ENABLE_CUDA_FEATURES)
    add_executable(leben_ml_cuda src/main.cpp ${common_sources} ${cuda_sources})
    add_executable(leben_ml_cuda_test src/test.cpp ${common_sources} ${cuda_sources})
    set_target_properties(leben_ml_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
    add_executable(leben_ml_cpu_only src/main.cpp ${common_sources} ${cpu_only_sources})
    add_executable(leben_ml_cpu_only_test src/test.cpp ${common_sources} ${cpu_only_sources})
endif()
