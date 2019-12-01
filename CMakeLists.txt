cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(nbody LANGUAGES CXX CUDA)
set(CMAKE_CXX_FLAGS "-O0 -g")

file(GLOB SOURCES "src/*.cpp" "include/*.hpp" "src/*.cu" "include/*.cuh")
add_library(nbody SHARED ${SOURCES})
target_compile_features(nbody PUBLIC cxx_std_11)
target_include_directories(nbody PRIVATE include)
set_target_properties(nbody PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(nbody_test nbody_test.cpp)
set_target_properties(nbody_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(nbody_test PRIVATE nbody)
target_include_directories(nbody_test PRIVATE include)

target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -arch=sm_70
                       >)

target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -G
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       >)

