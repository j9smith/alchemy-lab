cmake_minimum_required(VERSION 3.22)
project(alchemy-runtime LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(httplib CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_path(TRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include/x86_64-linux-gnu
)
find_library(TRT_LIB nvinfer
    HINTS /usr/lib/x86_64-linux-gnu
)
find_library(TRT_PLUGIN_LIB nvinfer_plugin
    HINTS /usr/lib/x86_64-linux-gnu
)

if(NOT TRT_INCLUDE_DIR OR NOT TRT_LIB)
    message(FATAL_ERROR "TensorRT not found")
endif()

add_executable(alchemy-runtime
    src/main.cpp
    src/batch_queue.cpp
    src/infer.cpp
    src/engine.cpp
    src/pipeline.cpp
    src/ddpm_scheduler.cu
    src/utils.cu
)

target_link_libraries(alchemy-runtime PRIVATE
    httplib::httplib
    ${TRT_LIB}
    ${TRT_PLUGIN_LIB}
    CUDA::cudart
    CUDA::curand
)

target_include_directories(alchemy-runtime PRIVATE
    ${TRT_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    src
)