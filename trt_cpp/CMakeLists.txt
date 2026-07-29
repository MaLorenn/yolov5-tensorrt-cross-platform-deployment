cmake_minimum_required(VERSION 3.10)

project(yolov5_trt_cpp)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    /usr/include/aarch64-linux-gnu
    /usr/local/cuda/include
)

link_directories(
    /usr/lib/aarch64-linux-gnu
    /usr/local/cuda/lib64
)

add_executable(
    yolov5_trt
    src/main.cpp
)

target_link_libraries(
    yolov5_trt
    nvinfer
    cudart
    ${OpenCV_LIBS}
)

add_executable(
    yolov5_trt_multithread
    src/main_multithread.cpp
)

target_link_libraries(
    yolov5_trt_multithread
    nvinfer
    cudart
    pthread
    ${OpenCV_LIBS}
)