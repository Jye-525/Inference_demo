cmake_minimum_required(VERSION 3.20)
project(inference_demo
        VERSION 1.0
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
find_package( OpenCV REQUIRED )

find_library(TENSORFLOW_LIB tensorflow HINT libtensorflow_gpu_c/lib)

add_executable(inference main.cpp cppflow/src/Model.cpp cppflow/src/Tensor.cpp)
target_include_directories(inference PRIVATE cppflow/include libtensorflow_gpu_c/include)
target_link_libraries (inference ${TENSORFLOW_LIB} ${OpenCV_LIBS} stdc++fs)
