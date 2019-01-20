cmake_minimum_required(VERSION 3.13)
project(kf_img_track)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(kf_img_track test1_random_const_estimate_offline.cpp)

target_link_libraries(kf_img_track ${OpenCV_LIBS})