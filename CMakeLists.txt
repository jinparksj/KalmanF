cmake_minimum_required(VERSION 3.13)
project(kf_img_track)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(kf_img_track 01_random_const_estimate_offline.cpp)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath='$ORIGIN/lib'")

target_link_libraries(kf_img_track ${OpenCV_LIBS})