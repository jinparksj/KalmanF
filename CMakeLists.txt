cmake_minimum_required(VERSION 3.13)
project(kf_img_track)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(kf_img_track meanShift_CVKF.cpp)
#01_random_const_estimate_offline.cpp
#02_random_const_estimate_online.cpp
#03_random_const_estimate_offline_cvkf.cpp
#04_random_const_estimate_online_cvkf.cpp
#Motion_Falling_Body_CVKF.cpp
#meanShift_CVKF.cpp

target_link_libraries(kf_img_track ${OpenCV_LIBS})
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath='$ORIGIN/lib'")

