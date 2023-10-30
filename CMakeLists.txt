cmake_minimum_required(VERSION 3.16)

project(ros_human_tracker LANGUAGES CXX CUDA)


set(CMAKE_MODULE_PATH ${CMAKE_ROOT}/Modules/ ${CMAKE_SOURCE_DIR}/cmake/ )

include(setup)

include(dependencies)

find_package(ament_cmake REQUIRED)

set(HTRAC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CMAKE_LINK_WHAT_YOU_USE TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")



include_directories(${HTRAC_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${ASSIMP_INCLUDE_DIR} src/3rdparty ${catkin_INCLUDE_DIRS})


#add_subdirectory(pybind11)
#pybind11_add_module(htrac src/openpose/sandbox/test_openpose.cpp)

add_subdirectory(src)

ament_package()
