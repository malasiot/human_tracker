include(ExternalProject)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc highgui imgcodecs)
find_package(Eigen3 3.3 REQUIRED)
find_package(OpenMP)
find_package(assimp REQUIRED)
# install libtorch and cudnn
# export CUDA_NVCC_EXECUTABLE=/usr/local/cuda-11.7/bin/nvcc
# export CUDA_BIN_PATH=/usr/local/cuda-11.7/
# export CUDA_INC_PATH=/usr/local/cuda-11.7/
# cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PATH_PREFIX=/usr/local/ -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda/ -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc -DCUDA_SDK_ROOT_DIR=/usr/local/cuda/ ../..
# call cmake with flags -DCMAKE_PREFIX_PATH=<libtorch_root>  -DCUDA_TOOLKIT_ROOT_DIR=<cuda_root> -DCMAKE_CUDA_COMPILER=<cuda_nvcc>
#find_package(Torch REQUIRED)
find_package(Ceres REQUIRED)
#${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}

find_library(NPPIF_LIBRARY nppif ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(NPPC_LIBRARY nppc ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

#set(CVX_ROOT ${CMAKE_BINARY_DIR}/3rdparty/cvx)
#set(CVX_LIB_DIR ${CVX_ROOT}/bin/lib)
#set(CVX_INCLUDE_DIR ${CVX_ROOT}/bin/include)

#ExternalProject_Add(
#  cvx_external
#  GIT_REPOSITORY "https://github.com/malasiot/cvx.git"
#  GIT_TAG "master"
#  UPDATE_COMMAND ""
#  PATCH_COMMAND ""
#  BINARY_DIR ${CVX_ROOT}/src/cvx
#  SOURCE_DIR ${CVX_ROOT}/src/cvx
#  INSTALL_DIR ${CVX_ROOT}/bin
#  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
#)

#add_library(cvx SHARED IMPORTED)
#set_target_properties(cvx PROPERTIES IMPORTED_LOCATION ${CVX_LIB_DIR}/libcvx.so)
#add_dependencies(cvx cvx_external)
find_package(cvx REQUIRED)
#find_package(xviz REQUIRED)
#find_package(gfx REQUIRED)
find_package(OpenPose REQUIRED)


include_directories(${EXTERNAL_INSTALL_LOCATION}/include ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${OpenPose_INCLUDE_DIRS} pybind11)
