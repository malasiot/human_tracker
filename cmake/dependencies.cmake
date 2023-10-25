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
find_package(Torch REQUIRED)
find_package(Ceres REQUIRED)
#${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}

find_library(NPPIF_LIBRARY nppif ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(NPPC_LIBRARY nppc ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

find_package(cvx REQUIRED)
find_package(xviz REQUIRED)
find_package(gfx REQUIRED)
find_package(OpenPose REQUIRED)



include_directories(${EXTERNAL_INSTALL_LOCATION}/include ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${OpenPose_INCLUDE_DIRS} pybind11)
