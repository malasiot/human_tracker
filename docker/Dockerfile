FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

CMD ["/bin/bash"]

USER 0

# Set up time zone.

ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

RUN \
 apt-get update && apt-get install -y \
  ssh cmake make build-essential git-core \
  libopencv-dev python3-opencv libeigen3-dev libf2c2-dev libblas-dev liblapack-dev libsqlite3-dev\
  libassimp-dev libglu1-mesa-dev libfreetype-dev libfontconfig1-dev libharfbuzz-dev qtbase5-dev

RUN git clone 'https://github.com/malasiot/cvx.git';cd cvx; mkdir build; cd build; cmake ..; make -j8 install ; cd ../.. ; rm -rf cvx ;
RUN git clone 'https://github.com/malasiot/xviz.git';cd xviz; mkdir build; cd build; cmake ..; make -j8 install ; cd ../.. ; rm -rf xviz ;


# torch
RUN apt-get install -y unzip
RUN cd /tmp/ ; wget 'https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip' ; unzip  libtorch-cxx11-abi-shared-with-deps-2.0.1+cu118.zip -d /usr/local/

# ceres
RUN apt-get install -y libatlas-base-dev libsuitesparse-dev libgflags-dev libgoogle-glog-dev
RUN cd /tmp/ ; git clone --branch 2.1.0 https://ceres-solver.googlesource.com/ceres-solver; cd ceres-solver ; mkdir build ; cd build ; cmake -DBUILD_SHARED_LIBS=On -DBUILD_EXAMPLES=Off -DUSE_CUDA=On .. ; make -j8 install ; cd ../../ ; rm -rf ceres-solver;

# openpose 
RUN apt-get install -y libboost-all-dev protobuf-compiler libhdf5-dev libgoogle-glog-dev
RUN cd /tmp/ ; git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose ; cd openpose ; mkdir build ; cd build ; cmake -DDOWNLOAD_BODY_25_MODEL=Off -DDOWNLOAD_HAND_MODEL=Off -DDOWNLOAD_FACE_MODEL=Off .. ; make -j8 ; make install ;

#realsense2
RUN apt-get install -y curl apt-transport-https
RUN mkdir -p /etc/apt/keyrings
RUN curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | tee /etc/apt/keyrings/librealsense.pgp > /dev/null
RUN echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | tee /etc/apt/sources.list.d/librealsense.list ; apt-get update
RUN apt-get install -y librealsense2-dev

# human tracker
RUN apt-get install -y libfaiss-dev

RUN apt-get install -y qtcreator

# ROS2

RUN apt update && apt install locales
RUN locale-gen en_US en_US.UTF-8
RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
RUN LANG=en_US.UTF-8

RUN apt install -y software-properties-common
RUN add-apt-repository universe

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN apt update 
RUN apt install -y ros-humble-desktop ros-dev-tools
