### Human Tracker

Real time human pose estimation and tracking from RGBD camera stream.

### Description

Implementation of an algorithm for fitting an articulated 3D human model on a 3D point cloud.

### Functionality

The provided ROS node will subscribe to color and depth image topics provided by an RGBD camera node (e.g. Primesense, Kinect). The color image is used for body part estimation using OpenPose library. This is used to initialize the tracker with pose and safe working area. For each depth frame the tracker estimates 3D human pose which is then published with a custom message.

### Technical Specifications

- C++, CUDA

- CUDA (Tested on CUDA 11.8)

- CUDA GPU enabled hardware (tested on NVidia GForce RTX 3060, GTX 1070)

### Design


### Use Cases


### Testing


### Limitations


