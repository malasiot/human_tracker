set(HTRAC_POSE_SRC
    modelling/skeleton.cpp
    modelling/skinned_mesh.cpp
    modelling/fit_skeleton.cpp
    modelling/primitive_mesh.cpp
    modelling/parameterization.cpp
    modelling/pose.cpp

    tracking/energy_term.cpp
    tracking/image_to_model_term.cpp
    tracking/image_to_model_term_gpu.cu
    tracking/model_to_image_term.cpp
    tracking/model_to_image_term_gpu.cu
    tracking/keypoints_2d_term.cpp
    tracking/collision_term.cpp
    tracking/collision_term_gpu.cu
    tracking/collision_data.cpp
    tracking/joint_limits_term.cpp
    tracking/keypoints_term.cpp
    tracking/keypoint_distance_field.cpp
    tracking/pose_from_keypoints.cpp
    tracking/context_gpu.cu
    tracking/model_fit.cu
    tracking/context.cpp
    tracking/sdf_model.cpp
    tracking/odf.cpp
    tracking/distance_transform.cpp
    tracking/primitive_sdf_gpu.cu

    openpose/keypoint_detector.cpp
    openpose/keypoint_detector_openpose.cpp

    datasets/itop_dataset.cpp
    datasets/certh_dataset.cpp

    util/pose_database.cpp
    util/mhx2_importer.cpp
    util/io_util.cpp
    util/pcl_util.cpp
    util/skeleton.cpp
    util/mesh.cpp
    util/depth_provider.cpp
)

add_library(htrac_pose SHARED ${HTRAC_POSE_SRC})
set_target_properties(htrac_pose PROPERTIES
                            CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(htrac_pose PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                            --expt-relaxed-constexpr
                            --extended-lambda
                            >)

target_link_libraries(htrac_pose "${TORCH_LIBRARIES}" ${OpenCV_LIBS}  cvx ceres assimp OpenMP::OpenMP_CXX
    ${NPPIF_LIBRARY} ${NPPC_LIBRARY}
    openpose gflags)


#add_executable(test_fit2 tracking/sandbox/test_fit2.cpp)
#target_link_libraries(test_fit2 htrac_pose xviz)

add_definitions(-DUSE_CUDNN)
add_executable(test_openpose openpose/sandbox/test_openpose.cpp)
target_link_libraries(test_openpose PRIVATE htrac_pose
#    pybind11::embed
)

find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(image_geometry REQUIRED)
find_package(human_tracker_msgs REQUIRED)

add_executable(test_camera_reader ros/test_camera_reader.cpp ros/camera_reader.cpp)
ament_target_dependencies(test_camera_reader rclcpp sensor_msgs std_msgs cv_bridge message_filters)
target_link_libraries(test_camera_reader htrac_pose cvx)

add_executable(openpose_tracker ros/openpose_tracker.cpp)
ament_target_dependencies(openpose_tracker
    rclcpp
    sensor_msgs
    std_msgs
    cv_bridge
    message_filters
    visualization_msgs
    tf2_eigen
    image_geometry
    human_tracker_msgs)

target_link_libraries(openpose_tracker htrac_pose cvx)

install(TARGETS
  test_camera_reader
  openpose_tracker
  DESTINATION lib/${PROJECT_NAME})



