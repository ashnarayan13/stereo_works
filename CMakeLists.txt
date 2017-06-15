cmake_minimum_required(VERSION 2.8.3)
project(stereo)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(OpenCV REQUIRED
NO_MODULE # should be optional, tells CMake to use config mode
PATHS /usr/local/include # look here
NO_DEFAULT_PATH) # and don't look anywhere else

find_package( CUDA REQUIRED )
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  message_filters
  roscpp
  rospy
  sensor_msgs
  std_msgs
  pcl_conversions
  pcl_ros roscpp
  sensor_msgs
  visualization_msgs
)


catkin_package(
)



###########
## Build ##
###########

#include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(
  ${catkin_INCLUDE_DIRS}
)



## Declare a C++ executable
#add_executable(stereo_node src/stereo_node.cpp)
add_executable(stereo_node_cam src/stereo_node_cam.cpp)
add_executable(stereo_node_recon src/stereo_node_recon.cpp)
add_executable(stereo_node_kitti src/stereo_node_kitti.cpp)
add_executable(flir_node src/flir_node.cpp)
add_executable(capture_images src/capture_images.cpp)


## Specify libraries to link a library or executable target against
#target_link_libraries(stereo_node ${catkin_LIBRARIES})
target_link_libraries(stereo_node_cam ${catkin_LIBRARIES})
target_link_libraries(stereo_node_recon ${catkin_LIBRARIES})
target_link_libraries(stereo_node_kitti ${catkin_LIBRARIES})
target_link_libraries(flir_node ${catkin_LIBRARIES} )
target_link_libraries(capture_images ${catkin_LIBRARIES})


