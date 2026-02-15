cmake_minimum_required(VERSION 3.8)
project(yolo_ros)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(message_filters REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(include)

add_library(${PROJECT_NAME} SHARED
  src/yolo_detect_node.cpp
  src/detector.cpp
  src/position_estimator.cpp
  src/detection_tracker.cpp
  src/camera_manager.cpp
  src/visualizer.cpp
)
ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  rclcpp_lifecycle
  rclcpp_components
  sensor_msgs
  vision_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  cv_bridge
  image_transport
  message_filters
  OpenCV
)

add_executable(yolo_detect_auto_activate src/main.cpp)
target_link_libraries(yolo_detect_auto_activate ${PROJECT_NAME})
ament_target_dependencies(yolo_detect_auto_activate
  rclcpp
  rclcpp_lifecycle
)

add_executable(batch_processor src/batch_processor.cpp)
target_link_libraries(batch_processor ${PROJECT_NAME})
ament_target_dependencies(batch_processor
  rclcpp
  sensor_msgs
  cv_bridge
  OpenCV
)

target_include_directories(${PROJECT_NAME} PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  ${cv_bridge_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} 
  ${OpenCV_LIBRARIES}
)

rclcpp_components_register_nodes(${PROJECT_NAME} "yolo_ros::YoloDetectNode")

install(TARGETS 
  ${PROJECT_NAME} 
  yolo_detect_auto_activate
  batch_processor
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()


ament_package()
