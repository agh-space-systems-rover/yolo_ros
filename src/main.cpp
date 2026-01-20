#include <rclcpp/rclcpp.hpp>
#include "yolo_ros/yolo_detect_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  auto node = std::make_shared<yolo_ros::YoloDetectNode>(options);

  // Auto-activate for convenience (mimicking yolo_detect_auto_activate)
  node->configure();
  node->activate();

  rclcpp::spin(node->get_node_base_interface());
  rclcpp::shutdown();
  return 0;
}
