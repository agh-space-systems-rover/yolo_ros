# YOLO-ROS (C++)

A high-performance ROS 2 Lifecycle node integrating YOLOv8 object detection with 3D position estimation and multi-camera support. This package uses OpenCV DNN module for inference.

## Features

- **Multi-Camera Support**: Synchronized processing of RGB and Depth streams from multiple cameras.
- **3D Position Estimation**: Projects 2D detections to 3D space using depth maps or fallback heuristics.
- **Temporal Tracking**: Filters out flickering detections and tracks objects over time.
- **Lifecycle Management**: Fully supports ROS 2 Lifecycle state transitions (Configure, Activate, Deactivate, Cleanup).
- **Visualization**: Publishes annotated images with bounding boxes and labels per camera stream.

## Architecture

The node follows a modular architecture derived from SOLID principles:

- **YoloDetectNode**: The orchestrator that manages lifecycle and data flow.
- **CameraManager**: Handles subscriptions and synchronization of camera streams.
- **Visualizer**: Manages image annotation and publishing.
- **IDetector**: Interface for object detection (implemented by `YoloOpenCVDetector`).
- **IPositionEstimator**: Interface for 3D projection logic.
- **ITracker**: Interface for temporal filtering and tracking.

See `docs/node_structure.puml` and `docs/cpp_structure.puml` for detailed diagrams.

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `rgbd_ids` | string[] | List of camera IDs (e.g., `["d455_front", "d455_back"]`). Topics will be derived as `/{id}/color/image_raw`. |
| `model` | string | Path to the ONNX model file. |
| `class_names` | string[] | List of class names corresponding to model outputs. |
| `class_radii` | double[] | Real-world radius (in meters) for each class, used for depth estimation fallback. |
| `confidence_threshold` | double | Minimum confidence score for detections. |
| `merge_radius` | double | Distance in meters to merge duplicate detections. |
| `temporal_window` | int | Size of the history buffer for tracking. |
| `temporal_threshold` | int | Minimum detections in window to confirm an object. |
| `publish_annotated` | bool | Whether to publish debug images with bounding boxes. |

## How to Launch

```python
from launch import LaunchDescription
from launch_ros.actions import LifecycleNode

def generate_launch_description():
    return LaunchDescription([
        LifecycleNode(
            package='yolo_ros',
            executable='yolo_detect_node',
            name='yolo_detect',
            namespace='',
            output='screen',
            parameters=[{
                'rgbd_ids': ['camera_front', 'camera_rear'],
                'model': '/path/to/model.onnx',
                'class_names': ['person', 'chair'],
                'class_radii': [0.3, 0.5],
                'confidence_threshold': 0.5,
                'publish_annotated': True,
                'color_transport': 'compressed',
                'depth_transport': 'compressedDepth'
            }]
        )
    ])
```

## Topics

### Subscribed
For each ID in `rgbd_ids`:
- `/{id}/color/image_raw` (or `/compressed`)
- `/{id}/depth/image_raw` (or `/compressedDepth`)
- `/{id}/color/camera_info`

### Published
- `/detections` (`vision_msgs/Detection2DArray`): Merged and filtered detections.
- `/{id}/yolo_annotated` (`sensor_msgs/Image`): Debug visualization (if enabled).
- `/tf`: Transform frames for detected objects.
