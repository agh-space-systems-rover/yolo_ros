from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def launch_setup(context, *args, **kwargs):
    # Resolve LaunchConfigurations to strings
    folder = LaunchConfiguration('folder').perform(context)
    model = LaunchConfiguration('model').perform(context)
    config = LaunchConfiguration('config').perform(context)
    
    # Base parameters from config file
    params = [config]
    
    # Override only if arguments are provided (non-empty)
    if folder:
        params.append({'input_folder': folder})
        
    if model:
        params.append({'model_path': model})
        
    return [
        Node(
            package='yolo_ros',
            executable='batch_processor',
            name='batch_processor',
            output='screen',
            parameters=params
        )
    ]

def generate_launch_description():
    pkg_share = FindPackageShare('yolo_ros')
    
    # Default config file path
    default_config = PathJoinSubstitution([pkg_share, 'config', 'arch.yaml'])

    # Default model path - assumes kalman_yolo is installed
    kalman_yolo_share = FindPackageShare('kalman_yolo')
    default_model = PathJoinSubstitution([kalman_yolo_share, 'models', 'arch2025.onnx'])

    return LaunchDescription([
        # Main arguments needed
        DeclareLaunchArgument('folder', default_value='', description='Path to folder containing images to process'),
        DeclareLaunchArgument('model', default_value=default_model, description='Path to YOLO ONNX model'),
        
        # Config file argument
        DeclareLaunchArgument('config', default_value=default_config, description='Path to config file (arch.yaml)'),

        OpaqueFunction(function=launch_setup)
    ])

