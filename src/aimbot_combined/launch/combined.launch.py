import launch
from launch_ros.actions import Node


def generate_launch_description():
    """Launch camera and detector nodes."""
    camera_node = Node(
        package='camera',
        executable='camera_node',
        name='camera_node',
        output='screen',
    )

    detector_node = Node(
        package='aim_auto',
        executable='detector_node',
        name='detector_node',
        output='screen',
    )

    return launch.LaunchDescription([camera_node, detector_node])