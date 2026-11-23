from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Robot controller
        Node(
            package='ros_robot_controller',
            executable='ros_robot_controller',
            name='ros_robot_controller',
            output='screen',
        ),

        # Joy node
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            output='screen',
        ),

        # PS4 teleop → /controller/cmd_vel
        Node(
            package='a1_control',
            executable='ps4_teleop',
            name='ps4_teleop',
            output='screen',
        ),

        # cmd_vel → モータ & サーボ
        Node(
            package='a1_vehicle',
            executable='cmdvel_to_motor',
            name='cmdvel_to_motor',
            output='screen',
        ),
    ])
