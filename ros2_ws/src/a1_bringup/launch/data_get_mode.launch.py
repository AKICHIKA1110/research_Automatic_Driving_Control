#!/usr/bin/env python3
import os
from datetime import datetime

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # ===== Angstrong カメラの設定ファイルパス =====
    ascamera_conf = os.path.join(
        get_package_share_directory('ascamera'),
        'configurationfiles'
    )

    # ===== rosbag の保存先（USB SSD） =====
    bag_root = os.environ.get('A1_BAG_ROOT', os.path.expanduser('~/rosbags'))   # 環境変数 A1_BAG_ROOT で変更可
    # ディレクトリが無ければ作る
    os.makedirs(bag_root, exist_ok=True)

    # タイムスタンプ付きの bag プレフィックス
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    bag_prefix = f'data_get_{run_id}'

    return LaunchDescription([
        # ロボットコントローラ（モータ／サーボ／IMU）
        Node(
            package='ros_robot_controller',
            executable='ros_robot_controller',
            name='ros_robot_controller',
            output='screen',
        ),

        # ジョイスティック入力
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            output='screen',
        ),

        # PS4コントローラ → /controller/cmd_vel
        Node(
            package='a1_control',
            executable='ps4_teleop',
            name='ps4_teleop',
            output='screen',
        ),

        # /controller/cmd_vel → モータ＆ステアリングサーボ
        Node(
            package='a1_vehicle',
            executable='cmdvel_to_motor',
            name='cmdvel_to_motor',
            output='screen',
        ),

        # 深度カメラ（Angstrong HP60C）
        Node(
            package='ascamera',
            executable='ascamera_node',
            name='ascamera_node',
            output='screen',
            parameters=[{
                'depth_width': 640,
                'depth_height': 480,
                'rgb_width': 640,
                'rgb_height': 480,
                'pub_tfTree': True,
                'confiPath': ascamera_conf,
            }],
        ),

        # === rosbag2 レコーダ ===
        # RGB画像・IMU・cmd_vel・joy・バッテリ電圧・ステアサーボ状態を、USB SSD 上の bag_root に保存
        ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'record',
                '/ascamera_node/rgb0/image', # 保存topic
                '/ros_robot_controller/imu_raw',
                '/controller/cmd_vel',
                '/joy',
                '/ros_robot_controller/battery',  # 電圧 [mV]、約1Hz
                '/ros_robot_controller/bus_servo/joint_states',  # ステアサーボ角 [rad]（500 中心）、約20Hz
                '/ros_robot_controller/bus_servo/state',         # ステアサーボ 位置・電圧・温度、約1Hz
                '/ros_robot_controller/motor/joint_states',      # 車輪速度 [rad/s]・エンコーダカウント、約50Hz（パッチ版ファームウェアのみ）
                '-o', bag_prefix,     # data_get_YYYYMMDD_HHMMSS
            ],
            cwd=bag_root,             # bag_root の中で実行
            output='screen',
        ),
    ])

