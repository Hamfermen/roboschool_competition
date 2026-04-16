from __future__ import annotations

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _launch_setup(context, *args, **kwargs):
    use_sim_time = LaunchConfiguration("use_sim_time")
    delete_db_on_start = LaunchConfiguration("delete_db_on_start").perform(
        context
    ).lower() in ("1", "true", "yes")
    approx_sync = LaunchConfiguration("approx_sync")
    qos_sensor_data = LaunchConfiguration("qos_sensor_data")
    rtabmap_args_raw = LaunchConfiguration("rtabmap_args").perform(context).strip()
    wait_imu_to_init = LaunchConfiguration("wait_imu_to_init")
    rviz = LaunchConfiguration("rviz")
    publish_camera_tf = LaunchConfiguration("publish_camera_tf")
    nav2_autostart = LaunchConfiguration("nav2_autostart")

    pkg_share = FindPackageShare("aliengo_competition")
    rtabmap_params = PathJoinSubstitution([pkg_share, "config", "rtabmap_params.yaml"])
    nav2_params = PathJoinSubstitution([pkg_share, "config", "nav2_params.yaml"])
    rviz_config = PathJoinSubstitution([pkg_share, "rviz", "nav2_view.rviz"])

    rtabmap_args = []
    if rtabmap_args_raw:
        rtabmap_args.extend(rtabmap_args_raw.split())
    if delete_db_on_start:
        rtabmap_args.append("--delete_db_on_start")

    # Optional static TF for camera if simulation bridge doesn't already publish it.
    static_camera_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_static_tf_pub",
        arguments=["0", "0", "0.25", "0", "0", "0", "base_link", "camera_link"],
        condition=IfCondition(publish_camera_tf),
    )

    # Optional lightweight robot_state_publisher with a minimal URDF skeleton.
    # This keeps TF tree consistent in tools that expect robot_description.
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "robot_description": (
                    "<robot name='aliengo'>"
                    "<link name='base_link'/>"
                    "<link name='camera_link'/>"
                    "<joint name='base_to_camera' type='fixed'>"
                    "<parent link='base_link'/>"
                    "<child link='camera_link'/>"
                    "<origin xyz='0 0 0.25' rpy='0 0 0'/>"
                    "</joint>"
                    "</robot>"
                ),
            }
        ],
    )

    # Fuse IMU + measured body twist into odom->base_link for robust short-term odometry.
    # RTAB-Map then closes loops and publishes map->odom.
    ekf_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "frequency": 50.0,
                "sensor_timeout": 0.1,
                "two_d_mode": True,
                "publish_tf": True,
                "map_frame": "map",
                "odom_frame": "odom",
                "base_link_frame": "base_link",
                "world_frame": "odom",
                "imu0": "/imu/data",
                "imu0_config": [
                    False,
                    False,
                    False,  # x y z
                    False,
                    False,
                    True,  # roll pitch yaw
                    False,
                    False,
                    False,  # vx vy vz
                    False,
                    False,
                    True,  # vroll vpitch vyaw
                    False,
                    False,
                    False,  # ax ay az
                ],
                "imu0_differential": False,
                "imu0_remove_gravitational_acceleration": True,
                "twist0": "/measured_vel",
                "twist0_config": [
                    False,
                    False,
                    False,  # x y z
                    False,
                    False,
                    False,  # roll pitch yaw
                    True,
                    True,
                    False,  # vx vy vz
                    False,
                    False,
                    True,  # vroll vpitch vyaw
                    False,
                    False,
                    False,  # ax ay az
                ],
                "twist0_differential": False,
            }
        ],
    )

    rgbd_odometry = Node(
        package="rtabmap_odom",
        executable="rgbd_odometry",
        name="rgbd_odometry",
        output="screen",
        parameters=[
            rtabmap_params,
            {
                "use_sim_time": use_sim_time,
                "frame_id": "base_link",
                "odom_frame_id": "odom",
                "publish_tf": False,  # EKF owns odom->base_link
                "wait_imu_to_init": wait_imu_to_init,
                "approx_sync": approx_sync,
                "qos_sensor_data": qos_sensor_data,
            },
        ],
        remappings=[
            ("rgb/image", "/camera/rgb/image_raw"),
            ("rgb/camera_info", "/camera/rgb/camera_info"),
            ("depth/image", "/camera/depth/image_raw"),
            ("imu", "/imu/data"),
        ],
    )

    rtabmap_node = Node(
        package="rtabmap_slam",
        executable="rtabmap",
        name="rtabmap",
        output="screen",
        arguments=rtabmap_args,
        parameters=[
            rtabmap_params,
            {
                "use_sim_time": use_sim_time,
                "frame_id": "base_link",
                "odom_frame_id": "odom",
                "map_frame_id": "map",
                "publish_tf": True,  # publishes map->odom
                "publish_map_tf": True,
                "approx_sync": approx_sync,
                "qos_sensor_data": qos_sensor_data,
            },
        ],
        remappings=[
            ("rgb/image", "/camera/rgb/image_raw"),
            ("rgb/camera_info", "/camera/rgb/camera_info"),
            ("depth/image", "/camera/depth/image_raw"),
            ("imu", "/imu/data"),
            ("odom", "/odometry/filtered"),
        ],
    )

    nav2_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("nav2_bringup"), "launch", "navigation_launch.py"]
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "params_file": nav2_params,
            "autostart": nav2_autostart,
        }.items(),
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(rviz),
    )

    return [
        static_camera_tf,
        robot_state_publisher,
        ekf_node,
        rgbd_odometry,
        rtabmap_node,
        nav2_bringup,
        rviz_node,
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("delete_db_on_start", default_value="true"),
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument("approx_sync", default_value="true"),
            DeclareLaunchArgument(
                "qos_sensor_data", default_value="2"
            ),  # 2=sensor_data QoS
            DeclareLaunchArgument("wait_imu_to_init", default_value="true"),
            DeclareLaunchArgument("publish_camera_tf", default_value="false"),
            DeclareLaunchArgument("nav2_autostart", default_value="true"),
            DeclareLaunchArgument("rtabmap_args", default_value=""),
            OpaqueFunction(function=_launch_setup),
        ]
    )
