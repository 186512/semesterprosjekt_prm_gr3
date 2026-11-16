import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

package_name = 'search_and_rescue'

# Namespace, spawn position (x, y) and yaw of every robot. All inside the start zone.

SPAWN_PER_WORLD = {
    '1': {
        'tb3_0': ('0.0', '-1.0', '0.0'),
        'tb3_1': ('0.0', '1.0', '0.0'),
        'tb3_2': ('-1.0', '0.0', '0.0'),
    },
    '2': {
        'tb3_0': ('-1.0', '1.0', '0.0'),
        'tb3_1': ('-1.0', '-1.0', '0.0'),
        'tb3_2': ('-2.0', '0.0', '0.0'),
    },
    '3': {
        'tb3_0': ('-1.0', '0.5', '0.0'),
        'tb3_1': ('-1.0', '-1.5', '0.0'),
        'tb3_2': ('-2.0', '-0.5', '0.0'),
    },
    '4': {
        'tb3_0': ('-0.5', '0.7', '0.0'),
        'tb3_1': ('-0.5', '-1.3', '0.0'),
        'tb3_2': ('-1.5', '-0.3', '0.0'),
    },
    '5': {
        'tb3_0': ('0.0', '-0.5', '0.0'),
        'tb3_1': ('0.0', '-2.5', '0.0'),
        'tb3_2': ('-1.0', '-1.5', '0.0'),
    },
}
DEFAULT_WORLD = '1' 

ROBOT_NAMES = ['tb3_0', 'tb3_1', 'tb3_2']

EXPLORERS = ['tb3_0', 'tb3_1']


def launch_setup(context):
    share = get_package_share_directory(package_name)
    world = LaunchConfiguration('world').perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time')
    robots = SPAWN_PER_WORLD.get(world, SPAWN_PER_WORLD[DEFAULT_WORLD])

    world_file_path = os.path.join(share, 'worlds', f'dat160_w{world}.world')
    map_file_path = os.path.join(share, 'maps', f'map_dat160_w{world}.yaml')
    rviz_config_file_path = os.path.join(share, 'rviz', 'model.rviz')

    actions = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')),
            launch_arguments={'world': world_file_path}.items()),

        Node(package='nav2_map_server', executable='map_server', name='map_server',
             output='screen',
             parameters=[{'yaml_filename': map_file_path, 'topic_name': 'map',
                          'frame_id': 'map', 'use_sim_time': use_sim_time}]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_localization', output='screen',
             parameters=[{'use_sim_time': use_sim_time}, {'autostart': True},
                         {'node_names': ['map_server']}]),

        Node(package='scoring', executable='scoring', name='scoring', output='screen',
             parameters=[{'use_sim_time': use_sim_time}]),
        Node(package='rviz2', executable='rviz2', name='rviz2', output='screen',
             arguments=['-d', rviz_config_file_path],
             parameters=[{'use_sim_time': use_sim_time}]),
    ]

    # Spawn every robot (also starts its ArUco detection and map->odom transform)
    for ns, (x, y, yaw) in robots.items():
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'spawn_robot.launch.py')),
            launch_arguments={'namespace': ns, 'x': x, 'y': y, 'yaw': yaw}.items()))

    # PRM explorers (replace the wall followers)
    for ns in EXPLORERS:
        actions.append(Node(
            package=package_name, executable='explorer', name='explorer', namespace=ns,
            output='screen',
            parameters=[{'use_sim_time': True,
                         'namespace': ns,
                         'num_robots': len(EXPLORERS),
                         'all_robots': list(ROBOT_NAMES)}]))

    actions += [
        Node(package=package_name, executable='multi_robot_coordinator', output='screen',
             parameters=[{'use_sim_time': True}]),
        # Rescue robot: same planner/controller as the explorers, in 'rescue' mode. It waits at
        # the start and drives to the big fire when the coordinator calls.
        Node(package=package_name, executable='explorer', name='rescue', namespace='tb3_2',
             output='screen',
             parameters=[{'use_sim_time': True,
                          'namespace': 'tb3_2',
                          'role': 'rescue',
                          'all_robots': list(ROBOT_NAMES)}]),
    ]
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='Use simulation (Gazebo) clock if true'),
        DeclareLaunchArgument('world', default_value='1', description='World number 1-5'),
        OpaqueFunction(function=launch_setup),
    ])
