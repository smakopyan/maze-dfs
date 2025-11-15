from setuptools import find_packages, setup

package_name = 'maze-dfs'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/robot_state_publisher.launch.py',
                                                'launch/spawn_turtlebot3.launch.py',
                                                'launch/trash_bins_maze.launch.py']),
        ('share/' + package_name + '/worlds', ['worlds/world_with_bins.sdf', 'worlds/world_with_bin.sdf']),
        ('share/' + package_name + '/config', ['config/map.pgm', 'config/map.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sa',
    maintainer_email='satenikak@yandex.ru',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
