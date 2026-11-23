from setuptools import setup

package_name = 'a1_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/manual_mode.launch.py',
            'launch/data_get_mode.launch.py',
            'launch/data_get_depth_mode.launch.py',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='akichika',
    maintainer_email='128462874+AKICHIKA1110@users.noreply.github.com',
    description='Launch files for the A1 Ackermann car (manual driving and rosbag data collection)',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'image_sync_throttle = a1_bringup.image_sync_throttle:main',
        ],
    },
)

