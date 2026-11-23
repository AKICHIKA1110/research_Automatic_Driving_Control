from setuptools import find_packages, setup

package_name = 'a1_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='akichika',
    maintainer_email='128462874+AKICHIKA1110@users.noreply.github.com',
    description='PS4 joystick teleoperation for the A1 car (Joy -> /controller/cmd_vel)',
    license='Apache-2.0',
    entry_points={
       'console_scripts': [
            'ps4_teleop = a1_control.ps4_teleop:main',
        ],
    },

)
