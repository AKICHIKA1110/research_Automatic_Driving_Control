from setuptools import find_packages, setup

package_name = 'a1_vehicle'

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
    description='Converts /controller/cmd_vel into rear DC motor speeds and front steering servo commands for the A1 car',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'cmdvel_to_motor = a1_vehicle.cmdvel_to_motor:main',
        ],
    },

)
