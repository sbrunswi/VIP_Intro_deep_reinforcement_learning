from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'pylon_2026_world'
share_dir = os.path.join("share", package_name)

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join(share_dir, "config"), glob("config/*.yaml")),
        (os.path.join(share_dir, 'logs'), [os.path.join('logs', 'README.md')]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wsribunma',
    maintainer_email='worawissribunma@gmail.com',
    description='Pylon course markers + lap scoring for RViz2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "course_node = pylon_2026_world.course_node:main",
            "purt_course_node = pylon_2026_world.purt_course_node:main"
        ],
    },
)
