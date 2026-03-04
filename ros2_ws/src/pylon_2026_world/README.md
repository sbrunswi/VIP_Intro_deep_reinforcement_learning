# Autonomosu Pylon Racing Competition Official Scoring Metrics
This package contains RViz2-based pylon course and lap time scoring for fixed-wing UAS racing.

## Build
```bash
cd ~/ros2_ws/src
git clone https://github.com/wsribunma/pylon_2026_world.git
cd ..
colcon build --symlink-install
. install/setup.bash
```

## Run
```bash
. install/setup.bash
ros2 run pylon_2026_world course.launch.py
```
