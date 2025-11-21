#!/bin/bash

# Script to launch all four nodes: camera_node, detector_node, tracker_node, serial_bridge_node

# Set up ROS2 environment
source install/local_setup.bash

echo "Starting camera_node..."
ros2 run camera camera_node &
CAMERA_PID=$!

echo "Starting detector_node..."
ros2 run aim_auto detector_node --ros-args -p target_color:=red &
#ros2 run aim_auto detector_ref_node &
DETECTOR_PID=$!

echo "Starting tracker_node..."
ros2 run aim_auto tracker_node &
TRACKER_PID=$!

#echo "Starting serial_bridge_node..."
#ros2 run aim_auto serial_bridge_node &
#SERIAL_PID=$!

echo "All nodes started!"
echo "Camera PID: $CAMERA_PID"
echo "Detector PID: $DETECTOR_PID"
echo "Tracker PID: $TRACKER_PID"
#echo "Serial PID: $SERIAL_PID"

# Wait for all processes
wait $CAMERA_PID $DETECTOR_PID $TRACKER_PID #$SERIAL_PID

echo "All nodes have stopped."