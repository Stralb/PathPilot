# Jetbot Path Tracking and Control System

## Overview

This Project provides a path tracking and control system for Jetbots using computer vision techniques. It detects paths, tracks objects, and adjusts motor speeds based on visual feedback.

## Key Features

1. **Path Detection**: Identifies and processes a white line to define a path.
2. **Object Tracking**: Tracks red and blue dots using OpenCV's CSRT tracker.
3. **Angle Calculation**: Computes angles between vectors to determine direction.
4. **Speed Adjustment**: Adjusts motor speeds based on tracking data and angle errors.
5. **Interactive UI**: Allows manual selection of tracking points.

## Algorithms and Techniques

- **Path Processing**: 
  - Binary image conversion and thinning to extract the midline.
  - Depth-First Search (DFS) for finding the shortest path between points.
- **Tracking**: 
  - Object tracking using CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability).
- **Angle Calculation**: 
  - Vector angle calculation to determine direction.
- **Speed Adjustment**: 
  - PID control for motor speed adjustment based on angle error.

## Installation

1. **Install Dependencies**

   ```bash
   pip install opencv-python numpy scikit-image opencv-contrib-python
   ```

## Usage

1. **Run the Script**

   ```bash
   python path_tracking.py
   ```

2. **Select Points**

   - Click on the video frame to select start and end points for path detection.
   - Select red and blue dots for tracking.

3. **Tracking and Control**

   - The script will track objects, compute angles, and adjust motor speeds in real-time based on visual feedback.
  
4. **Files**

The system consists of two main scripts:

    path_tracking.py:
        Handles path detection, object tracking, and angle calculations.
        Adjusts motor speed instructions based on visual feedback.

    jetbot_control.py:
        Reads motor speed instructions from a file and controls the JetBot accordingly.

## Functions

- `extract_coordinates()`: Extracts coordinates of white pixels.
- `create_line_matrix()`: Creates a matrix from line coordinates.
- `find_and_draw_area()`: Finds and draws the path between points.
- `thin_yellow_coordinates()`: Thins the yellow line for midline extraction.
- `dfs()`: Finds the shortest path using depth-first search.
- `calculate_angle()`: Computes the angle between vectors.
- `adjust_motor_speeds()`: Adjusts motor speeds based on angle error.

## Contributing

Contributions are welcome. Submit pull requests or open issues to improve functionality.

This overview covers the main functionalities, algorithms used, installation steps, and basic usage instructions. Let me know if you need any more details or adjustments!
