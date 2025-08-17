# 🤖 Autonomous Jetbot: Path Tracking & Control

A real-time computer vision system for path tracking, object detection, and autonomous navigation.

## 🚀 Live Demonstration

Witness the Jetbot in action! These videos showcase the system's ability to autonomously follow a path, track objects, and adjust its movement in real-time.

| Video | Description |
|-------|-------------|
| [Pathpilot Demo](../Videos/pathpilot2.mp4) | The Jetbot navigating the path and making real-time adjustments. |
| [Object Tracking Demo](../Videos/pathpilot1.mp4) | Demonstration of the system tracking red and blue dots. |
| [Full Course Demo](../Videos/pathpilot3.mp4) | The Jetbot successfully completing the course autonomously. |

## 💡 Overview

This project provides a robust control system for Jetbots, blending computer vision with robotics to achieve autonomous navigation. The system dynamically processes live camera feeds to detect a predefined path, identify specific objects, and make precise motor adjustments. This enables the Jetbot to autonomously navigate complex environments with no human intervention.

## ✨ Key Features

- **Autonomous Path Following**: Identifies and follows a white line, even with curves and turns, by extracting its midline.
- **Real-time Object Tracking**: Utilizes the robust CSRT algorithm to track red and blue dots, adding an extra layer of navigation and object avoidance.
- **Dynamic Speed Control**: Employs a PID controller to calculate and adjust motor speeds in real-time based on the angle of the path and position errors.
- **Interactive UI**: Allows users to manually select and redefine tracking points on the fly for increased flexibility.

## 🧠 Under the Hood

The system relies on a blend of proven algorithms and techniques to achieve its functionality:

- **Path Processing**: Converts the video feed into a binary image, applies thinning to extract the thinnest possible midline, and uses Depth-First Search (DFS) to find the shortest path between user-defined points.
- **Object Tracking**: Leverages OpenCV's CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability) for high-performance and reliable tracking of objects.
- **Vector Geometry**: Calculates the angle between vectors to determine the Jetbot's heading relative to the path, providing the necessary data for the PID controller.

## 🛠️ Installation & Usage

### Prerequisites
- A Jetbot with a camera module.
- Python 3.8 or higher installed on your system.

### Setting Up the Environment

1. Clone this repository to your Jetbot:
   ```bash
   git clone https://github.com/yourusername/pathpilot.git
   cd pathpilot
   ```

2. Create a virtual environment:
   ```bash
   # On Windows
   python -m venv venv
   
   # On Linux/macOS
   python3 -m venv venv
   ```

3. Activate the virtual environment:
   ```bash
   # On Windows
   venv\Scripts\activate
   
   # On Linux/macOS
   source venv/bin/activate
   ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Files
- `path_tracking.py`: The main script that handles all vision-based processing, including path detection and object tracking.
- `jetbot_control.py`: The script that reads motor speed instructions and controls the Jetbot's motors.
- `requirements.txt`: Contains all the necessary Python dependencies for the project.

### Running the System
1. Ensure your virtual environment is activated.
2. Run the main script:
   ```bash
   python path_tracking.py
   ```
3. Once the live video feed appears, click on the screen to set your desired tracking points for the path and the objects.

### Troubleshooting
- If you encounter issues with OpenCV on the Jetbot, you may need to build it from source for optimal performance.
- Ensure your camera is properly connected and accessible by the system.
- For performance issues, consider reducing the resolution in the path_tracking.py script.

## 🤝 Contributing

Contributions are what make the open-source community so amazing. If you have a suggestion for improving this project, feel free to open an issue or submit a pull request.