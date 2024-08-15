# PathPilot

PathPilot is an advanced robotics project where a small robot, the JetBot, is guided by an overhead camera. This system leverages OpenCV and Python to analyze the course from above, plan the optimal path, and navigate the JetBot with precision. It combines aerial analysis with autonomous navigation to achieve accurate pathfinding and obstacle avoidance.

## Features

- **Overhead Camera Analysis:** Uses OpenCV to capture and analyze the course.
- **Path Planning:** Computes the optimal route for the JetBot.
- **Autonomous Navigation:** JetBot follows the planned path based on real-time data.

## Setup Instructions

### Prerequisites

- **Hardware:**
  - JetBot
  - Overhead camera
  - Computer with Python and Jupyter Notebook installed
- **Software:**
  - Python 3.x
  - OpenCV (`opencv-python`)
  - Jupyter Notebook

### 1. JetBot Setup

1. **Code Deployment:**
   - Download the JetBot code provided in the `.txt` file.
   - Open Jupyter Notebook on your PC.
   - Create a new notebook and copy the code from the `.txt` file into a code cell.

2. **Network Configuration:**
   - Ensure that your JetBot and PC are connected to the same Wi-Fi network. This setup will allow seamless communication between the JetBot and your computer.

### 2. Python Code Configuration

1. **Update IP Address:**
   - Locate the Python script provided in the repository.
   - Open the script in a text editor.
   - Find the section where the IP address of the JetBot is specified and replace it with your JetBot’s IP address.

2. **Camera Setup:**
   - Connect an external camera to your computer.
   - Ensure that the camera is properly configured and recognized by your system.

3. **Run the Script:**
   - Execute the Python script in your environment.
   - Follow the interactive prompts provided by the script to calibrate the system and start the navigation process.

### Usage

1. **Start Analysis:**
   - Run the Jupyter Notebook with the JetBot code to initiate the overhead camera analysis.
   - The camera will capture images of the course and process them using OpenCV.

2. **Monitor Navigation:**
   - The JetBot will begin following the planned path as determined by the overhead camera analysis.
   - Monitor the JetBot’s progress and adjust parameters if necessary based on real-time feedback.

### Troubleshooting

- **Connection Issues:** Ensure both the JetBot and PC are on the same network and check for any network-related issues.
- **Camera Problems:** Verify that the external camera is properly connected and configured. Check camera drivers and connections if the camera is not recognized.
- **Code Errors:** Double-check that the JetBot’s IP address is correctly updated in the Python script and ensure all required libraries are installed.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenCV for computer vision tasks
- Jupyter Notebook for interactive development
- JetBot for autonomous robotics

For further information or support, please refer to the [Documentation](docs) or open an issue in the repository.

---

This detailed README provides a comprehensive guide for setting up and using the PathPilot project, including prerequisites, setup instructions, usage, and troubleshooting tips.
