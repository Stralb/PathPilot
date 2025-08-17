import cv2
import math
import socket
import time
import numpy as np
from skimage.morphology import skeletonize
import heapq
from collections import deque
import sys

# Function to extract coordinates from a white line
def extract_coordinates(line_mask):
    coordinates = []
    for y in range(line_mask.shape[0]):
        for x in range(line_mask.shape[1]):
            if line_mask[y, x] > 0:
                coordinates.append((x, y))
    return coordinates

# Function to create a matrix with 1s at white line coordinates and 0s elsewhere
def create_line_matrix(line_coordinates, shape):
    line_matrix = np.zeros(shape, dtype=np.uint8)
    for coord in line_coordinates:
        line_matrix[coord[1], coord[0]] = 1
    return line_matrix

# Function to find the area around the shortest path until the white line and fill it with yellow
def find_and_draw_area(start, end, line_matrix, frame):
    # Initialize yellow area coordinates
    yellow_area_coordinates = []

    # Define the yellow color
    yellow_color = (0, 255, 255)

    # Create a copy of the frame to work with
    filled_frame = frame.copy()

    # Define a function to check if a coordinate is within bounds and not part of a white line
    def is_valid(x, y):
        return 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0] and not line_matrix[y][x] and filled_frame[y, x, 0] != yellow_color[0]

    # Define directions for exploring adjacent cells
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    # Perform flood fill from start and end points using an iterative approach
    stack = [(start[0], start[1])]
    while stack:
        x, y = stack.pop()
        if is_valid(x, y):
            filled_frame[y, x] = yellow_color
            yellow_area_coordinates.append((x, y))
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if is_valid(nx, ny):
                    stack.append((nx, ny))

        # Add termination condition to avoid infinite looping
        if len(yellow_area_coordinates) >= frame.shape[0] * frame.shape[1]:
            break

    # Draw the yellow area coordinates on the frame
    for x, y in yellow_area_coordinates:
        cv2.circle(frame, (x, y), 1, yellow_color, -1)  # Draw a filled circle

    return yellow_area_coordinates

# Function to convert yellow area to binary image
def yellow_to_binary(frame, yellow_area_coordinates):
    binary_frame = np.zeros_like(frame[:, :, 0])
    for x, y in yellow_area_coordinates:
        binary_frame[y, x] = 255
    return binary_frame

# Function to thin yellow coordinates using OpenCV's thinning algorithm
def thin_yellow_coordinates(binary_yellow):
    thinned_yellow = cv2.ximgproc.thinning(binary_yellow)
    return thinned_yellow



def dfs(current, end, visited, path, midline, tolerance=3):
    if np.linalg.norm(np.array(current) - np.array(end)) <= tolerance:
        return path + [end]

    visited.add(current)

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for dx, dy in directions:
        next_x, next_y = current[0] + dx, current[1] + dy
        if 0 <= next_x < midline.shape[1] and 0 <= next_y < midline.shape[0] and midline[next_y, next_x] > 0 and (next_x, next_y) not in visited:
            result = dfs((next_x, next_y), end, visited, path + [(next_x, next_y)], midline, tolerance)
            if result:
                return result
    
    print("No valid path found from", current, "to", end)  # Debugging output
    return []

# Increase recursion limit for testing purposes
sys.setrecursionlimit(10000)


def connect_and_draw(start, end, midline, frame):
    # Define directions for exploring adjacent cells (8-connectivity)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (-1, 1), (1, -1)]

    if midline[start[1], start[0]] == 0:
        start = find_closest_coordinate(start, midline)
    if midline[end[1], end[0]] == 0:
        end = find_closest_coordinate(end, midline)

    # Initialize visited set
    visited = set()

    # Initialize connected line coordinates
    connected_line_coordinates = []

    # Initialize stack for iterative DFS
    stack = [(start, [])]  # Each item in the stack includes the current point and the path taken so far

    # Perform iterative depth-first search (DFS) to find connected line segment
    while stack:
        current, path = stack.pop()
        visited.add(current)

        # Add the current point to the path
        path.append(current)

        # Check if the current point is the end point
        if current == end:
            connected_line_coordinates = path  # Set the connected line coordinates to the final path
            break  # Exit loop if end point is reached

        # Explore adjacent cells
        for dx, dy in directions:
            next_x, next_y = current[0] + dx, current[1] + dy
            if (
                0 <= next_x < frame.shape[1]
                and 0 <= next_y < frame.shape[0]
                and midline[next_y, next_x] > 0
                and (next_x, next_y) not in visited
            ):
                # Add the next point and the updated path to the stack
                stack.append(((next_x, next_y), path.copy()))  # Use path.copy() to avoid modifying the same list

    return connected_line_coordinates


def find_closest_coordinate(point, midline, tolerance=30):
    # Example implementation to find closest coordinate in midline with tolerance
    if(midline is not None):
        closest_coord = None
        min_distance = float('inf')

        for y in range(point[1] - tolerance, point[1] + tolerance + 1):
            for x in range(point[0] - tolerance, point[0] + tolerance + 1):
                if 0 <= y < midline.shape[0] and 0 <= x < midline.shape[1] and midline[y, x] > 0:
                    distance = np.linalg.norm(np.array(point) - np.array((x, y)))
                    if distance < min_distance:
                        closest_coord = (x, y)
                        min_distance = distance

        return closest_coord

# Function to get start and end points manually by clicking on the frame
def select_points(event, x, y, flags, param):
    global start_point, end_point
    if event == cv2.EVENT_LBUTTONDOWN:
        if start_point is None:
            start_point = (x, y)
            print("Start Point Selected:", start_point)
        elif end_point is None:
            end_point = (x, y)
            print("End Point Selected:", end_point)

# Function to get start and end points manually by clicking on the frame
def get_start_end_points(frame, midline):
    global start_point, end_point
    start_point = None
    end_point = None

    cv2.namedWindow('1. Select Start and End Points')
    cv2.setMouseCallback('1. Select Start and End Points', select_points)

    while True:
        frame_with_points = frame.copy()  # Create a copy of the frame to draw points on
        # Draw start and end points if they are selected
        if start_point is not None:
            cv2.circle(frame_with_points, start_point, 5, (0, 0, 255), -1)  # Red circle for start point
        if end_point is not None:
            cv2.circle(frame_with_points, end_point, 5, (0, 255, 0), -1)  # Green circle for end point

        cv2.imshow('1. Select Start and End Points', frame_with_points)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif start_point is not None and end_point is not None:
            break

    cv2.destroyAllWindows()
    if midline is not None:
        start_point = find_closest_coordinate(start_point, midline)
        end_point = find_closest_coordinate(end_point, midline)
    return start_point, end_point



def return_line():
    midline =None
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Thresholding to isolate white line
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)


    # Find contours of white line
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract coordinates of white lines (assuming you have a function extract_coordinates)
    line_coordinates = extract_coordinates(thresh)

    # Create a matrix with 1s at white line coordinates and 0s elsewhere
    line_matrix = create_line_matrix(line_coordinates, thresh.shape)

    # Get start and end points manually
    start_point, end_point = get_start_end_points(frame.copy(), midline)
    
    # Find and draw the area around the path until the white line in yellow
    yellow_area_coordinates = find_and_draw_area(start_point, end_point, line_matrix, frame)

    # Convert yellow area to binary image
    binary_yellow = yellow_to_binary(frame, yellow_area_coordinates)

    # Thin the yellow coordinates
    thinned_yellow = thin_yellow_coordinates(binary_yellow)

    # Draw thinned yellow coordinates in red on the frame
    frame[thinned_yellow > 0] = (0, 0, 255)  # Draw thinned yellow coordinates in red

    # Skeletonize the binary image to get midline
    midline = skeletonize(binary_yellow / 255).astype(np.uint8) * 255

    # Draw midline on the frame
    frame[midline > 0] = (0, 0, 255)  # Draw midline in red

    # Get connected line coordinates
    connected_line_coordinates = connect_and_draw(start_point, end_point, midline, frame)
   

    for coord in connected_line_coordinates:
        cv2.circle(frame, coord, 1, (0, 0, 0), -1)

    return connected_line_coordinates


# Global variables to store selected point coordinates and tracking status
selected_red_point = None
selected_blue_point = None
start_tracking = False

# Mouse callback function to capture click event and store coordinates for red dot
def select_red_point(event, x, y, flags, param):
    global selected_red_point, start_tracking
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_red_point = (x, y)
        start_tracking = True  # Set tracking status to True when red point is selected

        # Draw the red dot at the selected point
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Red color

# Mouse callback function to capture click event and store coordinates for blue dot
def select_blue_point(event, x, y, flags, param):
    global selected_blue_point
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_blue_point = (x, y)
        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)  # Red color

# Set up video capture device
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 700)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
screen_width, screen_height = 700, 500  # Replace with your actual screen resolution


lists = return_line()

# Read the first frame
ret, frame = cap.read()
if not ret or frame is None:
    print("Error reading frame from video stream.")
    exit()

# Manually select the object to track by drawing a bounding box
bbox = cv2.selectROI("2. Select Object to Track", frame, False)
if bbox == (0, 0, 0, 0):
    print("Empty bounding box selected. Please try again.")
    exit()

# Create a new window for displaying the frame with instructions
cv2.namedWindow('2.1 Object Selection, select the direction of the Jetbot', cv2.WINDOW_NORMAL)
cv2.resizeWindow('2.1 Object Selection, select the direction of the Jetbot', 800, 600)  # Set window size (adjust as needed)


# Set mouse callback to capture red dot selection within the same frame
cv2.setMouseCallback("2.1 Object Selection, select the direction of the Jetbot", select_red_point)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    # Display the frame with bounding box and instructions for selecting the red dot
    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (255, 255, 0), 2)
    cv2.putText(frame, 'Select a red dot inside the bounding box and press Enter', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow('2.1 Object Selection, select the direction of the Jetbot', frame)

    # Check for key press to start dot tracking and exit
    key = cv2.waitKey(1) & 0xFF
    if key == 13:  # Enter key press after selecting the red dot
        if selected_red_point is None:
            print("Please select a red dot inside the bounding box before pressing Enter to start tracking.")
        else:
            break
    elif key == ord('q'):  # 'q' key press to exit
        break

# Initialize CSRT tracker for the object
tracker_obj = cv2.TrackerCSRT_create()
tracker_obj.init(frame, bbox)

# Check if selected_red_point is valid before initializing the red dot tracker
if selected_red_point is not None:
    # Use the selected red dot as the center of the dot's bounding box
    red_dot_bbox = (selected_red_point[0] - 5, selected_red_point[1] - 5, 10, 10)  # Adjust size as needed
    tracker_red_dot = cv2.TrackerCSRT_create()
    tracker_red_dot.init(frame, red_dot_bbox)
else:
    print("Error: No valid red dot selected.")

# Create a new window for displaying the frame with instructions
cv2.namedWindow('2.2 Object Selection, select the middle of the Jetbot', cv2.WINDOW_NORMAL)
cv2.resizeWindow('2.2 Object Selection, select the middle of the Jetbot', 800, 600)  # Set window size (adjust as needed)


# Set mouse callback to capture red dot selection within the same frame
cv2.setMouseCallback("2.2 Object Selection, select the middle of the Jetbot", select_blue_point)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    # Display the frame with bounding box and instructions for selecting the blue dot
    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 255, 0), 2)
    cv2.putText(frame, 'Select a blue dot inside the bounding box and press Enter', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    cv2.imshow('2.2 Object Selection, select the middle of the Jetbot', frame)

    # Check for key press to start dot tracking and exit
    key = cv2.waitKey(1) & 0xFF
    if key == 13:  # Enter key press after selecting the blue dot
        if selected_blue_point is None:
            print("Please select a blue dot inside the bounding box.")
        else:
            break
    elif key == ord('q'):  # 'q' key press to exit
        break

# Check if selected_blue_point is valid before initializing the blue dot tracker
if selected_blue_point is not None:
    # Use the selected blue dot as the center of the dot's bounding box
    blue_dot_bbox = (selected_blue_point[0] - 5, selected_blue_point[1] - 5, 10, 10)  # Adjust size as needed
    tracker_blue_dot = cv2.TrackerCSRT_create()
    tracker_blue_dot.init(frame, blue_dot_bbox)
else:
    print("Error: No valid blue dot selected.")

# Create the "Object Tracking" window
cv2.namedWindow('3. Object Tracking', cv2.WINDOW_NORMAL)
cv2.resizeWindow('3. Object Tracking', 800, 600)  # Set window size (adjust as needed)


def calculate_angle(vector1, vector2):
    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    magnitude_vector1 = math.sqrt(sum(a * a for a in vector1))
    magnitude_vector2 = math.sqrt(sum(b * b for b in vector2))
    cosine_angle = dot_product / (magnitude_vector1 * magnitude_vector2)

    # Ensure the cosine angle is within [-1, 1] due to numerical precision
    cosine_angle = max(min(cosine_angle, 1), -1)

    angle_rad = math.acos(cosine_angle)
    angle_deg = math.degrees(angle_rad)

    # Check if the angle should be negative (between 180 and 360 degrees)
    if vector1[0] * vector2[1] - vector1[1] * vector2[0] < 0:
        angle_deg = -angle_deg

    return angle_deg


def calculate_shortest_angle(current_angle, target_angle):
    error_angle = target_angle - current_angle
    if error_angle > 180:
        error_angle -= 360
    elif error_angle < -180:
        error_angle += 360
    return error_angle

def adjust_motor_speeds(error_angle, max_speed):
    if error_angle < 0:
        # Target is to the right, adjust speeds for a right turn
        left_speed = max_speed
        right_speed = -max_speed
    elif error_angle > 0:
        # Target is to the left, adjust speeds for a left turn
        left_speed = -max_speed
        right_speed = max_speed
    else:
        # Error angle is zero, both motors remain at zero speed
        left_speed = 0
        right_speed = 0

    return left_speed, right_speed

def calculate_distance_from_goal(current_position, goal_position):
    # Unpack current and goal positions
    x_current, y_current = current_position
    x_goal, y_goal = goal_position

    # Calculate squared differences in coordinates
    x_diff_sq = (x_goal - x_current) ** 2
    y_diff_sq = (y_goal - y_current) ** 2

    # Calculate Euclidean distance
    distance = math.sqrt(x_diff_sq + y_diff_sq)
    
    return distance

# Initialize PID variables
kp = 1.0
ki = 0.0
kd = 0.0
target_angle = 0.0
integral = 0.0
last_error = 0.0
start_index = 0  # Index of the current start coordinate

# Start with the first coordinate in the list
start = lists[start_index]
threshold_error = 10  # Example threshold error for feedback adjustment
feedback_factor = 0.9  # Example feedback factor for adjusting PID constants
curr_speed = 0.1
# Main object tracking loop
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    # Update the tracker for the object with the new frame
    success_obj, bbox_obj = tracker_obj.update(frame)

    if success_obj:
        # Tracking successful for the object, draw bounding box around it
        x_obj, y_obj, w_obj, h_obj = [int(i) for i in bbox_obj]
        cv2.rectangle(frame, (x_obj, y_obj), (x_obj + w_obj, y_obj + h_obj), (0, 255, 0), 2)
        cv2.putText(frame, 'Tracked Object', (x_obj, y_obj - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Update the tracker for the red dot with the new frame
    success_red_dot, red_dot_bbox = tracker_red_dot.update(frame)

    if success_red_dot:
        # Tracking successful for the red dot, draw it on the frame in red color
        x_red_dot, y_red_dot, _, _ = [int(i) for i in red_dot_bbox]
        cv2.circle(frame, (x_red_dot, y_red_dot), 5, (0, 0, 255), -1)  # Red color

    # Update the tracker for the blue dot with the new frame
    success_blue_dot, blue_dot_bbox = tracker_blue_dot.update(frame)

    if success_blue_dot:
        # Tracking successful for the blue dot, draw it on the frame in blue color
        x_blue_dot, y_blue_dot, _, _ = [int(i) for i in blue_dot_bbox]
        cv2.circle(frame, (x_blue_dot, y_blue_dot), 5, (255, 0, 0), -1)  # Blue color

        # Draw a line between the start point and the blue coordinate
        cv2.line(frame, start, (x_blue_dot, y_blue_dot), (255, 0, 0), 2)

        # Calculate the angle between the red-blue line and the blue-start line
        vector_red_blue = (x_red_dot - x_blue_dot, y_red_dot - y_blue_dot)
        vector_blue_start = (start[0] - x_blue_dot, start[1] - y_blue_dot)

        dot_product = vector_red_blue[0] * vector_blue_start[0] + vector_red_blue[1] * vector_blue_start[1]
        magnitude_red_blue = math.sqrt(vector_red_blue[0] ** 2 + vector_red_blue[1] ** 2)
        magnitude_blue_start = math.sqrt(vector_blue_start[0] ** 2 + vector_blue_start[1] ** 2)

        if magnitude_red_blue != 0 and magnitude_blue_start != 0:
            angle_deg = calculate_angle(vector_red_blue, vector_blue_start)

            # PID Control
            error = target_angle - angle_deg
            integral += error
            derivative = error - last_error
            output = kp * error + ki * integral + kd * derivative
            last_error = error

            if abs(error) > threshold_error:
                # Adjust kp, ki, kd based on performance metrics (e.g., error, overshoot, settling time)
                kp_new = kp * feedback_factor
                ki_new = ki * feedback_factor
                kd_new = kd * feedback_factor

                def smooth_transition(current, target, rate):
                    return current + (target - current) * rate

                transition_rate = 0.1  # Adjust as needed
                # Apply the updated PID constants
                kp = smooth_transition(kp, kp_new, transition_rate)
                ki = smooth_transition(ki, ki_new, transition_rate)
                kd = smooth_transition(kd, kd_new, transition_rate)

                # Reset integral and last_error for smoother tuning
                integral = 0.0
                last_error = 0.0

            # Example usage:
            current_angle = angle_deg  # Example current angle (replace with actual values)
            target_angle = 0  # Example target angle (usually 0 degrees for alignment)


            def adjust_max_speed(angle_deg, max_speed, speed_ranges):
                for angle_range, speed_multiplier in speed_ranges.items():
                    if angle_range[0] <= abs(angle_deg) < angle_range[1]:
                        max_speed *= speed_multiplier
                        break  # Exit loop once the appropriate range is found and speed is adjusted

                return max_speed

            # Example speed ranges and corresponding speed multipliers
            speed_ranges = {
                (0, 5): 1.2,    # If angle is between 0 and 5 degrees, increase speed by 20%
                (5, float('inf')): 0.7  # If angle is 5 degrees or more, decrease speed by 20%
            }

            # Example usage
            current_angle = angle_deg  # Example current angle (replace with actual values)
            max_speed = 0.2  # Initial maximum speed

            # Adjust max_speed based on current angle using the defined speed_ranges
            max_speed = adjust_max_speed(current_angle, max_speed, speed_ranges)

            # Calculate the shortest angle between current angle and target angle
            error_angle = calculate_shortest_angle(current_angle, target_angle)

            # Adjust motor speeds based on the error angle
            left_speed, right_speed = adjust_motor_speeds(error_angle, max_speed)
       
            distance_from_goal = calculate_distance_from_goal((x_red_dot, y_red_dot), start)
            #print(distance_from_goal)

            def euclidean_distance(point1, point2):
                """Calculate the Euclidean distance between two points."""
                return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

            def closest_point_on_line(line_points, point):
                """Find the closest point on the line to a given point."""
                closest_point = line_points[0]
                min_distance = euclidean_distance(line_points[0], point)
                for line_point in line_points:
                    distance = euclidean_distance(line_point, point)
                    if distance < min_distance:
                        min_distance = distance
                        closest_point = line_point
                return closest_point

            def adjust_increment_value(blue_coordinate, red_coordinate, path_coordinates):
                """Adjust the increment value based on the lateral distance from the path."""
                closest_point_blue = closest_point_on_line(path_coordinates, blue_coordinate)
                closest_point_red = closest_point_on_line(path_coordinates, red_coordinate)
                
                lateral_distance_blue = euclidean_distance(blue_coordinate, closest_point_blue)
                lateral_distance_red = euclidean_distance(red_coordinate, closest_point_red)
                
                # Define range for lateral distance adjustment
                min_lateral_distance = 5
                max_lateral_distance = 200
                
                # Define range for increment value adjustment
                min_increment = 5
                max_increment = 30
                
                # Calculate a scaled increment value based on lateral distance
                if lateral_distance_blue < min_lateral_distance and lateral_distance_red < min_lateral_distance:
                    increment_value = max_increment  # Maximum increment value when well-aligned
                elif lateral_distance_blue > max_lateral_distance or lateral_distance_red > max_lateral_distance:
                    increment_value = min_increment  # Minimum increment value if far from the path
                else:
                    # Scale increment value linearly between min and max based on lateral distance
                    range_ratio = (lateral_distance_blue - min_lateral_distance) / (max_lateral_distance - min_lateral_distance)
                    increment_value = min_increment + (max_increment - min_increment) * range_ratio

                print(int(increment_value))
                return int(increment_value)
            
            # Check if the Jetbot is aligned within tolerance
            if abs(error) < 8:
                # If aligned, set equal positive speeds for both motors to drive forward
                left_speed = max_speed*1.4
                right_speed = max_speed*1.4
                if distance_from_goal <= 30:  # Stop if very close to the goal
                    left_speed = 0.0
                    right_speed = 0.0
                    #print(adjust_start_index(lists, start_index, lookahead=lookahead_range))
                    increment_value = adjust_increment_value((x_blue_dot, y_blue_dot), (x_red_dot, y_red_dot), lists)
                    start_index += increment_value
                    if start_index >= len(lists):
                        left_speed = 0.0
                        right_speed = 0.0
                    else:
                        start = lists[start_index]
      

            duration = 0.006  # Movement duration (adjust as needed)
            movement_command = f"move({duration}, {left_speed}, {right_speed})"  # Left Speed, Right Speed

            try:
                s = socket.socket()
                host = '10.0.0.4'  # Replace 'jetbot_ip_address' with the actual IP address of your Jetbot
                port = 12464
                s.connect((host, port))
                s.send(f"set_pid({kp},{ki},{kd},{target_angle});{movement_command}".encode())
                # Close the socket connection after exiting the main loop
                s.close()
            except Exception as e:
                print(f"Error sending instructions to Jetbot: {e}")

            # Display angle information on the frame
            angle_text = f'Angle: {angle_deg:.2f} degrees, PID Output: {output:.2f}'
            cv2.putText(frame, angle_text, (80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    if lists:
        for coord in lists:
            cv2.circle(frame, coord, 1, (255, 255, 5), -1)  # Blue color
            
        cv2.circle(frame, start, 3, (240, 32, 160), -1) 
        cv2.circle(frame, end_point, 5, (0, 255, 0), -1)  
    
    # Display the frame with tracking results
    cv2.imshow('3. Object Tracking', frame)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close all windows
cap.release()
cv2.destroyAllWindows()
