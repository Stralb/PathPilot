import socket
from jetbot import Robot
import time
import logging

robot = Robot()

# Global variables for PID control
kp = 1.0
ki = 0.0
kd = 0.0
target_angle = 0.0
integral = 0.0
last_error = 0.0

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def pid_controller(current_angle):
    global integral, last_error
    error = target_angle - current_angle
    integral += error
    derivative = error - last_error
    output = kp * error + ki * integral + kd * derivative
    last_error = error
    return output

def parse_instruction(instruction):
    try:
        command, value = instruction.strip().split('(')
        value = value.strip(')')
        return command, value
    except ValueError:
        return None, None

def execute_instruction(instruction):
    global kp, ki, kd, target_angle
    command, value = parse_instruction(instruction)
    if command == "set_pid":
        kp, ki, kd, target_angle = map(float, value.split(','))
        logger.info(f"PID Parameters Updated: KP={kp}, KI={ki}, KD={kd}, Target Angle={target_angle}")
        return True
    elif command == "move":
        duration, left_speed, right_speed = map(float, value.split(','))
        move(duration, left_speed, right_speed)
        return True
    else:
        logger.error("Invalid command")
        robot.stop()
        return False

def move(duration, left_speed, right_speed):
    robot.set_motors(left_speed, right_speed)
    time.sleep(duration)
    robot.stop()

def main():
    s = socket.socket()
    port = 12464
    s.bind(('', port))
    s.listen(5)
    print("Socket is up and running.")

    while True:
        try:
            c, addr = s.accept()
            print("Socket Up and running with a connection from", addr)

            received_data = c.recv(1024).decode().lower()

            if received_data.strip() == "bye":
                break

            instructions = received_data.split(';')
            for instruction in instructions:
                if instruction:
                    if not execute_instruction(instruction):
                        break

            c.close()
        except Exception as e:
            print(f"Error in socket handling: {e}")
            # Optionally, handle the error or retry the connection here


if __name__ == "__main__":
    main()
