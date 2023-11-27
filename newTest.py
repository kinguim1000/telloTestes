import cv2

import numpy as np

from djitellopy import Tello

from ultralytics import YOLO

import sys

sys.path.insert(0, r"C:\Users\abeer\OneDrive\Desktop\ultralytics")

from ultralytics.utils.ops import non_max_suppression, scale_coords

# Load the YOLOv8 model

model = YOLO(r"C:\\FYP\\Tello Programs\\My TELLO Journey\\TELLO Ball Tracker(TBT)\\Tello Ball Tracker - TBT.v3i.yolov8\\runs\detect\\train\\weights\\best.pt")

# Initialize DJI Tello

tello = Tello()

tello.connect()

tello.streamon()

# Initial speed

S = 60

# PID coefficients for the bounding box's x, y, and area coordinates

pid_x = [0.5, 0.5, 0]

pid_y = [0.5, 0.5, 0]

pid_area = [0.5, 0.5, 0]

# Initialize the error and integral terms

error_x, integral_x = 0, 0

error_y, integral_y = 0, 0

error_area, integral_area = 0, 0

# The target area of the bounding box (area of the frame to be covered by the bounding box)

target_area = 720*960/5

# The target position of the bounding box's center (center of the frame)

target_x = 960 // 2

target_y = 720 // 2

# Take off

tello.takeoff()

# Fly to human height

tello.move_up(160)

while True:

    # Get the current frame

    frame_read = tello.get_frame_read()

    frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)

    frame = cv2.resize(frame, (960, 720))

    # Perform object detection

    results = model.predict(frame, show=False, conf=0.9)

    # If an object was detected

    if results[0] is not None and len(results[0].boxes) > 0:

        # Get the bounding box with the highest objectness score

        max_score_idx = results[0].boxes.data[:, 4].argmax()
        x1, y1, x2, y2 = results[0].boxes.data[max_score_idx, :4].tolist()

        # Calculate the bounding box's area and center

        area = (x2 - x1) * (y2 - y1)

        center_x = (x1 + x2) / 2

        center_y = (y1 + y2) / 2

        # Calculate the error terms

        error_x_prev = error_x

        error_y_prev = error_y

        error_area_prev = error_area

        error_x = target_x - center_x

        error_y = target_y - center_y

        error_area = target_area - area

        # Calculate the integral terms

        integral_x += error_x

        integral_y += error_y

        integral_area += error_area

        # Calculate the derivative terms

        derivative_x = error_x - error_x_prev

        derivative_y = error_y - error_y_prev

        derivative_area = error_area - error_area_prev

        # Calculate the adjustment values for the drone's position

        adjustment_x = sum([pid_x[i] * error_x for i, error_x in enumerate([error_x, integral_x, derivative_x])])

        adjustment_y = sum([pid_y[i] * error_y for i, error_y in enumerate([error_y, integral_y, derivative_y])])

        adjustment_area = sum([pid_area[i] * error_area for i, error_area in enumerate([error_area, integral_area, derivative_area])])

        # Adjust the drone's position

        if adjustment_x > 0:

            tello.move_right(min(S, adjustment_x))

        elif adjustment_x < 0:

            tello.move_left(min(S, -adjustment_x))

        if adjustment_y > 0:

            tello.move_up(min(S, adjustment_y))

        elif adjustment_y < 0:

            tello.move_down(min(S, -adjustment_y))

        if adjustment_area > 0:

            tello.move_forward(min(S, adjustment_area))

        elif adjustment_area < 0:

            tello.move_back(min(S, -adjustment_area))

    # If no object was detected

    else:

        # Hover in place

        tello.send_rc_control(0, 0, 0, 0)

    # Display the frame

    cv2.imshow('Tello Tracking...', frame)

    # Break the loop if 'q' is pressed

    if cv2.waitKey(1) & 0xFF == ord('q'):

        tello.land()

        break

cv2.destroyAllWindows()