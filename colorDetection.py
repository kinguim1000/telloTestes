import numpy as np
import cv2
from djitellopy import Tello

tello = Tello()
tello.connect()
tello.streamon()
tello.takeoff()

cv2.namedWindow("Multiple Color Detection in Real-Time", cv2.WINDOW_NORMAL)

while True:
    img = tello.get_frame_read().frame

    hsvFrame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Set range for red color and define a mask
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    # Set range for green color and define a mask
    green_lower = np.array([25, 52, 72], np.uint8)
    green_upper = np.array([102, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

    # Set range for blue color and define a mask
    blue_lower = np.array([94, 80, 2], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

    kernel = np.ones((5, 5), "uint8")

    red_mask = cv2.dilate(red_mask, kernel)
    res_red = cv2.bitwise_and(img, img, mask=red_mask)

    green_mask = cv2.dilate(green_mask, kernel)
    res_green = cv2.bitwise_and(img, img, mask=green_mask)

    blue_mask = cv2.dilate(blue_mask, kernel)
    res_blue = cv2.bitwise_and(img, img, mask=blue_mask)

    def track_color(contour, color):
        for pic, c in enumerate(contour):
            area = cv2.contourArea(c)
            if area > 300:
                x, y, w, h = cv2.boundingRect(c)
                img = cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, color + " Color", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    track_color(contours, (0, 0, 255))  # Red

    contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    track_color(contours, (0, 255, 0))  # Green

    contours, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    track_color(contours, (255, 0, 0))  # Blue

    cv2.imshow("Multiple Color Detection in Real-Time", img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        tello.land()
        cv2.destroyAllWindows()
        break
