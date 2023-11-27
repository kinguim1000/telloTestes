import cv2
from djitellopy import Tello

tello = Tello()
tello.connect()

tello.streamon()
frame_read = tello.get_frame_read()
print(tello.get_battery())

cv2.imwrite("picture.png", frame_read.frame)

