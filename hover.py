from djitellopy import Tello
import time
tello = Tello()

tello.connect()
tello.takeoff()
time.sleep(1)
tello.move_left(20)
time.sleep(1)
tello.rotate_clockwise(90)
time.sleep(1)
tello.move_forward(20)

tello.land()


