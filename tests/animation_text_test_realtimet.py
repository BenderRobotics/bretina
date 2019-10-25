import cv2 as cv
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
import bretina
import brest
import logging
import array

from brest.misc import OCR

"""
raaltime test for animaton function
"""

# size of chessboard (number of white/black pairs)
chessboard_size = (15, 8.5)
# size of real display in px
display_size = (480, 272)
# scale between camera resolution and real display
scale = 3
# border (in pixels) around cropped display
border = 4

box = [82, 120, 450, 150]
# calibrate camera
args = {'interface': {'index': 0}}
camera = brest.cameras.Backfly(args)
input("chessboard")
chessboard_img = camera.acquire_image()
maps, transformation, resolution = bretina.get_rectification(chessboard_img, scale, chessboard_size, display_size, border=border)
chessboard_img = bretina.rectify(chessboard_img, maps, transformation, resolution)
input("red")
red_img = camera.acquire_image()
red_img = bretina.rectify(red_img, maps, transformation, resolution)
input("blue")
blue_img = camera.acquire_image()
blue_img = bretina.rectify(blue_img, maps, transformation, resolution)
input("green")
green_img = camera.acquire_image()
green_img = bretina.rectify(green_img, maps, transformation, resolution)
histogram_calibration_data, rgb_calibration_data = bretina.color_calibration(chessboard_img, chessboard_size, red_img, green_img, blue_img)

# animation function test
input("animeted text")
animated_text = bretina.SlidingTextReader()
active = True

while active:
    img = camera.acquire_image()
    img = bretina.rectify(img, maps, transformation, resolution)
    img = bretina.calibrate_hist(img, histogram_calibration_data)
    img = bretina.calibrate_rgb(img, rgb_calibration_data)
    img = bretina.crop(img, box, scale, border)
    active = animated_text.unite_animation_text(img)

print(bretina.read_text(animated_text.get_image(), multiline=False))
# final image
cv.imshow("img", animated_text.get_image())
cv.waitKey()
cv.destroyAllWindows()
