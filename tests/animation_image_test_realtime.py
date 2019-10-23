import cv2 as cv
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
import bretina
import brest
import logging
import array
import time

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

box = [112, 10, 152, 50]
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
input("animation")

images = []
end_time = 5
period = 0.25
start_time = time.time()
while end_time > time.time() - start_time:
    start = time.time()
    images.append([time.time()-start_time, camera.acquire_image()])
    end = time.time()
    sleep_time = period - (end - start) if period > end - start else 0
    time.sleep(sleep_time)

size = (27, 24)
img7_1_template_small = cv.imread('images/img/homescreen/error.png')
img7_1_template_resized = bretina.resize(img7_1_template_small, scale)
template = bretina.separate_animation_template(img7_1_template_resized, size, scale)

for x, image in enumerate(images):
    img = bretina.rectify(image[1], maps, transformation, resolution)
    img = bretina.calibrate_hist(img, histogram_calibration_data)
    img = bretina.calibrate_rgb(img, rgb_calibration_data)
    img = bretina.crop(img, box, scale, border)
    result = []
    for img_template in template:
        result.append(bretina.recognize_image(img, img_template))
    val = max(result)                                               
    images[x] = [image[0],result.index(val) if val > 0.3 else None]
duty_cycle, period = bretina.recognize_animation(images)
    
print(duty_cycle, period)


