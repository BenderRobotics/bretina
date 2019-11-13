import cv2
import sys
import os

sys.path.insert(0, os.path.abspath('..'))
import bretina

"""
test for active_color function
"""
img = cv2.imread('images/2019-11-13_10-51-03___main__.TestHomeScreen.test_item31-src.png')
img = bretina.crop(img, [233, 205, 267, 239], 3)
color = bretina.active_color(img)
print(bretina.color_str(color))

color = bretina.active_color(img, bgcolor="black")
print(bretina.color_str(color))

color = bretina.dominant_color(img)
print(bretina.color_str(color))

img = cv2.imread('images/2019-11-13_10-51-13___main__.TestHomeScreen.test_item31-src.png')
img = bretina.crop(img, [233, 205, 267, 239], 3)
color = bretina.active_color(img, bgcolor="black")
print(bretina.color_str(color))
