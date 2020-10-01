import cv2
import sys
import os

sys.path.insert(0, os.path.abspath('..'))
import bretina

"""
test for active_color function
"""
img = cv2.imread('images/item1_1.png')
color = bretina.active_color(img, n=2)
print(bretina.color_str(color))

color = bretina.active_color(img, n=2, bgcolor="black")
print(bretina.color_str(color))

color = bretina.dominant_color(img, n=2)
print(bretina.color_str(color))
