import cv2
import sys
import os

sys.path.insert(0, os.path.abspath('..'))
import bretina

"""
test for recognize_image function
"""


img1_1 = cv2.imread('images/item1_1.png')
img2 = cv2.imread('images/item2.png')
tank = cv2.imread('images/tank.bmp')

img1_1_template_small = cv2.imread('images/img/homescreen/quiet.png')
img2_template_1_small = cv2.imread('images/img/homescreen/wall_mounted.png')
img2_template_2_small = cv2.imread('images/img/homescreen/floor_standing.png')

# resize image
img1_1_template_resized = bretina.resize(img1_1_template_small, 3)
img2_template_1_resized = bretina.resize(img2_template_1_small, 3)
img2_template_2_resized = bretina.resize(img2_template_2_small, 3)


print("{:14}: {}".format("quiet x quiet", bretina.recognize_image(img1_1, img1_1_template_resized)))
print("{:14}: {}".format("floor x wall", bretina.recognize_image(img2, img2_template_1_resized)))
print("{:14}: {}".format("floor x floor", bretina.recognize_image(img2, img2_template_2_resized)))
