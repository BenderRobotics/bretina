import cv2 as cv
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
import bretina

"""
raaltime test for animaton function
"""

# scale between camera resolution and real display
scale = 3
imgs = [f for f in os.listdir("images/animated_img") if f.endswith(".png")]

# size of template image and load template image
size = (24, 27)
img7_1_template_small = cv.imread('images/img/homescreen/error.png')

images = []
# load carputed images
for x, pic in enumerate(imgs):
    time = x*1
    images.append({'time': time, 'image': cv.imread('images/animated_img/'+pic)})

# recognize image animation, return mean image conformity and period conformity
conformity, period = bretina.recognize_animation(images, img7_1_template_small, size, scale, 2)

print('conformity: ', conformity)
print('period: ', period)


