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

# size of template image
size = (24, 27)
img7_1_template_small = cv.imread('images/img/homescreen/error.png')
img7_1_template_resized = bretina.resize(img7_1_template_small, scale)
template = bretina.separate_animation_template(img7_1_template_resized, size, scale)
images = []

for x, pic in enumerate(imgs):
    
    time = x*0.25
    images.append({'time': time, 'image': cv.imread('images/animated_img/'+pic)})

conformity, period = bretina.recognize_animation(images, template, 0.5)

print('conformity: ', conformity)
print('period: ', period)


