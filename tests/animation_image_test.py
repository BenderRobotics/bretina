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
imgs=([f for f in os.listdir("images/animated_img") if f.endswith(".png")])

size = (27, 24)
img7_1_template_small = cv.imread('images/img/homescreen/error.png')
img7_1_template_resized = bretina.resize(img7_1_template_small, scale)
template = bretina.separate_animation_template(img7_1_template_resized, size, scale)
images = []
for x, pic in enumerate(imgs):
    result = []
    time = x*0.25
    img = cv.imread('images/animated_img/'+pic)
    for img_template in template:
        result.append(bretina.recognize_image(img, img_template))
    val = max(result)
    images.append([time, result.index(val) if val > 0.3 else None])
duty_cycle, period = bretina.recognize_animation(images)
    
print(duty_cycle, period)


