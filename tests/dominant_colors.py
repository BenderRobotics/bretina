import cv2
import sys
import os

sys.path.insert(0, os.path.abspath('..'))
import bretina

"""
test for active_color function
"""
img = cv2.imread('../docs/_static/fig_active_color_1.png')
colors = bretina.active_color(img)
print(colors)
