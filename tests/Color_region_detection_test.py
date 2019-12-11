import cv2
import sys
import os

sys.path.insert(0, os.path.abspath('..'))
import bretina

"""
test Color region detection
"""

scale = 3
img = cv2.imread('images/color_region/2019-12-09_12-41-58-833___main__.TestVisualTextMenu.test_menu_texts-src.png')

# get box
box = bretina.color_region_detection(img, '#1360FA', scale)
print('box: ' + str(box))

# show box
if box is not None:
    img =  bretina.draw_border(img, box, scale)
    cv2.imshow('a',img)
    cv2.waitKey()
    cv2.destroyAllWindows()
