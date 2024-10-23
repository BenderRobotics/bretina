import cv2
import sys
import os

sys.path.insert(0, os.path.abspath('..'))
import bretina

"""
test for read function
"""

#: image of 
img = cv2.imread('images/hmscr.png')
item3_1 = {"box": [237,210,266,233]}
item6_date = {"box": [300, 12, 474, 40]}
item6_time = {"box": [370, 42, 474, 70]}
# test of 
img_3_1 = bretina.item_crop_box(img, item3_1, 3, 4)
img_6_date = bretina.item_crop_box(img, item6_date, 3, 4)
img_6_time = bretina.item_crop_box(img, item6_time, 3, 4)
print(bretina.read_text(img_3_1, 'eng'))
print(bretina.read_text(img_6_date, 'eng'))
print(bretina.read_text(img_6_time, 'eng', 'm'))
# test of 
cv2.imshow("img", img)
cv2.waitKey()
cv2.destroyAllWindows()


