import cv2
import sys
import os

sys.path.insert(0, os.path.abspath('..'))
import bretina

"""
test for read_text and crop function
"""

# image of aquired homescreen and text box definition
img = cv2.imread('images/hmscr.png')
item3_1 = {"box": [237,210,266,233]}
item6_date = {"box": [300, 12, 474, 40]}
item6_time = {"box": [370, 42, 474, 70]}
item6 = {"box": [300, 12, 474, 70]}

# test of function
img3_1 = bretina.crop(img, item3_1["box"], 3, 4)
img6_date = bretina.crop(img, item6_date["box"], 3, 4)
img6_time = bretina.crop(img, item6_time["box"], 3, 4)
img6 = bretina.crop(img, item6["box"], 3, 4)


print(bretina.read_text(img3_1, 3, 'eng', False))
# sinleline
print(bretina.read_text(img6_date, 3))
print(bretina.read_text(img6_time, 3))
# multiline
print(bretina.read_text(img6, 3))
# used screen
cv2.imshow("img", img)
cv2.waitKey()
cv2.destroyAllWindows()


