import cv2
import sys
import os

sys.path.insert(0, os.path.abspath('..'))
import bretina

"""
test for read_text and crop function
"""

# Uncomment following line to set path to the local installation of the Tesseract OCR
# bretina.TESSERACT_PATH = 'C:\\Program Files\\Tesseract-OCR\\'

# image of aquired homescreen and text box definition
img = cv2.imread('images/hmscr.png')
img6 = cv2.imread("images/text_multiline_2.png")

item3_1 = [237, 210, 266, 233]
item6_date = [300, 12, 474, 40]
item6_time = [370, 42, 474, 70]
item6 = [300, 12, 474, 70]

scale = 3.0

# test of function
img3_1 = bretina.crop(img, item3_1, scale, 4)
img6_date = bretina.crop(img, item6_date, scale, 4)
img6_time = bretina.crop(img, item6_time, scale, 4)

# singleline
print(bretina.read_text(img3_1, 'eng', False))

multiline = bretina.text_rows(img6_date, scale)[0] > 1
print(bretina.read_text(img6_date, multiline=multiline))

multiline = bretina.text_rows(img6_time, scale)[0] > 1
print(bretina.read_text(img6_time, multiline=multiline))

# multiline
multiline = bretina.text_rows(img6, scale)[0] > 1
print(bretina.read_text(img6, multiline=multiline))

# used screen
cv2.imshow("img", img)
cv2.waitKey()
cv2.destroyAllWindows()
