import cv2
import sys
import os

sys.path.insert(0, os.path.abspath('..'))
import bretina

"""
test for detection of the number of rows
"""

images = ['images/text_multiline_1.png',
          'images/text_multiline_2.png',
          'images/text_multiline_3.png',
          'images/text_multiline_4.png',
          'images/text_multiline_5.png']

for path in images:
    img = cv2.imread(path)
    cnt, regions = bretina.text_rows(img, 3)
    print("Found {cnt} row(s) of text in {path} at {regions}".format(cnt=cnt, path=path, regions=regions))
