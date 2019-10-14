import cv2
import sys
import os

sys.path.insert(0, os.path.abspath('..'))
import bretina

"""
test for detection of the number of cols and its regions
"""

images = ['images/text_multicols_1.png',
          'images/text_multicols_2.png']

for path in images:
    img = cv2.imread(path)
    cnt, regions = bretina.text_cols(img)
    print("Found {cnt} col(s) of text in {path} at {regions}".format(cnt=cnt, path=path, regions=regions))
