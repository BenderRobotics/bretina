import cv2
import sys
import os

sys.path.insert(0, os.path.abspath('..'))
import bretina

"""
test for detection of the number of cols and its regions
"""

images = ['images/cols/cols2-001.jpg',
          'images/cols/cols2-002.jpg',
          'images/cols/cols2-003.png',
          'images/cols/cols2-004.png',
          'images/cols/cols2-005.png',
          'images/cols/cols2-006.png',
          'images/cols/cols2-007.png',
          'images/cols/cols2-008.png',
          'images/cols/cols2-009.png',
          'images/cols/cols2-010.png',
          'images/cols/cols2-011.png',
          'images/cols/cols4-001.png',
          'images/cols/cols4-002.png',]

for path in images:
    print(path)

    img = cv2.imread(path)

    if 'cols2' in path:
        count = 2
    elif 'cols4' in path:
        count = 4
    else:
        count = 1

    regions = bretina.split_cols(img, 3, col_count=count)
    for region in regions:
        img = bretina.draw_border(img, [region[0], 0, region[1], img.shape[0]-1])

    cv2.imshow('', img)
    cv2.waitKey()

print('done')