import numpy as np
import cv2 as cv
import sys
import os

sys.path.insert(0, os.path.abspath('..'))

from bretina.polyline import get_polyline_coordinates

for path in os.listdir('images/curves/'):
    if path.endswith(('.png', '.jpg', '.bmp')):
        img = cv.imread(os.path.join('images/curves/', path), cv.IMREAD_UNCHANGED)
        area = (0, 0, img.shape[1], img.shape[0])
        try:
            paths = get_polyline_coordinates(img, area, scale=1, threshold=35, blank=None, suppress_noise=False, max_line_gab=5)
            print(paths)
        except Exception as ex:
            print(ex)
