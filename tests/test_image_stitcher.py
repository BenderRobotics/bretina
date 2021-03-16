import cv2 as cv
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath('..'))
from bretina import ImageStitcher


"""
Test of the image stitching
"""

for folder in ["01"]:
    data = [f for f in os.listdir("test_image_stitcher\\" + folder) if f.endswith((".bmp", ".png", ".jpg"))]
    stitcher = ImageStitcher(cut_off_bg_threshold=80, bgcolor=None, cut_off_bg=True)

    finished = False
    i = 0

    imgs = [cv.imread(f"test_image_stitcher\\{folder}\\{f}") for f in data]

    while i < len(data) and not finished:
        finished, merged_img = stitcher.add(imgs[i])
        i += 1

    # Show output image
    cv.imshow("folder", merged_img)
    cv.waitKey()
    cv.destroyAllWindows()

exit()

for folder in ["05"]:
    data = [f for f in os.listdir("test_image_stitcher\\" + folder) if f.endswith((".bmp", ".png", ".jpg"))]
    stitcher = ImageStitcher(cut_off_bg_threshold=80)

    finished = False
    i = 0

    while i < len(data):
        img = cv.imread(f"test_image_stitcher\\{folder}\\{data[i]}")
        finished, merged_img = stitcher.add(img, cut_off_bg=False)
        i += 1

    # Show output image
    cv.imshow("folder", merged_img)
    cv.waitKey()
    cv.destroyAllWindows()
