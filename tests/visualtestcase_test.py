import os
import sys
import cv2
import logging
import brest

sys.path.insert(0, os.path.abspath('..'))
from bretina import VisualTestCase

vtc = VisualTestCase()



vtc.img = cv2.imread('images/hmscr.png')

try:
    vtc.assertEmpty([230, 130, 250, 150])
    vtc.assertNotEmpty([242, 180, 263, 197])
    vtc.assertEmpty([242, 180, 263, 197])
except:
    pass


vtc.save_img(vtc.img, "test1", [50, 2, 150, 200], "Expected area to be blank")
vtc.save_img(vtc.img, "test2", [460, 100, 475, 130], "Expected area to be blank")

# Calibration, pre-processing
# --------------------------------
chess_img = cv2.imread('images/uncropped/chessboard.png')
r_img = cv2.imread('images/uncropped/red.png')
g_img = cv2.imread('images/uncropped/green.png')
b_img = cv2.imread('images/uncropped/blue.png')
raw_img = cv2.imread("images/uncropped/homescreen.png")

vtc.calibrate(chess_img, r_img, g_img, b_img)
vtc.img = vtc._preprocess(raw_img)

# Save img
# ---------------------------------
vtc.save_img(vtc.img, "test1", [50, 2, 150, 200], "Expected area to be blank")
vtc.save_img(vtc.img, "test2", [50, 100, 150, 200], "Expected area to be blank")

cv2.imshow("original", raw_img)
cv2.imshow("pre-processing", vtc.img)
cv2.waitKey()
