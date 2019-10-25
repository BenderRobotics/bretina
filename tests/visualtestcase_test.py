import os
import sys
import cv2
import logging
import brest

sys.path.insert(0, os.path.abspath('..'))
from bretina import VisualTestCase

vtc = VisualTestCase()

# Calibration, pre-processing
# --------------------------------
chess_img = cv2.imread('images/uncropped/chessboard.png')
r_img = cv2.imread('images/uncropped/red.png')
g_img = cv2.imread('images/uncropped/green.png')
b_img = cv2.imread('images/uncropped/blue.png')
raw_img = cv2.imread("images/uncropped/homescreen.png")

vtc.calibrate(chess_img, r_img, g_img, b_img)
pre_img = vtc._preprocess(raw_img)

cv2.imshow("original", raw_img)
cv2.imshow("pre-processing", pre_img)
cv2.waitKey()
