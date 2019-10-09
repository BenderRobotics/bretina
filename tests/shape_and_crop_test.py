"""
test for shape and crop function
"""

import cv2
import sys
import os

sys.path.insert(0, os.path.abspath('..'))
import bretina

#: image of captured chessboard
chessboard_img = cv2.imread('images/chessboard.png')
#: size of chessboard (number of white/black pairs)
chessboard_size = 15, 8.5
#: size of real display in px
display_size = 480, 272
#: scale between camera resolution and real display
scale = 3
#: border (in pixels) around cropped display
border = 4

# test of shape function (from chessboard image make data to undistorted and crop actual surface of display)
calibration_data = bretina.shape(chessboard_img, chessboard_size, display_size, scale, border)

# test of crop function (use calibration data to undistorted and crop inserted image)
cv2.imshow("img", bretina.crop(chessboard_img, calibration_data))
cv2.waitKey()
cv2.destroyAllWindows()
