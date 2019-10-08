"""
test for shape and crop function
"""

import cv2
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from bretina import shape, crop

#: image of captured chessboard
chessboard_img = cv2.imread('images/chessboard.png')
#: size of chessboard (number of white/black pairs)
chessboard_size = 15, 8.5
#: size of real display in px
display_size = 480, 272
#: scale between camera resolutin and real display
scale = 3
#: border (in pixels) around cropped dsplay
border = 4

# test of shape function (from chessboard image make data to undistord and crop actual surface of display)
calibration_data = shape(chessboard_img, chessboard_size, display_size, scale, border)

# test of crop function (use calibration data to undistord and crop inserted image)
cv2.imshow("img",crop(chessboard_img, calibration_data))
cv2.waitKey()
cv2.destroyAllWindows()
