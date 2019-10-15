import cv2
import sys
import os

sys.path.insert(0, os.path.abspath('..'))
import bretina

"""
test for shape and crop function
"""

# image of captured chessboard
chessboard_img = cv2.imread('images/chessboard_bad.png')
# size of chessboard (number of white/black pairs)
chessboard_size = (15, 8.5)
# size of real display in px
display_size = (480, 272)
# scale between camera resolution and real display
scale = 3
# border (in pixels) around cropped display
border = 4

# test of shape function (from chessboard image make data to undistorted and crop actual surface of display)
maps, transformation, resolution = bretina.get_rectification(chessboard_img, scale, chessboard_size, display_size, border=border)

# test of rectify function (use calibration data to undistorted and crop inserted image)
rectified_img = bretina.rectify(chessboard_img, maps, transformation, resolution)

cv2.imshow("source", chessboard_img)
cv2.imshow("output", rectified_img)
cv2.waitKey()
cv2.destroyAllWindows()
