import cv2
import sys
import os

sys.path.insert(0, os.path.abspath('..'))
import bretina

"""
test for color_calibratin functions (histogram and rgb calibration)
"""

# cropped image of captured chessboard, red, blue and green screen
chessboard_img = cv2.imread('images/ch.png')
red_img = cv2.imread('images/red.png')
green_img = cv2.imread('images/green.png')
blue_img = cv2.imread('images/blue.png')
# size of chessboard (number of white/black pairs)
chessboard_size = (15, 8.5)

# get calibration data for histogram and color calibration
histogram_calibration_data, rgb_calibration_data = bretina.color_calibration(chessboard_img, chessboard_size, red_img, green_img, blue_img)
# test of calibration functions
ch2 = bretina.calibrate_hist(chessboard_img, histogram_calibration_data)
cv2.imshow("img", ch2)
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imshow("img", bretina.calibrate_rgb(ch2, rgb_calibration_data))
cv2.waitKey()
cv2.destroyAllWindows()
