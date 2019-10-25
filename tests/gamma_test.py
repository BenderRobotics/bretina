import cv2
import sys
import os
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.abspath('..'))
import bretina

"""
test for shape and crop function
"""

for path in ['images/gamma_01.png',
             'images/gamma_02.png',
             'images/gamma_03.png',
             'images/gamma_04.png']:
    img = cv2.imread(path)
    gamma = bretina.gamma_calibration(img)
    plt.subplot(3, 1, 1)
    plt.imshow(img)
    plt.subplot(3, 1, 2)
    plt.imshow(bretina.adjust_gamma(img, gamma))
    plt.subplot(3, 1, 3)
    plt.imshow(cv2.imread('images/gamma_01.png'))
    plt.show()
