import cv2 as cv
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath('..'))
import bretina

bretina.TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR"

"""
test for animation function
"""

imgs = [f for f in os.listdir("images/animated_text") if f.endswith(".png")]
animated_text = bretina.SlidingTextReader()

# RGB image: animation function test 1
# ------------------------------------
active = True
i = 0

while active:
    img = cv.imread('images/animated_text/' + imgs[i])
    active = animated_text.unite_animation_text(img)
    i += 1

print(bretina.read_text(animated_text.get_image(), multiline=False))

# final image
cv.imshow("img", animated_text.get_image())
cv.waitKey()
cv.destroyAllWindows()

# Gray scale image: animation function test
# -----------------------------------------
active = True
i = 0

while active:
    img = cv.imread('images/animated_text/' + imgs[i], cv.IMREAD_GRAYSCALE)
    active = animated_text.unite_animation_text(img)
    i += 1

print(bretina.read_text(animated_text.get_image(), multiline=False))

# final image
cv.imshow("img", animated_text.get_image())
cv.waitKey()
cv.destroyAllWindows()

# RGB image (lightness only): animation function test
# ---------------------------------------------------
active = True
i = 0

while active:
    img = cv.imread('images/animated_text/' + imgs[i])
    img = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    img = img[:, :, 1]
    active = animated_text.unite_animation_text(img)
    i += 1

print(bretina.read_text(animated_text.get_image(), multiline=False))

# final image
cv.imshow("img", animated_text.get_image())
cv.waitKey()
cv.destroyAllWindows()

# RGB image: animation function test 2
# ------------------------------------
active = True
i = 0
imgs = [f for f in os.listdir("images/animated_text_2") if f.endswith(".bmp")]

while active:
    img = cv.imread('images/animated_text_2/' + imgs[i])
    active = animated_text.unite_animation_text(img)
    i += 1

print(bretina.read_text(animated_text.get_image(), multiline=False))

# final image
cv.imshow("img", animated_text.get_image())
cv.waitKey()
cv.destroyAllWindows()
