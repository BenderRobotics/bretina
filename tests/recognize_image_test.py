import cv2
import sys
import os

sys.path.insert(0, os.path.abspath('..'))
import bretina

"""
test for recognize_image function
"""

# image of aquired homescreen, box and template images definition 
img = cv2.imread('images/hmscr.png')
item1_1 = {"box": [381,150,421,195], "images": ['quiet.png']}
item2 = {"box": [255,155,345,255], "images": ['wall_mounted.png','floor_standing.png']}
item7 = {"box": [112,10,152,50], "images": ['error.png','malfunction.png']}
path = 'images/homescreen3/'

img1_1 = bretina.crop(img, item1_1["box"], 3, 4)
img2 = bretina.crop(img, item2["box"], 3, 4)
img7 = bretina.crop(img, item7["box"], 3, 4)

print(bretina.recognize_image(img1_1, item1_1["images"], path))
print(bretina.recognize_image(img2, item2["images"], path))
print(bretina.recognize_image(img7, item7["images"], path))
# used screen
cv2.imshow("img", img)
cv2.waitKey()
cv2.destroyAllWindows()


