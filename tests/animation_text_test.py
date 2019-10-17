import cv2 as cv
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
import bretina

"""
test for animaton function
"""
imgs=([f for f in os.listdir("images/animated_text") if f.endswith(".png")])

animated_text = bretina.ReadAnimation()
active = True
i = 0

# animation function test
while active:
    img = cv.imread('images/animated_text/'+imgs[i])
    active = animated_text.unite_animation_text(img)
    i += 1
    
print(bretina.read_text(animated_text.image(), multiline=False))

# final image
cv.imshow("img", animated_text.image())
cv.waitKey()
cv.destroyAllWindows()
