# -*- coding: utf-8 -*-
# import os
import sys
import cv2
import os
import logging

sys.path.insert(0, os.path.abspath('..'))
from bretina import VisualTestCase

vtc = VisualTestCase()
vtc.img = cv2.imread('images/hmscr.png')

#try:
#    vtc.assertEmpty([230, 130, 250, 150])
#except:
#    pass
#
#try:
#    vtc.assertNotEmpty([242, 180, 263, 197])
#except:
#    pass
#
#try:
#    vtc.assertEmpty([242, 180, 263, 197], msg="test1Lorem ipsum dolor sit amet, consectetuer adipiscing elit. \n Maecenas ipsum velit, consectetuer eu lobortis ut, dictum at dui. \n  Proin mattis lacinia justo.")
#except Exception as ex:
#    pass
#

vtc.save_img(vtc.img, name="test1", border_box=[50, 2, 150, 200], msg="test1Lorem ipsum dolor sit amet, consectetuer adipiscing elit. \n\n\n Maecenas ipsum velit, consectetuer eu lobortis ut, dictum at dui. \n  Proin mattis lacinia justo.\n\n-   4. Complex is better than complicated.\n?            ^                     ---- ^\n+   4. Complicated is better than complex.\n?           ++++ ^                      ^\n")
vtc.save_img(vtc.img, name="test2", border_box=[460, 100, 475, 130], msg="Λορεμ ιπσθμ δολορ σιτ αμετ, μινιμ cομπλεcτιτθρ δεφινιτιονεμ qθο αν, ιδ εοσ ασσθεvεριτ cομπλεcτιτθρ, εθμ ιμπεδιτ δισσεντιασ τε. Ηισ cθ αθγθε φαcιλισ. Δθο εξ ιριθρε νονθμεσ cονvενιρε. Cθμ ινσολενσ ατομορθμ ρεπθδιαρε. Λορεμ ιπσθμ δολορ σιτ αμετ, μινιμ cομπλεcτιτθρ δεφινιτιονεμ qθο αν, ιδ εοσ ασσθεvεριτ cομπλεcτιτθρ, εθμ ιμπεδιτ δισσεντιασ τε. Ηισ cθ αθγθε φαcιλισ. Δθο εξ ιριθρε νονθμεσ cονvενιρε. Cθμ ινσολενσ ατομορθμ ρεπθδιαρε. Λορεμ ιπσθμ δολορ σιτ αμετ, μινιμ cομπλεcτιτθρ δεφινιτιονεμ qθο αν, ιδ εοσ ασσθεvεριτ cομπλεcτιτθρ, εθμ ιμπεδιτ δισσεντιασ τε. Ηισ cθ αθγθε φαcιλισ. Δθο εξ ιριθρε νονθμεσ cονvενιρε. Cθμ ινσολενσ ατομορθμ ρεπθδιαρε.")

quit()

# Calibration, pre-processing
# --------------------------------
chess_img = cv2.imread('images/uncropped/chessboard.png')
r_img = cv2.imread('images/uncropped/red.png')
g_img = cv2.imread('images/uncropped/green.png')
b_img = cv2.imread('images/uncropped/blue.png')
raw_img = cv2.imread("images/uncropped/homescreen.png")

vtc.img = vtc._preprocess(raw_img)

# Save img
# ---------------------------------
vtc.save_img(vtc.img, "test1", [50, 2, 150, 200], "Expected area to be blank")
vtc.save_img(vtc.img, "test2", [50, 100, 150, 200], "Expected area to be blank")

cv2.imshow("original", raw_img)
cv2.imshow("pre-processing", vtc.img)
cv2.waitKey()
