import cv2
import sys
import os
import numpy as np
from pprint import pprint as prt

sys.path.insert(0, os.path.abspath('..'))
import bretina

"""
test for image histogram difference function 

"""

#
item_11 = cv2.imread('images/smart_energy.png', cv2.IMREAD_UNCHANGED)
item_11_src = cv2.imread('images/img/homescreen/smart_energy_1_tower.png', cv2.IMREAD_UNCHANGED)
item_11_src = bretina.resize(item_11_src, 3)

print("{:32}: {}".format("smart energy x smart energy", bretina.img_hist_diff(item_11, item_11_src, bgcolor="black")))

item_rgb = cv2.cvtColor(item_11, cv2.COLOR_BGR2RGB)
print("{:32}: {}".format("swap B and R chanel", bretina.img_hist_diff(item_rgb, item_11_src, bgcolor="black")))

item_11[:,:,0] +=40
print("{:32}: {}".format("blue chanel + 40", bretina.img_hist_diff(item_11, item_11_src, bgcolor="black")))
