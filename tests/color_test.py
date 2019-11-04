import cv2
import sys
import os
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.abspath('..'))
import bretina

"""
test for shape and crop function
"""

print(bretina.color("#FF0000"))
print(bretina.color("#F00"))
print(bretina.color("RED "))
print(bretina.color("red"))
print(bretina.color((0, 0, 255)))

print(bretina.color("very red "))    # will raise exception
