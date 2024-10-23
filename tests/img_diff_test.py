import cv2
import sys
import os
import numpy as np
from pprint import pprint as prt

sys.path.insert(0, os.path.abspath('..'))
import bretina

"""
test for recognize_image function
"""

tank = cv2.imread('images/icon/tank.bmp')
tank_white = cv2.imread('images/icon/tank_white.bmp', cv2.IMREAD_UNCHANGED)

tank_template = cv2.imread('images/img/ico_dhw_40.png', cv2.IMREAD_UNCHANGED)
tank_template = bretina.resize(tank_template, 3)

print("{:32}: {:7f}".format("tank x tank", bretina.img_diff(tank, tank_template)))
print("{:32}: {:7f}".format("tank x tank", bretina.img_diff(tank_white, tank_template)))

#
item_11 = cv2.imread('images/item1_1.png', cv2.IMREAD_UNCHANGED)
item_11_src = cv2.imread('images/img/homescreen/quiet.png', cv2.IMREAD_UNCHANGED)
item_11_src = bretina.resize(item_11_src, 3)

print("{:32}: {:7f}".format("quiet x quit", bretina.img_diff(item_11, item_11_src)))

#
item_11 = cv2.imread('images/empty.png', cv2.IMREAD_UNCHANGED)
item_11_src = cv2.imread('images/img/ico_malfunction_40.png', cv2.IMREAD_UNCHANGED)
item_11_src = bretina.resize(item_11_src, 3)

print("{:32}: {}".format("empty x malfunction", bretina.img_diff(item_11, item_11_src)))

#
item_11 = cv2.imread('images/empty.png', cv2.IMREAD_UNCHANGED)
item_11_src = cv2.imread('images/img/homescreen/quiet.png', cv2.IMREAD_UNCHANGED)
item_11_src = bretina.resize(item_11_src, 3)

print("{:32}: {}".format("empty x quiet", bretina.img_diff(item_11, item_11_src)))

#
item_11 = cv2.imread('images/legionella.png', cv2.IMREAD_UNCHANGED)
item_11_src = cv2.imread('images/img/homescreen/legionella.png', cv2.IMREAD_UNCHANGED)
item_11_src = bretina.resize(item_11_src, 3)

print("{:32}: {}".format("legionella x legionella", bretina.img_diff(item_11, item_11_src, edges=True)))

#
item_2 = cv2.imread('images/item2.png', cv2.IMREAD_UNCHANGED)
item_2_src = cv2.imread('images/img/homescreen/floor_standing_alpha.png', cv2.IMREAD_UNCHANGED)
item_2_src = bretina.resize(item_2_src, 3)

print("{:32}: {:7f}".format("floor x floor", bretina.img_diff(item_2, item_2_src, blank=[150, 80, 215, 140])))

#
item_2 = cv2.imread('images/item2.png', cv2.IMREAD_UNCHANGED)
item_2_src = cv2.imread('images/img/homescreen/wall_mounted_alpha.png', cv2.IMREAD_UNCHANGED)
item_2_src = bretina.resize(item_2_src, 3)

print("{:32}: {:7f}".format("wall x floor", bretina.img_diff(item_2, item_2_src, blank=[150, 15, 215, 60])))

#
item_2 = cv2.imread('images/dhw.png', cv2.IMREAD_UNCHANGED)
item_2_src = cv2.imread('images/img/homescreen/dhw.png', cv2.IMREAD_UNCHANGED)
item_2_src = bretina.resize(item_2_src, 3)

print("{:32}: {:7f}".format("dhw x dhw", bretina.img_diff(item_2, item_2_src, bgcolor="#E4F4FF")))

#
item_2 = cv2.imread('images/drt.png', cv2.IMREAD_UNCHANGED)
item_2_src = cv2.imread('images/img/homescreen/daikin_room_thermostat.png', cv2.IMREAD_UNCHANGED)
item_2_src = bretina.resize(item_2_src, 3)

print("{:32}: {:7f}".format("drt x drt", bretina.img_diff(item_2, item_2_src, bgcolor="#E4F4FF")))

#
item_2 = cv2.imread('images/drt.png', cv2.IMREAD_UNCHANGED)
item_2_src = cv2.imread('images/img/homescreen/external_room_thermostat_alpha.png', cv2.IMREAD_UNCHANGED)
item_2_src = bretina.resize(item_2_src, 3)

print("{:32}: {:7f}".format("drt x ert", bretina.img_diff(item_2, item_2_src, bgcolor="#E4F4FF")))

#
item_2 = cv2.imread('images/radiator.png', cv2.IMREAD_UNCHANGED)
item_2_src = cv2.imread('images/img/homescreen/radiator_alpha.png', cv2.IMREAD_UNCHANGED)
item_2_src = bretina.resize(item_2_src, 3)

print("{:32}: {:7f}".format("radiator x radiator", bretina.img_diff(item_2, item_2_src, bgcolor="#E4F4FF")))

#
item_2 = cv2.imread('images/radiator.png', cv2.IMREAD_UNCHANGED)
item_2_src = cv2.imread('images/img/homescreen/underfloor_heating_alpha.png', cv2.IMREAD_UNCHANGED)
item_2_src = bretina.resize(item_2_src, 3)

print("{:32}: {:7f}".format("radiator x ufh", bretina.img_diff(item_2, item_2_src, bgcolor="#E4F4FF")))

#
item_2 = cv2.imread('images/radiator.png', cv2.IMREAD_UNCHANGED)
item_2_src = cv2.imread('images/img/homescreen/fancoil_unit_alpha.png', cv2.IMREAD_UNCHANGED)
item_2_src = bretina.resize(item_2_src, 3)

print("{:32}: {:7f}".format("radiator x fancoil", bretina.img_diff(item_2, item_2_src, bgcolor="#E4F4FF")))

#
item_2 = cv2.imread('images/outdoor_unit4.png', cv2.IMREAD_UNCHANGED)
item_2_src = cv2.imread('images/img/homescreen/outdoor_unit4_alpha.png', cv2.IMREAD_UNCHANGED)
item_2_src = bretina.resize(item_2_src, 3)

print("{:32}: {:7f}".format("ou4 x ou4", bretina.img_diff(item_2, item_2_src, bgcolor="#000")))

#
mallfunction = cv2.imread('images/icon/bell/mallfunction.bmp')
user = cv2.imread('images/icon/bell/user.bmp')
error = cv2.imread('images/img/ico_error_40.png', cv2.IMREAD_UNCHANGED)
profile = cv2.imread('images/img/ico_profile_40.png', cv2.IMREAD_UNCHANGED)

print("{:32}: {:7f}".format("mallfunction x error", bretina.img_diff(mallfunction, error)))
print("{:32}: {:7f}".format("mallfunction x profile", bretina.img_diff(mallfunction, profile)))
print("{:32}: {:7f}".format("user x error", bretina.img_diff(user, error)))
print("{:32}: {:7f}".format("user x profile", bretina.img_diff(user, profile)))

print("{:32}: {:7f}".format("mallfunction x mallfunction", bretina.img_diff(mallfunction, mallfunction)))
print("{:32}: {:7f}".format("error x error", bretina.img_diff(error, error)))
print("{:32}: {:7f}".format("user x user", bretina.img_diff(user, user)))
print("{:32}: {:7f}".format("profile x profile", bretina.img_diff(profile, profile)))


h_c_ico = cv2.imread('images/img/ico_ch_40.png', cv2.IMREAD_UNCHANGED)
cooling = cv2.imread('images/img/ico_cooling_40.png', cv2.IMREAD_UNCHANGED)
heating = cv2.imread('images/img/ico_heating_40.png', cv2.IMREAD_UNCHANGED)
on_off = cv2.imread('images/img/ico_onoff_40.png', cv2.IMREAD_UNCHANGED)
info = cv2.imread('images/img/ico_info_40.png', cv2.IMREAD_UNCHANGED)
set_ico = cv2.imread('images/img/ico_set_40.png', cv2.IMREAD_UNCHANGED)
h_c_ico = bretina.resize(h_c_ico, 3)
cooling = bretina.resize(cooling, 3)
heating = bretina.resize(heating, 3)
on_off = bretina.resize(on_off, 3)
info = bretina.resize(info, 3)
set_ico = bretina.resize(set_ico, 3)


imgs = [f for f in os.listdir("images/icon/round") if f.endswith(".bmp")]

for i in range(0, len(imgs)):
    img = cv2.imread('images/icon/round/'+imgs[i])
    print("{:32}: {:7f}".format(imgs[i] + " x heat/cool", bretina.img_diff(img, h_c_ico)))
    print("{:32}: {:7f}".format(imgs[i] + " x cooling", bretina.img_diff(img, cooling)))
    print("{:32}: {:7f}".format(imgs[i] + " x heating", bretina.img_diff(img, heating)))
    print("{:32}: {:7f}".format(imgs[i] + " x on off", bretina.img_diff(img, on_off)))
    print("{:32}: {:7f}".format(imgs[i] + " x info", bretina.img_diff(img, info)))
    print("{:32}: {:7f}".format(imgs[i] + " x set", bretina.img_diff(img, set_ico)))
    print("")

menu = cv2.imread('images/img/ico_menu_40.png', cv2.IMREAD_UNCHANGED)
radiator = cv2.imread('images/img/ico_radiator_40.png', cv2.IMREAD_UNCHANGED)
underfloor = cv2.imread('images/img/ico_underfloor_40.png', cv2.IMREAD_UNCHANGED)
fancoil = cv2.imread('images/img/ico_fancoil_40.png', cv2.IMREAD_UNCHANGED)
menu = bretina.resize(menu, 3)
radiator = bretina.resize(radiator, 3)
underfloor = bretina.resize(underfloor, 3)
fancoil = bretina.resize(fancoil, 3)

icons = (menu, radiator, underfloor, fancoil)

print('radiator_ico')

confusion = np.zeros((len(icons), len(icons)))

for x in range(len(icons)):
    for y in range(len(icons)):
        confusion[x, y] = bretina.img_diff(icons[x], icons[y])

prt(confusion)