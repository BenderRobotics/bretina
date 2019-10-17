import cv2
import sys
import os

sys.path.insert(0, os.path.abspath('..'))
import bretina

"""
test for recognize_image function
"""

tank = cv2.imread('images/icon/tank.bmp')
tank_white = cv2.imread('images/icon/tank_white.bmp')


tank_template = cv2.imread('images/img/ico_dhw_40.png')


# resize image
tank_template_resized = bretina.resize(tank_template, 3)

print('tank')
print("{:14}: {}".format("tank blue x tank", bretina.recognize_image(tank, tank_template_resized)))
print("{:14}: {}".format("tank white x tank", bretina.recognize_image(tank_white, tank_template_resized)))
print("")

mall = cv2.imread('images/icon/bell/mallfunction.bmp')
user = cv2.imread('images/icon/bell/user.bmp')


error = cv2.imread('images/img/ico_error_40.png')
prof = cv2.imread('images/img/ico_profile_40.png')


# resize image
error = bretina.resize(error, 3)
prof = bretina.resize(prof, 3)
print('bell_shape')
print("{:14}: {}".format("bell x bell", bretina.recognize_image(mall, error)))
print("{:14}: {}".format("bell x user", bretina.recognize_image(mall, prof)))
print("{:14}: {}".format("user x bell", bretina.recognize_image(user, error)))
print("{:14}: {}".format("user x user", bretina.recognize_image(user, prof)))
print("")

h_c_ico = cv2.imread('images/img/ico_ch_40.png')
cooling = cv2.imread('images/img/ico_cooling_40.png')
heating = cv2.imread('images/img/ico_heating_40.png')
onoff = cv2.imread('images/img/ico_onoff_40.png')
info = cv2.imread('images/img/ico_info_40.png')
set_ico = cv2.imread('images/img/ico_set_40.png')
h_c_ico = bretina.resize(h_c_ico, 3)
cooling = bretina.resize(cooling, 3)
heating = bretina.resize(heating, 3)
onoff = bretina.resize(onoff, 3)
info = bretina.resize(info, 3)
set_ico = bretina.resize(set_ico, 3)


imgs= [f for f in os.listdir("images/icon/round") if f.endswith(".bmp")]

for i in range(0, len(imgs)):
    img = cv2.imread('images/icon/round/'+imgs[i])
    print(imgs[i])
    print("{:14}: {}".format("x heat/cool", bretina.recognize_image(img, h_c_ico)))
    print("{:14}: {}".format("x cooling", bretina.recognize_image(img, cooling)))
    print("{:14}: {}".format("x heating", bretina.recognize_image(img, heating)))
    print("{:14}: {}".format("x onoff", bretina.recognize_image(img, onoff)))
    print("{:14}: {}".format("x info", bretina.recognize_image(img, info)))
    print("{:14}: {}".format("x set", bretina.recognize_image(img, set_ico)))
    print("")

menu = cv2.imread('images/img/ico_menu_40.png')
radiator = cv2.imread('images/img/ico_radiator_40.png')
underfloor = cv2.imread('images/img/ico_underfloor_40.png')
fancoil = cv2.imread('images/img/ico_fancoil_40.png')
menu = bretina.resize(menu, 3)
radiator = bretina.resize(radiator, 3)
underfloor = bretina.resize(underfloor, 3)
fancoil = bretina.resize(fancoil, 3)
icons = (menu, radiator, underfloor, fancoil)

print('radiator_ico')

for icon in icons:
    for icon2 in icons:
        print(bretina.recognize_image(icon, icon2))
