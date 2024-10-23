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
tank_template_resized = bretina.resize_image(tank_template, 3) 

print('tank')
print(bretina.recognize_image(tank, tank_template_resized))
print(bretina.recognize_image(tank_white, tank_template_resized))

mall = cv2.imread('images/icon/bell/mallfunction.bmp')
user = cv2.imread('images/icon/bell/user.bmp')


error = cv2.imread('images/img/ico_error_40.png')
prof = cv2.imread('images/img/ico_profile_40.png')


# resize image
error = bretina.resize_image(error, 3) 
prof = bretina.resize_image(prof, 3) 
print('bell_shape')
print(bretina.recognize_image(mall, error))
print(bretina.recognize_image(mall, prof))
print(bretina.recognize_image(user, error))
print(bretina.recognize_image(user, prof))


h_c_ico = cv2.imread('images/img/ico_ch_40.png')
cooling = cv2.imread('images/img/ico_cooling_40.png')
heating = cv2.imread('images/img/ico_heating_40.png')
onoff = cv2.imread('images/img/ico_onoff_40.png')
info = cv2.imread('images/img/ico_info_40.png')
set_ico = cv2.imread('images/img/ico_set_40.png')
h_c_ico = bretina.resize_image(h_c_ico, 3) 
cooling = bretina.resize_image(cooling, 3) 
heating = bretina.resize_image(heating, 3) 
onoff = bretina.resize_image(onoff, 3) 
info = bretina.resize_image(info, 3) 
set_ico = bretina.resize_image(set_ico, 3) 


imgs=([f for f in os.listdir("images/icon/round") if f.endswith(".bmp")]  )
for i in range(0, len(imgs)):
    img = cv2.imread('images/icon/round/'+imgs[i])
    print imgs[i]
    print(bretina.recognize_image(img, h_c_ico))
    print(bretina.recognize_image(img, cooling))
    print(bretina.recognize_image(img, heating))
    print(bretina.recognize_image(img, onoff))
    print(bretina.recognize_image(img, info))
    print(bretina.recognize_image(img, set_ico))

menu = cv2.imread('images/img/ico_menu_40.png')
radiator = cv2.imread('images/img/ico_radiator_40.png')
underfloor = cv2.imread('images/img/ico_underfloor_40.png')
fancoil = cv2.imread('images/img/ico_fancoil_40.png')
menu = bretina.resize_image(menu, 3) 
radiator = bretina.resize_image(radiator, 3) 
underfloor = bretina.resize_image(underfloor, 3) 
fancoil = bretina.resize_image(fancoil, 3) 
icons = (menu, radiator, underfloor, fancoil)

print('radiator_ico')
for icon in icons:
    for icon2 in icons:
        print(bretina.recognize_image(icon, icon2))
