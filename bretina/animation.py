import numpy as np
import cv2 as cv
import bretina
import os
from datetime import datetime


class ReadAnimation():
    '''
    recognize animated image and read running text
    '''
    

    def __init__(self):
        self.text_img = None

    
    def unite_animation_text(self, img):
        """
        read horizontally moving text
    
        :param item: boundaries of text in screen (in resolution of display)
            or "none" value if input screen is cropped around text
        :type item: dict ({"box": [width left border, height upper border,
            width right border, height lower border]}
        :param t: refresh time period (in seconds)
        :type t: float
        :return: read text, if is animation
        :rtype: string, bool
        """
    
        direction = [0, 0, 0]
        
        if self.text_img is None:
            self.h = img.shape[0]
            self.w = img.shape[1]
            text_img = np.zeros((self.h, 10*self.w, 3), np.uint8)
            self.text_img[0:self.h, 5*self.w:6*self.w] = img
            self.min_pos = 5*self.w
            self.max_pos = 6*self.w
            return True
            
        l_loc = self.min_pos
        res = cv.matchTemplate(fin_img, img, cv.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    
        if max_loc[0] < self.max_pos:
            if max_loc[0] < self.min_pos:
                self.text_img[0:h, max_loc[0]:self.min_pos] = img[0:h,
                                                   0:(self.min_pos-max_loc[0])]
        else:
            fin_img[0:h, self.max_pos + w:max_loc[0] +
                    w] = img[0:h, w - (max_loc[0] - self.max_pos):w]
        fin_img[0:h, max_loc[0]:w+max_loc[0]] = cv.addWeighted(
            fin_img[0:h, max_loc[0]:w + max_loc[0]], 0.5, img, 0.5, 0)
    
        self.min_pos = min(max_loc[0], self.min_pos)
        self.max_pos = max(max_loc[0] + self.w, self.max_pos)
    
        d = max_loc[0]-l_loc
        l_loc = max_loc[0]
    
        if direction == 0:
            direction = d
            direction_change = 0
            counter = 0
    
        if d < 0:
            if direction > 0:
                direction_change += 1
                direction = d
                counter = 0
                
        elif d > 0:
            if direction < 0:
                direction_change += 1
                direction = d
                counter = 0
        else:
            counter = counter+1
            if counter > 5:
                active = False
                
        if direction[1] == 2:
            self.text_img = self.text_img[0:h, self.min_pos:self.max_pos]
            return False
        
        
    
        return True
     
    def read_animation_text(self, cam, scale, border, width_px, height_px)
        acive = True
        while active:
            img = load_img()
            active = self.unite_animation_text(img):
        text = self.read_text(None, fin, False)    
    
    def carpute_iamge(self, item, t=0.1, t_end=1):
		Pass
		
    def recognize_image_animated(self, item):
        """
        recognize animation in image
    
        :param item: boundaries of text in screen (in resolution of display)
            or "none" value if input screen is cropped around text,
            array of artwork images name
        :type item: dict ({"box": [width left border, height upper border,
            width right border, height lower border],
            "images": array of image names}
        :param t: refresh time period (in seconds)
        :type t: float
        :param t_end: max time of animation recognition
        :type t_end: int/float
        :return: recognized image, animation period, duty cycle
        :rtype: string, float, float
        """
        if self.cam is not None:
            img = self.__load_img()
        else:
            self.logger.error('No camera connected', extra=self.log_args)
            raise SystemExit
    
        if item["box"] is not None:
            boundaries = self.__boundaries_from_box(item)
        else:
            endY, endX = img.shape[:2]
            boundaries = [0, endY, 0, endX]
    
        new_item = {"box": None, "images": item["images"]}
        roi = img[boundaries[0]:boundaries[1], boundaries[2]:boundaries[3]]
        start = time.time()
        animation = {0: self.recognize_image(new_item, roi, 0.3, False)}
    
        for x in range(int(t_end/t)):
            img = self.__load_img()
            img = img[boundaries[0]:boundaries[1], boundaries[2]:boundaries[3]]
            animation[time.time()-start] = self.recognize_image(new_item, img, 0.3, False)
    
        read_item = []
        duty_cycles = {}
        duty_cycles_zero = {}
        periods = []
        item = {}
    
        for x, time in enumerate(animation):
            try:
                i = item[animation[time]]
            except:
                item[animation[time]] = len(read_item)
                read_item.append([animation[time], time, 1, x])
                continue
    
            if read_item[i][3] == x-1:
                read_item[i] = [animation[time], read_item[i][1], (read_item[i][2])+1, x]
            else:
                periods.append(time-read_item[i][1])
                read_item[i] = [animation[time], time, (read_item[i][2]+1), x]
    
                try:
                    zero_time = duty_cycles_zero[animation[time]]
                    duty_cycles[animation[time]] = [read_item[i][2]-zero_time,
                                                    duty_cycles[animation[time]][1]+1]
                except:
                    duty_cycles[animation[time]] = [0, 0]
                    duty_cycles_zero[animation[time]] = read_item[i][2]
    
        count_period = 0
        duty_cycle = {}
    
        if len(periods) == 0:
            period = 0
            duty_cycle[read_item[0][0]] = 1
        else:
            for period in periods:
                count_period += period
            period = count_period/len(periods)
    
            for item in duty_cycles:
                if duty_cycles[item][0] == 0:
                    duty_cycle[item] = 1
                    print(duty_cycles[item])
                else:
                    duty_cycle[item] = (
                        duty_cycles[item][1]/duty_cycles[item][0])
    
        return(duty_cycle, period)
