import numpy as np
import cv2 as cv
import bretina


class ReadAnimation():
    '''
    recognize animated image and read running text
    '''
    

    def __init__(self):
        self.text_img = None
        self.direction = 0
        self.direction_change = 0
        self.counter = 0
        
    def unite_animation_text(self, img):
        """
        read horizontally moving text
    
        :param img: cropped image around moving text prom camera
        :type img: cv2 image (b,g,r matrix)
        :return: if animation was reeded
        :rtype: bool
        """
          
        if self.text_img is None:
            self.h = img.shape[0]
            self.w = img.shape[1]
            self.text_img = np.zeros((self.h, 10*self.w, 3), np.uint8)
            self.text_img[0:self.h, 5*self.w:6*self.w] = img
            self.min_pos = 5*self.w
            self.max_pos = 6*self.w
            return True
            
        l_loc = self.min_pos
        res = cv.matchTemplate(self.text_img, img, cv.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    
        if max_loc[0] < self.max_pos:
            if max_loc[0] < self.min_pos:
                self.text_img[0:self.h, max_loc[0]:self.min_pos] = img[0:self.h,
                                                   0:(self.min_pos-max_loc[0])]
        else:
            self.text_img[0:self.h, self.max_pos + self.w:max_loc[0] +
                    self.w] = img[0:self.h, self.w - (max_loc[0] - self.max_pos):self.w]
        self.text_img[0:self.h, max_loc[0]:self.w+max_loc[0]] = cv.addWeighted(
           self.text_img[0:self.h, max_loc[0]:self.w + max_loc[0]], 0.5, img, 0.5, 0)
    
        self.min_pos = min(max_loc[0], self.min_pos)
        self.max_pos = max(max_loc[0] + self.w, self.max_pos)
    
        d = max_loc[0]-l_loc
        l_loc = max_loc[0]
        
        if self.direction == 0:
            self.direction = d
            self.direction_change = 0
            self.counter = 0
    
        if d < 0:
            if self.direction > 0:
                self.direction_change += 1
                self.direction = d
                self.counter = 0
                
        elif d > 0:
            if self.direction < 0:
                self.direction_change += 1
                self.direction = d
                self.counter = 0
        else:
            self.counter = self.counter+1
            if self.counter > 5:
                self.text_img = self.text_img[0:self.h, self.min_pos:self.max_pos]
                return False
                
        if self.direction_change == 2:
            self.text_img = self.text_img[0:self.h, self.min_pos:self.max_pos]
            return False
        return True
    
    def image(self):
        """
		return final image
		
        :return: cropped image around text
        :rtype: cv2 image (b,g,r matrix)
        """
        return self.text_img
