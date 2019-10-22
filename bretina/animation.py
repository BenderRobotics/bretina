import numpy as np
import cv2 as cv
import bretina


class SlidingTextReader():
    '''
    Recognize animated image and read running text
    '''

    def __init__(self):
        self._reset()

    def unite_animation_text(self, img):
        """
        Reads horizontally moving text

        :param img: cropped image around moving text prom camera
        :type  img: cv2 image (b,g,r matrix)
        :return: True if animation was detected
        :rtype: bool
        """

        if self.text_img is None:
            self.h = img.shape[0]
            self.w = img.shape[1]
            self.text_img = np.zeros((self.h, 3*self.w, 3), np.uint8)
            self.text_img[:, self.w : 2*self.w] = img
            self.min_pos = self.w
            self.max_pos = 2*self.w
            self.united_img = None
            self.l_loc = self.min_pos
            return True
            
        res = cv.matchTemplate(self.text_img, img, cv.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    
        if max_loc[0] < self.max_pos:
            if max_loc[0] < self.min_pos:
                target_stop = self.min_pos-max_loc[0]
                self.text_img[:, max_loc[0]:self.min_pos] = img[:, 0 : target_stop]
        else:
            target_start_1 = self.max_pos + self.w
            target_stop_1 = max_loc[0] + self.w
            target_start_2 = target_start_1 - max_loc[0]
            self.text_img[:, target_start_1 : target_stop_1] = img[:, target_start_2 : self.w]
        target_stop = self.w + max_loc[0]
        self.text_img[:, max_loc[0] : target_stop] = cv.addWeighted(
            self.text_img[:, max_loc[0] : target_stop], 0.5, img, 0.5, 0)
    
        self.min_pos = min(max_loc[0], self.min_pos)
        self.max_pos = max(target_stop, self.max_pos)
    
        d = max_loc[0] - self.l_loc
        self.l_loc = max_loc[0]
        upper_boundary = self.text_img.shape[1] - self.w

        if self.max_pos > upper_boundary:
            blank_img = np.zeros((self.h, self.max_pos-upper_boundary, 3), np.uint8)
            self.text_img = np.concatenate((self.text_img, blank_img), axis=1)
            
        if self.min_pos < self.w:
            shift = self.w - self.min_pos 
            blank_img = np.zeros((self.h, shift, 3), np.uint8)
            self.text_img = np.concatenate((blank_img, self.text_img), axis=1)
            self.min_pos += shift
            self.max_pos += shift
            self.l_loc += shift
            
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
            self.counter += 1
            if self.counter > 5:
                self.united_img = self.text_img[:, self.min_pos:self.max_pos]
                self._reset()
                return False
                
        if self.direction_change == 4:
            self.united_img = self.text_img[:, self.min_pos:self.max_pos]
            self._reset()
            return False

        return True

    def get_image(self):
        """
        Final image combined from frames.
		
        :return: cropped image around text
        :rtype: cv2 image (b,g,r matrix)
        """
        return self.united_img

    def _reset(self):
        """
        Resets internal states of the class.
        """
        self.text_img = None
        self.direction = 0
        self.direction_change = 0
        self.counter = 0
