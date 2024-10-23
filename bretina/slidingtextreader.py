import numpy as np
import cv2 as cv
import bretina
import time


class SlidingTextReader():
    '''
    Recognize animated image and read running text
    '''

    def __init__(self):
        self._reset()

    def unite_animation_text(self, img, absolute_counter=100, bgcolor=None, transparent=False, zone=40):
        """
        Reads horizontally moving text

        :param img: cropped image around moving text prom camera
        :type  img: cv2 image (b,g,r matrix)
        :param absolute_counter: maximum iteration fot try read sliding text
        :type  absolute_counter: False or int
        :param transparent: set if mask background as transparent
        :type  transparent: bool
        :param bgcolor: color of the background
        :param zone: distance between set background color and color which been transparent
        :type  zone: int
        :return: True if animation was detected
        :rtype: bool
        """
        # mask background if activated
        if transparent:
            if (bgcolor is not None) and (img is not None):
                b, g, r = bretina.color(bgcolor)
                lower = np.maximum((b-zone, g-zone, r-zone), (0, 0, 0))
                upper = np.minimum((b+zone, g+zone, r+zone), (255, 255, 255))
                mask = cv.inRange(img, lower, upper)

        # initial setup parameters
        if self.text_img is None:
            self.color = True if (len(img.shape) == 3 and img.shape[2] == 3) else False
            self.h = img.shape[0]
            self.w = img.shape[1]
            self.text_img = self._blank_image(self.h, 3*self.w)
            if transparent:
                self.text_img[:, self.w:2*self.w] = cv.bitwise_and(img, img, mask=255-mask)
            else:
                self.text_img[:, self.w:2*self.w] = img
            self.min_pos = self.w
            self.max_pos = 2*self.w
            self.united_img = None
            self.l_loc = self.min_pos
            return True
        res = cv.matchTemplate(self.text_img, img, cv.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        # compare actual and last pozition
        d = max_loc[0] - self.l_loc

        # reset if image is the same and direction not changet or if image is different
        if (max_val < 0.9 and abs(d) < 15) or max_val < 0.75:
            if self.absolute_counter < absolute_counter/3:
                self.text_img = self._blank_image(self.h, 3 * self.w)
                if transparent:
                    self.text_img[:, self.w:2*self.w] = cv.bitwise_and(img, img, mask=255-mask)
                else:
                    self.text_img[:, self.w:2*self.w] = img
                self.min_pos = self.w
                self.max_pos = 2*self.w
                self.united_img = None
                self.l_loc = self.min_pos
                self.direction = 0
                self.direction_change = 0
            return True

        # compound images
        if max_loc[0] < self.max_pos:
            if max_loc[0] < self.min_pos:
                target_stop = self.min_pos-max_loc[0]
                self.text_img[:, max_loc[0]:self.min_pos] = img[:, 0:target_stop]
        else:
            target_start_1 = self.max_pos + self.w
            target_stop_1 = max_loc[0] + self.w
            target_start_2 = target_start_1 - max_loc[0]
            self.text_img[:, target_start_1:target_stop_1] = img[:, target_start_2:self.w]
        target_stop = self.w + max_loc[0]
        if transparent:
            img = cv.bitwise_and(img, img, mask=255-mask)
            mix_img = cv.bitwise_and(self.text_img[:, max_loc[0]:target_stop], self.text_img[:, max_loc[0]:target_stop], mask=mask)
            mix_img = cv.add(img, mix_img)
            self.text_img[:, max_loc[0]:target_stop] = cv.addWeighted(
                self.text_img[:, max_loc[0]:target_stop], 0.5, mix_img, 0.5, 0)
        else:
            self.text_img[:, max_loc[0]:target_stop] = cv.addWeighted(
                self.text_img[:, max_loc[0]:target_stop], 0.5, img, 0.5, 0)

        self.min_pos = min(max_loc[0], self.min_pos)
        self.max_pos = max(target_stop, self.max_pos)

        self.l_loc = max_loc[0]
        upper_boundary = self.text_img.shape[1] - self.w

        # compare position with boundaries of compound text, extended it if is larger than image
        if self.max_pos > upper_boundary:
            blank_img = self._blank_image(self.h, self.max_pos-upper_boundary)
            self.text_img = np.concatenate((self.text_img, blank_img), axis=1)
        if self.min_pos < self.w:
            shift = self.w - self.min_pos
            blank_img = self._blank_image(self.h, shift)
            self.text_img = np.concatenate((blank_img, self.text_img), axis=1)
            self.min_pos += shift
            self.max_pos += shift
            self.l_loc += shift

        # initial set of direction, if
        if self.direction == 0:
            self.direction = d if d != 0 else 1

        # if position and direction is changed, add direction counter and reset counter
        if d != 0:
            if (self.direction > 0) == (d < 0):
                self.direction_change += 1
                self.direction = d
                self.counter = 0

        # id position not changed for 10 loops then stop function
        else:
            self.counter += 1
            if self.counter > 10:
                self.united_img = self.text_img[:, self.min_pos:self.max_pos]
                self._reset()
                return False

        # if animation go throught three cycles (6 direction change) stop function
        if self.direction_change == 6:
            self.united_img = self.text_img[:, self.min_pos:self.max_pos]
            self._reset()
            return False

        if absolute_counter:
            self.absolute_counter += 1
            if self.absolute_counter > absolute_counter:
                self.united_img = self.text_img[:, self.min_pos:self.max_pos]
                self._reset()
                return False

        return True

    def get_image(self):
        """
        Final image combined from frames.

        :return:
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
        self.absolute_counter = 0
        self.t0 = time.time()

    def _blank_image(self, h, w):
        """
        Creates blank image with the given width and height

        :param int h: height of blank image
        :param int w: width of blank image
        :return: blank image
        :rtype: cv2 image (one or 3 color channels)
        """
        if self.color:
            blank_image = np.zeros((h, w, 3), np.uint8)
        else:
            blank_image = np.zeros((h, w), np.uint8)
        return blank_image
