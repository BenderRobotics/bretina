import unittest
import numpy as np
import cv2 as cv
import bretina
import os
import inspect
from datetime import datetime


class VisualTestCase(unittest.TestCase):
    '''
    Region rectangle is represented as tuple: (x, y, w, h)
    '''
    LIMIT_EMPTY_STD = 16.0
    LIMIT_COLOR_DISTANCE = 50.0

    COLOR_DISTANCE_CHROMA = 1
    COLOR_DISTANCE_RGB = 2
    REGION_SCALE = 2.0
    RESOLUTION = (int(REGION_SCALE * 480), int(REGION_SCALE * 272))

    def __init__(self):
        path = 'img/supernoisy.png'

        self.TEST_CASE_NAME = ""

        # colorized image, apply:
        # - image de-noising
        # - cropping
        # - resizing
        self.img_color = cv.imread(path)
        self.img_color = cv.fastNlMeansDenoisingColored(self.img_color, None, 3, 3, 5, 11)
        self.img_color = cv.resize(self.img_color, self.RESOLUTION, interpolation=cv.INTER_NEAREST)

        # grayscale image, apply:
        # - histogram normalization with CLAHE
        # - bilateral filtering to reduce noise
        # - cropping
        # - resizing
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.img_grayscale = cv.imread(path, 0)
        self.img_grayscale = clahe.apply(self.img_grayscale)
        self.img_grayscale = cv.bilateralFilter(self.img_grayscale, 9, 40, 10)
        self.img_grayscale = cv.resize(self.img_grayscale, self.RESOLUTION, interpolation=cv.INTER_NEAREST)

        # Calibration (https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html)
        # TODO

    def _save_img(self, img, region):
        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + self.TEST_CASE_NAME + ".jpg"
        path = os.path.join(os.getcwd(), name)
        cv.imwrite(path, bretina.border(img, region))

    @classmethod
    def _scale(cls, region):
        return tuple(cls.RESOLUTION * dim for dim in region)

    def assertEmpty(self, region, bgcolor=None, metrics=COLOR_DISTANCE_CHROMA, msg=""):
        '''
        Check if the region is empty. Checks standart deviation of the color lightness
        and optionally average color to be bgcolor.
        '''
        region = self._scale(region)
        roi = bretina.crop(self.img_grayscale, region)
        std = bretina.lightness_std(roi)
        self.assertLess(std, self.LIMIT_EMPTY_STD)    # check if standart deviation of the lightness is low

        # check if average color is close to expected background
        if bgcolor is not None:
            roi = bretina.crop(self.img_color, region)
            avgcolor = bretina.mean_color(roi)

            if metrics == self.COLOR_DISTANCE_CHROMA:
                dist = bretina.hue_distance(avgcolor, bgcolor)
            else:
                dist = bretina.color_distance(avgcolor, bgcolor)

            self.assertLess(dist, self.LIMIT_COLOR_DISTANCE)

    def assertNotEmpty(self, region, msg=""):
        '''
        Checks if region is not empty by standart deviation of the lightness.
        '''
        region = self._scale(region)
        roi = bretina.crop(self.img_grayscale, region)
        std = bretina.lightness_std(roi)
        self.assertGreater(std, self.LIMIT_EMPTY_STD)   # check if standart deviation of the lightness is high

    def assertColor(self, region, color, bgcolor=None, metrics=COLOR_DISTANCE_RGB, msg=""):
        '''
        Checks if the most dominant color is the given color. Background color can be specified.
        '''
        region = self._scale(region)
        roi = bretina.crop(self.img_color, region)
        dominant_color = bretina.active_color(roi, bgcolor=bgcolor)

        # get distance of nominal color to expected color
        if metrics == self.COLOR_DISTANCE_CHROMA:
            dist = bretina.hue_distance(dominant_color, color)
        else:
            dist = bretina.color_distance(dominant_color, color)

        # test if color is close to expected
        if dist > self.LIMIT_COLOR_DISTANCE:
            self._save_img(self.img_color, region)
            self.fail("Color {} is too far from {} (expected) : {}".format(
                bretina.color_str(dominant_color),
                bretina.color_str(color),
                msg))

    def assertText(self, region, tol, msg=""):
        '''
        Checks if text is present in the given region.
        '''
        # https://www.pyimagesearch.com/2017/07/10/using-tesseract-ocr-python/
        region = self._scale(region)
        pass

    def assertImage(self, region, name, msg=""):
        '''
        Checks if image is present in the given region.
        '''
        # SSIM looks good https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
        # use template matching to align https://docs.opencv.org/master/d4/dc6/tutorial_py_template_matching.html
        region = self._scale(region)
        pass

    def assertSprite(self, region, name, frameDelay, frameNum, msg=""):
        '''
        Checks if sprite image animation is present in the given region.
        '''
        region = self._scale(region)
        pass

    def assertShape(self, region, name, msg=""):
        '''
        Checks if image is present in the given region without controlling the color.
        '''
        region = self._scale(region)
        pass

    def assertFormat(self, region, regexp, msg=""):
        '''
        Cheks if the text in the given region follows the specified format set by the regexp.
        '''
        region = self._scale(region)
        pass
