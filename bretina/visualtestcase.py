"""Visual Test Case implementation"""

import unittest
import numpy as np
import bretina
import cv2
import os

from datetime import datetime


class VisualTestCase(unittest.TestCase):
    """
    """

    LIMIT_EMPTY_STD = 16.0
    LIMIT_COLOR_DISTANCE = 50.0
    LIMIT_IMAGE_MATCH = 0.8

    COLOR_DISTANCE_HUE = 1
    COLOR_DISTANCE_RGB = 2
    CHESSBOARD_SIZE = (8, 8)
    DISPLAY_SIZE = (480, 272)
    SCALE = 3.0
    BORDER = 4

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.TEST_CASE_NAME = ""
        self.dstmaps = None             # rectification un-distortion maps
        self.transformation = None      # rectification transformation matrix
        self.resolution = None          # final resolution
        self.template_path = "./"       # path to the source image templates

    def _preprocess(self, img_raw):
        """
        Executes rectification, color adjustment and crop to display are on the acquired image.

        :param img_raw: Input image from the camera
        :type  img_raw: cv2 image
        :return: pre-processed image
        :rtype: cv2 image
        """
        assert ((self.dstmaps is not None) and
                (self.transformation is not None) and
                (self.resolution is not None)),
                "Calibration to get rectification parameters needs to be done first! Use `calibrate` method."
        assert ((self.hist_calibration is not None) and
                (self.rgb_calibration is not None)),
                "Calibration to get color adjustment parameters needs to be done first! Use `calibrate` method."

        # Apply calibrations
        img_calib = bretina.rectify(img_raw, self.dstmaps, self.transformation, self.resolution)
        img_calib = bretina.calibrate_hist(img_calib, self.hist_calibration)
        img_calib = bretina.calibrate_rgb(img_calib, self.rgb_calibration)

        # Filters to remove noise
        img_lab = cv2.cvtColor(img_calib, cv2.COLOR_BGR2LAB)
        img_lab[:, :, 0] = cv2.bilateralFilter(img_lab[:, :, 0], 9, 40, 10)
        img_color = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
        img_color = cv2.fastNlMeansDenoisingColored(img_color, None, 3, 3, 5, 11)
        return img_color

    def calibrate(self, chessboard, red, green, blue):
        """
        Does the calibration on the calibration images.

        :param chessboard: image of the chessboard pattern
        :type  chessboard: cv2 image
        :param red: image of the red calibration screen
        :type  red: cv2 image
        :param green: image of the green calibration screen
        :type  green: cv2 image
        :param blue: image of the blue calibration screen
        :type  blue: cv2 image
        """
        self.dstmaps, self.transformation, self.resolution = bretina.get_rectification(chessboard, SCALE, CHESSBOARD_SIZE, DISPLAY_SIZE, border=BORDER)
        chessboard = bretina.rectify(chessboard, self.dstmaps, self.transformation, self.resolution)
        self.hist_calibration, self.rgb_calibration = bretina.color_calibration(chessboard, CHESSBOARD_SIZE, red, green, blue)

    def capture(self):
        """
        Captures image from the camera and does the preprocessing. Pre-processed image is
        stored in the `self.img`.
        """
        raw = self.camera.acquire_image()
        self.img = self._preprocess(raw)

    def save_img(self, name, border_box=None):
        """
        Writes the actual image to the file with the name based on the current time and the given name.

        :param name: name of the test to use it in the output filename
        :type  name: str
        :param border_box: specify this parameter to draw a red rectangle to this region in the stored image
        :type  border_box: Tuple[left, top, right, bottom]
        """
        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + name + ".png"
        path = os.path.join(os.getcwd(), name)

        if border_box is not None:
            img = bretina.draw_border(self.img, border_box, SCALE)
        else:
            img = self.img

        cv2.imwrite(path, img)

    def setUp(self):
        """
        Hook method for setting up the test fixture before exercising it.
        """
        self.TEST_CASE_NAME = self.id()
        self.capture()

    def tearDown(self):
        """
        Hook method for deconstructing the test fixture after testing it.
        """
        pass

    # ---------------------------------------------------------------------------------
    # - Asserts
    # ---------------------------------------------------------------------------------

    def assertEmpty(self, region, bgcolor=None, msg=""):
        """
        Check if the region is empty. Checks standart deviation of the color lightness
        and optionally average color to be bgcolor.


        :param region: boundaries of intrested area
        :type  region: [left, top, right, bottom]
        :param bgcolor: background color, compared with actual background if not None
        :type  bgcolor: str or Tuple(B, G, R)
        :param msg: optional assertion message
        :type  msg: str
        """
        roi = bretina.crop(self.img, region, SCALE)
        roi_gray = bretina.img_to_grayscale(roi_gray)
        std = bretina.lightness_std(roi_gray)

        # check if standart deviation of the lightness is low
        if std > self.LIMIT_EMPTY_STD:
            figure = self.draw_border(self.img, region, SCALE)
            self.save_img(figure, self.TEST_CASE_NAME)
            message = "Region '{region}' is not empty (STD {std:3d} > {limit:3d}): {msg}"
            message = message.format(region=region, std=std, limit=self.LIMIT_EMPTY_STD, msg=msg)
            self.fail(msg=message)

        # check if average color is close to expected background
        if bgcolor is not None:
            avgcolor = bretina.mean_color(roi)

            if metrics == self.COLOR_DISTANCE_HUE:
                dist = bretina.hue_distance(avgcolor, bgcolor)
            else:
                dist = bretina.color_distance(avgcolor, bgcolor)

            if dist > self.LIMIT_COLOR_DISTANCE:
                figure = self.draw_border(self.img, region, SCALE)
                self.save_img(figure, self.TEST_CASE_NAME)
                message = "Region background color '{region}' is not as expected {background} != {expected}: {msg}"
                message = message.format(region=region, background=avgcolor, expected=bgcolor, msg=msg)
                self.fail(msg=message)

    def assertNotEmpty(self, region, msg=""):
        """
        Checks if region is not empty by standart deviation of the lightness.

        :param region: boundaries of intrested area
        :type  region: [left, top, right, bottom]
        :param msg: optional assertion message
        :type  msg: str
        """
        roi = bretina.crop(self.img, region, SCALE)
        roi_gray = bretina.img_to_grayscale(roi)
        std = bretina.lightness_std(roi_gray)

        # check if standart deviation of the lightness is high
        if std <= self.LIMIT_EMPTY_STD:
            figure = self.draw_border(self.img, region, SCALE)
            self.save_img(figure, self.TEST_CASE_NAME)
            message = "Region '{region}' is empty (STD {std:3d} <= {limit:3d}): {msg}"
            message = message.format(region=region, std=std, limit=self.LIMIT_EMPTY_STD, msg=msg)
            self.fail(msg=message)

    def assertColor(self, region, color, bgcolor=None, metrics=COLOR_DISTANCE_RGB, msg=""):
        """
        Checks if the most dominant color is the given color. Background color can be specified.

        :param region: boundaries of intrested area
        :type  region: [left, top, right, bottom]
        :param color: expected color
        :type  color: str or Tuple(B, G, R)
        :param bgcolor: background color, set to None to determine automatically
        :type  bgcolor: str or Tuple(B, G, R)
        :param metrics: metrics of comparision
            * COLOR_DISTANCE_RGB - sum{|RGB(A) - RGB(B)|} / 3
            * COLOR_DISTANCE_HUE - |Hue(A) - Hue(B)|
        :param msg: optional assertion message
        :type  msg: str
        """
        roi = bretina.crop(self.img, region, SCALE)
        dominant_color = bretina.active_color(roi, bgcolor=bgcolor)

        # get distance to expected color
        if metrics == self.COLOR_DISTANCE_HUE:
            dist = bretina.hue_distance(dominant_color, color)
        else:
            dist = bretina.color_distance(dominant_color, color)

        # test if color is close to the expected
        if dist > self.LIMIT_COLOR_DISTANCE:
            figure = self.draw_border(self.img, region, SCALE)
            self.save_img(figure, self.TEST_CASE_NAME)
            message = "Color {color} is too far from {expected} (distance {distance:3d} > {limit:3d}): {msg}"
            message = message.format(color=bretina.color_str(dominant_color),
                                     expected=bretina.color_str(color),
                                     distance=dist,
                                     limit=self.LIMIT_COLOR_DISTANCE,
                                     msg=msg)
            self.fail(msg=message)

    def assertText(self, region, text, language="eng", msg=""):
        """
        Checks the text in the given region.

        :param region: boundaries of intrested area
        :type  region: [left, top, right, bottom]
        :param text: expected text co compare
        :type  text: str
        :param language: language of the string, use 3-letter ISO codes: https://github.com/tesseract-ocr/tesseract/wiki/Data-Files
        :type  language: str
        :param msg: optional assertion message
        :type  msg: str
        """
        roi = bretina.crop(self.img, region, SCALE, border=5)
        multiline = bretina.text_rows(roi, scale)[0] > 1
        readout = bretina.read_text(roi, language, multiline)
        text = text.strip()
        readout = readout.strip()

        if readout != text:
            figure = self.draw_border(self.img, region, SCALE)
            self.save_img(figure, self.TEST_CASE_NAME)
            message = "Text '{readout}' is not the expected '{expected}': {msg}'
            message = message.format(readout=readout, expected=text, msg=msg)
            self.fail(msg=message)

    def assertImage(self, region, template_name, msg=""):
        """
        Checks if image is present in the given region.

        :param region: boundaries of intrested area
        :type  region: [left, top, right, bottom]
        :param template_name: filaname of the expected image relative to `self.template_path`
        :type  template_name: str
        :param msg: optional assertion message
        :type  msg: str
        """
        roi = bretina.crop(self.img, region, SCALE)
        path = os.join(self.template_path, template_name)
        template = cv2.imread(path)
        template = bretina.resize(template, SCALE)
        match = bretina.recognize_image(roi, template)

        if match < LIMIT_IMAGE_MATCH:
            figure = self.draw_border(self.img, region, SCALE)
            self.save_img(figure, self.TEST_CASE_NAME)
            message = "Template '{name}' does not match with given region content, matching level {level:3d} < {limit:3d}: {msg}"
            message = message.format(name=name, level=match, limit=LIMIT_IMAGE_MATCH, msg=msg)
            self.fail(msg=message)
        elif match >= LIMIT_IMAGE_MATCH and match <= (LIMIT_IMAGE_MATCH + 0.5):
            message = "Template '{name}' matching level {level:3d} is close to the limit {limit:3d}."
            message = message.format(name=name, level=match, limit=LIMIT_IMAGE_MATCH)
            self.log.warning(message)
