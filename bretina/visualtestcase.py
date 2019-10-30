"""Visual Test Case implementation"""

import unittest
import numpy as np
import bretina
import cv2
import os

from datetime import datetime


class VisualTestCase(unittest.TestCase):
    """
    A class whose instances are single test cases.

    VisualTestCase is a subclass of the standart unittest TestCase,
    therefore there are all features from the unittest module and some
    additional asserts for the image processing.

    By default, the test code itself should be placed in a method named
    'runTest'.

    If the fixture may be used for many test cases, create as
    many test methods as are needed. When instantiating such a TestCase
    subclass, specify in the constructor arguments the name of the test method
    that the instance is to execute.

    Test authors should subclass TestCase for their own tests. Construction
    and deconstruction of the test's environment ('fixture') can be
    implemented by overriding the 'setUp' and 'tearDown' methods respectively.

    If it is necessary to override the __init__ method, the base class
    __init__ method must always be called. It is important that subclasses
    should not change the signature of their __init__ method, since instances
    of the classes are instantiated automatically by parts of the framework
    in order to be run.

    When subclassing TestCase, you can set these attributes:
    * failureException: determines which exception will be raised when
        the instance's assertion methods fail; test methods raising this
        exception will be deemed to have 'failed' rather than 'errored'.
    * longMessage: determines whether long messages (including repr of
        objects used in assert methods) will be printed on failure in *addition*
        to any explicit message passed.
    * maxDiff: sets the maximum length of a diff in failure messages
        by assert methods using difflib. It is looked up as an instance
        attribute so can be configured by individual tests if required.
    """

    LIMIT_EMPTY_STD = 16.0
    LIMIT_COLOR_DISTANCE = 50.0
    LIMIT_IMAGE_MATCH = 0.8

    COLOR_DISTANCE_HUE = 1
    COLOR_DISTANCE_RGB = 2
    CHESSBOARD_SIZE = (15, 8.5)
    DISPLAY_SIZE = (480, 272)
    SCALE = 3.0
    BORDER = 4

    #: Sets if the bilateral filtering is applied during pre-processing
    PRE_BIL_FILTER_APPLY = True
    #: Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace.
    PRE_BIL_FILTER_DIAMETER = 5
    #: Filter sigma in the color space. A larger value of the parameter means that farther colors within
    # the pixel neighborhood will be mixed together, resulting in larger areas of semi-equal color.
    PRE_BIL_FILTER_SIGMA_COLOR = 20.0
    #: Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each
    # other as long as their colors are close enough. When PRE_FILTER_DIAMETER>0, it specifies the neighborhood size regardless
    # of sigmaSpace. Otherwise, d is proportional to sigmaSpace.
    PRE_BIL_FILTER_SIGMA_SPACE = 10.0

    #: Sets if the Non-local Means De-noising algorithm is applied during pre-processing
    PRE_DENOISE_APPLY = True
    #: Parameter regulating filter strength for luminance component. Bigger h value perfectly removes noise but also removes
    #  image details, smaller h value preserves details but also preserves some noise
    PRE_DENOISE_H_LIGHT = 3
    #: The same as h but for color components. For most images value equals 10 will be enough to remove colored noise and do not distort colors
    PRE_DENOISE_H_COLOR = 3
    #: Size in pixels of the template patch that is used to compute weights. Should be odd.
    PRE_DENOISE_TEMP_WIN_SIZE = 7
    #: Size in pixels of the window that is used to compute weighted average for given pixel. Should be odd.
    #  Affect performance linearly: greater searchWindowsSize - greater de-noising time.
    PRE_DENOISE_SEARCH_WIN_SIZE = 11

    def __init__(self, methodName='runTest', templatePath='./'):
        super().__init__(methodName)
        self.TEST_CASE_NAME = ""
        self.dstmaps = None                 # rectification un-distortion maps
        self.transformation = None          # rectification transformation matrix
        self.resolution = None              # final resolution
        self.template_path = templatePath   # path to the source image templates

    def _preprocess(self, img_raw):
        """
        Executes rectification, color adjustment and crop to display are on the acquired image.

        :param img_raw: Input image from the camera
        :type  img_raw: image
        :return: pre-processed image
        :rtype: image
        """
        assert self.dstmaps is not None, "Un-distortion maps are required, use `calibrate` method."
        assert self.transformation is not None, "Transformation matrix are required, use `calibrate` method."
        assert self.resolution is not None, "Final resolution is required, use `calibrate` method."
        assert self.hist_calibration is not None, "Histogram calibration parameters are required, use `calibrate` method."
        assert self.rgb_calibration is not None, "RGB calibration parameters are required, use `calibrate` method."

        # Apply calibrations
        img = bretina.rectify(img_raw, self.dstmaps, self.transformation, self.resolution)
        img = bretina.calibrate_hist(img, self.hist_calibration)
        img = bretina.calibrate_rgb(img, self.rgb_calibration)

        # Bilateral filter
        if self.PRE_BIL_FILTER_APPLY:
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            img_lab[:, :, 0] = cv2.bilateralFilter(img_lab[:, :, 0], self.PRE_BIL_FILTER_DIAMETER, self.PRE_BIL_FILTER_SIGMA_COLOR, self.PRE_BIL_FILTER_SIGMA_SPACE)
            img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

        # Non-local means de-noising
        if self.PRE_DENOISE_APPLY:
            img = cv2.fastNlMeansDenoisingColored(img, None, self.PRE_DENOISE_H_LIGHT, self.PRE_DENOISE_H_COLOR, self.PRE_DENOISE_TEMP_WIN_SIZE, self.PRE_DENOISE_SEARCH_WIN_SIZE)

        return img

    def calibrate(self, chessboard, red, green, blue):
        """
        Does the calibration on the calibration images.

        :param chessboard: image of the chessboard pattern
        :type  chessboard: image
        :param red: image of the red calibration screen
        :type  red: image
        :param green: image of the green calibration screen
        :type  green: image
        :param blue: image of the blue calibration screen
        :type  blue: image
        """
        self.dstmaps, self.transformation, self.resolution = bretina.get_rectification(chessboard, self.SCALE, self.CHESSBOARD_SIZE, self.DISPLAY_SIZE, border=self.BORDER)
        chessboard = bretina.rectify(chessboard, self.dstmaps, self.transformation, self.resolution)
        self.hist_calibration, self.rgb_calibration = bretina.color_calibration(chessboard, self.CHESSBOARD_SIZE, red, green, blue)

    def capture(self):
        """
        Captures image from the camera and does the preprocessing. Pre-processed image is
        stored in the `self.img`.
        """
        raw = self.camera.acquire_image()
        self.img = self._preprocess(raw)

    def save_img(self, img, name, border_box=None, msg=None):
        """
        Writes the actual image to the file with the name based on the current time and the given name.

        :param name: name of the test to use it in the output filename
        :type  name: str
        :param border_box: specify this parameter to draw a red rectangle to this region in the stored image
        :type  border_box: Tuple[left, top, right, bottom]
        """
        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if name is not None and name:
            filename += "_" + str(name)

        path = os.path.join(os.getcwd(), filename + ".png")

        if border_box is not None:
            img = bretina.draw_border(img, border_box, self.SCALE)
        else:
            border_box = [0, img.shape[0] / self.SCALE, img.shape[1] / self.SCALE, img.shape[0] / self.SCALE]

        if msg is not None:
            font_name = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            margin = 6
            img_width = img.shape[1]
            img_height = img.shape[0]

            size = cv2.getTextSize(msg, font_name, font_scale, thickness)
            text_width = size[0][0]
            text_height = size[0][1]
            line_height = text_height + size[1]

            border_box = [max(border_box[0], 0),
                          max(border_box[1], 0),
                          min(border_box[2], img_width-1),
                          min(border_box[3], img_height-1)]

            left = border_box[0] * self.SCALE
            bottom = border_box[1] * self.SCALE - margin

            # overflow of image width - shift text to right
            if (left + text_width) > img_width:
                left = max(0, border_box[2] * self.SCALE - text_width)

            # overflow of image top - move text to bottom of the region
            if (bottom - line_height) < 0:
                bottom = min(img_height-1, border_box[3] * self.SCALE + margin + text_height)

            text_org = (int(left), int(bottom))
            back_pt1 = (int(left), int(bottom + size[1]))
            back_pt2 = (int(left + text_width), int(bottom - text_height))

            cv2.rectangle(img, back_pt1, back_pt2, bretina.COLOR_BLACK, -1)  # -1 is for filled
            cv2.putText(img, msg, text_org, font_name, font_scale, bretina.COLOR_RED, thickness)

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
        roi = bretina.crop(self.img, region, self.SCALE)
        roi_gray = bretina.img_to_grayscale(roi)
        std = bretina.lightness_std(roi_gray)

        # check if standart deviation of the lightness is low
        if std > self.LIMIT_EMPTY_STD:
            figure = bretina.draw_border(self.img, region, self.SCALE)
            message = "Region '{region}' is not empty (STD {std:.2f} > {limit:.2f}): {msg}"
            message = message.format(region=region, std=std, limit=self.LIMIT_EMPTY_STD, msg=msg)
            self.save_img(figure, self.TEST_CASE_NAME, region, msg=message)
            self.fail(msg=message)

        # check if average color is close to expected background
        if bgcolor is not None:
            avgcolor = bretina.mean_color(roi)

            if metrics == self.COLOR_DISTANCE_HUE:
                dist = bretina.hue_distance(avgcolor, bgcolor)
            else:
                dist = bretina.color_distance(avgcolor, bgcolor)

            if dist > self.LIMIT_COLOR_DISTANCE:
                figure = bretina.draw_border(self.img, region, self.SCALE)
                message = "Region background color '{region}' is not as expected {background} != {expected}: {msg}"
                message = message.format(region=region, background=avgcolor, expected=bgcolor, msg=msg)
                self.save_img(figure, self.TEST_CASE_NAME, region, msg=message)
                self.fail(msg=message)

    def assertNotEmpty(self, region, msg=""):
        """
        Checks if region is not empty by standart deviation of the lightness.

        :param region: boundaries of intrested area
        :type  region: [left, top, right, bottom]
        :param msg: optional assertion message
        :type  msg: str
        """
        roi = bretina.crop(self.img, region, self.SCALE)
        roi_gray = bretina.img_to_grayscale(roi)
        std = bretina.lightness_std(roi_gray)

        # check if standart deviation of the lightness is high
        if std <= self.LIMIT_EMPTY_STD:
            figure = bretina.draw_border(self.img, region, self.SCALE)
            message = "Region '{region}' is empty (STD {std}} <= {limit:.2f}): {msg}"
            message = message.format(region=region, std=std, limit=self.LIMIT_EMPTY_STD, msg=msg)
            self.save_img(figure, self.TEST_CASE_NAME, region, msg=message)
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
        roi = bretina.crop(self.img, region, self.SCALE)
        dominant_color = bretina.active_color(roi, bgcolor=bgcolor)

        # get distance to expected color
        if metrics == self.COLOR_DISTANCE_HUE:
            dist = bretina.hue_distance(dominant_color, color)
        else:
            dist = bretina.color_distance(dominant_color, color)

        # test if color is close to the expected
        if dist > self.LIMIT_COLOR_DISTANCE:
            figure = bretina.draw_border(self.img, region, self.SCALE)
            message = "Color {color} is too far from {expected} (distance {distance:.2f} > {limit:.2f}): {msg}"
            message = message.format(color=bretina.color_str(dominant_color),
                                     expected=bretina.color_str(color),
                                     distance=dist,
                                     limit=self.LIMIT_COLOR_DISTANCE,
                                     msg=msg)
            self.save_img(figure, self.TEST_CASE_NAME, region, msg=message)
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
        roi = bretina.crop(self.img, region, self.SCALE, border=5)
        multiline = bretina.text_rows(roi, scale)[0] > 1
        readout = bretina.read_text(roi, language, multiline)
        text = text.strip()
        readout = readout.strip()

        if readout != text:
            figure = bretina.draw_border(self.img, region, self.SCALE)
            message = "Text '{readout}' does not match expected '{expected}': {msg}"
            message = message.format(readout=readout, expected=text, msg=msg)
            self.save_img(figure, self.TEST_CASE_NAME, region, msg=message)
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
        roi = bretina.crop(self.img, region, self.SCALE)
        path = os.join(self.template_path, template_name)
        template = cv2.imread(path)
        template = bretina.resize(template, self.SCALE)
        match = bretina.recognize_image(roi, template)

        if match < self.LIMIT_IMAGE_MATCH:
            figure = bretina.draw_border(self.img, region, self.SCALE)
            message = "Template '{name}' does not match with given region content, matching level {level:.2f} < {limit:.2f}: {msg}"
            message = message.format(name=name, level=match, limit=self.LIMIT_IMAGE_MATCH, msg=msg)
            self.save_img(figure, self.TEST_CASE_NAME, region, msg=message)
            self.fail(msg=message)
        elif match >= self.LIMIT_IMAGE_MATCH and match <= (self.LIMIT_IMAGE_MATCH + 0.5):
            message = "Template '{name}' matching level {level:.2f} is close to the limit {limit:.2f}."
            message = message.format(name=name, level=match, limit=LIMIT_IMAGE_MATCH)
            self.log.warning(message)
