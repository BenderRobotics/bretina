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
    LIMIT_COLOR_DISTANCE = 30.0
    LIMIT_IMAGE_MATCH = 0.74
    CHESSBOARD_SIZE = (15, 8.5)
    DISPLAY_SIZE = (480, 272)
    SCALE = 3.0
    BORDER = 4

    #: path where the log images should be stored
    LOG_PATH = './log/'
    TEMPLATE_PATH = './'
    LOG_IMG_FORMAT = "JPG"
    SRC_IMG_FORMAT = "PNG"
    SIMILAR_CHARACTERS = ["1ilI|", "0oOQ"]

    #: set to true to save also source image when assert fails
    SAVE_SOURCE_IMG = False

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

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.TEST_CASE_NAME = ""
        self.img = None                             #: here is stored the currently captured image
        self.imgs = None                            #:

    def _preprocess(self, img):
        """
        Apliing filters on the acquired image.

        :param img: Input image from the camera
        :type  img: image
        :return: pre-processed image
        :rtype: image
        """
        # Bilateral filter
        if self.PRE_BIL_FILTER_APPLY:
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            img_lab[:, :, 0] = cv2.bilateralFilter(img_lab[:, :, 0], self.PRE_BIL_FILTER_DIAMETER, self.PRE_BIL_FILTER_SIGMA_COLOR, self.PRE_BIL_FILTER_SIGMA_SPACE)
            img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

        # Non-local means de-noising
        if self.PRE_DENOISE_APPLY:
            img = cv2.fastNlMeansDenoisingColored(img, None, self.PRE_DENOISE_H_LIGHT, self.PRE_DENOISE_H_COLOR, self.PRE_DENOISE_TEMP_WIN_SIZE, self.PRE_DENOISE_SEARCH_WIN_SIZE)

        return img

    def capture(self):
        """
        Captures image from the camera and does the preprocessing.

        Pre-processed image is stored in the `self.img`.
        """
        img = self.camera.acquire_calibrated_image()
        self.img = self._preprocess(img)

    def capture_images(self, num_images, period):
        """
        Captures image from the camera and does the preprocessing.

        Sequence of pre-processed images is stored in the `self.imgs`.
        """
        raws = self.camera.acquire_calibrated_images(num_images, period)
        self.imgs = [self._preprocess(raw) for raw in raws]

    def save_img(self, img, name, border_box=None, msg=None):
        """
        Writes the actual image to the file with the name based on the current time and the given name.

        :param name: name of the test to use it in the output filename
        :type  name: str
        :param border_box: specify this parameter to draw a red rectangle to this region in the stored image
        :type  border_box: Tuple[left, top, right, bottom]
        """
        now = datetime.now()
        filename = now.strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]

        if (name is not None) and (len(str(name)) > 0):
            filename += "_" + str(name)

        directory = os.path.join(self.LOG_PATH, now.strftime('%Y-%m-%d'))

        if not os.path.isdir(directory):
            os.mkdirs(directory)

        extension = self.LOG_IMG_FORMAT.lower()
        if not extension.startswith('.'):
            extension = '.' + extension

        path = os.path.join(directory, filename + extension)

        if self.SAVE_SOURCE_IMG:
            extension = self.SRC_IMG_FORMAT.lower()
            if not extension.startswith('.'):
                extension = '.' + extension

            cv2.imwrite(os.path.join(directory, filename + '-src' + extension), img)

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

    def tearDown(self):
        """
        Hook method for deconstructing the test fixture after testing it.
        """
        pass

    # ---------------------------------------------------------------------------------
    # - Asserts
    # ---------------------------------------------------------------------------------

    def assertEmpty(self, region, bgcolor=None, metric=None, msg=""):
        """
        Check if the region is empty. Checks standart deviation of the color lightness
        and optionally average color to be bgcolor.

        :param region: boundaries of intrested area
        :type  region: [left, top, right, bottom]
        :param bgcolor: background color, compared with actual background if not None
        :type  bgcolor: str or Tuple(B, G, R)
        :param metrics: function to use to calculate the color distance `d = metrics((B, G, R), (B, G, R))`
        :type  metrics: callable
        :param msg: optional assertion message
        :type  msg: str
        """
        roi = bretina.crop(self.img, region, self.SCALE)
        roi_gray = bretina.img_to_grayscale(roi)
        std = bretina.lightness_std(roi_gray)

        # check if standart deviation of the lightness is low
        if std > self.LIMIT_EMPTY_STD:
            message = "Region '{region}' is not empty (STD {std:.2f} > {limit:.2f}): {msg}"
            message = message.format(region=region, std=std, limit=self.LIMIT_EMPTY_STD, msg=msg)
            self.log.error(message)
            self.save_img(self.img, self.TEST_CASE_NAME, region, msg=message)
            self.fail(msg=message)
        else:
            self.log.debug("Region '{region}' is empty (STD {std:.2f} <= {limit:.2f})".format(region=region,
                                                                                              std=std,
                                                                                              limit=self.LIMIT_EMPTY_STD))

        # check if average color is close to expected background
        if bgcolor is not None:
            if metric is None:
                metric = bretina.rgb_rms_distance
            else:
                assert callable(metric), "`metric` parameter has to be callable function with two parameters"

            avgcolor = bretina.mean_color(roi)
            dist = metric(avgcolor, bgcolor)

            if dist > self.LIMIT_COLOR_DISTANCE:
                message = "Region '{region}' background color is not as expected {background} != {expected} (distance {distance:.2f}): {msg}"
                message = message.format(region=region,
                                         background=bretina.color_str(avgcolor),
                                         expected=bretina.color_str(bgcolor),
                                         distance=dist,
                                         msg=msg)
                self.log.error(message)
                self.save_img(self.img, self.TEST_CASE_NAME, region, msg=message)
                self.fail(msg=message)
            else:
                self.log.debug("Region '{region}' background {background} equals {expected} (distance {distance:.2f})".format(
                    region=region,
                    background=bretina.color_str(avgcolor),
                    expected=bretina.color_str(bgcolor),
                    distance=dist))

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
            message = "Region '{region}' is empty (STD {std}} <= {limit:.2f}): {msg}".format(region=region,
                                                                                             std=std,
                                                                                             limit=self.LIMIT_EMPTY_STD,
                                                                                             msg=msg)
            self.log.error(message)
            self.save_img(self.img, self.TEST_CASE_NAME, region, msg=message)
            self.fail(msg=message)
        else:
            self.log.debug("Region '{region}' is not empty (STD {std}} > {limit:.2f})".format(region=region,
                                                                                              std=std,
                                                                                              limit=self.LIMIT_EMPTY_STD))

    def assertColor(self, region, color, bgcolor=None, metric=None, msg=""):
        """
        Checks if the most dominant color is the given color. Background color can be specified.

        :param region: boundaries of intrested area
        :type  region: [left, top, right, bottom]
        :param color: expected color
        :type  color: str or Tuple(B, G, R)
        :param bgcolor: background color, set to None to determine automatically
        :type  bgcolor: str or Tuple(B, G, R)
        :param metric: function to use to calculate the color distance `d = metrics((B, G, R), (B, G, R))`
        :type  metric: callable
        :param msg: optional assertion message
        :type  msg: str
        """
        if metric is None:
            metric = bretina.rgb_rms_distance
        else:
            assert callable(metric), "`metric` parameter has to be callable function with two parameters"

        roi = bretina.crop(self.img, region, self.SCALE)
        dominant_color = bretina.active_color(roi, bgcolor=bgcolor)
        dist = metric(dominant_color, color)

        # test if color is close to the expected
        if dist > self.LIMIT_COLOR_DISTANCE:
            message = "Color {color} is too far from {expected} (distance {distance:.2f} > {limit:.2f}): {msg}".format(
                            color=bretina.color_str(dominant_color),
                            expected=bretina.color_str(color),
                            distance=dist,
                            limit=self.LIMIT_COLOR_DISTANCE,
                            msg=msg)
            self.log.error(message)
            self.save_img(self.img, self.TEST_CASE_NAME, region, msg=message)
            self.fail(msg=message)
        else:
            self.log.debug("Color {color} equals to {expected} ({distance:.2f} <= {limit:.2f})".format(color=bretina.color_str(dominant_color),
                                                                                                       expected=bretina.color_str(color),
                                                                                                       distance=dist,
                                                                                                       limit=self.LIMIT_COLOR_DISTANCE))

    def assertText(self, region, text,
                   language="eng", msg="", circle=False, bgcolor=None, chars=None, floodfill=False, sliding=False, ratio=None, simchars=None):
        """
        Checks the text in the given region.

        When `ratio` is not specified, the comparision method is ignores differences in `SIMILAR_CHARACTERS`.
        You can override `SIMILAR_CHARACTERS` with `simchars` parameter. If ratio is specified, than it is
        used as comparision method. Set ratio to '1' for exact match.

        :param list region: boundaries of intrested area [left, top, right, bottom]
        :param str text: expected text co compare
        :param str language: language of the string, use 3-letter ISO codes: https://github.com/tesseract-ocr/tesseract/wiki/Data-Files
        :param str msg: optional assertion message
        :param bool circle: optional flag to tell OCR engine that the text is in circle
        :param bgcolor: background color
        :param str chars: optional limit of the used characters in the OCR
        :param bool floodfill: optional argument to apply flood fill to unify background
        :param bool sliding: optional argument
            - `False` to prohibit sliding text animation recognition
            - `True` to check also sliding text animation, can lead to long process time
        :param float ratio: measure of the sequences similarity as a float in the range [0, 1], see https://docs.python.org/3.8/library/difflib.html#difflib.SequenceMatcher.ratio
        :param list simchars: allowed similar chars in text comparision, e.g. ["1l", "0O"]. Differences in these characters are not taken as differences.
        """
        roi = bretina.crop(self.img, region, self.SCALE, border=5)
        multiline = bretina.text_rows(roi, self.SCALE)[0] > 1
        readout = bretina.read_text(roi, language, multiline, circle=circle, bgcolor=bgcolor, chars=chars, floodfill=floodfill)

        if simchars is None:
            simchars = self.SIMILAR_CHARACTERS

        # Local compare function
        def equals(a, b):
            if ratio is not None:
                assert ratio <= 1.0 and ratio >= 0.0, '`ratio` has to be float in range [0, 1], {} given'.format(ratio)
                return bretina.equal_str_ratio(a, b, ratio)
            else:
                return bretina.equal_str(a, b, simchars)

        # For single line text try to use sliding text reader
        if not equals(readout, text) and not multiline and sliding:
            active = True
            sliding_text = bretina.SlidingTextReader()

            # Gather sliding animation frames
            while active:
                img = self.camera.acquire_calibrated_image()
                img = self._preprocess(img)
                img = bretina.crop(img, region, self.SCALE, border=self.BORDER)
                active = sliding_text.unite_animation_text(img, 20, bg_color='black', transparent=True)

            roi = sliding_text.get_image()
            readout = bretina.read_text(roi, language, False, circle=circle, bgcolor=bgcolor, chars=chars, floodfill=floodfill)

            if not equals(readout, text):
                if roi.shape[1] < self.img.shape[1]:
                    top = int(region[3] * self.SCALE)
                    left = int(region[0] * self.SCALE)
                    if left+roi.shape[1] < self.img.shape[1]:
                        self.img[top:top+roi.shape[0], left:left+roi.shape[1]] = roi
                    else:
                        self.img[top:top+roi.shape[0], self.img.shape[1]-roi.shape[1]:self.img.shape[1]] = roi
                else:
                    self.save_img(roi, self.TEST_CASE_NAME)

        if not equals(readout, text):
            message = "Text '{readout}' does not match expected '{expected}': {msg}".format(readout=readout,
                                                                                            expected=text,
                                                                                            msg=msg)
            self.log.error(message)
            self.save_img(self.img, self.TEST_CASE_NAME, region, msg=message)
            self.fail(msg=message)
        else:
            try:
                self.log.debug("Text '{readout}' matched with '{expected}'".format(readout=readout,
                                                                                   expected=text))
            except UnicodeEncodeError as ex:
                pass

    def assertImage(self, region, template_name, msg=""):
        """
        Checks if image is present in the given region.

        :param region: boundaries of intrested area
        :type  region: [left, top, right, bottom]
        :param template_name: file name of the expected image relative to `self.template_path`
        :type  template_name: str
        :param msg: optional assertion message
        :type  msg: str
        """
        roi = bretina.crop(self.img, region, self.SCALE)
        path = os.path.join(self.template_path, template_name)
        template = cv2.imread(path)

        if template is None:
            message = 'Template file {} is missing! Full path: {}'.format(template_name, path)
            self.log.error(message)
            self.fail(message)

        template = bretina.resize(template, self.SCALE)
        match = bretina.recognize_image(roi, template)

        if match < self.LIMIT_IMAGE_MATCH:
            message = "Template '{name}' does not match with given region content, matching level {level:.2f} < {limit:.2f}: {msg}".format(
                            name=template_name,
                            level=match,
                            limit=self.LIMIT_IMAGE_MATCH, msg=msg)
            self.log.error(message)
            self.save_img(self.img, self.TEST_CASE_NAME, region, msg=message)
            self.fail(msg=message)
        elif match >= self.LIMIT_IMAGE_MATCH and match <= (self.LIMIT_IMAGE_MATCH + 0.05):
            message = "Template '{name}' matching level {level:.2f} is close to the limit {limit:.2f}.".format(
                            name=template_name,
                            level=match,
                            limit=self.LIMIT_IMAGE_MATCH)
            self.log.warning(message)
        else:
            self.log.debug("Template '{name}' matched ({level:.2f} >= {limit:.2f})".format(name=template_name,
                                                                                           level=match,
                                                                                           limit=self.LIMIT_IMAGE_MATCH))

    def assertEmptyAnimation(self, region, bgcolor=None, metric=None, msg=""):
        """
        Check if the region is empty. Checks standart deviation of the color lightness
        and optionally average color to be bgcolor.

        :param region: boundaries of intrested area
        :type  region: [left, top, right, bottom]
        :param bgcolor: background color, compared with actual background if not None
        :type  bgcolor: str or Tuple(B, G, R)
        :param metrics: function to use to calculate the color distance `d = metrics((B, G, R), (B, G, R))`
        :type  metrics: callable
        :param msg: optional assertion message
        :type  msg: str
        """
        roi = [bretina.crop(img, region, self.SCALE) for img in self.imgs]
        roi_gray = [bretina.img_to_grayscale(img) for img in roi]
        std = [bretina.lightness_std(img) for img in roi_gray]
        position = np.argmax(std)

        # check if standart deviation of the lightness is low
        if max(std) > self.LIMIT_EMPTY_STD:
            message = "Region '{region}' is not empty (STD {std:.2f} > {limit:.2f}): {msg}".format(region=region,
                                                                                                   std=max(std),
                                                                                                   limit=self.LIMIT_EMPTY_STD,
                                                                                                   msg=msg)
            self.log.error(message)
            self.save_img(self.img, self.TEST_CASE_NAME, region, msg=message)
            self.fail(msg=message)
        else:
            self.log.debug("Region '{region}' is empty (STD {std:.2f} > {limit:.2f})".format(region=region,
                                                                                             std=max(std),
                                                                                             limit=self.LIMIT_EMPTY_STD))

        # check if average color is close to expected background
        if bgcolor is not None:
            if metric is None:
                metric = bretina.rgb_rms_distance
            else:
                assert callable(metric), "`metric` parameter has to be callable function with two parameters"

            avgcolors = [bretina.mean_color(img) for img in roi]
            avgcolor = max(avgcolors) if metric(max(avgcolors), bgcolor) > metric(min(avgcolors), bgcolor) else max(avgcolors)
            dist = max(metric(max(avgcolors), bgcolor), metric(min(avgcolors), bgcolor))

            if dist > self.LIMIT_COLOR_DISTANCE:
                message = "Region {region} background color is not as expected {background} != {expected} (distance {distance:.2f}): {msg}".format(
                                region=region,
                                background=bretina.color_str(avgcolor),
                                expected=bretina.color_str(bgcolor),
                                distance=dist,
                                msg=msg)
                self.log.error(message)
                self.save_img(self.imgs[0], self.TEST_CASE_NAME, region, msg=message)
                self.fail(msg=message)
            else:
                self.log.debug("Background color distance ({distance:.2f} <= {limit:.2f}).".format(distance=dist, limit=self.LIMIT_COLOR_DISTANCE))

    def assertImageAnimation(self, region, template_name, animation_active, size, msg=""):
        """
        Checks if the image animation is present in the given region.

        :param region: boundaries of intrested area
        :type  region: [left, top, right, bottom]
        :param template_name: file name of the expected image relative to `self.template_path`
        :type  template_name: str
        :param msg: optional assertion message
        :type  msg: str
        """
        roi = [bretina.crop(img, region, self.SCALE) for img in self.imgs]
        path = os.path.join(self.template_path, template_name)
        template = cv2.imread(path)

        if template is None:
            message = 'Template file {} is missing! Full path: {}'.format(template_name, path)
            self.log.error(message)
            self.fail(message)

        conformity, animation = bretina.recognize_animation(roi, template, size, self.SCALE)

        template_crop = bretina.crop(template, [0, 0, size[0], size[1]], 1, 0)
        position = np.argmax([bretina.recognize_image(img, template_crop) for img in roi])

        if conformity < self.LIMIT_IMAGE_MATCH:
            message = "Template '{name}' does not match with given region content, matching level {level:.2f} < {limit:.2f}: {msg}".format(
                        name=template_name,
                        level=conformity,
                        limit=self.LIMIT_IMAGE_MATCH,
                        msg=msg)
            self.log.error(message)
            self.save_img(self.imgs[position], self.TEST_CASE_NAME, region, msg=message)
            self.fail(msg=message)
        elif conformity >= self.LIMIT_IMAGE_MATCH and conformity <= (self.LIMIT_IMAGE_MATCH + 0.05):
            message = "Template '{name}' matching level {level:.2f} is close to the limit {limit:.2f}.".format(
                            name=template_name,
                            level=conformity,
                            limit=self.LIMIT_IMAGE_MATCH)
            self.log.warning(message)
        else:
            self.log.debug("Animation template '{name}' matched ({level:.2f} > {limit:.2f})".format(name=template_name,
                                                                                                    level=conformity,
                                                                                                    limit=self.LIMIT_IMAGE_MATCH))

        if animation != animation_active:
            message = "Template '{name}' does not meets the assumption that animation is {theoretic:.2f} but is {real:.2f}: {msg}".format(
                        name=template_name,
                        theoretic=animation_active,
                        real=animation,
                        msg=msg)
            self.log.error(message)
            self.save_img(self.imgs[0], self.TEST_CASE_NAME, region, msg=message)
            self.fail(msg=message)
