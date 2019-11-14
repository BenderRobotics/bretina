"""Visual Test Case implementation"""

import unittest
import numpy as np
import textwrap
import difflib
import bretina
import cv2 as cv
import os

from PIL import Image, ImageFont, ImageDraw
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

    #: Default threshold value for the assertEmpty and assertNotEmpty.
    LIMIT_EMPTY_STD = 16.0
    #: Default threshold value for the color asserts.
    LIMIT_COLOR_DISTANCE = 30.0
    #: Default threshold value for the image asserts.
    LIMIT_IMAGE_MATCH = 0.74
    #: Max len of string for which is the diff displayed
    MAX_STRING_DIFF_LEN = 50

    CHESSBOARD_SIZE = (15, 8.5)
    DISPLAY_SIZE = (480, 272)
    #: Scaling
    SCALE = 3.0
    #: Border
    BORDER = 4

    #: path where the log images should be stored
    LOG_PATH = './log/'
    TEMPLATE_PATH = './'
    LOG_IMG_FORMAT = "JPG"
    SRC_IMG_FORMAT = "PNG"
    CONFUSABLE_CHARACTERS = ["1ilI|", "0oOQ", ".,;", ";j", "G6"]

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
        Apling filters on the acquired image.

        :param img: Input image from the camera
        :type  img: image
        :return: pre-processed image
        :rtype: image
        """
        # Bilateral filter
        if self.PRE_BIL_FILTER_APPLY:
            img_lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
            img_lab[:, :, 0] = cv.bilateralFilter(img_lab[:, :, 0], self.PRE_BIL_FILTER_DIAMETER, self.PRE_BIL_FILTER_SIGMA_COLOR, self.PRE_BIL_FILTER_SIGMA_SPACE)
            img = cv.cvtColor(img_lab, cv.COLOR_LAB2BGR)

        # Non-local means de-noising
        if self.PRE_DENOISE_APPLY:
            img = cv.fastNlMeansDenoisingColored(img, None, self.PRE_DENOISE_H_LIGHT, self.PRE_DENOISE_H_COLOR, self.PRE_DENOISE_TEMP_WIN_SIZE, self.PRE_DENOISE_SEARCH_WIN_SIZE)

        return img

    def _diff_string(self, a, b):
        """
        Get string diff deltas.

        :param str a:
        :param str b:
        :return: 3 rows of human readable deltas in string
        :rtype list:
        """
        d = difflib.Differ()
        diff = d.compare(a, b)

        l1 = ""
        l2 = ""
        l3 = ""

        for d in diff:
            if d.startswith("-"):
                l1 += d[-1]
                l2 += " "
                l3 += "^"
            elif d.startswith("+"):
                l1 += " "
                l2 += d[-1]
                l3 += "^"
            elif d.startswith(" "):
                l1 += d[-1]
                l2 += d[-1]
                l3 += " "

        return [l1, l2, l3]

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
            os.makedirs(directory)

        extension = self.LOG_IMG_FORMAT.lower()
        if not extension.startswith('.'):
            extension = '.' + extension

        path = os.path.join(directory, filename + extension)

        if self.SAVE_SOURCE_IMG:
            extension = self.SRC_IMG_FORMAT.lower()
            if not extension.startswith('.'):
                extension = '.' + extension

            cv.imwrite(os.path.join(directory, filename + '-src' + extension), img)

        if border_box is not None:
            img = bretina.draw_border(img, border_box, self.SCALE)
        else:
            border_box = [0, img.shape[0] / self.SCALE, img.shape[1] / self.SCALE, img.shape[0] / self.SCALE]

        if msg is not None:
            margin = 6
            font_size = 20
            spacing = int(font_size * 0.4)
            img_width = img.shape[1]
            img_height = img.shape[0]

            # Convert the image to RGB (OpenCV uses BGR)
            rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            draw = ImageDraw.Draw(pil_img)
            font = ImageFont.truetype("consola.ttf", font_size)

            if font is None:
                font = ImageFont.truetype("arial.ttf", font_size)

            max_chars = img_width / (font_size / 6)  # some estimation of max chars based on font size and image size
            lines = msg.splitlines()
            cnt = 0

            while True and cnt < 10:
                cnt += 1
                for line in lines:
                    line_width, _ = font.getsize(line)

                    if line_width > img_width:
                        max_chars = min(max_chars, len(line) * 0.85)

                wrapped_lines = []

                for line in msg.splitlines():
                    if len(line) > max_chars:
                        wrapped_lines += textwrap.wrap(line, width=max_chars)
                    else:
                        wrapped_lines.append(line)

                if len(lines) < len(wrapped_lines):
                    lines = wrapped_lines
                else:
                    break

            msg = '\n'.join(lines)
            text_width, text_height = draw.multiline_textsize(msg, font, spacing)
            border_box = [max(border_box[0], 0),
                          max(border_box[1], 0),
                          min(border_box[2], img_width-1),
                          min(border_box[3], img_height-1)]

            left = border_box[0] * self.SCALE
            bottom = border_box[1] * self.SCALE - margin - 1
            top = bottom - text_height

            # overflow of image width - shift text to right
            if (left + text_width) > img_width:
                left = max(0, border_box[2] * self.SCALE - text_width)

            # overflow of image top - move text to bottom of the region
            if top < 0:
                top = min(img_height-1, border_box[3] * self.SCALE + margin + 1)

            text_pt = (int(left), int(top))
            back_left = int(max(0, left - margin))
            back_top = int(max(0, top - margin))
            back_right = int(min(img_width - 1, left + text_width + margin))
            back_bottom = int(min(img_height - 1, top + text_height + margin))

            draw.rectangle([back_left, back_top, back_right, back_bottom], fill="#000000")
            draw.multiline_text(text_pt, msg, fill="#FF0000", font=font, spacing=spacing)

            # Get back the image to OpenCV
            img = cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR)

        cv.imwrite(path, img)

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

    def assertEmpty(self, region, threshold=None, bgcolor=None, bgcolor_threshold=None, metric=None, msg=""):
        """
        Check if the region is empty. Checks standart deviation of the color lightness
        and optionally average color to be bgcolor.

        :param region: boundaries of intrested area
        :type  region: [left, top, right, bottom]
        :param float threshold: threshold of the test, `LIMIT_EMPTY_STD` by default
        :param bgcolor: background color, compared with actual background if not None
        :type  bgcolor: str or Tuple(B, G, R)
        :param float bgcolor_threshold: threshold of the background color comparision, `LIMIT_COLOR_DISTANCE` by default
        :param metrics: function to use to calculate the color distance `d = metrics((B, G, R), (B, G, R))`
        :type  metrics: callable
        :param msg: optional assertion message
        :type  msg: str
        """
        if threshold is None:
            threshold = self.LIMIT_EMPTY_STD

        assert threshold >= 0.0, '`threshold` has to be a positive float'

        roi = bretina.crop(self.img, region, self.SCALE)
        roi_gray = bretina.img_to_grayscale(roi)
        std = bretina.lightness_std(roi_gray)

        # check if standart deviation of the lightness is low
        if std > threshold:
            message = "Region '{region}' is not empty (STD {std:.2f} > {limit:.2f}): {msg}"
            message = message.format(region=region, std=std, limit=threshold, msg=msg)
            self.log.error(message)
            self.save_img(self.img, self.TEST_CASE_NAME, region, msg=message)
            self.fail(msg=message)
        else:
            self.log.debug("Region '{region}' is empty (STD {std:.2f} <= {limit:.2f})".format(region=region,
                                                                                              std=std,
                                                                                              limit=threshold))

        # check if average color is close to expected background
        if bgcolor is not None:
            if metric is None:
                metric = bretina.rgb_rms_distance
            else:
                assert callable(metric), "`metric` parameter has to be callable function with two parameters"

            avgcolor = bretina.mean_color(roi)
            dist = metric(avgcolor, bgcolor)

            if bgcolor_threshold is None:
                bgcolor_threshold = self.LIMIT_COLOR_DISTANCE

            if dist > bgcolor_threshold:
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

    def assertNotEmpty(self, region, threshold=None, msg=""):
        """
        Checks if region is not empty by standart deviation of the lightness.

        :param region: boundaries of intrested area
        :type  region: [left, top, right, bottom]
        :param float threshold: threshold of the test, `LIMIT_EMPTY_STD` by default
        :param str msg: optional assertion message
        """
        if threshold is None:
            threshold = self.LIMIT_EMPTY_STD

        assert threshold >= 0.0, '`threshold` has to be a positive float'

        roi = bretina.crop(self.img, region, self.SCALE)
        roi_gray = bretina.img_to_grayscale(roi)
        std = bretina.lightness_std(roi_gray)

        # check if standart deviation of the lightness is high
        if std <= threshold:
            message = "Region '{region}' is empty (STD {std}} <= {limit:.2f}): {msg}".format(region=region,
                                                                                             std=std,
                                                                                             limit=threshold,
                                                                                             msg=msg)
            self.log.error(message)
            self.save_img(self.img, self.TEST_CASE_NAME, region, msg=message)
            self.fail(msg=message)
        else:
            self.log.debug("Region '{region}' is not empty (STD {std}} > {limit:.2f})".format(region=region,
                                                                                              std=std,
                                                                                              limit=threshold))

    def assertColor(self, region, color, threshold=None, bgcolor=None, metric=None, msg=""):
        """
        Checks if the most dominant color is the given color. Background color can be specified.

        :param region: boundaries of intrested area
        :type  region: [left, top, right, bottom]
        :param color: expected color
        :type  color: str or Tuple(B, G, R)
        :param float threshold: threshold of the test, `LIMIT_COLOR_DISTANCE` by default
        :param bgcolor: background color, set to None to determine automatically
        :type  bgcolor: str or Tuple(B, G, R)
        :param metric: function to use to calculate the color distance `d = metrics((B, G, R), (B, G, R))`
        :type  metric: callable
        :param msg: optional assertion message
        :type  msg: str
        """
        if metric is None:
            metric = bretina.rgb_rms_distance

        assert callable(metric), "`metric` parameter has to be callable function with two parameters"

        if threshold is None:
            threshold = self.LIMIT_COLOR_DISTANCE

        assert threshold >= 0.0, '`threshold` has to be a positive float'

        roi = bretina.crop(self.img, region, self.SCALE)
        dominant_color = bretina.active_color(roi, bgcolor=bgcolor)
        dist = metric(dominant_color, color)

        # test if color is close to the expected
        if dist > threshold:
            message = "Color {color} is too far from {expected} (distance {distance:.2f} > {limit:.2f}): {msg}".format(
                            color=bretina.color_str(dominant_color),
                            expected=bretina.color_str(color),
                            distance=dist,
                            limit=threshold,
                            msg=msg)
            self.log.error(message)
            self.save_img(self.img, self.TEST_CASE_NAME, region, msg=message)
            self.fail(msg=message)
        else:
            self.log.debug("Color {color} equals to {expected} ({distance:.2f} <= {limit:.2f})".format(color=bretina.color_str(dominant_color),
                                                                                                       expected=bretina.color_str(color),
                                                                                                       distance=dist,
                                                                                                       limit=threshold))

    def assertText(self, region, text,
                   language="eng", msg="", circle=False, bgcolor=None, chars=None, floodfill=False, sliding=False, ratio=None, simchars=None):
        """
        Checks the text in the given region.

        When `ratio` is not specified, the comparision method is ignores differences in `CONFUSABLE_CHARACTERS`.
        You can override `CONFUSABLE_CHARACTERS` with `simchars` parameter. If ratio is specified, than it is
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
            simchars = self.CONFUSABLE_CHARACTERS

        # Local compare function
        def equals(a, b):
            if ratio is not None:
                assert ratio <= 1.0 and ratio >= 0.0, '`ratio` has to be float in range [0, 1], {} given'.format(ratio)
                return bretina.equal_str_ratio(a, b, ratio)
            else:
                return bretina.equal_str(a, b, simchars)

        # For single line text try to use sliding text reader
        if not equals(readout, text) and not multiline and sliding:
            cnt, regions = bretina.text_cols(roi, self.SCALE, 'black', limit=0.10)
            active = regions[cnt-1][1] > (roi.shape[1] * 0.9)
            sliding_text = bretina.SlidingTextReader()

            # Gather sliding animation frames
            if active:
                while active:
                    img = self.camera.acquire_calibrated_image()
                    img = self._preprocess(img)
                    img = bretina.crop(img, region, self.SCALE, border=self.BORDER)
                    active = sliding_text.unite_animation_text(img, 20, bg_color='black', transparent=True)

                roi = sliding_text.get_image()
                readout = bretina.read_text(roi, language, False, circle=circle, bgcolor=bgcolor, chars=chars, floodfill=floodfill)

                if not equals(readout, text):
                    top = int(region[3] * self.SCALE)
                    if roi.shape[1] < self.img.shape[1]:
                        left = int(region[0] * self.SCALE)
                        bottom = top + roi.shape[0]
                        if left+roi.shape[1] < self.img.shape[1]:
                            right = left + roi.shape[1]
                        else:
                            left = self.img.shape[1]-roi.shape[1]
                            right = self.img.shape[1]
                    else:
                        width = self.img.shape[1]
                        height = roi.shape[0] * self.img.shape[1] / roi.shape[0]
                        roi = cv.resize(roi, (width, height), interpolation=cv.INTER_CUBIC)
                        bottom = top + height
                        right = self.img.shape[1]
                        left = 0
                    self.img[top:bottom, left:right] = roi
                    self.img = cv.rectangle(self.img, (left, top), (right, bottom), bretina.COLOR_GREEN)

        if not equals(readout, text):
            message = "Text '{readout}' does not match expected '{expected}': {msg}".format(readout=readout,
                                                                                            expected=text,
                                                                                            msg=msg)
            self.log.error(message)

            # show also diffs for short texts
            if len(text) <= self.MAX_STRING_DIFF_LEN:
                message += "\n\n" + "\n".join(self._diff_string(readout, text))

            self.save_img(self.img, self.TEST_CASE_NAME, region, msg=message)
            self.fail(msg=message)
        else:
            try:
                self.log.debug("Text '{readout}' matched with '{expected}'".format(readout=readout,
                                                                                   expected=text))
            except UnicodeEncodeError as ex:
                pass

    def assertImage(self, region, template_name, threshold=None, msg=""):
        """
        Checks if image is present in the given region.

        :param region: boundaries of intrested area
        :type  region: [left, top, right, bottom]
        :param str template_name: file name of the expected image relative to `self.template_path`
        :param float threshold: threshold value used in the test for the image, `LIMIT_IMAGE_MATCH` is the default
        :param str msg: optional assertion message
        """
        if threshold is None:
            threshold = self.LIMIT_IMAGE_MATCH

        assert threshold <= 1.0 and threshold >= 0.0, "`threshold` has to be float in range [0, 1]"

        roi = bretina.crop(self.img, region, self.SCALE)
        path = os.path.join(self.template_path, template_name)
        template = cv.imread(path)

        if template is None:
            message = 'Template file {} is missing! Full path: {}'.format(template_name, path)
            self.log.error(message)
            self.fail(message)

        template = bretina.resize(template, self.SCALE)
        match = bretina.recognize_image(roi, template)

        if match < threshold:
            message = "Template '{name}' does not match with given region content, matching level {level:.2f} < {limit:.2f}: {msg}".format(
                            name=template_name,
                            level=match,
                            limit=threshold,
                            msg=msg)
            self.log.error(message)
            self.save_img(self.img, self.TEST_CASE_NAME, region, msg=message)
            self.fail(msg=message)
        elif match >= threshold and match <= (threshold + 0.05):
            message = "Template '{name}' matching level {level:.2f} is close to the limit {limit:.2f}.".format(
                            name=template_name,
                            level=match,
                            limit=threshold)
            self.log.warning(message)
        else:
            self.log.debug("Template '{name}' matched ({level:.2f} >= {limit:.2f})".format(name=template_name,
                                                                                           level=match,
                                                                                           limit=threshold))

    def assertEmptyAnimation(self, region, threshold=None, bgcolor=None, bgcolor_threshold=None, metric=None, msg=""):
        """
        Check if the region is empty. Checks standart deviation of the color lightness
        and optionally average color to be bgcolor.

        :param region: boundaries of intrested area
        :type  region: [left, top, right, bottom]
        :param float threshold: threshold of the test, `LIMIT_EMPTY_STD` by default
        :param bgcolor: background color, compared with actual background if not None
        :type  bgcolor: str or Tuple(B, G, R)
        :param float bgcolor_threshold: threshold of the background color test, `LIMIT_COLOR_DISTANCE` by default
        :param metrics: function to use to calculate the color distance `d = metrics((B, G, R), (B, G, R))`
        :type  metrics: callable
        :param str msg: optional assertion message
        """
        if threshold is None:
            threshold = self.LIMIT_EMPTY_STD

        assert threshold >= 0.0, '`threshold` has to be a positive float'

        roi = [bretina.crop(img, region, self.SCALE) for img in self.imgs]
        roi_gray = [bretina.img_to_grayscale(img) for img in roi]
        std = [bretina.lightness_std(img) for img in roi_gray]
        position = np.argmax(std)

        # check if standart deviation of the lightness is low
        if max(std) > threshold:
            message = "Region '{region}' is not empty (STD {std:.2f} > {limit:.2f}): {msg}".format(region=region,
                                                                                                   std=max(std),
                                                                                                   limit=threshold,
                                                                                                   msg=msg)
            self.log.error(message)
            self.save_img(self.img, self.TEST_CASE_NAME, region, msg=message)
            self.fail(msg=message)
        else:
            self.log.debug("Region '{region}' is empty (STD {std:.2f} > {limit:.2f})".format(region=region,
                                                                                             std=max(std),
                                                                                             limit=threshold))

        # check if average color is close to expected background
        if bgcolor is not None:
            if metric is None:
                metric = bretina.rgb_rms_distance
            else:
                assert callable(metric), "`metric` parameter has to be callable function with two parameters"

            avgcolors = [bretina.mean_color(img) for img in roi]
            avgcolor = max(avgcolors) if metric(max(avgcolors), bgcolor) > metric(min(avgcolors), bgcolor) else max(avgcolors)
            dist = max(metric(max(avgcolors), bgcolor), metric(min(avgcolors), bgcolor))

            if bgcolor_threshold is None:
                bgcolor_threshold = self.LIMIT_COLOR_DISTANCE

            if dist > bgcolor_threshold:
                message = "Region {region} background color is not as expected {background} != {expected} (distance {distance:.2f} > {limit:.2f}): {msg}".format(
                                region=region,
                                background=bretina.color_str(avgcolor),
                                expected=bretina.color_str(bgcolor),
                                distance=dist,
                                limit=bgcolor_threshold,
                                msg=msg)
                self.log.error(message)
                self.save_img(self.imgs[0], self.TEST_CASE_NAME, region, msg=message)
                self.fail(msg=message)
            else:
                self.log.debug("Background color distance ({distance:.2f} <= {limit:.2f}).".format(distance=dist, limit=bgcolor_threshold))

    def assertImageAnimation(self, region, template_name, animation_active, size, threshold=None, msg=""):
        """
        Checks if the image animation is present in the given region.

        :param region: boundaries of intrested area
        :type  region: [left, top, right, bottom]
        :param str template_name: file name of the expected image relative to `self.template_path`
        :param float threshold: threshold value used in the test for the image, `LIMIT_IMAGE_MATCH` is the default
        :param msg: optional assertion message
        """
        if threshold is None:
            threshold = self.LIMIT_IMAGE_MATCH

        assert threshold <= 1.0 and threshold >= 0.0, "`threshold` has to be float in range [0, 1]"

        roi = [bretina.crop(img, region, self.SCALE) for img in self.imgs]
        path = os.path.join(self.template_path, template_name)
        template = cv.imread(path)

        if template is None:
            message = 'Template file {} is missing! Full path: {}'.format(template_name, path)
            self.log.error(message)
            self.fail(message)

        conformity, animation = bretina.recognize_animation(roi, template, size, self.SCALE)

        template_crop = bretina.crop(template, [0, 0, size[0], size[1]], 1, 0)
        position = np.argmax([bretina.recognize_image(img, template_crop) for img in roi])

        if conformity < threshold:
            message = "Template '{name}' does not match with given region content, matching level {level:.2f} < {limit:.2f}: {msg}".format(
                        name=template_name,
                        level=conformity,
                        limit=threshold,
                        msg=msg)
            self.log.error(message)
            self.save_img(self.imgs[position], self.TEST_CASE_NAME, region, msg=message)
            self.fail(msg=message)
        elif conformity >= threshold and conformity <= (threshold + 0.05):
            message = "Template '{name}' matching level {level:.2f} is close to the limit {limit:.2f}.".format(
                            name=template_name,
                            level=conformity,
                            limit=threshold)
            self.log.warning(message)
        else:
            self.log.debug("Animation template '{name}' matched ({level:.2f} > {limit:.2f})".format(name=template_name,
                                                                                                    level=conformity,
                                                                                                    limit=threshold))

        if animation != animation_active:
            message = "Template '{name}' does not meets the assumption that animation is {theoretic:.2f} but is {real:.2f}: {msg}".format(
                        name=template_name,
                        theoretic=animation_active,
                        real=animation,
                        msg=msg)
            self.log.error(message)
            self.save_img(self.imgs[0], self.TEST_CASE_NAME, region, msg=message)
            self.fail(msg=message)
