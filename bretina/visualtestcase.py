"""Visual Test Case implementation"""

import unittest
import numpy as np
import textwrap
import difflib
import bretina
import time
import cv2 as cv
import os

from PIL import Image, ImageFont, ImageDraw
from datetime import datetime

#: Name of the default color metric
DEFAULT_COLOR_METRIC = "rgb_rms_distance"


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
    LIMIT_COLOR_DISTANCE = 50.0
    #: Default threshold value for the image asserts, if diff is > LIMIT_IMAGE_MATCH, assert fails.
    LIMIT_IMAGE_MATCH = 1.0
    #: Max len of string for which is the diff displayed
    MAX_STRING_DIFF_LEN = 80
    #: Scaling
    SCALE = 3.0
    #: Border
    BORDER = 0

    #: path where the log images should be stored
    LOG_PATH = './log/'
    TEMPLATE_PATH = './'
    LOG_IMG_FORMAT = "JPG"
    PASS_IMG_FORMAT = "JPG"
    SRC_IMG_FORMAT = "PNG"

    #: set to true to save also source image when assert fails
    SAVE_SOURCE_IMG = False
    #: set to true to save also source image when assert pass
    SAVE_PASS_IMG = False

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
    PRE_DENOISE_APPLY = False
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
        self._DEFAULT_COLOR_METRIC = getattr(bretina, DEFAULT_COLOR_METRIC)
        self.img = None                             #: here is stored the currently captured image
        self.imgs = None                            #:
        self.camera = None

    def _preprocess(self, img):
        """
        Appling filters on the acquired image.

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

    def capture(self, delay=0):
        """
        Captures image from the camera and does the preprocessing. Pre-processed image is stored in the `self.img`.

        :param float delay: delay in [s] before camera captures an image
        """
        if delay > 0:
            time.sleep(delay)

        img = self.camera.acquire_calibrated_image()
        self.img = self._preprocess(img)

    def capture_images(self, num_images, period):
        """
        Captures image from the camera and does the preprocessing.

        Sequence of pre-processed images is stored in the `self.imgs`.
        """
        raws = self.camera.acquire_calibrated_images(num_images, period)
        self.imgs = [self._preprocess(raw) for raw in raws]

    def save_img(self, img, name, img_format="jpg", border_box=None, msg=None, color='red'):
        """
        Writes the actual image to the file with the name based on the current time and the given name.

        :param str name: name of the file
        :param str format: image file format ('jpg', 'png', 'bmp')
        :param border_box: specify this parameter to draw a rectangle to this region in the stored image
        :type  border_box: Tuple[left, top, right, bottom]
        :param color: color of rectangle and text
        """
        color = bretina.color(color)
        now = datetime.now()
        filename = now.strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]

        if (name is not None) and (len(str(name)) > 0):
            filename += "_" + str(name)

        directory = os.path.join(self.LOG_PATH, now.strftime('%Y-%m-%d'))

        if not os.path.isdir(directory):
            os.makedirs(directory)

        extension = img_format.lower()
        if not extension.startswith('.'):
            extension = '.' + extension

        path = os.path.join(directory, filename + extension)

        if border_box is not None:
            img = bretina.draw_border(img, border_box, self.SCALE, color=color)
        else:
            border_box = [0, img.shape[0] / self.SCALE, img.shape[1] / self.SCALE, img.shape[0] / self.SCALE]

        if msg is not None:
            margin = 6
            font_size = 26
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

            while cnt < 10:
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

            # overflow of image width - shift text to left
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
            draw.multiline_text(text_pt, msg, fill=bretina.color_str(color), font=font, spacing=spacing)

            # Get back the image to OpenCV
            img = cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR)

        cv.imwrite(path, img)

        if self.log is not None:
            self.log.debug(bretina.ImageRecord(img))

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
            message = f"Region '{region}' not empty (STD {std:.2f} > {threshold:.2f}): {msg}"
            self.log.error(message)
            self.save_img(self.img, self.id(), self.LOG_IMG_FORMAT, region, message, bretina.COLOR_RED)

            if self.SAVE_SOURCE_IMG:
                self.save_img(self.img, self.id() + "-src", img_format=self.SRC_IMG_FORMAT)

            self.fail(msg=message)
        # when OK
        else:
            message = f"Region '{region}' empty (STD {std:.2f} <= {threshold:.2f})"
            self.log.debug(message)

            if self.SAVE_PASS_IMG:
                self.save_img(self.img, self.id() + "-pass", self.PASS_IMG_FORMAT, region, message, bretina.COLOR_GREEN)

        # check if average color is close to expected background
        if bgcolor is not None:
            if metric is None:
                metric = self._DEFAULT_COLOR_METRIC
            else:
                assert callable(metric), "`metric` parameter has to be callable function with two parameters"

            avgcolor = bretina.mean_color(roi)
            dist = metric(avgcolor, bgcolor)

            if bgcolor_threshold is None:
                bgcolor_threshold = self.LIMIT_COLOR_DISTANCE

            # check distance from background color
            if dist > bgcolor_threshold:
                message = f"Background {bretina.color_str(avgcolor)} != {bretina.color_str(bgcolor)} (expected) (distance {dist:.2f} > {bgcolor_threshold:.2f}): {msg}"
                self.log.error(message)
                self.save_img(self.img, self.id(), self.LOG_IMG_FORMAT, region, message, bretina.COLOR_RED)

                if self.SAVE_SOURCE_IMG:
                    self.save_img(self.img, self.id() + "-src", img_format=self.SRC_IMG_FORMAT)

                self.fail(msg=message)
            # when OK
            else:
                message = f"Background {bretina.color_str(avgcolor)} == {bretina.color_str(bgcolor)} (expected) (distance {dist:.2f} <= {bgcolor_threshold:.2f})"
                self.log.debug(message)

                if self.SAVE_PASS_IMG:
                    self.save_img(self.img, self.id() + "-pass", self.PASS_IMG_FORMAT, region, message, bretina.COLOR_GREEN)

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
            message = f"Region '{region}' empty (STD {std} <= {threshold:.2f}): {msg}"
            self.log.error(message)
            self.save_img(self.img, self.id(), self.LOG_IMG_FORMAT, region, message, bretina.COLOR_RED)

            if self.SAVE_SOURCE_IMG:
                self.save_img(self.img, self.id() + "-src", img_format=self.SRC_IMG_FORMAT)

            self.fail(msg=message)
        # when OK
        else:
            message = f"Region '{region}' not empty (STD {std} > {threshold:.2f})"
            self.log.debug(message)

            if self.SAVE_PASS_IMG:
                self.save_img(self.img, self.id() + "-pass", self.PASS_IMG_FORMAT, region, message, bretina.COLOR_GREEN)

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
            metric = self._DEFAULT_COLOR_METRIC

        assert callable(metric), "`metric` parameter has to be callable function with two parameters"

        if threshold is None:
            threshold = self.LIMIT_COLOR_DISTANCE

        assert threshold >= 0.0, '`threshold` has to be a positive float'

        roi = bretina.crop(self.img, region, self.SCALE)

        if bgcolor is None:
            dominant_color = bretina.dominant_color(roi)
        else:
            dominant_color = bretina.active_color(roi, bgcolor=bgcolor)

        dist = metric(dominant_color, color)

        # test if color is close to the expected
        if dist > threshold:
            message = f"Color {bretina.color_str(dominant_color)} != {bretina.color_str(color)} (expected) (distance {dist:.2f} > {threshold:.2f}): {msg}"
            self.log.error(message)
            self.save_img(self.img, self.id(), self.LOG_IMG_FORMAT, region, message, bretina.COLOR_RED)

            if self.SAVE_SOURCE_IMG:
                self.save_img(self.img, self.id() + "-src", img_format=self.SRC_IMG_FORMAT)

            self.fail(msg=message)
        # when OK
        else:
            message = f"Color {bretina.color_str(dominant_color)} == {bretina.color_str(color)} (expected) (distance {dist:.2f} <= {threshold:.2f})"
            self.log.debug(message)

            if self.SAVE_PASS_IMG:
                self.save_img(self.img, self.id() + "-pass", self.PASS_IMG_FORMAT, region, message, bretina.COLOR_GREEN)

    def assertText(self, region, text,
                   language="eng", msg="", circle=False, bgcolor=None, chars=None, langchars=False, floodfill=False, sliding=False, threshold=1.0, simchars=None, ligatures=None, ignore_accents=True):
        """
        Checks the text in the given region.

        :param list region: boundaries of intrested area [left, top, right, bottom]
        :param str text: expected text co compare
        :param str language: language of the string, use 3-letter ISO codes: https://github.com/tesseract-ocr/tesseract/wiki/Data-Files
        :param str msg: optional assertion message
        :param bool circle: optional flag to tell OCR engine that the text is in circle
        :param bgcolor: background color
        :param str chars: optional limit of the used characters in the OCR
        :param bool langchars: recognized characters are limited only to chars in the `language`
        :param bool floodfill: optional argument to apply flood fill to unify background
        :param bool sliding: optional argument
            - `False` to prohibit sliding text animation recognition
            - `True` to check also sliding text animation, can lead to long process time
        :param float threshold: measure of the sequences similarity as a float in the range [0, 1], see
            https://docs.python.org/3.8/library/difflib.html#difflib.SequenceMatcher.ratio
        :param list simchars: allowed similar chars in text comparision, e.g. ["1l", "0O"]. Differences in these characters are not taken as differences.
        :param list ligatures: list of char combinations which shall be unified to prevent confusion e.g. [("τπ", "πτ")]
        :param bool ignore_accents: when set to `True`, given and OCR-ed texts are cleared from diacritic, accents, umlauts, ... before comparision
            (e.g. "příliš žluťoučký kůň" is treated as "prilis zlutoucky kun").
        """
        border = 5
        sliding_counter = 50

        # remove accents from the expected text
        if ignore_accents:
            text = bretina.remove_accents(text)

        # get string from image
        roi = bretina.crop(self.img, region, self.SCALE, border=border)
        multiline = bretina.text_rows(roi, self.SCALE)[0] > 1
        readout = bretina.read_text(roi, language, multiline, circle=circle, bgcolor=bgcolor, chars=chars, floodfill=floodfill, langchars=langchars)

        # remove accents from the OCR-ed text
        if ignore_accents:
            readout = bretina.remove_accents(readout)

        if simchars is None:
            simchars = bretina.CONFUSABLE_CHARACTERS

        if ligatures is None:
            ligatures = bretina.LIGATURE_CHARACTERS

        assert threshold <= 1.0 and threshold >= 0.0, '`threshold` has to be float in range [0, 1], {} given'.format(threshold)

        # check equality of the strings
        equal, equal_ratio = bretina.equal_str_ratio(readout, text, simchars, ligatures, threshold)

        # if not equal, for single line text try to use sliding text reader if sliding is not prohibited
        if not equal and not multiline and sliding:
            # but first verify if the text covers more than 90% of the region
            cnt, regions = bretina.text_cols(roi, self.SCALE, bgcolor='black')

            if (cnt > 0) and (regions[-1][1] > (roi.shape[1] * 0.9)):
                # gather sliding animation frames
                sliding_text = bretina.SlidingTextReader()
                active = sliding_text.unite_animation_text(roi, sliding_counter, bgcolor='black', transparent=True)

                while active:
                    img = self.camera.acquire_calibrated_image()
                    img = bretina.crop(img, region, self.SCALE, border=border)
                    img = self._preprocess(img)
                    active = sliding_text.unite_animation_text(img, sliding_counter, bgcolor='black', transparent=True)

                roi = sliding_text.get_image()
                readout = bretina.read_text(roi, language, False, circle=circle, bgcolor=bgcolor, chars=chars, floodfill=floodfill, langchars=langchars)

                # remove accents from the OCRed text
                if ignore_accents:
                    readout = bretina.remove_accents(readout)

                equal, equal_ratio = bretina.equal_str_ratio(readout, text, simchars, ligatures, threshold)

                # TODO: put this part into special func or something
                if not equal:
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
                    self.img = cv.rectangle(self.img, (left, top), (right, bottom), bretina.COLOR_YELLOW)

        if not equal:
            message = f"Text '{readout}' != '{text}' (expected) ({equal_ratio:.3f} < {threshold:.3f}): {msg}"
            self.log.error(message)

            # show also diffs for short texts
            # TODO: dont show differences which are not considered due to simchars
            if len(text) <= self.MAX_STRING_DIFF_LEN:
                message += "\n\n" + "\n".join(self._diff_string(readout, text))

            self.save_img(self.img, self.id(), self.LOG_IMG_FORMAT, region, message, bretina.COLOR_RED)

            if self.SAVE_SOURCE_IMG:
                self.save_img(self.img, self.id() + "-src", img_format=self.SRC_IMG_FORMAT)

            self.fail(msg=message)
        # when OK
        else:
            message = f"Text '{readout}' == '{text}' (expected) ({equal_ratio:.3f} >= {threshold:.3f})"
            self.log.debug(message)

            if self.SAVE_PASS_IMG:
                self.save_img(self.img, self.id() + "-pass", self.PASS_IMG_FORMAT, region, message, bretina.COLOR_GREEN)

    def assertImage(self, region, template_name, threshold=None, edges=False, inv=None, bgcolor=None, blank=None, msg=""):
        """
        Checks if image is present in the given region.

        :param region: boundaries of intrested area
        :type  region: [left, top, right, bottom]
        :param str template_name: file name of the expected image relative to `self.template_path`
        :param float threshold: threshold value used in the test for the image, `LIMIT_IMAGE_MATCH` is the default
        :param bool edges: controls if the comparision shall be done on edges only
        :param bool inv: specifies if image is inverted
                        - [True]   images are inverted before processing (use for dark lines on light background)
                        - [False]  images are not inverted before processing (use for light lines on dark background)
                        - [None]   inversion is decided automatically based on `img` background
        :param bgcolor: specify color which is used to fill transparent areas in png with alpha channel, decided automatically when None
        :param list blank: list of areas which shall be masked
        :param str msg: optional assertion message
        """
        if threshold is None:
            threshold = self.LIMIT_IMAGE_MATCH

        assert threshold >= 0.0, "`threshold` has to be float in range [0, 1]"

        roi = bretina.crop(self.img, region, self.SCALE)
        path = os.path.join(self.template_path, template_name)
        template = cv.imread(path, cv.IMREAD_UNCHANGED)

        if template is None:
            message = 'Template file {} is missing! Full path: {}'.format(template_name, path)
            self.log.error(message)
            self.fail(message)

        # resize blanked areas
        if blank is not None:
            assert isinstance(blank, list), '`blank` has to be list'

            # make list if only one area is given
            if len(blank) > 0 and not isinstance(blank[0], list):
                blank = [blank]

            for i in range(len(blank)):
                for j in range(len(blank[i])):
                    blank[i][j] *= self.SCALE

        # get difference between template and ROI
        template = bretina.resize(template, self.SCALE)
        diff = bretina.img_diff(roi, template, edges=edges, inv=inv, bgcolor=bgcolor, blank=blank)

        # check the diff level
        if diff > threshold:
            message = f"Image '{template_name}' is different ({diff:.3f} > {threshold:.3f}): {msg}"
            self.log.error(message)
            self.save_img(self.img, self.id(), self.LOG_IMG_FORMAT, region, message, bretina.COLOR_RED)

            if self.SAVE_SOURCE_IMG:
                self.save_img(self.img, self.id() + "-src", img_format=self.SRC_IMG_FORMAT)

            self.fail(msg=message)
        # diff level is close to the limit, show warning
        elif diff <= threshold and diff >= (threshold * 1.1):
            message = f"Image '{template_name}' difference {diff:.3f} close to limit {threshold:.3f}."
            self.log.warning(message)

            if self.SAVE_PASS_IMG:
                self.save_img(self.img, self.id() + "-pass", self.PASS_IMG_FORMAT, region, message, bretina.COLOR_ORANGE)
        # when OK
        else:
            message = f"Image '{template_name}' matched ({diff:.5f} <= {threshold:.5f})"
            self.log.debug(message)

            if self.SAVE_PASS_IMG:
                self.save_img(self.img, self.id() + "-pass", self.PASS_IMG_FORMAT, region, message, bretina.COLOR_GREEN)

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
            message = f"Region '{region}' not empty (STD {max(std):.2f} > {threshold:.2f}): {msg}"
            self.log.error(message)
            self.save_img(self.imgs[position], self.id(), self.LOG_IMG_FORMAT, region, message, bretina.COLOR_RED)

            if self.SAVE_SOURCE_IMG:
                self.save_img(self.imgs[position], self.id() + "-src", img_format=self.SRC_IMG_FORMAT)

            self.fail(msg=message)
        # when OK
        else:
            message = f"Region '{region}' empty (STD {max(std):.2f} > {threshold:.2f})"
            self.log.debug(message)

            if self.SAVE_PASS_IMG:
                self.save_img(self.imgs[position], self.id() + "-pass", self.PASS_IMG_FORMAT, region, message, bretina.COLOR_GREEN)

        # check if average color is close to expected background
        if bgcolor is not None:
            if metric is None:
                metric = self._DEFAULT_COLOR_METRIC
            else:
                assert callable(metric), "`metric` parameter has to be callable function with two parameters"

            avgcolors = [bretina.mean_color(img) for img in roi]
            avgcolor = max(avgcolors) if metric(max(avgcolors), bgcolor) > metric(min(avgcolors), bgcolor) else max(avgcolors)
            dist = max(metric(max(avgcolors), bgcolor), metric(min(avgcolors), bgcolor))

            if bgcolor_threshold is None:
                bgcolor_threshold = self.LIMIT_COLOR_DISTANCE

            # check backgrould color is close to the expected one
            if dist > bgcolor_threshold:
                message = f"Region {region} background {bretina.color_str(avgcolor)} != {bretina.color_str(bgcolor)} (expected) ({dist:.2f} > {bgcolor_threshold:.2f}): {msg}"
                self.log.error(message)
                self.save_img(self.imgs[0], self.id(), self.LOG_IMG_FORMAT, region, message, bretina.COLOR_RED)

                if self.SAVE_SOURCE_IMG:
                    self.save_img(self.imgs[0], self.id() + "-src", img_format=self.SRC_IMG_FORMAT)

                self.fail(msg=message)
            # when OK
            else:
                message = f"Background {bretina.color_str(avgcolor)} == {bretina.color_str(bgcolor)} ({dist:.2f} <= {bgcolor_threshold:.2f})."
                self.log.debug(message)

                if self.SAVE_PASS_IMG:
                    self.save_img(self.imgs[0], self.id() + "-pass", self.PASS_IMG_FORMAT, region, message, bretina.COLOR_GREEN)

    def assertImageAnimation(self, region, template_name, animation_active, size, threshold=None, bgcolor=None, msg="", split_threshold=64):
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

        assert threshold >= 0.0, "`threshold` has to be float in range [0, 1]"

        roi = [bretina.crop(img, region, self.SCALE) for img in self.imgs]
        path = os.path.join(self.template_path, template_name)
        template = cv.imread(path)

        if template is None:
            message = f'Template file {template_name} is missing! Full path: {path}'
            self.log.error(message)
            self.fail(message)


        diff, animation = bretina.recognize_animation(roi, template, size, self.SCALE, split_threshold=split_threshold)

        template_crop = bretina.crop(template, [0, 0, size[0], size[1]], 1, 0)
        template_crop = bretina.resize(template_crop, self.SCALE)
        position = np.argmax([bretina.img_diff(img, template_crop, bgcolor=bgcolor) for img in roi])

        # check difference with the animation
        if diff > threshold:
            message = f"Animation '{template_name}' not matched {diff:.2f} < {threshold:.2f}: {msg}"
            self.log.error(message)
            self.save_img(self.imgs[0], self.id(), self.LOG_IMG_FORMAT, region, message, bretina.COLOR_RED)

            if self.SAVE_SOURCE_IMG:
                self.save_img(self.imgs[0], self.id() + "-src", img_format=self.SRC_IMG_FORMAT)

            self.fail(msg=message)
        # show warning if difference is close to the threshold
        elif diff <= threshold and diff >= (threshold * 1.1):
            message = f"Animation '{template_name}' matched but close to limit ({diff:.2f} >= {threshold:.2f})."
            self.log.warning(message)

            if self.SAVE_PASS_IMG:
                self.save_img(self.imgs[0], self.id() + "-pass", self.PASS_IMG_FORMAT, region, message, bretina.COLOR_ORANGE)
        # when OK
        else:
            message = f"Animation '{template_name}' matched ({diff:.2f} >= {threshold:.2f})"
            self.log.debug(message)

            if self.SAVE_PASS_IMG:
                self.save_img(self.imgs[0], self.id() + "-pass", self.PASS_IMG_FORMAT, region, message, bretina.COLOR_GREEN)

        if animation != animation_active:
            message = f"Animation '{template_name}' activity {animation} != {animation_active} (expected): {msg}"
            self.log.error(message)
            self.save_img(self.imgs[0], self.id(), self.LOG_IMG_FORMAT, region, message, bretina.COLOR_RED)

            if self.SAVE_SOURCE_IMG:
                self.save_img(self.imgs[0], self.id() + "-src", img_format=self.SRC_IMG_FORMAT)

            self.fail(msg=message)
