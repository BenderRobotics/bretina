"""Visual Test Case implementation"""

import unittest
import numpy as np
import itertools
import logging
import bretina
import time
import cv2 as cv
import os
import re

from bretina.polyline import get_polyline_coordinates, linear_transform_points
from htmllogging import ImageRecord

#: Name of the default color metric
DEFAULT_COLOR_METRIC = "rgb_rms_distance"


class VisualTestCase(unittest.TestCase):
    """
    A class whose instances are single test cases.

    VisualTestCase is a subclass of the standart unittest TestCase,
    therefore there are all features from the unittest module and some
    additional asserts for the image processing.

    If the fixture may be used for many test cases, create as
    many test methods as are needed.

    Test authors should subclass VisualTestCase for their own tests.
    Construction and deconstruction of the test's environment ('fixture') can
    be implemented by overriding the 'setUp' and 'tearDown' methods respectively.

    If it is necessary to override the `__init__` method, the base class
    `__init__` method must always be called. It is important that subclasses
    should not change the signature of their `__init__` method, since instances
    of the classes are instantiated automatically by parts of the framework
    in order to be run.
    """

    #: Default threshold value for the assertEmpty and assertNotEmpty.
    LIMIT_EMPTY_STD = 16.0
    #: Default threshold value for the assertColor.
    LIMIT_COLOR_DISTANCE = 50.0
    #: Default threshold value for the assertCurve.
    LIMIT_POINTS_FINDING = 35.0
    #: Default threshold value for the assertImage, if diff is > LIMIT_IMAGE_MATCH, assert fails.
    LIMIT_IMAGE_MATCH = 1.0
    #: Max len of string for which is the diff displayed.
    MAX_STRING_DIFF_LEN = 60
    #: Scaling between image dimensions and the given coordinates.
    SCALE = 3.0

    #: Path where the log images should be stored.
    LOG_PATH = './log/'
    #: Format of the log image.
    LOG_IMG_FORMAT = "JPG"
    #: Format of the pass image.
    PASS_IMG_FORMAT = "JPG"
    #: Format of the fail image.
    SRC_IMG_FORMAT = "PNG"

    #: Set to true to save also source image when assert fails.
    SAVE_SOURCE_IMG = False
    #: Set to true to save also source image when assert pass.
    SAVE_PASS_IMG = False

    #: Sets if the bilateral filtering is applied during pre-processing.
    PRE_BIL_FILTER_APPLY = True
    #: Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace.
    PRE_BIL_FILTER_DIAMETER = 5
    #: Filter sigma in the color space. A larger value of the parameter means that farther colors within
    #: the pixel neighborhood will be mixed together, resulting in larger areas of semi-equal color.
    PRE_BIL_FILTER_SIGMA_COLOR = 20.0
    #: Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence
    #: each other as long as their colors are close enough. When PRE_FILTER_DIAMETER>0, it specifies the neighborhood
    #: size regardless of sigmaSpace. Otherwise, d is proportional to sigmaSpace.
    PRE_BIL_FILTER_SIGMA_SPACE = 10.0

    #: Sets if the Non-local Means De-noising algorithm is applied during pre-processing.
    PRE_DENOISE_APPLY = False
    #: Parameter regulating filter strength for luminance component. Bigger h value perfectly removes noise but also
    #: removes image details, smaller h value preserves details but also preserves some noise.
    PRE_DENOISE_H_LIGHT = 3
    #: The same as h but for color components. For most images value equals 10 will be enough to remove colored noise
    #: and do not distort colors.
    PRE_DENOISE_H_COLOR = 3
    #: Size in pixels of the template patch that is used to compute weights. Should be odd.
    PRE_DENOISE_TEMP_WIN_SIZE = 7
    #: Size in pixels of the window that is used to compute weighted average for given pixel. Should be odd.
    #: Affect performance linearly: greater searchWindowsSize - greater de-noising time.
    PRE_DENOISE_SEARCH_WIN_SIZE = 11
    #: level of loging used for the error records
    ERROR_LOG_LEVEL = logging.ERROR
    #: max len of the test ID
    MAX_TEST_ID_LEN = 120
    #: Path to the templates
    template_path = ''

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self._DEFAULT_COLOR_METRIC = getattr(bretina, DEFAULT_COLOR_METRIC)
        self.img = None             #: Here is stored the currently captured image.
        self.imgs = None            #: Here is stored sequence of the images for the animation asserts.
        self.camera = None          #: Reference to the camera interface.
        self.log = logging.getLogger()
        self._available_tessdata = bretina.get_tesseract_trained_data()

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

    def _test_id(self):
        """
        Returns ID of the current test or sub-test

        :return:  ID of the current test or sub-test
        :rtype: str
        """
        test_id = ''
        invalid_characters = ("'", '"', '/', '\\', '?', '<', '>', ':', '|', '*')

        try:
            # use subtest ID if possible (already contains test ID), but it will fail for non sub tests
            test_id = self._subtest.id()
        except Exception:
            # fallback to test ID
            test_id = self.id()
        finally:
            if '.' in test_id:
                test_id = '.'.join(test_id.split('.')[1:])

            for char in invalid_characters:
                test_id = test_id.replace(char, '')

            return test_id[:self.MAX_TEST_ID_LEN]

    def capture(self, delay=0.25):
        """
        Captures image from the camera and does the preprocessing. Pre-processed image is stored in the `self.img`.

        Captured image goes through the preprocessing sequence:

        * `PRE_BIL_FILTER_APPLY == True`: Bilateral filter (http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html)
          is applied on the captured images with the settings controlled by `PRE_BIL_FILTER_DIAMETER`,
          `PRE_BIL_FILTER_SIGMA_COLOR`, `PRE_BIL_FILTER_SIGMA_SPACE`. Bilateral filtering tries to remove noise and
          still preserve sharp edges.

        * `PRE_DENOISE_APPLY == True`: Perform image denoising using Non-local Means Denoising algorithm (http://www.ipol.im/pub/algo/bcm_non_local_means_denoising).
          Reduces a gaussian white noise, options controlled by `PRE_DENOISE_H_LIGHT`, `PRE_DENOISE_H_COLOR`,
          `PRE_DENOISE_TEMP_WIN_SIZE`, `PRE_DENOISE_SEARCH_WIN_SIZE`.

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

        :param int num_images: number of images to capture
        :param float period: time [s] in between two frames
        """
        raws = self.camera.acquire_calibrated_images(num_images, period)
        self.imgs = [self._preprocess(raw) for raw in raws]

    def save_img(self, img, name, format="jpg", border_box=None, msg=None, color='red', put_img=None, log_level=logging.DEBUG):
        """
        Writes the actual image to the file with the name based on the current time and the given name.

        :param img: image o be saved (in OpenCV/numpy format)
        :param str name: name of the file
        :param str img_format: image file format ('jpg', 'png', 'bmp')
        :param border_box: specify this parameter to draw a rectangle to this region in the stored image
        :type  border_box: Tuple[left, top, right, bottom]
        :param str msg: additional message to add to the saved image
        :param color: color of rectangle and text
        :param put_img: put additional image or color to picture
        :type  put_img: OpenCV image or color code
        :param int log_level: level of the log which is used to log the image
        """
        font_size = 22
        color = bretina.color(color)

        if border_box is not None:
            border_box = tuple(int(round(self.SCALE * p)) for p in border_box)
            img = bretina.draw_border(img, border_box, color=color)
        else:
            border_box = (0, 0, img.shape[1], img.shape[0])

        if put_img is not None:
            if isinstance(put_img, (str, tuple, list, set)):
                put_img = bretina.draw_color_sample(put_img, font_size=font_size)

            img = bretina.draw_image(img, put_img, border_box, border_color=bretina.COLOR_YELLOW)

        if msg:
            img = bretina.write_image_text(img, msg, font_size=font_size, color=color)

        path = bretina.get_image_filename(directory=self.LOG_PATH, name=name, extension=format.lower())
        cv.imwrite(path, img)

        if self.log is not None:
            try:
                log_level = int(log_level)
            except ValueError:
                log_level = logging.ERROR
            finally:
                self.log.log(log_level, ImageRecord(img))

        try:
            if self._subtest is not None:
                if not hasattr(self._subtest, 'extra'):
                    self._subtest.extra = []
                self._subtest.extra.append(img)
            else:
                self.extra.append(img)
        except AttributeError:
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
        :param metric: function to use to calculate the color distance `d = metrics((B, G, R), (B, G, R))`
        :type  metric: callable
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
            self.log.log(self.ERROR_LOG_LEVEL, message)
            self.save_img(self.img,
                          name=self._test_id(),
                          format=self.LOG_IMG_FORMAT,
                          border_box=region,
                          msg=message,
                          color=bretina.COLOR_RED,
                          log_level=self.ERROR_LOG_LEVEL)

            if self.SAVE_SOURCE_IMG:
                self.save_img(self.img,
                            name=self._test_id() + "-src",
                            format=self.SRC_IMG_FORMAT,
                            log_level=logging.INFO)

            self.fail(msg=message)
        # when OK
        else:
            message = f"Region '{region}' empty (STD {std:.2f} <= {threshold:.2f})"
            self.log.debug(message)

            if self.SAVE_PASS_IMG:
                self.save_img(self.img,
                              name=self._test_id() + "-pass",
                              format=self.PASS_IMG_FORMAT,
                              border_box=region,
                              msg=message,
                              color=bretina.COLOR_GREEN,
                              log_level=logging.INFO)

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
                self.log.log(self.ERROR_LOG_LEVEL, message)
                self.save_img(self.img,
                              name=self._test_id(),
                              format=self.LOG_IMG_FORMAT,
                              border_box=region,
                              msg=message,
                              color=bretina.COLOR_RED,
                              log_level=self.ERROR_LOG_LEVEL)

                if self.SAVE_SOURCE_IMG:
                    self.save_img(self.img,
                                  name=self._test_id() + "-src",
                                  format=self.SRC_IMG_FORMAT,
                                  log_level=logging.INFO)

                self.fail(msg=message)
            # when OK
            else:
                message = f"Background {bretina.color_str(avgcolor)} == {bretina.color_str(bgcolor)} (expected) (distance {dist:.2f} <= {bgcolor_threshold:.2f})"
                self.log.debug(message)

                if self.SAVE_PASS_IMG:
                    self.save_img(self.img,
                                  name=self._test_id() + "-pass",
                                  format=self.PASS_IMG_FORMAT,
                                  border_box=region,
                                  msg=message,
                                  color=bretina.COLOR_GREEN,
                                  log_level=logging.INFO)

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
            self.log.log(self.ERROR_LOG_LEVEL, message)
            self.save_img(self.img,
                          name=self._test_id(),
                          format=self.LOG_IMG_FORMAT,
                          border_box=region,
                          msg=message,
                          color=bretina.COLOR_RED,
                          log_level=self.ERROR_LOG_LEVEL)

            if self.SAVE_SOURCE_IMG:
                self.save_img(self.img,
                              name=self._test_id() + "-src",
                              img_format=self.SRC_IMG_FORMAT,
                              log_level=logging.INFO)

            self.fail(msg=message)
        # when OK
        else:
            message = f"Region '{region}' not empty (STD {std} > {threshold:.2f})"
            self.log.debug(message)

            if self.SAVE_PASS_IMG:
                self.save_img(self.img,
                              name=self._test_id() + "-pass",
                              format=self.PASS_IMG_FORMAT,
                              border_box=region,
                              msg=message,
                              color=bretina.COLOR_GREEN,
                              log_level=logging.INFO)

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
        colors = [bretina.color_str(dominant_color), bretina.color_str(color)]
        # test if color is close to the expected
        if dist > threshold:
            message = f"Color {bretina.color_str(dominant_color)} != {bretina.color_str(color)} (expected) (distance {dist:.2f} > {threshold:.2f}): {msg}"
            self.log.log(self.ERROR_LOG_LEVEL, message)
            self.save_img(self.img,
                          name=self._test_id(),
                          format=self.LOG_IMG_FORMAT,
                          border_box=region,
                          msg=message,
                          color=bretina.COLOR_RED,
                          put_img=colors,
                          log_level=self.ERROR_LOG_LEVEL)

            if self.SAVE_SOURCE_IMG:
                self.save_img(self.img,
                              name=self._test_id() + "-src",
                              format=self.SRC_IMG_FORMAT,
                              log_level=logging.INFO)

            self.fail(msg=message)
        # when OK
        else:
            message = f"Color {bretina.color_str(dominant_color)} == {bretina.color_str(color)} (expected) (distance {dist:.2f} <= {threshold:.2f})"
            self.log.debug(message)

            if self.SAVE_PASS_IMG:
                self.save_img(self.img,
                              name=self._test_id() + "-pass",
                              format=self.PASS_IMG_FORMAT,
                              border_box=region,
                              msg=message,
                              color=bretina.COLOR_GREEN,
                              put_img=colors,
                              log_level=logging.INFO)

    def assertNotColor(self, region, color, threshold=None, bgcolor=None, metric=None, msg=""):
        """
        Checks if the most dominant color is not the given color. Background color can be specified.

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
        colors = [bretina.color_str(dominant_color), bretina.color_str(color)]
        # test if color is not close to the expected
        if dist < threshold:
            message = f"Color {bretina.color_str(dominant_color)} == {bretina.color_str(color)} (expected) (distance {dist:.2f} < {threshold:.2f}): {msg}"
            self.log.log(self.ERROR_LOG_LEVEL, message)
            self.save_img(self.img,
                          name=self._test_id(),
                          format=self.LOG_IMG_FORMAT,
                          border_box=region,
                          msg=message,
                          color=bretina.COLOR_RED,
                          put_img=colors,
                          log_level=self.ERROR_LOG_LEVEL)

            if self.SAVE_SOURCE_IMG:
                self.save_img(self.img,
                              name=self._test_id() + "-src",
                              format=self.SRC_IMG_FORMAT,
                              log_level=logging.INFO)

            self.fail(msg=message)
        # when OK
        else:
            message = f"Color {bretina.color_str(dominant_color)} != {bretina.color_str(color)} (expected) (distance {dist:.2f} >= {threshold:.2f})"
            self.log.debug(message)

            if self.SAVE_PASS_IMG:
                self.save_img(self.img,
                              name=self._test_id() + "-pass",
                              format=self.PASS_IMG_FORMAT,
                              border_box=region,
                              msg=message,
                              color=bretina.COLOR_GREEN,
                              put_img=colors,
                              log_level=logging.INFO)

    def assertText(self, region, text,
                   language="eng", msg="", circle=False, bgcolor=None, chars=None, floodfill=False, sliding=False,
                   threshold=1, simchars=None, ligatures=None, ignore_accents=True, singlechar=False,
                   expendable_chars=None, patterns=None):
        """
        Checks the text in the given region.

        :param list region: boundaries of intrested area [left, top, right, bottom]
        :param str text: expected text co compare
        :param str language: language of the string or collection of languages (tuple), use 3-letter ISO codes:
                             https://github.com/tesseract-ocr/tesseract/wiki/Data-Files
        :param str msg: optional assertion message
        :param bool circle: optional flag to tell OCR engine that the text is in circle
        :param bgcolor: background color
        :param str chars: optional limit of the used characters in the OCR
        :param bool floodfill: optional argument to apply flood fill to unify background
        :param bool sliding: optional argument
            - `False` to prohibit sliding text animation recognition
            - `True` to check also sliding text animation, can lead to long process time
        :param float threshold: how many errors is ignored in the diff
        :param list simchars: allowed similar chars in text comparision, e.g. ["1l", "0O"]. Differences in these
                              characters are not taken as differences.
        :param list ligatures: list of char combinations which shall be unified to prevent confusion e.g. [("τπ", "πτ")]
        :param bool ignore_accents: when set to `True`, given and OCR-ed texts are cleared from diacritic, accents,
                                    umlauts, ... before comparision
            (e.g. "příliš žluťoučký kůň" is treated as "prilis zlutoucky kun").
        :param bool singlechar: treat the analysed text as single character (uses special setting of the OCR engine)
        :param list expendable_chars: set of chars which may are allowed to be missing in the text
        :param patterns: single pattern e.g "\d\A\A" or list of patterns, e.g.["\A\d\p\d\d", "\d\A\A"]
        :type patterns: string or list of strings
        """
        combined_img = None
        # remove accents from the expected text
        if ignore_accents:
            text = bretina.remove_accents(text)

        # if only one character is checked, use singlechar option
        if len(text.strip()) == 1:
            singlechar = True

        # build a tuple of languages from the normalized languages
        if language is None:
            language = ['eng']
        elif isinstance(language, str):
            language = [bretina.normalize_lang_name(language)]
        elif isinstance(language, (list, tuple, set, frozenset)):
            language = list(set([bretina.normalize_lang_name(lang) for lang in language]))
        else:
            raise TypeError(f'Param `language` does not support type of {type(language)}, parameter has to be string or list of strings.')

        # combine all given languages
        lang_table = [language] * len(language)
        lang_combinations = itertools.product(*lang_table)
        lang_combinations = sorted(list(set([tuple(sorted(set(i))) for i in lang_combinations])))
        language = ['+'.join(l) for l in lang_combinations]

        # check if tested string is in format which is tested only in one language
        if bretina.LANGUAGE_LIMITED is not None:
            for lang, patterns in bretina.LANGUAGE_LIMITED.items():
                for pattern in patterns:
                    if re.match(pattern, text):
                        if not bretina.normalize_lang_name(lang) in language:
                            self.log.warning(f"Using {lang} instead of {language} because '{text}' is matching {lang}-only pattern.")
                        language = [lang]
                        break

        # crop the region of interest
        roi = bretina.crop(self.img, region, self.SCALE)
        multiline = bretina.text_rows(roi, self.SCALE)[0] > 1

        assert threshold >= 0, f'`threshold` has to be positive integer, {threshold} given'
        threshold = int(threshold)

        # load default simchars and ligatures
        if simchars is None:
            simchars = bretina.CONFUSABLE_CHARACTERS

        if ligatures is None:
            ligatures = bretina.LIGATURE_CHARACTERS

        if expendable_chars is None:
            expendable_chars = bretina.EXPENDABLE_CHARACTERS

        def _get_diffs(img_roi, languages):
            _diff_count = 2**32
            _diffs = ''
            _diff_lang = ''
            _readout = ''

            for lang in languages:
                # try all installed training data
                for tessdata in self._available_tessdata:
                    # get string from image
                    lang_readout = bretina.read_text(img_roi, lang, multiline, circle=circle, bgcolor=bgcolor,
                                                     chars=chars, floodfill=floodfill, singlechar=singlechar,
                                                     tessdata=tessdata, patterns=patterns)

                    # remove accents from the OCR-ed text
                    if ignore_accents:
                        lang_readout = bretina.remove_accents(lang_readout)

                    # check equality of the strings
                    lang_diff_count, lang_diffs = bretina.compare_str(lang_readout, text, simchars=simchars,
                                                                      ligatures=ligatures,
                                                                      expendable_chars=expendable_chars)

                    # find the language with the minimum difference
                    if lang_diff_count < _diff_count:
                        _diff_count = lang_diff_count
                        _diffs = lang_diffs
                        _diff_lang = lang
                        _readout = lang_readout
                        _tessdata = tessdata

                    # no need to check other languages when this one is without an error
                    if _diff_count == 0:
                        break

                if _diff_count == 0:
                    break

            return _diff_count, _diffs, _diff_lang, _readout

        diff_count, diffs, diff_lang, readout = _get_diffs(roi, language)

        # if not equal, for single line text try to use sliding text reader if sliding is not prohibited
        if (diff_count > threshold) and not multiline and sliding:
            # but first verify if the text covers more than 90% of the region
            cnt, regions = bretina.text_cols(roi, self.SCALE, bgcolor=bgcolor)

            if (cnt > 0) and (regions[-1][1] > (roi.shape[1] * 0.9)):
                # gather sliding animation frames
                retries = 0
                period = 0.050  # [s]

                while retries < 3:
                    stitcher = bretina.ImageStitcher(axis='h', bgcolor=bgcolor, cut_off_bg=True)
                    stitched, combined_img = stitcher.add(roi)

                    while not stitched:
                        toc = time.time() + period
                        img = self.camera.acquire_calibrated_image()
                        img = bretina.crop(img, region, self.SCALE)
                        img = self._preprocess(img)
                        stitched, combined_img = stitcher.add(img)
                        now = time.time()

                        if toc > now:
                            time.sleep(toc - now)

                    slide_diff_count, slide_diffs, slide_diff_lang, slide_readout = _get_diffs(combined_img, language)

                    if slide_diff_count > threshold:
                        retries += 1
                        self.log.debug(f'{slide_diff_count} differences detected, retry stitching {retries}. time.')
                    else:
                        break

                # take the diff from the slide only if it is better than without slide
                if slide_diff_count < diff_count:
                    diff_count = slide_diff_count
                    diffs = slide_diffs
                    diff_lang = slide_diff_lang
                    readout = slide_readout
                else:
                    combined_img = None

        if diff_count > threshold:
            message = f"Text [{diff_lang}] '{readout}' != '{text}' (expected) ({diff_count} > {threshold}): {msg}"
            # remove new lines from the given text and put spaces instead (yes, regex could handle this, but nah...)
            message = message.replace(' \n', '\n').replace('\n ', '\n').replace('\n', ' ')
            self.log.log(self.ERROR_LOG_LEVEL, message)

            # show also diffs for short texts
            message += "\n................................\n"
            message += bretina.format_diff(diffs, max_len=self.MAX_STRING_DIFF_LEN)

            self.save_img(self.img,
                          name=self._test_id(),
                          format=self.LOG_IMG_FORMAT,
                          border_box=region,
                          msg=message,
                          color=bretina.COLOR_RED,
                          put_img=combined_img,
                          log_level=self.ERROR_LOG_LEVEL)

            if self.SAVE_SOURCE_IMG:
                self.save_img(self.img,
                              name=self._test_id() + "-src",
                              format=self.SRC_IMG_FORMAT,
                              log_level=logging.INFO)

            self.fail(msg=message)
        # when OK
        else:
            message = f"Text [{diff_lang}] '{readout}' == '{text}' (expected) ({diff_count} <= {threshold})"
            message = message.replace(' \n', '\n').replace('\n ', '\n').replace('\n', ' ')
            self.log.debug(message)

            if self.SAVE_PASS_IMG:
                self.save_img(self.img,
                              name=self._test_id() + "-pass",
                              format=self.PASS_IMG_FORMAT,
                              border_box=region,
                              msg=message,
                              color=bretina.COLOR_GREEN,
                              put_img=combined_img,
                              log_level=logging.INFO)

    def assertImage(self, region, template_name, threshold=None, edges=False, inv=None, bgcolor=None, alpha_color=None,
                    blank=None, template_roi=None, msg=""):
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
        :param bgcolor: specify color which is used to fill transparent areas in png with alpha channel, decided
                        automatically when None
        :param str alpha_color: creates an alpha channel and fills areas with 0% transparency with the desired color
        :param list blank: list of areas which shall be masked
        :param list template_roi: region of interest in the template image [left, top, right, bottom] - only this part
                                  area of the tempalate image will be used for the comparision.
        :param str msg: optional assertion message
        """
        if threshold is None:
            threshold = self.LIMIT_IMAGE_MATCH

        assert threshold >= 0.0, "`threshold` has to be float in range [0, 1]"

        roi = bretina.crop(self.img, region, self.SCALE)
        path = os.path.join(self.template_path, template_name)
        template = cv.imread(path, cv.IMREAD_UNCHANGED)
        template_original = template.copy()

        if template is None:
            message = 'Template file {} is missing! Full path: {}'.format(template_name, path)
            self.log.log(self.ERROR_LOG_LEVEL, message)
            self.fail(message)

        # crop only region-of-interest if specified
        if template_roi is not None:
            if not isinstance(template_roi, (list, tuple)) or (len(template_roi) != 4):
                raise ValueError(f'Argument `template_roi` has to be sequence in format [left, top, right, bottom], `{template_roi}` given')

            template = bretina.crop(template, template_roi, 1.0)
            template_original = bretina.draw_border(template_original, template_roi,
                                                    color=bretina.color('magenta'))

        if alpha_color is not None:
            # alpha_channel
            alpha = bretina.img_to_grayscale(template)

            # create new BGRA img
            template = np.ones([template.shape[0], template.shape[1], 4], dtype=np.uint8)
            color = bretina.color(alpha_color)

            template[:, :, 0:3] = color
            template[:, :, 3] = alpha

        # resize blanked areas
        if blank is not None:
            assert isinstance(blank, (list, tuple, set, frozenset)), '`blank` has to be list'

            # make list if only one area is given
            if len(blank) > 0 and not isinstance(blank[0], (list, tuple, set, frozenset)):
                blank = [blank]

            for i in range(len(blank)):
                for j in range(len(blank[i])):
                    blank[i][j] *= self.SCALE

        # get difference between template and ROI
        template = bretina.resize(template, self.SCALE)
        template_original = bretina.resize(template_original, self.SCALE)
        diff = bretina.img_diff(roi, template, edges=edges, inv=inv, bgcolor=bgcolor, blank=blank)

        # check the diff level
        if diff > threshold:
            message = f"Image '{template_name}' is different ({diff:.3f} > {threshold:.3f}): {msg}"
            self.log.log(self.ERROR_LOG_LEVEL, message)
            self.save_img(self.img,
                          name=self._test_id(),
                          format=self.LOG_IMG_FORMAT,
                          border_box=region,
                          msg=message,
                          color=bretina.COLOR_RED,
                          put_img=template_original,
                          log_level=self.ERROR_LOG_LEVEL)

            if self.SAVE_SOURCE_IMG:
                self.save_img(self.img,
                              name=self._test_id() + "-src",
                              format=self.SRC_IMG_FORMAT,
                              log_level=logging.INFO)

            self.fail(msg=message)
        # diff level is close to the limit, show warning
        elif diff <= threshold and diff >= (threshold * 1.1):
            message = f"Image '{template_name}' difference {diff:.3f} close to limit {threshold:.3f}."
            self.log.warning(message)

            if self.SAVE_PASS_IMG:
                self.save_img(self.img,
                              name=self._test_id() + "-pass",
                              format=self.PASS_IMG_FORMAT,
                              border_box=region,
                              msg=message,
                              color=bretina.COLOR_ORANGE,
                              put_img=template_original,
                              log_level=logging.INFO)
        # when OK
        else:
            message = f"Image '{template_name}' matched ({diff:.5f} <= {threshold:.5f})"
            self.log.debug(message)

            if self.SAVE_PASS_IMG:
                self.save_img(self.img,
                              name=self._test_id() + "-pass",
                              format=self.PASS_IMG_FORMAT,
                              border_box=region,
                              msg=message,
                              color=bretina.COLOR_GREEN,
                              put_img=template_original,
                              log_level=logging.INFO)

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
        :param metric: function to use to calculate the color distance `d = metrics((B, G, R), (B, G, R))`
        :type  metric: callable
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
            self.log.log(self.ERROR_LOG_LEVEL, message)
            self.save_img(self.imgs[position],
                          name=self._test_id(),
                          format=self.LOG_IMG_FORMAT,
                          border_box=region,
                          msg=message,
                          color=bretina.COLOR_RED,
                          log_level=self.ERROR_LOG_LEVEL)

            if self.SAVE_SOURCE_IMG:
                self.save_img(self.imgs[position],
                              name=self._test_id() + "-src",
                              format=self.SRC_IMG_FORMAT,
                              log_level=logging.INFO)

            self.fail(msg=message)
        # when OK
        else:
            message = f"Region '{region}' empty (STD {max(std):.2f} > {threshold:.2f})"
            self.log.debug(message)

            if self.SAVE_PASS_IMG:
                self.save_img(self.imgs[position],
                              name=self._test_id() + "-pass",
                              format=self.PASS_IMG_FORMAT,
                              border_box=region,
                              msg=message,
                              color=bretina.COLOR_GREEN,
                              log_level=logging.INFO)

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

            # check background color is close to the expected one
            if dist > bgcolor_threshold:
                message = f"Region {region} background {bretina.color_str(avgcolor)} != {bretina.color_str(bgcolor)} (expected) ({dist:.2f} > {bgcolor_threshold:.2f}): {msg}"
                self.log.log(self.ERROR_LOG_LEVEL, message)
                self.save_img(self.imgs[0],
                              name=self._test_id(),
                              format=self.LOG_IMG_FORMAT,
                              border_box=region,
                              msg=message,
                              color=bretina.COLOR_RED,
                              log_level=self.ERROR_LOG_LEVEL)

                if self.SAVE_SOURCE_IMG:
                    self.save_img(self.imgs[0],
                                  name=self._test_id() + "-src",
                                  format=self.SRC_IMG_FORMAT,
                                  log_level=logging.INFO)

                self.fail(msg=message)
            # when OK
            else:
                message = f"Background {bretina.color_str(avgcolor)} == {bretina.color_str(bgcolor)} ({dist:.2f} <= {bgcolor_threshold:.2f})."
                self.log.debug(message)

                if self.SAVE_PASS_IMG:
                    self.save_img(self.imgs[0],
                                  name=self._test_id() + "-pass",
                                  format=self.PASS_IMG_FORMAT,
                                  border_box=region,
                                  msg=message,
                                  color=bretina.COLOR_GREEN,
                                  log_level=logging.INFO)

    def assertImageAnimation(self, region, template_name, animation_active, size, threshold=None, bgcolor=None, msg="", split_threshold=64):
        """
        Checks if the image animation is present in the given region.

        :param region: boundaries of intrested area
        :type  region: [left, top, right, bottom]
        :param str template_name: file name of the expected animation sequence image relative to `self.template_path`
        :param bool animation_active:
            - True - animation is expected to be running,
            - False - animation is expected to be freezed
        :param size: size of the one animation frame in the template image.
        :type  size: [width, height]
        :param float threshold: threshold value used in the test for the image, `LIMIT_IMAGE_MATCH` is the default
        :param bgcolor: expected background color (for partially transparent images)
        :param msg: optional assertion message
        :param int split_threshold: ? TODO
        """
        if threshold is None:
            threshold = self.LIMIT_IMAGE_MATCH

        assert threshold >= 0.0, "`threshold` has to be float in range [0, 1]"

        roi = [bretina.crop(img, region, self.SCALE) for img in self.imgs]
        path = os.path.join(self.template_path, template_name)
        template = cv.imread(path)

        if template is None:
            message = f'Template file {template_name} is missing! Full path: {path}'
            self.log.log(self.ERROR_LOG_LEVEL, message)
            self.fail(message)

        diff, animation = bretina.recognize_animation(roi, template, size, self.SCALE, split_threshold=split_threshold)

        template_crop = bretina.crop(template, (0, 0, size[0], size[1]), 1)
        template_crop = bretina.resize(template_crop, self.SCALE)
        position = np.argmax([bretina.img_diff(img, template_crop, bgcolor=bgcolor) for img in roi])

        # check difference with the animation
        if diff > threshold:
            message = f"Animation '{template_name}' not matched {diff:.2f} > {threshold:.2f}: {msg}"
            self.log.log(self.ERROR_LOG_LEVEL, message)
            self.save_img(self.imgs[0],
                          name=self._test_id(),
                          format=self.LOG_IMG_FORMAT,
                          border_box=region,
                          msg=message,
                          color=bretina.COLOR_RED,
                          put_img=template,
                          log_level=self.ERROR_LOG_LEVEL)

            if self.SAVE_SOURCE_IMG:
                self.save_img(self.imgs[0],
                              name=self._test_id() + "-src",
                              format=self.SRC_IMG_FORMAT,
                              log_level=logging.INFO)

            self.fail(msg=message)
        # show warning if difference is close to the threshold
        elif diff >= (threshold * 0.95):
            message = f"Animation '{template_name}' matched but close to limit ({diff:.2f} ~ {threshold:.2f})."
            self.log.warning(message)

            if self.SAVE_PASS_IMG:
                self.save_img(self.imgs[0],
                              name=self._test_id() + "-pass",
                              format=self.PASS_IMG_FORMAT,
                              border_box=region,
                              msg=message,
                              color=bretina.COLOR_ORANGE,
                              put_img=template,
                              log_level=logging.INFO)
        # when OK
        else:
            message = f"Animation '{template_name}' matched ({diff:.2f} <= {threshold:.2f})"
            self.log.debug(message)

            if self.SAVE_PASS_IMG:
                self.save_img(self.imgs[0],
                              name=self._test_id() + "-pass",
                              format=self.PASS_IMG_FORMAT,
                              border_box=region,
                              msg=message,
                              color=bretina.COLOR_GREEN,
                              put_img=template,
                              log_level=logging.INFO)

        if animation != animation_active:
            message = f"Animation '{template_name}' activity {animation} != {animation_active} (expected): {msg}"
            self.log.log(self.ERROR_LOG_LEVEL, message)
            self.save_img(self.imgs[0],
                          name=self._test_id(),
                          format=self.LOG_IMG_FORMAT,
                          border_box=region,
                          msg=message,
                          color=bretina.COLOR_RED,
                          put_img=template,
                          log_level=self.ERROR_LOG_LEVEL)

            if self.SAVE_SOURCE_IMG:
                self.save_img(self.imgs[0],
                              name=self._test_id() + "-src",
                              format=self.SRC_IMG_FORMAT,
                              log_level=logging.INFO)

            self.fail(msg=message)

    def assertHist(self, region, template_name, threshold=None, bgcolor=None, blank=None, msg=""):
        """
        Checks if image histogram is similar to compared one.

        :param region: boundaries of intrested area
        :type  region: [left, top, right, bottom]
        :param str template_name: file name of the expected image relative to `self.template_path`
        :param float threshold: threshold value used in the test for the image, `LIMIT_IMAGE_MATCH` is the default
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
            self.log.log(self.ERROR_LOG_LEVEL, message)
            self.fail(message)

        # resize blanked areas
        if blank is not None:
            assert isinstance(blank, (list, tuple, set, frozenset)), '`blank` has to be list'

            # make list if only one area is given
            if len(blank) > 0 and not isinstance(blank[0], (list, tuple, set, frozenset)):
                blank = [blank]

            for i in range(len(blank)):
                for j in range(len(blank[i])):
                    blank[i][j] *= self.SCALE

        # get difference between template and ROI
        template = bretina.resize(template, self.SCALE)
        diff = bretina.img_hist_diff(roi, template, bgcolor=bgcolor, blank=blank)

        # check the diff level
        if diff > threshold:
            message = f"Image '{template_name}' have different histogram ({diff:.3f} > {threshold:.3f}): {msg}"
            self.log.log(self.ERROR_LOG_LEVEL, message)
            self.save_img(self.img,
                          name=self._test_id(),
                          format=self.LOG_IMG_FORMAT,
                          border_box=region,
                          msg=message,
                          color=bretina.COLOR_RED,
                          put_img=template,
                          log_level=self.ERROR_LOG_LEVEL)

            if self.SAVE_SOURCE_IMG:
                self.save_img(self.img,
                              name=self._test_id() + "-src",
                              format=self.SRC_IMG_FORMAT,
                              log_level=logging.INFO)

            self.fail(msg=message)
        # diff level is close to the limit, show warning
        elif diff <= threshold and diff >= (threshold * 1.1):
            message = f"Image '{template_name}' histogram difference {diff:.3f} close to limit {threshold:.3f}."
            self.log.warning(message)

            if self.SAVE_PASS_IMG:
                self.save_img(self.img,
                              name=self._test_id() + "-pass",
                              format=self.PASS_IMG_FORMAT,
                              border_box=region,
                              msg=message,
                              color=bretina.COLOR_ORANGE,
                              put_img=template,
                              log_level=logging.INFO)
        # when OK
        else:
            message = f"Image '{template_name}' histogram matched ({diff:.5f} <= {threshold:.5f})"
            self.log.debug(message)

            if self.SAVE_PASS_IMG:
                self.save_img(self.img,
                              name=self._test_id() + "-pass",
                              format=self.PASS_IMG_FORMAT,
                              border_box=region,
                              msg=message,
                              color=bretina.COLOR_GREEN,
                              put_img=template,
                              log_level=logging.INFO)

    def assertCurve(self, region, expected_points, scale=None, threshold=None, blank=None, suppress_noise=False, max_line_gab=None,
                    transform=True, tolerance=None, msg=""):
        """
        Check if found polyline is the same as expected one. Checks if all found coordinates of polyline
        match with expected one. The units of expected points can be of a different type (not only pixels).

        :param region: boundaries of intrested area
        :type  region: [left, top, right, bottom]
        :param expected_points: coordinates of expected polyline in format [[x1, y1], [x2, y2],...]
        :type expected_points : nested list
        :param scale: target scaling
        :type  scale: float
        :param blank: areas which shall be masked e.g.[[10, 20, 30, 40], [], ...]
        :type  blank: nested list
        :param suppress_noise: removes noise around the polyline, (True - remove noise, False - don`t remove noise)
        :type suppress_noise : bool
        :param max_line_gab: maximum allowed gap between points on the same line to link them
        :type max_line_gab : int
        :param transform: flag which indicates if found points should be transform to different units or not
        :type transform : bool
        :param tolerance: set tolerance zone (point +-tolerance) to evaluate if found point is the same as expected one
        :type tolerance : int
        :param msg: assertion message
        :type msg : str
        """
        # checking of type and format of expected points
        assert isinstance(expected_points, list), f"`expected _points` should be type of `list`, `{type(expected_points)}` given"

        for coodinates in expected_points:
            assert isinstance(coodinates, list), f"each part of the `expected_points` has to be type of `list`"

        if scale is None:
            scale = self.SCALE

        if threshold is None:
            threshold = self.LIMIT_POINTS_FINDING

        assert scale > 0, "`scale` has to be positive float"
        assert tolerance >= 0, "tolerance must be positive"
        assert threshold >= 0, "`threshold` has to be a positive float"
        assert isinstance(transform, bool), f"`transform` has to be type of `bool`, `{type(transform)}` given"

        # find all coordinates of polyline in selected area
        paths = get_polyline_coordinates(self.img, region, scale=scale, threshold=threshold,
                                         blank=blank, suppress_noise=suppress_noise, max_line_gab=max_line_gab)
        merged_paths = [coord for path in paths for coord in path]

        merged_paths = sorted(merged_paths, key=lambda x: x[0])
        expected_points = sorted(expected_points, key=lambda x: x[0])

        # save all found x-coordinates and y-coordinates to separete lists
        points_px = []
        points_real = []

        for index, item in enumerate([merged_paths, expected_points]):
            for pos in [0, 1]:
                points = [coord[pos] for coord in item]
                if index == 0:
                    points_px.append(points)
                else:
                    points_real.append(points)

        # verifies if found points match with expected ones
        for index, coord_px in enumerate(points_px):
            points_real_copy = points_real[index].copy()

            # linear transformation of found points is executed
            if transform or scale > 1:
                obtained_real = linear_transform_points(coord_px, points_real[index], index)
            else:
                obtained_real = points_real[index].copy()

            ok =  True  # flag which indicates if obtained points are the same as expected point
            not_found = 0  # number of point which were checked but not found between expected points
            diff = len(obtained_real) - len(points_real[index])  # difference between number of found points and expected points

            # determines how many obtained points are different from expected one
            for pos, point in enumerate(obtained_real):
                if points_real_copy:
                    for pos_, expected_point in enumerate(points_real_copy):
                        range_ = list(range(expected_point - tolerance, expected_point + tolerance + 1))

                        if point in range_:
                            points_real_copy.pop(pos_)
                            break
                    else:
                        not_found += 1
                else:
                    not_found += len(obtained_real) - pos

            if (diff >= 0 and not_found > diff) or (diff < 0 and not_found > 0):
                ok = False

            # draw found polyline to new img
            found_img = np.zeros(self.img.shape)
            for line in paths:
                for j in range(len(line)-1):
                    found_img = cv.line(found_img, tuple(line[j]), tuple(line[j+1]), 255, 6)

            # evaluation of the obtained result
            if not ok:
                message = f"Obtained points of found polyline does not match with expected one.\nObtained points: {obtained_real} Expected points: {points_real[index]}: {msg}"
                self.log.log(self.ERROR_LOG_LEVEL, msg)
                self.save_img(self.img,
                              name=self._test_id(),
                              format=self.LOG_IMG_FORMAT,
                              border_box=region,
                              msg=msg,
                              color=bretina.COLOR_RED,
                              put_img=found_img,
                              log_level=self.ERROR_LOG_LEVEL)

                if self.SAVE_SOURCE_IMG:
                    self.save_img(self.img,
                                  name=self._test_id() + "-src",
                                  format=self.SRC_IMG_FORMAT,
                                  log_level=logging.INFO)

                self.fail(msg=message)
            else:
                message = f"Obtained points of found polyline match with expected one.\nObtained points: {obtained_real} Expected points: {points_real[index]}"
                self.log.info(message)

                if self.SAVE_PASS_IMG:
                    self.save_img(self.img,
                                  name=self._test_id() + "-pass",
                                  format=self.PASS_IMG_FORMAT,
                                  border_box=region,
                                  msg=msg,
                                  color=bretina.COLOR_GREEN,
                                  put_img=found_img,
                                  log_level=logging.INFO)
