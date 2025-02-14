# Bretina project changelog

## 0.7.5 (2023-08-23)

- fix deprecation of `multiline_textbbox` method in pillow 10.0 (refs #7535).

## 0.7.4 (2023-08-07)

- handle deprecation of `font.getsize` method in Pillow 10.0 (refs #7535).

## 0.7.3 (2022-05-26)

- added image saving if ZeroDivisionError occurs during linear transformation in curve assertion (refs #5933).

## 0.7.2 (2022-05-09)

- draw log image text in white and with color border (refs #5579),
- optimized searching for the tesseract.exe.

## 0.7.1 (2022-03-10)

- reduced minimum number of non-background pixels for the color assertion.
- allow to work even when Tesseract.exe is not located (to be able to run on
  Linux where such a file does not exist).
- fixed drawing image overflowing the canvas image.
- font-size for message text is selected according to the resolution of the
  saving image.

## 0.7.0 (2022-03-08)

- split Serbian Latin and Cyrillic languages into two codes - do not combine them implicitly.
- allow implicitly differences in hyphens.
- relocated message in artefact image bellow the source image (refs #5579).
- reduce size and quality of the image artefact (refs #5579).
- allow difference in dot and hyphen character.
- reworked color assertion (refs #5098).

## 0.6.10 (2022-02-17)

- ignore differences in non breaking hyphens during check of the string equality (refs #5395).

## 0.6.9 (2022-02-15)

- fix logging level during image record saving
- ignore differences in hyphens during check of the string equality (refs #5395).

## 0.6.8 (2022-01-04)

- prevent collision in generated extra file names (refs #5307).
- prevent invalid log level value during image logging.

## 0.6.7 (2022-01-04)

- modified OCR text comparision to allow differences in the similar lower-upper case characters.

## 0.6.6 (2021-11-24)

- replace invalid characters in the test ID (refs #5307).

## 0.6.5 (2021-11-15)

- reduced max length of the test ID to 160 characters.

## 0.6.4 (2021-11-09)

- use sub-test ID as the image artefact name if possible (refs #5230),
- add image artefact to the extra test and sub-test data.

## 0.6.3 (2021-09-28)

- increased limits of the image stitcher, added fix period of stitching.

## 0.6.2 (2021-09-13)

- added retry during assertion of the stitched text.

## 0.6.1 (2021-08-25)

- added padding parameter to the cols detection (refs #4873),
- added property to access number of added frames without a change in the stitched image,
- reduced image cute-off part during merging to 5%.

## 0.6.0 (2021-06-30)

- added Zhang-Suen thinning algorithm to Polyline (refs #3691),
- added Serbian and Montenegrin support (refs #4409).

## 0.5.0 (2021-03-16)

- fixed wording in animation assertion log,
- reworked sliding text animation processing, created `ImageStitcher` class replacing `SlidingTextReader` (refs #2781),
- removed `merge_images` function,
- renamed `color` parameter to `channels` of the `_blank_image` function.

## 0.4.0 (2021-03-04)

- added support for Tesseract OCR text patterns (refs #4074),
- added support to use only part of the template image in the image assertion,
- removed unused `TEMPLATE_PATH` definition,
- allowed to use pre-release in the version name.

## 0.3.0 (2021-02-22)

- improved execution time of the text assertion.
- added support for expendable characters in the text assertion.

## 0.2.2 (2021-02-18)

- fixed listing of the available `tessdata` datasets.

## 0.2.1 (2021-02-05)

- removed hash of the referenced htmllogging package.

## 0.2.0 (2021-02-03)

- modified text OCR to try all available Tesseract trained datasets,
- changed the default Tesseract installation directory to the more commonly used,
- during the text assertion, all given languages are combined and all combinations are tested - the best match is used
  as an result,
- removed text assertion `langchars` and `deflang` options.

## 0.1.0 (2021-01-22)

- added function to split image into given number of columns (refs #3964).

## 0.0.9 (2021-01-18)

- added abilities of the HTML log filtering based on the severity level.

## 0.0.8 (2020-11-27)

- added support for multiple languages in the OCR text assertion (refs #3798)

## 0.0.7 (2020-09-11)

- fixed typo of alpha to alpha_color in VisualTestCase.assertImage().

## 0.0.6 (2020-09-10)

- added support to assert alpha-only images.

## 0.0.5 (2020-09-04)

- added support for TRACE and FAIL log levels,
- improved documentation.

## 0.0.4 (2019-03-03)

- added `assertNotColor` function,
- added Region Of Interest parameter to region color detection.

## 0.0.3 (2019-10-19)

- updated image logging to prevent logging of images to the normal text logs.

## 0.0.2 (2020-02-18)

- changed string compare strategy to absolute diff instead of relative ratio (refs #2640),
- added support for ignoring duplicated chars in string compare,
- added histogram assert function (refs #2630)
- added support for multiple languages in the tesseract parameter
- added sanitation of new lines in text assertion message,
- added default delay to capture function,
- added single char option to use tesseract single char page segmentation mode,
- changed default limit value for cols detection,
- improved sliding text recognition (refs #2781),
- added functionality to specify language of the OCR text assert with the regular expression (refs #2852),
- added support for string diffs on long texts,
- added image merge function (refs #2865),
- changed logging to use the same log folder during the entire test run,
- added function for language name normalization.

## 0.0.1 (2019-12-16)

- first version of the library, included all major functions.
