# Bretina project changelog

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
