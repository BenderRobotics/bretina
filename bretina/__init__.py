"""
    bretina
    =======

    Bender Robotics module for visual based testing support.

    :copyright: 2019 Bender Robotics
"""

__version__ = '0.0.1'
__all__ = ['VisualTestCase', 'SlidingTextReader', 'HtmlHandler', 'ImageRecord',
           '__version__']

import numpy as np
import cv2 as cv
import os
import time
import math
import difflib
import logging
import itertools
import pytesseract
import unicodedata

from bretina.visualtestcase import VisualTestCase
from bretina.slidingtextreader import SlidingTextReader
from bretina.htmllogging import HtmlHandler, ImageRecord

#: List of ligatures, these char sequences are unified.
#: E.g. greek word 'δυσλειτουργία' (malfunction) contains sequence 'ιτ' which will
#: be replaced with 'π' and will be treated as equal to the word 'δυσλεπουργία' (dyslexia).
#: Motivation fro this replacement is that these characters can look similar on the display
#: and therefore can not be recognized correctly
LIGATURE_CHARACTERS = None

#: List of confusable characters, when OCR-ed and expected text differs in the chars
#: in chars which are listed bellow, this difference is not considred as difference.
#: E.g with "ćčc" in CONFUSABLE_CHARACTERS strings "čep", "cep" and "ćep" will be treated
#: as equal.
CONFUSABLE_CHARACTERS = None

#: Default path to the Tesseract OCR engine installation
TESSERACT_PATH = 'C:\\Tesseract-OCR\\'

#: Standart color definitions in BGR
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_CYAN = (255, 255, 0)
COLOR_MAGENTA = (255, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_ORANGE = (0, 128, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GRAY = (127, 127, 127)
COLOR_WHITE = (255, 255, 255)

#: Map of HTML color names to hex codes
COLORS = {
    'aliceblue':            '#F0F8FF',
    'antiquewhite':         '#FAEBD7',
    'aqua':                 '#00FFFF',
    'aquamarine':           '#7FFFD4',
    'azure':                '#F0FFFF',
    'beige':                '#F5F5DC',
    'bisque':               '#FFE4C4',
    'black':                '#000000',
    'blanchedalmond':       '#FFEBCD',
    'blue':                 '#0000FF',
    'blueviolet':           '#8A2BE2',
    'brown':                '#A52A2A',
    'burlywood':            '#DEB887',
    'cadetblue':            '#5F9EA0',
    'chartreuse':           '#7FFF00',
    'chocolate':            '#D2691E',
    'coral':                '#FF7F50',
    'cornflowerblue':       '#6495ED',
    'cornsilk':             '#FFF8DC',
    'crimson':              '#DC143C',
    'cyan':                 '#00FFFF',
    'darkblue':             '#00008B',
    'darkcyan':             '#008B8B',
    'darkgoldenrod':        '#B8860B',
    'darkgray':             '#A9A9A9',
    'darkgrey':             '#A9A9A9',
    'darkgreen':            '#006400',
    'darkkhaki':            '#BDB76B',
    'darkmagenta':          '#8B008B',
    'darkolivegreen':       '#556B2F',
    'darkorange':           '#FF8C00',
    'darkorchid':           '#9932CC',
    'darkred':              '#8B0000',
    'darksalmon':           '#E9967A',
    'darkseagreen':         '#8FBC8F',
    'darkslateblue':        '#483D8B',
    'darkslategray':        '#2F4F4F',
    'darkslategrey':        '#2F4F4F',
    'darkturquoise':        '#00CED1',
    'darkviolet':           '#9400D3',
    'deeppink':             '#FF1493',
    'deepskyblue':          '#00BFFF',
    'dimgray':              '#696969',
    'dimgrey':              '#696969',
    'dodgerblue':           '#1E90FF',
    'firebrick':            '#B22222',
    'floralwhite':          '#FFFAF0',
    'forestgreen':          '#228B22',
    'fuchsia':              '#FF00FF',
    'gainsboro':            '#DCDCDC',
    'ghostwhite':           '#F8F8FF',
    'gold':                 '#FFD700',
    'goldenrod':            '#DAA520',
    'gray':                 '#808080',
    'grey':                 '#808080',
    'green':                '#008000',
    'greenyellow':          '#ADFF2F',
    'honeydew':             '#F0FFF0',
    'hotpink':              '#FF69B4',
    'indianred ':           '#CD5C5C',
    'indigo ':              '#4B0082',
    'ivory':                '#FFFFF0',
    'khaki':                '#F0E68C',
    'lavender':             '#E6E6FA',
    'lavenderblush  ':      '#FFF0F5',
    'lawngreen':            '#7CFC00',
    'lemonchiffon':         '#FFFACD',
    'lightblue':            '#ADD8E6',
    'lightcoral':           '#F08080',
    'lightcyan  ':          '#E0FFFF',
    'lightgoldenrodyellow': '#FAFAD2',
    'lightgray':            '#D3D3D3',
    'lightgrey':            '#D3D3D3',
    'lightgreen':           '#90EE90',
    'lightpink':            '#FFB6C1',
    'lightsalmon':          '#FFA07A',
    'lightseagreen':        '#20B2AA',
    'lightskyblue':         '#87CEFA',
    'lightslategray':       '#778899',
    'lightslategrey':       '#778899',
    'lightsteelblue':       '#B0C4DE',
    'lightyellow':          '#FFFFE0',
    'lime':                 '#00FF00',
    'limegreen':            '#32CD32',
    'linen':                '#FAF0E6',
    'magenta':              '#FF00FF',
    'maroon':               '#800000',
    'mediumaquamarine':     '#66CDAA',
    'mediumblue':           '#0000CD',
    'mediumorchid':         '#BA55D3',
    'mediumpurple':         '#9370DB',
    'mediumseagreen':       '#3CB371',
    'mediumslateblue':      '#7B68EE',
    'mediumspringgreen':    '#00FA9A',
    'mediumturquoise':      '#48D1CC',
    'mediumvioletred':      '#C71585',
    'midnightblue':         '#191970',
    'mintcream':            '#F5FFFA',
    'mistyrose':            '#FFE4E1',
    'moccasin':             '#FFE4B5',
    'navajowhite':          '#FFDEAD',
    'navy':                 '#000080',
    'oldlace':              '#FDF5E6',
    'olive':                '#808000',
    'olivedrab':            '#6B8E23',
    'orange':               '#FFA500',
    'orangered':            '#FF4500',
    'orchid':               '#DA70D6',
    'palegoldenrod':        '#EEE8AA',
    'palegreen':            '#98FB98',
    'paleturquoise':        '#AFEEEE',
    'palevioletred':        '#DB7093',
    'papayawhip':           '#FFEFD5',
    'peachpuff':            '#FFDAB9',
    'peru':                 '#CD853F',
    'pink':                 '#FFC0CB',
    'plum':                 '#DDA0DD',
    'powderblue':           '#B0E0E6',
    'purple':               '#800080',
    'rebeccapurple':        '#663399',
    'red':                  '#FF0000',
    'rosybrown':            '#BC8F8F',
    'royalblue':            '#4169E1',
    'saddlebrown':          '#8B4513',
    'salmon':               '#FA8072',
    'sandybrown':           '#F4A460',
    'seagreen':             '#2E8B57',
    'seashell':             '#FFF5EE',
    'sienna':               '#A0522D',
    'silver':               '#C0C0C0',
    'skyblue':              '#87CEEB',
    'slateblue':            '#6A5ACD',
    'slategray':            '#708090',
    'slategrey':            '#708090',
    'snow':                 '#FFFAFA',
    'springgreen':          '#00FF7F',
    'steelblue':            '#4682B4',
    'tan':                  '#D2B48C',
    'teal':                 '#008080',
    'thistle':              '#D8BFD8',
    'tomato':               '#FF6347',
    'turquoise':            '#40E0D0',
    'violet':               '#EE82EE',
    'wheat':                '#F5DEB3',
    'white':                '#FFFFFF',
    'whitesmoke':           '#F5F5F5',
    'yellow':               '#FFFF00',
    'yellowgreen':          '#9ACD32',
}


def dominant_colors(img, n=3):
    """
    Returns list of dominant colors in the image ordered according to its occurency (major first),
    internally performs k-means clustering for the segmentation

    :param array_like img: source image
    :param int n: number of colors in the output pallet
    :return: list of (B, G, R) color tuples
    :rtype: list
    """
    if len(img.shape) == 3 and img.shape[2] >= 3:
        pixels = np.float32(img[:, :, :3].reshape(-1, 3))
    else:
        pixels = np.float32(img.reshape(-1))

    # k-means clustering
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv.kmeans(pixels, n, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    indexes = np.argsort(counts)[::-1]
    return palette[indexes]


def dominant_color(img, n=3):
    """
    Gets the most dominant color in the image using `dominant_colors` function.

    :param img: source image
    :param n: number of colors in segmentation
    :return: (B, G, R) color tuple
    """
    return dominant_colors(img, n)[0]


def active_color(img, bgcolor=None):
    """
    Gets the most dominant color which is not the background color.

    :param array_like img: source image
    :param bgcolor: color of the image background, recognized automatically if `None` or not set
    :return: (B, G, R) color tuple
    :rtype: tuple
    """
    colors = dominant_colors(img, 3)

    # if background color is not specified, determine background from the outline border
    if bgcolor is None:
        bgcolor = background_color(img)

    # get index of the bg in pallet as minimum of distance, active color is 0 if
    # it is not background (major color)
    bg_index = np.argmin([rgb_distance(bgcolor, c) for c in colors])
    color_index = 1 if bg_index == 0 else 0
    return colors[color_index]


def mean_color(img):
    """
    Returns mean value of pixels colors.

    :param img: source image
    :return: (B, G, R) color tuple
    :rtype: tuple
    """
    channels = img.shape[2] if len(img.shape) == 3 else 1
    pixels = np.float32(img.reshape(-1, channels))
    return np.mean(pixels, axis=0)


def background_color(img):
    """
    Mean color from the 2-pixel width border.

    :param img: source image
    :return: mean color of the image border
    :rtype: tuple
    """
    colors = 3 if (len(img.shape) == 3 and img.shape[2] == 3) else 1
    # take pixels from top, bottom, left and right border lines
    pixels = np.concatenate((np.float32(img[0:2, :].reshape(-1, colors)),
                             np.float32(img[-3:-1, :].reshape(-1, colors)),
                             np.float32(img[:, 0:2].reshape(-1, colors)),
                             np.float32(img[:, -3:-1].reshape(-1, colors))))
    return np.mean(pixels, axis=0)


def background_lightness(img):
    """
    Lightness of the background color.

    Calculates background color with ``background_color`` function and returns mean value
    of R, G and B.

    :param img: source image
    :return: lightness of the background color
    :rtype: int
    """
    bgcolor = background_color(img)
    return int(np.around(np.mean(bgcolor)))


def color_std(img):
    """
    Get standart deviation of the color information in the given image.

    :param img: source image
    :return: standart deviation of the B, G, R color channels
    :rtype: tuple
    """
    pixels = np.float32(img.reshape(-1, 3))
    return np.std(pixels, axis=0)


def lightness_std(img):
    """
    Get standart deviation of the given image lightness information.

    :param img: source image
    :return: standart deviation of the lightness in the given image
    :rtype: float
    """
    gray = img_to_grayscale(img)
    pixels = np.float32(gray.reshape(-1, 1))
    return np.std(pixels)


def rgb_distance(color_a, color_b):
    """
    Gets distance metric of two colors as mean absolute value of differences in R, G, B channels.

    :math:`distance = (|R_1 - R_2| + |G_1 - G_2| + |B_1 - B_2|)/3`

    :param color_a: string or tuple representation of the color A
    :param color_b: string or tuple representation of the color B
    :return: mean distance in RGB
    :rtype: float
    """
    a = [float(_) for _ in color(color_a)]
    b = [float(_) for _ in color(color_b)]

    return (np.absolute(a[0] - b[0]) +
            np.absolute(a[1] - b[1]) +
            np.absolute(a[2] - b[2])) / 3.0


def rgb_rms_distance(color_a, color_b):
    """
    Gets distance metric of two colors as root mean square of differences in R, G, B channels.

    :math:`distance = \sqrt{(R_1 - R_2)^2 + (G_1 - G_2)^2 + (B_1 - B_2)^2}`

    :param color_a: string or tuple representation of the color A
    :param color_b: string or tuple representation of the color B
    :return: mean distance in RGB
    :rtype: float
    """
    a = [float(_) for _ in color(color_a)]
    b = [float(_) for _ in color(color_b)]

    return np.sqrt(((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2) / 3.0)


def hue_distance(color_a, color_b):
    """
    Gets distance metric of two colors in Hue channel.

    :math:`distance = |H_1 - H_2|`

    :param color_a: string or tuple representation of the color A
    :param color_b: string or tuple representation of the color B
    :return: distance in the Hue channel (note that Hue range is 0-180 in cv)
    :rtype: int
    """
    # make two 1px size images of given colors to have color transformation function available
    img_a = np.zeros((1, 1, 3), np.uint8)
    img_a[0, 0] = color(color_a)

    img_b = np.zeros((1, 1, 3), np.uint8)
    img_b[0, 0] = color(color_b)

    a = cv.cvtColor(img_a, cv.COLOR_BGR2HSV)[0, 0]
    b = cv.cvtColor(img_b, cv.COLOR_BGR2HSV)[0, 0]
    a = [float(_) for _ in a]
    b = [float(_) for _ in b]

    d = np.absolute(a[0] - b[0])

    # because 180 is same as 0 degree, return 180-d to have shortest angular distance
    if d > 90:
        return 180 - d
    else:
        return d


def lightness_distance(color_a, color_b):
    """
    Gets two colors distance metric in lightness (L-channel in LAB color space).

    :math:`distance = |L_1 - L_2|`

    :param color_a: string or tuple representation of the color A
    :param color_b: string or tuple representation of the color B
    :return: distance in the Lightness channel (based on LAB color space)
    :rtype: int
    """
    img_a = np.zeros((1, 1, 3), np.uint8)
    img_a[0, 0] = color(color_a)

    img_b = np.zeros((1, 1, 3), np.uint8)
    img_b[0, 0] = color(color_b)

    a = cv.cvtColor(img_a, cv.COLOR_BGR2LAB)[0, 0]
    b = cv.cvtColor(img_b, cv.COLOR_BGR2LAB)[0, 0]
    a = [float(_) for _ in a]
    b = [float(_) for _ in b]

    return np.absolute(a[0] - b[0])


def lab_distance(color_a, color_b):
    """
    Gets distance metric in LAB color space based on CIE76 formula.

    :math:`distance = \sqrt{(L_1 - L_2)^2 + (A_1 - A_2)^2 + (B_1 - B_2)^2}`

    :param color_a: string or tuple representation of the color A
    :param color_b: string or tuple representation of the color B
    :return: distance in the LAB color space (sqrt{dL^2 + dA^2 + dB^2})
    :rtype: int
    """
    img_a = np.zeros((1, 1, 3), np.uint8)
    img_a[0, 0] = color(color_a)

    img_b = np.zeros((1, 1, 3), np.uint8)
    img_b[0, 0] = color(color_b)

    a = cv.cvtColor(img_a, cv.COLOR_BGR2LAB)[0, 0]
    b = cv.cvtColor(img_b, cv.COLOR_BGR2LAB)[0, 0]
    a = [float(_) for _ in a]
    b = [float(_) for _ in b]

    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)


def ab_distance(color_a, color_b):
    """
    Gets distance metric in LAB color space as distance in A-B plane.

    :math:`distance = \sqrt{(A_1 - A_2)^2 + (B_1 - B_2)^2}`

    :param color_a: string or tuple representation of the color A
    :param color_b: string or tuple representation of the color B
    :return: distance in the LAB color space (sqrt{dA^2 + dB^2})
    :rtype: int
    """
    img_a = np.zeros((1, 1, 3), np.uint8)
    img_a[0, 0] = color(color_a)

    img_b = np.zeros((1, 1, 3), np.uint8)
    img_b[0, 0] = color(color_b)

    a = cv.cvtColor(img_a, cv.COLOR_BGR2LAB)[0, 0]
    b = cv.cvtColor(img_b, cv.COLOR_BGR2LAB)[0, 0]
    a = [float(_) for _ in a]
    b = [float(_) for _ in b]

    return np.sqrt((a[1] - b[1])**2 + (a[2] - b[2])**2)


def color(color):
    """
    Converts hex string color '#RRGGBB' to tuple representation (B, G, R).

    :param color: #RRGGBB color string or HTML color name (black) or (B, G, R) tuple
    :type color: str
    :return: (B, G, R) tuple
    :rtype: tuple
    :raises: ValueError -- when the given color is not in recognized format
    """
    if type(color) == str:
        color = color.lower().strip()
        # take color code from map if color is a keyword
        if color in COLORS:
            color = COLORS[color]

        # make long hex from short hex (#FFF -> #FFFFFF)
        if color[0] == '#' and len(color) == 4:
            color = '#{r}{r}{g}{g}{b}{b}'.format(r=color[1], g=color[2], b=color[3])

        # color should be a valid hex string, otherwise raise error
        if color[0] == '#' and len(color) == 7:
            # convert from hex color representation
            h = color.lstrip('#')
            return tuple(int(h[i:i+2], 16) for i in (4, 2, 0))
    elif isinstance(color, (int, float)):
        return (int(color), int(color), int(color))
    elif len(color) == 1:
        return (color[0], color[0], color[0])
    elif len(color) == 3:
        return tuple(color)

    raise ValueError('{} not recognized as a valid color definition.'.format(repr(color)))


def color_str(color):
    """
    Converts color from (B, G, R) tuple to ``#RRGGBB`` string representation.

    :param tuple color: (B, G, R) sequence
    :return: string representation in hex code
    :rtype: str
    """
    if type(color) == str:
        return color
    else:
        return '#{r:02x}{g:02x}{b:02x}'.format(r=int(color[2]),
                                               g=int(color[1]),
                                               b=int(color[0]))


def draw_border(img, box, scale=1, color=COLOR_RED, padding=0, thickness=1):
    """
    Draws rectangle around specified region.

    :param array_like img: cv image
    :param tuple box: border box coordinates (left, top, right, bottom)
    :param float scale: scaling factor
    :param tuple color: color of the border
    :param int padding: additional border padding
    :param int thickness: thickness of the line
    :return: copy of the given image with the border
    :rtype: array_like
    """
    figure = img.copy()

    max_x = figure.shape[1] - 1
    max_y = figure.shape[0] - 1
    start_x = np.clip(int(round(box[0] * scale - padding)), 0, max_x)
    start_y = np.clip(int(round(box[1] * scale - padding)), 0, max_y)
    end_x = np.clip(int(round(box[2] * scale + padding)), 0, max_x)
    end_y = np.clip(int(round(box[3] * scale + padding)), 0, max_y)

    return cv.rectangle(figure, (start_x, start_y), (end_x, end_y), color, thickness=thickness)


def img_to_grayscale(img):
    """
    Converts image to gray-scale.

    :param array_like img: cv image
    :return: image converted to grayscale
    """
    if len(img.shape) == 3:
        if img.shape[2] == 1:
            return img
        elif img.shape[2] >= 3:
            return cv.cvtColor(img[:, :, :3], cv.COLOR_BGR2GRAY)
        else:
            raise Exception(f"Unsupported shape of image {img.shape}")
    else:
        return img


def text_rows(img, scale, bgcolor=None, min_height=10, limit=0.025):
    """
    Gets number of text rows in the given image.

    :param array_like img: image to process
    :param float scale: allows to optimize for different resolution, scale=1 is for font size = 16px.
    :param tuple bgcolor: background color (optional). If not set, the background color is detected automatically.
    :param int min_height: minimum height of row in pixels in original image (is multipled by scale), rows with less pixels are not detected.
    :param float limit: line coverage with pixels of text used for the row detection. Set to lower value for higher sensitivity (0.05 means that 5% of row has to be text pixels)
    :return: (count, regions)
        - count - number of detected text lines
        - regions - tuple of regions where the text rows are detected, each region is represented with tuple (y_from, y_to)
    :rtype: Tuple
    """
    assert img is not None

    min_pixels = img.shape[1] * limit * 255                 # defines how many white pixel in row is minimum for row detection (relatively to the image width)
    kernel_dim = int(2*scale - 1)
    kernel = np.ones((kernel_dim, kernel_dim), np.uint8)    # kernel for dilatation/erosion operations

    if bgcolor is None:
        bg_light = background_lightness(img)
    else:
        bg_light = np.mean(color(bgcolor))

    img = img_to_grayscale(img)
    # thresholding on the image, if image is with dark background, use inverted to have white values in the letters
    ret, thresh = cv.threshold(img, 127, 255, (cv.THRESH_BINARY if bg_light < 128 else cv.THRESH_BINARY_INV) + cv.THRESH_OTSU)
    # apply opening (erosion followed by dilation) to remove pepper and salt artifacts
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    # get sum of pixels in rows and make 0/1 thresholding based on minimum pixel count
    row_sum = np.sum(opening, axis=1, dtype=np.int32)
    row_sum = np.where(row_sum < min_pixels, 0, 1)
    # put 0 at the beginning and end to eliminate option that the letters starts right at the top
    row_sum = np.append([0], row_sum)
    row_sum = np.append(row_sum, [0])
    # get count of rows as the number of 0->1 transitions
    regions = []
    row_start = 0

    for i in range(len(row_sum) - 1):
        if row_sum[i+1] > row_sum[i]:
            row_start = i
        elif row_sum[i+1] < row_sum[i]:
            if (i - row_start) >= min_height * scale:
                regions.append((row_start, i+1))

    return len(regions), tuple(regions)


def text_cols(img, scale, bgcolor=None, min_width=20, limit=0.1):
    """
    Gets regions of text cols in the given image.

    :param img: image to process
    :param scale: allows to optimize for different resolution, scale=1 is for font size = 16px.
    :type  scale: float
    :param bgcolor: background color (optional). If not set, the background color is detected automatically.
    :param min_width: minimum width of column in pixels, rows with less pixels are not detected.
    :param limit: col coverage with pixels of text used for the column detection. Set to lower value for higher sensitivity (0.05 means that 5% of row has to be text pixels).
    :return:
        - count - number of detected text columns
        - regions - list of regions where the text columns are detected, each region is represented with tuple (x_from, x_to)
    """
    assert img is not None

    min_pixels = img.shape[0] * limit * 255                 # defines how many white pixel in col is minimum for detection (relatively to the image height)
    kernel_dim = int(2*scale - 1)
    kernel = np.ones((kernel_dim, kernel_dim), np.uint8)    # kernel for dilatation/erosion operations

    if bgcolor is None:
        bg_light = background_lightness(img)
    else:
        bg_light = np.mean(color(bgcolor))

    img = img_to_grayscale(img)
    # thresholding on the image, if image is with dark background, use inverted to have white values in the letters
    ret, thresh = cv.threshold(img, 127, 255, (cv.THRESH_BINARY if bg_light < 128 else cv.THRESH_BINARY_INV) + cv.THRESH_OTSU)
    # apply opening (erosion followed by dilation) to remove pepper and salt artifacts and dilatation to fill gaps in between characters
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    dilateted = cv.dilate(opening, kernel, iterations=6)
    # get sum of pixels in cols and make 0/1 thresholding based on minimum pixel count
    col_sum = np.sum(dilateted, axis=0, dtype=np.int32)
    col_sum = np.where(col_sum < min_pixels, 0, 1)
    # put 0 at the beginning to eliminate option that the letters starts right at the top
    col_sum = np.append([0], col_sum)
    col_sum = np.append(col_sum, [0])
    # get count of cols as the number of 0->1 transitions
    regions = []
    col_start = 0

    for i in range(len(col_sum) - 1):
        if col_sum[i+1] > col_sum[i]:
            col_start = i
        elif col_sum[i+1] < col_sum[i]:
            if (i - col_start) >= min_width:
                regions.append((col_start, i+1))

    return len(regions), regions


def gamma_calibration(gradient_img):
    """
    Provides gamma value based on the black-white horizontal gradient image.

    :param gradient_img: black-white horizontal gradient image
    :type  gradient_img: img
    :return: value of the gamma
    """
    img = img_to_grayscale(gradient_img)
    width = img.shape[1]
    img_curve = np.mean(img, axis=0)
    img_curve = [y / 255.0 for y in img_curve]
    ideal_curve = [i / (width-1) for i in range(width)]

    diff = 1.0
    gamma = 1.0

    # iterative gamma adjustment
    for _ in range(50):
        # apply new gamma
        gamma_curve = [(i ** (1.0 / gamma)) for i in img_curve]
        # diff between applied gamma and the ideal curve
        diff = sum([(i - g) for i, g in zip(gamma_curve, ideal_curve)]) / width
        # termination criteria
        if abs(diff) < 0.0001:
            break
        gamma -= diff

    return gamma


def adjust_gamma(img, gamma):
    """
    Applies gamma correction on the given image.

    Gamma values < 1 will shift the image towards the darker end of the spectrum
    while gamma values > 1 will make the image appear lighter. A gamma value of G=1
    will have no affect on the input image.

    :param img: image to adjust
    :type  img: cv2 image
    :param gamma: gamma value
    :type  gamma: float
    :return: adjusted image
    :rtype: cv2 image
    """
    # Create lookup table and use it to apply gamma correction
    invG = 1.0 / gamma
    table = np.array([((i / 255.0) ** invG) * 255 for i in range(256)]).astype('uint8')
    return cv.LUT(img, table)


def crop(img, box, scale, border=0):
    """
    Crops image with given box borders.

    :param img: source image
    :type  img: cv2 image (b,g,r matrix)
    :param box: boundaries of intrested area
    :type  box: [left, top, right, bottom]
    :param scale: target scaling
    :type  scale: float
    :param border: border (in pixels) around cropped display
    :type  border: int
    :return: cropped image
    :rtype: cv2 image (b,g,r matrix)
    """
    max_x = img.shape[1] - 1
    max_y = img.shape[0] - 1
    start_x = np.clip(int(round(box[0]*scale - border)), 0, max_x)
    start_y = np.clip(int(round(box[1]*scale - border)), 0, max_y)
    end_x = np.clip(int(round(box[2]*scale + border)), 0, max_x)
    end_y = np.clip(int(round(box[3]*scale + border)), 0, max_y)

    roi = img[start_y:end_y, start_x:end_x]
    return roi


def read_text(img, language='eng', multiline=False, circle=False, bgcolor=None, chars=None, floodfill=False, langchars=False):
    """
    Reads text from image with use of the Tesseract ORC engine.

    Install Tesseract OCR engine (https://github.com/tesseract-ocr/tesseract/wiki) and add the path to
    the installation folder to your system PATH variable or set the path to `bretina.TESSERACT_PATH`.

    There are several options how to improve quality of the text recognition:

    -   Specify `bgcolor` parameter - the OCR works fine only for black letters on the light background,
        therefore inversion is done when light letters on dark background are recognized. If bgcolor is not set,
        bretina will try to recognize background automatically and this recognition may fail.
    -   Select correct `language`. You may need to install the language data file from
        https://github.com/tesseract-ocr/tesseract/wiki/Data-Files.
    -   If you want to recognize only numbers or mathematical expressions, use special language "equ"
        (`language="equ"`), but you will also need to install the `equ` training data to tesseract.
    -   If you expect only limited set of letters, you can use `chars` parameter, e.g. `chars='ABC'` will
        recognize only characters 'A', 'B', 'C'. Supported wildcards are:

        - **%d** for integral numbers,
        - **%f** for floating point numbers and
        - **%w** for letters.

        Wildcards can be combined with additional characters and other wildcards, e.g. `chars='%d%w?'` will
        recognize all integer numbers, all small and capital letters and question mark.
    -   Enable `floodfill` parameter for the unification of background.

    :param img: image of text
    :type  img: cv2 image (b,g,r matrix)
    :param str language: language of text (use three letter ISO code https://github.com/tesseract-ocr/tesseract/wiki/Data-Files)
    :param bool multiline: control, if the text is treated as multiline or not
    :param bool circle: controls, if the text is treated as text in a circle
    :param str bgcolor: allowes to specify background color of the text, determined automatically if None
    :param str chars: string consisting of the allowed chars
    :param bool floodfill: flag to use flood fill for the background
    :param bool langchars: flag indicates, if the localized language chars shall be used
    :return: read text
    :rtype: string
    """
    # Translation table from various language names
    LANG_CODES = {
        'belarusian': 'bel',
        'bulgarian': 'bul',
        'croatian': 'hrv',
        'czech': 'ces',
        'danish': 'dan',
        'dutch': 'nld',
        'english': 'eng',
        'estonian': 'est',
        'finnish': 'fin',
        'french': 'fra',
        'german': 'deu',
        'greek': 'ell',
        'hungarian': 'hun',
        'italian': 'ita',
        'latvian': 'lav',
        'lithuanian': 'lit',
        'norwegian': 'nor',
        'macedonian': 'mkd',
        'polish': 'pol',
        'portuguese': 'por',
        'romanian': 'ron',
        'russian': 'rus',
        'slovak': 'slk',
        'slovenian': 'slv',
        'spanish': 'spa',
        'swedish': 'swe',
        'turkish': 'tur',
        'ukrainian': 'ukr'
    }

    BORDER = 10     #: [px] additional padding add to the image
    SCRIPT_CYRILLIC = ['bel', 'bul', 'mkd', 'rus', 'ukr']
    SCRIPT_GREEK = ['ell']

    CHARS_COMMON = "1234567890().,;:?!/=+‒\"'’"
    CHARS_CYRILLIC = CHARS_COMMON + "АБВГҐДЂЃЕЁЄЖЗЅИІЇЙЈКЛЉМНЊОПРСТЋЌУЎUФХЦЧЏШЩЪЫЬЭЮЯабвгґдђѓеёєжзѕиіїйјклљмнњопрстћќуўuфхцчџшщъыьэюя"
    CHARS_GREEK = CHARS_COMMON + "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρσςτυφχψω"

    # Options of Tesseract page segmentation mode:
    TESSERACT_PAGE_SEGMENTATION_MODE_00 = '--psm 0'        # Orientation and script detection (OSD) only.
    TESSERACT_PAGE_SEGMENTATION_MODE_01 = '--psm 1'        # Automatic page segmentation with OSD.
    TESSERACT_PAGE_SEGMENTATION_MODE_02 = '--psm 2'        # Automatic page segmentation, but no OSD, or OCR. (not implemented)
    TESSERACT_PAGE_SEGMENTATION_MODE_03 = '--psm 3'        # Fully automatic page segmentation, but no OSD. (Default)
    TESSERACT_PAGE_SEGMENTATION_MODE_04 = '--psm 4'        # Assume a single column of text of variable sizes.
    TESSERACT_PAGE_SEGMENTATION_MODE_05 = '--psm 5'        # Assume a single uniform block of vertically aligned text.
    TESSERACT_PAGE_SEGMENTATION_MODE_06 = '--psm 6'        # Assume a single uniform block of text.
    TESSERACT_PAGE_SEGMENTATION_MODE_07 = '--psm 7'        # Treat the image as a single text line.
    TESSERACT_PAGE_SEGMENTATION_MODE_08 = '--psm 8'        # Treat the image as a single word.
    TESSERACT_PAGE_SEGMENTATION_MODE_09 = '--psm 9'        # Treat the image as a single word in a circle.
    TESSERACT_PAGE_SEGMENTATION_MODE_10 = '--psm 10'      # Treat the image as a single character.
    TESSERACT_PAGE_SEGMENTATION_MODE_11 = '--psm 11'      # Sparse text. Find as much text as possible in no particular order.
    TESSERACT_PAGE_SEGMENTATION_MODE_12 = '--psm 12'      # Sparse text with OSD.
    TESSERACT_PAGE_SEGMENTATION_MODE_13 = '--psm 13'      # Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.

    WHITELIST_EXPRESIONS = {
        '%d': '-0123456789',
        '%f': '-0123456789.',
        '%w': 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    }

    tesseract_cmd = pytesseract.pytesseract.tesseract_cmd

    # Try to find tesseract in %PATH and TESSERACT_PATH
    if not os.path.isfile(tesseract_cmd):
        os_path = os.environ.get('PATH').split(';')
        os_path.append(TESSERACT_PATH)

        for p in os_path:
            path = os.path.join(p, 'tesseract.exe')
            if os.path.isfile(path):
                tesseract_cmd = path
                break

    scaling = 200.0 / max(img.shape[0:2])
    scaling = max(1, scaling)
    img = resize(img, scaling)

    # Check if tesseract was located
    if os.path.isfile(tesseract_cmd):
        # set path to tesseract OCR engine
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    else:
        raise Exception('Tesseract OCR engine not found in system PATH and `bretina.TESSERACT_PATH`.')

    # Convert to grayscale and invert if background is not light
    img = img_to_grayscale(img)

    if bgcolor is not None:
        bg_light = np.mean(color(bgcolor))
    else:
        bg_light = background_lightness(img)

    if bg_light < 127:
        img = 255 - img

    _, img = cv.threshold(img, 127, 255, cv.THRESH_OTSU + cv.THRESH_BINARY)

    # Floodfill of the image background
    if floodfill:
        h, w = img.shape[:2]
        if h > 1 and w > 1:
            mask = np.zeros((h + 2, w + 2), np.uint8)
            # Start from all corners
            for seed in [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]:
                try:
                    cv.floodFill(img, mask, seed, 255)
                except Exception as ex:
                    pass

    # Add padding, tesseract works better with it
    img = cv.copyMakeBorder(img, BORDER, BORDER, BORDER, BORDER, cv.BORDER_CONSTANT, value=COLOR_WHITE)

    # Special page segmentation mode for text in circle
    if circle:
        psm_opt = TESSERACT_PAGE_SEGMENTATION_MODE_09
    elif multiline:
        psm_opt = TESSERACT_PAGE_SEGMENTATION_MODE_03
    else:
        psm_opt = TESSERACT_PAGE_SEGMENTATION_MODE_07

    # Standardize language
    language = language.lower()

    if language in LANG_CODES:
        language = LANG_CODES[language]

    assert not ((chars is not None) and langchars), 'Argument `langchars` can not be used together with `chars`.'

    # Add language specific characters
    if langchars:
        if language in SCRIPT_CYRILLIC:
            chars = CHARS_CYRILLIC
        elif language in SCRIPT_GREEK:
            chars = CHARS_GREEK
        else:
            chars = None            # latin scripts are not limited

    # Create whitelist of characters
    whitelist = ''

    if chars is not None and len(chars) > 0:
        for s, val in WHITELIST_EXPRESIONS.items():
            chars = chars.replace(s, val)
        whitelist = '-c tessedit_char_whitelist=' + chars

    # Create config and call OCR
    config = '-l {lang} {psm} {whitelist}'.format(
        lang=language,
        psm=psm_opt,
        whitelist=whitelist)
    text = pytesseract.image_to_string(img, config=config)

    return text


def img_diff(img, template, edges=False, inv=None, bgcolor=None, blank=None, split_threshold=64):
    """
    Calculates difference of two images.

    :param img: image taken from camera
    :param template: source image
    :param bool edges: controls if the comparision shall be done on edges only
    :param bool inv: specifies if image is inverted
                     - [True]   images are inverted before processing (use for dark lines on light background)
                     - [False]  images are not inverted before processing (use for light lines on dark background)
                     - [None]   inversion is decided automatically based on `img` background
    :param bgcolor: specify color which is used to fill transparent areas in png with alpha channel, decided automatically when None
    :param list blank: list of areas which shall be masked
    :param int split_threshold: value used for thresholding
    :return: difference ration of two images, different pixels / template pixels
    """
    scaling = 120.0 / max(template.shape[0:2])
    scaling = max(1, scaling)
    img = resize(img, scaling)
    template = resize(template, scaling)

    alpha = np.ones(template.shape[0:2], dtype=np.uint8) * 255

    # get alpha channel and mask the template
    if len(template.shape) == 3 and template.shape[2] == 4:
        # only if there is an information in the alpha channel
        if lightness_std(template[:, :, 3]) > 5:
            alpha = template[:, :, 3]
            _, alpha = cv.threshold(alpha, 127, 255, cv.THRESH_BINARY)
            template = cv.bitwise_and(template[:, :, :3], template[:, :, :3], mask=alpha)

            temp_bg = template.copy()

            if bgcolor is None:
                temp_bg[:] = dominant_color(img)
            else:
                temp_bg[:] = color(bgcolor)

            temp_bg = cv.bitwise_and(temp_bg, temp_bg, mask=255-alpha)
            template = cv.add(template, temp_bg)
        else:
            template = template[:, :, :3]

    # add blanked areas to alpha mask
    if blank is not None:
        assert isinstance(blank, (list, tuple, set, frozenset)), '`blank` has to be list'

        # make list if only one area is given
        if len(blank) > 0 and not isinstance(blank[0], (list, tuple, set, frozenset)):
            blank = [blank]

        for area in blank:
            # rescale and set mask in area to 0
            area = [int(round(a * scaling)) for a in area]
            alpha[area[1]:area[3], area[0]:area[2]] *= 0

    img_gray = img_to_grayscale(img)
    src_gray = img_to_grayscale(template)

    res = cv.matchTemplate(img_gray, src_gray, cv.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    # crop only region with maximum similarity
    x, y = max_loc
    h, w = src_gray.shape
    img_gray = img_gray[y:y+h, x:x+w]

    if inv or (inv is None and np.mean(background_color(img)) > 127):
        img_gray = 255 - img_gray
        src_gray = 255 - src_gray

    if edges:
        img_gray = cv.Canny(img_gray, 150, 150)
        src_gray = cv.Canny(src_gray, 150, 150)
        kernel = np.ones((9, 9), np.uint8)
        img_gray = cv.morphologyEx(img_gray, cv.MORPH_CLOSE, kernel)
        src_gray = cv.morphologyEx(src_gray, cv.MORPH_CLOSE, kernel)

    # mask alpha
    img_gray = cv.bitwise_and(img_gray, img_gray, mask=alpha)

    _, img_gray = cv.threshold(img_gray, split_threshold, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    _, src_gray = cv.threshold(src_gray, split_threshold, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # get difference
    diff = np.absolute(src_gray.astype(int) - img_gray.astype(int)).astype('uint8')

    # remove small fragments
    kernel = np.ones((5, 5), np.uint8)
    diff = cv.morphologyEx(diff, cv.MORPH_OPEN, kernel)

    # mask alpha
    diff = cv.bitwise_and(diff, diff, mask=alpha)

    # sum pixels and get difference ratio
    n_img = np.sum(img_gray)
    n_src = np.sum(src_gray)
    n_alpha = np.sum(alpha)
    n_dif = np.sum(diff)
    ratio = n_dif / n_alpha * 64.0

    #### some temp ploting
    """
    source = np.concatenate((img_gray, src_gray), axis=1)
    full = np.concatenate((source, diff), axis=1)
    full_col = np.zeros((full.shape[0], full.shape[1], 3), dtype=np.uint8)

    if ratio > 1.0:
        full_col[:, :, 2] = full
    else:
        full_col[:, :, 1] = full

    cv.imshow(str(ratio), full_col)
    cv.waitKey()
    cv.destroyAllWindows()
    """
    ####
    return ratio


def resize(img, scale):
    """
    Resize image to a given scale and returns its copy

    :param img: source image
    :type  img: cv2 image (b,g,r matrix)
    :param scale: scale between source and target resolution
    :type  scale: float
    :return: scaled image
    :rtype: cv2 image (b,g,r matrix)
    """
    if scale == 1:
        return img.copy()
    else:
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        image_resized = cv.resize(img.copy(), (width, height), interpolation=cv.INTER_CUBIC)
        return image_resized


def recognize_animation(images, template, size, scale, split_threshold=64):
    """
    Recognize image animation and return duty cycles and animation period

    :param images: array of images
    :type  images: array
    :param template: template for animated image }animation in rows/coloms
    :type  template: cv2 image
    :param size: expected size of one separated image
    :type  size: tuple (width, height) int/float
    :param scale: scale between source and target resolution
    :type  scale: float
    :return: image difference, animation
    :rtype: float, bool
    """
    # load template images (resize and separate)
    blank = None
    templates = separate_animation_template(template, size, scale)
    assert len(templates) != 0, 'no usable template to test animation, bad template size or path'

    for x, img_template in enumerate(templates):
        if lightness_std(img_template) < 5:
            blank = x
    read_item = {}
    periods = []

    for x, image in enumerate(images):
        result = []
        if lightness_std(image) < 16:
            min_val = 0
            i = blank
        else:
            # compare template images with captured
            for img_template in templates:
                result.append(img_diff(image, img_template, split_threshold=split_threshold))
            min_val = min(result)
            i = result.index(min_val)

        # choose the least different template
        if i not in read_item:
            read_item[i] = [[min_val], 1, x]
            continue

        # identify if image was captured in same period
        if read_item[i][2] != x-1:
            periods.append(x-read_item[i][2])

        read_item[i][0].append(min_val)
        read_item[i][1] += 1
        read_item[i][2] = x

    # identify if image is blinking, compute period difference
    if len(periods) == 0:
        animation = False
    else:
        animation = True

    # count image difference
    diff = []
    for x in read_item:
        if x != blank:
            diff.append(np.mean(read_item[x][0]))
    if len(diff) == 0:
        # there is no confirmity, return maximum difference
        difference = float('Inf')
    else:
        difference = np.mean(diff)
    return (difference, animation)


def separate_animation_template(img, size, scale):
    """
    Separate individual images from one composite image

    :param img: composite image
    :type  img: cv2 image (b,g,r matrix)
    :param size: expected size of one separated image
    :type  size: tuple (width, height) int/float
    :param scale: scale between source and target resolution
    :type  scale: float
    :return: array of seperated images
    :rtype: array of cv2 image (b,g,r matrix)
    """
    img = resize(img, scale)
    width = img.shape[1]
    height = img.shape[0]
    size = (int(size[0]*scale), int(size[1]*scale))
    templates = []

    for row in range(int(height // size[1])):
        for column in range(int(width // size[0])):
            templates.append(img[row*size[1]:(1+row)*size[1], column*size[0]:(1+column)*size[0]])
    return templates


def format_diff(diff):
    """
    Converts diff list to human readable form in form of 3-line text

    Input coding:
        - `  x` char `x` common to both sequences
        - `- x` char `x` unique to sequence 1
        - `+ x` char `x` unique to sequence 2
        - `~- x` char `x` unique to sequence 1 but not considered as difference
        - `~+ x` char `x` unique to sequence 2  but not considered as difference

    `~` is a special mark indicating that the difference was evaluated as not significant
    (e.g. `v` vs `V`).

    Output marks:
        - ^ is used to mark difference
        - ~ is used to mark allowed difference (not included in the ratio calculation)

    Example:
    Diff made of strings "vigo" and "Viga" shall be ['~- v', '~+ V', '  i', '  g', '- o', '+ a']
    and the outpus is formated as

    ```
    v igo
     Vig a
    ~~  ^^
    ```

    :param list diff: list of difflib codes (https://docs.python.org/3.8/library/difflib.html)
    :return: 3 rows of human readable text
    :rtype string:
    """
    l1 = ""
    l2 = ""
    l3 = ""

    for d in diff:
        if d.startswith("~"):
            mark = "~"
            d = d[1:]
        else:
            mark = "^"

        if d.startswith("-"):
            l1 += d[-1]
            l2 += " "
            l3 += mark
        elif d.startswith("+"):
            l1 += " "
            l2 += d[-1]
            l3 += mark
        elif d.startswith(" "):
            l1 += d[-1]
            l2 += d[-1]
            l3 += " "

    return "\n".join((l1, l2, l3))


def compare_str(a, b, simchars=None, ligatures=None):
    """
    Compares two strings and returns result, allowes to define similar
    characters which are not considered as difference.

    In default version, the trimmed strings are compared (" A " == "A")
    When `simchars` argument is set, more complex algorithm is used and
    some difference are ignored.

    :param str a: left side of the string comparision
    :param str b: right side of the string comparision
    :param list simchars: e.g. ["1il", "0oO"] or None
    :param list ligatures: list of ligatures
    :return: tuple diffs (int, tuple(string)).
             **int**: number of differences
             **tuple(string)** string with diff codes
    :rtype: tuple(bool, float)
    """
    assert isinstance(a, str), f'`a` has to be string, {type(a)} given'
    assert isinstance(b, str), f'`b` has to be string, {type(b)} given'

    # remove white spaces
    a = ' '.join(a.split())
    b = ' '.join(b.split())

    # replace ligatures
    if ligatures is not None:
        for lig in ligatures:
            a = a.replace(lig[0], lig[1])
            b = b.replace(lig[0], lig[1])

    # quick check of the equal strings leading to the fast True
    if a == b:
        return 0, (f'  {_}' for _ in a)

    res = []
    sims = []

    # generate all combinations of the given similar characters
    if isinstance(simchars, str):
        simchars = [simchars]

    assert all(isinstance(el, str) for el in simchars), '`simchars` argument has to be list of strings, e.g. ["1il", "0oO"]'

    for string in simchars:
        sims += list(itertools.permutations(string, 2))

    # get list of differences, filter spaces
    df = difflib.ndiff(a, b)
    df = filter(lambda x: x not in ('+  ', '-  ', '?  '), df)

    # remove differences matching simchars
    for d in df:
        # '-': char only in A, '+': char only in B
        if len(res) > 0:
            for i in range(len(res)-1, -1, -1):
                if (res[i][0] not in (' ', '~')) and (not d.startswith(' ')):
                    if not res[i].startswith(d[0]) and ((res[i][-1], d[-1]) in sims):
                        res[i] = '~' + res[i]
                        d = '~' + d     # to prevent pop
                        break
        res.append(d)

    diffs = list(filter(lambda x: x[0] in ('+', '-', '?'), res))
    r = math.ceil(len(diffs) / 2)

    return int(r), res


def remove_accents(s):
    """
    Sanitizes given string, removes accents, umlauts etc.

    :param str s: string to remove accents
    :return: sanitized string without accents
    :rtype: str
    """
    # the character category "Mn" stands for Nonspacing_Mark
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))

def color_region_detection(img, desired_color, scale, padding=10, tolerance=50):
    """
    :param img: opencv image
    :param color: color to find
    :param padding: (optional) optional parameter to add some padding to the box
    :param tolerance: set tolerance zone (color +-tolerance) to find desired color
    """
    assert tolerance >= 0, 'tolerance must be positive'
    b, g, r = color(desired_color)
    lower = np.maximum((b-tolerance, g-tolerance, r-tolerance), (0, 0, 0))
    upper = np.minimum((b+tolerance, g+tolerance, r+tolerance), (255, 255, 255))
    mask = cv.inRange(img, lower, upper)

    # remove small fragments
    kernel_size = int(1 * scale)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    bound = cv.boundingRect(mask)

    white_pix = np.sum(mask)/255
    all_pix = mask.shape[0]*mask.shape[1]

    if white_pix/all_pix*10000 < 1:
        return None
    left, top, width, height = bound
    right = left + width
    bottom = top + height
    left = max(left // scale - padding, 0)
    top = max(top // scale - padding, 0)
    right = min(right // scale + padding, img.shape[1])
    bottom = min(bottom // scale + padding, img.shape[0])
    return tuple(int(_) for _ in (left, top, right, bottom))
