"""
    bretina
    =======

    Bender Robotics module for visual based testing support.

    :copyright: 2020 Bender Robotics
"""

__version__ = '0.7.0'
__all__ = ['VisualTestCase', 'ImageStitcher', '__version__', 'polyline']

import numpy as np
import cv2 as cv
import os
import math
import difflib
import itertools
import pytesseract
import unicodedata
import tempfile
import textwrap

from datetime import datetime
from PIL import Image, ImageFont, ImageDraw
from string import ascii_letters

from bretina.visualtestcase import VisualTestCase
from bretina.imagestitcher import ImageStitcher

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
CONFUSABLE_CHARACTERS = []

#: List of ignored characters, when OCR-ed and expected text differs in the chars
#: in chars which are listed bellow, this difference is not considered. With '°', '10 °C'
#: and '10 C' are treated as equal
EXPENDABLE_CHARACTERS = []

#: Default path to the Tesseract OCR engine installation
TESSERACT_PATH = 'C:\\Program Files (x86)\\Tesseract-OCR'

#: Default path to the Tesseract OCR engine trained data
TESSDATA_PATH = 'C:\\Tesseract-Data\\'

#: Limit of the applied languages based on the string format
LANGUAGE_LIMITED = None

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

#: Translation table from various language names
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
    'ukrainian': 'ukr',
    'serbian': 'srp_latn', # consider latin variant of serbian as default one
    'serbian_cyrl': 'srp',
    'montenegrin': 'srp_latn', # tesseract does not have Montenegrin, but it is close to Serbian
    'montenegrin_cyrl': 'srp', # tesseract does not have Montenegrin, but it is close to Serbian
}


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


def dominant_colors(img, n=3):
    """
    Returns list of dominant colors in the image ordered according to its occurency (major first),
    internally performs k-means clustering (https://docs.opencv.org/master/d1/d5c/tutorial_py_kmeans_opencv.html) for the segmentation

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
    return [tuple(color) for color in palette[indexes]]


def dominant_color(img, n=3):
    """
    Gets the most dominant color in the image using `dominant_colors` function.

    :param img: source image
    :param n: number of colors in segmentation
    :return: (B, G, R) color tuple
    :rtype: tuple
    """
    return dominant_colors(img, n)[0]


def active_color(img, bgcolor=None, n=3):
    """
    Gets the most dominant color which is not the background color.

    :param array_like img: source image
    :param bgcolor: color of the image background, recognized automatically if `None` or not set
    :param n: number of colors in segmentation
    :return: (B, G, R) color tuple
    :rtype: tuple
    """
    colors = dominant_colors(img, n)

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
    return tuple(np.mean(pixels, axis=0))


def background_color(img, border=2, mean=True):
    """
    Returns Mean color from the border of the image.

    :param img: source image
    :param border: [px] width of the border to calculate the mean
    :param bool mean: True - mean is used to calculate the background color, False - majority color is used
    :return: mean or major color of the image border
    :rtype: tuple
    """
    colors = 3 if (len(img.shape) == 3 and img.shape[2] == 3) else 1
    # take pixels from top, bottom, left and right border lines
    pixels = np.concatenate((np.float32(img[:border, :].reshape(-1, colors)),
                             np.float32(img[-border:, :].reshape(-1, colors)),
                             np.float32(img[:, :border].reshape(-1, colors)),
                             np.float32(img[:, -border:].reshape(-1, colors))))

    if mean:
        return tuple(np.mean(pixels, axis=0))
    else:
        # k-means clustering
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv.KMEANS_RANDOM_CENTERS
        _, labels, palette = cv.kmeans(pixels, 3, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)
        indexes = np.argsort(counts)[::-1]
        return tuple(palette[indexes][0])


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


def draw_border(img, box, color=COLOR_RED, padding=0, thickness=1):
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
    start_x = np.clip(int(round(box[0] - padding)), 0, max_x)
    start_y = np.clip(int(round(box[1] - padding)), 0, max_y)
    end_x = np.clip(int(round(box[2] + padding)), 0, max_x)
    end_y = np.clip(int(round(box[3] + padding)), 0, max_y)

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
        - regions - tuple of regions where the text rows are detected, each region is represented with tuple (`y_from`, `y_to`)
    :rtype: Tuple
    """
    assert img is not None
    regions = []

    if (img.shape[0] * img.shape[1]) > 0:
        # defines how many white pixel in row is minimum for row detection (relatively to the image width)
        min_pixels = img.shape[1] * limit * 255
        # kernel for dilatation/erosion operations
        kernel_dim = int(2*scale - 1)
        kernel = np.ones((kernel_dim, kernel_dim), np.uint8)

        try:
            if bgcolor is None:
                bg_light = background_lightness(img)
            else:
                bg_light = np.mean(color(bgcolor))
        except ValueError:
            bg_light = np.mean(img[0, 0, :])

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
        row_start = 0

        for i in range(len(row_sum) - 1):
            if row_sum[i+1] > row_sum[i]:
                row_start = i
            elif row_sum[i+1] < row_sum[i]:
                if (i - row_start) >= min_height * scale:
                    regions.append((row_start, i+1))

    return len(regions), tuple(regions)


def text_cols(img, scale, bgcolor=None, min_width=20, limit=0.025):
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
        - regions - list of regions where the text columns are detected, each region is represented with tuple (`x_from`, `x_to`)
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


def split_cols(img, scale, col_count=2, bgcolor=None, limit=0.025, padding=0):
    """
    Identifies given number of columns in the image and returns regions of these.

    :param img: image to process
    :param scale: allows to optimize for different resolution, scale=1 is for font size = 16px.
    :type  scale: float
    :param col_count: number of columns to identify
    :type col_count: int
    :param bgcolor: background color (optional). If not set, the background color is detected automatically.
    :param limit: col coverage with pixels of text used for the column detection. Set to lower value
                  for higher sensitivity (0.05 means that 5% of row has to be text pixels).
    :param padding: additional space [px] to extend the regions, applied padding is multiplied by `scale`
    :return: regions - list of regions where the text columns are detected, each region is represented with tuple (`x_from`, `x_to`)
    """
    assert img is not None
    height, width = img.shape[0], img.shape[1]
    min_pixels = height * limit * 255                 # defines how many white pixel in col is minimum for detection (relatively to the image height)
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
    # put 0 at the beginning and end to eliminate option that the letters starts right at the start of the row
    col_sum = np.append([0], col_sum)
    col_sum = np.append(col_sum, [0])
    #
    edges = []
    lens = []

    for i in range(len(col_sum) - 1):
        if col_sum[i+1] != col_sum[i]:
            edges.append(i)

            if len(edges) > 1:
                lens.append(i+1 - edges[-2])

    while len(edges) > (2 * col_count):
        min_len = min(lens[1:-1])
        min_index = lens.index(min_len)
        # remove two closes edges
        edges.pop(min_index)
        edges.pop(min_index)
        lens.pop(min_index)
        lens[min_index] += min_len
        lens[min_index - 1] += lens.pop(min_index)

    regions = []

    # only one region detected
    if len(edges) < 2:
        regions = [(0, width-1)]
    else:
        for i in range(0, len(edges)-1, 2):
            regions.append([edges[i], edges[i+1]])

        # apply padding
        for i in range(len(regions)):
            width = regions[i][1] - regions[i][0]

            # shift right side by padding if this is not the last region
            if i < (len(regions)-1):
                space_right = regions[i+1][0] - regions[i][1]
                regions[i][1] += min(space_right / 2, padding * scale)

            # shift left side by padding if this is not the first region
            if i > 0:
                space_left = regions[i][0] - regions[i-1][1]
                regions[i][0] -= min(space_left / 2, padding * scale)

    return [tuple(_) for _ in regions]


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
    if box is None:
        return img

    max_x = img.shape[1] - 1
    max_y = img.shape[0] - 1
    start_x = np.clip(int(round(box[0]*scale - border)), 0, max_x)
    start_y = np.clip(int(round(box[1]*scale - border)), 0, max_y)
    end_x = np.clip(int(round(box[2]*scale + border)), 0, max_x)
    end_y = np.clip(int(round(box[3]*scale + border)), 0, max_y)

    roi = img[start_y:end_y, start_x:end_x]
    return roi


def read_text(img, language='eng', multiline=False, circle=False, bgcolor=None, chars=None, floodfill=False,
              singlechar=False, tessdata=None, patterns=None):
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
    :param str language: language of text (use three letter ISO code https://github.com/tesseract-ocr/tesseract/wiki/Data-Files),
        you can also allow multiple languages with '+' operator (e.g. 'hun+eng' preferrers hungarian but uses also eng)
    :param bool multiline: control, if the text is treated as multiline or not
    :param bool circle: controls, if the text is treated as text in a circle
    :param str bgcolor: allowes to specify background color of the text, determined automatically if None
    :param str chars: string consisting of the allowed chars
    :param bool floodfill: flag to use flood fill for the background
    :param str tessdata: path to the tesseract trained datasets which shall be used for recognition
    :param patterns: single pattern e.g "\d\A\A" or list of patterns, e.g.["\A\d\p\d\d", "\d\A\A"]
    :type patterns: string or list of strings
    :return: read text
    :rtype: string
    """
    BORDER = 10     #: [px] additional padding add to the image

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
    TESSERACT_PAGE_SEGMENTATION_MODE_10 = '--psm 10'       # Treat the image as a single character.
    TESSERACT_PAGE_SEGMENTATION_MODE_11 = '--psm 11'       # Sparse text. Find as much text as possible in no particular order.
    TESSERACT_PAGE_SEGMENTATION_MODE_12 = '--psm 12'       # Sparse text with OSD.
    TESSERACT_PAGE_SEGMENTATION_MODE_13 = '--psm 13'       # Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.

    WHITELIST_EXPRESIONS = {
        '%d': '-0123456789',
        '%f': '-0123456789.,',
        '%w': 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    }

    scaling = 200.0 / max(img.shape[0:2])
    scaling = max(1, scaling)
    img = resize(img, scaling)

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
    elif singlechar:
        psm_opt = TESSERACT_PAGE_SEGMENTATION_MODE_10
    else:
        psm_opt = TESSERACT_PAGE_SEGMENTATION_MODE_07

    languages = language.split('+')
    languages = [normalize_lang_name(lang) for lang in languages]
    languages = [lang for lang in languages if lang]            # filter empty strings
    languages = sorted(set(languages), key=languages.index)     # use set to remove duplicit langs
    language = '+'.join(languages)                              # join back to one expresion

    # Create whitelist of characters
    whitelist = ''

    if chars is not None and len(chars) > 0:
        for s, val in WHITELIST_EXPRESIONS.items():
            chars = chars.replace(s, val)
        whitelist = '-c tessedit_char_whitelist=' + chars

    if tessdata is not None:
        tessdata = f'--tessdata-dir "{tessdata}"'
    else:
        tessdata = ''

    # User patterns
    if patterns is not None:
        patterns = patterns if isinstance(patterns, (list, tuple, set)) else [patterns]
        file_patterns = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',  newline='\n', suffix='.patterns', delete=False)
        for pattern in patterns:
            file_patterns.write(pattern + '\n')
        file_patterns_name = file_patterns.name.replace('\\', '/' )
        patterns_config = f'--user-patterns {file_patterns_name}'
        file_patterns.close()
    else:
        patterns_config = ''

    # Find tesseract engine (and prepare pytesseract.pytesseract.tesseract_cmd)
    get_tesseract_location()

    # Create config from not empty flags and call OCR
    config = ' '.join([f for f in (psm_opt, tessdata, whitelist, patterns_config) if f])
    text = pytesseract.image_to_string(img, lang=language, config=config)

    # Delete patterns temp file
    if patterns is not None:
        os.remove(file_patterns.name)

    return text


def get_tesseract_trained_data():
    """
    Lists all available Tesseract trained data sets (in Tesseract installation directory and in TESSDATA_PATH)

    :return: list of paths to the trained data
    :rtype: list
    """
    tess_path = get_tesseract_location()
    tess_data_dir = []
    dirs = []

    if tess_path is not None:
        tess_dir, _ = os.path.split(tess_path)
        dirs.append(tess_dir)

    if os.path.isdir(TESSDATA_PATH):
        dirs.append(TESSDATA_PATH)

    for d in dirs:
        for f in os.scandir(d):
            if f.is_dir():
                root, directory = os.path.split(f.path)
                if directory.startswith('tessdata'):
                    tess_data_dir.append(f.path)

    if not tess_data_dir:
        return [None]
    else:
        return tess_data_dir


def get_tesseract_location():
    """
    Find tesseract engine installation and prepares pytesseract.pytesseract.tesseract_cmd.

    :return: path to the tesseract.exe
    :rtype: str
    """
    tesseract_cmd = pytesseract.pytesseract.tesseract_cmd

    # Try to find tesseract in %PATH and TESSERACT_PATH
    if not os.path.isfile(tesseract_cmd):
        os_path = os.environ.get('PATH', '').split(';')
        os_path.append(TESSERACT_PATH)

        for p in os_path:
            path = os.path.join(p, 'tesseract.exe')

            if os.path.isfile(path):
                tesseract_cmd = path
                break

    # Check if tesseract was located
    if os.path.isfile(tesseract_cmd):
        # set path to tesseract OCR engine
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        return tesseract_cmd
    else:
        print('WARNING: Tesseract OCR engine not found in system `PATH` and `bretina.TESSERACT_PATH`.')
        return None

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
    :param template: template for animated image
    :type  template: cv2 image
    :param size: expected size of one separated image in px
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
        # there is no conformity, return maximum difference
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
    if len(templates) < 2:
        templates.append(_blank_image(size[1], size[0], 3))
    return templates


def format_diff(diff, max_len=0):
    """
    Converts diff list to human readable form in form of 3-line text

    Input coding:
        - ``x``: char ``x`` common to both sequences
        - ``- x``: char ``x`` unique to sequence 1
        - ``+ x``: char ``x`` unique to sequence 2
        - ``~- x``: char ``x`` unique to sequence 1 but not considered as difference
        - ``~+ x``: char ``x`` unique to sequence 2  but not considered as difference

    ``~`` is a special mark indicating that the difference was evaluated as not significant
    (e.g. ``v`` vs ``V``).

    Output marks:
        - ``^`` is used to mark difference
        - ``~`` is used to mark allowed difference (not included in the ratio calculation)

    Example:
    Diff made of strings "vigo" and "Viga" shall be ``['~- v', '~+ V', '  i', '  g', '- o', '+ a']``
    and the outpus is formated as::

         vig o
        V iga
        ~~  ^^

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

    if max_len > 0 and len(l3) > max_len:
        l1_short = "… "
        l2_short = "… "
        l3_short = "  "
        flag = False
        span = 8

        for i, char in enumerate(l3):
            if any(map(lambda x: x != " ", l3[max(0, i-span):min(len(l3)-1, i+span)])):
                l1_short += l1[i]
                l2_short += l2[i]
                l3_short += l3[i]
                flag = True
            elif flag:
                l1_short += " … "
                l2_short += " … "
                l3_short += "   "
                flag = False

        return "\n".join((l1_short, l2_short, l3_short))
    else:
        return "\n".join((l1, l2, l3))


def string_equal(a, b, tolerate_similar_case=True):
    """
    Compares if two strings are equal. Allows to ignore difference in similar letters case ('I' vs 'i')

    :param str a: left side operand
    :param str b: right side operand
    :param bool tolerate_similar_case: True - differences in similar letters case are ignored
    :return: True - strings are equal, False - strings are not equal
    :rtype: bool
    """
    if tolerate_similar_case:
        CASE_SIMILAR_LETTERS = ('C', 'I', 'J', 'K', 'O', 'P', 'S', 'U', 'V', 'W', 'X', 'Y', 'Z')
    else:
        CASE_SIMILAR_LETTERS = tuple()

    # very quick check of the equal strings leading to the fast True
    if a == b:
        return True
    # if both strings has the same length
    elif len(a) == len(b):
        diffs = [abs(ord(l) - ord(r)) for l, r in zip(a, b)]

        # if there is any difference (!= 0) which is not in case (A - a = 32), return false
        if any(d not in (0, 32) for d in diffs):
            return False

        # if there is un-allowed diff in case, return false
        if any((d == 32) and (char not in CASE_SIMILAR_LETTERS) for char, d in zip(a.upper(), diffs)):
            return False

        return True
    # if strings are not same and have different lengths
    else:
        return False


def compare_str(a, b, simchars=None, ligatures=None, ignore_duplicate=True, expendable_chars=[]):
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
    :param bool ignore_duplicate: set to true to ignore duplicated chars e.g. "aapple" vs "apple"
    :param list expendable_chars: set of chars which may are allowed to be missing in the text
    :return: tuple diffs (int, tuple(string)).
             **int**: number of differences
             **tuple(string)** string with diff codes
    :rtype: tuple(bool, float)
    """
    assert isinstance(a, str), f'`a` has to be string, {type(a)} given'
    assert isinstance(b, str), f'`b` has to be string, {type(b)} given'

    # remove multiple white spaces
    a = ' '.join(a.split())
    b = ' '.join(b.split())

    # replace ligatures
    if ligatures is not None:
        for lig in ligatures:
            a = a.replace(lig[0], lig[1])
            b = b.replace(lig[0], lig[1])

    # quick check of the equal strings leading to the fast True
    if string_equal(a, b):
        return 0, (f'  {_}' for _ in a)

    res = []
    sims = set()

    # generate all combinations of the given similar characters
    if isinstance(simchars, str):
        simchars = [simchars]

    assert all(isinstance(el, str) for el in simchars), '`simchars` argument has to be list of strings, e.g. ["1il", "0oO"]'

    # create tuple of chars which are ignored
    expendables = tuple([f'- {c}' for c in expendable_chars] + [f'+ {c}' for c in expendable_chars])

    for string in simchars:
        sims.update(itertools.permutations(string, 2))

    # get list of differences, filter spaces
    df = difflib.ndiff(a, b)

    # remove differences matching simchars
    for d in df:
        # ignore differences in spaces and hyphens
        if (d in ('-  ', '+  ',
                  '- ·', '+ ·',
                  '- -', '+ -',
                  '- ‐', '+ ‐',
                  '- ‑', '+ ‑',
                  '- ‒', '+ ‒',
                  '- –', '+ –',
                  '- —', '+ —',
                  '- ―', '+ ―',
                  '- −', '+ −',)):
            d = '~' + d
        # '-': char only in A, '+': char only in B
        elif len(res) > 0:
            for i in range(len(res)-1, -1, -1):
                if (res[i][0] not in (' ', '~')) and (not d.startswith(' ')):
                    # only if first char of d and res is not same and combination is in sims
                    if not res[i].startswith(d[0]) and ((res[i][-1], d[-1]) in sims):
                        res[i] = '~' + res[i]
                        d = '~' + d
                        break
        res.append(d)

    # if duplicated chars are ignores, check if there is a same char before or after each diff,
    # if so, replace with "~"
    if ignore_duplicate:
        for i in range(len(res)-1):
            if res[i][0] in ('-', '+') and res[i+1].startswith(' ') and res[i][-1] == res[i+1][-1]:
                res[i] = "~" + res[i]

        for i in range(1, len(res)):
            if res[i][0] in ('-', '+') and res[i-1].startswith(' ') and res[i][-1] == res[i-1][-1]:
                res[i] = "~" + res[i]

    for i, d in enumerate(res):
        if d in expendables:
            res[i] = '~' + res[i]

    diffs = list(filter(lambda x: x[0] in ('+', '-', '?'), res))
    r = math.ceil(len(diffs) / 2)

    return int(r), res


def remove_accents(s):
    """
    Sanitizes given string, removes accents, umlauts etc.
    Sentence "příliš žluťoučký kůň úpěl ďábelské ódy" is turned to "prilis zlutoucky kun upel dabelske ody".

    :param str s: string to remove accents
    :return: sanitized string without accents
    :rtype: str
    """
    # the character category "Mn" stands for Nonspacing_Mark
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))


def normalize_lang_name(language):
    """
    Normalize given language name to ISO code.

    :param str language: language name (e.g. "English")
    :return: iso language code (e.g. "eng")
    :rtype: str
    """
    language = language.strip().lower()

    if language in LANG_CODES:
        language = LANG_CODES[language]

    return language


def color_region_detection(img, desired_color, scale, roi=None, padding=10, tolerance=50):
    """
    This function is used to localize region of the given color in the source image.

    :param img: opencv image where to locate the color
    :param desired_color: color to locate
    :param scale: scale between source and target resolution
    :param roi: region of interest - limits area where the color is located
    :param padding: (optional) optional parameter to add some padding to the box
    :param tolerance: set tolerance zone (color +-tolerance) to find desired color
    :return: (left, top, right, bottom) tuple of the color region or `None` when the color is not localized
    :rtype: tuple
    """
    assert tolerance >= 0, 'tolerance must be positive'
    img = crop(img.copy(), roi, scale)
    b, g, r = color(desired_color)
    lower = np.maximum((b-tolerance, g-tolerance, r-tolerance), (0, 0, 0))
    upper = np.minimum((b+tolerance, g+tolerance, r+tolerance), (255, 255, 255))
    mask = cv.inRange(img, lower, upper)

    # remove small fragments
    kernel_size = int(1 * scale)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    bound = cv.boundingRect(mask)

    white_pix = np.sum(mask) / 255
    all_pix = mask.shape[0] * mask.shape[1]

    if ((white_pix / all_pix) * 10000) < 1:
        return None

    left, top, width, height = bound

    if roi is not None:
        assert isinstance(roi, (list, tuple)), '`roi` has to be collection'
        assert len(roi) == 4, '`roi` has to be collection of length 4 - [left, top, right, bottom]'

        left += (roi[0] * scale)
        top += (roi[1] * scale)

    right = left + width
    bottom = top + height
    left = max(left // scale - padding, 0)
    top = max(top // scale - padding, 0)
    right = min(right // scale + padding, img.shape[1])
    bottom = min(bottom // scale + padding, img.shape[0])

    return tuple(int(_) for _ in (left, top, right, bottom))


def img_hist_diff(img, template, bgcolor=None, blank=None):
    """
    Calculates the difference between two images based on their color histograms.

    :param img: image taken from camera
    :param template: source image
    :param bgcolor: specify color which is used to fill transparent areas in png with alpha channel, decided automatically when `None`
    :param list blank: list of areas which shall be masked
    :return: difference ration of two image histograms (0 - same, 1 - different)
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
        assert isinstance(blank, list), '`blank` has to be list'

        # make list if only one area is given
        if len(blank) > 0 and not isinstance(blank[0], list):
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
    img_cropp = img[y:y+h, x:x+w]

    kernel = np.ones((3, 3), np.uint8)
    img_cropp = cv.morphologyEx(img_cropp, cv.MORPH_OPEN, kernel)
    img_cropp = cv.GaussianBlur(img_cropp, (5, 5), 3)
    template = cv.morphologyEx(template, cv.MORPH_OPEN, kernel)
    template = cv.GaussianBlur(template, (5, 5), 3)

    # Convert it to LAB
    img_lab = cv.cvtColor(img_cropp, cv.COLOR_BGR2LAB)
    templ_lab = cv.cvtColor(template, cv.COLOR_BGR2LAB)

    hist = cv.calcHist([img_lab], [1, 2], alpha, [256, 256], [0, 256, 0, 256])
    hist2 = cv.calcHist([templ_lab], [1, 2], alpha, [256, 256], [0, 256, 0, 256])

    # size of compared area (unmasked pixels for all colors)
    compared = np.sum(alpha) / 255 * 3
    return cv.compareHist(hist, hist2, cv.HISTCMP_KL_DIV)/compared


def _blank_image(h, w, channels):
    """
    Creates blank image of the given size and channel numbers

    :param int h: height of image
    :param int w: width of image
    :param int channels: number of color chanels (1 = monochrome)
    """
    h = int(h)
    w = int(w)

    if channels > 1:
        channels = int(channels)
        blank_image = np.zeros((h, w, channels), np.uint8)
    else:
        blank_image = np.zeros((h, w), np.uint8)

    return blank_image


def get_font(size=22):
    """
    Returns TTF font loaded from the system

    :param size: size of the font in px
    :return: TTF font
    """
    fonts = ('consola.ttf',
             'cour.ttf',
             'lucon.ttf',
             'arial.ttf',
             '/usr/share/fonts/truetype/freefont/FreeMono.ttf',
             '/usr/local/lib/X11/fonts/freefont-ttf/FreeMono.ttf',)
    font = None

    for font_name in fonts:
        try:
            font = ImageFont.truetype(font_name, size)
        except:
            ...
        else:
            if font is not None:
                return font

    return ImageFont.load_default()


def get_image_filename(directory, name=None, extension="jpg"):
    """
    Gets full filename for the image based on the current time.

    Checks filesystem and gives only filenames which does not exist yet.

    :param directory: directory where to store the image (can contain date-time
                      tokens '%Y', '%m', '%d', '%H', '%M', '%S', '%f')
    :param name: name of the file to be added to the timestamp
    :param extension: filename extension
    :return: filename
    """
    max_attempts = 1000
    # create basename from date time and the given name
    now = datetime.now()
    filename = now.strftime('%Y%m%d_%H%M%S%f')[:-3]

    if name:
        filename += "_" + str(name)

    for token in ('%Y', '%m', '%d', '%H', '%M', '%S', '%f'):
        directory = directory.replace(token, now.strftime(token))

    # create dir if not exist
    if not os.path.isdir(directory):
        os.makedirs(directory)

    extension = extension.strip('.').lower()

    for num in range(1, max_attempts):
        path = os.path.join(directory, f'{filename}-{num:03}.{extension}')

        if not os.path.isfile(path):
            return path

    raise OSError(f'Filename {filename} already exists in {max_attempts} copies.')


def fit_text_lines(text, width, font_size):
    """
    Wraps individual lines of the multiline text to fit to the given width [px].

    :param text: multiline text
    :param width: max width where we have to fit to
    :param font_size: size of the font
    :return: text
    """
    font = get_font(size=font_size)
    avg_char_width = sum(font.getsize(char)[0] for char in ascii_letters) / len(ascii_letters)
    max_char_count = int((width * 0.95) / avg_char_width)

    lines = text.split('\n')
    wrapped_lines = []

    for line in lines:
        wrapped_lines.append(textwrap.fill(text=line, width=max_char_count))

    return '\n'.join(wrapped_lines)


def write_image_text(image, text, font_size, color, background_color="#000000", margin=8):
    """
    Writes text to the bottom (extends) of the image

    :param image: image to write text to
    :param text: text to write
    :param font_size: font size to use
    :param color: color to use
    :param background_color: text background color
    :param margin: margin of the text are in [px]
    :return: new image
    """
    margin = int(margin)
    spacing = font_size * 0.4
    font = get_font(size=font_size)
    img = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    text = fit_text_lines(text, img.width - 2 * margin, font_size)
    draw = ImageDraw.Draw(img)

    text_width, text_height = draw.multiline_textsize(text, font, spacing)
    text_width = int((2 * margin) + text_width + 0.5)
    text_height = int((2 * margin) + text_height + 0.5)
    text_left = margin
    text_right = text_width - margin
    text_top = int(img.height) + margin
    text_bottom = int(img.height) + text_height - margin

    # Text is wider than image -> extend image on the right side
    if text_right > img.width:
        blank = Image.new('RGB', (text_right - img.width, img.height))
        canvas = Image.new('RGB', (text_right, img.height))
        canvas.paste(img, (0, 0))
        canvas.paste(blank, (img.width, 0))
        img = canvas

    # extend bottom side of the image
    blank = Image.new('RGB', (img.width, text_height))
    canvas = Image.new('RGB', (img.width, img.height + text_height))
    canvas.paste(img, (0, 0))
    canvas.paste(blank, (0, img.height))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle((text_left, text_top, text_right - 1, text_bottom - 1),
                   fill=color_str(background_color))
    draw.multiline_text((text_left, text_top),
                        text,
                        fill=color_str(color),
                        font=font,
                        spacing=spacing)

    # return back the image in OpenCV format
    return cv.cvtColor(np.array(canvas), cv.COLOR_RGB2BGR)


def draw_color_sample(color_list, font_size):
    """
    Creates image with the color samples as filled rectangle and hex color code.

    :param color_list: list of colors
    :param font_size: size of the font
    :return: image with color samples
    """
    margin = 10
    canvas_width, canvas_height = 0, 0
    font = get_font(font_size)

    if isinstance(color_list, str):
        color_list = [color_list]

    for color_name in color_list:
        width, height = font.getsize(color_name)
        canvas_width += margin + width + margin
        canvas_height = max(canvas_height,  margin + height + margin)

    canvas = Image.new('RGB', (canvas_width, canvas_height))
    draw = ImageDraw.Draw(canvas)
    left = 0.0

    for color_name in color_list:
        lightness = sum(color(color_name)) / 3
        text_color = 'white' if lightness < 128 else 'black'
        width, _ = font.getsize(color_name)
        width = margin + width + margin
        draw.rectangle((left, 0.0, left + width, canvas_height), fill=color_name, outline=None, width=0)
        draw.multiline_text((left + margin, margin), color_name, fill=text_color, font=font)
        left += width

    return cv.cvtColor(np.array(canvas), cv.COLOR_RGB2BGR)


def draw_image(canvas, image, surround_box, border_color=None):
    """
    Draws image on the canvas

    :param canvas: canvas
    :param image: image to draw
    :param surround_box: image position where to place it
    :param border_color: color of the border of the image
    :return: canvas with image
    """
    # if image is BGRA or gray convert to BGR
    if len(image.shape) == 3 and image.shape[2] >= 3:
        image = image[:, :, :3]
    else:
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)

    height = image.shape[0]
    width = image.shape[1]
    left, top = surround_box[0], surround_box[3] # by default at left-bottom corner of the box

    if width > canvas.shape[1]:
        width = canvas.shape[1]
        height = int(width / image.shape[1] * height)
        image = cv.resize(image, (width, height), interpolation=cv.INTER_CUBIC)

    if height > canvas.shape[0]:
        height = canvas.shape[0]
        width = int(height / width * image.shape[1])
        image = cv.resize(image, (width, height), interpolation=cv.INTER_CUBIC)

    # if we would overflow when placed bellow
    if (top + height) > canvas.shape[0]:
        # try to put it on the left side
        if surround_box[0] - width > 0:
            left = surround_box[0] - width - 1
            top = surround_box[1]
        # or right side
        elif (surround_box[2] + width + 1) < canvas.shape[1]:
            left = surround_box[2] + 1
            top = surround_box[1]
        # or above
        elif (surround_box[1] - height) > 0:
            left = surround_box[0]
            top = surround_box[1] - height - 1
        # or extend canvas
        else:
            extension_height = top + height - canvas.shape[0] + 1
            extension = np.zeros((extension_height, canvas.shape[1], 3), np.uint8)
            canvas = np.concatenate((canvas, extension), axis=0)

    # check if we overflow horizontally
    if (left + width) > canvas.shape[1]:
        # try to shift it to the left
        if (top >= surround_box[3]) or (top + height <= surround_box[1]):
            left = 0

        if (left + width) > canvas.shape[1]:
            extension_width = left + width - canvas.shape[1] + 1
            extension = np.zeros((canvas.shape[0], extension_width, 3), np.uint8)
            canvas = np.concatenate((canvas, extension), axis=1)

    right = left + width
    bottom = top + height

    # final check of the size one more time
    if right >= canvas.shape[1]:
        width = canvas.shape[1] - left - 1
        height = int(width / image.shape[1] * height)

    if bottom >= canvas.shape[0]:
        height = canvas.shape[0] - top - 1
        width = int(height / image.shape[0] * width)

    if (width != image.shape[1]) or (height != image.shape[0]):
        image = cv.resize(image, (width, height), interpolation=cv.INTER_CUBIC)
        right = left + width
        bottom = top + height

    canvas[top:bottom, left:right] = image

    if border_color is not None:
        border_color = color(border_color)
        canvas = cv.rectangle(canvas, (left, top), (right, bottom), border_color)

    return canvas
