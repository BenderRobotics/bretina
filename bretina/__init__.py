"""
    bretina
    ~~~~~

    Bender Robotics module for visual based testing.

    :copyright: 2019 Bender Robotics
"""

__version__ = '0.0.1'
__all__ = ['VisualTestCase', 'SlidingTextReader', '__version__']

import numpy as np
import cv2 as cv
import os
import time
import difflib
import logging
import itertools
import pytesseract
from bretina.visualtestcase import VisualTestCase
from bretina.slidingtextreader import SlidingTextReader

# Standart color definitions in BGR
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_CYAN = (255, 255, 0)
COLOR_MAGENTA = (255, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_ORANGE = (0, 128, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GRAY = (127, 127, 127)

#: default path to the Tesseract OCR engine installation
TESSERACT_PATH = 'C:\\Tesseract-OCR\\'

#: map of HTML color names to hex codes
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
    Returns list of dominant colors in the image.

    :param img: source image
    :param n: number of colors
    :return: list of (B, G, R) color tuples
    """
    if len(img.shape) == 3 and img.shape[2] == 3:
        pixels = np.float32(img.reshape(-1, 3))
    else:
        pixels = np.float32(img.reshape(-1))

    # k-means clustering
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv.kmeans(pixels, n, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    indexes = np.argsort(counts)[::-1]
    return palette[indexes]


def dominant_color(img):
    """
    Gets the most dominant color in the image.

    :param img: source image
    :return: (B, G, R) color tuple
    """
    return dominant_colors(img)[0]


def active_color(img, bgcolor=None):
    """
    Gets the most dominant color which is not the background color.

    :param img: source image
    :param bgcolor: color of the image background, recognized automatically if `None` or not set
    :return: (B, G, R) color tuple
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
    Mean of each chromatic channel.

    :param img: source image
    """
    channels = img.shape[2] if len(img.shape) == 3 else 1
    pixels = np.float32(img.reshape(-1, channels))
    return np.mean(pixels, axis=0)


def background_color(img):
    """
    Mean color from the 2-pixel width border.

    :param img: source image
    :return: mean color of the image border
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

    :param img: source image
    :return: lightness of the background color
    """
    bgcolor = background_color(img)
    return np.mean(bgcolor)


def color_std(img):
    """
    Get standart deviation of the given image.

    :param img: source image
    :return: standart deviation of the B, G, R color channels
    :rtype: Tuple(std_b, std_g, std_r)
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
    Gets distance metric of two colors as mean absolute value of differences in R, G, B channels.

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
    Gets distance metric of lightness of two colors.

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
    elif type(color) == int or type(color) == float:
        return (color, color, color)
    elif len(color) == 1:
        return (color[0], color[0], color[0])
    elif len(color) == 3:
        return color

    raise Exception('{} not recognized as a valid color definition.'.format(repr(color)))


def color_str(color):
    """
    Converts color from (B, G, R) tuple to '#RRGGBB' string.

    :param color: (B, G, R) sequence
    :type color: tuple
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

    :param img: cv image
    :param box: border box coordinates [left, top, right, bottom]
    :param scale: scaling factor
    :param color: color of the border
    :param int padding: additional border padding
    :param int thickness: thickness of the line
    :return: copy of the given image with the red border
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

    :param img: cv image
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


def text_rows(img, scale, bgcolor=None, min_height=10, limit=0.05):
    """
    Gets number of text rows in the given image.

    :param img: image to process
    :param scale: allows to optimize for different resolution, scale=1 is for font size = 16px.
    :type  scale: float
    :param bgcolor: background color (optional). If not set, the background color is detected automatically.
    :param min_height: minimum height of row in pixels, rows with less pixels are not detected.
    :type  min_height: float
    :param limit: line coverage with pixels of text used for the row detection. Set to lower value for higher sensitivity (0.05 means that 5% of row has to be text pixels)
    :type  limit: float
    :return:
        - count - number of detected text lines
        - regions - list of regions where the text rows are detected, each region is represented with tuple (y_from, y_to)
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
    row_sum = np.sum(opening, axis=1)
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
            if (i - row_start) >= min_height:
                regions.append((row_start, i+1))

    return len(regions), regions


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
    col_sum = np.sum(dilateted, axis=0)
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
    - Specify `bgcolor` parameter - the OCR works fine only for black letters on the light background,
      therefore inversion is done when light letters on dark background are recognized. If bgcolor is not set,
      bretina will try to recognize background automatically and this recognition may fail.
    - Select correct `language`. You may need to install the language data file from
      https://github.com/tesseract-ocr/tesseract/wiki/Data-Files.
    - If you want to recognize only numbers or mathematical expressions, use special language "equ"
      (`language="equ"`).
    - If you expect only limited set of letters, you can use `chars` parameter, e.g. `chars='ABC'` will
      recognize only characters 'A', 'B', 'C'. There are some wildcards:
        - `%d` for integer numbers,
        - `%f` for float numbers,
        - `%w` for letters.
      Wildcards can be combined with additional characters and other wildcards, e.g. `chars='%d%w?'` will
      recognize all integer numbers, all small and capital letters and question mark.
    - Enable `floodfill` parameter for the unification of background.

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


def img_diff(img, template, edges=False, inv=None, bgcolor=None, blank=None):
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
    :return: difference ration of two images, different pixels / template pixels
    """
    scaling = 120.0 / max(template.shape[0:2])

    if scaling > 1:
        img = resize(img.copy(), scaling)
        template = resize(template.copy(), scaling)

    alpha = np.ones(template.shape[0:2], dtype=np.uint8) * 255

    # get alpha channel and mask the template
    if len(template.shape) == 3 and template.shape[2] == 4:
        print(template.shape[2])

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

    _, img_gray = cv.threshold(img_gray, 64, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    _, src_gray = cv.threshold(src_gray, 64, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # get difference
    diff = np.absolute(src_gray.astype(int) - img_gray.astype(int)).astype('uint8')

    # remove small fragments
    kernel = np.ones((5, 5), np.uint8)
    diff = cv.morphologyEx(diff, cv.MORPH_OPEN, kernel)

    # add blanked areas to alpha mask
    if blank is not None:
        assert isinstance(blank, list), '`blank` has to be list'

        # make list if only one area is given
        if len(blank) > 0 and not isinstance(blank[0], list):
            blank = [blank]

        for area in blank:
            alpha[area[1]:area[3], area[0]:area[2]] *= 0

    # mask alpha
    diff = cv.bitwise_and(diff, diff, mask=alpha)

    # sum pixels and get difference ratio
    n_img = np.sum(img_gray)
    n_src = np.sum(src_gray)
    n_alpha = np.sum(alpha)
    n_dif = np.sum(diff)
    ratio = n_dif / n_alpha * 64.0

    #### some temp ploting
    '''
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
    '''
    ####
    return ratio


def recognize_image(img, template, bgcolor=None, white=False):
    """
    Compare given image and template.

    :param image: image where is template searched
    :type  image: cv2 image (b,g,r matrix)
    :param template: template image
    :type  template: cv2 image (b,g,r matrix)
    :param bgcolor: color of template background
    :param white: add white to mask template too
    :type  white: bool
    :return: degree of conformity (0 - 1)
    :rtype: float
    """

    if len(template.shape) == 3:
        colors = 3
    else:
        colors = 1
    mask = None

    # mask backgroung color for template
    if colors == 3 and template.shape[2] == 4:
        if lightness_std(template[:, :, 3]) > 5:
            mask = 255 - template[:, :, 3]
            mask = cv.GaussianBlur(mask, (3, 3), 3)
            mask = cv.equalizeHist(mask)
            ret, mask = cv.threshold(mask, 30, 255, cv.THRESH_BINARY)
            template = cv.bitwise_and(template[:, :, :3], template[:, :, :3], mask=255-mask)
            temp_bg = template.copy()
            if np.mean(background_color(img)) > 100:
                temp_bg[:] = color('white')
            else:
                temp_bg[:] = color('black')
            temp_bg = cv.bitwise_and(temp_bg, temp_bg, mask=mask)
            template = cv.add(template, temp_bg)
        else:
            template = template[:, :, :3]
    if mask is None:
        if bgcolor is None:
            bgcolor = background_color(template)
        b, g, r = color(bgcolor)
        if colors == 3:
            lower = np.maximum((b-15, g-15, r-15), (0, 0, 0))
            upper = np.minimum((b+15, g+15, r+15), (255, 255, 255))
        else:
            lower = np.maximum((b+g+r)/3-20, 0)
            upper = np.minimum((b+g+r)/3+20, 255)
        mask = cv.inRange(template, lower, upper)

    # lightening dark template
    pixels = np.float32(template.reshape(-1, colors))
    if np.mean(np.mean(pixels, axis=0)) < 30:
        template = adjust_gamma(template, 2)
        img = adjust_gamma(img, 7)

    if white:
        if colors == 3:
            mask2 = cv.inRange(template, (250, 250, 250), (255, 255, 255))
        else:
            mask2 = cv.inRange(template, 250, 255)
        mask = cv.add(mask, mask2)

    mask = cv.GaussianBlur(mask, (13, 13), 4)
    mask = cv.equalizeHist(mask)
    ret, mask = cv.threshold(mask, 100, 255, cv.THRESH_BINARY)

    im = img.copy()
    img = img_to_grayscale(img)
    template = img_to_grayscale(template)

    # apply closing to remove small fragments
    kernel = np.ones((3, 3), np.uint8)
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=1)
    template = cv.morphologyEx(template, cv.MORPH_CLOSE, kernel, iterations=1)

    # thresholding
    ret, img = cv.threshold(img, 64, 255, cv.THRESH_TOZERO + cv.THRESH_OTSU)
    ret, template = cv.threshold(template, 64, 255, cv.THRESH_TOZERO + cv.THRESH_OTSU)

    # transfer to edges
    img = cv.Canny(img, 150, 150)
    template = cv.Canny(template, 150, 150)

    # make edges wider with bluring
    img = cv.GaussianBlur(img, (19, 19), 3)
    template = cv.GaussianBlur(template, (19, 19), 3)

    # equalize lightness in both images
    img = cv.equalizeHist(img)
    template = cv.equalizeHist(template)

    # thresholding to make gray pixels white again
    ret, img = cv.threshold(img, 64, 255, cv.THRESH_BINARY)
    ret, template = cv.threshold(template, 64, 255, cv.THRESH_BINARY)

    # match with template
    res = cv.matchTemplate(img, template, cv.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    im = im[max_loc[1]:max_loc[1]+template.shape[0], max_loc[0]:max_loc[0]+template.shape[1]]
    im2 = im.copy()
    if colors == 3:
        colors = dominant_colors(im, 3)
        bgcolor = background_color(im)
        # get index of the bg in pallet as minimum of distance, active color is 0 if
        # it is not background (major color)
        bg_index = np.argmin([rgb_distance(bgcolor, c) for c in colors])
        color_index = 0 if bg_index == 0 else 1
        im2[:] = colors[color_index]
    im = cv.bitwise_and(im, im, mask=255-mask)
    im2 = cv.bitwise_and(im2, im2, mask=mask)
    im = cv.add(im, im2)
    im = img_to_grayscale(im)
    im = cv.Canny(im, 30, 100)
    im = cv.GaussianBlur(im, (15, 15), 8)
    im = cv.equalizeHist(im)
    ret, im = cv.threshold(im, 64, 255, cv.THRESH_BINARY)

    res = cv.matchTemplate(im, template, cv.TM_CCORR_NORMED)
    min_val, max_val2, min_loc, max_loc = cv.minMaxLoc(res)
    return max(max_val, max_val2)


def resize(img, scale):
    """
    Resize image to a given scale

    :param img: source image
    :type  img: cv2 image (b,g,r matrix)
    :param scale: scale between source and target resolution
    :type  scale: float
    :return: scaled image
    :rtype: cv2 image (b,g,r matrix)
    """
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    image_resized = cv.resize(img, (width, height), interpolation=cv.INTER_CUBIC)
    return image_resized


def recognize_animation(images, template, size, scale):
    """
    Recognize image animation and return duty cycles and animation period

    :param images: images with time information
    :type  images: dict {'time': time, 'image': cv2 image}
    :param template: template for animated image }animation in rows/coloms
    :type  template: cv2 image
    :param size: expected size of one separated image
    :type  size: tuple (width, height) int/float
    :param scale: scale between source and target resolution
    :type  scale: float
    :return: image conformity, period conformity (1 - best match, 0 - no match)
    :rtype: float, float
    """
    # load template images (resize and separate)
    blank = None
    templates = separate_animation_template(template, size, scale)

    for x, img_template in enumerate(templates):
        if lightness_std(img_template) < 5:
            blank = x

    read_item = {}
    periods = []

    for x, image in enumerate(images):
        result = []
        # compare template images with captured
        for img_template in templates:
            result.append(recognize_image(image, img_template))
        max_val = max(result)
        if max_val < 0.1:
            i = blank
        else:
            i = result.index(max_val)

        # choose most similar template
        if i not in read_item:
            read_item[i] = [[max_val], 1, x]
            continue

        # identify if image was captured in same period

        if read_item[i][2] != x-1:
            periods.append(x-read_item[i][2])

        read_item[i][0].append(max_val)
        read_item[i][1] += 1
        read_item[i][2] = x

    # identify if image is blinking, compute period conformity
    if len(periods) == 0:
        animation = False
    else:
        animation = True

    # count image conformity
    conf = []
    for x in read_item:
        if x != blank:
            conf.append(np.mean(read_item[x][0]))
    if len(conf) == 0:
        conformity = 0
    else:
        conformity = np.mean(conf)
    return(conformity, animation)


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
        for colum in range(int(width // size[0])):
            templates.append(img[row*size[1]:(1+row)*size[1], colum*size[0]:(1+colum)*size[0]])
    return templates


def equal_str_ratio(a, b, ratio):
    """
    Compares two strings and returns result, allowes
    to define a measure of the sequences’ similarity
    as a float in the range [0, 1].

    Where T is the total number of elements in both sequences,
    and M is the number of matches, this is 2.0*M / T.
    Note that this is 1.0 if the sequences are identical,
    and 0.0 if they have nothing in common.

    :param str a: left side of the string comparision
    :param str b: right side of the string comparision
    :param float ratio: measure of the sequences’ similarity as a float in the range [0, 1]
    :return: True if strings are equal, False if not
    :rtype: bool
    """
    # quick check of the same strings
    if a.strip() == b.strip():
        return True
    # ratio of 1.0 can not be satisfied due to previous condition
    elif ratio == 1.0:
        return False
    else:
        seq = difflib.SequenceMatcher(lambda x: x is ' ', a.strip(), b.strip())
        return seq.ratio() >= ratio


def equal_str(a, b, simchars, ratio=1.0):
    """
    Compares two strings and returns result, allowes to define similar
    characters which are not considered as difference.

    In default version, the trimmed strings are compared (" A " == "A")
    When `simchars` argument is set, more complex algorithm is used and
    some difference are ignored.

    :param str a: left side of the string comparision
    :param str b: right side of the string comparision
    :param list simchars: e.g. ["1il", "0oO"] or None
    :param float ratio: can be used to allow small differences, calculated as (1 - M/T) where T is the average number of elements in both sequences, and M is the number of differences.
    :return: True if strings are equal, False if not
    :rtype: bool
    """
    assert isinstance(a, str), '`a` has to be string, {} given'.format(type(a))
    assert isinstance(b, str), '`b` has to be string, {} given'.format(type(b))

    a = a.strip()
    b = b.strip()

    # quick check of the same strings
    if a == b:
        return True

    # if simchars is not specified, there is no hope for True
    if simchars is None:
        return False
    else:
        if isinstance(simchars, str):
            simchars = [simchars]

        assert all(isinstance(el, str) for el in simchars), '`simchars` argument has to be list of strings, e.g. ["1il", "0oO"]'

        # get possible allowed substitutions
        sims = list()

        for string in simchars:
            sims += list(itertools.permutations(string, 2))

        # get list of differences, filter spaces
        df = filter(lambda x: x not in ['+  ', '-  ', '?  '], difflib.ndiff(a, b))

        prev_char = None
        prev_diff = None
        res = []

        for d in df:
            # '-': char only in A, '+': char only in B
            if d.startswith('-'):
                diff = -1
            elif d.startswith('+'):
                diff = +1
            else:
                diff = 0
            # take the char only (last from the diff)
            char = d[-1]

            # if the difference means that there is different char in A and B (- in one, + in other)
            if (prev_char is not None) and (prev_diff + diff == 0) and (diff != 0):
                # if this combination is in sims, remove last from res
                #  buffer and change new one to valid code (starts with space)
                if (char, prev_char) in sims:
                    res.pop()
                    d = '  ' + char

            prev_char = char
            prev_diff = diff

            if not d.startswith(' '):
                res.append(d)

        if ratio == 1.0:
            return (len(res) == 0)
        else:
            # if ratio is not 1, then it is calculated as complement to ratio of differences over average of input length
            t = (len(a) + len(b)) / 2
            r = 1.0 - (len(res) / t)
            return (r >= ratio)
