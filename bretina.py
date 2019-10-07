import numpy as np
import cv2 as cv
import time

# Standart color definitions in BGR
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_CYAN = (255, 255, 0)
COLOR_MAGENTA = (255, 0, 255)
COLOR_YELLOW = (0, 255, 255)


def crop(img, region):
    '''
    Crops image by region
    '''
    return img[region[1]:(region[1]+region[3]), region[0]:(region[0]+region[2])]


def dominant_colors(img, n=2):
    '''
    Returns list of dominant colors in the image
    '''
    pixels = np.float32(img.reshape(-1, 3))

    # k-means clustering
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv.kmeans(pixels, n, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    return palette


def dominant_color(img):
    return dominant_colors(img)[0]


def active_color(img, bgcolor=None):
    colors = dominant_colors(img, 2)

    # if background color is not specified, determine background from the outline border
    if bgcolor is None:
        bgcolor = background_color(img)

    # get index of the bg in pallet as minimum of distance, item color is the other index
    bg_index = np.argmin([color_distance(bgcolor, c) for c in colors])
    color_index = 0 if bg_index == 1 else 1
    return colors[color_index]


def mean_color(img):
    '''
    Mean of each chromatic channel
    '''
    channels = img.shape[2] if len(img.shape) == 3 else 1
    pixels = np.float32(img.reshape(-1, channels))
    return np.mean(pixels, axis=0)


def background_color(img):
    '''
    Mean color from the 2-pixel width border
    '''
    # take pixels from top, bottom, left and right border lines
    pixels = np.concatenate((np.float32(img[0:2, :].reshape(-1, 3)),
                             np.float32(img[-3:-1, :].reshape(-1, 3)),
                             np.float32(img[:, 0:2].reshape(-1, 3)),
                             np.float32(img[:, -3:-1].reshape(-1, 3))))
    return np.mean(pixels, axis=0)


def color_std(img):
    '''
    Get standart deviation of the given image
    '''
    pixels = np.float32(img.reshape(-1, 3))
    return np.std(pixels, axis=0)


def lightness_std(img):
    '''
    Get standart deviation of the given image lightness information
    '''
    if len(img.shape) == 3 and img.shape[2] == 3:   # if image has 3 channels
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img
    pixels = np.float32(gray.reshape(-1, 1))
    return np.std(pixels, axis=0)


def color_distance(color_a, color_b):
    '''
    Gets distance metric of two colors as mean absolute value of differences in R, G, B channels
    '''
    a = color(color_a)
    b = color(color_b)
    return np.sum(np.absolute(a - b)) / 3.0


def hue_distance(color_a, color_b):
    '''
    Gets distance metric of two colors
    '''
    # make two 1px size images of given colors to have color transformation function available
    img_a = np.zeros((1, 1, 3), np.uint8)
    img_a[0, 0] = color(color_a)

    img_b = np.zeros((1, 1, 3), np.uint8)
    img_b[0, 0] = color(color_b)

    a = cv.cvtColor(img_a, cv.COLOR_BGR2HSV)[0, 0]
    b = cv.cvtColor(img_b, cv.COLOR_BGR2HSV)[0, 0]
    d = np.absolute(a[0] - b[0])

    # because 360 is same as 0 degree, return 360 - d to have smaller angular distance
    if d > 180:
        return 360 - d
    else:
        return d


def lightness_distance(color_a, color_b):
    '''
    Gets distance metric of lightness of two colors
    '''
    img_a = np.zeros((1, 1, 3), np.uint8)
    img_a[0, 0] = color(color_a)

    img_b = np.zeros((1, 1, 3), np.uint8)
    img_b[0, 0] = color(color_b)

    a = cv.cvtColor(img_a, cv.COLOR_BGR2LAB)[0, 0]
    b = cv.cvtColor(img_b, cv.COLOR_BGR2LAB)[0, 0]
    return np.absolute(a[0] - b[0])


def color(color):
    '''
    Converts hex string color "#RRGGBB" to tuple representation (B, G, R)
    '''
    if type(color) == str:
        # convert from hex color representation
        h = color.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (4, 2, 0))
    else:
        return color


def color_str(color):
    '''
    Converts color from BGR tuple to string notation
    '''
    if type(color) == str:
        return color
    else:
        return hex(int(round(color[2]) * 255*255 +
                       round(color[1]) * 255
                       round(color[0]))).replace('0x', '#')


def border(img, region):
    '''
    Draws red border around specified region
    '''
    left_top = (region[0], region[1])
    right_bottom = (region[0]+region[2], region[1]+region[3])

    if len(img.shape) != 3 or img.shape[2] == 1:   # if image has 3 channels
        figure = cv.cvtColor(img.copy(), cv.COLOR_GRAY2BGR)
    else:
        figure = img.copy()

    return cv.rectangle(figure, left_top, right_bottom, COLOR_RED)
