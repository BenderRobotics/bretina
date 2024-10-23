"""
    bretina
    ~~~~~

    Bender Robotics module for visual based testing.

    :copyright: 2019 Bender Robotics
"""

__version__ = '0.0.1'
__all__ = ['VisualTestCase', '__version__']

import numpy as np
import cv2 as cv
import os
import time
import logging
import pytesseract
from bretina.visualtestcase import VisualTestCase

# Standart color definitions in BGR
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_CYAN = (255, 255, 0)
COLOR_MAGENTA = (255, 0, 255)
COLOR_YELLOW = (0, 255, 255)


TESSERACT_PATH = "C:\\Tesseract-OCR\\"

def dominant_colors(img, n=2):
    """
    Returns list of dominant colors in the image.

    :param img: source image
    :param n: number of colors
    :return: list of (B, G, R) color tuples
    """
    pixels = np.float32(img.reshape(-1, 3))

    # k-means clustering
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv.kmeans(pixels, n, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    return palette


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
    colors = dominant_colors(img, 2)

    # if background color is not specified, determine background from the outline border
    if bgcolor is None:
        bgcolor = background_color(img)

    # get index of the bg in pallet as minimum of distance, item color is the other index
    bg_index = np.argmin([color_distance(bgcolor, c) for c in colors])
    color_index = 0 if bg_index == 1 else 1
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
    # take pixels from top, bottom, left and right border lines
    pixels = np.concatenate((np.float32(img[0:2, :].reshape(-1, 3)),
                             np.float32(img[-3:-1, :].reshape(-1, 3)),
                             np.float32(img[:, 0:2].reshape(-1, 3)),
                             np.float32(img[:, -3:-1].reshape(-1, 3))))
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
    img = img_to_grayscale(img)
    pixels = np.float32(gray.reshape(-1, 1))
    return np.std(pixels, axis=0)


def color_distance(color_a, color_b):
    """
    Gets distance metric of two colors as mean absolute value of differences in R, G, B channels.

    :param color_a: string or tuple representation of the color A
    :param color_b: string or tuple representation of the color B
    :return: mean distance in RGB
    :rtype: float
    """
    a = color(color_a)
    b = color(color_b)
    return np.sum(np.absolute(a - b)) / 3.0


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
    return np.absolute(a[0] - b[0])


def color(color):
    """
    Converts hex string color "#RRGGBB" to tuple representation (B, G, R).

    :param color: #RRGGBB color string
    :type color: str
    :return: (B, G, R) tuple
    """
    if type(color) == str:
        # convert from hex color representation
        h = color.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (4, 2, 0))
    else:
        return color


def color_str(color):
    """
    Converts color from (B, G, R) tuple to "#RRGGBB" string.

    :param color: (B, G, R) sequence
    :type color: Tuple(B, G, R)
    """
    if type(color) == str:
        return color
    else:
        return "#{r:02x}{g:02x}{b:02x}".format(r=color[2], g=color[1], b=color[0])


def draw_border(img, box):
    """
    Draws red border around specified region

    :param img: cv image
    :param box: border box coordinates
    :type box: [left, top, right, bottom]
    :return: copy of the given image with the red border
    """
    figure = img.copy()
    return cv.rectangle(figure, (region[0], region[1]), (region[2], region[3]), COLOR_RED)


def img_to_grayscale(img):
    """
    Converts image to gray-scale.

    :param img: cv image
    :return: image converted to grayscale
    """
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        return img


def text_rows(img, scale, bgcolor=None, min_height=10, limit=0.05):
    """
    Gets number of text rows in the given image.

    :param img: image to process
    :param scale: allows to optimize for different resolution, scale=1 is for font size = 16px.
    :param bgcolor: background color (optional). If not set, the background color is detected automatically.
    :param min_height: minimum height of row in pixels, rows with less pixels are not detected.
    :param limit: line coverage with pixels of text used for the row detection. Set to lower value for higher sensitivity (0.05 means that 5% of row has to be text pixels)
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


def get_rectification(img, scale, chessboard_size, display_size, border=0):
    """
    Get rectification parameters from captured chessboard calibration image.

    :param img: acquired image of chessboard on display
    :type img: cv2 image (b,g,r matrix)
    :param scale: scale between camera resolution and real display
    :type scale: float
    :param chessboard_size: size of chessboard (number of white/black pairs)
    :type chessboard_size: [width, height] int/float
    :param display_size: display size (in px)
    :type display_size: [width, height] int/float
    :param border: border (in pixels) around cropped display
    :type border: int
    :return:
        - dstmaps: (x, y) distortion remap matrix,
        - transformation: perspective transformation & crop matrix,
        - final resolution.
    """

    # termination criteria - epsilon reached and number of iterations
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # size of chessboard (from white/black pairs to rows and cols) and image
    w_ch, h_ch = int(chessboard_size[0]*2 - 3), int(chessboard_size[1]*2 - 3)
    # image size (columns, rows)
    size = (img.shape[1], img.shape[0])
    # prepare object points
    object_points = np.zeros((w_ch*h_ch, 3), np.float32)
    object_points[:, :2] = np.mgrid[0:h_ch, 0:w_ch].T.reshape(-1, 2)

    # find the chessboard corners
    img_gray = img_to_grayscale(img)
    ret, corners = cv.findChessboardCorners(img_gray, (h_ch, w_ch), None)

    # raise exception and terminate if chessboard is not found in the image
    if not ret:
        raise Exception("Image of chessboard [{}x{}] not found in the image.".format(w_ch, h_ch))

    # find corners with higher precision
    corners_sub = cv.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)

    # - Find intrinsic and extrinsic parameters from view of a calibration pattern,
    # - get new camera matrix based on found parameters,
    # - computes the un-distortion and rectification transformation map,
    # - and apply a geometrical transformation to an image.
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera([object_points], [corners_sub], size, None, None)
    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, size, 1, size)
    dstmaps = cv.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, size, cv.CV_32FC1)
    img_remaped = cv.remap(img, dstmaps[0], dstmaps[1], cv.INTER_LINEAR)
    img_gray = img_to_grayscale(img_remaped)

    # find the chessboard corners in the transformed image
    ret, corners = cv.findChessboardCorners(img_gray, (h_ch, w_ch), None)
    corners_sub = cv.cornerSubPix(img_gray, corners, (3, 3), (-1, -1), criteria)
    final_resolution = (int(display_size[0]*scale + border*2),
                        int(display_size[1]*scale + border*2))
    # corner points found in the chessboard image
    source_points = np.float32([corners_sub[-1, 0],
                                corners_sub[h_ch-1, 0],
                                corners_sub[-h_ch, 0],
                                corners_sub[0, 0]])
    # expected coordinations of the chessboard corners
    ch_b = display_size[0] * scale / chessboard_size[0] + border
    dest_points = np.float32([[ch_b, final_resolution[1]-ch_b],
                              [final_resolution[0]-ch_b, final_resolution[1]-ch_b],
                              [ch_b, ch_b],
                              [final_resolution[0]-ch_b, ch_b]])
    transformation = cv.getPerspectiveTransform(source_points, dest_points)

    return (dstmaps, transformation, final_resolution)


def rectify(img, dstmaps, transformation, resolution):
    """
    Applies distortion correction and perspective transformation and returns image
    with desired resolution. Use get_transformation() to get necessary parameters.

    :param img: acquired image
    :type img: cv2 image (b,g,r matrix)
    :param dstmaps: x and y rectification maps
    :param transformation: perspective transformation matrix
    :param resolution: tuple of final resolution (width, height)
    :return: undistorted and cropped image
    :rtype: cv2 image (b,g,r matrix)
    """
    img_remaped = cv.remap(img, dstmaps[0], dstmaps[1], cv.INTER_LINEAR)
    img_final = cv.warpPerspective(img_remaped, transformation, resolution)
    return img_final


def color_calibration(chessboard_img, chessboard_size, r, g, b):
    """
    Create calibration parameters from displayed chessboard and red, green
    and blue screen to rgb color calibration and histogram color calibration

    :param chessboard_img: acquired image of chessboard on display
    :type chessboard_img: cv2 image (b,g,r matrix)
    :param chessboard_size: size of chessboard (number of white/black pairs)
    :type chessboard_size: (width, height) int/float
    :param r: acquired image of red screen
    :type r: cv2 image (b,g,r matrix)
    :param g: acquired image of green screen
    :type g: cv2 image (b,g,r matrix)
    :param b: acquired image of blue screen
    :type b: cv2 image (b,g,r matrix)
    """
    # BGR screen cropped to better function
    b = b[50:-50, 50:-50]
    g = g[50:-50, 50:-50]
    r = r[50:-50, 50:-50]

    hist_bins = 255
    hist_range = (0, 255)
    hist_threshold = 10

    crb, prb = np.histogram(np.ma.masked_less(r[:, :, 0], hist_threshold), bins=hist_bins, range=hist_range)
    crg, prg = np.histogram(np.ma.masked_less(r[:, :, 1], hist_threshold), bins=hist_bins, range=hist_range)
    crr, prr = np.histogram(np.ma.masked_less(r[:, :, 2], hist_threshold), bins=hist_bins, range=hist_range)
    cbb, pbb = np.histogram(np.ma.masked_less(b[:, :, 0], hist_threshold), bins=hist_bins, range=hist_range)
    cbg, pbg = np.histogram(np.ma.masked_less(b[:, :, 1], hist_threshold), bins=hist_bins, range=hist_range)
    cbr, pbr = np.histogram(np.ma.masked_less(b[:, :, 2], hist_threshold), bins=hist_bins, range=hist_range)
    cgb, pgb = np.histogram(np.ma.masked_less(g[:, :, 0], hist_threshold), bins=hist_bins, range=hist_range)
    cgg, pgg = np.histogram(np.ma.masked_less(g[:, :, 1], hist_threshold), bins=hist_bins, range=hist_range)
    cgr, pgr = np.histogram(np.ma.masked_less(g[:, :, 2], hist_threshold), bins=hist_bins, range=hist_range)
    cbb = np.where(0.95*max(cbb) < cbb, 0.95*max(cbb), cbb)
    cgg = np.where(0.95*max(cgg) < cgg, 0.95*max(cgg), cgg)
    crr = np.where(0.95*max(crr) < crr, 0.95*max(crr), crr)

    bmin = min(np.argmax(crb), np.argmax(cgb))
    gmin = min(np.argmax(crg), np.argmax(cbg))
    rmin = min(np.argmax(cbr), np.argmax(cbg))

    if cbb[-1] > 10:
        bmax = np.argmax(cbb)
    else:
        bmax = 255
    if cgg[-1] > 10:
        gmax = np.argmax(cgg)
    else:
        gmax = 255
    if crr[-1] > 10:
        rmax = np.argmax(crr)
    else:
        rmax = 255
    bmean = int(np.mean(chessboard_img[:, :, 0]))
    gmean = int(np.mean(chessboard_img[:, :, 1]))
    rmean = int(np.mean(chessboard_img[:, :, 2]))

    bp = [bmin, bmean, bmax]
    gp = [gmin, gmean, gmax]
    rp = [rmin, rmean, rmax]
    histogram_calibration_data = [bp, gp, rp]

    b = calibrate_hist(b, histogram_calibration_data)
    g = calibrate_hist(g, histogram_calibration_data)
    r = calibrate_hist(r, histogram_calibration_data)
    chessboard_img = calibrate_hist(chessboard_img, histogram_calibration_data)

    Bi = [np.mean(b[:, :, 0]), np.mean(b[:, :, 1]), np.mean(b[:, :, 2])]
    Gi = [np.mean(g[:, :, 0]), np.mean(g[:, :, 1]), np.mean(g[:, :, 2])]
    Ri = [np.mean(r[:, :, 0]), np.mean(r[:, :, 1]), np.mean(r[:, :, 2])]

    chb = cv.GaussianBlur(chessboard_img, (9, 9), 10)
    chb = cv.GaussianBlur(chb, (5, 5), 20)
    h, w = chb.shape[:2]  # image size
    ws = int(w/chessboard_size[0])
    hs = int(h/chessboard_size[1])

    i = 0
    wb = 0
    wg = 0
    wr = 0
    kb = 0
    kg = 0
    kr = 0

    for x in range(int(chessboard_size[0])):
        for y in range(int(chessboard_size[1])):
            # calculation of center for white and black square in black/white pair
            x1 = int(ws*x+ws/4)
            x2 = int(ws*x+ws*3/4)
            y1 = int(hs*y+hs/4)
            y2 = int(hs*y+hs*3/4)

            wb += ((int(chb[y1, x1, 2])+int(chb[y2, x2, 2]))/2)
            wg += ((int(chb[y1, x1, 2])+int(chb[y2, x2, 2]))/2)
            wr += ((int(chb[y1, x1, 2])+int(chb[y2, x2, 2]))/2)
            kb += ((int(chb[y1, x2, 0])+int(chb[y2, x1, 0]))/2)
            kg += ((int(chb[y1, x2, 1])+int(chb[y2, x1, 1]))/2)
            kr += ((int(chb[y1, x2, 2])+int(chb[y2, x1, 2]))/2)
            i += 1

    Wi = [wb/(i-1), wg/(i-1), wr/(i-1)]
    Ki = [kb/(i-1), kg/(i-1), kr/(i-1)]
    I = [Bi, Gi, Ri, Wi, Ki, Wi, Ki, Wi, Ki]

    Bt = [255, 0, 0]
    Gt = [0, 255, 0]
    Rt = [0, 0, 255]
    Wt = [255, 255, 255]
    Kt = [0, 0, 0]
    T = [Bt, Gt, Rt, Wt, Kt, Wt, Kt, Wt, Kt]

    invI = np.linalg.pinv(I)

    rgb_calibration_data = np.dot(invI, T)
    return (histogram_calibration_data, rgb_calibration_data)


def calibrate_hist(img, histogram_calibration_data):
    """
    Histogram calibration on acquired image

    :param img: acquired image
    :type img: cv2 image (b,g,r matrix)
    :param histogram_calibration_data: data for histogram calibration
    :type histogram_calibration_data: calibratin matrix 3x3
    :return: histogram calibrated image
    :rtype: cv2 image (b,g,r matrix)
    """

    imgo = img.copy()
    for x in range(0, 3):
        ar = img[:, :, x]
        p = histogram_calibration_data[x]
        # set mean value of color from chessboar image (black/white image) to center of histogram
        #(format init16 for possible calculation out of space uinit8)
        k = np.where(ar<p[1],np.array(ar*(127/p[1])),np.array((ar-p[1])*(127/(255-p[1]))+127)).astype('int16')
        # stretching the histogram
        imgo[:, :, x] = np.clip((k-p[0])*(255/(p[2]-p[0])),0,255).astype('uint8')
    return imgo


def calibrate_rgb(img, rgb_calibration_data):
    """
    RGB color calibration on acquired image

    :param img: acquired image
    :type img: cv2 image (b,g,r matrix)
    :param rgb_calibration_data: data for RGB color calibration
    :type rgb_calibration_data: calibratin matrix 3x3
    :return: RGB color calibrated image
    :rtype: cv2 image (b,g,r matrix)
    """

    imgo = img.copy()
    bb = (img[:, :, 0])
    bg = (img[:, :, 1])
    br = (img[:, :, 2])
    gb = (img[:, :, 0])
    gg = (img[:, :, 1])
    gr = (img[:, :, 2])
    rb = (img[:, :, 0])
    rg = (img[:, :, 1])
    rr = (img[:, :, 2])

    bo = bb*rgb_calibration_data[0, 0]+bg*rgb_calibration_data[1, 0]+br*rgb_calibration_data[2, 0]
    go = gb*rgb_calibration_data[0, 1]+gg*rgb_calibration_data[1, 1]+gr*rgb_calibration_data[2, 1]
    ro = rb*rgb_calibration_data[0, 2]+rg*rgb_calibration_data[1, 2]+rr*rgb_calibration_data[2, 2]
    imgo[:, :, 0] = np.clip(bo, 0, 255).astype('uint8')
    imgo[:, :, 1] = np.clip(go, 0, 255).astype('uint8')
    imgo[:, :, 2] = np.clip(ro, 0, 255).astype('uint8')
    return imgo


def crop(img, box, scale, border=0):
    """
    Crops image with given box borders.

    :param img: source image
    :type img: cv2 image (b,g,r matrix)
    :param box: boundaries of intrested area
    :type box: [left, top, right, bottom]
    :param scale: target scaling
    :type scale: float
    :param border: border (in pixels) around cropped display
    :type border: int
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


def read_text(img, language='eng', multiline=False):
    """
    Reads text from image with use of the Tesseract ORC engine.
    Install Tesseract OCR engine (https://github.com/tesseract-ocr/tesseract/wiki) and set the
    path to the installation to `bretina.TESSERACT_PATH` ('C:\\Tesseract-OCR\\' for instance).

    :param img: image of text
    :type img: cv2 image (b,g,r matrix)
    :param language: language of text (use three letter ISO code
        https://github.com/tesseract-ocr/tesseract/wiki/Data-Files)
    :type language: string
    :return: read text
    :rtype: string
    """
    # Options of Tesseract page segmentation mode:
    TESSERACT_PAGE_SEGMENTATION_MODE_0 = '0'        # Orientation and script detection (OSD) only.
    TESSERACT_PAGE_SEGMENTATION_MODE_1 = '1'        # Automatic page segmentation with OSD.
    TESSERACT_PAGE_SEGMENTATION_MODE_2 = '2'        # Automatic page segmentation, but no OSD, or OCR. (not implemented)
    TESSERACT_PAGE_SEGMENTATION_MODE_3 = '3'        # Fully automatic page segmentation, but no OSD. (Default)
    TESSERACT_PAGE_SEGMENTATION_MODE_4 = '4'        # Assume a single column of text of variable sizes.
    TESSERACT_PAGE_SEGMENTATION_MODE_5 = '5'        # Assume a single uniform block of vertically aligned text.
    TESSERACT_PAGE_SEGMENTATION_MODE_6 = '6'        # Assume a single uniform block of text.
    TESSERACT_PAGE_SEGMENTATION_MODE_7 = '7'        # Treat the image as a single text line.
    TESSERACT_PAGE_SEGMENTATION_MODE_8 = '8'        # Treat the image as a single word.
    TESSERACT_PAGE_SEGMENTATION_MODE_9 = '9'        # Treat the image as a single word in a circle.
    TESSERACT_PAGE_SEGMENTATION_MODE_10 = '10'      # Treat the image as a single character.
    TESSERACT_PAGE_SEGMENTATION_MODE_11 = '11'      # Sparse text. Find as much text as possible in no particular order.
    TESSERACT_PAGE_SEGMENTATION_MODE_12 = '12'      # Sparse text with OSD.
    TESSERACT_PAGE_SEGMENTATION_MODE_13 = '13'      # Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.

    # Options of Tesseract OCR engine mode:
    TESSERACT_OCR_ENGINE_MODE_0 = '0'               # Legacy engine only.
    TESSERACT_OCR_ENGINE_MODE_1 = '1'               # Neural nets LSTM engine only.
    TESSERACT_OCR_ENGINE_MODE_2 = '2'               # Legacy + LSTM engines.
    TESSERACT_OCR_ENGINE_MODE_3 = '3'               # Default, based on what is available.

    # set path to tesseract OCR engine
    pytesseract.pytesseract.tesseract_cmd = os.path.join(TESSERACT_PATH, 'tesseract.exe')

    if background_lightness(img) < 120:
        img = 255 - img

    ret, img = cv.threshold(img, 200, 200, cv.THRESH_TRUNC)
    img = cv.GaussianBlur(img, (3, 3), 2)

    if multiline:
        psm_opt = TESSERACT_PAGE_SEGMENTATION_MODE_3
    else:
        psm_opt = TESSERACT_PAGE_SEGMENTATION_MODE_7

    config = "-l {lang} --oem {oem} --psm {psm}".format(lang=language,
                                                        oem=TESSERACT_OCR_ENGINE_MODE_3,
                                                        psm=psm_opt)
    text = pytesseract.image_to_string(img, config=config)
    return text


def recognize_image(img, template):
    """
    Compare given image and template.

    :param image: image where is template searched
    :type image: cv2 image (b,g,r matrix)
    :param template: template image
    :type image: cv2 image (b,g,r matrix)
    :return: degree of conformity
    :rtype: float
    """

    image = cv.GaussianBlur(image, (7, 7), 2)
    template = cv.GaussianBlur(template, (7, 7), 2)
    ret,image = cv.threshold(image,200,200,cv.THRESH_TRUNC)
    ret,template = cv.threshold(template,200,200,cv.THRESH_TRUNC)
    image = cv.GaussianBlur(image, (5, 5), 4)
    template = cv.GaussianBlur(template, (5, 5), 4)
    # transfer to edges
    img = cv.Canny(img, 150, 150)
    template = cv.Canny(template, 150, 150)
    res = cv.matchTemplate(img, template, cv.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    return max_val


def resize(img, scale):
    """
    Resize image to a given scale

    :param img: source image
    :type img: cv2 image (b,g,r matrix)
    :param scale: scale between source and target resolution
    :type scale: float
    :return: scaled image
    :rtype: cv2 image (b,g,r matrix)
    """
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    image_resized = cv.resize(img, (width, height), interpolation=cv.INTER_CUBIC)
    return image_resized

