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
    Mean color from the 2-pixel width border.
    '''
    # take pixels from top, bottom, left and right border lines
    pixels = np.concatenate((np.float32(img[0:2, :].reshape(-1, 3)),
                             np.float32(img[-3:-1, :].reshape(-1, 3)),
                             np.float32(img[:, 0:2].reshape(-1, 3)),
                             np.float32(img[:, -3:-1].reshape(-1, 3))))
    return np.mean(pixels, axis=0)


def background_lightness(img):
    '''
    Lightness of the background color.
    '''
    bgcolor = background_color(img)
    return np.mean(bgcolor)


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
    img = img_to_grayscale(img)
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

    # because 180 is same as 0 degree, return 180-d to have shortest angular distance
    if d > 90:
        return 180 - d
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
    Converts color from (B, G, R) tuple to "#RRGGBB" string
    '''
    if type(color) == str:
        return color
    else:
        return "#{r:02x}{g:02x}{b:02x}".format(r=color[2], g=color[1], b=color[0])


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


def img_to_grayscale(img):
    '''
    Converts image to gray-scale.
    :param img: cv image
    :return: image converted to grayscale
    '''
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        return img


def text_rows(img, scale, bgcolor=None, min_height=10, limit=0.05):
    '''
    Gets number of text rows in the given image.

    :param img: image to process
    :param scale: allows to optimize for different resolution, scale=1 is for font size = 16px.
    :param bgcolor: background color (optional). If not set, the background color is detected automatically.
    :param min_height: minimum height of row in pixels, rows with less pixels are not detected.
    :param limit: line coverage with pixels of text used for the row detection. Set to lower value for higher sensitivity (0.05 means that 5% of row has to be text pixels)
    :return:
        - count - number of detected text lines
        - regions - list of regions where the text rows are detected, each region is represented with tuple (y_from, y_to)
    '''
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
    '''
    Gets regions of text cols in the given image.

    :param img: image to process
    :param scale: allows to optimize for different resolution, scale=1 is for font size = 16px.
    :param bgcolor: background color (optional). If not set, the background color is detected automatically.
    :param min_width: minimum width of column in pixels, rows with less pixels are not detected.
    :param limit: col coverage with pixels of text used for the column detection. Set to lower value for higher sensitivity (0.05 means that 5% of row has to be text pixels).
    :return:
        - count - number of detected text columns
        - regions - list of regions where the text columns are detected, each region is represented with tuple (x_from, x_to)
    '''
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


def get_transformation(img, scale, chessboard_size, display_size, border=0.0):
    """
    create calibration parameters from displayed chessboard to undistorted and crop acquired images

    :param img: acquired image of chessboard on display
    :type img: cv2 image (b,g,r matrix)
    :param chessboard_size: size of chessboard (number of white/black pairs)
    :type chessboard_size: [width, height] int/float
    :param display_size: display size (in px)
    :type display_size: [width, height] int/float
    :param scale: scale between camera resolution and real display
    :type scale: int
    :param border: border (in pixels) around cropped display
    :type border: int
    :return: parameters of undistorted matrix, crop matrix and final resolution
    :rtype: tuple of array
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
    undistort_maps = cv.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, size, cv.CV_32FC1)
    img_remaped = cv.remap(img, undistort_maps[0], undistort_maps[1], cv.INTER_LINEAR)
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
    perspective_transformation = cv.getPerspectiveTransform(source_points, dest_points)

    return (undistort_maps, perspective_transformation, final_resolution)


def crop(img, calibration_data):
    """
    undistorted and crop acquired image

    :param img: acquired image
    :type img: cv2 image (b,g,r matrix)
    :param calibration_data: parameters of undistorted matrix, crop matrix and final resolution
    :type calibration_data: tuple of array
    :return: undistorted and cropped image
    :rtype: cv2 image (b,g,r matrix)
    """

    # undistort
    dst = cv.remap(img, calibration_data[0][0], calibration_data[0][1], cv.INTER_LINEAR)
    # crop
    fin = cv.warpPerspective(dst, calibration_data[1], calibration_data[2])
    return (fin)


def color_calibration(chessboard_img, chessboard_size, r, g, b):
    """
    create calibration parameters from displayed chessboard and red, green and blue screen to rgb color calibration and histogram color calibration

    :param chessboard_img: acquired image of chessboard on display
    :type chessboard_img: cv2 image (b,g,r matrix)
    :param chessboard_size: size of chessboard (number of white/black pairs)
    :type chessboard_size: [width, height] int/float
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
    histgram calibration on acquired image

    :param img: acquired image
    :type img: cv2 image (b,g,r matrix)
    :param histogram_calibration_data: data for histogram calibration
    :type histogram_calibration_data: calibratin matrix 3x3
    :return: histgram calibrated image
    :rtype: cv2 image (b,g,r matrix)
    """

    imgo = img.copy()
    for x in range(0, 3):
        ar = img[:, :, x]
        p = histogram_calibration_data[x]
        k = np.where(ar<p[1],np.array(ar*(127/p[1])),np.array((ar-p[1])*(127/(255-p[1]))+127)).astype('uint8')
        k = np.where(k<p[0], 0, np.array(k-p[0]))
        k = np.where(k <(p[2]-p[0]), np.array(k*(255/(p[2]-p[0]))), 255).astype('uint8')
        imgo[:, :, x] = k
    return imgo


def calibrate_rgb(img, rgb_calibration_data):
    """
    rgb color calibration on acquired image

    :param img: acquired image
    :type img: cv2 image (b,g,r matrix)
    :param rgb_calibration_data: data for rgb color calibration
    :type rgb_calibration_data: calibratin matrix 3x3
    :return: rgb color calibrated image
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
    bo = np.where(bo < 0, 0, bo)
    go = np.where(go < 0, 0, go)
    ro = np.where(ro < 0, 0, ro)
    bo = np.where(bo > 255, 255, bo).astype('uint8')
    go = np.where(go > 255, 255, go).astype('uint8')
    ro = np.where(ro > 255, 255, ro).astype('uint8')

    imgo[:, :, 0] = bo
    imgo[:, :, 1] = go
    imgo[:, :, 2] = ro
    return imgo


def item_crop_box(img, item, scale, border):
    startX = int(item["box"][0] * scale + border)
    startY = int(item["box"][1] * scale + border)
    endX = int(item["box"][2] * scale + border)
    endY = int(item["box"][3] * scale + border)
    roi = img[startY:endY, startX:endX]
    return (roi)


def read_text(img, language, text_line='singleline'):
    """
    read text from image

    :param img: image cropped around text
    :type img: cv2 image (b,g,r matrix)
    :param language: language of text (use three letter ISO code https://github.com/tesseract-ocr/tesseract/wiki/Data-Files)
    :type language: string
    :return: read text
    :rtype: string
    """
    pytesseract.pytesseract.tesseract_cmd = r"C:\Tesseract-OCR\tesseract.exe"

    if background_lightness(img) < 120:
        img = 255 - img

    gray = cv.medianBlur(img, 3)
    img = cv.GaussianBlur(img, (3, 3), 2)
    # in order to apply Tesseract v4 to OCR text we must supply
    # (1) a language, (2) an OEM flag of 4 (0 - 3), indicating that the we
    # wish to use the LSTM neural net model for OCR, and finally
    # (3) an OEM value, in this case, 7 which implies that we are
    # treating the ROI as a single line of text
    if text_line == 'singleline':
        config = ("-l " + language + " --oem 3 --psm 7")
    else:
        config = ("-l " + language + " --oem 3 --psm 3")
    text = pytesseract.image_to_string(img, config=config)

    return (text)


def recognize_image(self, item, img=None, image_matching=0.3, overwrite=True):
    """
    compare image from box at screen with artwork

    :param item: boundaries of text in screen (in resolution of display) or "none" value if input screen is cropped around text, array of artwork images name
    :type item: dict ({"box": [width left border, height upper border, width right border, height lower border], "images": array of image names}
    :param img: acquired image
    :type img: cv2 image (b,g,r matrix)
    :param image_matching: the boundary for recognizing the picture's conformity with the template (1 is the same picture, 0 is no match)
    :type image_matching: float 0 - 1
    :param overwrite: overwrite image in self.img
    :type overwrite: bool
    :return: recognized image
    :rtype: string
    """
    if overwrite:
        self.__is_img_source(img)
        inv = self.img.copy()
    else:
        inv = img.copy()

    bgr = self.background_color(item, inv)
    c = (bgr[0]+bgr[1]+bgr[2])/3

    if c < 120:
        inv = 255-inv
    img_gray = cv.cvtColor(inv, cv.COLOR_BGR2GRAY)
    roi = img_gray.copy()

    if item["box"] is not None:
        boundaries = self.__boundaries_from_box(item)
        # extract the actual padded ROI
        roi = img_gray[boundaries[0]:boundaries[1],
                       boundaries[2]:boundaries[3]]

    imgBlurred = cv.GaussianBlur(roi, (5, 5), 0)        # smoothing
    # transfer to edges
    edges = cv.Canny(imgBlurred, 150, 200)
    edges = cv.GaussianBlur(edges, (5, 5), 0)           # smoothing
    maximum = []
    for (items) in item["images"]:
        icon = cv.imread(self.path+items, 0)
        imgBlurred2 = cv.GaussianBlur(icon, (5, 5), 0)
        edg = cv.Canny(imgBlurred2, 150, 200)
        edg = cv.GaussianBlur(edg, (5, 5), 0)
        res = cv.matchTemplate(edges, edg, cv.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        maximum.append(max_val)

    val = max(maximum)
    pos = maximum.index(val)
    if val > image_matching:
        a = item["images"][pos]
    else:
        a = None
    return (a)


def recognize_image_low_contrast(self, item, img=None, image_matching=0.3, overwrite=True):
    """
    compare image from box at screen with artwork, it is assumption that image from box has low contrast and can't be transformed to edges

    :param item: boundaries of text in screen (in resolution of display) or "none" value if input screen is cropped around text, array of artwork images name
    :type item: dict ({"box": [width left border, height upper border, width right border, height lower border], "images": array of image names}
    :param img: acquired image
    :type img: cv2 image (b,g,r matrix)
    :param image_matching: the boundary for recognizing the picture's conformity with the template (1 is the same picture, 0 is no match)
    :type image_matching: float 0 - 1
    :param overwrite: overwrite image in self.img
    :type overwrite: bool
    :return: recognized image
    :rtype: string
    """
    if overwrite:
        self.__is_img_source(img)
        inv = self.img.copy()
    else:
        inv = img.copy()

    bgr = self.background_color(item, inv)
    c = (bgr[0]+bgr[1]+bgr[2])/3

    if c < 120:
        inv = 255-inv

    img_gray = cv.cvtColor(inv, cv.COLOR_BGR2GRAY)
    roi = img_gray.copy()

    if item["box"] is not None:
        boundaries = self.__boundaries_from_box(item)
        # extract the actual padded ROI
        roi = img_gray[boundaries[0]:boundaries[1],
                       boundaries[2]:boundaries[3]]

    imgBlurred = cv.GaussianBlur(roi, (5, 5), 0)        # smoothing
    # transfer to edges
    edges = cv.Canny(imgBlurred, 150, 200)
    edges = cv.GaussianBlur(edges, (5, 5), 0)           # smoothing
    maximum = []

    for (items) in item["images"]:
        icon = cv.imread(self.path+items, 0)
        res = cv.matchTemplate(edges, icon, cv.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        maximum.append(max_val)

    val = max(maximum)
    pos = maximum.index(val)

    if val > image_matching:
        a = item["images"][pos]
    else:
        a = None
    return (a)


def clear_image(self):
    """
    clear acquired image saved in local memory
    """
    self.img = None


def return_image(self):
    """
    return acquired image saved in local memory
    """
    return(self.img)


def show_image(self):
    """
    show acquired image saved in local memory
    """
    cv.imshow("img", self.img)
    cv.waitKey()
    cv.destroyAllWindows()


def __is_img_source(self, img):
    """
    load image from different sources

    try if is set image in function (img) or if is image in local memory (self.img) or is necessary to acquire new image

    :param img: acquired image
    :type img: cv2 image (b,g,r matrix)
    """
    if img is not None:
        self.img = img
    else:
        if self.img is not None:
            pass
        else:
            if self.cam is not None:
                self.img = self.__load_img()
            else:
                self.logger.error(
                    'No camera or image source', extra=self.log_args)
                raise SystemExit


def __load_img(self):
    """
    acquire and calibrate image from camera

    :return: calibrated image
    :rtype: cv2 image (b,g,r matrix)
    """
    img = self.cam.acquire_image()
    img = self.crop(img)
    img = self.calibrate_hist(img)
    img = self.calibrate_rgb(img)
    return img


def __boundaries_from_box(self, item):
    """
    resize boundaries to acquired image

    :param item: boundaries of text in screen (in resolution of display) or "none" value if input screen is cropped around text
    :type item: dict ({"box": [width left border, height upper border, width right border, height lower border]}
    :return: resized boundaries
    :rtype: array
    """
    startX = int(item["box"][0] * self.scale + self.border)
    startY = int(item["box"][1] * self.scale + self.border)
    endX = int(item["box"][2] * self.scale + self.border)
    endY = int(item["box"][3] * self.scale + self.border)
    return [startY, endY, startX, endX]


def read_animation_text(self, item, t=0.5):
    """
    read horizontally moving text

    :param item: boundaries of text in screen (in resolution of display) or "none" value if input screen is cropped around text
    :type item: dict ({"box": [width left border, height upper border, width right border, height lower border]}
    :param t: refresh time period (in seconds)
    :type t: float
    :return: read text, if is animation
    :rtype: string, bool
    """

    if self.cam is not None:
        img = self.__load_img()
    else:
        self.logger.error('No camera connected', extra=self.log_args)
        raise SystemExit

    if item["box"] is not None:
        boundaries = self.__boundaries_from_box(item)
    else:
        endY, endX = img.shape[:2]
        boundaries = [0, endY, 0, endX]
    direction = [0, 0, 0]
    # extract the actual padded ROI
    roi = img[boundaries[0]:boundaries[1], boundaries[2]:boundaries[3]]
    h, w = roi.shape[:2]
    fin_img = np.zeros((h, 10*w, 3), np.uint8)
    fin_img[0:h, 5*w:6*w] = img
    min_pos = 5*w
    max_pos = 6*w
    active = True
    l_loc = min_pos

    while active:
        time.sleep(t)
        img = self.__load_img()
        img = img[boundaries[0]:boundaries[1], boundaries[2]:boundaries[3]]

        res = cv.matchTemplate(fin_img, img, cv.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        if max_loc[0] < max_pos:
            if max_loc[0] < min_pos:
                fin_img[0:h, max_loc[0]:min_pos] = img[0:h,
                                                       0:(min_pos-max_loc[0])]
        else:
            fin_img[0:h, max_pos+w:max_loc[0] +
                    w] = img[0:h, w-(max_loc[0]-max_pos):w]
        fin_img[0:h, max_loc[0]:w+max_loc[0]] = cv.addWeighted(
            fin_img[0:h, max_loc[0]:w+max_loc[0]], 0.5, img, 0.5, 0)

        min_pos = min(max_loc[0], min_pos)
        max_pos = max(max_loc[0], max_pos)

        d = max_loc[0]-l_loc
        l_loc = max_loc[0]

        if direction[0] == 0:
            direction[0] = d
            counter = 0

        if d < 0:
            if direction[0] > 0:
                direction[1] = direction[1]+1
                direction[0] = d
                counter = 0
        elif d > 0:
            if direction[0] < 0:
                direction[1] = direction[1]+1
                direction[0] = d
                counter = 0
        else:
            counter = counter+1
            if counter > 5:
                active = False
        if direction[1] == 2:
            break
    fin = fin_img[0:h, min_pos:w+max_pos]
    text = self.read_text(None, fin, False)

    return (text, active)


def recognize_image_animated(self, item, t=0.1, t_end=1):
    """
    recognize animation in image

    :param item: boundaries of text in screen (in resolution of display) or "none" value if input screen is cropped around text, array of artwork images name
    :type item: dict ({"box": [width left border, height upper border, width right border, height lower border], "images": array of image names}
    :param t: refresh time period (in seconds)
    :type t: float
    :param t_end: max time of animation recognition
    :type t_end: int/float
    :return: recognized image, animation period, duty cycle
    :rtype: string, float, float
    """
    if self.cam is not None:
        img = self.__load_img()
    else:
        self.logger.error('No camera connected', extra=self.log_args)
        raise SystemExit

    if item["box"] is not None:
        boundaries = self.__boundaries_from_box(item)
    else:
        endY, endX = img.shape[:2]
        boundaries = [0, endY, 0, endX]

    new_item = {"box": None, "images": item["images"]}
    roi = img[boundaries[0]:boundaries[1], boundaries[2]:boundaries[3]]
    start = time.time()
    animation = {0: self.recognize_image(new_item, roi, 0.3, False)}

    for x in range(int(t_end/t)):
        img = self.__load_img()
        img = img[boundaries[0]:boundaries[1], boundaries[2]:boundaries[3]]
        animation[time.time()-start] = self.recognize_image(new_item, img, 0.3, False)

    read_item = []
    duty_cycles = {}
    duty_cycles_zero = {}
    periods = []
    item = {}

    for x, time in enumerate(animation):
        try:
            i = item[animation[time]]
        except:
            item[animation[time]] = len(read_item)
            read_item.append([animation[time], time, 1, x])
            continue

        if read_item[i][3] == x-1:
            read_item[i] = [animation[time], read_item[i][1], (read_item[i][2])+1, x]
        else:
            periods.append(time-read_item[i][1])
            read_item[i] = [animation[time], time, (read_item[i][2]+1), x]

            try:
                zero_time = duty_cycles_zero[animation[time]]
                duty_cycles[animation[time]] = [read_item[i][2]-zero_time, duty_cycles[animation[time]][1]+1]
            except:
                duty_cycles[animation[time]] = [0, 0]
                duty_cycles_zero[animation[time]] = read_item[i][2]

    count_period = 0
    duty_cycle = {}

    if len(periods) == 0:
        period = 0
        duty_cycle[read_item[0][0]] = 1
    else:
        for period in periods:
            count_period += period
        period = count_period/len(periods)

        for item in duty_cycles:
            if duty_cycles[item][0] == 0:
                duty_cycle[item] = 1
                print(duty_cycles[item])
            else:
                duty_cycle[item] = (
                    duty_cycles[item][1]/duty_cycles[item][0])

    return(duty_cycle, period)
