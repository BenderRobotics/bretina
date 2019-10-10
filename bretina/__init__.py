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


def shape(chessboard_img, chessboard_size, display_size, scale, border):
    """
    create calibration parameters from displayed chessboard to undistorted and crop acquired images

    :param chessboard_img: acquired image of chessboard on display
    :type chessboard_img: cv2 image (b,g,r matrix)
    :param chessboard_size: size of chessboard (number of white/black pairs)
    :type chessboard_size: [width, height] int/float
    :param display_size: display size (in px)
    :type display_size: [width, height] int/float
    :param scale: scale between camera resolutin and real display
    :type scale: int
    :param border: border (in pixels) around cropped dsplay
    :type border: int
    :return: parametrs of undistorted matrix, crop matrix and final resolution
    :rtype: touple of array
    """


    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    w_ch, h_ch = int(chessboard_size[0]*2-3), int(chessboard_size[1]*2-3)
    # prepare object points
    objp = np.zeros((w_ch*h_ch, 3), np.float32)
    objp[:, :2] = np.mgrid[0:h_ch, 0:w_ch].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    gray = cv.cvtColor(chessboard_img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (h_ch, w_ch), None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        h,  w = chessboard_img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # undistort
        undistort_parametr = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
        dst = cv.remap(chessboard_img, undistort_parametr[0], undistort_parametr[1], cv.INTER_LINEAR)

        # crop
        gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (h_ch, w_ch), None)
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)
        imgpoints.append(corners2)
        pts1 = np.float32([corners2[-1, 0], corners2[h_ch-1, 0], corners2[-h_ch, 0], corners2[0, 0]])
        ch_b = display_size[0]*scale/chessboard_size[0]+border
        fin_resolution = display_size[0]*scale+border*2, display_size[1]*scale+border*2
        pts2 = np.float32([[ch_b, fin_resolution[1]-ch_b], [fin_resolution[0]-ch_b, fin_resolution[1]-ch_b], [ch_b, ch_b], [fin_resolution[0]-ch_b, ch_b]])  # chessboard borders
        crop_parametr = cv.getPerspectiveTransform(pts1, pts2)
    return (undistort_parametr, crop_parametr, fin_resolution)



def crop(img, calibration_data):
    """
    undistorted and crop acquired image

    :param img: acquired image
    :type img: cv2 image (b,g,r matrix)
    :param calibration_data: parametrs of undistorted matrix, crop matrix and final resolution
    :type calibration_data: touple of array
    :return: undistorted and cropped image
    :rtype: cv2 image (b,g,r matrix)
    """

    # undistort
    dst = cv.remap(img, calibration_data[0][0],calibration_data[0][1], cv.INTER_LINEAR)
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

    crb, prb = np.histogram(np.ma.masked_less(r[:, :, 0], 10), bins=(255), range=(0, 255))
    crg, prg = np.histogram(np.ma.masked_less(r[:, :, 1], 10), bins=(255), range=(0, 255))
    crr, prr = np.histogram(np.ma.masked_less(r[:, :, 2], 10), bins=(255), range=(0, 255))
    cbb, pbb = np.histogram(np.ma.masked_less(b[:, :, 0], 10), bins=(255), range=(0, 255))
    cbg, pbg = np.histogram(np.ma.masked_less(b[:, :, 1], 10), bins=(255), range=(0, 255))
    cbr, pbr = np.histogram(np.ma.masked_less(b[:, :, 2], 10), bins=(255), range=(0, 255))
    cgb, pgb = np.histogram(np.ma.masked_less(g[:, :, 0], 10), bins=(255), range=(0, 255))
    cgg, pgg = np.histogram(np.ma.masked_less(g[:, :, 1], 10), bins=(255), range=(0, 255))
    cgr, pgr = np.histogram(np.ma.masked_less(g[:, :, 2], 10), bins=(255), range=(0, 255))
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
        k = np.where(ar < p[1], np.array(
            ar*(127/p[1])), np.array((ar-p[1])*(127/(255-p[1]))+127)).astype('uint8')
        k = np.where(k < p[0], 0, np.array(k-p[0]))
        k = np.where(k < (p[2]-p[0]), np.array(k*(255/(p[2]-p[0]))), 255).astype('uint8')
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


def read_text(img, language="english"):
    """
    read text from image

    :param item: boundaries of text in screen (in resolution of display) or "none" value if input screen is cropped around text
    :type item: dict ({"box": [width left border, height upper border, width right border, height lower border]}
    :param img: acquired image
    :type img: cv2 image (b,g,r matrix)
    :param overwrite: overwrite image in self.img
    :type overwrite: bool
    :return: read text
    :rtype: string
    """
    pytesseract.pytesseract.tesseract_cmd = r"Tesseract-OCR\tesseract.exe"

    if overwrite:
        self.__is_img_source(img)
        inv = self.img.copy()
    else:
        inv = img.copy()

    if item["box"] is None:
        roi = inv.copy()
    else:
        orig = inv.copy()
        boundaries = self.__boundaries_from_box(item)
        # extract the actual padded ROI
        roi = inv[boundaries[0]:boundaries[1], boundaries[2]:boundaries[3]]

    bgr = self.background_color(item, roi)
    c = (bgr[0]+bgr[1]+bgr[2])/3
    if c < 120:
        roi = 255-roi

    roi = cv.GaussianBlur(roi, (3, 3), 3)       # smoothing
    # in order to apply Tesseract v4 to OCR text we must supply
    # (1) a language, (2) an OEM flag of 4, indicating that the we
    # wish to use the LSTM neural net model for OCR, and finally
    # (3) an OEM value, in this case, 7 which implies that we are
    # treating the ROI as a single line of text
    config = ("-l eng --oem 3 --psm 7")
    text = pytesseract.image_to_string(roi, config=config)
    #text = "".join([c if ord(c) < 128 else "" for c in text]).strip()

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


def background_color(self, item, img=None):
    """
    determines the most frequent color in background, return r, g, b

    :param item: boundaries of box in screen (in resolution of display) or "none" for whole screen
    :type item: dict ({"box": [width left border, height upper border, width right border, height lower border]}
    :param img: acquired image
    :type img: cv2 image (b,g,r matrix)
    :return: blue, green and red most frequent intensity
    :rtype: int, int, int
    """
    self.__is_img_source(img)

    if img is not None:
        img_g = img.copy()
    else:
        img_g = self.img.copy()

    img_g = cv.GaussianBlur(img_g, (5, 5), 5)
    img_g = cv.GaussianBlur(img_g, (11, 11), 5)
    #img_g = cv.addWeighted(img_g, 1.01, img_g, 0, 0.01)

    if item["box"] is None:
        roi = img_g.copy()
    else:
        orig = img_g.copy()
        boundaries = self.__boundaries_from_box(item)
        # extract the actual padded ROI
        roi = orig[boundaries[0]:boundaries[1],
                   boundaries[2]:boundaries[3]]

    r1 = np.ma.masked_less(roi[:, :, 0], 5)
    r2 = np.ma.masked_less(roi[:, :, 1], 5)
    r3 = np.ma.masked_less(roi[:, :, 2], 5)

    crb, prb = np.histogram(r1, bins=(255), range=(0, 255))
    crg, prg = np.histogram(r2, bins=(255), range=(0, 255))
    crr, prr = np.histogram(r3, bins=(255), range=(0, 255))
    b = int(np.argmax(crb))
    g = int(np.argmax(crg))
    r = int(np.argmax(crr))
    return([r, g, b])


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
