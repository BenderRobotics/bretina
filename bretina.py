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














import logging


class OCR:
    """
    Base init of OCR class

    :param scale: scale between camera resolutin and real display
    :type scale: int/float
    :param border: border (in pixels) around cropped dsplay
    :type border: int
    :param path: path to artwork images
    :type path: string
    :param width_px: width of real display in px
    :type width_px: int
    :param height_px: width of real display in px
    :type height_px: int
    :param cam: imported camera
    :type cam: camera class variable
    """

    def __init__(self, scale, border, width_px = None, height_px = None, path = None, cam = None):

        self.log_args = {'class_name': self.__class__.__module__ + '.' + self.__class__.__name__}
        self.logger = logging.getLogger('brest')

        #: x parametrs of undistort matrix
        self.mx = None
        #: y parametrs of undistort matrix
        self.my = None
        #: crop matrix
        self.M = None
        #: rgb color calibration data
        self.rgb = None
        #: histofram color caibration data
        self.hist = None
        #: acquired image saved in local memory for future use
        self.img = None
        #: width of real display in px
        self.width_px = width_px
        #: height of real display in px
        self.height_px = height_px
        #: scale between camera resolutin and real display
        self.scale = scale
        #: border (in pixels) around cropped dsplay
        self.border = border
        #: imported camera
        self.cam = cam
        #: width of chessboard (number of white/black pairs)
        self.chessboard_width = None
        #: height of chessboard (number of white/black pairs)
        self.chessboard_height = None
        #: path to images used to recognize image
        self.path = path

    def shape(self, chessboard_img, chessboard_width, chessboard_height):
        """
        create calibratin parameters for crop and udistort inage

        create calibration parameters from displayed chessboard to undistort and crop acquired images

        :param chessboard_img: acquired image of chessboard on display
        :type chessboard_img: cv2 image (b,g,r matrix)
        :param chessboard_width: width of chessboard (number of white/black pairs)
        :type chessboard_width: int/float
        :param chessboard_height: height of chessboard (number of white/black pairs)
        :type chessboard_height: int/float
        """

        import cv2
        import numpy as np
        if self.width_px is None or self.height_px is None:
            self.logger.error('display height and width not defined', extra=self.log_args)
            raise SystemExit
        self.chessboard_width = chessboard_width
        self.chessboard_height = chessboard_height


        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        w_ch=int(chessboard_width*2-3)
        h_ch=int(chessboard_height*2-3)
        # prepare object points
        objp = np.zeros((w_ch*h_ch,3), np.float32)
        objp[:,:2] = np.mgrid[0:h_ch,0:w_ch].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        gray = cv2.cvtColor(chessboard_img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (h_ch,w_ch),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)

                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
                h,  w = chessboard_img.shape[:2]
                newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

                # undistort
                mx,my = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
                dst = cv2.remap(chessboard_img,mx,my,cv2.INTER_LINEAR)


                # crop
                gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (h_ch,w_ch),None)
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),criteria)
                imgpoints.append(corners2)
                pts1 = np.float32([corners2[-1,0],corners2[h_ch-1,0],corners2[-h_ch,0],corners2[0,0]])
                ch_b = self.width_px*self.scale/self.chessboard_width+self.border
                self.fin_res_w = self.width_px*self.scale+self.border*2
                self.fin_res_h = self.height_px*self.scale+self.border*2
                pts2 = np.float32([[ch_b,self.fin_res_h-ch_b],[self.fin_res_w-ch_b,self.fin_res_h-ch_b],[ch_b,ch_b],[self.fin_res_w-ch_b,ch_b]]) #chessboard borders
                M = cv2.getPerspectiveTransform(pts1,pts2)
                dst = cv2.warpPerspective(dst,M,(self.fin_res_w,self.fin_res_h))
        self.mx=mx
        self.my=my
        self.M=M

    def crop(self,img):
        """
        undistort and crop acquired image

        :param img: acquired image
        :type img: cv2 image (b,g,r matrix)
        :return: undistorted and croped image
        :rtype: cv2 image (b,g,r matrix)
        """

        import cv2

        if self.mx is None:
            self.logger.warning('No calibration data to crop image', extra=self.log_args)
            return img
        # undistort
        dst = cv2.remap(img,self.mx,self.my,cv2.INTER_LINEAR)
        # crop
        fin = cv2.warpPerspective(dst,self.M,(self.fin_res_w,self.fin_res_h))
        return (fin)

    def col_cal(self, chessboard_img, r, g, b):
        """
        create calibratin parameters histogram and rgb color calibration

        create calibration parameters from displayed chessboard and red, green and blue screen to rgb color calibration and histogram color calibration


        :param chessboard_img: acquired image of chessboard on display
        :type chessboard_img: cv2 image (b,g,r matrix)
        :param r: acquired image of red screen
        :type r: cv2 image (b,g,r matrix)
        :param g: acquired image of green screen
        :type g: cv2 image (b,g,r matrix)
        :param b: acquired image of blue screen
        :type b: cv2 image (b,g,r matrix)

        """
        import cv2
        import numpy as np

        # BGR screen cropped to better function
        b = b[50:-50, 50:-50]
        g = g[50:-50, 50:-50]
        r = r[50:-50, 50:-50]

        r1=np.ma.masked_less(r[:,:,0],20)
        r2=np.ma.masked_less(r[:,:,1],20)
        r3=np.ma.masked_less(r[:,:,2],20)
        b1=np.ma.masked_less(b[:,:,0],20)
        b2=np.ma.masked_less(b[:,:,1],20)
        b3=np.ma.masked_less(b[:,:,2],20)
        g1=np.ma.masked_less(g[:,:,0],20)
        g2=np.ma.masked_less(g[:,:,1],20)
        g3=np.ma.masked_less(g[:,:,2],20)
        crb,prb=np.histogram(r1, bins=(2550), range=(0,255))
        crg,prg=np.histogram(r2, bins=(2550), range=(0,255))
        crr,prr=np.histogram(r3, bins=(2550), range=(0,255))
        cbb,pbb=np.histogram(b1, bins=(2550), range=(0,255))
        cbg,pbg=np.histogram(b2, bins=(2550), range=(0,255))
        cbr,pbr=np.histogram(b3, bins=(2550), range=(0,255))
        cgb,pgb=np.histogram(g1, bins=(2550), range=(0,255))
        cgg,pgg=np.histogram(g2, bins=(2550), range=(0,255))
        cgr,pgr=np.histogram(g3, bins=(2550), range=(0,255))
        mcbb=max(cbb)
        cbb=np.where(0.95*mcbb<cbb,0.95*mcbb,cbb)
        mcgg=max(cgg)
        cgg=np.where(0.95*mcgg<cgg,0.95*mcgg,cgg)
        mcrr=max(crr)
        crr=np.where(0.95*mcrr<crr,0.95*mcrr,crr)

        bmin=min(int(np.argmax(crb)/10),int(np.argmax(cgb)/10))
        gmin=min(int(np.argmax(crg)/10),int(np.argmax(cbg)/10))
        rmin=min(int(np.argmax(cbr)/10),int(np.argmax(cbg)/10))

        if cbb[-1]>10:
            bmax=int(np.argmax(cbb)/10)
        else:
            bmax=255
        if cgg[-1]>10:
            gmax=int(np.argmax(cgg)/10)
        else:
            gmax=255
        if crr[-1]>10:
            rmax=int(np.argmax(crr)/10)
        else:
            rmax=255
        bmean=int(np.mean(chessboard_img[:,:,0]))
        gmean=int(np.mean(chessboard_img[:,:,1]))
        rmean=int(np.mean(chessboard_img[:,:,2]))

        bp=[bmin,bmean,bmax]
        gp=[gmin,gmean,gmax]
        rp=[rmin,rmean,rmax]
        self.hist=[bp,gp,rp] # histogram calibration matrix


        # BGR screen cropped to better function
        b = self.calibrate_hist(b)
        g = self.calibrate_hist(g)
        r = self.calibrate_hist(r)
        bb=np.mean(b[:,:,0])
        bg=np.mean(b[:,:,1])
        br=np.mean(b[:,:,2])
        gb=np.mean(g[:,:,0])
        gg=np.mean(g[:,:,1])
        gr=np.mean(g[:,:,2])
        rb=np.mean(r[:,:,0])
        rg=np.mean(r[:,:,1])
        rr=np.mean(r[:,:,2])

        chb = cv2.GaussianBlur(chessboard_img, (9,9), 10)
        chb = cv2.GaussianBlur(chb, (5,5), 20)
        chdim=[self.chessboard_width,self.chessboard_height] #chessboard size
        h,  w = chb.shape[:2] #image size
        hs=int(h/chdim[1])
        ws=int(w/chdim[0])

        i=0
        wb=np.zeros(int(chdim[0])*int(chdim[1]))
        wg=np.zeros(int(chdim[0])*int(chdim[1]))
        wr=np.zeros(int(chdim[0])*int(chdim[1]))
        kb=np.zeros(int(chdim[0])*int(chdim[1]))
        kg=np.zeros(int(chdim[0])*int(chdim[1]))
        kr=np.zeros(int(chdim[0])*int(chdim[1]))
        for x in range (int(chdim[0])):
            for y in range (int(chdim[1])):
                x1=int(ws*x+ws/4)
                x2=int(ws*x+ws*3/4)
                y1=int(hs*y+hs/4)
                y2=int(hs*y+hs*3/4)

                wb[i]=((int(chb[y1,x1,0])+int(chb[y2,x2,0]))/2)
                wg[i]=((int(chb[y1,x1,1])+int(chb[y2,x2,1]))/2)
                wr[i]=((int(chb[y1,x1,2])+int(chb[y2,x2,2]))/2)
                kb[i]=((int(chb[y1,x2,0])+int(chb[y2,x1,0]))/2)
                kg[i]=((int(chb[y1,x2,1])+int(chb[y2,x1,1]))/2)
                kr[i]=((int(chb[y1,x2,2])+int(chb[y2,x1,2]))/2)
                i=i+1

        wb=np.mean(wb)
        wg=np.mean(wg)
        wr=np.mean(wr)
        kb=np.mean(kb)
        kg=np.mean(kg)
        kr=np.mean(kr)
        Bi=[bb,bg,br]
        Gi=[gb,gg,gr]
        Ri=[rb,rg,rr]
        Wi=[wb,wg,wr]
        Ki=[kb,kg,kr]
        I=[Bi,Gi,Ri,Wi,Ki,Wi,Ki,Wi,Ki]

        Bt=[255,0,0]
        Gt=[0,255,0]
        Rt=[0,0,255]
        Wt=[255,255,255]
        Kt=[0,0,0]
        T=[Bt,Gt,Rt,Wt,Kt,Wt,Kt,Wt,Kt]

        invI=np.linalg.pinv(I)

        self.rgb=np.dot(invI,T)


    def calibrate_hist(self, img):
        """
        histgram calibration on acquired image

        :param img: acquired image
        :type img: cv2 image (b,g,r matrix)
        :return: histgram calibrated image
        :rtype: cv2 image (b,g,r matrix)

        """
        import cv2
        import numpy as np

        if self.hist is None:
            self.logger.warning('No calibration data to histogram color calibration', extra=self.log_args)
            return img
        pp=self.hist

        imgo=img.copy()
        for x in range(0, 3):
            ar=img[:,:,x]
            p=pp[x]
            k=np.where(ar<p[1],np.array(ar*(127/p[1])),np.array((ar-p[1])*(127/(255-p[1]))+127)).astype('uint8')
            k=np.where(k<p[0], 0, np.array(k-p[0]))
            k=np.where(k <(p[2]-p[0]), np.array(k*(255/(p[2]-p[0]))), 255).astype('uint8')
            imgo[:,:,x]=k
        return imgo

    def calibrate_rgb(self, img):
        """
        rgb color calibration on acquired image

        :param img: acquired image
        :type img: cv2 image (b,g,r matrix)
        :return: rgb color calibrated image
        :rtype: cv2 image (b,g,r matrix)

        """
        import numpy as np

        if self.rgb is None:
            self.logger.warning('No calibration data to RGB color calibration', extra=self.log_args)
            return img
        t=self.rgb

        imgo=img.copy()
        bb=(img[:,:,0])
        bg=(img[:,:,1])
        br=(img[:,:,2])
        gb=(img[:,:,0])
        gg=(img[:,:,1])
        gr=(img[:,:,2])
        rb=(img[:,:,0])
        rg=(img[:,:,1])
        rr=(img[:,:,2])
        bo=bb*t[0,0]+bg*t[1,0]+br*t[2,0]
        go=gb*t[0,1]+gg*t[1,1]+gr*t[2,1]
        ro=rb*t[0,2]+rg*t[1,2]+rr*t[2,2]
        bo=np.where(bo<0, 0, bo)
        go=np.where(go<0, 0, go)
        ro=np.where(ro<0, 0, ro)

        bo=np.where(bo>255, 255, bo).astype('uint8')
        go=np.where(go>255, 255, go).astype('uint8')
        ro=np.where(ro>255, 255, ro).astype('uint8')

        imgo[:,:,0]=bo
        imgo[:,:,1]=go
        imgo[:,:,2]=ro
        return imgo


    def read_text(self, item, img = None, overwrite = True):
        """
        read text from image

        :param item: boundaris of text in screen (in resolution of display) or "none" value if input screen is cropped around text
        :type item: dict ({"box": [width left border, height upper border, width right border, height lower border]}
        :param img: acquired image
        :type img: cv2 image (b,g,r matrix)
        :param overwrite: overwrite image in self.img
        :type overwrite: bool
        :return: read text
        :rtype: string
        """

        import cv2
        try:
            import pytesseract
        except ModuleNotFoundError:
            raise ModuleNotFoundError(f'To use {self.__class__.__name__} class you have to install `pytesseract` module')

        pytesseract.pytesseract.tesseract_cmd = r"Tesseract-OCR\tesseract.exe"

        if overwrite:
            self.__is_img_source(img)
            inv=self.img.copy()
        else:
            inv=img.copy()

        if item["box"] is None:
            roi=inv.copy()
        else:
            orig = inv.copy()
            boundaries = self.__boundaries_from_box(item)
            # extract the actual padded ROI
            roi = inv[boundaries[0]:boundaries[1], boundaries[2]:boundaries[3]]

        bgr=self.background_color(item, roi)
        c=(bgr[0]+bgr[1]+bgr[2])/3
        if c<120:
            roi=255-roi

        roi = cv2.GaussianBlur(roi, (3,3), 3)       # smoothing
        # in order to apply Tesseract v4 to OCR text we must supply
        # (1) a language, (2) an OEM flag of 4, indicating that the we
        # wish to use the LSTM neural net model for OCR, and finally
        # (3) an OEM value, in this case, 7 which implies that we are
        # treating the ROI as a single line of text
        config = ("-l eng --oem 3 --psm 7")
        text = pytesseract.image_to_string(roi, config=config)
        #text = "".join([c if ord(c) < 128 else "" for c in text]).strip()

        return (text)

    def recognize_image(self, item, img = None, image_matching=0.3, overwrite = True):
        """
        compare image from box at screen with artwork

        :param item: boundaris of text in screen (in resolution of display) or "none" value if input screen is cropped around text, array of artwork images name
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
        import cv2

        if overwrite:
            self.__is_img_source(img)
            inv=self.img.copy()
        else:
            inv=img.copy()

        bgr=self.background_color(item, inv)
        c=(bgr[0]+bgr[1]+bgr[2])/3

        if c<120:
            inv=255-inv
        img_gray = cv2.cvtColor(inv, cv2.COLOR_BGR2GRAY)
        roi = img_gray.copy()
        if item["box"] is not None:
            boundaries = self.__boundaries_from_box(item)
            # extract the actual padded ROI
            roi = img_gray[boundaries[0]:boundaries[1], boundaries[2]:boundaries[3]]

        imgBlurred = cv2.GaussianBlur(roi, (5,5), 0)        # smoothing
        edges = cv2.Canny(imgBlurred,150,200)               # transfer to edges
        edges = cv2.GaussianBlur(edges, (5,5), 0)           # smoothing
        maximum=[]
        for (items) in item["images"]:
            icon = cv2.imread(self.path+items,0)
            imgBlurred2 = cv2.GaussianBlur(icon, (5,5), 0)
            edg = cv2.Canny(imgBlurred2,150,200)
            edg = cv2.GaussianBlur(edg, (5,5), 0)
            res = cv2.matchTemplate(edges,edg,cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            maximum.append(max_val)

        val=max(maximum)
        pos=maximum.index(val)
        if val > image_matching:
            a = item["images"][pos]
        else:
            a = None
        return (a)
    def recognize_image_low_contrast(self, item, img = None, image_matching=0.3, overwrite = True):
        """
        compare image from box at screen with artwork

        compare image from box at screen with artwork, it is assumption that image from box has low contrast and can't be transformed to edges

        :param item: boundaris of text in screen (in resolution of display) or "none" value if input screen is cropped around text, array of artwork images name
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
        import cv2

        if overwrite:
            self.__is_img_source(img)
            inv=self.img.copy()
        else:
            inv=img.copy()

        bgr=self.background_color(item, inv)
        c=(bgr[0]+bgr[1]+bgr[2])/3

        if c<120:
            inv=255-inv
        img_gray = cv2.cvtColor(inv, cv2.COLOR_BGR2GRAY)
        roi = img_gray.copy()
        if item["box"] is not None:
            boundaries = self.__boundaries_from_box(item)
            # extract the actual padded ROI
            roi = img_gray[boundaries[0]:boundaries[1], boundaries[2]:boundaries[3]]

        imgBlurred = cv2.GaussianBlur(roi, (5,5), 0)        # smoothing
        edges = cv2.Canny(imgBlurred,150,200)               # transfer to edges
        edges = cv2.GaussianBlur(edges, (5,5), 0)           # smoothing
        maximum=[]
        for (items) in item["images"]:
            icon = cv2.imread(self.path+items,0)

            res = cv2.matchTemplate(edges,icon,cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            maximum.append(max_val)

        val=max(maximum)
        pos=maximum.index(val)
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
        import cv2
        cv2.imshow("img",self.img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def background_color(self, item, img = None):
        """
        determines the most frequent color in background, return r, g, b

        :param item: boundaris of box in screen (in resolution of display) or "none" for whole screen
        :type item: dict ({"box": [width left border, height upper border, width right border, height lower border]}
        :param img: acquired image
        :type img: cv2 image (b,g,r matrix)
        :return: blue, green and red most frequent intensity
        :rtype: int, int, int
        """
        import cv2
        import numpy as np
        self.__is_img_source(img)
        if img is not None:
            img_g=img.copy()
        else:
            img_g=self.img.copy()
        img_g = cv2.GaussianBlur(img_g, (5,5), 5)
        img_g = cv2.GaussianBlur(img_g, (11,11), 5)
        #img_g = cv2.addWeighted(img_g, 1.01, img_g, 0, 0.01)

        if item["box"] is None:
            roi=img_g.copy()
        else:
            orig = img_g.copy()
            boundaries = self.__boundaries_from_box(item)
            # extract the actual padded ROI
            roi = orig[boundaries[0]:boundaries[1], boundaries[2]:boundaries[3]]

        r1=np.ma.masked_less(roi[:,:,0],5)
        r2=np.ma.masked_less(roi[:,:,1],5)
        r3=np.ma.masked_less(roi[:,:,2],5)

        crb,prb=np.histogram(r1, bins=(255), range=(0,255))
        crg,prg=np.histogram(r2, bins=(255), range=(0,255))
        crr,prr=np.histogram(r3, bins=(255), range=(0,255))
        b=int(np.argmax(crb))
        g=int(np.argmax(crg))
        r=int(np.argmax(crr))
        return([r,g,b])

    def __is_img_source(self, img):
        """
        load image from difrent sources

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
                    self.img=self.__load_img()
                else:
                    self.logger.error('No camera or image source', extra=self.log_args)
                    raise SystemExit

    def __load_img(self):
        """
        acquire and calibrate image from camera

        :return: calibrated image
        :rtype: cv2 image (b,g,r matrix)
        """

        img=self.cam.acquire_image()
        img=self.crop(img)
        img=self.calibrate_hist(img)
        img=self.calibrate_rgb(img)
        return img

    def __boundaries_from_box(self, item):
        """
        resize boundaries to acquired image

        :param item: boundaris of text in screen (in resolution of display) or "none" value if input screen is cropped around text
        :type item: dict ({"box": [width left border, height upper border, width right border, height lower border]}
        :return: resized boundaries
        :rtype: array
        """
        startX = int(item["box"][0] * self.scale + self.border)
        startY = int(item["box"][1] * self.scale + self.border)
        endX = int(item["box"][2] * self.scale + self.border)
        endY = int(item["box"][3] * self.scale + self.border)
        return [startY, endY, startX, endX]

    def read_animation_text(self, item, t = 0.5):
        """
        read moving text

        read horizontaly moving text

        :param item: boundaris of text in screen (in resolution of display) or "none" value if input screen is cropped around text
        :type item: dict ({"box": [width left border, height upper border, width right border, height lower border]}
        :param t: refresh time period (in seconds)
        :type t: float
        :return: read text, if is animation
        :rtype: string, bool
        """

        import time
        import cv2
        import numpy as np

        if self.cam is not None:
            img=self.__load_img()
        else:
            self.logger.error('No camera connected', extra=self.log_args)
            raise SystemExit
        if item["box"] is not None:
            boundaries = self.__boundaries_from_box(item)

        else:
            endY, endX = img.shape[:2]
            boundaries = [0, endY, 0, endX]
        direction=[0,0,0]
        # extract the actual padded ROI
        roi = img[boundaries[0]:boundaries[1], boundaries[2]:boundaries[3]]
        h, w = roi.shape[:2]
        fin_img = np.zeros((h,10*w,3), np.uint8)
        fin_img[0:h, 5*w:6*w]=img
        min_pos=5*w
        max_pos=6*w
        active = True
        l_loc=min_pos
        while(active):
            time.sleep(t)
            img=self.__load_img()
            img = img[boundaries[0]:boundaries[1], boundaries[2]:boundaries[3]]

            res = cv2.matchTemplate(fin_img,img,cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if max_loc[0]<max_pos:
                if max_loc[0]<min_pos:
                    fin_img[0:h, max_loc[0]:min_pos]=img[0:h, 0:(min_pos-max_loc[0])]
            else:
                fin_img[0:h, max_pos+w:max_loc[0]+w]=img[0:h, w-(max_loc[0]-max_pos):w]
            fin_img[0:h, max_loc[0]:w+max_loc[0]]=cv2.addWeighted(fin_img[0:h, max_loc[0]:w+max_loc[0]],0.5,img,0.5,0)

            min_pos = min(max_loc[0],min_pos)
            max_pos = max(max_loc[0],max_pos)

            d = max_loc[0]-l_loc
            l_loc = max_loc[0]
            if direction[0] == 0:
                direction[0] = d
                counter = 0
            if d<0:
                if direction[0]>0:
                    direction[1] = direction[1]+1
                    direction[0] = d
                    counter = 0

            elif d>0:
                if direction[0]<0:
                    direction[1] = direction[1]+1
                    direction[0] = d
                    counter = 0
            else:
                counter=counter+1
                if counter > 5:
                    active = False
            if direction[1] == 2:
                break
        fin=fin_img[0:h, min_pos:w+max_pos]
        text=self.read_text(None, fin, False)

        return(text, active)

    def recognize_image_animated(self, item, t = 0.1, t_end = 1):

        """
        recognize animation in image

        :param item: boundaris of text in screen (in resolution of display) or "none" value if input screen is cropped around text, array of artwork images name
        :type item: dict ({"box": [width left border, height upper border, width right border, height lower border], "images": array of image names}
        :param t: refresh time period (in seconds)
        :type t: float
        :param t_end: max time of animation recognition
        :type t_end: int/float
        :return: recognized image, animation period, duty cycle
        :rtype: string, float, float
        """
        import time
        import cv2
        import numpy as np

        if self.cam is not None:
            img=self.__load_img()
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
        animation={0: self.recognize_image(new_item, roi, 0.3, False)}
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
                    duty_cycle[item] = (duty_cycles[item][1]/duty_cycles[item][0])

        return(duty_cycle, period)
