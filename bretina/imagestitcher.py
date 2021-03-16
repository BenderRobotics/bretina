import numpy as np
import cv2 as cv
import bretina


class ImageStitcher():
    '''
    Combines given images (stitches them) until the final image is created.
    '''
    #: Minimal value of the template match. When it is bellow this value, the merged image is cleaned and stitching
    #: starts from the blank image.
    MINIMAL_MATCH = 0.8

    def __init__(self, axis=None, bgcolor=None, convergence_limit=10, max_iterations=200, cut_off_bg=True,
                 cut_off_bg_threshold=40):
        """
        :param str axis: axis of stitching
                            - 'h': horizontal
                            - 'v': vertical
                            - None [default]: both
        :param bgcolor: expected background color, taken automatically from the image if None.
        :param int convergence_limit: when the output is not changed for this number of added images, the operation is
                                      finished.
        :param int max_iterations: maximum absolute iteration count
        :param bool cut_off_bg: True - background of new images is removed during merging.
        :param int cut_off_bg_threshold: color distance threshold to identify background pixels
        """
        self._axis = axis.lower() if isinstance(axis, str) else None
        self._bgcolor = bgcolor
        self._counter = 0
        self._no_change_count = 0
        self._combined_img = None
        self._canvas = None
        self._max_iterations = max_iterations
        self._convergence_limit = convergence_limit
        self._cut_off_bg = cut_off_bg
        self._cut_off_bg_thr = cut_off_bg_threshold
        self._w = 0
        self._h = 0
        self._ch = 3

    def _background_mask(self, img, bgcolor):
        """
        Determines mask of the image background based on the background color.

        :param img: image to process
        :type  img: cv2 image (b, g, r matrix)
        :param bgcolor: background color
        :type  bgcolor: tuple (b, g, r)
        :return: mask matrix
        """
        b, g, r = bgcolor
        lower = np.maximum((b-self._cut_off_bg_thr, g-self._cut_off_bg_thr, r-self._cut_off_bg_thr), (0, 0, 0))
        upper = np.minimum((b+self._cut_off_bg_thr, g+self._cut_off_bg_thr, r+self._cut_off_bg_thr), (255, 255, 255))
        mask = cv.inRange(img, lower, upper)
        return mask

    def _merge(self, img, location):
        """
        Merges new image with the already merged images at the given location.

        :param img: new image to merge
        :param location: where the new image shall be added
        :type location: tuple (x, y)
        """
        location = list(location)

        # Cut part of the image near the edge to prevent added black field
        if (location[0] == self._roi[0]) and (location[1] == self._roi[1]):
            pass
        elif (location[1] == self._roi[1]) and (location[0] > self._roi[0]):
            shift = int(img.shape[1] * 0.1)
            img = img[:, shift:]
            location[0] += shift
        elif (location[1] == self._roi[1]) and (location[0] < self._roi[0]):
            shift = int(img.shape[1] * 0.9)
            img = img[:, :shift]
        elif (location[0] == self._roi[0]) and (location[1] > self._roi[1]):
            shift = int(img.shape[0] * 0.1)
            img = img[shift:, :]
            location[1] += shift
        elif (location[0] == self._roi[0]) and (location[1] < self._roi[1]):
            shift = int(img.shape[0] * 0.9)
            img = img[:shift, :]

        img_w, img_h = img.shape[1], img.shape[0]
        x_1 = location[0]
        x_2 = location[0] + img_w
        y_1 = location[1]
        y_2 = location[1] + img_h

        # If we have the background mask, mask the image background and use the pixels from the canvas instead
        if self._cut_off_bg:
            mask = self._background_mask(img, self._bgcolor)
            canvas_roi = self._canvas[y_1:y_2, x_1:x_2]
            canvas_roi = cv.bitwise_or(canvas_roi, canvas_roi, mask=mask)
            img = cv.bitwise_or(canvas_roi, img)

        # Add the aligned image to the merged image
        self._canvas[y_1:y_2, x_1:x_2] = img

        # Crop the merged image out of the canvas
        roi = (min(self._roi[0], x_1),
               min(self._roi[1], y_1),
               max(self._roi[2], x_2),
               max(self._roi[3], y_2))

        self._combined_img = self._canvas[roi[1]:roi[3], roi[0]:roi[2]]
        self._w = self._combined_img.shape[1]
        self._h = self._combined_img.shape[0]
        self._roi = (0, 0, self._w, self._h)

    def _align(self, img):
        """
        Aligns new image to the rest of the merged images.

        :param img: new image to align
        :return: location where the new image shall be added
        :rtype: tuple(correlation, tuple(x, y))
        """
        roi_w, roi_h = (self._combined_img.shape[1], self._combined_img.shape[0])
        canvas_w = roi_w * (2 if self._axis != 'v' else 1)
        canvas_h = roi_h * (2 if self._axis != 'h' else 1)
        self._roi = ((canvas_w - roi_w) // 2,
                     (canvas_h - roi_h) // 2,
                     (canvas_w - roi_w) // 2 + roi_w,
                     (canvas_h - roi_h) // 2 + roi_h)

        # Create the canvas (double the size) and put the merged image in the middle, rest fill with bg color
        self._canvas = np.zeros((canvas_h, canvas_w, self._ch), np.uint8)
        self._canvas[:] = self._bgcolor
        self._canvas[self._roi[1]:self._roi[3], self._roi[0]:self._roi[2]] = self._combined_img

        # Align new img with the canvas
        res = cv.matchTemplate(self._canvas, img, cv.TM_CCORR_NORMED)
        _, value, __, location = cv.minMaxLoc(res)
        return value, location

    def reset(self, img):
        """
        Resets the already merged image and starts over.

        :param img: new image to start with
        """
        self._w = img.shape[1]
        self._h = img.shape[0]
        self._ch = 1 if (len(img.shape) == 2) else img.shape[2]
        self._roi = (0, 0, self._w, self._h)
        self._no_change_count = 0
        self._combined_img = img

        # Get background if not specified
        if self._bgcolor is None:
            self._bgcolor = bretina.background_color(img, mean=False)
        else:
            self._bgcolor = bretina.color(self._bgcolor)

    def add(self, img):
        """
        Adds image to the already stitched image

        :param img: cropped image around moving text prom camera
        :type  img: cv2 image (b,g,r matrix)
        :return: tuple
                    - bool finished: False until the final image is created,
                    - merged image
        :rtype: tuple (bool, cv2 image)
        """
        assert img is not None, '`img` has to be given'

        if self._combined_img is None:
            self.reset(img)
        else:
            # Mask background if requested and replace the pixels with background color
            if self._cut_off_bg:
                mask = self._background_mask(img, self._bgcolor)
                img[mask != 0] = self._bgcolor

            self._counter += 1
            dimension = (self._w, self._h)
            value, location = self._align(img)

            # Add this picture only if the maximum is high
            if value > self.MINIMAL_MATCH:
                self._merge(img, location)

                if dimension == (self._w, self._h):
                    self._no_change_count += 1
                else:
                    self._no_change_count = 0
            # Otherwise reset
            else:
                self.reset(img)

        # Return False until FINAL_LIMIT times in a row no change is detected
        finished = (self._no_change_count >= self._convergence_limit) or (self._counter >= self._max_iterations)
        return finished, self._combined_img

    def result(self):
        """
        Returns result of the stitching.

        :return: merged image
        :rtype: cv2 image)
        """
        assert self._combined_img is not None, 'No images given to perfrom stitching, use `add` method first.'
        return self._combined_img
