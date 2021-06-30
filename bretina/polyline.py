import numpy as np
import cv2 as cv
import bretina
import math


def create_binary_img(img, threshold=127):
    """
    Creates binary image from selected image.

    :param array_like img: cv image
    :param threshold: value used for thresholding
    :type threshold : int

    :return: binary image
    """
    assert isinstance(threshold, (int, float)), f"Parameter `threshold` has to be type of `int` or `float`, type {type(threshold)} is not supported."

    # converts image to grayscale
    gray = bretina.img_to_grayscale(img)

    # gets background lightness
    bg_light = bretina.background_lightness(gray)

    # creates binary image
    ret, binary_img = cv.threshold(gray, threshold, 255, (cv.THRESH_BINARY if bg_light < 128 else cv.THRESH_BINARY_INV) + cv.THRESH_OTSU)

    return binary_img


def thinning(img):
    """
    Returns skeletonized version of image.

    :param array_like img: cv image
    :return: skeletonized image
    """
    # creates binary img
    binary_img = create_binary_img(img)

    skelet = binary_img.copy() / 255
    change = True  # flag which indicates whether change in the img occurred

    def _transitions(adjacent_pixels):
        """
        Returns number of transitions from 0 to 1 of adjacent pixels.

        :param adjacent_pixels
        :type adjacent_pixels: list

        :return number of found transitions
        :rtype: int
        """
        count = 0  # number of pairs which meet the cond2 - transitions from 0 to 1
        extend_adjacent_pixels = adjacent_pixels + [adjacent_pixels[0]]

        for index in range(len(adjacent_pixels)):
            if extend_adjacent_pixels[index] == 0 and extend_adjacent_pixels[index+1] == 1:
                count += 1

        return count

    white_pixels = [[] for _ in range(img.shape[0])]  # empty list of all positions of the white pixels in the img

    # finds all white pixels in the img and saves them to list
    for x in range(1, img.shape[1] - 1):
        for y in range(1, img.shape[0] - 1):
            if skelet[y][x] == 1:
                white_pixels[y].append(x)

    # Zhang-Suen thinning algorithm
    while change:
        change = False

        for step in range(2):
            pix_to_change = []  # position of pixels which value should be change to zero

            for y, _ in enumerate(white_pixels):
                changes = []

                for index, x in enumerate(_):
                    # 8 neighbors of selected point in image in clockwise order
                    adjacent_pixels = [skelet[y-1][x], skelet[y-1][x+1], skelet[y][x+1], skelet[y+1][x+1],
                                        skelet[y+1][x], skelet[y+1][x-1], skelet[y][x-1], skelet[y-1][x-1]]

                    # conditions which have to be met:
                    # cond0: selected pixel is 1 and has eight neighbors
                    # cond1: 2 < = N(P1) < = 6
                    # cond2: number of transitions from 0 to 1 = 1
                    # cond3: P2 * P4 * P6 = 0
                    # cond4: P4 * P6 * P8 = 0

                    cond0 = skelet[y][x]
                    cond1 = sum(adjacent_pixels)
                    cond2 = _transitions(adjacent_pixels)
                    cond3 = adjacent_pixels[0] * adjacent_pixels[2] * adjacent_pixels[4]
                    cond4 = adjacent_pixels[2] * adjacent_pixels[4] * adjacent_pixels[6]

                    if cond0 == 1 and (cond1 >= 2 and cond1 <= 6) and cond2 == 1 and cond3 == 0 and cond4 == 0:
                        change = True
                        changes.append(x)
                        pix_to_change.append([x, y])

                white_pixels[y] = list(set(white_pixels[y]) - set(changes))

            # modification of the img
            for x, y in pix_to_change: 
                skelet[y][x] = 0

    skeletonized_img = (skelet * 255).astype('uint8')

    return skeletonized_img


def get_polyline_width(img):
    """
    Returns average polyline width.

    :param array_like img: cv image
    :return: average polyline width
    :rtype: float
    """
    # creates binary img
    binary_img = create_binary_img(img)

    # creates img skeleton
    skeletonize_img = thinning(binary_img)

    # width calculation
    sum_skelet = np.sum(skeletonize_img) / 255
    assert sum_skelet != 0, "There is no line in selected img, so the width of the line can`t be modified."

    sum_img = np.sum(binary_img) / 255
    width = round(sum_img / sum_skelet, 1)

    return width


def change_polyline_width(img, required_width=5):
    """
    Modifies the polyline width in img to the desired one.

    :param array_like img: part of the image with polyline
    :param required_width: expected polyline width [px]
    :type required_width: int

    :return: binary image with changed polylines
    """
    assert isinstance(required_width, int), f"`required_width` has to be type of `int`, `{type(required_width)}` given"
    assert len(img.shape) <= 2, f"input img has to have only 2 dimensions (grayscale), {img.shape[2]} dimensions given"

    # determines the current line width
    line_width = get_polyline_width(img)

    # determines how many iterations are needed to create the desired polyline width
    diff = int(abs(line_width - required_width))
    if (diff % 2) != 0:
        diff += 1

    iterations = int(diff / 2)  # only half of the calculated iterations are needed due to the size of the kernel used
    kernel = np.array([[0,1,0], [1,1,1], [0,1,0]], dtype=np.uint8)

    # polyline modification in img
    if line_width > required_width:
        modified = cv.erode(img, kernel, iterations=iterations)
    elif line_width < required_width:
        modified = cv.dilate(img, kernel, iterations=iterations)
    else:
        modified = img

    return modified


def get_polyline_first_point(img):
    """
    Returns coordinates of the first point of the searched polyline in pixels.

    :param array_like img: part of image where is searched polyline

    :return: found coordinates of the first point [px]
    :rtype: list [x, y]
    """
    first_point = []
    dominant_color = 0  # its binary img, so only two colors are possible (0: black, 255: white)

    width = img.shape[1]  # width of the selected img [px]
    height = img.shape[0]  # height of the selected img [px]

    # creates binary img
    binary_img = create_binary_img(img)

    # finds first white pixel in the image from beginning of the coordinate system
    for actual_x in range(width):
        for actual_y in range(height):
            if binary_img[actual_y, actual_x] > dominant_color:
                first_point.extend([actual_x, actual_y])  # coordinates of the beginning of the curve [x, y]
                break
        if first_point:
            break

    return first_point


def get_number_of_polylines(img, min_line_len=15):
    """
    Returns a number that expresses the number of polylines found in the image.

    :param array_like img: part of image where polylines are
    :param min_line_len: the minimum length that the polyline must have in order to be considered as a polyline [px]
    :type min_line_len : int

    :return: number of found polylines
    :rtype: int
    """
    assert isinstance(min_line_len, int), f"`min_line_len` has to be type of `int`, `{type(min_line_len)}` given"

    num_of_lines = 0  # number of polylines in the image

    # creates binary img
    binary_img = create_binary_img(img)

    # get the number of polylines in the image
    contours, hierarchy = cv.findContours(binary_img, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]

    for cnt in zip(contours, hierarchy):
        contour = cnt[0]  # current contour
        current_hierarchy = cnt[1]  # hierarchy of current contour

        if current_hierarchy[3] < 0:  # only contours of holes will have the parent in 2-lvl hierarchy
            if cv.arcLength(contour, False) > min_line_len:
                num_of_lines += 1
            else:
                cv.drawContours(img, [cnt[0]], -1, (0, 0, 0), -1)

    return num_of_lines


def split_polylines_to_seprate_imgs(img):
    """
    Returns array of images, in which each layer contains exactly one polyline.

    :param array_like img: part of image where polylines are
    :return: array in which each polyline is saved in a separate image
    :rtype: 3D array
    """
    # creates binary img
    binary_img = create_binary_img(img)

    # create array in which each polyline will be drawn separately
    num_of_lines = get_number_of_polylines(binary_img, min_line_len=15)
    imgs = np.zeros((img.shape[0], img.shape[1], num_of_lines), dtype=np.uint8)

    # finds external contour of each polyline
    contours, hierarchy = cv.findContours(binary_img, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]

    layer = 0  # current layer in 3d array where the contour should be saved
    for cnt in zip(contours, hierarchy):
        contour = cnt[0]  # current contour
        current_hierarchy = cnt[1]  # hierarchy of current contour

        if current_hierarchy[3] < 0:  # only contours of holes will have the parent in 2-lvl hierarchy
            mask = np.zeros((binary_img.shape[0], binary_img.shape[1]), dtype=np.uint8)
            cv.drawContours(mask, [contour], -1, 255, -1)

            current_img = cv.bitwise_and(binary_img, binary_img, mask=mask)  # image which contains only one contour
            imgs[:, :, layer] = current_img
            binary_img = binary_img - current_img  # remove contour from original img
            layer += 1

    return imgs


def linear_transform_points(points_px, points_real, axis=0):
    """
    Function which transform coordinates of the points in pixels to coordinates of the points in required units.

    :param points_px: list of points in pixels
    :type points_px : list
    :param points_real: list of expected values of points in required units
    :type points_real :list
    :param axis: x-axis: 0, y-axis: 1
    :type axis : int

    :return: searched values in required units
    :rtype: list
    """
    assert isinstance(points_px, list), f"`points_px` has to be type of `list`, `{type(points_px)}` given"
    assert isinstance(points_real, list), f"`points_real` has to be type of `list`, `{type(points_real)}` given"

    min_pos_real = points_real.index(min(points_real))  # position with minimal value in list of points_real
    max_pos_real = points_real.index(max(points_real))  # position with maximal value in list of points_real

    # real units increases from left bottom corner, but pixels in python from left upper corner
    # because of this, minimal and maximal found position on y-axis has to be switched
    if axis == 0:
        min_pos_px = points_px.index(min(points_px))  # position with minimal value in list of points_px
        max_pos_px = points_px.index(max(points_px))  # position with maximal value in list of points_px
    elif axis == 1:
        max_pos_px = points_px.index(min(points_px))  # position with minimal value in list of points_px
        min_pos_px = points_px.index(max(points_px))  # position with maximal value in list of points_px

    # Y - points in pixels
    # y - real points

    # coordinates linear transformation (two points needed)
    # y = ax + b
    # y1 = A*y1_px + B -> B = y1 - A*y1_px
    # y2 = A*y2_px + B ->  y2 = A*y2_px + y1 - A*y1_px
    
    # A = (y2 - y1)/(y2_px - y1_px)
    # Y_2 - searched value in pixels

    a = (points_real[min_pos_real] - points_real[max_pos_real]) / (points_px[min_pos_px] - points_px[max_pos_px])
    y1 = points_real[max_pos_real]
    y1_px = points_px[max_pos_px]

    searched_points = []
    for y2_px in points_px:
        y = a*y2_px + y1 - a*y1_px
        searched_points.append(int(round(y)))

    return searched_points


def find_peaks(data, ref_point=[0, 0], absolute_height=None, absolute_distance=None, relative_height=None,
               relative_distance=None, type_of_peaks=None):
    """
    Finds peaks in the data based on specified parameters.

    :param data: data which contains the peaks
    :type data :list
    :param ref_point: point from where the absolute height and width will be calculated
    :type ref_point : list, tuple
    :param absolute_height: required height of the peaks from reference point
    :type absolute_height : numeric types: int, float or sequence types: list, tuple
    :param absolute_distance: required distance of the peaks from reference point
    :type absolute_distance : numeric types: int, float or sequence types: list, tuple
    :param relative_height: required height of the each peak from its start to the end
    :type relative_height : numeric types: int, float or sequence types: list, tuple
    :param relative_distance: required distance between two peaks
    :type relative_distance : numeric types: int, float or sequence types: list, tuple
    :param type_of_peaks: describes which type of peaks is need as result, - (`max` - maximal, `min` - minimal)
    :type type_of_peaks: str

    :return: indices of peaks that satisfy all given conditions
    :type: list
    """
    assert isinstance(data, list), f"`data` has to be type of `list`, `{type(data)}` given"
    assert isinstance(ref_point, (list, tuple)), f"`ref_point` has to be type of `list` or `tuple`, `{type(ref_point)}` given"

    arguments = locals().copy()  # all current local params (dict)
    params = ["absolute_height", "absolute_distance", "relative_distance", "relative_height"]

    # verification if all arguments were set in correct format
    for key in arguments.keys():
        if key in params:
            if arguments[key] is not None:
                assert isinstance(arguments[key], (int, float, list, tuple)), f"{key} has to be numeric or sequence type, `{type(locals()[key])}` given"

                # subsequent modification of argument if only minimal value was set
                if isinstance(arguments[key], (int, float)) and 'height' in key:
                    arguments[key] = [arguments[key], arguments[key] + 2*max(data)]
                else:
                    arguments[key] = [arguments[key], arguments[key] + 2*len(data)]
            else:
                if 'height' in key:
                    arguments[key] = [0, 2*max(data)]
                else:
                    arguments[key] = [0, 2*len(data)]

    if type_of_peaks:
        assert isinstance(type_of_peaks, str), f"`type_of_peaks` has to be type of `str`, `{type(type_of_peaks)}` given"

    def get_peaks(selected_data):
        """
        Finds all peaks in selected data and divides them into maxima and minima.

        :param selected data: data in which peaks should be found
        :type selected_data : list

        :return: nested list in which peaks are divided into maxima and minima
        :type: list
        """
        peaks = [[], []]  # indices of found peaks [0] - max, [1] - min
        actual_flag = 0  # flag which indicates whether the data are increasing or decreasing (0 - decreasing, 1 - increasing)
        previous_flag = 0  # flag in which is stored previous state

        # setting previous state
        if data == selected_data:
            diff = data[1] - data[0]
        else:
            diff = data[selected_data[1]] - data[selected_data[0]]

        if diff > 0:
            previous_flag = 1

        for j in range(1, len(selected_data)-1):
            # calculates the difference between the current and the next value
            if data == selected_data:
                diff = data[j+1] - data[j]
            else:
                diff = data[selected_data[j+1]] - data[selected_data[j]]

            # changes the value of the flag according to calculated difference
            if diff == 0:
                continue
            elif diff > 0:
                actual_flag = 1
            else:
                actual_flag = 0

            # split found peaks to min and max peaks
            if previous_flag != actual_flag:
                # peaks with maximal values
                if actual_flag == 0:
                    peaks[0].append(j)
                # peaks with minimal values
                else:
                    peaks[1].append(j)
            previous_flag = actual_flag

        return peaks

    def get_local_minmax(selected_data, positions, flag):
        """
        Verifies if there are local maxima or minima in data between two final peaks.

        :param selected data: data in which peaks should be found
        :type selected_data : list
        :param positions: positions of all found peaks (all minima and maxima)
        :type positions :list
        :param flag: flag which indicates whether the maxima or minima are being sought (0 - maxima, 1 - minima)
        :type: bool

        :return: nested list in which peaks are divided to min and max
        :type: list
        """
        if not selected_data[0]:
            new_data = selected_data[1]  # evaluated data
        else:
            found_peaks = get_peaks(selected_data[0])

            # if flag indicates max, a local minimum is added to the selected data that contains the found max peaks and vice versa
            if flag == 0:
                new_data = selected_data[1]
                if found_peaks[1]:
                    for j in found_peaks[1]:
                        new_data.append(selected_data[0][j])

            else:
                new_data = selected_data[1]
                if found_peaks[0]:
                    for j in found_peaks[0]:
                        new_data.append(selected_data[0][j])

        new_data = list(set(new_data))
        new_data.sort()

        possible_peaks = get_peaks(new_data)
        final_peaks = []  # new peaks which were found in data

        # recalculation to total positions
        for peaks in possible_peaks:
            found_pos = []
            for pos in peaks:
                found_pos.append(new_data[pos])
            final_peaks.append(found_pos)

        return final_peaks

    def verify_relative_height(peaks, positions):
        """
        Verifies relative height of peaks and removes these which do not meet the condition.

        :param peaks: peaks which should be verified
        :type peaks : list
        :param positions: positions of all found peaks (all found minima and maxima)
        :type positions :list

        :return: all peaks which meet the condition
        :type: list
        """
        found_peaks = []

        for pos in peaks:
            rel_height = []  # relative height in format [left side, right side]
            pos_ = positions.index(pos)  # current position in all positions of found peaks

            # calculation of relative height from both sides
            rel_height.extend([abs(data[pos] - data[positions[pos_-1]]), abs(data[pos] - data[positions[pos_+1]])])
            rel_height = min(rel_height)

            # verification if smaller value of relative height meets the condition
            if rel_height >= arguments['relative_height'][0] and rel_height < arguments['relative_height'][1]:
                found_peaks.append(positions[pos_])

        return found_peaks

    def verify_relative_distance(peaks):
        """
        Verifies relative distance of peaks and removes these which do not meet the condition.

        :param peaks: peaks which should be verified
        :type peaks : list

        :return: all peaks which meet the condition
        :type: list
        """
        found_peaks = []

        # calculation of distance between first and last peak
        rel_dist = peaks[-1] - peaks[0]

        # verification of relative distance is lower/higher then requested one
        if rel_dist >= arguments['relative_distance'][0] and rel_dist < arguments['relative_distance'][1]:
            found_peaks.extend([peaks[0], peaks[-1]])

            # if number of found peaks is higher then 2, relative distance between each peak has to be verified
            if len(peaks) > 2:
                for j in range(1, len(peaks) - 1):

                    # calculation of distance between two peaks
                    rel_dist = []
                    rel_dist.extend([peaks[j] - peaks[j-1], peaks[j+1] - peaks[j]])
                    rel_dist = min(rel_dist)

                    if rel_dist >= arguments['relative_distance'][0] and rel_dist < arguments['relative_distance'][1]:
                        found_peaks.append(peaks[j])

        return found_peaks

    def verify_absolute_height_distance(peaks):
        """
        Verifies absolute distance and height of peaks from reference point and removes these which do not meet the condition.

        :param peaks: peaks which should be verified
        :type peaks : list

        :return: all peaks which meet the condition
        :type: list
        """
        result = [[], []]  # result in format [[maxima], [minima]]

        for j in range(len(peaks)):
            for peak in peaks[j]:

                # calculation of actual absolute height and distance from reference point
                abs_dist = peak - ref_point[0]  # determines the distance of the peak from the reference point
                abs_height = abs(data[peak] - ref_point[1])  # determines the peak height from the selected reference point

                # verification of absolute height and distance
                if abs_height >= arguments['absolute_height'][0] and abs_height < arguments['absolute_height'][1]:
                    if abs_dist >= arguments['absolute_distance'][0] and abs_dist < arguments['absolute_distance'][1]:
                        result[j].append(peak)

        return result

    # ----- main code -----
    found_peaks = get_peaks(data)

    # all found positions of vertices also together with the last point position
    positions = found_peaks[0] + found_peaks[1] + [len(data)-1]
    positions.sort()

    count = 0  # flag which indicates which part of found peaks is being processed (0 - max, 1 - min)
    result = []  # peaks that meet all conditions
    final_peaks = []  # found peaks for current processed part of all peaks

    for peaks in found_peaks:
        if peaks:
            final_peaks = verify_relative_height(peaks, positions)

            # first and last position of the data added to peaks which meet the requirements
            final_peaks += [0, len(data)-1]
            final_peaks.sort()

            # first and last position of the data added to found peaks
            peaks += [0, len(data)-1]
            peaks.sort()

            # verification if there are local maxima or minima between found peaks
            for j in range(1, len(final_peaks)):

                # selects peaks between two final peaks from peaks found
                selected_data = []
                for i in [count-1, count]:
                    selected_data.append([elem for elem in found_peaks[i] if (elem >= final_peaks[j-1] and elem < final_peaks[j])])

                    # adds current two final peaks to selected data a sorts it
                    for k in range(len(selected_data)):
                        selected_data[k].extend([final_peaks[j-1], final_peaks[j]])
                        selected_data[k].sort()

                # finds peaks in new data, get local max/min
                possible_peaks = get_local_minmax(selected_data, positions, count)

                # creates new positions of found peaks and sorts them
                new_positions =  possible_peaks[0] + possible_peaks[1] + [final_peaks[j-1]] + [final_peaks[j]]
                new_positions.sort()

                if count == 0:
                    new_final_peaks = verify_relative_height(possible_peaks[0], new_positions)
                else:
                    new_final_peaks = verify_relative_height(possible_peaks[1], new_positions)
                final_peaks.extend(new_final_peaks)

            # sorts the final peaks and removes first and last point
            final_peaks.sort()
            final_peaks = final_peaks[1:-1]

            # if there is more then one peak in final peaks, verifies relative distance between these peaks
            if len(final_peaks) > 1:
                final_peaks = verify_relative_distance(final_peaks)

            final_peaks.sort()
            result.append(final_peaks)

        else:
            result.append(final_peaks)
        count += 1

    # verification if found peaks meet the conditions of absolute height and distance
    result = verify_absolute_height_distance(result)

    # split solution according to selected type_of_peaks
    if type_of_peaks:
        if 'max' in type_of_peaks:
            return result[0]

        elif 'min' in type_of_peaks:
            return result[1]

    result = result[0] + result[1]
    result.sort()

    return result


def generate_ends_of_lines_kernels():
    """
    Generate 9x9 kernels which are needed for correct detection of ends of line in skeleton.
    All generated kernels can be found in https://legacy.imagemagick.org/Usage/morphology/#lineends.

    :return: kernels which represent end of line
    :type: list
    """
    # template of the kernel
    template = np.array((
            [-1, -1, -1],
            [-1, 1, -1],
            [-1, -1, -1]), dtype="int")

    kernels = []

    # creates kernels which represent end of line
    for x in range(template.shape[1]):
        for y in range(template.shape[0]):
            kernel = template.copy()
            kernel[x, y] = 1
            kernels.append(kernel)

    return kernels


def generate_intersection_kernels():
    """
    Generate 9x9 kernels which are needed for correct detection of line intersection in skeleton.
    All generated kernels can be found in https://legacy.imagemagick.org/Usage/morphology/#linejunctions.

    :return: kernels which represent line intersection
    :type: list
    """
    iterable = list(range(8))  # marked neighbors of selected point
    sizes = [3, 4]  # size of different combinations that are possible
    found_comb = []  # final found combinations, which meet the conditions
    banned_diff = 1  # serching only nonadjacent combinations, so difference beetween two elements must be higher

    for actual_size in sizes:
        for combination in itertools.combinations(iterable, actual_size):
            combination = list(combination)
            diff = np.diff(combination)  # difference beetween individual elements

            if not banned_diff in diff:
                # first and last element of the found combination can`t be 0 and 7
                # because these are adjacent elements in matrix 9x9
                if combination[-1] - combination[0] != iterable[-1] - iterable[0]:
                    found_comb.append(combination)

    # template of the kernel
    template = np.zeros((3, 3), dtype=np.uint8)
    template[1 , 1] = 1  # center pixel is always 1

    # maps the numbers from 0-7 into the 8 pixels surrounding of the center pixel in a 9 x 9 matrix clockwise. Up pixel = 0.
    positions = [(0,1), (0,2), (1,2), (2,2), (2,1), (2,0), (1,0), (0,0)]

    kernels = []
    # creates kernels according to obtained combinations
    for comb in found_comb:
        kernel = template.copy()
        for elem in comb:
            kernel[positions[elem][0], positions[elem][1]] = 1
        kernels.append(kernel)

    return kernels


def find_line_intersection(img):
    """
    Returns image with found line intersections.

    :param array_like img: cv image
    :return: binary image with found intersections
    """
    img = create_binary_img(img)

    # new img where found intersecion will be drawn.
    output_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    kernels = generate_intersection_kernels()
    for j in range(len(kernels)):
        out = cv.morphologyEx(img, cv.MORPH_HITMISS, kernels[j])
        output_img += out

    return output_img


def find_ends_of_lines(img):
    """
    Returns image with found ends of lines.

    :param array_like img: cv image
    :return: binary image with found ends of lines
    """
    img = create_binary_img(img)

    # new img where found intersecion will be drawn.
    output_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    kernels = generate_ends_of_lines_kernels()
    for j in range(len(kernels)):
        out = cv.morphologyEx(img, cv.MORPH_HITMISS, kernels[j])
        output_img += out

    return output_img


def get_num_of_white_pixels_for_each_direction(img, point):
    """
    At the selected point, the space for each angle in the range (0, 360, step) will be searched gradually.
    For each angle, the number of consecutive white pixels is determined from the starting point
    and the coordinates of the last white point in that direction are also determined.

    :param array_like img: img that contains a modified line
    :param point: coordinates of the point
    :type point : list e.g. [x1, y2]

    :return:
        - data - number of consecutive white pixels for each direction
        - coordinates - coordinates of last white point in each direction
    :type:
        - list
        - list
    """
    assert isinstance(point, list), f"`point` has to by type list, {type(point)} given."

    data = []  # list of number of white pixels, which were found in each verified direction
    all_coord = [] # list of all coordinates of the points at which the line ends, one for each direction

    step = 0.5  # value by which the angle will be changed, in degrees
    counter_length = 5 # length of list which will be contain values of several consecutive pixels that have been verified

    # for each direction finds number of consecutive white pixels from starting point
    # and also stores the coordinates of the - list
    for angle in np.arange(0, 360, step):

        t = 1 # line parameter
        out_of_img = 0  # out of image flag for line movement
        count = 0  # variable which contains actual number of white pixels, which were found in selected direction

        counter = [1]  # list which contains values of several consecutive pixels that have been verified (true-pixel is white, false-pixel is black)
        angle_rad = np.pi / 180 * angle  # angle in radians, it is needed for numpy trigonometric functions

        while True:
            # parametric form of line
            x = point[0] + t * np.cos(angle_rad)
            y = point[1] + t * np.sin(angle_rad)

            real_x = int(x)
            real_y = int(y)

            # if the calculated value of y is within the image range, it is verified whether the given pixel on [real_x, real_y] is white
            if (real_y > 0 and real_y < img.shape[0]) and (real_x > 0 and real_x < img.shape[1]):
                if img[real_y, real_x] == 255:
                    count += 1
                    counter.append(1)
                else:
                    counter.append(0)
            else:
                out_of_img = 1

            if len(counter) > counter_length:
                counter.pop(0)

            # if sum of values several consecutive pixels is zero or it is the end of the image,
            # cycle ends and coordinates of last white pixel are saved
            if len(counter) == counter_length or out_of_img:
                if sum(counter) == 0 or out_of_img:
                    # shift of line parameter t
                    if out_of_img:
                        shift = 1
                    else:
                        shift = len(counter) + 1

                    # coordinates of last point in the img
                    x = point[0] + (t - shift) * np.cos(angle_rad)
                    y = point[1] + (t - shift) * np.sin(angle_rad)

                    real_x = max(0, int(round(x)))
                    real_y = max(0, int(round(y)))

                    all_coord.append([real_x, real_y])
                    break
            t += 1
        data.append(count)

    return data, all_coord


def get_coordinates(current_path, paths, img, count, banned_angles=[]):
    """
    In the data created for a particular point, it finds the angles at which the number of white pixels is largest.
    For each such angle, it is checked whether the last white pixel in the given direction is new or not and adds it to the current path.

    :param current_path: found points which define part of polyline in format: [[x1,y1], [x2,y2],...]]
    :type current_path : nested list
    :param paths: found points which define polyline in format: [[[x1,y1], [x2,y2],...],[[x1,y1], [x2,y2],...]]
    :type paths : nested list
    :param array_like img: part of the image with polyline
    :param banned_angles: angles that have already been evaluated
    :type banned_angles : list
    :param count: number of function calls
    :type: int

    :returns the polyline points that are stored as paths so that the original polyline can be reconstructed 
    :rtype: nested list
    """
    assert isinstance(paths, list), f"`paths` has to be type of `list`, `{type(paths)}` given"

    # if the end of the line is not found within count = 50, the line search ends
    if count > 50:
        paths.append(current_path)
        return paths

    # gets number of white points and coordinates of last white point in each direction
    data, all_coord = get_num_of_white_pixels_for_each_direction(img, current_path[-1])

    # the directions in which the most white pixels were found
    peaks = find_peaks(data, absolute_height=15, relative_height=15, relative_distance=5, type_of_peaks='max')

    # if peaks exist at the beginning or end of the data, they will be added to found peaks
    peaks = add_begin_end_peaks(data, peaks)

    angles = []  # list of found angles
    angles.extend((360 + np.array(peaks) / 2) % 360)  # all found angles which should be verified
    end_of_line = True  # flag for dead end

    for angle in angles:
        # coordinates of the last white point for the current angle
        coord = all_coord[int(angle*2)]

        # calculation of angle tolerance
        # a^2 + b^2 = c^2
        line_length = math.sqrt(abs(coord[0] - current_path[-1][0]) ** 2 + abs(coord[1] - current_path[-1][1]) ** 2)
        # atan(line_width / line_length) conversion to degrees
        tolerance = round(180 / np.pi * np.arctan(get_polyline_width(img) / line_length)) + 5

        # verifies if selected angle is in banned angles
        if banned_angles:
            for banned_angle in banned_angles:
                diff = abs(banned_angle - angle)
                diff = min(diff, 360 - diff)

                if diff < tolerance:
                    break
        else:
            diff = tolerance + 1

        # if angle is not in banned angles, found coordinates for current angle are added to line coordinates
        if diff > tolerance:
            end_of_line = True

            # verification if found point is the new one
            if paths:
                for path in paths:
                    exist_in_paths = check_point_on_line_segment(coord, path, line_width=10)
                    if exist_in_paths:
                        break
            else:
                exist_in_paths = False  # flag which indicates whether the found coordinates are already in found paths

            exist_in_current_path = check_point_uniqueness(coord, current_path, surrounding=10)  # flag which indicates whether the found coordinates are already in current path
            current_path.append(coord)

            if not exist_in_paths and not exist_in_current_path:
                end_of_line = False

                # recursion
                paths = get_coordinates(current_path.copy(), paths.copy(), img, count+1, [(angle + 180) % 360])

                banned_angles.append(angle)  # current angle has been already checked, so it is added to banned angles
                current_path.pop(-1)  # closed recursion, last point has to be removed
            else:
                # if found point already exists it is necessary to verify if current path is unique
                if paths:
                    for path in paths:
                        match = 0  # the number of points that match

                        for j in range(len(current_path)):
                            exist_in_current_path = check_point_uniqueness(current_path[j], path, surrounding=10)
                            if exist_in_current_path:
                                match += 1
                            else:
                                break

                        # if path is not unique only point where path starts is saved
                        if match == len(current_path):
                            current_path = [current_path[-2]]
                            break
                    else:
                        current_path.append(current_path[-2])
                else:
                    current_path.append(current_path[-2])

    # the last point is deleted if the same point is in (position - 2)
    if len(current_path) > 2:
        if current_path[-1] == current_path[-3]:
            current_path.pop(-1)

    # if end of line is found and all conditions are met current path is added to paths
    if end_of_line and angles:
        if len(current_path) > 1:
            append = True  # flag to add a path to all found paths 

            # if there are already some paths, then only the part of the newly found path
            # that does not belong to the paths will be added
            if paths:
                for path in paths:
                    for j in range(len(path)-1):
                        if path[j+1] != current_path[j+1]:
                            current_path = current_path[j:]
                            break

                # verifying that the found path exists in the paths,
                # if it already exists, it will not be added to them
                for path in paths:
                    match = 0  # the number of points that match

                    for j in range(len(current_path)):
                        exist_in_current_path = check_point_on_line_segment(current_path[j], path, line_width=10)
                        if exist_in_current_path:
                            match += 1
                        else:
                            break

                    if match == len(current_path):
                        append = False

            if append:
                paths.append(current_path)

    return paths


def add_missing_points(img, paths):
    """
    Checks if there is another part of the polyline whose coordinates have not been added,
    and if such a part exists, it will add the missing coordinates to the currently found paths.

    :param array_like img: cv image with curve
    :param paths: found points which define polyline in format: [[[x1,y1], [x2,y2],...],[[x1,y1], [x2,y2],...]]
    :type paths : nested list

    :return all found coordinates of the polyline
    :type: nested list
    """
    assert isinstance(paths, list), f"`paths` has to be type of `list`, `{type(paths)}` given"

    # image dimension
    width = img.shape[1]
    height = img.shape[0]

    # draw line to img according to assigned coordinates
    new_img = np.zeros(img.shape)
    for line in paths:
        for j in range(len(line)-1):
            new_img = cv.line(new_img, tuple(line[j]), tuple(line[j+1]), 255, 10)

    # creates skeleton from input img
    skeleton = thinning(img)

    # draws ends of lines to new img
    ends_of_lines = find_ends_of_lines(skeleton)

    # coordinates which should be added to found coordinates but in first searching they were not found
    new_peaks = ends_of_lines - cv.bitwise_and(new_img, new_img, mask=ends_of_lines)
    miss_coord = []

    for actual_x in range(width):
        for actual_y in range(height):
            if new_peaks[actual_y, actual_x] > 0:
                miss_coord.append([actual_x, actual_y])

    for coord in miss_coord:
        paths = get_coordinates([coord.copy()], paths.copy(), img, 0, banned_angles=[])

    return paths


def check_point_uniqueness(point, existing_points, surrounding=0):
    """
    Returns flag which indicates whether the current point is in surrounding of another point or not.

    :param point: coordinates of the point in format [x, y]
    :type point : list
    :param existing_points: existing points with which the current point has to be compared e.g. [[x1, y1], [x2, y2],...]
    :type existing_points : nested list
    :param surrounding: tolerance in which the verified point is considered to be an existing point
    :type surrounding : int

    :return flag which indicates if selected point should be added to existing points or not
    :type: bool
    """
    assert isinstance(point, list), f"`point` has to be type of `list`, `{type(point)}` given"
    assert len(point) == 2, f"`point` shall be list with two elements, {point} ({len(point)} elements) given"

    assert isinstance(existing_points, list), f"`existing_points` has to be type of `list`, `{type(existing_points)}` given"
    assert any(isinstance(p, list) for p in existing_points), f"all elements of `existing_points` has to be type of `list`"

    assert isinstance(surrounding, int), f"`surrounding` has to be type of `int`, `{type(surrounding)}` given"

    for ref_point in existing_points:
        # generates close surrounding of the reference point
        surrounding_x = [ref_point[0] - surrounding, ref_point[0] + surrounding]
        surrounding_y = [ref_point[1] - surrounding, ref_point[1] + surrounding]

        if point[0] <= surrounding_x[1] and point[0] >= surrounding_x[0]:
            if point[1] <= surrounding_y[1] and point[1]  >= surrounding_y[0]:
                return True

    return False


def check_point_on_line_segment(point, existing_points, line_width=0):
    """
    Returns flag which indicates whether the current point is between two points.

    :param point: coordinates of the point in format [x, y]
    :type point : list
    :param existing_points: existing points between which it is verified whether the given point lies or not e.g. [[x1, y1], [x2, y2],...]
    :type existing_points : nested list
    :param line_width: tolerance at which the verified point is considered to lie between the lines
    :type line_width : int

    :return flag which indicates if selected point should be added to existing points or not
    :type :bool
    """
    assert isinstance(point, list), f"`point` has to be type of `list`, `{type(point)}` given"
    assert len(point) == 2, f"`point` shall be list with two elements, {point} ({len(point)} elements) given"

    assert isinstance(existing_points, list), f"`existing_points` has to be type of `list`, `{type(existing_points)}` given"
    assert any(isinstance(p, list) for p in existing_points), f"all elements of `existing_points` has to be type of `list`"
    assert len(point) >= 2, f"`existing_points` shall be list with minimal two points, {point} ({len(point)} points) given"

    assert isinstance(line_width, int), f"`line_width` has to be type of `int`, `{type(line_width)}` given"

    for j in range(len(existing_points) - 1):
        a = existing_points[j]
        b = existing_points[j+1]

        # vector perpendicular to direction vector
        # direction vector: u = [b_x - a_x, b_y - a_y] -> perpendicular vector v = [b_y - a_y, a_x - b_x]
        v_x = b[1] - a[1]
        v_y = a[0] - b[0]
        v = np.array([v_x, v_y])

        # norm calculation
        norm = math.sqrt(v_x**2 + v_y**2)

        v = (v / norm) * (line_width / 2)
        contour = []  # contour

        # calculation of border points
        for u in [a, b]:
            for t in [-1, 1]:
                x = round(u[0] + v[0] * t)
                y = round(u[1] + v[1] * t)

                contour.append([x, y])

        # correct points order for 
        contour = np.array([contour[0], contour[2], contour[3], contour[1]], dtype=np.int0)

        inside = cv.pointPolygonTest(contour, tuple(point), False)

        if inside > 0:
            return True

    return False


def add_begin_end_peaks(data, peaks):
    """
    The original data is shifted so that it is possible to verify
    whether there is a peak at the beginning or end of the original data.

    : param data: data which contains the peaks
    : type data : list
    : param peaks: positions of already found peaks in sent data
    : type peaks : list

    : return peaks which are extended by peaks found at the beginning or end if such vertices exist
    : rtype: list
    """
    assert isinstance(data, list), f"`data` has to be type of `list`, `{type(data)}` given"
    assert isinstance(peaks, list), f"`peaks` has to be type of `list`, `{type(peaks)}` given"

    shift = 50  # shifts data by 50 positions
    new_data = data[shift:] + data[0:shift]  # shifted data
    new_peaks = find_peaks(new_data, absolute_height=15, relative_height=5, relative_distance=15, type_of_peaks='max')

    found_peaks = []  # list of peaks found in `new_peaks`

    if new_peaks is not None:
        for peak in new_peaks:
            peak = peak + shift  # moves the vertex to its original position

            if peak >= len(new_data):
                peak -= 360 * 2

            found_peaks.append(peak)

    # if the peak exists in `peaks` but was not found after shifting the data,
    # it will be removed, because it describes only the local maximum
    for peak in peaks:
        if (peak < shift) or (peak > (360 * 2 - shift)):
            if peak not in found_peaks:
                peaks.pop(peaks.index(peak))

    peaks += found_peaks
    peaks = list(set(peaks))
    peaks.sort()

    return peaks


def get_polyline_coordinates(img, box, scale=1, threshold=125, blank=None, suppress_noise=False, max_line_gab=5):
    """
    For each line in the selected area, obtain a list of coordinates from which it is possible to re-construct the original line.

    :param img: source image
    :type img : cv2 image (b,g,r matrix)
    :param box: boundaries of intrested area
    :type box : [left, top, right, bottom]
    :param scale: target scaling
    :type scale : float
    :param blank: areas which shall be masked e.g.[[10, 20, 30, 40], [], ...]
    :type blank : list of lists
    :param suppress_noise: if there is noise around the line, such as characters, it will try to remove that noise (True) else (False)
    :type suppress_noise : bool
    :param max_line_gab: maximum allowed gap between points on the same line to link them
    :type max_line_gab : int
    """
    roi = bretina.crop(img, box, scale)

    # image dimension
    width = roi.shape[1]
    height = roi.shape[0]

    # creates binary img
    binary_img = create_binary_img(roi, threshold=threshold)

    if blank is not None:
        assert isinstance(blank, (list, tuple, set, frozenset)), "`blank` has to be list"
        assert isinstance(blank[0], (list, tuple, set, frozenset)), "`blank area` has to be list"

        # from all areas which shall be masked white points are removed
        for area in blank:

            # check if selected area consists of 4 points
            assert len(area) == 4, "`selected blank area does not have enough points (four points needed)"

            # resize blanked areas depending on scale
            start_x = int(round(area[0]*scale))
            start_y = int(round(area[1]*scale))

            end_x = int(round(area[2]*scale))
            end_y = int(round(area[3]*scale))

            # white points removal
            binary_img[start_y:end_y, start_x:end_x] = 0

    if suppress_noise:
        # removes salt and pepper noise
        binary_img = cv.medianBlur(binary_img, 3)

        # removes lines which are smaller then 15 px
        min_line_len = 15
        contours, _ = cv.findContours(binary_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv.arcLength(cnt, False) < min_line_len:
                cv.drawContours(binary_img, [cnt], -1, (0, 0, 0), -1)

    # remove black spaces between lines
    if (max_line_gab % 2) != 0 or max_line_gab == 0:
        max_line_gab += 1
    iterations = int(max_line_gab / 2)

    kernel1 = np.ones((3, 3), np.uint8)
    kernel2 = np.array([[0,1,0], [1,1,1], [0,1,0]], np.uint8)

    img = cv.dilate(binary_img, kernel1, iterations=iterations)
    img = cv.erode(img, kernel2, iterations=iterations)

    # creates a separate image for each curve in the image
    imgs = split_polylines_to_seprate_imgs(img)

    paths = []  # all found coordinates

    # finds coordinates for each line in img separately
    for j in range(imgs.shape[2]):

        # line in current img is modified to required width
        img = imgs[:, :, j]
        img = change_polyline_width(img, required_width=6)

        # draw black border
        img = cv.copyMakeBorder(img, 1, 1, 1, 1, cv.BORDER_CONSTANT, None, 0)

        # gets first point of the polyline and adds it to found coordinates
        first_point = get_polyline_first_point(img)
        coordinates = [[coord + 1 for coord in first_point]]

        if not any(coordinates):
            continue

        # paths represent nested list, each list in represent one found path
        paths = get_coordinates(coordinates.copy(), paths.copy(), img, 0, banned_angles=[])
        paths = add_missing_points(img, paths.copy())

    assert any(paths), f'No paths for selected img exist, verify that polyline is wider than 1 [px] and longer then 15 [px]'

    return paths
