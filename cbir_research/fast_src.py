import cv2
import math
import fast.fast9 as fast
import numpy as np
import time
import itertools
import csv
import imutils
from pymongo import MongoClient
import pickle
# from bson.binary import Binary
from matplotlib import pyplot as plt
from os.path import exists
import os
import threading
import os
import multiprocessing

import concurrent.futures


class ThreadWithReturnValue(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                        **self._kwargs)

    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return


# show image on new window
def show_image(win, img):
    cv2.imshow('{}'.format(win), img)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('s'):
        cv2.imwrite('{}.jpg'.format(win), img)
        cv2.destroyAllWindows()


# calculate the intensities of the 16 surrounding pixels
def get_pixels(img, y, x, threshold):
    px = img[y, x]
    pixel_circle = [
        int(img[y, x + 3]) - px, int(img[y + 1, x + 3]) - px, int(img[y + 2, x + 2]) - px, int(img[y + 3, x + 1]) - px,
        int(img[y + 3, x]) - px, int(img[y + 3, x - 1]) - px, int(img[y + 2, x - 2]) - px, int(img[y + 1, x - 3]) - px,
        int(img[y, x - 3]) - px, int(img[y - 1, x - 3]) - px, int(img[y - 2, x - 2]) - px, int(img[y - 3, x - 1]) - px,
        int(img[y - 3, x]) - px, int(img[y - 3, x + 1]) - px, int(img[y - 2, x + 2]) - px, int(img[y - 1, x + 3]) - px]

    compared_list = [1 if item > threshold else -1 if item < -threshold else 0 for item in pixel_circle]

    return compared_list


'''
fast detection algorithm:
iterate through each pixel in the image and find pixels that contain x number of consecutive surrounding pixels
with itensities either above or below the threshold.   
'''


def fast_multi_scale_setup(img, level, n, sf, q1):
    _kp, _ = fast_test(img, n, threshold=20, non_max=1)
    harris_kp = harris_corner(img, _kp, n, 0.04)
    kp_scaled_up = []
    for i in range(len(harris_kp)):
        x = int(harris_kp[i][0] * (sf ** level))
        y = int(harris_kp[i][1] * (sf ** level))
        kp_scaled_up.append((x, y))
    q1.put(kp_scaled_up)

def set_up_scales_multi(img, scale_factor, nlevels, nfeatures):
    orb_kp = run_orb(img, nfeatures)
    factor = 1.0 / scale_factor
    ndesired_features_per_scale = nfeatures * (1 - factor) / (1 - factor ** nlevels)

    sum_features = 0
    nfeatures_per_level = []
    for level in range(nlevels - 1):
        nfeatures_per_level.append(round(ndesired_features_per_scale))
        sum_features += nfeatures_per_level[level]
        ndesired_features_per_scale *= factor
    nfeatures_per_level.append(max(nfeatures - sum_features, 0))
    rows, cols = img.shape
    _matched_list = []
    ratio_list = []
    _keypoints_list = []
    scaled_images = []

    for level in range(nlevels):
        w = int(cols / scale_factor ** level)
        scaled_images.append(imutils.resize(img, width=w))



    q1 = multiprocessing.Queue()
    threads = []

    for i in range(8):
        threads.append(multiprocessing.Process(target=fast_multi_scale_setup, args=(
        scaled_images[i], i, nfeatures_per_level[i] * 2, scale_factor, q1)))

    for i in threads:
        i.start()

    for i in threads:
        i.join()

    for i in range(8):
        _keypoints_list+= q1.get()
    print(_keypoints_list)
    print(len(_keypoints_list))
    show_image('', mark_keypoints(_keypoints_list, img))

    #
    # for level in range(nlevels):
    #     print(level)
    #     current_kp = []
    #     scaled_img = img
    #     w = int(cols / scale_factor ** level)
    #     scaled_img = imutils.resize(scaled_img, width=w)
    #     start = time.process_time()
    #     _kp, _ = fast_test(scaled_img, nfeatures_per_level[level] * 2, threshold=20, non_max=1)
    #     # print("Processing time:", time.process_time() - start)
    #     # show_image('fast at: {}'.format(level), mark_keypoints(_kp, scaled_img))
    #     harris_kp = harris_corner(scaled_img, _kp, nfeatures_per_level[level], 0.04)
    #     for i in range(len(harris_kp)):
    #         x = int(harris_kp[i][0] * (scale_factor ** level))
    #         y = int(harris_kp[i][1] * (scale_factor ** level))
    #         _keypoints_list.append((x, y))
    #         current_kp.append((x, y))
    #     # show_image('Scale level {}'.format(level + 1), mark_keypoints(current_kp, img))
    #     # print('current # of kp: ', len(_keypoints_list))
    #     # if level % 2 == 0:
    #     matched_points = get_matched_point(orb_kp, _keypoints_list)
    #     # get_average_distance(_keypoints_list, orb_kp)
    #     # print('cumulative matches: ', len(matched_points))
    #     _matched_list.append(len(matched_points))
    # for level in range(nlevels):
    #     ratio_list.append(_matched_list[level] / max(_matched_list))
    #     # if len(matched_points) > nfeatures/4:
    #     #     break
    #     # some conversion to OPENCV's keypoint object
    #     # pass them into brief
    #     # show_image('harris at: {}'.format(level), mark_keypoints(harris_kp, scaled_img))
    # print('length', len(_keypoints_list))
    # return _keypoints_list, ratio_list
    return _keypoints_list, ''

# def set_up_scales(img, scale_factor, nlevels, nfeatures):
#     orb_kp = run_orb(img, nfeatures)
#     factor = 1.0 / scale_factor
#     ndesired_features_per_scale = nfeatures * (1 - factor) / (1 - factor ** nlevels)
#
#     sum_features = 0
#     nfeatures_per_level = []
#     for level in range(nlevels - 1):
#         nfeatures_per_level.append(round(ndesired_features_per_scale))
#         sum_features += nfeatures_per_level[level]
#         ndesired_features_per_scale *= factor
#     nfeatures_per_level.append(max(nfeatures - sum_features, 0))
#
#     _keypoints_list = []
#     rows, cols = img.shape
#     _matched_list = []
#     ratio_list = []
#
#     quadrant = 0
#     for level in range(nlevels):
#         current_kp = []
#         scaled_img = img
#         w = int(cols / scale_factor ** level)
#         scaled_img = imutils.resize(scaled_img, width=w)
#         print('shape', scaled_img.shape)
#         row, col = scaled_img.shape
#
#         if quadrant == 0:
#             scaled_img = scaled_img[:row // 2, :col // 2]
#         elif quadrant == 1:
#             scaled_img = scaled_img[:row // 2, col // 2:]
#         elif quadrant == 2:
#             scaled_img = scaled_img[row // 2:, : col // 2]
#         elif quadrant == 3:
#             scaled_img = scaled_img[row // 2:, col // 2:]
#         show_image('quadrant', scaled_img)
#         start = time.process_time()
#         _kp, _ = fast_test(scaled_img, nfeatures_per_level[level] * 2 // 4, threshold=20, non_max=1)
#         print("Processing time:", time.process_time() - start)
#         # show_image('fast at: {}'.format(level), mark_keypoints(_kp, scaled_img))
#         harris_kp = harris_corner(scaled_img, _kp, nfeatures_per_level[level] // 4, 0.04)
#         for i in range(len(harris_kp)):
#             x = harris_kp[i][0]
#             y = harris_kp[i][1]
#             if quadrant == 1:
#                 x = x + col // 2
#             elif quadrant == 2:
#                 y = y + row // 2
#             elif quadrant == 3:
#                 x = x + col // 2
#                 y = y + row // 2
#             x = int(x * (scale_factor ** level))
#             y = int(y * (scale_factor ** level))
#             _keypoints_list.append((x, y))
#             current_kp.append((x, y))
#         show_image('Scale level {}'.format(level + 1), mark_keypoints(current_kp, img))
#         print('current # of kp: ', len(_keypoints_list))
#         # if level % 2 == 0:
#         matched_points = get_matched_point(orb_kp, _keypoints_list)
#         get_average_distance(_keypoints_list, orb_kp)
#         print('cumulative matches: ', len(matched_points))
#         _matched_list.append(len(matched_points))
#         quadrant += 1
#         if quadrant > 3:
#             quadrant = 0
#
#     for level in range(nlevels):
#         ratio_list.append(_matched_list[level] / max(_matched_list))
#         # if len(matched_points) > nfeatures/4:
#         #     break
#         # some conversion to OPENCV's keypoint object
#         # pass them into brief
#         # show_image('harris at: {}'.format(level), mark_keypoints(harris_kp, scaled_img))
#     print('length', len(_keypoints_list))
#     return _keypoints_list, ratio_list


def show_metric():
    average = get_metric()
    levels = [x for x in range(0, 9)]
    lines = plt.plot(levels, average)
    plt.setp(lines, color='blue', linewidth=3.0)
    plt.ylabel('% of matches found')
    plt.xlabel('scale')
    plt.axis([0, 8, 0, 100])
    plt.grid(color='black', linestyle='-', linewidth=1)
    plt.show()


def get_metric():
    total_ratio = []
    row_length = 14

    for i in range(row_length):
        imgpath = 'box/frames/frame_{}.png'.format(i)
        img = cv2.imread(imgpath, 0)
        keypoints, ratio = set_up_scales(img, scale_factor=1.2, nlevels=8, nfeatures=500)
        total_ratio.append((ratio))

    average = 8 * [0]
    for col in range(8):
        for row in range(row_length):
            average[col] += total_ratio[row][col]
        average[col] /= row_length
        average[col] *= 100
    average.insert(0, 0)
    return average


def fast_test(img, n, threshold, non_max):
    fast_n = 9
    keypoints = []
    scores = []
    rows, cols = img.shape
    y_start = 4
    y_end = rows - 4
    x_start = 4
    x_end = cols - 4

    for y in range(y_start, y_end):
        for x in range(x_start, x_end):
            pixel_list = get_pixels(img, y, x, threshold)
            if pixel_list.count(-1) >= fast_n or pixel_list.count(1) >= fast_n:
                consecutive = [len(list(g)) for _, g in itertools.groupby(pixel_list)]
                if pixel_list[0] == pixel_list[-1]:
                    if len(consecutive) > 1:
                        consecutive[0] += consecutive.pop()
                if max(consecutive) >= fast_n:
                    keypoints.append((x, y))
            else:
                continue
    scores = [corner_score(img, keypoints[i][0], keypoints[i][1]) for i in range(len(keypoints))]

    if non_max:
        sc = np.zeros(img.shape)
        for i in range(len(keypoints)):
            sc[keypoints[i][1], keypoints[i][0]] = scores[i]

        nonmax_corners = []
        nonmax_scores = []

        for i in range(len(keypoints)):
            x = keypoints[i][0]
            y = keypoints[i][1]
            s = scores[i]
            if s >= sc[y - 1][x + 1] and s >= sc[y - 1][x] and s >= sc[y - 1][x - 1] and s >= sc[y][x + 1] \
                    and s >= sc[y][x - 1] and s >= sc[y + 1][x + 1] and s >= sc[y + 1][x] and s >= sc[y + 1][x - 1]:
                nonmax_corners.append((x, y))
                nonmax_scores.append(s)
        zipped_pairs = zip(nonmax_scores, nonmax_corners)
        nonmax_corners = [x for _, x in sorted(zipped_pairs, reverse=True)]
        nonmax_corners = nonmax_corners[0: n]
        return nonmax_corners, nonmax_scores
    return keypoints, scores


def is_a_corner(img, x, y, b):
    fast_n = 9
    pixel_list = get_pixels(img, y, x, b)

    if pixel_list.count(-1) >= fast_n or pixel_list.count(1) >= fast_n:
        consecutive = [len(list(g)) for _, g in itertools.groupby(pixel_list)]
        if pixel_list[0] == pixel_list[-1]:
            if len(consecutive) > 1:
                consecutive[0] += consecutive.pop()
        if max(consecutive) >= fast_n:
            return 1
    else:
        return 0
    return 0


def corner_score(img, x, y):
    bmin = 0
    bmax = 255
    b = (bmax + bmin) / 2

    while True:
        if is_a_corner(img, x, y, b):
            bmin = int(b)
        else:
            bmax = int(b)
        if bmin == bmax - 1 or bmin == bmax:
            return bmin
        b = (bmin + bmax) / 2


# def gradient_x(img):
#     kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
#     return sig.convolve2d(img, kernel_x, mode='same')
#
#
# def gradient_y(img):
#     kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
#     return sig.convolve2d(img, kernel_y, mode='same')


def harris_corner(img, keypoints, n, k, window_size=9):
    harris_list = []
    threshold = 10000
    # I_x = gradient_x(img)
    # I_y = gradient_y(img)
    # Ixx = ndi.gaussian_filter(I_x ** 2, sigma=1)
    # Ixy = ndi.gaussian_filter(I_y * I_x, sigma=1)
    # Iyy = ndi.gaussian_filter(I_y ** 2, sigma=1)
    dy, dx = np.gradient(img)

    Ixx = dx ** 2
    Ixy = dy * dx
    Iyy = dy ** 2
    offset = int(window_size / 2)
    for item in keypoints:
        x = item[0]
        y = item[1]
        windowIxx = Ixx[y - offset: y + offset + 1, x - offset: x + offset + 1]
        windowIxy = Ixy[y - offset: y + offset + 1, x - offset: x + offset + 1]
        windowIyy = Iyy[y - offset: y + offset + 1, x - offset: x + offset + 1]
        Sxx = windowIxx.sum()
        Sxy = windowIxy.sum()
        Syy = windowIyy.sum()
        det = (Sxx * Syy) - (Sxy ** 2)
        trace = Sxx + Syy
        r = det - k * (trace ** 2)
        # print(r)
        if r > threshold:
            harris_list.append((item[0], item[1], r))
    harris_list.sort(key=lambda x: x[2], reverse=True)
    harris_list = harris_list[:n]

    return harris_list


def intensity_centroid(img, keypoints, patchsize):
    half_patchsize = patchsize // 2
    orientation_list = []
    for i in keypoints:
        x = i[0]
        y = i[1]
        gray_patch = img[y - half_patchsize: y + half_patchsize + 1, x - half_patchsize: x + half_patchsize + 1]
        ret, thresh = cv2.threshold(gray_patch, 127, 255, 0)
        M = cv2.moments(thresh)
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        # print('cX, cY in 31 x 31: {}'.format((cX, cY)))
        color_patch = cv2.cvtColor(gray_patch, cv2.COLOR_GRAY2RGB)
        cv2.circle(color_patch, (cX, cY), 1, (255, 255, 255), -1)
        cX = x + (cX - 16)
        cY = y + (cY - 16)

        print('cX, cY in original image: {}'.format((cX, cY)))
        orientation = math.atan2((cY - y) * -1, cX - x)
        cv2.circle(color_patch, (16, 16), 1, (0, 0, 255), -1)
        print('keypoint in original image: {}'.format((x, y)))
        print('in radians: ', orientation)
        print('in degrees: ', math.degrees(orientation))
        # show_image('patch', color_patch)
        orientation_list.append(orientation)
    return orientation_list

    # mu = [None] * len(kp)
    # for i in range(len(kp)):
    #     mu[i] = cv2.moments(kp[i])
    # print(mu)
    # mc = [None] * len(kp)
    # for i in range(len(kp)):
    #     mc[i] = math.atan2(mu[i]['m01'] / (mu[i]['m00'] + 1e-5), mu[i]['m10'] / (mu[i]['m00'] + 1e-5))
    # return mc


def get_average_distance(kp1, kp2):
    sum = 0
    for i in kp1:
        distance = []
        for j in kp2:
            distance.append(euclidean_distance((i[0], i[1]), (j[0], j[1])))
        sum += min(distance)

    print('Average distance with matching points', (sum / len(kp1)))
    return sum / len(kp1)


def get_minimum_distance(kp1, kp2):
    min_list = []

    for i in kp1:
        min_d = []
        for j in kp2:
            min_d.append(euclidean_distance((i[0], i[1]), (j[0], j[1])))
        min_list.append(min(min_d))

    return min_list


def get_matched_point(kp1, kp2):
    # count = 0
    # matching_points = []
    # for i in kp1:
    #     if (i[0], i[1]) in kp2:
    #         count += 1
    #         matching_points.append(i)
    # return matching_points
    count = 0
    for i in kp1:
        distance = []
        for j in kp2:
            distance.append(euclidean_distance((i[0], i[1]), (j[0], j[1])))
        if min(distance) < 5:
            count += 1
    # print('Points found within 5 pixels', count)
    return count


def mark_keypoints(kp, img):
    color_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(len(kp)):
        color_image = cv2.circle(color_image, (kp[i][0], kp[i][1]), 3, (0, 0, 255), -1)
    return color_image


# used to check the euclidean distance between points
def euclidean_distance(p, q):
    return math.sqrt((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2)


def draw_histogram(distance):
    print(distance)
    under_five_count = [0] * 5
    for i in distance:
        if i < 1:
            under_five_count[0] += 1
        elif i < 2:
            under_five_count[1] += 1
        elif i < 3:
            under_five_count[2] += 1
        elif i < 4:
            under_five_count[3] += 1
        elif i < 5:
            under_five_count[4] += 1
    print(under_five_count)
    arr = np.array(distance)
    a = np.hstack(arr)
    plt.ylabel('Number of keypoints')
    plt.xlabel('Distance from nearest matched keypoints (in pixels)')
    _ = plt.hist(a, bins='auto', color='blue')
    plt.show()
    return under_five_count


def run_orb(img, nfeatures):
    orb = cv2.ORB_create(nfeatures=nfeatures)
    pts = orb.detect(img)
    orb_kp = cv2.KeyPoint_convert(pts)
    orb_kp1 = []

    for i in orb_kp:
        orb_kp1.append((int(i[0]), int(i[1])))
    orb_kp1 = list(set(orb_kp1))

    return orb_kp1


def run_fast():
    f = 'apple_0'
    imgpath = 'test_images/dataset/{}.jpg'.format(f)
    img = cv2.imread(imgpath, 0)

    orb_kp = run_orb(img, 500)
    keypoints, _ = set_up_scales_multi(img, scale_factor=1.2, nlevels=8, nfeatures=500)
    print(keypoints)

    img2 = cv2.imread(imgpath, 0)
    row, col = img2.shape

    distance = get_minimum_distance(keypoints, orb_kp)
    draw_histogram(distance)

    keypoints_per_quadrant = [[] for _ in range(4)]

    # for k in keypoints:
    #     if k[0] < col // 2 and k[1] < row // 2:
    #         keypoints_per_quadrant[0].append((k[0], k[1]))
    #     elif k[0] > col // 2 and k[1] < row // 2:
    #         keypoints_per_quadrant[1].append((k[0] - col // 2, k[1]))
    #     elif k[0] < col // 2 and k[1] > row // 2:
    #         keypoints_per_quadrant[2].append((k[0], k[1] - row // 2))
    #     elif k[0] > col // 2 and k[1] > row // 2:
    #         keypoints_per_quadrant[3].append((k[0] - col // 2, k[1] - row // 2))
    #
    # for i in keypoints_per_quadrant:
    #     show_image('quadrant', mark_keypoints(i, img2))

    split = 4
    # quadrant_kp = []
    # for quadrant in range(4):
    #     print('quadrant is:', quadrant)
    #     if quadrant == 0:
    #         scaled_img = img2[:row // 2, :col // 2]
    #         x_shift = 0
    #         y_shift = 0
    #     elif quadrant == 1:
    #         scaled_img = img2[:row // 2, col // 2:]
    #         x_shift = col // 2
    #         y_shift = 0
    #     elif quadrant == 2:
    #         scaled_img = img2[row // 2:, : col // 2]
    #         x_shift = 0
    #         y_shift = row // 2
    #     elif quadrant == 3:
    #         scaled_img = img2[row // 2:, col // 2:]
    #         x_shift = col // 2
    #         y_shift = row // 2
    #     temp, _ = set_up_scales(scaled_img, scale_factor=1.2, nlevels=8, nfeatures=125)
    #
    #     for i in temp:
    #         quadrant_kp.append((i[0] + x_shift, i[1] + y_shift))
    #     # show_image('quadrant {}'.format(quadrant + 1), mark_keypoints(quadrant_kp, img2))
    #     print()
    #     print('length of keypoints by first method {}'.format(len(keypoints)))
    #     print('length of keypoints by second method {}'.format(len(quadrant_kp)))
    #     print(len(get_matched_point(quadrant_kp, keypoints)))
    #     get_average_distance(quadrant_kp, keypoints)
    #     distance = get_minimum_distance(quadrant_kp, keypoints)
    #     distance_count = draw_histogram(distance)
    #
    #     print()

    show_image(f, mark_keypoints(keypoints, img))
    show_image('orb', mark_keypoints(orb_kp, img2))
    print(get_matched_point(keypoints, orb_kp))
    get_average_distance(keypoints, orb_kp)
    kp_o1, des1 = get_brief_descriptors(img, keypoints)

    # intensity_centroid(img, kp, 31)
    return keypoints, img


def add_box_frames_to_db():
    for i in range(14):
        imgpath = 'box/frames/frame_{}.png'.format(i)
        img = cv2.imread(imgpath, 0)
        keypoints = set_up_scales(img, scale_factor=1.2, nlevels=8, nfeatures=500)
        # kp_o, des = get_brief_descriptors(img, keypoints)
        # add_to_database(imgpath, keypoints, des)


def add_to_database(imgpath, keypoints, descriptors):
    conn = MongoClient('localhost', 27017)
    db = conn["image_db"]
    collection = db['images']
    entry = {
        'imgpath': imgpath,
        'keypoints': keypoints,
        'descriptors': Binary(pickle.dumps(descriptors, protocol=2), subtype=128)
    }
    collection.insert_one(entry)


def get_brief_descriptors(img, coordinates_list):
    keypoints_objects = cv2.KeyPoint_convert(coordinates_list)
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    _, des = brief.compute(img, keypoints_objects)
    return keypoints_objects, des


def fast_multi_setup(img, quadrant, row, col, q1, _kp_list):
    # orb_kp = run_orb(img, 125)
    _kp, _ = fast_test(img, 250, threshold=20, non_max=1)
    harris_kp = harris_corner(img, _kp, 125, 0.04)

    for i in range(len(harris_kp)):
        x = harris_kp[i][0]
        y = harris_kp[i][1]
        if quadrant == 1:
            y = y + row // 2

        _kp_list.append((x, y))

    # for i in range(len(orb_kp)):
    #     x = orb_kp[i][0]
    #     y = orb_kp[i][1]
    #     if quadrant == 1:
    #         x = x + col // 2
    #     elif quadrant == 2:
    #         y = y + row // 2
    #     elif quadrant == 3:
    #         x = x + col // 2
    #         y = y + row // 2
    #     _orb_list.append((x, y))

    q1.put(_kp_list)
    # q2.put(_orb_list)


def main():
    run_fast()
    # ext_list = ['jpg', 'png', 'jfif']
    # count_list = []
    # path = 'test_images/dataset'
    # dir = os.listdir(path)
    # _keypoints_list = [[] for _x in range(len(dir))]
    # _orb_kp_list = [[] for _x in range(len(dir))]
    # _image_sizes = []
    # start = time.time()
    # img_num = 0
    # print(len(path))
    # print(len(dir))
    # for filename in os.listdir(path):
    #     f = os.path.join(path, filename)
    #
    #     if os.path.isfile(f):
    #         img = cv2.imread(f, 0)
    #         row, col = img.shape
    #         img_start = time.time()
    #         imgpath = f
    #         img = cv2.imread(imgpath, 0)
    #         # look at quadrant at a time
    #         _image_sizes.append(img.shape)
    #
    #         _kp_list = []
    #         # _orb_list = []
    #         row, col = img.shape
    #
    #         q1 = multiprocessing.Queue()
    #         # q2 = multiprocessing.Queue()
    #         quadrants = [img[:row // 2, :col],
    #                      img[row // 2:, :col]]
    #
    #         threads = []
    #         for i in range(2):
    #             threads.append(
    #                 multiprocessing.Process(target=fast_multi_setup,
    #                                         args=(quadrants[i], i, row, col, q1, _kp_list)))
    #
    #         for i in threads:
    #             i.start()
    #
    #         for i in threads:
    #             i.join()
    #
    #         # _keypoints_list[img_num] += _kp_list
    #         # _orb_kp_list[img_num] += _orb_list
    #
    #         for i in range(2):
    #             _keypoints_list[img_num] += q1.get()
    #             # _orb_kp_list[img_num] += q2.get()
    #
    #         # count = get_matched_point(_keypoints_list[img_num], _orb_kp_list[img_num])
    #         # count_list.append(count)
    #         img_num += 1
    #
    #
    # # for i in range(len(count_list)):
    # #     print('kp: {}\t\torb kp: {}\tmatch count: {} ({:.4f}%)\t {}'.format(len(_keypoints_list[i]), len(_orb_kp_list[i]),
    # #                                                                         count_list[i],
    # #                                                                         count_list[i] / len(_orb_kp_list[i]),
    # #                                                                         time.time() - img_start))
    #
    # print("Processing time:", time.time() - start)


#             for quadrant in range(0, 4):
#                 current_kp = []
#                 current_orb_kp = []
#                 scaled_img = img
#                 row, col = img.shape
#
#                 if quadrant == 0:
#                     scaled_img = scaled_img[:row // 2, :col // 2]
#                 elif quadrant == 1:
#                     scaled_img = scaled_img[:row // 2, col // 2:]
#                 elif quadrant == 2:
#                     scaled_img = scaled_img[row // 2:, :col // 2]
#                 elif quadrant == 3:
#                     scaled_img = scaled_img[row // 2:, col // 2:]
#
#                 orb_kp = run_orb(scaled_img, 125)
#                 orb_len = len(orb_kp)
#                 _kp, _ = fast_test(scaled_img, 250, threshold=20, non_max=1)
#                 # print("Processing time:", time.process_time() - start)
#                 # show_image('fast at: {}'.format(level), mark_keypoints(_kp, scaled_img))
#                 harris_kp = harris_corner(scaled_img, _kp, 125, 0.04)
#                 # for i in range(len(harris_kp)):
#                 #     x = int(harris_kp[i][0] * (scale_factor ** level))
#                 #     y = int(harris_kp[i][1] * (scale_factor ** level))
#                 #     _keypoints_list.append((x, y))
#                 #     current_kp.append((x, y))
#                 # show_image('Scale level {}'.format(level + 1), mark_keypoints(current_kp, img))
#                 # print('current # of kp: ', len(_keypoints_list))
#                 # if level % 2 == 0:
#
#                 for i in range(len(harris_kp)):
#                     x = harris_kp[i][0]
#                     y = harris_kp[i][1]
#                     if quadrant == 1:
#                         x = x + col // 2
#                     elif quadrant == 2:
#                         y = y + row // 2
#                     elif quadrant == 3:
#                         x = x + col // 2
#                         y = y + row // 2
#                     current_kp.append((x, y))
#
#                 for i in range(len(orb_kp)):
#                     x = orb_kp[i][0]
#                     y = orb_kp[i][1]
#                     if quadrant == 1:
#                         x = x + col // 2
#                     elif quadrant == 2:
#                         y = y + row // 2
#                     elif quadrant == 3:
#                         x = x + col // 2
#                         y = y + row // 2
#                     current_orb_kp.append((x, y))
#
#                 _keypoints_list[img_num] += current_kp
#                 _orb_kp_list[img_num] += current_orb_kp
#
#                 # print('quadrant: {}'.format(quadrant))
#                 # print('kp found: {}\torb kp found: {}'.format(len(_keypoints_list[img_num]),
#                 #                                               len(_orb_kp_list[img_num])))
#                 count = get_matched_point(_keypoints_list[img_num], _orb_kp_list[img_num])
#                 # print('% of orb kp found: {}'.format(count / len(_orb_kp_list[img_num])))
#                 # avg = get_average_distance(_keypoints_list[img_num], _orb_kp_list[img_num])
#
#                 # print('cumulative matches: ', len(matched_points))
#                 # show_image('_kp', mark_keypoints(_keypoints_list[img_num], img))
#                 # show_image('orb', mark_keypoints(_orb_kp_list[img_num], img))
#                 if quadrant == 3:
#                     count_list.append(count)
#
# for i in range(len(count_list)):
#     print('kp: {}\t\torb kp: {}\tmatch count: {} ({:.4f}%)'.format(len(_keypoints_list[i]), len(_orb_kp_list[i]),
#                                                                    count_list[i],
#                                                                    count_list[i] / len(_orb_kp_list[i])))
# print("Processing time:", time.time() - start)


if __name__ == '__main__':
    main()

'''
def main():
    imgpath = 'test_images/dog.jfif'
    img = cv2.imread(imgpath, 0)


    start = time.process_time()
    keypoints = fast_test(img)
    print("Processing time:", time.process_time() - start)

    print(len(keypoints) == len(set(keypoints)))
    show_image('fast', mark_keypoints(keypoints, imgpath))

    start_FAST12 = time.process_time()
    corners = fast.detect(img, 20, 0)
    print("FAST-9 Processing time:", time.process_time() - start_FAST12)

    show_image('original_fast', mark_keypoints(corners, imgpath))

    # nonmax_corners, _ = non_max(keypoints, img)
    # for i in range(len(nonmax_corners)):
    #     if list[i] > 0.01 * max(list):
    #         img3 = cv2.circle(img2, (keypoints[i][1], keypoints[i][0]), 2, (0, 0, 255), -1)
    #     else:
    #         img3 = cv2.circle(img2, (keypoints[i][1], keypoints[i][0]), 2, (0, 255, 0), -1)
    # show_image('image', img3)

    # harris_response(img,img3, keypoints, 0.04)

    # print(keypoints)
    # print(corners)

    count = 0
    matching_points = get_matched_point(corners, keypoints)

    unique_corners = [item for item in corners if item not in matching_points]
    unique_keypoints = [item for item in keypoints if item not in matching_points]
    print('Keypoints length:', len(keypoints))
    print("FAST-9's corners length:", len(corners))
    print('Unique_keypoints length:', len(unique_keypoints))
    print('Unique_corners length:', len(unique_corners))
    print('Matching count:', count)

    # file = open ('output.csv', 'w', newline = '')

    # with file:
    #     writer = csv.writer(file)
    #     for i in unique_keypoints:
    #         list = [i[0],i[1]]
    #         writer.writerow(list)
    #     for i in unique_corners:
    #         list = [i[0],i[1]]
    #         writer.writerow(list)
    # sum = 0
    # for i in keypoints:
    #     distance = []
    #     for j in corners:
    #         distance.append(euclidean_distance(i, j))
    #     sum += min(distance)
    # print('Average distance with matching points', (sum / len(keypoints)))
    #
    # sum = 0
    # if unique_keypoints:
    #     for i in unique_keypoints:
    #         distance = []
    #         for j in unique_corners:
    #             distance.append(euclidean_distance(i, j))
    #         sum += min(distance)
    #     print('Average distance without matching points', (sum / len(unique_keypoints)))

    orb = cv2.ORB_create(nfeatures=500)
    pts = orb.detect(img)
    orb_kp = cv2.KeyPoint_convert(pts)
    print(orb_kp)
    h_kp = harris_corner(img, keypoints, len(orb_kp), 0.04)

    show_image('harris', mark_keypoints(h_kp, imgpath))
    kp = []
    for i in range(len(orb_kp)):
        kp.append((int(orb_kp[i][0]), int(orb_kp[i][1])))
    with open('h_kp.csv', mode='w', newline='') as h_file:
        h_writer = csv.writer(h_file, delimiter=',')
        for i in h_kp:
            h_writer.writerow(i)
    with open('orb_kp.csv', mode='w', newline='') as orb_file:
        orb_writer = csv.writer(orb_file, delimiter=',')
        for i in kp:
            orb_writer.writerow(i)

    kp = list(set(kp))
    show_image('orb', mark_keypoints(kp, imgpath))
    print(len(kp))
    get_average_distance(h_kp, kp)
    get_matched_point(h_kp, kp)
    intensity_centroid(img, h_kp, 31)

'''
