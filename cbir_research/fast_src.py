import cv2
import math
import fast.fast9 as fast
import numpy as np
import time
import itertools
import csv
import imutils


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


def set_up_scales(img):
    scale_factor = 1.2
    nlevels = 8
    nfeatures = 500
    factor = 1.0 / scale_factor
    ndesired_features_per_scale = nfeatures * (1 - factor) / (1 - factor ** nlevels)

    sum_features = 0
    nfeatures_per_level = []
    for level in range(nlevels - 1):
        nfeatures_per_level.append(round(ndesired_features_per_scale))
        sum_features += nfeatures_per_level[level]
        ndesired_features_per_scale *= factor
    nfeatures_per_level.append(max(nfeatures - sum_features, 0))

    _keypoints_list = []
    rows, cols = img.shape
    for level in range(nlevels):
        scaled_img = img
        w = int(cols / scale_factor ** level)
        scaled_img = imutils.resize(scaled_img, width=w)
        start = time.process_time()
        _kp, _ = fast_test(scaled_img, nfeatures_per_level[level] * 2, threshold=20, non_max=1, )
        print("Processing time:", time.process_time() - start)
        show_image('fast at: {}'.format(level), mark_keypoints(_kp, scaled_img))
        harris_kp = harris_corner(scaled_img, _kp, nfeatures_per_level[level], 0.04)
        print(len(harris_kp))
        _keypoints_list.append(harris_kp)
        # some conversion to OPENCV's keypoint object
        # pass them into brief
        show_image('harris at: {}'.format(level), mark_keypoints(harris_kp, scaled_img))

    all_scale_keypoints = []
    for level in range(nlevels):
        for i in range(len(_keypoints_list[level])):
            x = int(_keypoints_list[level][i][0] * (scale_factor ** level))
            y = int(_keypoints_list[level][i][1] * (scale_factor ** level))
            all_scale_keypoints.append((x, y))

    print('length', len(all_scale_keypoints))
    show_image('ours', mark_keypoints(all_scale_keypoints, img))
    return all_scale_keypoints


def fast_test(img, n, threshold, non_max):
    fast_n = 9
    keypoints = []
    scores = []
    rows, cols = img.shape

    for y in range(4, rows - 4):
        for x in range(4, cols - 4):
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
        nonmax_corners = nonmax_corners [0: n]
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
    kp = []
    half_patchsize = patchsize // 2
    for i in keypoints:
        x = i[0]
        y = i[1]
        kp.append((x, y))
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

        # print('cX, cY in original image: {}'.format((cX, cY)))
        orientation = math.atan2((cY - y) * -1, cX - x)
        cv2.circle(color_patch, (16, 16), 1, (0, 0, 255), -1)
        # print('keypoint in original image: {}'.format((x, y)))
        # print('in radians: ', orientation)
        # print('in degrees: ', math.degrees(orientation))
        # show_image('patch', color_patch)
        kp.append((x, y, orientation))
    return kp
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
            distance.append(euclidean_distance((i[0], i[1]), j))
        sum += min(distance)

    print('Average distance with matching points', (sum / len(kp1)))


def get_matched_point(kp1, kp2):
    count = 0
    matching_points = []
    for i in kp1:
        if (i[0], i[1]) in kp2:
            count += 1
            matching_points.append(i)
    print('count:', count)
    return matching_points


def mark_keypoints(kp, img):
    color_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(len(kp)):
        color_image = cv2.circle(color_image, (kp[i][0], kp[i][1]), 2, (0, 0, 255), -1)
    return color_image


# used to check the euclidean distance between points
def euclidean_distance(p, q):
    return math.sqrt((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2)


def run_fast():
    imgpath = 'test_images/cathedral_700.jpg'
    img = cv2.imread(imgpath, 0)

    # corners, scores = fast_test(img, 1)
    # corners2, _ = fast_test(img, 0)
    # print(len(corners))
    # print(len(corners2))
    # show_image('fast', mark_keypoints(corners, img))
    # show_image('fast2', mark_keypoints(corners2, img))
    orb = cv2.ORB_create(nfeatures=500)
    pts = orb.detect(img)
    orb_kp = cv2.KeyPoint_convert(pts)

    orb_kp1 = []
    for i in orb_kp:
        orb_kp1.append((int(i[0]), int(i[1])))
    orb_kp1 = list(set(orb_kp1))
    print(len(orb_kp))
    print(len(orb_kp1))
    kp = set_up_scales(img)
    # show_image('orb', mark_keypoints(orb_kp, img))

    show_image('orb', mark_keypoints(orb_kp1, img))
    matching_keypoints = get_matched_point(orb_kp1, kp)
    get_average_distance(kp, orb_kp1)
    # intensity_centroid(img, kp, 31)
    return kp, img


def main():
    run_fast()


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

