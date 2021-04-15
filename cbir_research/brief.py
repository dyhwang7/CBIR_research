import math
import cv2
import time
import itertools
import fast.fast9 as fast
import numpy as np


# Start: From previous main...
def get_pixels(img, y, x, threshold):
    px = img[y, x]
    pixel_circle = [int(img[y, x + 3]) - px, int(img[y + 1, x + 3]) - px, int(img[y + 2, x + 2]) - px,
                    int(img[y + 3, x + 1]) - px,
                    int(img[y + 3, x]) - px, int(img[y + 3, x - 1]) - px, int(img[y + 2, x - 2]) - px,
                    int(img[y + 1, x - 3]) - px,
                    int(img[y, x - 3]) - px, int(img[y - 1, x - 3]) - px, int(img[y - 2, x - 2]) - px,
                    int(img[y - 3, x - 1]) - px,
                    int(img[y - 3, x]) - px, int(img[y - 3, x + 1]) - px, int(img[y - 2, x + 2]) - px,
                    int(img[y - 1, x + 3]) - px]
    compared_list = []

    for item in pixel_circle:
        if item > threshold:
            compared_list.append(1)
        elif item < - threshold:
            compared_list.append(-1)
        else:
            compared_list.append(0)
    return compared_list


def fast_test(img, threshold):
    fast_n = 9
    keypoints = []
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
    return keypoints


def draw_keypoints(keypoint, img, win):
    for i in range(len(keypoint)):
        img = cv2.circle(img, (keypoint[i][0], keypoint[i][1]), 2, (0, 0, 255), -1)
    show_image(win, img)


def match_keypoints(keypoints, corners):
    count = 0
    matching_points = []

    for corner in corners:
        if corner in keypoints:
            count += 1
            matching_points.append(corner)

    unique_corners = [item for item in corners if item not in matching_points]
    unique_keypoints = [item for item in keypoints if item not in matching_points]
    print('Keypoints length:', len(keypoints))
    print("FAST-9's corners length:", len(corners))
    print('Unique_keypoints length:', len(unique_keypoints))
    print('Unique_corners length:', len(unique_corners))
    print('Matching count:', count)
    return unique_keypoints, unique_corners


def euclidean_distance(p, q):
    return math.sqrt((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2)


def check_distance(keypoints, corners, unique_keypoints, unique_corners):
    sum = 0
    for i in keypoints:
        distance = []
        for j in corners:
            distance.append(euclidean_distance(i, j))
        sum += min(distance)
    print('Average distance with matching points', (sum / len(keypoints)))

    sum = 0
    if unique_keypoints:
        for i in unique_keypoints:
            distance = []
            for j in unique_corners:
                distance.append(euclidean_distance(i, j))
            sum += min(distance)
        print('Average distance without matching points', (sum / len(unique_keypoints)))


def full_keypoint_test(img_dir, threshold, win):
    img = cv2.imread(img_dir, 0)
    img2 = cv2.imread(img_dir)

    start = time.process_time()
    keypoints = fast_test(img, threshold)
    print("Processing time:", time.process_time() - start)

    print(len(keypoints) == len(set(keypoints)))
    draw_keypoints(keypoints, img2, win)
    return img2, keypoints


def show_image(win, img):
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow('{}'.format(win), img)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('s'):
        cv2.imwrite('{}.jpg'.format(win), img)
        cv2.destroyAllWindows()


# End: From previous main
gaussian = [8, -3, 9, 5,
            4, 2, 7, -12,
            -11, 9, -8, 2,
            7, -12, 12, -13,
            2, -13, 2, 12,
            1, -7, 1, 6,
            -2, -10, -2, -4,
            -13, -13, -11, -8,
            -13, -3, -12, -9,
            10, 4, 11, 9,
            -13, -8, -8, -9,
            -11, 7, -9, 12,
            7, 7, 12, 6,
            -4, -5, -3, 0,
            -13, 2, -12, -3,
            -9, 0, -7, 5,
            12, -6, 12, -1,
            -3, 6, -2, 12,
            -6, -13, -4, -8,
            11, -13, 12, -8,
            4, 7, 5, 1,
            5, -3, 10, -3,
            3, -7, 6, 12,
            -8, -7, -6, -2,
            -2, 11, -1, -10,
            -13, 12, -8, 10,
            -7, 3, -5, -3,
            -4, 2, -3, 7,
            -10, -12, -6, 11,
            5, -12, 6, -7,
            5, -6, 7, -1,
            1, 0, 4, -5,
            9, 11, 11, -13,
            4, 7, 4, 12,
            2, -1, 4, 4,
            -4, -12, -2, 7,
            -8, -5, -7, -10,
            4, 11, 9, 12,
            0, -8, 1, -13,
            -13, -2, -8, 2,
            -3, -2, -2, 3,
            -6, 9, -4, -9,
            8, 12, 10, 7,
            0, 9, 1, 3,
            7, -5, 11, -10,
            -13, -6, -11, 0,
            10, 7, 12, 1,
            -6, -3, -6, 12,
            10, -9, 12, -4,
            -13, 8, -8, -12,
            -13, 0, -8, -4,
            3, 3, 7, 8,
            5, 7, 10, -7,
            -1, 7, 1, -12,
            3, -10, 5, 6,
            2, -4, 3, -10,
            -13, 0, -13, 5,
            -13, -7, -12, 12,
            -13, 3, -11, 8,
            -7, 12, -4, 7,
            6, -10, 12, 8,
            -9, -1, -7, -6,
            -2, -5, 0, 12,
            -12, 5, -7, 5,
            3, -10, 8, -13,
            -7, -7, -4, 5,
            -3, -2, -1, -7,
            2, 9, 5, -11,
            -11, -13, -5, -13,
            -1, 6, 0, -1,
            5, -3, 5, 2,
            -4, -13, -4, 12,
            -9, -6, -9, 6,
            -12, -10, -8, -4,
            10, 2, 12, -3,
            7, 12, 12, 12,
            -7, -13, -6, 5,
            -4, 9, -3, 4,
            7, -1, 12, 2,
            -7, 6, -5, 1,
            -13, 11, -12, 5,
            -3, 7, -2, -6,
            7, -8, 12, -7,
            -13, -7, -11, -12,
            1, -3, 12, 12,
            2, -6, 3, 0,
            -4, 3, -2, -13,
            -1, -13, 1, 9,
            7, 1, 8, -6,
            1, -1, 3, 12,
            9, 1, 12, 6,
            -1, -9, -1, 3,
            -13, -13, -10, 5,
            7, 7, 10, 12,
            12, -5, 12, 9,
            6, 3, 7, 11,
            5, -13, 6, 10,
            2, -12, 2, 3,
            3, 8, 4, -6,
            2, 6, 12, -13,
            9, -12, 10, 3,
            -8, 4, -7, 9,
            -11, 12, -4, -6,
            1, 12, 2, -8,
            6, -9, 7, -4,
            2, 3, 3, -2,
            6, 3, 11, 0,
            3, -3, 8, -8,
            7, 8, 9, 3,
            -11, -5, -6, -4,
            -10, 11, -5, 10,
            -5, -8, -3, 12,
            -10, 5, -9, 0,
            8, -1, 12, -6,
            4, -6, 6, -11,
            -10, 12, -8, 7,
            4, -2, 6, 7,
            -2, 0, -2, 12,
            -5, -8, -5, 2,
            7, -6, 10, 12,
            -9, -13, -8, -8,
            -5, -13, -5, -2,
            8, -8, 9, -13,
            -9, -11, -9, 0,
            1, -8, 1, -2,
            7, -4, 9, 1,
            -2, 1, -1, -4,
            11, -6, 12, -11,
            -12, -9, -6, 4,
            3, 7, 7, 12,
            5, 5, 10, 8,
            0, -4, 2, 8,
            -9, 12, -5, -13,
            0, 7, 2, 12,
            -1, 2, 1, 7,
            5, 11, 7, -9,
            3, 5, 6, -8,
            -13, -4, -8, 9,
            -5, 9, -3, -3,
            -4, -7, -3, -12,
            6, 5, 8, 0,
            -7, 6, -6, 12,
            -13, 6, -5, -2,
            1, -10, 3, 10,
            4, 1, 8, -4,
            -2, -2, 2, -13,
            2, -12, 12, 12,
            -2, -13, 0, -6,
            4, 1, 9, 3,
            -6, -10, -3, -5,
            -3, -13, -1, 1,
            7, 5, 12, -11,
            4, -2, 5, -7,
            -13, 9, -9, -5,
            7, 1, 8, 6,
            7, -8, 7, 6,
            -7, -4, -7, 1,
            -8, 11, -7, -8,
            -13, 6, -12, -8,
            2, 4, 3, 9,
            10, -5, 12, 3,
            -6, -5, -6, 7,
            8, -3, 9, -8,
            2, -12, 2, 8,
            -11, -2, -10, 3,
            -12, -13, -7, -9,
            -11, 0, -10, -5,
            5, -3, 11, 8,
            -2, -13, -1, 12,
            -1, -8, 0, 9,
            -13, -11, -12, -5,
            -10, -2, -10, 11,
            -3, 9, -2, -13,
            2, -3, 3, 2,
            -9, -13, -4, 0,
            -4, 6, -3, -10,
            -4, 12, -2, -7,
            -6, -11, -4, 9,
            6, -3, 6, 11,
            -13, 11, -5, 5,
            11, 11, 12, 6,
            7, -5, 12, -2,
            -1, 12, 0, 7,
            -4, -8, -3, -2,
            -7, 1, -6, 7,
            -13, -12, -8, -13,
            -7, -2, -6, -8,
            -8, 5, -6, -9,
            -5, -1, -4, 5,
            -13, 7, -8, 10,
            1, 5, 5, -13,
            1, 0, 10, -13,
            9, 12, 10, -1,
            5, -8, 10, -9,
            -1, 11, 1, -13,
            -9, -3, -6, 2,
            -1, -10, 1, 12,
            -13, 1, -8, -10,
            8, -11, 10, -6,
            2, -13, 3, -6,
            7, -13, 12, -9,
            -10, -10, -5, -7,
            -10, -8, -8, -13,
            4, -6, 8, 5,
            3, 12, 8, -13,
            -4, 2, -3, -3,
            5, -13, 10, -12,
            4, -13, 5, -1,
            -9, 9, -4, 3,
            0, 3, 3, -9,
            -12, 1, -6, 1,
            3, 2, 4, -8,
            -10, -10, -10, 9,
            8, -13, 12, 12,
            -8, -12, -6, -5,
            2, 2, 3, 7,
            10, 6, 11, -8,
            6, 8, 8, -12,
            -7, 10, -6, 5,
            -3, -9, -3, 9,
            -1, -13, -1, 5,
            -3, -7, -3, 4,
            -8, -2, -8, 3,
            4, 2, 12, 12,
            2, -5, 3, 11,
            6, -9, 11, -13,
            3, -1, 7, 12,
            11, -1, 12, 4,
            -3, 0, -3, 6,
            4, -11, 4, 12,
            2, -4, 2, 1,
            -10, -6, -8, 1,
            -13, 7, -11, 1,
            -13, 12, -11, -13,
            6, 0, 11, -13,
            0, -1, 1, 4,
            -13, 3, -9, -2,
            -9, 8, -6, -3,
            -13, -6, -8, -2,
            5, -9, 8, 10,
            2, 7, 3, -9,
            -1, -6, -1, -1,
            9, 5, 11, -2,
            11, -3, 12, -8,
            3, 0, 3, 5,
            -1, 4, 0, 10,
            3, -6, 4, 5,
            -13, 0, -10, 5,
            5, 8, 12, 11,
            8, 9, 9, -6,
            7, -4, 8, -12,
            -10, 4, -10, 9,
            7, 3, 12, 4,
            9, -7, 10, -2,
            7, 0, 12, -2,
            -1, -6, 0, -11]
gaussian_X = []
gaussian_Y = []
for point in range(0, len(gaussian), 4):
    gaussian_X.append((gaussian[point], gaussian[point + 1]))
    gaussian_Y.append((gaussian[point + 2], gaussian[point + 3]))
S = [[gaussian_X], [gaussian_Y]]


def get_r_theta(theta):
    R = [[math.cos(theta), math.sin(theta)],
         [-math.sin(theta), math.cos(theta)]]
    return R


def get_crop_patch(keypoint, img):
    x = keypoint[0] - 15
    w = keypoint[0] + 16
    y = keypoint[1] - 15
    h = keypoint[1] + 16
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img_gray.shape
    if x < 0:
        x = 0
    elif x > width:
        x = width
    if w > width:
        w = width
    elif w < 0:
        w = 0
    if y < 0:
        y = 0
    elif y > height:
        y = height
    if h > height:
        h = height
    elif h < 0:
        h = 0
    return img_gray[y:h, x:w]


def get_subwindow_avg(px, img):
    height, width = img.shape
    x_ = int(px[1]) - 2
    y_ = int(px[0]) - 2
    if (y_ < 0):
        y_ = y_ * -1
    if (x_ < 0):
        x_ = x_ * -1
    px_patch = []
    for i in range(x_, x_ + 5):
        for j in range(y_, y_ + 5):
            y = j
            x = i
            if i < 0:
                x = 0
            elif i >= height:
                x = height - 1
            if j < 0:
                y = 0
            elif j >= width:
                y = width - 1
            px_patch.append((y, x))

    total = 0
    for i in range(0, 25):
        x = px_patch[i][0]
        y = px_patch[i][1]
        total = total + img[y, x]

    return total / len(px_patch)


def run_brief(img, keypoints, keypoints_with_orientation):
    descriptors = []

    for i in range(len(keypoints)):
        img_patch = get_crop_patch(keypoints[i], img)
        # show_image('Cropped Patch', img_patch)
        img_patch_blur = cv2.integral(img_patch)
        orientation = keypoints_with_orientation[i]
        orientation = orientation[2]
        R = get_r_theta(orientation)
        descriptor = ""
        for j in range(0, len(gaussian_X)):
            _x = (gaussian_X[j][0] + 15, gaussian_X[j][1] + 15)
            _y = (gaussian_Y[j][0] + 15, gaussian_Y[j][1] + 15)
            R = np.array(R)
            S_x = np.array(_x)
            S_theta_x = np.matmul(R, S_x)
            S_y = np.array(_y)
            S_theta_y = np.matmul(R, S_y)
            X_subwindow_avg = get_subwindow_avg(S_theta_x, img_patch_blur)
            Y_subwindow_avg = get_subwindow_avg(S_theta_y, img_patch_blur)
            if X_subwindow_avg < Y_subwindow_avg:
                descriptor = descriptor + "1"
            else:
                descriptor = descriptor + "0"

        descriptors.append(descriptor)

    return descriptors


def run_brief_test(keypoints, keypoint_with_orientation, img):
    star = cv2.xfeatures2d.StarDetector_create()
    kp = star.detect(img, None)
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    kp, des = brief.compute(img, kp)

    kp_pixels = cv2.KeyPoint_convert(kp)

    start = time.process_time()
    descriptors = run_brief(img, keypoints, keypoint_with_orientation)
    print("Processing time:", time.process_time() - start)
    count = 0
    for i in range(len(kp_pixels)):
        for j in range(len(keypoints)):
            if kp_pixels[i][0] == keypoints[j][0] and kp_pixels[i][1] == keypoints[j][1]:
                print('\nkp_pixels: ({}, {})'.format(kp_pixels[i][0], kp_pixels[i][1]))
                print('keypoints: ({}, {})'.format(keypoints[j][0], keypoints[j][1]))
                print("brief's descriptor ", des[i])
                new_des = ''
                for k in range(len(des[i])):
                    new_des += '{0:08b}'.format(des[i][k])
                print("brief's binary descriptor: ", new_des)
                print("our descriptor:            ", descriptors[j])
                print('our des length:          ', len(descriptors[j]))
                XOR = int(new_des, 2) ^ int(descriptors[j], 2)
                XOR = bin(XOR)[2:].zfill(len(new_des))
                hamming_distance = [ones for ones in XOR[2:] if ones == '1']
                print('XOR: ', XOR)
                print('hamming_distance: ', len(hamming_distance))
                count += 1
    print(count)

    duplicates = len(descriptors) - len(set(descriptors))
    if duplicates == 0:
        print("No duplicate descriptors")
    else:
        print("Total duplicate descriptors: ", duplicates)
