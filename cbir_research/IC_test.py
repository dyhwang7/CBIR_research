import cv2
import math
import fast.fast9 as fast
import numpy as np
from scipy import signal as sig
from scipy import ndimage as ndi
import time
import itertools
import math
import csv


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


'''
fast detection algorithm:
iterate through each pixel in the image and find pixels that contain x number of consecutive surrounding pixels
with itensities either above or below the threshold. 
'''


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


def gradient_x(img):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    return sig.convolve2d(img, kernel_x, mode='same')


def gradient_y(img):
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
    return sig.convolve2d(img, kernel_y, mode='same')


def harris_corner(img, keypoints, n, k, window_size=3, ):
    img2 = cv2.imread('test_images/frame_0.png')
    harris_list = []
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
        Sxx = np.sum(Ixx[y - offset: y + offset + 1, x - offset: x + offset + 1])
        Syy = np.sum(Iyy[y - offset: y + offset + 1, x - offset: x + offset + 1])
        Sxy = np.sum(Ixy[y - offset: y + offset + 1, x - offset: x + offset + 1])

        det = (Sxx * Syy) - (Sxy ** 2)
        trace = Sxx + Syy
        r = det - k * (trace ** 2)
        # print(r)
        harris_list.append({'keypoint': item, 'harris_response': r})
    sorted_list = sorted(harris_list, key=lambda i: i['harris_response'], reverse=True)
    sorted_list = sorted_list[:n]
    j = 0
    h = []
    for i in sorted_list:
        j += 1
        img2 = cv2.circle(img2, (i['keypoint'][0], i['keypoint'][1]), 2, (0, 0, 255), -1)
        # else:
        #     # img2 = cv2.circle(img2, (keypoints[i][0], keypoints[i][1]), 2, (0, 255, 0), -1)

    show_image('harris', img2)
    # print('j:', j)
    # print('harris length', len(h))
    return sorted_list


# used to check the euclidean distance between points
def euclidean_distance(p, q):
    return math.sqrt((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2)


def main():
    img = cv2.imread('test_images/frame_0.png', 0)
    img2 = cv2.imread('test_images/frame_0.png')
    img3 = cv2.imread('test_images/frame_0.png')
    img4 = cv2.imread('test_images/frame_0.png')
    img5 = cv2.imread('test_images/frame_0.png')
    threshold = 20

    start = time.process_time()
    keypoints = fast_test(img, threshold)
    print("Processing time:", time.process_time() - start)

    print(len(keypoints) == len(set(keypoints)))
    for i in range(len(keypoints)):
        img2 = cv2.circle(img2, (keypoints[i][0], keypoints[i][1]), 2, (0, 0, 255), -1)
    show_image('img2', img2)

    start_FAST12 = time.process_time()
    corners = fast.detect(img, 20, 0)
    print("FAST-9 Processing time:", time.process_time() - start_FAST12)

    # for i in range(len(corners)):
    #     img3 = cv2.circle(img3, (corners[i][0], corners[i][1]), 2, (0, 0, 255), -1)
    # show_image('img3', img3)

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
    matching_points = []

    for corner in corners:
        if corner in keypoints:
            count += 1
            matching_points.append(corner)
    #
    # unique_corners = [item for item in corners if item not in matching_points]
    # unique_keypoints = [item for item in keypoints if item not in matching_points]
    # print('Keypoints length:', len(keypoints))
    # print("FAST-9's corners length:", len(corners))
    # print('Unique_keypoints length:', len(unique_keypoints))
    # print('Unique_corners length:', len(unique_corners))
    # print('Matching count:', count)

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

    orb = cv2.ORB_create()
    pts = orb.detect(img)
    orb_kp = cv2.KeyPoint_convert(pts)
    h_kp = harris_corner(img, keypoints, len(orb_kp), 0.04)
    kp = []
    for i in range(len(orb_kp)):
        kp.append((round(orb_kp[i][0]), round(orb_kp[i][1])))
    with open('h_kp.csv', mode='w', newline='') as h_file:
        h_writer = csv.writer(h_file, delimiter=',')
        for i in h_kp:
            coord = (i['keypoint'][0], i['keypoint'][1], i['harris_response'])
            h_writer.writerow(coord)
    with open('orb_kp.csv', mode='w', newline='') as orb_file:
        orb_writer = csv.writer(orb_file, delimiter=',')
        for i in kp:
            orb_writer.writerow(i)
    print(h_kp)
    print(kp)
    kp = list(set(kp))

    # for i in range (10):
    #     print('harris:', h_kp[i])
    #     print('orb', kp[i])
    print(len(kp))
    for i in range(len(kp)):
        img4 = cv2.circle(img4, (kp[i][0], kp[i][1]), 2, (0, 0, 255), -1)
    show_image('img4', img4)
    count = 0
    matching_points = []

    for h in h_kp:
        if ((h['keypoint'][0], h['keypoint'][1]) in kp):
            count += 1
            matching_points.append(h)
    print('count:', count)
    sum = 0
    for i in h_kp:
        distance = []
        for j in kp:
            distance.append(euclidean_distance((i['keypoint'][0], i['keypoint'][1]), j))
        sum += min(distance)

    print('Average distance with matching points', (sum / len(h_kp)))

    mu = [None] * len(keypoints)
    for i in range(len(keypoints)):
        mu[i] = cv2.moments(keypoints[i])
    print(mu)
if __name__ == "__main__":
    main()


