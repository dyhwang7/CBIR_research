import cv2
import math
import fast.fast9 as fast

import time
import itertools


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


# used to check the euclidean distance between points
def euclidean_distance(p, q):
    return math.sqrt((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2)


def main():
    img = cv2.imread('test_images/frame_0.png', 0)
    img2 = cv2.imread('test_images/frame_0.png')
    img3 = cv2.imread('test_images/frame_0.png')
    threshold = 20

    start = time.process_time()
    keypoints = fast_test(img, threshold)
    print("Processing time:", time.process_time() - start)

    print(len(keypoints) == len(set(keypoints)))
    for i in range(len(keypoints)):
        img2 = cv2.circle(img3, (keypoints[i][0], keypoints[i][1]), 2, (0, 0, 255), -1)
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


if __name__ == "__main__":
    main()
