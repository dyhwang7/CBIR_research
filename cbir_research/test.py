import cv2
import math
import fast.fast9 as fast


def show_image(win, img):
    cv2.imshow('{}'.format(win), img)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('s'):
        cv2.imwrite('{}.jpg'.format(win), img)
        cv2.destroyAllWindows()


def intensity_centroid(img):
    mu = cv2.moments(img)
    print(mu)
    # for i in keypoints:
    #     kp.append((i[0], i[1]))
    # mu = [None] * len(kp)
    # for i in range(len(kp)):
    #     mu[i] = cv2.moments(kp[i])
    # print(mu)
    # mc = [None] * len(kp)
    # for i in range(len(kp)):
    #     mc[i] = math.atan2(mu[i]['m01'] / (mu[i]['m00'] + 1e-5), mu[i]['m10'] / (mu[i]['m00'] + 1e-5))
    # return mc


img = cv2.imread('test_images/dog.jfif', 0)

# rows, cols = img.shape
# x = 274
# y = 182
# h = 7
# s = h // 2
# img[y,x] = 0
# y_min = y - s
# y_max = y + s + 1
# x_min = x - s
# x_max = x + s + 1
# kp_x = 3
# kp_y = 3
# if x - s < 0:
#     x_min = 0
#     kp_x -= abs(x-s)
# if x + s + 1 > cols:
#     x_max = cols -1
#
# if y - s < 0:
#     y_min = 0
#     kp_y -= abs(y-s)
# if y + s + 1 > rows:
#     y_max = cols -1
#
# print(y_min, y_max, x_min, x_max)
# patch = img[y_min: y_max, x_min: x_max]
# print(patch)
# print(patch[kp_y, kp_x])
# img2 = img[0:150,0:150]
# show_image('img', img2)
# intensity_centroid(img2)
#
# scale_factor = 1.2
# nlevels = 8
# nfeatures = 500
# factor = 1.0/scale_factor
# ndesired_features_per_scale = nfeatures * (1-factor)/(1-factor**nlevels)
#
# sum_features = 0
# nfeatures_per_level = []
# for level in range(nlevels - 1):
#     print(level)
#     nfeatures_per_level.append(round(ndesired_features_per_scale))
#     print(round(ndesired_features_per_scale))
#     print()
#     sum_features += nfeatures_per_level[level]
#     ndesired_features_per_scale *= factor
#     print(ndesired_features_per_scale)
# nfeatures_per_level.append(max(nfeatures - sum_features, 0))

corners = fast.detect(img, 20, 0)