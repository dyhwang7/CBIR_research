import cv2
from matplotlib import pyplot as plt
from pymongo import MongoClient
from fast_src import mark_keypoints, show_image
import fast_src
import pickle

def compare_to_database():
    conn = MongoClient('localhost', 27017)
    db = conn["image_db"]
    collection = db['images']

    for item in collection.find():
        if item['imgpath'] == 'box/frames/frame_0.png':
            query_imgpath = item['imgpath']
            query_img = cv2.imread(query_imgpath, 0)
            query_keypoints = cv2.KeyPoint_convert(item['keypoints'])
            query_descriptors = pickle.loads(item['descriptors'])

    for item in collection.find():
        imgpath = item['imgpath']
        keypoints = cv2.KeyPoint_convert(item['keypoints'])
        descriptors = pickle.loads(item['descriptors'])
        img = cv2.imread(imgpath, 0)
        draw_matches_bf(query_img, img, query_keypoints, keypoints, query_descriptors, descriptors)


def draw_matches_knn(img1, img2, kp1, kp2, des1, des2):
    index_params = dict(algorithm=6,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2
    search_params = dict()

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    try:
        matchesMask = [[0, 0] for i in range(len(matches))]

        #ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matchesMask[i] = [1, 0]

        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=0)

        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

        plt.imshow(img3), plt.show()
    except ValueError:
        print('no matches')

def draw_matches_bf(img1, img2, kp1, kp2, des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)
    plt.imshow(img3), plt.show()

# conn = MongoClient('localhost', 27017)
# db = conn["image_db"]
# collection = db['images']
#
# for item in collection.find():
#     if item['imgpath'] == 'box/frames/frame_0.png':
#         query_imgpath = item['imgpath']
#         query_img = cv2.imread(query_imgpath, 0)
#         query_keypoints = item['keypoints']
#         query_keypoints_o = cv2.KeyPoint_convert(query_keypoints)
#         query_descriptors = pickle.loads(item['descriptors'])
# show_image('query', mark_keypoints(query_keypoints, query_img))
# rotated_img = cv2.imread('box/frames/frame_0_r90.png', 0)
# cropped_img = cv2.imread('box/frames/frame_0_cropped.jpg', 0)

# orb = cv2.ORB_create(nfeatures=500)
# rotated_kp_o = orb.detect(rotated_img)
# cropped_kp_o = orb.detect(cropped_img)
#
# rotated_kp_o, rotated_des = orb.compute(rotated_img, rotated_kp_o)
# cropped_kp_o, cropped_des = orb.compute(cropped_img, cropped_kp_o)

# rotated_kp = fast_src.set_up_scales(rotated_img, 1.2, 8, 500)
# cropped_kp = fast_src.set_up_scales(cropped_img, 1.2, 8, 500)
# show_image('rotated', mark_keypoints(rotated_kp, rotated_img))
# show_image('cropped', mark_keypoints(cropped_kp, cropped_img))
#
# rotated_kp_o, rotated_des = fast_src.get_brief_descriptors(rotated_img, rotated_kp)
# cropped_kp_o, cropped_des = fast_src.get_brief_descriptors(cropped_img, cropped_kp)
#


compare_to_database()