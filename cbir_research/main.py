import fast_src
import brief
import cv2

# fast_src.run_fast()
imgpath = 'test_images/cathedral_700.jpg'
img = cv2.imread(imgpath, 0)
keypoints, scores = fast_src.fast_test(img, 1000, threshold=20, non_max=1)
kp_orientation = fast_src.intensity_centroid(img, keypoints, 31)
kp_w_orientation = []
for i in range(1, len(kp_orientation), 2):
    kp_w_orientation.append(kp_orientation[i])

img = cv2.imread(imgpath)
# .show_image('Cathedral', img)
brief.run_brief_test(keypoints, kp_w_orientation, img)
