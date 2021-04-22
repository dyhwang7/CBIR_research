import fast_src
import cv2


imgpath = 'test_images/cathedral_700.jpg'
img = cv2.imread(imgpath, 0)


# _kp, _ = fast_src.fast_test(img, 500, threshold=20, non_max=1)
row, col = img.shape
half_row = row // 2
half_col = col // 2


kp = [[half_col, i] for i in range (row)]
color_img = fast_src.mark_keypoints(kp, img)

print('row, col', row, col)
print('')
q1 = color_img[:half_row, :half_col]
q2 = color_img[:half_row, half_col:]
q3 = color_img[half_row:, :half_col]
q4 = color_img[half_row:, half_col:]
fast_src.show_image('img', color_img)


for i in [q1, q2, q3, q4]:
    fast_src.show_image('quarter', i)