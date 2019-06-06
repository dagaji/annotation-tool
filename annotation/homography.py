import cv2
import numpy as np
from matplotlib import pyplot as plt
import pdb


def homography(image_a, image_b, draw_matches=True):

    image_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    image_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()

    kp_a, des_a = sift.detectAndCompute(image_a, None)
    kp_b, des_b = sift.detectAndCompute(image_b, None)

    # Brute force matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_a, trainDescriptors=des_b, k=2)

    # Lowes Ratio
    good_matches = []
    for m, n in matches:
        if m.distance < .75 * n.distance:
            good_matches.append(m)

    if draw_matches:
        aux_img = cv2.drawMatches(image_a, kp_a, image_b, kp_b, sorted(good_matches, key = lambda x:x.distance)[:10], None, flags=2)
        plt.figure()
        plt.imshow(aux_img),plt.show()

    src_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches])\
        .reshape(-1, 1, 2)
    dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches])\
        .reshape(-1, 1, 2)

    if len(src_pts) > 4:
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5)
    else:
        M = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    return M



def warp_image(image, homography, alpha_channel=True, return_warped=True):

    if alpha_channel:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    h, w = image.shape[:2]

    p = np.array([[0, w, w, 0], [0, 0, h, h], [1, 1, 1, 1]])
    p_prime = np.dot(homography, p)

    yrow = p_prime[1] / p_prime[2]
    xrow = p_prime[0] / p_prime[2]
    xmin = min(xrow)
    ymax = max(yrow)
    ymin = min(yrow)
    xmax = max(xrow)

    new_mat = np.array([[1, 0, -1 * xmin], [0, 1, -1 * ymin], [0, 0, 1]])
    homography = np.dot(new_mat, homography)

    width = int(round(xmax - xmin))
    heigth = int(round(ymax - ymin))

    size = (width, heigth)
    shift = (int(xmin), int(ymin))

    if not return_warped:
    	return size, shift

    if alpha_channel:
        warped_bgr = cv2.warpPerspective(src=image[...,:-1], M=homography, dsize=size, flags=cv2.INTER_LINEAR)
        warped_alpha = cv2.warpPerspective(src=image[...,-1], M=homography, dsize=size, flags=cv2.INTER_NEAREST)
        warped = np.dstack((warped_bgr, warped_alpha))
    else:
        warped = cv2.warpPerspective(src=image, M=homography, dsize=size, flags=cv2.INTER_LINEAR)

    return warped, shift