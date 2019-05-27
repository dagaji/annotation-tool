import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import argparse
import time
import pdb
import vis
import os.path
from downscale import _downscale as downscale
import operator
import sys; this = sys.modules[__name__]

this.camera_name = None
this._dist = None
this._mtx = None

def initialize(camera_name):

    if (this.camera_name is None):
        standard_method_dir = os.path.join('datos_calib_v2', camera_name, "standard")
        if not os.path.exists(standard_method_dir):
            msg = "Camera {0} is not defined"
            raise RuntimeError(msg.format(this.camera_name))
    else:
        msg = "Camera name is already initialized to {0}."
        raise RuntimeError(msg.format(this.camera_name))

    this.camera_name = camera_name; print this.camera_name
    this._dist = np.load(os.path.join(standard_method_dir,'dist.npy'))
    this._mtx = np.load(os.path.join(standard_method_dir,'mtx.npy'))



def undistort(img, return_FOV=False):

    h, w = img.shape[:2]

    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(this._mtx, this._dist,(w,h),0,(w,h))
    dst = cv2.undistort(img, this._mtx, this._dist, None, newcameramtx)

    if return_FOV:

        FOV_X=2*np.arctan2(w, 2*newcameramtx[0,0]) * 180.0/np.pi
        FOV_Y=2*np.arctan2(h, 2*newcameramtx[1,1]) * 180.0/np.pi

        return dst, (FOV_X, FOV_Y)

    else:
        return dst


def homography(image_a, image_b, draw_matches=True):

    image_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    image_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()

    kp_a, des_a = sift.detectAndCompute(image_a, None)
    kp_b, des_b = sift.detectAndCompute(image_b, None)

    # print "des_a: {}".format(des_a.shape)
    # print "des_b: {}".format(des_b.shape)

    # Brute force matching
    #pdb.set_trace()
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_a, trainDescriptors=des_b, k=2)

    # print "matches: {}".format(len(matches))
    # print

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

    #print len(src_pts)
    #pdb.set_trace()
    if len(src_pts) > 4:
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5)
    else:
        M = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    return M



def warp_image(image, homography, alpha_channel=True):

    if alpha_channel:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    h, w = image.shape[:2]

    # Find min and max x, y of new image
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

    if alpha_channel:
        warped_bgr = cv2.warpPerspective(src=image[...,:-1], M=homography, dsize=size, flags=cv2.INTER_LINEAR)
        warped_alpha = cv2.warpPerspective(src=image[...,-1], M=homography, dsize=size, flags=cv2.INTER_NEAREST)
        warped = np.dstack((warped_bgr, warped_alpha))
    else:
        warped = cv2.warpPerspective(src=image, M=homography, dsize=size, flags=cv2.INTER_LINEAR)

    shift = (int(xmin), int(ymin))

    return warped, shift


def string2msec(time_string):
    time_min = int(time_string.split(':')[0])
    time_sec = int(time_string.split(':')[1])
    time_sec += time_min * 60
    time_msec = 1000 * time_sec
    return time_msec



def msec2string(time_msec):
    time_sec = time_msec / 1000
    time_min = time_sec / 60
    time_string = "{}:{:02d}".format(time_min, time_sec - time_min * 60)
    return time_string



def get_offset(shift):
    offset_x = max(0, shift[0])
    offset_y = max(0, shift[1])
    return (offset_x, offset_y)



def paste(new_img, img, offset):
    h, w, z = img.shape
    offset_x, offset_y = offset
    new_img[offset_y:offset_y + h, offset_x:offset_x + w] = img
    return new_img



def merge_images(image1, image2, shift, blend=True, alpha=0.5):

    h1, w1, z1 = image1.shape
    h2, w2, z2 = image2.shape

    offset1 = get_offset((-shift[0], -shift[1]))
    offset2 = get_offset(shift)
    
    nw, nh = map(max, map(operator.add, offset1, (w1, h1)), map(operator.add, offset2, (w2, h2)))

    new_image = np.zeros((nh, nw, 3))
    new_image = paste(new_image, image1, offset1)
    new_image = paste(new_image, image2, offset2)

    if blend:

        new_image_aux = np.zeros((nh, nw, 3))
        new_image_aux = paste(new_image_aux, image2, offset2)
        new_image_aux = paste(new_image_aux, image1, offset1)

        new_image *= alpha
        new_image += (1 - alpha) * new_image_aux
        new_image = np.uint8(new_image)

    return new_image, offset1, offset2


def get_padding(sz, _sz):
    
    pad_amount = _sz - sz 
    
    if pad_amount % 2:
        
        padding = (pad_amount / 2 , pad_amount - pad_amount / 2)
    else:
        padding = (pad_amount / 2, pad_amount / 2)
        
    return padding


def pad_img(img, sz, pad_value=0):

    if img.shape == 2:
        img = np.dstack((img,img,img))

    H, W = sz
    height, width = img.shape[:2]
    x_pad = get_padding(width, W)
    y_pad = get_padding(height, H)
    img_padded = cv2.copyMakeBorder(img, y_pad[0], y_pad[1], x_pad[0], x_pad[1], cv2.BORDER_CONSTANT, value=[pad_value, pad_value, pad_value])
    return img_padded
