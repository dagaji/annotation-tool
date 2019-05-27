import utils
from utils import homography, warp_image, string2msec, msec2string, merge_images, get_offset, undistort
import utils
import json
import cv2
import os.path
import vis
import argparse
from downscale import _downscale as downscale
import pdb
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import PIL.Image
import PIL.ImageDraw
from math import ceil
from matplotlib.patches import Circle
import pickle

def preprocess_frame(img):
	img = undistort(img)
	img = downscale(img, 1000, False, pad_img=False)
	return img

def make_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--work_dir', type=str, required=True)
	parser.add_argument('--video_path', type=str, required=True)
	parser.add_argument('--start_time', type=str, required=True)
	parser.add_argument('--end_time', type=str, required=True)
	parser.add_argument('--labels', type=str, default="labels_mapping.txt")
	parser.add_argument('--fps', type=float, default=1.0)
	parser.add_argument('--label_name', type=str, default="road")
	parser.add_argument('--nframes', type=int, default=1)
	return parser

if __name__ == "__main__":
    
    args = make_parser().parse_args()
    
    with open(args.labels, 'r') as f:
			labels_mapping = dict()
			for line in f:
				line = line.replace(" ", "")
				label_name = line.split(':')[0]
				label_integer = int(line.split(':')[1])
				labels_mapping[label_name] = label_integer
    
    parameters_dir = "{}/parameters/".format(args.work_dir)
    panos_dir = "{}/panos/".format(args.work_dir)
    masks_dir = "{}/masks/".format(args.work_dir)
    masks_test_dir = "{}/masks_test/".format(args.work_dir)
    vis_dir = "{}/vis/".format(args.work_dir)
    annotations_dir = "{}/annotations/".format(args.work_dir)
    imgs_dir = "{}/images/".format(args.work_dir)
    
    start_time_msec = string2msec(args.start_time)
    end_time_msec = string2msec(args.end_time)
    delay_msec = int(1000 * (args.nframes / args.fps))
    
    vidcap = cv2.VideoCapture(args.video_path)
    vidcap.set(cv2.CAP_PROP_POS_MSEC, start_time_msec)
    success, start_frame = vidcap.read()
    current_time = start_time_msec
    
    while current_time <= end_time_msec:
        current_time_string = msec2string(current_time) + ".png"
        vidcap.set(cv2.CAP_PROP_POS_MSEC, current_time)
        success, current_frame = vidcap.read()
        if success:
            current_frame= preprocess_frame(current_frame)
            H, W = current_frame.shape[:2]
            mask = labels_mapping[args.label_name] * np.ones((H, W), dtype=np.uint8)
            vis_img = vis.vis_seg(current_frame, np.ones((H, W), dtype=np.uint8), vis.make_palette(2))
            cv2.imwrite(os.path.join(masks_dir, current_time_string), utils.pad_img(mask, (544, 1024)))
            cv2.imwrite(os.path.join(masks_test_dir, current_time_string), utils.pad_img(mask, (544, 1024)))
            cv2.imwrite(os.path.join(imgs_dir, current_time_string), utils.pad_img(current_frame, (544, 1024)))
            cv2.imwrite(os.path.join(vis_dir, current_time_string), vis_img)
            
        print msec2string(current_time)
        current_time += delay_msec
        
    
