import utils
from utils import homography, warp_image, string2msec, msec2string, merge_images, undistort
import os.path
import os
import json
import cv2
from Tkinter import * 
from matplotlib import pyplot as plt
import argparse
import numpy as np
from downscale import _downscale as downscale
import pdb
import pickle
from distutils.dir_util import copy_tree
from pathlib import Path


def make_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--video_path', type=str, required=True)
	parser.add_argument('--start_time', type=str, required=True)
	parser.add_argument('--end_time', type=str, required=True)
	parser.add_argument('--fps', type=float, default=1.0)
	parser.add_argument('--camera', type=str, required=True)
	parser.add_argument('--duration', type=str)
	return parser

def preprocess_frame(frame):
	frame = undistort(frame)
	frame = downscale(frame, 1000, False, pad_img=False)
	return frame

def output_limits(frames_list, M_list):
    
    offset_list = []
    warped_list = []
    for i in range(len(frames_list)):
        warped, offset = warp_image(frames_list[i], M_list[i])
        offset_list.append(offset)
        warped_list.append(warped)
        
    offset_x_list = [offset_x for offset_x, _ in offset_list]
    xmax_list = []
    for offset_x, img in zip(offset_x_list, warped_list):
        xmax_list.append(offset_x + img.shape[1])
        
    offset_y_list = [offset_y for _, offset_y in offset_list]
    ymax_list = []
    for offset_y, img in zip(offset_y_list, warped_list):
        ymax_list.append(offset_y + img.shape[0])
        
    xlim = np.array([[xlim1, xlim2] for xlim1, xlim2 in zip(offset_x_list, xmax_list)])
    ylim = np.array([[ylim1, ylim2] for ylim1, ylim2 in zip(offset_y_list, ymax_list)])
    
    return xlim, ylim

def compute_Minv(warped, sz):

	y, x = np.where(warped[...,-1])
	pts = np.array(zip(x,y))

	rect = np.zeros((4, 2), dtype = np.float32)

	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	H, W = sz
	pts_dst = np.float32([[0,0],[W,0],[W,H],[0,H]])

	M = cv2.getPerspectiveTransform(rect, pts_dst)

	return M

def generate_pano(warped_list, offset_list, sz):

	H, W = sz
	alpha_channels_list = []
	for warped, offset in zip(warped_list, offset_list):
		alpha_channel = np.zeros((H, W), dtype=bool)
		y, x = np.where(warped[...,-1] == 255)
		coords = np.array(zip(x,y))
		#pdb.set_trace()
		coords += np.array(offset)
		r = np.squeeze(coords[:,1])
		c = np.squeeze(coords[:,0])
		alpha_channel[r,c] = True
		alpha_channels_list.append(alpha_channel)

	pano = np.zeros((H, W, 3), dtype=np.uint8)
	for i in range(len(alpha_channels_list)):
		warped_ = warped_list[i][...,:-1]
		alpha_channel = warped_list[i][...,-1] == 255
		pano[alpha_channels_list[i]] = warped_[alpha_channel]
    
	return pano, alpha_channels_list

def process_interval(vidcap, start_time_msec, end_time_msec, work_dir, delay_msec=1000):

	vidcap.set(cv2.CAP_PROP_POS_MSEC, start_time_msec)
	success, start_frame = vidcap.read()
	if success:
		start_frame = preprocess_frame(start_frame)

	# plt.imshow(start_frame)
	# plt.show()

	M_list = [np.eye(3)]
	frames_list = [start_frame]
	current_time = start_time_msec + delay_msec

	while current_time <= end_time_msec:
		vidcap.set(cv2.CAP_PROP_POS_MSEC, current_time)
		success, current_frame = vidcap.read()
		if success:
			current_frame = preprocess_frame(current_frame)
			frames_list.append(current_frame)
			# plt.figure()
			# plt.imshow(frames_list[-2])
			# plt.figure()
			# plt.imshow(frames_list[-1])
			# plt.show()
			M = homography(frames_list[-2], frames_list[-1], draw_matches=False)
			# print M
			# pdb.set_trace()
			M_list.append(np.dot(M_list[-1], M))
		print msec2string(current_time)
		current_time += delay_msec


	center_image_idx = len(frames_list) / 2 + len(frames_list) % 2 - 1

	Minv = np.linalg.inv(M_list[center_image_idx])
	_M_list = [np.dot(Minv, M) for M in M_list]

	xlim, ylim = output_limits(frames_list, _M_list)

	# print xlim
	# pdb.set_trace()

	xmin = np.min(xlim[:,0])
	xmax = np.max(xlim[:,1])
	W = int(np.round(xmax - xmin))

	ymin = np.min(ylim[:,0])
	ymax = np.max(ylim[:,1])
	H = int(np.round(ymax - ymin))

	_warped_list = []
	_offset_list = []
	Minv_list = []
	coords1_list = []
	for i in range(len(frames_list)):
	    warped, offset = warp_image(frames_list[i], _M_list[i])
	    Minv = compute_Minv(warped, frames_list[0].shape[:2])
	    Minv_list.append(Minv)
	    coords1_list.append(warped[...,-1] == 255)
	    _offset_list.append(offset)
	    _warped_list.append(warped)

	_offset_list = [(offset_x - xmin, offset_y - ymin) for offset_x, offset_y in _offset_list]

	pano, coords2_list = generate_pano(_warped_list, _offset_list, (H,W))

	start_time = msec2string(start_time_msec)
	end_time = msec2string(end_time_msec)

	cv2.imwrite('{}/panos/{}-{}.png'.format(work_dir, start_time, end_time), pano)

	with open('{}/parameters/{}-{}.pickle'.format(work_dir, start_time, end_time), 'wb') as handle:
	    pickle.dump((Minv_list, coords1_list, coords2_list), handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":

	args = make_parser().parse_args()

	utils.initialize(args.camera)

	video_name = Path(args.video_path).parts[-1].split(".")[0]
	work_dir = os.path.join("data", video_name)
	if not os.path.exists(work_dir):
		copy_tree("plantilla", work_dir)

	start_time_msec = string2msec(args.start_time)
	end_time_msec = string2msec(args.end_time)
	delay_msec = int(1000 / args.fps)

	vidcap = cv2.VideoCapture(args.video_path)

	if args.duration is not None:
		duration_msec = string2msec(args.duration)
		num_itervals = (end_time_msec - start_time_msec) / duration_msec
	else:
		num_itervals = 1
		duration_msec = end_time_msec - start_time_msec

	_start_time_msec = start_time_msec
	for _ in range(num_itervals):
		_end_time_msec = _start_time_msec + duration_msec
		process_interval(vidcap, _start_time_msec, _end_time_msec, work_dir, delay_msec=delay_msec)
		_start_time_msec = _end_time_msec + delay_msec




	#######################################################################################

	# vidcap.set(cv2.CAP_PROP_POS_MSEC, start_time_msec)
	# success, start_frame = vidcap.read()
	# if success:
	# 	start_frame = preprocess_frame(start_frame)


	# M_list = [np.eye(3)]
	# frames_list = [start_frame]
	# current_time = start_time_msec + delay_msec

	# while current_time <= end_time_msec:
	# 	vidcap.set(cv2.CAP_PROP_POS_MSEC, current_time)
	# 	success, current_frame = vidcap.read()
	# 	if success:
	# 		current_frame = preprocess_frame(current_frame)
	# 		frames_list.append(current_frame)
	# 		M = homography(frames_list[-2], frames_list[-1], draw_matches=False)
	# 		#M_list.append(np.dot(M, M_list[-1]))
	# 		M_list.append(np.dot(M_list[-1], M))
	# 	print msec2string(current_time)
	# 	current_time += delay_msec


	# #xlim, ylim = output_limits(frames_list, M_list)

	# #avg_x = np.mean(xlim, axis=1)
	# #avg_y = np.mean(ylim, axis=1)

	# #idx = np.argsort(avg_y)
	# #center_idx = (len(M_list) / 2)
	# #center_image_idx = idx[center_idx]

	# center_image_idx = len(frames_list) / 2 + len(frames_list) % 2 - 1

	# Minv = np.linalg.inv(M_list[center_image_idx])
	# _M_list = [np.dot(Minv, M) for M in M_list]
	# #_M_list = M_list

	# xlim, ylim = output_limits(frames_list, _M_list)

	# xmin = np.min(xlim[:,0])
	# xmax = np.max(xlim[:,1])
	# W = int(np.round(xmax - xmin))

	# ymin = np.min(ylim[:,0])
	# ymax = np.max(ylim[:,1])
	# H = int(np.round(ymax - ymin))

	# _warped_list = []
	# _offset_list = []
	# Minv_list = []
	# coords1_list = []
	# for i in range(len(frames_list)):
	#     warped, offset = warp_image(frames_list[i], _M_list[i])
	#     Minv = compute_Minv(warped, frames_list[0].shape[:2])
	#     Minv_list.append(Minv)
	#     coords1_list.append(warped[...,-1] == 255)
	#     _offset_list.append(offset)
	#     _warped_list.append(warped)

	# _offset_list = [(offset_x - xmin, offset_y - ymin) for offset_x, offset_y in _offset_list]

	# #pdb.set_trace()
	    
	# pano, coords2_list = generate_pano(_warped_list, _offset_list, (H,W))

	# #plt.imshow(pano[...,::-1])
	# #plt.show()

	# cv2.imwrite('{}/panos/{}-{}.png'.format(work_dir, args.start_time, args.end_time), pano)

	# with open('{}/parameters/{}-{}.pickle'.format(work_dir, args.start_time, args.end_time), 'wb') as handle:
	#     pickle.dump((Minv_list, coords1_list, coords2_list), handle, protocol=pickle.HIGHEST_PROTOCOL)



