import utils
import os.path
import os
import cv2
import argparse
import numpy as np
import pdb
import pickle
from distutils.dir_util import copy_tree
from pathlib import Path
import homography as homo
import matplotlib.pyplot as plt


def output_limits(frames_list, M_list):
    
    offset_list = []
    size_list = []
    for i in range(len(frames_list)):
        size, offset = homo.warp_image(frames_list[i], M_list[i], return_warped=False)
        offset_list.append(offset)
        size_list.append(size)
        
    offset_x_list = [offset_x for offset_x, _ in offset_list]
    xmax_list = []
    for offset_x, size in zip(offset_x_list, size_list):
        xmax_list.append(offset_x + size[0])
        
    offset_y_list = [offset_y for _, offset_y in offset_list]
    ymax_list = []
    for offset_y, size in zip(offset_y_list, size_list):
        ymax_list.append(offset_y + size[1])
        
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

def paste_imgs(warped_list, offset_list, sz):

	H, W = sz
	alpha_channels_list = []
	for warped, offset in zip(warped_list, offset_list):
		alpha_channel = np.zeros((H, W), dtype=bool)
		y, x = np.where(warped[...,-1] == 255)
		coords = np.array(zip(x,y))
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

def create_pano(frames_list):

	M_list = [np.eye(3)]
	for i in range(1, len(frames_list)):
		M = homo.homography(frames_list[i-1], frames_list[i], draw_matches=False)
		M_list.append(np.dot(M_list[-1],M))

	center_image_idx = len(frames_list) / 2 + len(frames_list) % 2 - 1

	Minv = np.linalg.inv(M_list[center_image_idx])
	_M_list = [np.dot(Minv, M) for M in M_list]

	xlim, ylim = output_limits(frames_list, _M_list)

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
	    warped, offset = homo.warp_image(frames_list[i], _M_list[i])
	    Minv = compute_Minv(warped, frames_list[0].shape[:2])
	    Minv_list.append(Minv)
	    coords1_list.append(warped[...,-1] == 255)
	    _offset_list.append(offset)
	    _warped_list.append(warped)

	_offset_list = [(offset_x - xmin, offset_y - ymin) for offset_x, offset_y in _offset_list]

	pano, coords2_list = paste_imgs(_warped_list, _offset_list, (H,W))

	return pano, (Minv_list, coords1_list, coords2_list)


if __name__ == "__main__":

	config_path = utils.get_config_path()
	video_config, labels_mapping = utils.load_config_info(config_path)
	work_dir = video_config['work_dir']

	if not os.path.exists(work_dir):
		copy_tree("plantilla", work_dir)

	start_time_msec = utils.string2msec(video_config['start_time'])
	end_time_msec = utils.string2msec(video_config['end_time'])
	duration_msec = utils.string2msec(video_config['duration'])

	if duration_msec < (end_time_msec - start_time_msec):
		num_itervals = (end_time_msec - start_time_msec) / duration_msec
	else:
		num_itervals = 1
		duration_msec = end_time_msec - start_time_msec

	end_time_msec = start_time_msec + duration_msec
	video_loader = utils.VideoLoader(video_config['video_path'], video_config['camera'], start_time_msec, end_time_msec)

	for _ in range(num_itervals):

		frames_list = []
		for frame in iter(video_loader):
			frames_list.append(frame)

		if len(frames_list) > 0:

			pano_name = "{}-{}".format(utils.msec2string(start_time_msec), utils.msec2string(end_time_msec))
			
			try:

				pano, parameters = create_pano(frames_list)

				print "\n>> Saving parameters of pano {}".format(pano_name)

				cv2.imwrite('{}/panos/{}.png'.format(work_dir, pano_name), pano)

				with open('{}/parameters/{}.pickle'.format(work_dir, pano_name), 'wb') as handle:
					pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

			except Exception as e:
				print str(e)
				print ">> Could not create pano {}".format(pano_name)
			
		start_time_msec = end_time_msec
		end_time_msec = start_time_msec + duration_msec
		video_loader.reset(start_time_msec, end_time_msec)