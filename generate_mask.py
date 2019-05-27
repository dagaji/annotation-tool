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

def make_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--video_path', type=str, required=True)
	parser.add_argument('--labels', type=str, default="labels_mapping.txt")
	parser.add_argument('--fps', type=float, default=1.0)
	parser.add_argument('--camera', type=str, required=True)
	return parser

def check_point(point, sz):

	h, w = sz

	if point[0] >= w:
		x = w - 1
	elif point[0] < 0:
		x = 0
	else:
		x = point[0]


	if point[1] >= h:
		y = h - 1
	elif h < 0:
		y = 0
	else:
		y = point[1]

	return (x, y)


def preprocess_img(img):
	img = undistort(img)
	img = downscale(img, 1000, False, pad_img=False)
	return img


def json2mask(annotation_path, labels_mapping, sz=(2160, 4096)):

	mask = labels_mapping["background"] * np.ones(sz).astype(np.uint8)
	test_mask = labels_mapping["background"] * np.ones(sz).astype(np.uint8)

	mask = PIL.Image.fromarray(mask)
	test_mask = PIL.Image.fromarray(test_mask)

	draw = PIL.ImageDraw.Draw(mask)
	test_draw = PIL.ImageDraw.Draw(test_mask)

	with open(annotation_path, 'r') as f:
		json_data = json.load(f)

	for shape in json_data["shapes"]:

		npoints = len(shape["points"])
		xy = [check_point(tuple(point), sz) for point in shape["points"]]
		label_mapping = labels_mapping[shape["label"]]

		if npoints == 2:
			draw.line(xy=xy, fill=label_mapping, width=16)
			test_draw.line(xy=xy, fill=255, width=16)
		elif npoints > 2:
			draw.polygon(xy=xy, fill=label_mapping)
			test_draw.polygon(xy=xy, fill=label_mapping)

	for shape in json_data["shapes"]:

		npoints = len(shape["points"])
		xy = [check_point(tuple(point), sz) for point in shape["points"]]
		label_mapping = labels_mapping[shape["label"]]

		if npoints == 2:
			test_draw.line(xy=xy, fill=label_mapping, width=8)

	concat_mask = np.dstack((np.array(mask), np.array(test_mask)))

	return concat_mask

def get_section(section_name):

	section_start = section_name.split("-")[0]
	section_end = section_name.split("-")[1]
	section = dict()
	section['start'] = section_start
	section['end'] = section_end
	return section

def remap_mask(mask, labels_mapping):

	mask_vis = mask.copy()
	mask_vis[mask_vis == 255] = 0
	mask_vis[mask_vis == labels_mapping["background"]] = 0
	labels_in = np.sort(np.unique(mask_vis))
	num_classes = len(labels_mapping)
	for label_idx in range(1, num_classes):
		label_in = labels_in[label_idx]
		mask_vis[mask_vis == label_in] = label_idx

	return mask_vis

def generate_masks_section(vidcap, fps, section, masks_dir, vis_dir, img_dir, masks_test_dir, full_mask, parameters, labels_mapping):
	

	def save(frame, mask, masks_test, vis_img, time_string):

		cv2.imwrite(os.path.join(masks_dir, "{}.png".format(time_string)), mask)
		cv2.imwrite(os.path.join(masks_test_dir, "{}.png".format(time_string)), mask_test)
		cv2.imwrite(os.path.join(vis_dir, "{}.png".format(time_string)), vis_img)
		cv2.imwrite(os.path.join(img_dir, "{}.png".format(time_string)), frame)


	def process_frame(frame, coords1, coords2, sz2, warp=True, Minv=None):

		H, W = sz2
		h, w = coords1.shape
		mask = labels_mapping["background"] * np.ones((h, w, 2), dtype=np.uint8)
		mask[coords1] = full_mask[coords2]

		#pdb.set_trace()
		
		if warp:
			mask = cv2.warpPerspective(mask, Minv, (W, H), flags=cv2.INTER_NEAREST)

		#pdb.set_trace()
		mask, mask_test = np.dsplit(mask, 2)
		mask = np.squeeze(mask)
		mask_test = np.squeeze(mask_test)
		mask_vis = remap_mask(mask_test, labels_mapping)

		num_classes = len(labels_mapping)
		#pdb.set_trace()
		vis_img = vis.vis_seg(frame, mask_vis, vis.make_palette(num_classes))

		mask = utils.pad_img(mask, (544, 1024))
		mask_test = utils.pad_img(mask_test, (544, 1024))

		return mask, mask_test, vis_img


	start_time_msec = string2msec(section["start"])
	delay_msec = int(1000 * (1 / fps))
	Minv_list, coords1_list, coords2_list = parameters

	vidcap.set(cv2.CAP_PROP_POS_MSEC,(start_time_msec))
	success, frame_start = vidcap.read()
	if success:
		frame_start = preprocess_img(frame_start)
		sz2 = frame_start.shape[:2]
		mask, mask_test, vis_img = process_frame(frame_start, coords1_list[0], coords2_list[0], sz2, Minv=Minv_list[0])
		save(utils.pad_img(frame_start, (544, 1024)), mask, mask_test, vis_img, section["start"])

	current_time = start_time_msec + delay_msec

	for Minv, coords1, coords2 in zip(Minv_list[1:], coords1_list[1:], coords2_list[1:]):
		vidcap.set(cv2.CAP_PROP_POS_MSEC,(current_time))
		success, frame = vidcap.read()
		if success:
			frame = preprocess_img(frame)
			mask, mask_test, vis_img = process_frame(frame, coords1, coords2, sz2, Minv=Minv)
			save(utils.pad_img(frame, (544, 1024)), mask, mask_test, vis_img, msec2string(current_time))
		current_time += delay_msec



if __name__ == "__main__":

	args = make_parser().parse_args()

	utils.initialize(args.camera)

	with open(args.labels, 'r') as f:
			labels_mapping = dict()
			for line in f:
				line = line.replace(" ", "")
				label_name = line.split(':')[0]
				label_integer = int(line.split(':')[1])
				labels_mapping[label_name] = label_integer

	video_name = Path(args.video_path).parts[-1].split(".")[0]
	work_dir = os.path.join("data", video_name)

	parameters_dir = "{}/parameters/".format(work_dir)
	panos_dir = "{}/panos/".format(work_dir)
	masks_dir = "{}/masks/".format(work_dir)
	masks_test_dir = "{}/masks_test/".format(work_dir)
	vis_dir = "{}/vis/".format(work_dir)
	annotations_dir = "{}/annotations/".format(work_dir)
	imgs_dir = "{}/images/".format(work_dir)

	vidcap = cv2.VideoCapture(args.video_path)

	for glob in Path(annotations_dir).glob("*.json"):

		section_name = os.path.splitext(os.path.basename(glob.parts[-1]))[0]

		pano_name = "{}.png".format(section_name)
		pano = cv2.imread(os.path.join(panos_dir, pano_name))

		full_mask = json2mask(os.path.join(annotations_dir, glob.parts[-1]), labels_mapping, sz=pano.shape[:2])
		full_mask_vis = remap_mask(full_mask[...,0], labels_mapping)
		num_classes = len(labels_mapping)
		#pdb.set_trace()
		vis_pano = vis.vis_seg(pano, full_mask_vis, vis.make_palette(num_classes))
		# plt.imshow(vis_pano)
		# plt.show()
		#pdb.set_trace()
		parameters_name = "{}.pickle".format(section_name)
		with open(os.path.join(parameters_dir, parameters_name), 'rb') as handle:
			parameters = pickle.load(handle)

		section = get_section(section_name)
		generate_masks_section(vidcap, args.fps, section, masks_dir, vis_dir, imgs_dir, masks_test_dir, full_mask, parameters, labels_mapping)
