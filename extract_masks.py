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
import config as cfg
import utils_v2

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
	return utils_v2.string2msec(section_start), utils_v2.string2msec(section_end)


class MaskExtractor:

	def __init__(self, work_dir, video_loader, labels_mapping):

		self.masks_dir = os.path.join(work_dir, 'masks')
		self.masks_test_dir = os.path.join(work_dir, 'masks_test')
		self.vis_dir = os.path.join(work_dir, 'vis')
		self.imgs_dir = os.path.join(work_dir, 'images')

		self.video_loader = video_loader

		self.labels_mapping = labels_mapping

	def extract(self, full_mask, parameters, section):

		Minv_list, coords1_list, coords2_list = parameters

		self.video_loader.reset(*section)

		assert len(video_loader) == len(Minv_list) == len(coords1_list) == len(coords2_list), "Could not extract masks"

		for idx, frame in enumerate(iter(self.video_loader)):

			Minv = Minv_list[idx]
			coords1 = coords1_list[idx]
			coords2 = coords2_list[idx]

			H, W = frame.shape[:2]
			h, w = coords1.shape

			mask = self.labels_mapping["background"] * np.ones((h, w, 2), dtype=np.uint8)
			mask[coords1] = full_mask[coords2]

			mask = cv2.warpPerspective(mask, Minv, (W, H), flags=cv2.INTER_NEAREST)

			mask, mask_test = np.dsplit(mask, 2)
			mask = np.squeeze(mask)
			mask_test = np.squeeze(mask_test)

			vis_img = vis.vis_seg(frame, mask, vis.make_palette(cfg.NUM_CLASSES))

			self.save(frame, mask, mask_test, vis_img)


	def save(self, frame, mask, mask_test, vis_img):

		time_msec = self.video_loader.get_last_frame_time()
		img_name = utils_v2.msec2string(time_msec) + '.png'

		cv2.imwrite(os.path.join(self.masks_dir, img_name), mask)
		cv2.imwrite(os.path.join(self.masks_test_dir, img_name), mask_test)
		cv2.imwrite(os.path.join(self.vis_dir, img_name), vis_img)
		cv2.imwrite(os.path.join(self.imgs_dir, img_name), frame)


if __name__ == "__main__":

	args = make_parser().parse_args()

	with open(args.labels, 'r') as f:
			labels_mapping = dict()
			for line in f:
				line = line.replace(" ", "")
				label_name = line.split(':')[0]
				label_integer = int(line.split(':')[1])
				labels_mapping[label_name] = label_integer

	video_name = Path(args.video_path).parts[-1].split(".")[0]
	work_dir = os.path.join(cfg.DATA_PATH, video_name)

	annotations_dir = os.path.join(work_dir, 'annotations')
	parameters_dir = os.path.join(work_dir, 'parameters')
	panos_dir = os.path.join(work_dir, 'panos')

	assert os.path.join(work_dir), "This video has not been labelled"

	video_loader = utils_v2.VideoLoader(args.video_path, args.camera)
	mask_extractor = MaskExtractor(work_dir, video_loader, labels_mapping)

	for glob in Path(annotations_dir).glob("*.json"):

		section_name = os.path.splitext(os.path.basename(glob.parts[-1]))[0]

		pano = cv2.imread(os.path.join(panos_dir, section_name + ".png"))
		full_mask = json2mask(os.path.join(annotations_dir, glob.parts[-1]), labels_mapping, sz=pano.shape[:2])

		with open(os.path.join(parameters_dir, section_name + ".pickle"), 'rb') as handle:
			parameters = pickle.load(handle)

		section = get_section(section_name)

		try:
			mask_extractor.extract(full_mask, parameters, section)
		except Exception as e:
			print str(e)

