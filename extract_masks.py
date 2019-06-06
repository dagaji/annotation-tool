import json
import cv2
import os.path
import vis
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
import utils

def json2mask(annotation_path, labels_mapping, sz):

	def check_point(point):

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
		xy = [check_point(tuple(point)) for point in shape["points"]]
		label_mapping = labels_mapping[shape["label"]]

		if npoints == 2:
			draw.line(xy=xy, fill=label_mapping, width=16)
			test_draw.line(xy=xy, fill=255, width=16)
		elif npoints > 2:
			draw.polygon(xy=xy, fill=label_mapping)
			test_draw.polygon(xy=xy, fill=label_mapping)

	for shape in json_data["shapes"]:

		npoints = len(shape["points"])
		xy = [check_point(tuple(point)) for point in shape["points"]]
		label_mapping = labels_mapping[shape["label"]]

		if npoints == 2:
			test_draw.line(xy=xy, fill=label_mapping, width=8)

	concat_mask = np.dstack((np.array(mask), np.array(test_mask)))

	return concat_mask

def get_section(section_name, step_msec):
	section_start = section_name.split("-")[0]
	section_end = section_name.split("-")[1]
	return range(utils.string2msec(section_start), utils.string2msec(section_end), step_msec)


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

		assert len(video_loader) == len(Minv_list) == len(coords1_list) == len(coords2_list), "Could not extract masks"

		for idx, time_msec in enumerate(section):

			frame = self.video_loader.frame_at(time_msec)

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

			vis_img = vis.vis_seg(frame, mask, vis.make_palette(len(self.labels_mapping)))

			self.save(frame, mask, mask_test, vis_img, utils.msec2string(time_msec))


	def save(self, frame, mask, mask_test, vis_img, time_string):

		img_name = utils.msec2string(time_msec) + '.png'
		cv2.imwrite(os.path.join(self.masks_dir, img_name), mask)
		cv2.imwrite(os.path.join(self.masks_test_dir, img_name), mask_test)
		cv2.imwrite(os.path.join(self.vis_dir, img_name), vis_img)
		cv2.imwrite(os.path.join(self.imgs_dir, img_name), frame)


if __name__ == "__main__":

	config_path = utils.get_config_path()
	video_config, labels_mapping = utils.load_config_info(config_path)
	work_dir = video_config['work_dir']
	step_msec = int(1000 / video_config['fps'])

	annotations_dir = os.path.join(work_dir, 'annotations')
	parameters_dir = os.path.join(work_dir, 'parameters')
	panos_dir = os.path.join(work_dir, 'panos')

	assert os.path.exists(work_dir), "This video has not been labelled"

	video_loader = utils.VideoLoader(video_config['video_path'], video_config['camera'])
	mask_extractor = MaskExtractor(work_dir, video_loader, labels_mapping)

	for glob in Path(annotations_dir).glob("*.json"):

		section_name = os.path.splitext(os.path.basename(glob.parts[-1]))[0]

		pano = cv2.imread(os.path.join(panos_dir, section_name + ".png"))
		full_mask = json2mask(os.path.join(annotations_dir, glob.parts[-1]), labels_mapping, pano.shape[:2])

		with open(os.path.join(parameters_dir, section_name + ".pickle"), 'rb') as handle:
			parameters = pickle.load(handle)

		section = get_section(section_name, step_msec)

		try:
			mask_extractor.extract(full_mask, parameters, section)
		except Exception as e:
			print str(e)

