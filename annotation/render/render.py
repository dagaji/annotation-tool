import json
import pdb
import numpy as np
import PIL.Image
import PIL.ImageDraw
from .render_registry import register_render

@register_render('labelme')
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