import cv2
from pathlib import Path
from argparse import ArgumentParser
import os.path
import pdb
from math import ceil
import shutil

def make_parser():
    p = ArgumentParser()
    p.add_argument('--path', type=str, required=True)
    p.add_argument('--save_path', type=str, required=True)
    p.add_argument('--max_dim', type=int, required=True)
    p.add_argument('--output_format', type=str, required=True)
    p.add_argument('-m', dest='is_mask', action='store_true')
    p.set_defaults(is_mask=False)
    return p

def get_padding(sz):
    
    pad_amount = int(ceil(float(sz) / 32) * 32 - sz)
    
    if pad_amount % 2:
        
        padding = (pad_amount / 2 , pad_amount - pad_amount / 2)
    else:
        padding = (pad_amount / 2, pad_amount / 2)
        
    return padding

def _downscale(img, max_dim, is_mask, pad_img=True):

    down_img = None
    height, width = img.shape[:2]
    # only shrink if img is bigger than required
    if max_dim < height or max_dim < width:
        # get scaling factor
        scaling_factor = max_dim / float(height)
        if max_dim/float(width) < scaling_factor:
            scaling_factor = max_dim / float(width)
        # resize image
        if not is_mask:
            down_img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
            pad_value = 0
        else:
            down_img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_NEAREST)
            pad_value = 255
        # pad image so that its dimension are multiple of 32
        if pad_img:
            new_height, new_width = down_img.shape[:2]
            x_pad = get_padding(new_width)
            y_pad = get_padding(new_height)
            down_img_padded = cv2.copyMakeBorder(down_img, y_pad[0], y_pad[1], x_pad[0], x_pad[1], cv2.BORDER_CONSTANT, value=[pad_value, pad_value, pad_value])
            return down_img_padded

    return down_img


if __name__ == "__main__":
    
    parser = make_parser()
    args = parser.parse_args()

    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path, ignore_errors=True)
    os.makedirs(args.save_path)

    path = Path(args.path)

    for glob in path.glob('*'):
        img = cv2.imread(os.path.join(args.path, glob.parts[-1]))
        down_img_padded = _downscale(img, args.max_dim, args.is_mask)
        img_name = os.path.splitext(glob.parts[-1])[0]
        cv2.imwrite(os.path.join(args.save_path, '{}.{}'.format(img_name, args.output_format)), down_img_padded)
    


