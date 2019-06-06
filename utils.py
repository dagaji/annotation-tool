import cv2
import numpy as np
import math
import argparse
import pdb
import os.path
import yaml
from pathlib import Path

ANNOTATION_TOOL_PATH = os.path.dirname(os.path.realpath(__file__))
CALIB_DATA_PATH = os.path.join(ANNOTATION_TOOL_PATH, 'calib_data')

class VideoLoader:

    def __init__(self, video_path, camera_dir, max_dim=1000):

        self.vidcap = cv2.VideoCapture(video_path)
        self.dist = np.load(os.path.join(camera_dir, 'dist.npy'))
        self.mtx = np.load(os.path.join(camera_dir, 'mtx.npy'))
        self.max_dim = max_dim

    def frame_at(self, time_msec):

        self.vidcap.set(cv2.CAP_PROP_POS_MSEC, time_msec)
        success, frame = self.vidcap.read()
        if success:
            print ">> Loading frame at {}".format(time_msec)
            frame = self.process_frame(frame)
            return frame
        else:
            return None

    def process_frame(self, frame):
        frame = undistort(frame, self.dist, self.mtx)
        frame = downscale(frame, self.max_dim)
        return frame


def downscale(img, max_dim):

    height, width = img.shape[:2]

    if max_dim < height or max_dim < width:
        scaling_factor = min(max_dim / float(width), max_dim / float(height))
        img_down = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        return img_down
    else:
        return None

def undistort(img, dist, mtx):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist,(w,h),0,(w,h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    return dst


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


def get_config_path():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    return args.config


def load_config_info(config_path):


    with open(config_path, 'r') as stream:
        video_config = yaml.safe_load(stream)

    camera_dir = os.path.join(CALIB_DATA_PATH, video_config.pop("camera"))

    assert os.path.exists(video_config['video_path']), "Video not found."
    assert os.path.exists(camera_dir), "Camera data not found."

    video_config['camera_dir'] = camera_dir

    video_config.setdefault('fps', 1.0)
    video_config.setdefault('duration', "0:15")

    dataset_config_dir = os.path.dirname(config_path)
    dataset_config_path = os.path.join(dataset_config_dir, "dataset.yml")

    with open(dataset_config_path, 'r') as stream:
        dataset_config = yaml.safe_load(stream)

    video_name = Path(video_config['video_path']).parts[-1].split(".")[0]
    work_dir = os.path.join(dataset_config['dataset_dir'], video_name)
    video_config['work_dir'] = work_dir


    dataset_config['LABELS']['background'] = dataset_config['LABELS'].pop(video_config['background'])
    del video_config['background']

    return video_config, dataset_config['LABELS']