import cv2
import numpy as np
import math
import argparse
import pdb
import os.path
import config as cfg

class VideoLoader:

    def __init__(self, video_path, camera_name, start_time_msec=0, end_time_msec=0, max_dim=1000, fps=1.0):

        assert os.path.exists(video_path), "Video not found."

        camera_data_path = os.path.join(cfg.CALIB_DATA_PATH, camera_name)
        assert os.path.exists(camera_data_path), "Camera data not found."

        self.vidcap = cv2.VideoCapture(video_path)

        self.dist = np.load(os.path.join(camera_data_path, 'dist.npy'))
        self.mtx = np.load(os.path.join(camera_data_path, 'mtx.npy'))

        self.start_time_msec = start_time_msec
        self.end_time_msec = end_time_msec
        self.current_time_msec = start_time_msec

        self.max_dim = max_dim

        self.delay_msec = int(1000 / fps)

    def reset(self, start_time_msec, end_time_msec):

        self.start_time_msec = start_time_msec
        self.end_time_msec = end_time_msec
        self.current_time_msec = start_time_msec

    def __iter__(self):
        self.current_time_msec = self.start_time_msec
        return self

    def __len__(self):
        return (self.end_time_msec - self.start_time_msec) / self.delay_msec + 1

    def next(self):

        success = (self.current_time_msec <= self.end_time_msec)

        if success:
            self.vidcap.set(cv2.CAP_PROP_POS_MSEC, self.current_time_msec)
            success, frame = self.vidcap.read()
            if success:
                print ">> Loading frame at {}".format(msec2string(self.current_time_msec))
                frame = self.process_frame(frame)
                self.current_time_msec += self.delay_msec

        if success:
            return frame
        else:
            raise StopIteration

    def process_frame(self, frame):
        frame = undistort(frame, self.dist, self.mtx)
        frame = downscale(frame, self.max_dim)
        return frame

    def get_last_frame_time(self):
        if self.current_time_msec == self.start_time_msec:
            return -1
        else:
            return self.current_time_msec - self.delay_msec


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