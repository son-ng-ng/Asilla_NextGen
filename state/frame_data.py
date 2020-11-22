import numpy as np
import cv2
import time
import os
import sys
import random
import copy
from threading import Thread, Lock

random.seed(7)
from collections import deque



class FrameData():
    def __init__(self, frmidx = 0, frame = None, poses=None):
        self.frmidx = frmidx
        self.org_frame = frame
        self.org_poses = poses
        self.track_data_ = {0: None}
        self.track_data = {}
        self.action_list = None
        self.time = time.time()
        self.result = {}
        self.mid_dict = {}

    def update(self, result):
        self.time = 1/35 * self.frmidx
        self.frmidx += 1
        if self.track_data_[0] is None:
            self.track_data_[0] = deque([result]*20, 20)
        else:
            self.track_data_[0].append(result)

        self.track_data[0] = np.array(self.track_data_[0])