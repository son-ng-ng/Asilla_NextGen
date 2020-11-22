import json
import cv2
import time
import os
import datetime
import numpy as np
import copy
import yaml

import math
from collections import deque
# from utils.logger import Logger

from state.state_cluster import StateCluster
'''
0 đang làm việc
1 nghỉ
'''


with open('state/hyp.yaml', 'r') as f:

    hyp = yaml.load(f, Loader=yaml.FullLoader)



class StateEstimate(object):
    def __init__(self):
        super(StateEstimate, self).__init__()
        angle1 = hyp['angle1']
        angle2 = hyp['angle2']
        angle3 = hyp['angle3']

        self.min_rest = hyp['min_rest']
        self.deque_state_size = hyp['deque_state_size']
        self.work_interval = hyp['work_interval']

        self.angle_gt_min = np.array([angle1[0], angle1[0], angle2[0], angle2[0], angle3[0], angle3[0]])
        self.angle_gt_max = np.array([angle1[1], angle1[1], angle2[1], angle2[1], angle3[1], angle3[1]])
        self.state_data = {}
        self.state_cluster = {}

        self.check_working = False
        self.step_time_product = []

    def predict(self, frm):

        self.frm = copy.deepcopy(frm) 
        _state_data = self.estimate_state(frm)

        self.count_product(_state_data)


    def draw(self, img):

        id_ = 0
        if id_ in self.state_data:
            img = cv2.rectangle(img, (670,0), (1279, (len(self.state_data[id_]['list_time']) + 2)*60 + 50 ), (255,255,255),-1)
            
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (680,30 + 60)
            fontScale              = 1
            fontColor              = (255,0,0)
            lineType               = 2

            img = cv2.putText(img,str('count : {}'.format(self.state_data[id_]['count_product'])), 
                                bottomLeftCornerOfText, 
                                font, 
                                fontScale,
                                fontColor,
                                lineType)

            for i, t in enumerate(self.state_data[id_]['list_time']):
                bottomLeftCornerOfText = (680,60*(i+2))
                img = cv2.putText(img,str('product {}: {} sec'.format(i+1, round(sum(self.state_data[id_]['step_time_product'][i]), 1) )), 
                                    bottomLeftCornerOfText, 
                                    font, 
                                    fontScale,
                                    fontColor,
                                    lineType)

                mess = ''

                if self.state_data[id_]['step_time_product'][i][0] < self.state_data[id_]['step_time_product'][i][1]:
                    self.state_data[id_]['step_time_product'][i] = self.state_data[id_]['step_time_product'][i][::-1]
            
                for step, time in enumerate(self.state_data[id_]['step_time_product'][i]):
                    t = 'step {}: {}'.format(step + 1, round(time,1))
                    mess = mess + t + ' - '
                mess = '({})'.format(mess[:-3])

                bottomLeftCornerOfText = (700,60*(i+2) + 30)
                img = cv2.putText(img, mess, 
                                    bottomLeftCornerOfText, 
                                    font, 
                                    fontScale,
                                    fontColor,
                                    lineType)
            
            i = len(self.state_data[id_]['list_time'])
            bottomLeftCornerOfText = (680, 60*(i+2) )
            
            if self.state_data[id_]['state'] == 1: 
                mess = 'product {}: waiting'.format(i+1)
            else:
                mess = 'product {}: doing'.format(i+1)
            img = cv2.putText(img, mess, 
                                bottomLeftCornerOfText, 
                                font,
                                fontScale,
                                fontColor,
                                lineType)
            # bottomLeftCornerOfText = (0, 960)

            # img = cv2.putText(img, str(self.state_data[id_]['list_state'][-1]), 
            #                     bottomLeftCornerOfText, 
            #                     font, 
            #                     fontScale,
            #                     fontColor,
            #                     lineType)



        return img

    def estimate_state(self, frm):
        state_data = {}
        for id_ in frm.track_data:
            # print('dfafadfdfafsfa', id_)
            # print('fdsfasfsfdsd', frm.track_data[id_])
            # print(type(frm.track_data[id_]))
            temp = []
            for pose in frm.track_data[id_]:
                # asilla pose 
                right_shoulder = pose[2]
                right_elbow = pose[3]
                right_wrist = pose[4]
                left_shoulder = pose[5]
                left_elbow = pose[6]
                left_wrist = pose[7]
                right_hip  = pose[8]
                right_knee = pose[9]
                left_hip = pose[11]
                left_knee = pose[12]

                #alpha pose
                # right_shoulder = pose[6]
                # right_elbow = pose[8]
                # right_wrist = pose[10]
                # left_shoulder = pose[5]
                # left_elbow = pose[7]
                # left_wrist = pose[9]
                # right_hip  = pose[12]
                # right_knee = pose[14]
                # left_hip = pose[11]
                # left_knee = pose[13]

                list_angle = []
                #angle1
                v1 = left_shoulder - right_shoulder
                v2 = right_elbow - right_shoulder
                list_angle.append(180.0*angle(v1, v2)/math.pi)

                v1 = right_shoulder - left_shoulder
                v2 = left_elbow - left_shoulder
                list_angle.append(180.0*angle(v1, v2)/math.pi)

                #angle2
                v1 = right_shoulder - right_elbow
                v2 = right_wrist - right_elbow
                list_angle.append(180.0*angle(v1, v2)/math.pi)
                
                v1 = left_shoulder - left_elbow
                v2 = left_wrist - left_elbow
                list_angle.append(180.0*angle(v1, v2)/math.pi)

                #angle3
                v1 = right_shoulder - right_hip
                v2 = right_knee - right_hip
                list_angle.append(180.0*angle(v1, v2)/math.pi)
                
                v1 = left_shoulder - left_hip
                v2 = left_knee - left_hip
                list_angle.append(180.0*angle(v1, v2)/math.pi)

                list_angle = np.array(list_angle)

                state = np.logical_and(list_angle >= self.angle_gt_min, list_angle <= self.angle_gt_max)
                state = sum(state) > 4 # >4 là đang nghỉ 

                temp.append(state)

            state_data[id_] = sum(temp) > 10     #20frame/2 
            # state_data[id_] = sum(temp) > 480/2     #120 frame/2 
            
        return state_data

    def count_product(self, _state_data):

        time_ = time.time()
        for id_ in _state_data:
            if id_ == 0:
                
                if id_ not in self.state_data:
                    self.state_data[id_] = {'state': 1,
                                            # 'list_state': deque([_state_data[id_]], self.deque_state_size), 
                                            'list_state': deque([1]*self.deque_state_size, self.deque_state_size), 
                                            'time': time_, 
                                            'count_product': 0, 
                                            'list_time': [], 
                                            'step_time_product' : []}
                    self.state_cluster[id_] = StateCluster()

                    # if self.check_working:
                        # self.state_cluster[id_].update(self.frm.track_data[id_])
                    
                else:
                    if self.state_data[id_]['state'] == 0 :
                        if sum(self.state_data[id_]['list_state']) < 5:
                            self.state_cluster[id_].update(self.frm.track_data[id_], self.frm.time)
                    
                    self.state_data[id_]['list_state'].append(_state_data[id_])
                    #rest to work
                    if self.state_data[id_]['state'] == 1 and \
                        sum(self.state_data[id_]['list_state']) < len(self.state_data[id_]['list_state'])/2:
                        
                        self.state_data[id_]['state'] = 0
                        self.state_data[id_]['time'] = time_
                    
                    #work to rest
                    elif self.state_data[id_]['state'] == 0 and \
                        sum(self.state_data[id_]['list_state']) >= len(self.state_data[id_]['list_state'])/2 and \
                        time_ - self.state_data[id_]['time'] > self.work_interval: 

                        
                        self.state_data[id_]['count_product'] += 1

                        self.state_data[id_]['state'] = 1
                        self.state_data[id_]['time'] = time_
                        
                        self.state_cluster[id_].cluster()
                        self.state_data[id_]['list_time'].append(int(sum(self.state_cluster[id_].state_cluster_time)))
                        self.state_data[id_]['step_time_product'].append(self.state_cluster[id_].state_cluster_time)
                        self.state_cluster[id_].reset()



def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
    try:
        if not ((length(v1) * length(v2))):
            return 0
        return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))
    except:
        return 0 