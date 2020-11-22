# resize 512 self.frame = FrameData(self.camid, self.frame_idx, frame,self.visual, resized_frame=resized_frame)
# tranpose np.transpose(resized_frame, [2,0,1]) 

# predict
from __future__ import print_function
import time
import sys
import os
import datetime
from tqdm import tqdm
import argparse
import traceback
import numpy as np
import pickle
import copy
# first load
import grpc
from grpc.pose_pb2 import PoseRequest, PoseResponse
from grpc import pose_pb2_grpc
import cv2
import matplotlib.pyplot as plt
import cv2
import numpy as np
from draw_objects import PoseVisualizer
from state.frame_data import FrameData
from state.state_estimate import StateEstimate

topology = [
    [1,3], [1,6], [1,2], [2,3], [2,6],
    [3,4], [4,5], [6,7], [7,8],[3, 15],
    [6, 15], [9, 15], [12, 15],[3, 9], [6, 12],
    [9,12], [9,10], [10,11], [12,13], [13, 14]
    ]
posevisual = PoseVisualizer(topology)

def grpc_waitio(stub, imgs, output_shapes):
    try:
        imgs = pickle.dumps(imgs)
        output_shapes = pickle.dumps(output_shapes)
        pose_response = stub.predict(PoseRequest(imgs=imgs, output_shapes=output_shapes))
        poses_list = pickle.loads(pose_response.poselist)
        return poses_list
    except Exception as e:
        print(e)
        traceback.print_exc()


channel = grpc.insecure_channel('40.87.27.144:50051')  #NV6
stub = pose_pb2_grpc.PoseStub(channel)


fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
writer = cv2.VideoWriter('out_14.avi', fourcc, 25, (1280,720))

frm_data = FrameData()
state_estimator = StateEstimate()

output_shapes = [720, 1280]

cap = cv2.VideoCapture('SCENARIO_001_cut.mp4')
c =1
while cap.isOpened():
    tic = time.time()
    r, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_CUBIC)
    resized_frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_CUBIC)

    resized_frame = np.transpose(resized_frame, [2,0,1]) 
    poses_lst = grpc_waitio(stub, [resized_frame], [output_shapes])
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    # print('update')
    frm_data.update(poses_lst[0][0])
    
    # print('state_estimator')
    state_estimator.predict(frm_data)

    # print('draw')
    frame = state_estimator.draw(frame)

    # state_estimator.state_data[0]['step_time_product'] = [[1,4,5,6], [1,4,5,6]]
    # if len(state_estimator.state_data[0]['step_time_product']) >= c:
    #     c +=1
        
    #     for i in range(len(state_estimator.state_data[0]['step_time_product'])):
    #         list_time = []
    #         list_name = []
    #         fig = plt.figure(figsize=(15,10))
    #         for j in range(len(state_estimator.state_data[0]['step_time_product'][i])):
    #             name = 'priduct {} step {}'.format(i, j )
    #             list_name.append(name)
    #             list_time.append(state_estimator.state_data[0]['step_time_product'][i][j])

    #         plt.xticks(range(len(list_time)), list_name)
    #         plt.plot(range(len(list_time)), list_time,  'ro-')
        
    #     # Add labels
    #     plt.title('')
    #     # plt.xlabel()
    #     plt.ylabel('Time /piece')
    #     plt.savefig("distribution.png")


    frame  = np.uint8(posevisual.draw_humans(frame, poses_lst[0][:1]))
    writer.write(frame)

    print('fps ', 1/(time.time() - tic))
    # print(poses_lst[0]) 
writer.release()