from sklearn.cluster import DBSCAN
import numpy as np
import time
from sklearn.cluster import KMeans
from collections import deque
# X = np.array([[1, 2], [2, 2], [2, 3],
            #   [8, 7], [8, 8], [25, 80]])

# X = np.load('test.npy')
# tic = time.time()
# clustering = DBSCAN(eps=0.5, min_samples=5).fit(X)
# print(1/(time.time() - tic))
# print(clustering.labels_)

import yaml
import json
import cv2
import time
import os
import datetime
import numpy as np
#first load


with open('state/hyp.yaml', 'r') as f:

    hyp = yaml.load(f, Loader=yaml.FullLoader)

class StateCluster(object):
    
    def __init__(self):
        super(StateCluster, self).__init__()
        self.max_state = 2
        self.reset()

    def reset(self):
        self.list_pose = []
        self.list_state = []
        self.list_time = []
        self.state_cluster_time = []


    
    def update(self, pose, time_):

        pose = pose[:, :14, :]
        pose = pose.reshape(20*14*2)
        self.list_pose.append(pose)
        if len(self.list_time) == 0:
            self.list_time.append(time_ - 2)
        else:
            self.list_time.append(time_)


    def state_cluster(self, list_cluster): 
        list_state = deque([], 15)
        list_time = deque([], 15)

        start_time = self.list_time[0]
        temp_state = None

        for i in range(len(list_cluster)):
            list_state.append(list_cluster[i])
            list_time.append(self.list_time[i])

            if len(list_time) == 15:
                c = -1
                index_ = -1 
                next_state = None
                for s in set(list_state):
                    if c < list_state.count(s):
                        c = list_state.count(s)
                        index_ = list_state.index(s)
                        next_state = s

                if temp_state == None:
                    temp_state = next_state
                else:
                    if temp_state != next_state:
                        list_time = list(list_time)
                        list_state = list(list_state)

                        end_time = list_time[index_]
                        self.state_cluster_time.append(round(end_time - start_time,1 ))
                        start_time =  end_time

                        list_state = deque(list_state[index_:], 15)
                        list_time = deque(list_time[index_:], 15)
                    
                        temp_state = next_state

        end_time = list_time[-1]
        self.state_cluster_time.append(round(end_time - start_time,1))

        if len(self.state_cluster_time) > self.max_state:
            self.state_cluster_time[self.max_state-1] = sum(self.state_cluster_time[self.max_state-1:])
            self.state_cluster_time = self.state_cluster_time[:self.max_state]

    def cluster(self):
        np.save('test.npy', self.list_pose)

        X = np.load('test.npy')
        X = X.reshape(len(X), 20 , 14, 2)
        X = X[:, :, 0:8 , :]

        print(0 in X)
        check_raise = False
        for i, list_pose in enumerate(X):
            for j in range(8):
                x = list_pose[:, j, :]
                # print(x)
                if 0 in x: 
                    check_raise = True
                for k in range(len(x)):
                    
                    if 0 in x[k]:
                        check = False
                        for l in range(k+1, len(x)):
                            if 0 not in x[l]:
                                check_tren = x[l]
                                check = True
                        
                        if k == 0:
                            if check:
                                x[k] = check_tren
                        elif k == len(x) - 1:
                            x[k] = x[k -1]
                        else:
                            if check:
                                x[k] = (check_tren + x[k-1])/2
                            else: 
                                x[k] = x[k - 1]
                                    
                # print(x)
                # if check_raise:
                #     raise 'dd'
                list_pose[:, j, :] = x

        self.list_pose = X.reshape(len(X), 20*8*2)

        # clustering = DBSCAN(eps= self.eps, min_samples=self.min_samples).fit(self.list_pose)
        kmeans = KMeans(n_clusters=3, random_state=0).fit(self.list_pose)
        print(kmeans.labels_)
        self.state_cluster(kmeans.labels_)
        print(self.state_cluster_time)
        
        # return kmeans.labels_















