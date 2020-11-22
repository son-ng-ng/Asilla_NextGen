import cv2
import numpy as np
from helper import *
import copy
p_color = [(0, 255, 255), (0, 191, 255),(0, 255, 102),(0, 77, 255), (0, 255, 0),
            (77,255,255), (77, 255, 204), (77,204,255), (191, 255, 77), (77,191,255), (191, 255, 77), 
            (204,77,255), (77,255,204), (191,77,255), (77,255,191), (127,77,255), (77,255,127), (255,255,70), (255,180,20), (20,180,255), (155,155,155)] 
line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50), 
            (77,255,222), (77,196,255), (77,135,255), (191,255,77), 
            (77,255,77), (77,222,255), (255,156,127), (0,127,255), 
            (255,127,77), (0,77,255), (127,127,255), (255,0,127), 
            (0,127,0), (255,255,128), (0,0 ,50), (0,150 ,50), (255,180,20), (20,180,255)]




def remove_abnormal(joints, coef=2, **kwargs):
    '''
    Remove abnormal joints,
    return False if not match the requirement
    '''
    if coef > 0:
        x_nz = joints[:, 0][np.nonzero(joints[:, 0])[0]]
        x_std, x_mean = x_nz.std(), x_nz.mean()
        y_nz = joints[:, 1][np.nonzero(joints[:, 1])[0]]
        y_std, y_mean = y_nz.std(), y_nz.mean()
        x_lower, x_upper = x_mean - coef * x_std, x_mean + coef * x_std
        y_lower, y_upper = y_mean - coef * y_std, y_mean + coef * y_std
         # print(x_std, y_std)
        for idx, joint in enumerate(joints):
            if not (x_upper >= joint[0] >= x_lower) or not (y_upper >= joint[1] >= y_lower):
                joint[:] = 0
    return joints
class DrawObjects(object):

    def __init__(self, topology):
        self.topology = topology
    def __call__(self, image, object_counts, objects, normalized_peaks):
        topology = self.topology
        height = image.shape[0]
        width = image.shape[1]
        K = topology.shape[0]
        count = int(object_counts[0])
        poses = np.zeros((count, 18, 2))
        K = topology.shape[0]
        for i in range(count):
            person = []
            color = (0, 255, 0)
            obj = objects[0][i]
            C = obj.shape[0]
            for j in range(C):
                k = int(obj[j])
                if k >= 0:
                    peak = normalized_peaks[0][j][k]
                    x = round(float(peak[1]) * width)
                    y = round(float(peak[0]) * height)
                    #cv2.circle(image, (x, y), 3, color, 2)
                    person.append([x,y])
                else:
                    person.append([0, 0])
            poses[i,:15] = np.array(person)

            # print(topology)
            # for k in range(K):
            #     c_a = topology[k][2]
            #     c_b = topology[k][3]
            #     if obj[c_a] >= 0 and obj[c_b] >= 0:
            #         peak0 = normalized_peaks[0][c_a][obj[c_a]]
            #         peak1 = normalized_peaks[0][c_b][obj[c_b]]
            #         x0 = round(float(peak0[1]) * width)
            #         y0 = round(float(peak0[0]) * height)
            #         x1 = round(float(peak1[1]) * width)
            #         y1 = round(float(peak1[0]) * height)
            #         cv2.line(image, (x0, y0), (x1, y1), color, 2)
        # for i in range(poses.shape[0]):
        #     poses[i,...] = remove_abnormal(poses[i,...])
        #print(poses)
        return image, poses

    def draw_objects(self, img, poses, MIN_THICK=5, MAX_THICK=20):
        ALPHA = 0.5
        output = copy.deepcopy(img)
        if poses.shape[0] == 0:
            return img
        for i in range(poses.shape[0]):
            x_nonzero = poses[i,:,0][poses[i,:,0] != 0]
            y_nonzero = poses[i,:,1][poses[i,:,1] != 0]
            if len(x_nonzero) == 0:
              continue
            box_ = [min(x_nonzero),
                    min(y_nonzero),
                    max(x_nonzero),
                    max(y_nonzero)]
            thickness = 1
            current_pose = poses[i]
            thickness = self._get_line_size(box_, current_pose, img.shape[0], MIN_THICK, MAX_THICK)
            self._draw_limbs(current_pose, img, thickness)
        img = self._make_transparent(img, output, ALPHA)
        return img
    def _draw_limbs(self, current_pose, img, thickness=2):
      for i, pair in enumerate(L_PAIR):
        if i > 14:
            break
        c_a, c_b, tag = pair[0], pair[1], pair[2]
        x0 = int(current_pose[c_a][0])
        y0 = int(current_pose[c_a][1])
        x1 = int(current_pose[c_b][0])
        y1 = int(current_pose[c_b][1])

        if (x0 > 0 and x1 > 0 and y0 > 0 and y1 > 0):
            if (x1- x0) > 0.3*img.shape[1] or (y0-y1) > 0.3*img.shape[0]:
                pass
                #self._get_abnormal(img, LINE_COLOR[i], current_pose)
                #print((x0,y0), (x1,y1))
            cv2.line(img, (x0, y0), (x1, y1), LINE_COLOR[i], thickness)
            cv2.putText(img, tag, (x1, y1), cv2.FONT_HERSHEY_PLAIN,
                      max(0.5, thickness/5), LINE_COLOR[i], 1)
            cv2.circle(img, (x0, y0), 1, LINE_COLOR[i], thickness)
            cv2.circle(img, (x1, y1), 1, LINE_COLOR[i], thickness)
    def _make_transparent(self, overlay, output, alpha=0.7):
      # apply the overlay
      cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
      return output
    def _get_line_size(self, box, poses, height, min_thick=1, max_thick=10):
      # {2,  "RShoulder"},
      # {5,  "LShoulder"},
      # {8,  "RHip"},
      # {11, "LHip"},
      if poses[2][1] < 1 :
        pose_height = 0
      else:
        pose_height = min(poses[8][1] - poses[2][1], poses[11][1] - poses[2][1])*3
      #print(pose_height, box[3] - box[1])
      return max(min_thick,
                int(max_thick * max(box[3] - box[1], pose_height) / height))
#poses nx14x2
class PoseVisualizer:
    def __init__(self, topology, keypoint_radius=2, line_thickness=2):
        self.topology = topology
        self.keypoint_radius = keypoint_radius
        self.line_thickness = line_thickness
        
    def draw_humans(self, frame, results):
        body_pair = np.array(self.topology) - 1
        for single_person_joints in results:
            part_line = {}
            for i in range(single_person_joints.shape[0]):
                x = int(single_person_joints[i, 0])
                y = int(single_person_joints[i, 1])
                if x==0 and y==0:
                    continue
                part_line[i] = (x, y)
                cv2.circle(frame, (int(x), int(y)), self.keypoint_radius, p_color[i], -1)
            for i, (start_p, end_p) in enumerate(body_pair):
                if start_p in part_line and end_p in part_line:
                    start_xy = part_line[start_p]
                    end_xy = part_line[end_p]

                    cv2.line(frame, start_xy, end_xy, line_color[i], self.line_thickness)
        return frame

