from concurrent import futures
import logging

import grpc
# from grpc_reflection.v1alpha import reflection

import pose_pb2
import pose_pb2_grpc
import pickle
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../pose'))
from pose_estimate import PoseEstimator


class Pose(pose_pb2_grpc.PoseServicer):

    def __init__(self):
        self.estimator = PoseEstimator.get_instance()

    def predict(self, request, context):
        imgs = pickle.loads(request.imgs)
        output_shapes = pickle.loads(request.output_shapes)
        poses_list = self.estimator.predict(imgs, output_shapes, is_validate=False)
        return pose_pb2.PoseResponse(poselist=pickle.dumps(poses_list))


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pose_pb2_grpc.add_PoseServicer_to_server(Pose(), server)
    # SERVICE_NAMES = (
    #     pose_pb2.DESCRIPTOR.services_by_name['Pose'].full_name,
    #     reflection.SERVICE_NAME,
    # )
    # reflection.enable_server_reflection(SERVICE_NAMES, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()