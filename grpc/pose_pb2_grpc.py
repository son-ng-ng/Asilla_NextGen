# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import pose_pb2 as pose__pb2


class PoseStub(object):
    """The greeting service definition.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.predict = channel.unary_unary(
                '/anolla.Pose/predict',
                request_serializer=pose__pb2.PoseRequest.SerializeToString,
                response_deserializer=pose__pb2.PoseResponse.FromString,
                )


class PoseServicer(object):
    """The greeting service definition.
    """

    def predict(self, request, context):
        """Sends a greeting
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_PoseServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'predict': grpc.unary_unary_rpc_method_handler(
                    servicer.predict,
                    request_deserializer=pose__pb2.PoseRequest.FromString,
                    response_serializer=pose__pb2.PoseResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'anolla.Pose', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Pose(object):
    """The greeting service definition.
    """

    @staticmethod
    def predict(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/anolla.Pose/predict',
            pose__pb2.PoseRequest.SerializeToString,
            pose__pb2.PoseResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
