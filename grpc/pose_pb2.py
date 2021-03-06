# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: pose.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='pose.proto',
  package='anolla',
  syntax='proto3',
  serialized_options=b'\n\025io.grpc.anolla.ActionB\010OcrProtoP\001\242\002\003HLW',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\npose.proto\x12\x06\x61nolla\"2\n\x0bPoseRequest\x12\x0c\n\x04imgs\x18\x01 \x01(\x0c\x12\x15\n\routput_shapes\x18\x02 \x01(\x0c\" \n\x0cPoseResponse\x12\x10\n\x08poselist\x18\x01 \x01(\x0c\x32>\n\x04Pose\x12\x36\n\x07predict\x12\x13.anolla.PoseRequest\x1a\x14.anolla.PoseResponse\"\x00\x42)\n\x15io.grpc.anolla.ActionB\x08OcrProtoP\x01\xa2\x02\x03HLWb\x06proto3'
)




_POSEREQUEST = _descriptor.Descriptor(
  name='PoseRequest',
  full_name='anolla.PoseRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='imgs', full_name='anolla.PoseRequest.imgs', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='output_shapes', full_name='anolla.PoseRequest.output_shapes', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=22,
  serialized_end=72,
)


_POSERESPONSE = _descriptor.Descriptor(
  name='PoseResponse',
  full_name='anolla.PoseResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='poselist', full_name='anolla.PoseResponse.poselist', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=74,
  serialized_end=106,
)

DESCRIPTOR.message_types_by_name['PoseRequest'] = _POSEREQUEST
DESCRIPTOR.message_types_by_name['PoseResponse'] = _POSERESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

PoseRequest = _reflection.GeneratedProtocolMessageType('PoseRequest', (_message.Message,), {
  'DESCRIPTOR' : _POSEREQUEST,
  '__module__' : 'pose_pb2'
  # @@protoc_insertion_point(class_scope:anolla.PoseRequest)
  })
_sym_db.RegisterMessage(PoseRequest)

PoseResponse = _reflection.GeneratedProtocolMessageType('PoseResponse', (_message.Message,), {
  'DESCRIPTOR' : _POSERESPONSE,
  '__module__' : 'pose_pb2'
  # @@protoc_insertion_point(class_scope:anolla.PoseResponse)
  })
_sym_db.RegisterMessage(PoseResponse)


DESCRIPTOR._options = None

_POSE = _descriptor.ServiceDescriptor(
  name='Pose',
  full_name='anolla.Pose',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=108,
  serialized_end=170,
  methods=[
  _descriptor.MethodDescriptor(
    name='predict',
    full_name='anolla.Pose.predict',
    index=0,
    containing_service=None,
    input_type=_POSEREQUEST,
    output_type=_POSERESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_POSE)

DESCRIPTOR.services_by_name['Pose'] = _POSE

# @@protoc_insertion_point(module_scope)
