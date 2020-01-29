from argparse import ArgumentParser
import numpy as np
from mxnet.contrib import onnx as onnx_mxnet
import onnx
import onnx_tf
import tensorflow as tf

parser = ArgumentParser(description='Convert MXNet models to TensorFlow Lite models')
parser.add_argument('--symbol')
parser.add_argument('--params')
parser.add_argument('--input_shape')
parser.add_argument('--input_name')
parser.add_argument('--output_name')

args = parser.parse_args()

symbol_path = args.symbol
params_path = args.params
input_shape = list(map(int, args.input_shape.split(",")))


onnx_path = '__model.onnx'

converted_model_path = onnx_mxnet.export_model(symbol_path, params_path, [input_shape], np.float32, onnx_path)

tf_path = '__model.pb'

onnx_model = onnx.load(onnx_path)
# Maybe you have to do this workaround https://github.com/onnx/onnx-tensorflow/issues/377#issuecomment-464714597
tf_rep = onnx_tf.backend.prepare(onnx_model)
tf_rep.export_graph(tf_path)

converter = tf.lite.TFLiteConverter.from_frozen_graph(
    tf_path,
    [args.input_name],
    [args.output_name] 
)
tflite_model = converter.convert()

tflite_path = '__model.tflite'
open(tflite_path, "wb").write(tflite_model)
