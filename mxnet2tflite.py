from argparse import ArgumentParser
import numpy as np
from mxnet.contrib import onnx as onnx_mxnet

parser = ArgumentParser(description='Convert MXNet models to TensorFlow Lite models')
parser.add_argument('--symbol')
parser.add_argument('--params')
parser.add_argument('--input_shape')

args = parser.parse_args()

symbol_path = args.symbol
params_path = args.params
input_shape = list(map(int, args.input_shape.split(",")))

onnx_file = '__model.onnx'

converted_model_path = onnx_mxnet.export_model(symbol_path, params_path, [input_shape], np.float32, onnx_file)