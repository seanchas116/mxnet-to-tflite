from argparse import ArgumentParser

parser = ArgumentParser(description='Convert MXNet models to TensorFlow Lite models')
parser.add_argument('--symbol')
parser.add_argument('--params')
parser.add_argument('--input_shape')

args = parser.parse_args()

symbol_path = args.symbol
params_path = args.params
input_shape = list(map(int, args.input_shape.split(",")))

print(input_shape)