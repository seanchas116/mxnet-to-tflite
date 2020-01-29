from argparse import ArgumentParser

parser = ArgumentParser(description='Convert MXNet models to TensorFlow Lite models')
parser.add_argument('--symbol')
parser.add_argument('--params')
parser.add_argument('--input_shape')

args = parser.parse_args()
print(args.symbol)
print(args.params)
print(args.input_shape)
