import os
import argparse

import torch
from vedacore.misc import Config, load_weights
from vedadet.models import build_detector
from volksdep.converters import torch2onnx


def parse_args():
    parser = argparse.ArgumentParser(description='Convert to Onnx model.')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('out', help='output onnx file name')
    parser.add_argument('--dummy_input_shape', default='3,800,1344',
                        type=str, help='model input shape like 3,800,1344. '
                                       'Shape format is CxHxW')
    parser.add_argument('--dynamic_shape', default=False, action='store_true',
                        help='whether to use dynamic shape')
    parser.add_argument('--opset_version', default=9, type=int,
                        help='onnx opset version')
    parser.add_argument('--do_constant_folding', default=False,
                        action='store_true',
                        help='whether to apply constant-folding optimization')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='whether print convert info')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = 'cpu'

    model = build_detector(cfg.model)
    load_weights(model, args.checkpoint)
    model.to(device)
    model.forward = model.forward_impl

    shape = map(int, args.dummy_input_shape.split(','))
    dummy_input = torch.randn(1, *shape)

    if args.dynamic_shape:
        print(f'Convert to Onnx with dynamic input shape and '
              f'opset version {args.opset_version}')
    else:
        print(f'Convert to Onnx with constant input shape '
              f'{args.dummy_input_shape} and '
              f'opset version {args.opset_version}')
    torch2onnx(model, dummy_input, args.out, dynamic_shape=args.dynamic_shape,
               opset_version=args.opset_version,
               do_constant_folding=args.do_constant_folding,
               verbose=args.verbose)
    print(f'Convert successfully, saved onnx file: {os.path.abspath(args.out)}')


if __name__ == '__main__':
    main()
