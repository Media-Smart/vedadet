## Introduction
vedadet is a single stage object detector toolbox based on PyTorch.

## Features

- **Modular Design**

  We re-design MMDetection based on our taste and needs. Specifically, we decompose detector into four parts: data pipeline, model, postprocessing and criterion which make it easy to convert PyTorch model into TensorRT engine and deploy it on NVIDIA devices such as Tesla V100, Jetson Nano and Jetson AGX Xavier, etc.

- **Support of several popular single stage detector**

  The toolbox supports several popular single stage detector out of the box, *e.g.* RetinaNet, FCOS, etc.
 
- **Friendly to TensorRT**
  
  Detectors can be easily converted to TensorRT engine.
  
- **Easy to deploy**
  
  It's simple to deploy the model accelerate by TensorRT on NVIDIA devices through [Python front-end](https://github.com/Media-Smart/flexinfer) or [C++ front-end](https://github.com/Media-Smart/cheetahinfer).

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Installation
### Requirements

- Linux
- Python 3.7+
- PyTorch 1.6.0 or higher
- CUDA 10.2 or higher

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04.6 LTS
- CUDA: 10.2
- PyTorch 1.6.0
- Python 3.8.5

### Install vedadet

a. Create a conda virtual environment and activate it.

```shell
conda create -n vedadet python=3.8.5 -y
conda activate vedadet
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), *e.g.*,

```shell
conda install pytorch torchvision -c pytorch
```

c. Clone the vedadet repository.

```shell
git clone https://github.com/Media-Smart/vedadet.git
cd vedadet
vedadet_root=${PWD}
```

d. Install vedadet.

```shell
pip install -r requirements/build.txt
pip install -v -e .
```

## Train

a. Config

Modify some configuration accordingly in the config file like `configs/trainval/retinanet/retinanet.py`

b. Multi-GPUs training
```shell
tools/dist_trainval.sh configs/trainval/retinanet/retinanet.py "0,1"
```

c. Single GPU training
```shell
CUDA_VISIBLE_DEVICES="0" python tools/trainval.py configs/trainval/retinanet/retinanet.py
```

## Test

a. Config

Modify some configuration accordingly in the config file like `configs/trainval/retinanet/retinanet.py`

b. Test
```shell
CUDA_VISIBLE_DEVICES="0" python tools/test.py configs/trainval/retinanet/retinanet.py weight_path
```

## Inference

a. Config

Modify some configuration accordingly in the config file like `configs/infer/retinanet/retinanet.py`

b. Inference

```shell
CUDA_VISIBLE_DEVICES="0" python tools/infer.py configs/infer/retinanet/retinanet.py image_path
```

## Deploy

a. Convert to Onnx

Firstly, install volksdep following the [official instructions](https://github.com/Media-Smart/volksdep).

Then, run the following code to convert PyTorch to Onnx. The input shape format is `CxHxW`. If you need the onnx model with constant input shape, please remove `--dynamic_shape` in the end.

```shell
CUDA_VISIBLE_DEVICES="0" python tools/torch2onnx.py configs/trainval/retinanet/retinanet.py weight_path out_path --dummy_input_shape 3,800,1344 --dynamic_shape
```

Here are some unsupported operations for model conversion.
- GN
- Deformable Conv

Please see more details in [this](https://pytorch.org/docs/stable/onnx.html).

b. Inference SDK

Firstly, install flexinfer following the [official instructions](https://github.com/Media-Smart/flexinfer).

Then, see the [example](https://github.com/Media-Smart/flexinfer/tree/master/examples/object_detection) for details.

## Contact

This repository is currently maintained by Yanjia Zhu ([@mike112223](http://github.com/mike112223)), Hongxiang Cai ([@hxcai](http://github.com/hxcai)), Yichao Xiong ([@mileistone](https://github.com/mileistone)).

## Credits
We got a lot of code from [mmcv](https://github.com/open-mmlab/mmcv) and [mmdetection](https://github.com/open-mmlab/mmdetection), thanks to [open-mmlab](https://github.com/open-mmlab).

