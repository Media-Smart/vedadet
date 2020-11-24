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

Modify some configuration accordingly in the config file like `configs/trainval/retinanet.py`

b. Multi-GPUs training
```shell
tools/dist_trainval.sh configs/trainval/retinanet.py "0,1"
```

c. Single GPU training
```shell
python tools/trainval.py configs/trainval/retinanet.py
```

## Test

a. Config

Modify some configuration accordingly in the config file like `configs/trainval/retinanet.py`

b. Test
```shell
python tools/trainval.py configs/trainval/retinanet.py
```

Logs will be generated at `${vedadet_root}/workdir`.

## Inference

a. Config

Modify some configuration accordingly in the config file like `configs/trainval/retinanet.py`

b. Inference

```shell
python tools/test.py configs/infer/retinanet.py image_path
```

## Deploy
a. Convert to TensorRT engine

To be done.

b. Inference SDK

To be done.

## Contact

This repository is currently maintained by Hongxiang Cai ([@hxcai](http://github.com/hxcai)), Yichao Xiong ([@mileistone](https://github.com/mileistone)).

## Credits
We got a lot of code from [mmcv](https://github.com/open-mmlab/mmcv) and [mmdetection](https://github.com/open-mmlab/mmdetection), thanks to [open-mmlab](https://github.com/open-mmlab).

