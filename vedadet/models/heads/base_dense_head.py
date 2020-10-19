# adapted from https://github.com/open-mmlab/mmcv or https://github.com/open-mmlab/mmdetection
from abc import ABCMeta, abstractmethod

import torch.nn as nn


class BaseDenseHead(nn.Module, metaclass=ABCMeta):
    """Base class for DenseHeads."""
    def __init__(self):
        super(BaseDenseHead, self).__init__()
