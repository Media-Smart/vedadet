# Copyright (c) Open-MMLab. All rights reserved.
from torch.nn.parallel.distributed import DistributedDataParallel

from .scatter_gather import scatter_kwargs


class MMDistributedDataParallel(DistributedDataParallel):
    """The DDP module that supports DataContainer.

    MMDDP has two main differences with PyTorch DDP:

    - It supports a custom type :class:`DataContainer` which allows more
      flexible control of input data.
    """

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)
