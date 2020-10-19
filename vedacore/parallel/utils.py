# Copyright (c) Open-MMLab. All rights reserved.
from torch.nn.parallel import DataParallel, DistributedDataParallel


def is_module_wrapper(module):
    """Check if a module is a module wrapper.

    The following 3 modules in MMCV (and their subclasses) are regarded as
    module wrappers: DataParallel, DistributedDataParallel,
    MMDistributedDataParallel (the deprecated version). You may add you own
    module wrapper by registering it to parallel.MODULE_WRAPPERS.

    Args:
        module (nn.Module): The module to be checked.

    Returns:
        bool: True if the input module is a module wrapper.
    """
    module_wrappers = (DataParallel, DistributedDataParallel)
    return isinstance(module, module_wrappers)
