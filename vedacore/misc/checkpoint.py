import os
import os.path as osp
import pkgutil
import torch
import torchvision
from collections import OrderedDict
from importlib import import_module
from torch.optim import Optimizer
from torch.utils import model_zoo

from ..parallel import get_dist_info, is_module_wrapper
from .utils import mkdir_or_exist


# adapted from https://github.com/open-mmlab/mmcv
def get_torchvision_models():
    model_urls = dict()
    for _, name, ispkg in pkgutil.walk_packages(torchvision.models.__path__):
        if ispkg:
            continue
        _zoo = import_module(f'torchvision.models.{name}')
        if hasattr(_zoo, 'model_urls'):
            _urls = getattr(_zoo, 'model_urls')
            model_urls.update(_urls)
    return model_urls


# adapted from https://github.com/open-mmlab/mmcv
def load_url_dist(url, model_dir=None):
    """ In distributed setting, this function only download checkpoint at
    local rank 0 """
    rank, world_size = get_dist_info()
    rank = int(os.environ.get('LOCAL_RANK', rank))
    if rank == 0:
        checkpoint = model_zoo.load_url(url, model_dir=model_dir)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            checkpoint = model_zoo.load_url(url, model_dir=model_dir)
    return checkpoint


# adapted from https://github.com/open-mmlab/mmcv
def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=''):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        if is_module_wrapper(module):
            module = module.module
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


# adapted from https://github.com/open-mmlab/mmcv
def _load_checkpoint(filepath, map_location=None):
    """Load checkpoint from somewhere (modelzoo, file, url).

    Args:
        filepath (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str | None): Same as :func:`torch.load`. Default: None.

    Returns:
        dict | OrderedDict: The loaded checkpoint. It can be either an
            OrderedDict storing model weights or a dict containing other
            information, which depends on the checkpoint.
    """
    if filepath.startswith('torchvision://'):
        model_urls = get_torchvision_models()
        model_name = filepath[14:]
        checkpoint = load_url_dist(model_urls[model_name])
    elif filepath.startswith(('http://', 'https://')):
        checkpoint = load_url_dist(filepath)
    else:
        if not osp.isfile(filepath):
            raise IOError(f'{filepath} is not a file')
        checkpoint = torch.load(filepath, map_location=map_location)
    return checkpoint


# adapted from https://github.com/open-mmlab/mmcv
def weights_to_cpu(state_dict):
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.

    Returns:
        OrderedDict: Model weights on CPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu


def optimizer_to_cpu(state_dict):
    """Copy a optimizer to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.

    Returns:
        OrderedDict: optimizer on CPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        tmp = dict()
        for k, v in val.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu()
            tmp[k] = v
        state_dict_cpu[key] = tmp
    return state_dict_cpu


def save_weights(model, filepath):
    """Save checkpoint to file.

    Args:
        model (Module): Module whose params are to be saved.
        filepath (str): Checkpoint filepath.
    """
    mkdir_or_exist(osp.dirname(filepath))
    if is_module_wrapper(model):
        model = model.module

    checkpoint = weights_to_cpu(model.state_dict())
    # immediately flush buffer
    with open(filepath, 'wb') as f:
        torch.save(checkpoint, f)
        f.flush()


def save_optimizer(optimizer, filepath):
    """Save checkpoint to file.

    The checkpoint will have 2 fields: ``meta``, ``state_dict``.
    By default ``meta`` will epoch and iteration info.

    Args:
        optimizer (:obj:`Optimizer`): Optimizer to be saved.
        meta (dict): Metadata to be saved in checkpoint.
        filepath (str): Checkpoint filepath.
    """
    mkdir_or_exist(osp.dirname(filepath))
    # save optimizer state dict in the checkpoint
    if isinstance(optimizer, Optimizer):
        state_dict = optimizer.state_dict()
        state_dict['state'] = optimizer_to_cpu(state_dict['state'])
    elif isinstance(optimizer, dict):
        state_dict = {}
        for name, optim in optimizer.items():
            state_dict[name] = optim.state_dict()
            state_dict[name]['state'] = optimizer_to_cpu(state_dict[name]['state'])

    # immediately flush buffer
    with open(filepath, 'wb') as f:
        torch.save(state_dict, f)
        f.flush()


def save_meta(meta, filepath):
    """Save checkpoint to file.

    The checkpoint will have 2 fields: ``meta``, ``state_dict``. By default
    ``meta`` will epoch and iteration info.

    Args:
        optimizer (:obj:`Optimizer`): Optimizer to be saved.
        meta (dict): Metadata to be saved in checkpoint.
        filepath (str): Checkpoint filepath.
    """
    mkdir_or_exist(osp.dirname(filepath))
    # immediately flush buffer
    with open(filepath, 'wb') as f:
        torch.save(meta, f)
        f.flush()


def load_weights(model,
                 filepath,
                 map_location=None,
                 strict=False,
                 logger=None,
                 prefix=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filepath (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    state_dict = _load_checkpoint(filepath, map_location)
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    if prefix is not None:
        state_dict = {'%s.%s' % (prefix, k): v for k, v in state_dict.items()}
    # load state_dict
    load_state_dict(model, state_dict, strict, logger)


def load_optimizer(optimizer, filepath, map_location=None):
    """Load checkpoint from a file or URI.

    Args:
        optimizer (Module): Optimizer to load checkpoint.
        filepath (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
    Returns:
        meta (dict): epoch and iter information
    """
    state_dict = _load_checkpoint(filepath, map_location)
    optimizer.load_state_dict(state_dict)


def load_meta(filepath, map_location=None):
    """Load checkpoint from a file or URI.

    Args:
        optimizer (Module): Optimizer to load checkpoint.
        filepath (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
    Returns:
        meta (dict): epoch and iter information
    """
    state_dict = _load_checkpoint(filepath, map_location)
    return state_dict
