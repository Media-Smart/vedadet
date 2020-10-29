# adapted from https://github.com/open-mmlab/mmcv
import copy
import inspect
import torch

from vedacore.misc import build_from_cfg, registry


def register_torch_optimizers():
    torch_optimizers = []
    for module_name in dir(torch.optim):
        if module_name.startswith('__'):
            continue
        _optim = getattr(torch.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim,
                                                  torch.optim.Optimizer):
            registry.register_module('optimizer')(_optim)
            torch_optimizers.append(module_name)
    return torch_optimizers


register_torch_optimizers()


def build_optimizer_constructor(cfg):
    return build_from_cfg(cfg, registry, 'optimizer_builder')


def build_optimizer(model, cfg):
    optimizer_cfg = copy.deepcopy(cfg)
    constructor_type = optimizer_cfg.pop('constructor',
                                         'DefaultOptimizerConstructor')
    paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
    optim_constructor = build_optimizer_constructor(
        dict(
            typename=constructor_type,
            optimizer_cfg=optimizer_cfg,
            paramwise_cfg=paramwise_cfg))
    optimizer = optim_constructor(model)
    return optimizer
