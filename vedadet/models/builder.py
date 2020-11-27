import torch.nn as nn

from vedacore.misc import build_from_cfg, registry, singleton_arg


def build(cfg, module_name, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, module_name, default_args)
            for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, module_name, default_args)


def build_backbone(cfg):
    """Build backbone."""
    return build(cfg, 'backbone')


def build_neck(cfg):
    """Build neck."""
    return build(cfg, 'neck')


def build_head(cfg):
    """Build head."""
    return build(cfg, 'head')


@singleton_arg
def build_detector(cfg):
    return build_from_cfg(cfg, registry, 'detector')
