import json
from vedacore.misc import registry, build_from_cfg, singleton_arg


@singleton_arg
def build_meshgrid(cfg):
    return build_from_cfg(cfg, registry, 'meshgrid')
