from vedacore.misc import build_from_cfg, registry, singleton_arg


@singleton_arg
def build_meshgrid(cfg):
    return build_from_cfg(cfg, registry, 'meshgrid')
