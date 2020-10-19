from vedacore.misc import registry, build_from_cfg


def build_criterion(cfg):
    return build_from_cfg(cfg, registry, 'criterion')
