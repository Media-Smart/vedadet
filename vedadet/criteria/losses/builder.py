from vedacore.misc import build_from_cfg, registry


def build_loss(cfg):
    return build_from_cfg(cfg, registry, 'loss')
