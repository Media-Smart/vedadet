from vedacore.misc import build_from_cfg, registry


def build_base_anchor(cfg):
    return build_from_cfg(cfg, registry, 'base_anchor')
