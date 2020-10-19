from vedacore.misc import registry, build_from_cfg


def build_base_anchor(cfg):
    return build_from_cfg(cfg, registry, 'base_anchor')
