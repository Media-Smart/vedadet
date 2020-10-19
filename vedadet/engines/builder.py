from vedacore.misc import registry, build_from_cfg


def build_engine(cfg):
    return build_from_cfg(cfg, registry, 'engine')
