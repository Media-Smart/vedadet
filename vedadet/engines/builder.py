from vedacore.misc import build_from_cfg, registry


def build_engine(cfg):
    return build_from_cfg(cfg, registry, 'engine')
