from vedacore.misc import build_from_cfg, registry, singleton_arg


def build_backbone(cfg):
    """Build backbone."""
    return build_from_cfg(cfg, registry, 'backbone')


def build_neck(cfg):
    """Build neck."""
    return build_from_cfg(cfg, registry, 'neck')


def build_head(cfg):
    """Build head."""
    return build_from_cfg(cfg, registry, 'head')


@singleton_arg
def build_detector(cfg):
    return build_from_cfg(cfg, registry, 'detector')
