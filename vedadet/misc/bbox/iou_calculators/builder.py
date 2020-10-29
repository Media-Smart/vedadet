from vedacore.misc import build_from_cfg, registry


def build_iou_calculator(cfg, default_args=None):
    """Builder of IoU calculator."""
    return build_from_cfg(cfg, registry, 'iou_calculator', default_args)
