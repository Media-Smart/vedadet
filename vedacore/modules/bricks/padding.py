import torch.nn as nn

PADDING_LAYERS = {
    'zero': nn.ZeroPad2d,
    'reflect': nn.ReflectionPad2d,
    'replicate': nn.ReplicationPad2d
}


def build_padding_layer(cfg, *args, **kwargs):
    """Build padding layer.

    Args:
        cfg (None or dict): The padding layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a padding layer.

    Returns:
        nn.Module: Created padding layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'typename' not in cfg:
        raise KeyError('the cfg dict must contain the key "typename"')

    cfg_ = cfg.copy()
    padding_type = cfg_.pop('typename')
    if padding_type not in PADDING_LAYERS:
        raise KeyError(f'Unrecognized padding type {padding_type}.')
    else:
        padding_layer = PADDING_LAYERS.get(padding_type)

    layer = padding_layer(*args, **kwargs, **cfg_)

    return layer
