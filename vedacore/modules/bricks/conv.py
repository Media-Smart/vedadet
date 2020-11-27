from torch import nn

from vedadet.ops import DeformConvPack, ModulatedDeformConvPack

CONV_LAYERS = {
    'Conv1d': nn.Conv1d,
    'Conv2d': nn.Conv2d,
    'Conv3d': nn.Conv3d,
    'Conv': nn.Conv2d,
    'DCN': DeformConvPack,
    'DCNv2': ModulatedDeformConvPack
}


def build_conv_layer(cfg, *args, **kwargs):
    """Build convolution layer.

    Args:
        cfg (None or dict): The conv layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.

    Returns:
        nn.Module: Created conv layer.
    """
    if cfg is None:
        cfg_ = dict(typename='Conv2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'typename' not in cfg:
            raise KeyError('the cfg dict must contain the key "typename"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('typename')
    if layer_type not in CONV_LAYERS:
        raise KeyError(f'Unrecognized norm type {layer_type}')
    else:
        conv_layer = CONV_LAYERS.get(layer_type)

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer
