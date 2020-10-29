# adapted from https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection
from vedacore.misc import registry
from .base_bbox_coder import BaseBBoxCoder


@registry.register_module('bbox_coder')
class PseudoBBoxCoder(BaseBBoxCoder):
    """Pseudo bounding box coder."""

    def __init__(self, **kwargs):
        super(BaseBBoxCoder, self).__init__(**kwargs)

    def encode(self, bboxes, gt_bboxes):
        """torch.Tensor: return the given ``bboxes``"""
        return gt_bboxes

    def decode(self, bboxes, pred_bboxes):
        """torch.Tensor: return the given ``pred_bboxes``"""
        return pred_bboxes
