import torch

from vedacore.misc import registry
from .infer_engine import InferEngine


@registry.register_module('engine')
class ValEngine(InferEngine):
    def __init__(self, detector, meshgrid, converter, eval_metric):
        super().__init__(detector, meshgrid, converter)
        self.eval_metric = eval_metric

    def forward(self, data):
        return self.forward_impl(**data)

    def forward_impl(self, img, img_metas):
        dets = self.infer(img[0], img_metas[0])
        return dets
