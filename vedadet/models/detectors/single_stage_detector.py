import torch
import torch.nn as nn

from vedacore.misc import registry
from ..builder import build_backbone, build_head, build_neck
from .base_detector import BaseDetector


@registry.register_module('detector')
class SingleStageDetector(BaseDetector):

    def __init__(self, backbone, head, neck=None):
        super().__init__()
        self.backbone = build_backbone(backbone)
        if neck:
            self.neck = build_neck(neck)
        else:
            self.neck = None
        self.bbox_head = build_head(head)

        self.init_weights()

    def init_weights(self):
        self.backbone.init_weights()
        if self.neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()

        self.bbox_head.init_weights()

    def forward_impl(self, x):
        feats = self.backbone(x)
        if self.neck:
            feats = self.neck(feats)
        feats = self.bbox_head(feats)
        return feats

    def forward(self, x, train=True):
        if train:
            self.train()
            feats = self.forward_impl(x)
        else:
            self.eval()
            with torch.no_grad():
                feats = self.forward_impl(x)
        return feats
