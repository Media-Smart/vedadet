from vedacore.misc import registry
from vedacore.optimizers import build_optimizer
from vedadet.criteria import build_criterion
from .base_engine import BaseEngine


@registry.register_module('engine')
class TrainEngine(BaseEngine):

    def __init__(self, model, criterion, optimizer):
        super().__init__(model)
        self.criterion = build_criterion(criterion)
        self.optimizer = build_optimizer(self.model, optimizer)

    def extract_feats(self, img):
        feats = self.model(img, train=True)
        return feats

    def forward(self, data):
        return self.forward_impl(**data)

    def forward_impl(self,
                     img,
                     img_metas,
                     gt_labels,
                     gt_bboxes,
                     gt_bboxes_ignore=None):
        feats = self.extract_feats(img)
        losses = self.criterion.loss(feats, img_metas, gt_labels, gt_bboxes,
                                     gt_bboxes_ignore)
        return losses
