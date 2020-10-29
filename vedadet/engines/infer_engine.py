import torch

from vedacore.misc import registry
from vedadet.bridge import build_converter, build_meshgrid
from .base_engine import BaseEngine


@registry.register_module('engine')
class InferEngine(BaseEngine):

    def __init__(self, detector, meshgrid, converter):
        super().__init__(detector)
        self.meshgrid = build_meshgrid(meshgrid)
        self.converter = build_converter(converter)

    def extract_feats(self, img):
        feats = self.detector(img, train=False)
        return feats

    def _single_infer(self, img, img_metas):
        feats = self.extract_feats(img)

        featmap_sizes = [feat.shape[-2:] for feat in feats[0]]
        dtype = feats[0][0].dtype
        device = feats[0][0].device
        anchor_mesh = self.meshgrid.gen_anchor_mesh(featmap_sizes, img_metas,
                                                    dtype, device)
        dets = self.converter.get_bboxes(anchor_mesh, img_metas, *feats)
        return dets

    def _aug_infer(self, img_list, img_metas_list):
        assert len(img_list) == len(img_metas_list)
        det_labels_list = []
        det_bboxes_list = []
        for idx in len(img_list):
            img = img_list[idx]
            img_metas = img_metas_list[idx]
            det_bboxes, det_labels = self._single_infer(img, img_metas)
            det_bboxes_list.append(det_bboxes)
            det_labels_list.append(det_labels)

        det_labels = torch.cat(det_labels_list, dim=0)
        det_bboxes = torch.cat(det_bboxes_list, dim=0)
        return det_labels, det_bboxes

    def infer(self, img, img_metas):
        if isinstance(img, list):
            return self._aug_infer(img, img_metas)
        else:
            res = self._single_infer(img, img_metas)
            return res
