import torch

from vedacore.misc import registry
from vedadet.bridge import build_converter, build_meshgrid
from vedadet.misc.bbox import bbox2result, multiclass_nms
from .base_engine import BaseEngine


@registry.register_module('engine')
class InferEngine(BaseEngine):

    def __init__(self, model, meshgrid, converter, num_classes, use_sigmoid,
                 test_cfg):
        super().__init__(model)
        self.meshgrid = build_meshgrid(meshgrid)
        self.converter = build_converter(converter)
        if use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.test_cfg = test_cfg

    def extract_feats(self, img):
        feats = self.model(img, train=False)
        return feats

    def _get_raw_dets(self, img, img_metas):
        """
        Args:
            img(torch.Tensor): shape N*3*H*W, N is batch size
            img_metas(list): len(img_metas) = N
        Returns:
            dets(list): len(dets) is the batch size, len(dets[ii]) = #classes,
                dets[ii][jj] is an np.array whose shape is N*5
        """
        feats = self.extract_feats(img)

        featmap_sizes = [feat.shape[-2:] for feat in feats[0]]
        dtype = feats[0][0].dtype
        device = feats[0][0].device
        anchor_mesh = self.meshgrid.gen_anchor_mesh(featmap_sizes, img_metas,
                                                    dtype, device)
        # bboxes, scores, score_factor
        dets = self.converter.get_bboxes(anchor_mesh, img_metas, *feats)

        return dets

    def _simple_infer(self, img, img_metas):
        """
        Args:
            img(torch.Tensor): shape N*3*H*W, N is batch size
            img_metas(list): len(img_metas) = N
        Returns:
            dets(list): len(dets) is the batch size, len(dets[ii]) = #classes,
                dets[ii][jj] is an np.array whose shape is N*5
        """
        dets = self._get_raw_dets(img, img_metas)
        batch_size = len(dets)

        result_list = []
        for ii in range(batch_size):
            bboxes, scores, centerness = dets[ii]
            det_bboxes, det_labels = multiclass_nms(
                bboxes,
                scores,
                self.test_cfg.score_thr,
                self.test_cfg.nms,
                self.test_cfg.max_per_img,
                score_factors=centerness)
            bbox_result = bbox2result(det_bboxes, det_labels,
                                      self.cls_out_channels)
            result_list.append(bbox_result)

        return result_list

    def _aug_infer(self, img_list, img_metas_list):
        assert len(img_list) == len(img_metas_list)
        dets = []
        ntransforms = len(img_list)
        for idx in range(len(img_list)):
            img = img_list[idx]
            img_metas = img_metas_list[idx]
            tdets = self._get_raw_dets(img, img_metas)
            dets.append(tdets)
        batch_size = len(dets[0])
        nclasses = len(dets[0][0])
        merged_dets = []
        for ii in range(batch_size):
            single_image = []
            for kk in range(nclasses):
                single_class = []
                for jj in range(ntransforms):
                    single_class.append(dets[jj][ii][kk])
                single_image.append(torch.cat(single_class, axis=0))
            merged_dets.append(single_image)

        result_list = []
        for ii in range(batch_size):
            bboxes, scores, centerness = merged_dets[ii]
            det_bboxes, det_labels = multiclass_nms(
                bboxes,
                scores,
                self.test_cfg.score_thr,
                self.test_cfg.nms,
                self.test_cfg.max_per_img,
                score_factors=centerness)
            bbox_result = bbox2result(det_bboxes, det_labels,
                                      self.cls_out_channels)
            result_list.append(bbox_result)

        return result_list

    def infer(self, img, img_metas):
        if len(img) == 1:
            return self._simple_infer(img[0], img_metas[0])
        else:
            return self._aug_infer(img, img_metas)
