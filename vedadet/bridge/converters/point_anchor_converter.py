import torch

from vedacore.misc import registry
from vedadet.misc.bbox import bbox_revert, distance2bbox
from .base_converter import BaseConverter

INF = 1e8


@registry.register_module('converter')
class PointAnchorConverter(BaseConverter):

    def __init__(self, num_classes, nms_pre, revert=True, use_sigmoid=True):
        super().__init__()
        assert use_sigmoid is True
        self.nms_pre = nms_pre
        self.revert = revert
        self.cls_out_channels = num_classes

    def get_bboxes(
        self,
        mlvl_points,
        img_metas,
        cls_scores,
        bbox_preds,
        centernesses,
    ):
        """Transform network output for a batch into bbox predictions.

        Aapted from https://github.com/open-mmlab/mmdetection

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple. \
                The first item is an (n, 5) tensor, where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1. The second item is a \
                (n,) tensor where each item is the predicted class label of \
                the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            result = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                             centerness_pred_list, mlvl_points,
                                             img_metas[img_id], self.nms_pre,
                                             self.revert)
            result_list.append(result)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           centernesses,
                           mlvl_points,
                           img_metas,
                           nms_pre,
                           revert=True):
        """Transform outputs for a single batch item into bbox predictions.

        Aapted from https://github.com/open-mmlab/mmdetection

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                Has shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points * 4, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            cfg (Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            revert (bool): If True, return boxes in original image space.
        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1.
        """
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        img_shape = img_metas['img_shape']
        scale_factor = img_metas['scale_factor']
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, centerness, points in zip(
                cls_scores, bbox_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since  v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        if revert:
            flip = img_metas['flip']
            flip_direction = img_metas['flip_direction']
            mlvl_bboxes = bbox_revert(mlvl_bboxes, img_shape, scale_factor,
                                      flip, flip_direction)
        return mlvl_bboxes, mlvl_scores, mlvl_centerness
