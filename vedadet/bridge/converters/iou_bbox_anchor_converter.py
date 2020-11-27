import torch

from vedacore.misc import registry
from vedadet.misc.bbox import bbox_revert, build_bbox_coder
from .base_converter import BaseConverter


@registry.register_module('converter')
class IoUBBoxAnchorConverter(BaseConverter):

    def __init__(self,
                 num_classes,
                 nms_pre,
                 bbox_coder,
                 alpha=0.4,
                 height_th=9,
                 revert=True,
                 use_sigmoid=True):
        super().__init__()
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.alpha = alpha
        self.height_th = height_th
        self.use_sigmoid_cls = use_sigmoid
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.nms_pre = nms_pre
        self.revert = revert

    def get_bboxes(self, mlvl_anchors, img_metas, cls_scores, bbox_preds,
                   iou_preds):
        """Transform network output for a batch into bbox predictions.

        Aapted from https://github.com/open-mmlab/mmdetection

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class labelof the
                corresponding box.

        Example:
            >>> self = AnchorHead(
            >>>     num_classes=9,
            >>>     in_channels=1,
            >>>     anchor_generator=dict(
            >>>         type='AnchorGenerator',
            >>>         scales=[8],
            >>>         ratios=[0.5, 1.0, 2.0],
            >>>         strides=[4,]))
            >>> img_metas = [{'img_shape': (32, 32, 3), 'scale_factor': 1}]
            >>> cfg = Config(dict(
            >>>     score_thr=0.00,
            >>>     nms=dict(type='nms', iou_thr=1.0),
            >>>     max_per_img=10))
            >>> feat = torch.rand(1, 1, 3, 3)
            >>> cls_score, bbox_pred = self.forward_single(feat)
            >>> # note the input lists are over different levels, not images
            >>> cls_scores, bbox_preds = [cls_score], [bbox_pred]
            >>> result_list = self.get_bboxes(cls_scores, bbox_preds,
            >>>                               img_metas, cfg)
            >>> det_bboxes, det_labels = result_list[0]
            >>> assert len(result_list) == 1
            >>> assert det_bboxes.shape[1] == 5
            >>> assert len(det_bboxes) == len(det_labels) == cfg.max_per_img
        """
        assert len(cls_scores) == len(bbox_preds) == len(iou_preds)
        num_levels = len(cls_scores)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            iou_pred_list = [
                iou_preds[i][img_id].detach() for i in range(num_levels)
            ]

            # TODO: hard code. 0 for anchor_list, 1 for valid_flag_list
            anchors = mlvl_anchors[0][img_id]
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                                iou_pred_list, anchors,
                                                img_metas[img_id],
                                                self.nms_pre, self.revert)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           iou_pred_list,
                           mlvl_anchors,
                           img_metas,
                           nms_pre,
                           revert=True):
        """Transform outputs for a single batch item into bbox predictions.

        Aapted from https://github.com/open-mmlab/mmdetection

        Args:
            cls_score_list (list[Tensor]): Box scores for a single scale level
                Has shape (num_anchors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas for a single
                scale level with shape (num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Box reference for a single scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """

        # if len(mlvl_anchors) > 1:
        #     mlvl_anchors = mlvl_anchors[0]

        def filter_boxes(boxes, min_scale, max_scale):
            ws = boxes[:, 2] - boxes[:, 0]
            hs = boxes[:, 3] - boxes[:, 1]
            scales = torch.sqrt(ws * hs)

            return (scales >= max(1, min_scale)) & (scales <= max_scale)

        img_shape = img_metas['img_shape']
        scale_factor = img_metas['scale_factor']
        assert len(cls_score_list) == len(bbox_pred_list) == len(
            iou_pred_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, iou_pred, anchors in zip(
                cls_score_list, bbox_pred_list, iou_pred_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            iou_pred = iou_pred.permute(1, 2, 0).reshape(-1, 1).sigmoid()
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)

            if self.alpha is None:
                scores *= iou_pred
            elif isinstance(self.alpha, float):
                scores = torch.pow(scores, 2 * self.alpha) * torch.pow(
                    iou_pred, 2 * (1 - self.alpha))
            else:
                raise ValueError("alpha must be float or None")

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since  v2.0
                    # BG cat_id: num_class
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_scores = torch.cat(mlvl_scores)

        keeps = filter_boxes(mlvl_bboxes, 1, 10000)
        mlvl_bboxes = mlvl_bboxes[keeps]
        mlvl_scores = mlvl_scores[keeps]

        if self.use_sigmoid_cls:
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since  v2.0
            # BG cat_id: num_class
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_centerness = mlvl_scores.new_ones(mlvl_scores.shape[0]).detach()
        if revert:
            flip = img_metas['flip']
            flip_direction = img_metas['flip_direction']
            mlvl_bboxes = bbox_revert(mlvl_bboxes, img_shape, scale_factor,
                                      flip, flip_direction)

        if self.height_th is not None:
            hs = mlvl_bboxes[:, 3] - mlvl_bboxes[:, 1]
            valid = (hs >= self.height_th)
            mlvl_bboxes, mlvl_scores, mlvl_centerness = (
                mlvl_bboxes[valid], mlvl_scores[valid], mlvl_centerness[valid])

        return mlvl_bboxes, mlvl_scores, mlvl_centerness
