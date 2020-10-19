import sys
sys.path.insert(0, '/media/data/home/yichaoxiong/workspace/vedadet')

import torch
import unittest

import mmcv.parallel as parallel

from vedadet.data.builder import build_dummy_dataloader
from vedadet.bridge.generators.meshgrids.point_meshgrid import PointMeshGrid
from vedadet.bridge.assigners.fcos_assigner import FCOSAssigner
from vedadet.models import build_model
from volkscv.loopers import EpochBasedLooper
from vedadet.criteria import FCOSCriterion
from vedadet.engines import build_engine
from volkscv.hooks import HookPool


class SimpleTest(unittest.TestCase):
    def test_true(self):
        self.assertTrue(True)

        device = 'cuda:0'
        num_classes = 80
        strides = [8, 16, 32, 64, 128]
        regress_ranges = ((-1, 64), (64, 128), (128, 256), (256, 512), (512,
                                                                        10000))
        model_cfg = dict(
            type='SingleStageDetector',
            device=device,
            backbone_cfg=dict(type='ResNet',
                              depth=50,
                              num_stages=4,
                              out_indices=(0, 1, 2, 3),
                              frozen_stages=1,
                              norm_cfg=dict(type='BN', requires_grad=False),
                              norm_eval=True,
                              style='caffe'
                              ),
            neck_cfg=dict(
                type='FPN',
                in_channels=[256, 512, 1024, 2048],
                out_channels=256,
                start_level=1,
                add_extra_convs=True,
                extra_convs_on_inputs=False,  # use P5
                num_outs=5,
                relu_before_extra_convs=True),
            head_cfg=dict(type='FCOSHead',
                          num_classes=num_classes,
                          in_channels=256,
                          stacked_convs=4,
                          feat_channels=256,
                          strides=strides,
                          norm_cfg=None,
                          loss_cls=dict(type='FocalLoss',
                                        use_sigmoid=True,
                                        gamma=2.0,
                                        alpha=0.25,
                                        loss_weight=1.0),
                          loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                          loss_centerness=dict(type='CrossEntropyLoss',
                                               use_sigmoid=True,
                                               loss_weight=1.0)),
            )

        meshgrid_cfg = dict(type='PointMeshGrid', strides=strides)

        criterion_cfg = dict(type='FCOSCriterion')

        assigner_cfg = dict(type='FCOSAssigner',
                            regress_ranges=regress_ranges,
                            strides=strides,
                            background_label=num_classes)

        optimizer_cfg = dict(type='SGD',
                         lr=0.0003,
                         momentum=0.9,
                         weight_decay=0.0001,
                         paramwise_cfg=dict(bias_lr_mult=2.,
                                            bias_decay_mult=0.))

        converter_cfg = dict(type='FCOSConverter',
                cls_out_channels=num_classes,
                nms_cfg = dict(
                    nms_pre=1000,
                    min_bbox_size=0,
                    score_thr=0.5,
                    nms=dict(type='nms', iou_thr=0.5),
                    max_per_img=100)
                )
        # heart
        heart = build_engine(dict(type='Heart',
                    model_cfg=model_cfg,
                    meshgrid_cfg=meshgrid_cfg)
                    )
        # engine
        val_engine = build_engine(
            dict(type='ValEngine',
                 heart=heart,
                 converter_cfg=converter_cfg,
                 eval_metric=None
                 )
            )
        train_engine = build_engine(
            dict(type='AnchorFreeTrainEngine',
                 heart=heart,
                 criterion_cfg=criterion_cfg,
                 assigner_cfg=assigner_cfg,
                 optimizer_cfg=optimizer_cfg)
            )

        with open(
                '/media/data/home/yichaoxiong/workspace/ection-2.2.1/data.pkl',
                'rb') as fd:
            import pickle
            batch = pickle.load(fd)

        imgs = batch['img'].to(device)
        img_metas = batch['img_metas']
        gt_labels = [ii.to(device) for ii in batch['gt_labels']]
        gt_bboxes = [ii.to(device) for ii in batch['gt_bboxes']]

        idx = 3
        imgs = batch['img'].to(device) #[idx].unsqueeze(0)
        img_metas = batch['img_metas']
        #img_metas = [batch['img_metas'][idx]]
        gt_labels = [ii.to(device) for ii in batch['gt_labels']]
        #gt_labels = [gt_labels[idx]]
        gt_bboxes = [ii.to(device) for ii in batch['gt_bboxes']]
        #gt_bboxes = [gt_bboxes[idx]]

        workflow = [('val', 1)]
        #workflow = [('train', 1), ('val', 1)]
        dataloaders = {'train': [(imgs, img_metas, gt_labels, gt_bboxes)],
                       'val': [(imgs, img_metas, gt_labels, gt_bboxes)]}

        dataloader = build_dummy_dataloader()
        dataloaders = {'train': dataloader, 'val': dataloader}
        engines = {'train': train_engine, 'val': val_engine}
        
        hook_cfgs = []
        optimizer_hook_cfg = dict(type='OptimizerHook')
        hook_cfgs.append(optimizer_hook_cfg)
        lr_scheduler_hook_cfg = dict(type='FixedLrSchedulerHook')
        hook_cfgs.append(lr_scheduler_hook_cfg)

        hook_pool = HookPool(hook_cfgs)

        looper = EpochBasedLooper(workflow, dataloaders, engines, hook_pool)
        looper.start(1)


if __name__ == '__main__':
    unittest.main()
