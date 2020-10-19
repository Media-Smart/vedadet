import sys
sys.path.insert(0, '/media/data/home/yichaoxiong/workspace/vedadet')

import torch
import unittest
from addict import Dict

import mmcv.parallel as parallel

#from vedadet.data.datasets import build_dataset, build_dataloader
from .datasets import build_dataset, build_dataloader
#from vedadet.bridge.generators.meshgrids.point_meshgrid import PointMeshGrid
from vedadet.models import build_detector
from vedacore.loopers import EpochBasedLooper
#from vedadet.criteria import FCOSCriterion
from vedadet.engines import build_engine
from vedadet.bridge import build_meshgrid
from vedacore.hooks import HookPool


class SimpleTest(unittest.TestCase):
    def test_true(self):
        self.assertTrue(True)

        dataset_type = 'CocoDataset'
        data_root = '/media/data/datasets/COCO2017/'
        img_norm_cfg = dict(
                #mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
                mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
        train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ]
        test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Normalize', **img_norm_cfg),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img']),
                ])
        ]
        data = dict(
            samples_per_gpu=1,
            workers_per_gpu=2,
            train=dict(type=dataset_type,
                       ann_file=data_root +
                       'annotations/instances_minitrain2017.json',
                       img_prefix=data_root + 'train2017/',
                       pipeline=train_pipeline),
            val=dict(type=dataset_type,
                       ann_file=data_root +
                       'annotations/instances_minitrain2017.json',
                       img_prefix=data_root + 'train2017/',
                       pipeline=test_pipeline,
                       test_mode=True,
                       ),
        )

        train_dataset = build_dataset(data['train'])
        val_dataset = build_dataset(data['val'])

        train_dataloader = build_dataloader(
                train_dataset,
                data['samples_per_gpu'],
                data['workers_per_gpu'],
                # cfg.gpus will be ignored if distributed
                1,
                dist=False)
        val_dataloader = build_dataloader(
                val_dataset,
                1,
                data['workers_per_gpu'],
                # cfg.gpus will be ignored if distributed
                1,
                dist=False,
                shuffle=False)

        device = 'cuda:0'
        num_classes = 80
        strides = [8, 16, 32, 64, 128]
        bbox_coder_cfg = dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0])
        detector_cfg = dict(
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
                add_extra_convs='on_input',
                num_outs=5),
            head_cfg=dict(
                type='RetinaHead',
                num_classes=80,
                in_channels=256,
                stacked_convs=4,
                feat_channels=256,
                anchor_generator=dict(
                    type='AnchorGenerator',
                    octave_base_scale=4,
                    scales_per_octave=3,
                    ratios=[0.5, 1.0, 2.0],
                    strides=[8, 16, 32, 64, 128]),
                bbox_coder=bbox_coder_cfg,
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0))
            )

        optimizer_cfg = dict(type='SGD',
                         lr=0.0003,
                         momentum=0.9,
                         weight_decay=0.0001,
                         paramwise_cfg=dict(bias_lr_mult=2.,
                                            bias_decay_mult=0.))

        meshgrid_cfg=dict(type='BBoxAnchorMeshGrid',
                             strides=strides,
                             base_anchor_cfg=dict(type='BBoxBaseAnchor',
                                 octave_base_scale=4,
                                 scales_per_octave=3,
                                 ratios=[0.5, 1.0, 2.0],
                                 base_sizes=strides)
                                )

        meshgrid = build_meshgrid(meshgrid_cfg)

        # heart
        detector = build_detector(detector_cfg)

        # TODO
        criterion_cfg = dict(type='BBoxAnchorCriterion',
                num_classes=num_classes,
                meshgrid=meshgrid,
                bbox_coder_cfg=bbox_coder_cfg,
                loss_cls_cfg=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                loss_bbox_cfg=dict(type='L1Loss', loss_weight=1.0),
                train_cfg=Dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.5,
                        neg_iou_thr=0.4,
                        min_pos_iou=0,
                        ignore_iof_thr=-1),
                    allowed_border=-1,
                    pos_weight=-1,
                    debug=False)
                )

        # engine
        # TODO
        train_engine = build_engine(
            dict(type='TrainEngine',
                 detector=detector,
                 criterion_cfg=criterion_cfg,
                 optimizer_cfg=optimizer_cfg)
            )

        converter_cfg = Dict(type='BBoxAnchorConverter',
                    num_classes=num_classes,
                    test_cfg = dict(
                        nms_pre=1000,
                        min_bbox_size=0,
                        score_thr=0.05,
                        nms=dict(type='nms', iou_thr=0.5),
                        max_per_img=100),
                    bbox_coder_cfg=bbox_coder_cfg,
                    rescale=True
                )
        val_engine = build_engine(
                dict(type='ValEngine',
                     detector=detector,
                     meshgrid=meshgrid,
                     converter_cfg=converter_cfg,
                     eval_metric=None
                     )
            )

        workflow = [('train', 1), ('val', 1)]
        dataloaders = {'train': [(imgs, img_metas, gt_labels, gt_bboxes)],
                       'val': [(imgs, img_metas, gt_labels, gt_bboxes)]}

        dataloaders = {'train': train_dataloader, 'val': val_dataloader}
        engines = {'train': train_engine, 'val': val_engine}
        
        hook_cfgs = []
        optimizer_hook_cfg = dict(type='OptimizerHook')
        hook_cfgs.append(optimizer_hook_cfg)
        lr_scheduler_hook_cfg = dict(type='FixedLrSchedulerHook')
        hook_cfgs.append(lr_scheduler_hook_cfg)
        eval_hook_cfg = dict(type='EvalHook')
        hook_cfgs.append(eval_hook_cfg)

        hook_pool = HookPool(hook_cfgs)

        looper = EpochBasedLooper(workflow, dataloaders, engines, hook_pool)
        looper.start(1)


if __name__ == '__main__':
    unittest.main()
