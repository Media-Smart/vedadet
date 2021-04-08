# 1. data
dataset_type = 'WIDERFaceDataset'
data_root = 'data/WIDERFace/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
size_divisor = 32

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        typename=dataset_type,
        ann_file=data_root + 'WIDER_train/train.txt',
        img_prefix=data_root + 'WIDER_train/',
        min_size=1,
        offset=0,
        pipeline=[
            dict(typename='LoadImageFromFile', to_float32=True),
            dict(typename='LoadAnnotations', with_bbox=True),
            dict(typename='RandomSquareCrop',
                 crop_choice=[0.3, 0.45, 0.6, 0.8, 1.0]),
            dict(
                typename='PhotoMetricDistortion',
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18),
            dict(typename='RandomFlip', flip_ratio=0.5),
            dict(typename='Resize', img_scale=(640, 640), keep_ratio=False),
            dict(typename='Normalize', **img_norm_cfg),
            dict(typename='DefaultFormatBundle'),
            dict(typename='Collect', keys=['img', 'gt_bboxes',
                                           'gt_labels', 'gt_bboxes_ignore']),
        ]),
    val=dict(
        typename=dataset_type,
        ann_file=data_root + 'WIDER_val/val.txt',
        img_prefix=data_root + 'WIDER_val/',
        min_size=1,
        offset=0,
        pipeline=[
            dict(typename='LoadImageFromFile'),
            dict(
                typename='MultiScaleFlipAug',
                img_scale=(1100, 1650),
                flip=False,
                transforms=[
                    dict(typename='Resize', keep_ratio=True),
                    dict(typename='RandomFlip', flip_ratio=0.0),
                    dict(typename='Normalize', **img_norm_cfg),
                    dict(typename='Pad', size_divisor=32, pad_val=0),
                    dict(typename='ImageToTensor', keys=['img']),
                    dict(typename='Collect', keys=['img'])
                ])
        ]),
)

# 2. model
num_classes = 1
strides = [4, 8, 16, 32, 64, 128]
use_sigmoid = True
scales_per_octave = 3
ratios = [1.3]
num_anchors = scales_per_octave * len(ratios)

model = dict(
    typename='SingleStageDetector',
    backbone=dict(
        typename='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(typename='GN', num_groups=32, requires_grad=True),
        norm_eval=False,
        dcn=dict(typename='DCN', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
        style='pytorch'),
    neck=[
        dict(
            typename='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=0,
            add_extra_convs='on_input',
            num_outs=6,
            norm_cfg=dict(typename='GN', num_groups=32, requires_grad=True),
            upsample_cfg=dict(mode='bilinear')),
        dict(
            typename='Inception',
            in_channel=256,
            num_levels=6,
            norm_cfg=dict(typename='GN', num_groups=32, requires_grad=True),
            share=True)
    ],
    head=dict(
        typename='IoUAwareRetinaHead',
        num_classes=num_classes,
        num_anchors=num_anchors,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        norm_cfg=dict(typename='GN', num_groups=32, requires_grad=True),
        use_sigmoid=use_sigmoid))

# 3. engines
meshgrid = dict(
    typename='BBoxAnchorMeshGrid',
    strides=strides,
    base_anchor=dict(
        typename='BBoxBaseAnchor',
        octave_base_scale=2**(4 / 3),
        scales_per_octave=scales_per_octave,
        ratios=ratios,
        base_sizes=strides))

bbox_coder = dict(
    typename='DeltaXYWHBBoxCoder',
    target_means=[.0, .0, .0, .0],
    target_stds=[0.1, 0.1, 0.2, 0.2])

train_engine = dict(
    typename='TrainEngine',
    model=model,
    criterion=dict(
        typename='IoUBBoxAnchorCriterion',
        num_classes=num_classes,
        meshgrid=meshgrid,
        bbox_coder=bbox_coder,
        loss_cls=dict(
            typename='FocalLoss',
            use_sigmoid=use_sigmoid,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        reg_decoded_bbox=True,
        loss_bbox=dict(typename='DIoULoss', loss_weight=2.0),
        loss_iou=dict(
            typename='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0),
        train_cfg=dict(
            assigner=dict(
                typename='MaxIoUAssigner',
                pos_iou_thr=0.35,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1,
                gpu_assign_thr=100),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)),
    optimizer=dict(typename='SGD', lr=3.75e-3, momentum=0.9, weight_decay=5e-4)) # 3 GPUS

## 3.2 val engine
val_engine = dict(
    typename='ValEngine',
    model=model,
    meshgrid=meshgrid,
    converter=dict(
        typename='IoUBBoxAnchorConverter',
        num_classes=num_classes,
        bbox_coder=bbox_coder,
        nms_pre=-1,
        use_sigmoid=use_sigmoid),
    num_classes=num_classes,
    test_cfg=dict(
        min_bbox_size=0,
        score_thr=0.01,
        nms=dict(typename='lb_nms', iou_thr=0.45),
        max_per_img=-1),
    use_sigmoid=use_sigmoid,
    eval_metric=None)

hooks = [
    dict(typename='OptimizerHook'),
    dict(
        typename='CosineRestartLrSchedulerHook',
        periods=[30] * 21,
        restart_weights=[1] * 21,
        warmup='linear',
        warmup_iters=500,
        warmup_ratio=1e-1,
        min_lr_ratio=1e-2),
    dict(typename='EvalHook'),
    dict(typename='SnapshotHook', interval=1),
    dict(typename='LoggerHook', interval=100)
]

# 5. work modes
modes = ['train']#, 'val']
max_epochs = 630

# 6. checkpoint
weights = dict(
    filepath='torchvision://resnet50',
    prefix='backbone')
# optimizer = dict(filepath='workdir/retinanet_mini/epoch_3_optim.pth')
# meta = dict(filepath='workdir/retinanet_mini/epoch_3_meta.pth')

# 7. misc
seed = 1234
dist_params = dict(backend='nccl')
log_level = 'INFO'
