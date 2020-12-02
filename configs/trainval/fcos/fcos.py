# 1. data
dataset_type = 'CocoDataset'
data_root = '/media/data/datasets/COCO2017/'
img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True)
size_divisor = 32

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
    train=dict(
        typename=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=[
            dict(typename='LoadImageFromFile'),
            dict(typename='LoadAnnotations', with_bbox=True),
            dict(typename='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(typename='RandomFlip', flip_ratio=0.5),
            dict(typename='Normalize', **img_norm_cfg),
            dict(typename='Pad', size_divisor=size_divisor),
            dict(typename='DefaultFormatBundle'),
            dict(
                typename='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels'])]),
    val=dict(
        typename=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=[
            dict(typename='LoadImageFromFile'),
            dict(
                typename='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(typename='Resize', keep_ratio=True),
                    dict(typename='RandomFlip'),
                    dict(typename='Normalize', **img_norm_cfg),
                    dict(typename='Pad', size_divisor=size_divisor),
                    dict(typename='DefaultFormatBundle'),
                    dict(typename='Collect', keys=['img']),
                ])]))

# 2. model
num_classes = 80
strides = [8, 16, 32, 64, 128]
use_sigmoid = True
regress_ranges = ((-1, 64), (64, 128), (128, 256),
                  (256, 512), (512, 10000))

model = dict(
    typename='SingleStageDetector',
    backbone=dict(
        typename='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(
            typename='BN',
            requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        typename='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        extra_convs_on_inputs=False,  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    head=dict(
        typename='FCOSHead',
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=strides,
        use_sigmoid=use_sigmoid,
        norm_cfg=None))

# 3. engines
meshgrid = dict(
    typename='PointAnchorMeshGrid',
    strides=strides)

train_engine = dict(
    typename='TrainEngine',
    model=model,
    criterion=dict(
        typename='PointAnchorCriterion',
        num_classes=num_classes,
        meshgrid=meshgrid,
        strides=strides,
        regress_ranges=regress_ranges,
        center_sampling=False,
        center_sample_radius=1.5,
        loss_cls=dict(
            typename='FocalLoss',
            use_sigmoid=use_sigmoid,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            typename='IoULoss',
            loss_weight=1.0),
        loss_centerness=dict(
            typename='CrossEntropyLoss',
            use_sigmoid=use_sigmoid,
            loss_weight=1.0)),
    optimizer=dict(
        typename='SGD',
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0001,
        paramwise_cfg=dict(
            bias_lr_mult=2.,
            bias_decay_mult=0.)))

## 3.2 val engine
val_engine = dict(
    typename='ValEngine',
    model=model,
    meshgrid=meshgrid,
    converter=dict(
        typename='PointAnchorConverter',
        num_classes=num_classes,
        nms_pre=1000,
        use_sigmoid=use_sigmoid),
    num_classes=num_classes,
    test_cfg=dict(
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(typename='nms', iou_thr=0.5),
        max_per_img=100),
    use_sigmoid=use_sigmoid,
    eval_metric=None)

# 4. hooks
hooks = [
    dict(
        typename='OptimizerHook',
        grad_clip=dict(
            max_norm=35,
            norm_type=2)),
    dict(
        typename='StepLrSchedulerHook',
        step=[8, 11],
        warmup='constant',
        warmup_iters=500,
        warmup_ratio=1.0 / 10),
    dict(
        typename='SnapshotHook',
        interval=1),
    dict(
        typename='LoggerHook',
        interval=10),
    dict(typename='EvalHook')]

# 5. work modes
modes = ['train', 'val']
max_epochs = 12

# 6. misc
weights = dict(
    filepath='torchvision://resnet50',
    prefix='backbone')
# optimizer = dict(filepath='workdir/fcos/epoch_1_optim.pth')
# meta = dict(filepath='workdir/fcos/epoch_1_meta.pth')

# 7. misc
seed = 0
dist_params = dict(backend='nccl')
log_level = 'INFO'
