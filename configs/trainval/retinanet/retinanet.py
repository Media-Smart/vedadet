# 1. data
dataset_type = 'CocoDataset'
data_root = 'data/COCO2017/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
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
            dict(
                typename='LoadAnnotations',
                with_bbox=True),
            dict(
                typename='Resize',
                img_scale=(1333, 800),
                keep_ratio=True),
            dict(
                typename='RandomFlip',
                flip_ratio=0.5),
            dict(
                typename='Normalize',
                **img_norm_cfg),
            dict(
                typename='Pad',
                size_divisor=size_divisor),
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
                    dict(
                        typename='Resize',
                        keep_ratio=True),
                    dict(typename='RandomFlip'),
                    dict(
                        typename='Normalize',
                        **img_norm_cfg),
                    dict(
                        typename='Pad',
                        size_divisor=size_divisor),
                    dict(typename='DefaultFormatBundle'),
                    dict(
                        typename='Collect',
                        keys=['img'])])]))

# 2. model
num_classes = 80
strides = [8, 16, 32, 64, 128]
use_sigmoid = True
scales_per_octave = 3
ratios = [0.5, 1.0, 2.0]
num_anchors = scales_per_octave * len(ratios)

model = dict(
    typename='SingleStageDetector',
    backbone=dict(
        typename='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,  # TODO
        norm_cfg=dict(
            typename='BN',
            requires_grad=True),  # TODO
        norm_eval=True,
        style='pytorch'),  # TODO
    neck=dict(
        typename='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    head=dict(
        typename='RetinaHead',
        num_classes=num_classes,
        num_anchors=num_anchors,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        use_sigmoid=use_sigmoid))

# 3. engines
meshgrid = dict(
    typename='BBoxAnchorMeshGrid',
    strides=strides,
    base_anchor=dict(
        typename='BBoxBaseAnchor',
        octave_base_scale=4,
        scales_per_octave=scales_per_octave,
        ratios=ratios,
        base_sizes=strides))

bbox_coder = dict(
    typename='DeltaXYWHBBoxCoder',
    target_means=[.0, .0, .0, .0],
    target_stds=[1.0, 1.0, 1.0, 1.0])

train_engine = dict(
    typename='TrainEngine',
    model=model,
    criterion=dict(
        typename='BBoxAnchorCriterion',
        num_classes=num_classes,
        meshgrid=meshgrid,
        bbox_coder=bbox_coder,
        loss_cls=dict(
            typename='FocalLoss',
            use_sigmoid=use_sigmoid,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            typename='L1Loss',
            loss_weight=1.0),
        train_cfg=dict(
            assigner=dict(
                typename='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)),
    optimizer=dict(
        typename='SGD',
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0001))

## 3.2 val engine
val_engine = dict(
    typename='ValEngine',
    model=model,
    meshgrid=meshgrid,
    converter=dict(
        typename='BBoxAnchorConverter',
        num_classes=num_classes,
        bbox_coder=bbox_coder,
        nms_pre=1000,
        use_sigmoid=use_sigmoid),
    num_classes=num_classes,
    test_cfg=dict(
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(
            typename='nms',
            iou_thr=0.5),
        max_per_img=100),
    use_sigmoid=use_sigmoid,
    eval_metric=None)

# 4. hooks
hooks = [
    dict(typename='OptimizerHook'),
    dict(
        typename='StepLrSchedulerHook',
        step=[8, 11],
        warmup='linear',
        warmup_iters=1000,
        warmup_ratio=0.001),
    dict(typename='EvalHook'),
    dict(
        typename='SnapshotHook',
        interval=1),
    dict(
        typename='LoggerHook',
        interval=10)]

# 5. work modes
modes = ['train', 'val']
max_epochs = 12

# 6. checkpoint
weights = dict(
    filepath='torchvision://resnet50',
    prefix='backbone')
# optimizer = dict(filepath='workdir/retinanet_mini/epoch_3_optim.pth')
# meta = dict(filepath='workdir/retinanet_mini/epoch_3_meta.pth')

# 7. misc
seed = 0
dist_params = dict(backend='nccl')
log_level = 'INFO'
