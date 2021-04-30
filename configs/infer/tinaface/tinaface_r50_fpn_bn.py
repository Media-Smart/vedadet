# 1. data
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
size_divisor = 32

data_pipeline = [
    dict(typename='LoadImageFromFile'),
    dict(
        typename='MultiScaleFlipAug',
        img_scale=(1100, 1650),
        flip=False,
        transforms=[
            dict(typename='Resize', keep_ratio=True),
            dict(typename='RandomFlip', flip_ratio=0.0),
            dict(typename='Normalize', **img_norm_cfg),
            dict(typename='Pad', size_divisor=size_divisor, pad_val=0),
            dict(typename='ImageToTensor', keys=['img']),
            dict(typename='Collect', keys=['img'])
        ])
]

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
        norm_cfg=dict(typename='BN'),
        norm_eval=False,
        dcn=None,
        style='pytorch'),
    neck=[
        dict(
            typename='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=0,
            add_extra_convs='on_input',
            num_outs=6,
            norm_cfg=dict(typename='BN'),
            upsample_cfg=dict(mode='bilinear')),
        dict(
            typename='Inception',
            in_channel=256,
            num_levels=6,
            norm_cfg=dict(typename='BN'),
            share=True)
    ],
    head=dict(
        typename='IoUAwareRetinaHead',
        num_classes=num_classes,
        num_anchors=num_anchors,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        norm_cfg=dict(typename='BN'),
        use_sigmoid=use_sigmoid))

# 3. engine
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

## infer engine
infer_engine = dict(
    typename='InferEngine',
    model=model,
    meshgrid=meshgrid,
    converter=dict(
        typename='IoUBBoxAnchorConverter',
        num_classes=num_classes,
        bbox_coder=bbox_coder,
        nms_pre=10000,
        use_sigmoid=use_sigmoid),
    num_classes=num_classes,
    test_cfg=dict(
        min_bbox_size=0,
        score_thr=0.4,
        nms=dict(
            typename='nms',
            iou_thr=0.45),
        max_per_img=300),
    use_sigmoid=use_sigmoid)

# 4. weights
weights = dict(filepath='your/weight/file/path')

# 5. show
class_names = ('face', )
