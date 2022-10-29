_base_ = [
    '../../_base_/models/ssd300.py', '../../_base_/datasets/coco_detection.py',
    '../../_base_/schedules/schedule_2x.py', '../../_base_/default_runtime.py'
]
# ssd因为要修改coco.py所以要重新编译mmdet

# dataset settings
dataset_type = 'CocoDataset'
data_root = '/home/D/Item/datasheet/ccm_aoi/mirror_surface/'

log_config = dict( 
    interval=10, 
    hooks=[ 
        dict(type='TextLoggerHook'), 
        dict(type='TensorboardLoggerHook') 
    ]) 

model = dict(
    bbox_head=dict(num_classes=3))
classes = ('huashang','sunshang','zangwu')

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(300, 300), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(300, 300),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
# 可能数据集划分有问题
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=3,
    train=dict(
        _delete_=True,
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'train.json',
            img_prefix=data_root + 'Image/',
            pipeline=train_pipeline)),
    val=dict(

            type=dataset_type,
            ann_file=data_root + 'valid.json',
            img_prefix=data_root + 'Image/',
            pipeline=test_pipeline),
    test=dict(
       
            type=dataset_type,
            ann_file=data_root + 'valid.json',
            img_prefix=data_root + 'Image/',
            pipeline=test_pipeline),
)
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=5e-2)
optimizer_config = dict(_delete_=True)
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='CheckInvalidLossHook', interval=50, priority='VERY_LOW')
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)
