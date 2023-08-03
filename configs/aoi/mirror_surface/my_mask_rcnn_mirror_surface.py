_base_ = '../../mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
custom_imports = dict(imports=['mmdet.datasets.pipelines.ImageCrop'], allow_failed_imports=False)
model = dict(
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    rpn_head=dict(
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        bbox_roi_extractor=dict(
            roi_layer=dict(
                type='RoIAlign',
                output_size=7,
                sampling_ratio=2,
                aligned=False)),
        bbox_head=dict(
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        mask_roi_extractor=dict(
            roi_layer=dict(
                type='RoIAlign',
                output_size=14,
                sampling_ratio=2,
                aligned=False))))
# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='ImageCrop', crop_size=(1300, 1100)),
    dict(type='Resize', img_scale=(1300, 1100), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2448, 2048),
        flip=False,
        transforms=[
            # dict(type='ImageCrop', crop_size=(1300, 1100)),
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


# lr_config = dict(step=[28, 34])
# runner = dict(type='EpochBasedRunner', max_epochs=36)

# 我们需要对头中的类别数量进行修改来匹配数据集的标注
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=2),
        mask_head=dict(num_classes=2)))

log_config = dict( 
    interval=10, 
    hooks=[ 
        dict(type='TextLoggerHook'), 
        dict(type='TensorboardLoggerHook') 
    ]) 

dataset_type = 'CocoDataset'
classes = ('zangwu','huashang')
data = dict(
    train=dict(
        pipeline=train_pipeline,
        img_prefix='/home/D/Item/datasheet/ccm_aoi/mirror_surface_mask/output_train_coco/',
        classes=classes,
        ann_file='/home/D/Item/datasheet/ccm_aoi/mirror_surface_mask/output_train_coco/annotations.json'),
    val=dict(
        pipeline=test_pipeline,
        img_prefix='/home/D/Item/datasheet/ccm_aoi/mirror_surface_mask/output_train_coco/',
        classes=classes,
        ann_file='/home/D/Item/datasheet/ccm_aoi/mirror_surface_mask/output_train_coco/annotations.json'),
    test=dict(
        pipeline=test_pipeline,
        img_prefix='/home/D/Item/datasheet/ccm_aoi/mirror_surface_mask/output_train_coco/',
        classes=classes,
        ann_file='/home/D/Item/datasheet/ccm_aoi/mirror_surface_mask/output_train_coco/annotations.json'))

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.38__segm_mAP-0.344_20200504_231812-0ebd1859.pth'