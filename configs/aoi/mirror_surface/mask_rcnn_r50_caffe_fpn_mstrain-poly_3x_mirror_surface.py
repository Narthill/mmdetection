_base_ = '../../mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'
# learning policy
lr_config = dict(step=[28, 34])
runner = dict(type='EpochBasedRunner', max_epochs=36)

# 我们需要对头中的类别数量进行修改来匹配数据集的标注
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=2),
        mask_head=dict(num_classes=2)))

# log_config = dict(
#     interval=50,
#     hooks=[
#         dict(type='TextLoggerHook'),
#         dict(type='MMDetWandbHook',
#              init_kwargs={
#                 'project': 'mmdetection',
#                 'group': 'maskrcnn-r50-fpn-1x-coco'
#              },
#              interval=50,
#              log_checkpoint=False,
#              log_checkpoint_metadata=True,
#              num_eval_images=13)
#         ])
log_config = dict( 
    interval=10, 
    hooks=[ 
        dict(type='TextLoggerHook'), 
        dict(type='TensorboardLoggerHook') 
    ]) 

# 修改数据集相关设置
dataset_type = 'CocoDataset'
classes = ('zangwu','huashang')
data = dict(
    train=dict(
        img_prefix='/home/D/Item/datasheet/ccm_aoi/mirror_surface_mask/output_train_coco/',
        classes=classes,
        ann_file='/home/D/Item/datasheet/ccm_aoi/mirror_surface_mask/output_train_coco/annotations.json'),
    val=dict(
        img_prefix='/home/D/Item/datasheet/ccm_aoi/mirror_surface_mask/output_train_coco/',
        classes=classes,
        ann_file='/home/D/Item/datasheet/ccm_aoi/mirror_surface_mask/output_train_coco/annotations.json'),
    test=dict(
        img_prefix='/home/D/Item/datasheet/ccm_aoi/mirror_surface_mask/output_train_coco/',
        classes=classes,
        ann_file='/home/D/Item/datasheet/ccm_aoi/mirror_surface_mask/output_train_coco/annotations.json'))

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)

# 我们可以使用预训练的 Mask R-CNN 来获取更好的性能
load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
