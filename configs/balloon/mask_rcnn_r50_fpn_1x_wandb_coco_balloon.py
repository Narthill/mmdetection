# 这个新的配置文件继承自一个原始配置文件，只需要突出必要的修改部分即可
_base_ = '../mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'

# 我们需要对头中的类别数量进行修改来匹配数据集的标注
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

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
classes = ('balloon',)
data = dict(
    train=dict(
        img_prefix='/home/D/Item/datasheet/balloon_dataset/balloon/train/',
        classes=classes,
        ann_file='/home/D/Item/datasheet/balloon_dataset/balloon/train/train.json'),
    val=dict(
        img_prefix='/home/D/Item/datasheet/balloon_dataset/balloon/val/',
        classes=classes,
        ann_file='/home/D/Item/datasheet/balloon_dataset/balloon/val/val.json'),
    test=dict(
        img_prefix='/home/D/Item/datasheet/balloon_dataset/balloon/val/',
        classes=classes,
        ann_file='/home/D/Item/datasheet/balloon_dataset/balloon/val/val.json'))

# 我们可以使用预训练的 Mask R-CNN 来获取更好的性能
# load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'



 
    
