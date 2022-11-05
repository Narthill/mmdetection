# 这个新的配置文件继承自一个原始配置文件，只需要突出必要的修改部分即可
_base_ = '../../retinanet/retinanet_r50_fpn_2x_coco.py'

log_config = dict( 
    interval=10, 
    hooks=[ 
        dict(type='TextLoggerHook'), 
        dict(type='TensorboardLoggerHook') 
    ]) 
# 我们需要对头中的类别数量进行修改来匹配数据集的标注
model = dict(
    bbox_head=dict(num_classes=2))
runner = dict(type='EpochBasedRunner', max_epochs=30)
# 修改数据集相关设置
dataset_type = 'CocoDataset'
classes = ('huashang','zangwu',)
data = dict(
    train=dict(
        img_prefix='/home/D/Item/datasheet/ccm_aoi/mirror_surface/Image/',
        classes=classes,
        ann_file='/home/D/Item/datasheet/ccm_aoi/mirror_surface/train.json'),
    val=dict(
        img_prefix='/home/D/Item/datasheet/ccm_aoi/mirror_surface/Image/',
        classes=classes,
        ann_file='/home/D/Item/datasheet/ccm_aoi/mirror_surface/valid.json'),
    test=dict(
        img_prefix='/home/D/Item/datasheet/ccm_aoi/mirror_surface/Image/',
        classes=classes,
        ann_file='/home/D/Item/datasheet/ccm_aoi/mirror_surface/valid.json'))

# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
# 我们可以使用预训练的 Mask R-CNN 来获取更好的性能
load_from = 'checkpoints/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth'
# 使用方法
# python tools/test.py configs/aluminum/faster_rcnn_r50_fpn_1x_coco_aluminum.py work_dirs/faster_rcnn_r50_fpn_1x_coco_aluminum/latest.pth --eval bbox --eval-options 'iou_thrs=[0.5]'
# 可视化