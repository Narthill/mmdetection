# 这个新的配置文件继承自一个原始配置文件，只需要突出必要的修改部分即可
_base_ = '../../retinanet/retinanet_r18_fpn_1x8_1x_coco.py'

log_config = dict( 
    interval=10, 
    hooks=[ 
        dict(type='TextLoggerHook'), 
        dict(type='TensorboardLoggerHook') 
    ]) 

lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)

# 我们需要对头中的类别数量进行修改来匹配数据集的标注
model = dict(
    bbox_head=dict(num_classes=16))


# 修改数据集相关设置
dataset_type = 'CocoDataset'
classes = ('5000080712',
           '5000082386',
           '5000085346',
           '5000085589',
           '5000086313',
           '5000086356',
           '5000086365',
           '5000086458',
           '5000197766',
           '5000340966',
           '5000351940',
           '5000358406',
           '5000360012',
           '9221080',
           '9221191',
           '9221406')
data = dict(
    train=dict(
        img_prefix='/home/D/Item/datasheet/coffe_class/Image/',
        classes=classes,
        ann_file='/home/D/Item/datasheet/coffe_class/train.json'),
    val=dict(
        img_prefix='/home/D/Item/datasheet/coffe_class/Image/',
        classes=classes,
        ann_file='/home/D/Item/datasheet/coffe_class/train.json'),
    test=dict(
        img_prefix='/home/D/Item/datasheet/coffe_class/Image/',
        classes=classes,
        ann_file='/home/D/Item/datasheet/coffe_class/train.json'))

# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
# 我们可以使用预训练的 Mask R-CNN 来获取更好的性能
load_from = 'checkpoints/retinanet_r18_fpn_1x8_1x_coco_20220407_171255-4ea310d7.pth'
# 使用方法
# python tools/test.py configs/aluminum/faster_rcnn_r50_fpn_1x_coco_aluminum.py work_dirs/faster_rcnn_r50_fpn_1x_coco_aluminum/latest.pth --eval bbox --eval-options 'iou_thrs=[0.5]'
# 可视化