# 这个新的配置文件继承自一个原始配置文件，只需要突出必要的修改部分即可
_base_ = '../../mask_rcnn/mask_rcnn_r101_fpn_2x_coco.py'

# # 我们需要对头中的类别数量进行修改来匹配数据集的标注
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
classes = ('lian_xi',)
data = dict(
    train=dict(
        img_prefix='/home/E/data/aoi/Connector/special/lianxi_coco_ds/output_train_coco/',
        classes=classes,
        ann_file='/home/E/data/aoi/Connector/special/lianxi_coco_ds/output_train_coco/annotations.json'),
    val=dict(
        img_prefix='/home/E/data/aoi/Connector/special/lianxi_coco_ds/output_train_coco/',
        classes=classes,
        ann_file='/home/E/data/aoi/Connector/special/lianxi_coco_ds/output_train_coco/annotations.json'),
    test=dict(
        img_prefix='/home/E/data/aoi/Connector/special/lianxi_coco_ds/output_train_coco/',
        classes=classes,
        ann_file='/home/E/data/aoi/Connector/special/lianxi_coco_ds/output_train_coco/annotations.json'))

# 一、weight decay（权值衰减）的使用既不是为了提高你所说的收敛精确度也不是为了提高收敛速度，
# 其最终目的是防止过拟合。在损失函数中，weight decay是放在正则项（regularization）前面的一个系数，
# 正则项一般指示模型的复杂度，所以weight decay的作用是调节模型复杂度对损失函数的影响，
# 若weight decay很大，则复杂的模型损失函数的值也就大。
# 二、momentum是梯度下降法中一种常用的加速技术。
# 链接：https://www.zhihu.com/question/24529483/answer/114711446

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

# 我们可以使用预训练的 Mask R-CNN 来获取更好的性能
load_from = 'work_dirs/mask_rcnn_r101_fpn_2x_coco_connector/latest.pth'