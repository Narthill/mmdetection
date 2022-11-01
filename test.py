from mmdet.apis import init_detector, inference_detector
import os
# config_file = 'configs/aluminum/faster_rcnn_r50_fpn_1x_coco_aluminum.py'
# # 从 model zoo 下载 checkpoint 并放在 `checkpoints/` 文件下
# # 网址为: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
# checkpoint_file = 'work_dirs/faster_rcnn_r50_fpn_1x_coco_aluminum/latest.pth'
# device = 'cpu'
# # 初始化检测器
# model = init_detector(config_file, checkpoint_file, device=device)
# # 推理演示图像
# res=inference_detector(model, '/home/D/Item/datasheet/aluminum/images/192.jpg')
# model.show_result('/home/D/Item/datasheet/aluminum/images/192.jpg',res,out_file="out_aluminum.jpg")


# from mmdet.apis import init_detector, inference_detectors

# config_file = 'configs/balloon/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_balloon.py'
# # 从 model zoo 下载 checkpoint 并放在 `checkpoints/` 文件下
# # 网址为: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
# checkpoint_file = 'work_dirs/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_balloon/latest.pth'
# device = 'cpu'
# # 初始化检测器
# model = init_detector(config_file, checkpoint_file, device=device)
# # 推理演示图像
# res=inference_detector(model, '/home/D/Item/datasheet/balloon_dataset/balloon/val/16335852991_f55de7958d_k.jpg')
# model.show_result('/home/D/Item/datasheet/balloon_dataset/balloon/val/16335852991_f55de7958d_k.jpg',res,out_file="out.jpg")

# 测试
# python tools/test.py configs/aluminum/faster_rcnn_r50_fpn_1x_coco_aluminum.py work_dirs/faster_rcnn_r50_fpn_1x_coco_aluminum/latest.pth --eval bbox --eval-options 'iou_thrs=[0.5]'

# OrderedDict([('bbox_mAP', 0.988), ('bbox_mAP_50', -1.0), ('bbox_mAP_75', -1.0), 
#              ('bbox_mAP_s', 0.955), ('bbox_mAP_m', 0.985), ('bbox_mAP_l', 1.0), 
#              ('bbox_mAP_copypaste', '0.988 -1.000 -1.000 0.955 0.985 1.000')])

config_file = 'configs/aoi/mirror_surface/faster_rcnn_r50_fpn_2x_coco_mirror_surface.py'
# 从 model zoo 下载 checkpoint 并放在 `checkpoints/` 文件下
# 网址为: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = 'work_dirs/faster_rcnn_r50_fpn_2x_coco_mirror_surface/latest.pth'
device = 'cuda'
# 初始化检测器
model = init_detector(config_file, checkpoint_file, device=device)
# 推理演示图像
test_dir=r'/home/E/data/aoi/2022-10-26/NG/镜面/img/'
test_filelist=os.listdir(test_dir)
for item in test_filelist:
    res=inference_detector(model, test_dir+item)
    model.show_result(test_dir+item,res,out_file="resImg/test_res/"+item)


# 测试
# python tools/test.py configs/aoi/mirror_surface/faster_rcnn_r50_fpn_1x_coco_mirror_surface.py work_dirs/faster_rcnn_r50_fpn_1x_coco_mirror_surface/latest.pth --eval bbox --eval-options 'iou_thrs=[0.5]' --show-dir resImg/mirror_surface_faster_rcnn



