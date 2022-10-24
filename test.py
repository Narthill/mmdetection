from mmdet.apis import init_detector, inference_detector

config_file = 'configs/aluminum/faster_rcnn_r50_fpn_1x_coco_aluminum.py'
# 从 model zoo 下载 checkpoint 并放在 `checkpoints/` 文件下
# 网址为: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = 'work_dirs/faster_rcnn_r50_fpn_1x_coco_aluminum/latest.pth'
device = 'cpu'
# 初始化检测器
model = init_detector(config_file, checkpoint_file, device=device)
# 推理演示图像
res=inference_detector(model, '/home/D/Item/datasheet/aluminum/images/334.jpg')
model.show_result('/home/D/Item/datasheet/aluminum/images/334.jpg',res,out_file="out_aluminum.jpg")


# from mmdet.apis import init_detector, inference_detector

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
