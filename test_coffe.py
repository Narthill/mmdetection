from mmdet.apis import init_detector, inference_detector
import os
import time

# config_file = 'configs/aoi/mirror_surface/faster_rcnn_r50_fpn_2x_coco_mirror_surface.py'
config_file = 'configs/coffe_barcode/ssd/ssd300_coco_coffe_barcode.py'
# 从 model zoo 下载 checkpoint 并放在 `checkpoints/` 文件下
# 网址为: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
# checkpoint_file = 'work_dirs/faster_rcnn_r50_fpn_2x_coco_mirror_surface/latest.pth'
checkpoint_file = 'work_dirs/ssd300_coco_coffe_barcode/latest.pth'
device = 'cuda'
# 初始化检测器
model = init_detector(config_file, checkpoint_file, device=device)
# 推理演示图像
test_dir=r'/home/D/Item/datasheet/coffe/Test/'
test_filelist=os.listdir(test_dir)
# for item in test_filelist:
#     time_start=time.time()
#     res=inference_detector(model, test_dir+item)
#     time_then=time.time()
#     print('inference cost',time_then-time_start,'s')
#     model.show_result(test_dir+item,res,score_thr=0.5,out_file="resImg/ssd300_coffe_bacode_test/"+item)

# for item in test_filelist:
item=test_filelist[0]
time_start=time.time()
res=inference_detector(model, test_dir+item)
time_then=time.time()
print('inference cost',time_then-time_start,'s')
print(res[0])