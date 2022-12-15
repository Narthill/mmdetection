from pylibdmtx import pylibdmtx
from mmdet.apis import init_detector, inference_detector
import os
import time
import cv2


# config_file = 'configs/aoi/mirror_surface/faster_rcnn_r50_fpn_2x_coco_mirror_surface.py'
config_file = 'configs/coffe_barcode/ssd/ssd300_coco_coffe_barcode.py'
# 从 model zoo 下载 checkpoint 并放在 `checkpoints/` 文件下
# 网址为: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
# checkpoint_file = 'work_dirs/faster_rcnn_r50_fpn_2x_coco_mirror_surface/latest.pth'
checkpoint_file = 'work_dirs/ssd300_coco_coffe_barcode/latest.pth'
device = 'cpu'
# 初始化检测器
model = init_detector(config_file, checkpoint_file, device=device)
# 推理演示图像
test_dir=r'/home/D/Item/datasheet/coffe/Image/'
test_filelist=os.listdir(test_dir)
time_start=time.time()
for item in test_filelist:
    time_start=time.time()
    res=inference_detector(model, test_dir+item)
#     model.show_result(test_dir+item,res,out_file="resImg/ssd300_coffe_bacode_simo_test_roi/"+item)
    time_then=time.time()
    print('inference cost',time_then-time_start,'s')
    src=cv2.imread(test_dir+item)
    barcode_img=src[int(res[0][0][1]-10):int(res[0][0][3]+10),int(res[0][0][0]-10):int(res[0][0][2]+10)]
    cv2.imwrite("resImg/ssd300_coffe_bacode_test_roi/"+item,barcode_img)
    # model.show_result(test_dir+item,res,score_thr=0.5,out_file="resImg/ssd300_coffe_bacode_simo_test_roi/"+item)


# for item in test_filelist:
# item=test_filelist[2]
# time_start=time.time()
# res=inference_detector(model, test_dir+item)
# time_then=time.time()
# print('inference cost',time_then-time_start,'s')
# # 生成200个预测，最上面是最大的，每个预测包含五个信息(左上角x,左上角y,右下角x,右下角y，权重)
# print(res[0][0])
# src=cv2.imread(test_dir+item,0)
# barcode_img=src[int(res[0][0][1]):int(res[0][0][3]+0.5),int(res[0][0][0]):int(res[0][0][2]+0.5)]
# all_barcode_info = pylibdmtx.decode(barcode_img, timeout=500, max_count=1)
# print(all_barcode_info)
# cv2.imshow("barcode_img",barcode_img)
# cv2.waitKey(0)