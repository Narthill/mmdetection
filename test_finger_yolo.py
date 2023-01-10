from pylibdmtx import pylibdmtx
from mmdet.apis import init_detector, inference_detector
import os
import time
import cv2


# config_file = 'configs/aoi/mirror_surface/faster_rcnn_r50_fpn_2x_coco_mirror_surface.py'
config_file = 'configs/aoi/connector/yolox_s_8x8_300e_coco_connector.py'
# 从 model zoo 下载 checkpoint 并放在 `checkpoints/` 文件下
# 网址为: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
# checkpoint_file = 'work_dirs/faster_rcnn_r50_fpn_2x_coco_mirror_surface/latest.pth'
checkpoint_file = 'work_dirs/yolox_s_8x8_300e_coco_connector/best_bbox_mAP_epoch_295.pth'
device = 'cuda'
# 初始化检测器
model = init_detector(config_file, checkpoint_file, device=device)
# 推理演示图像
# test_dir=r'/home/E/data/aoi/Connector/fail1'
# test_dir=r'/home/E/data/aoi/Connector/fail2'
# test_dir=r'/home/E/data/aoi/Connector/fail3'
test_dir=r'/home/E/data/aoi/Connector/connector_detection_ds/images'

test_filelist=os.listdir(test_dir+"/")
time_start=time.time()
for item in test_filelist:
    time_start=time.time()
    res=inference_detector(model, test_dir+"/"+item)
    
    time_then=time.time()
    print('inference cost',time_then-time_start,'s')
    src=cv2.imread(test_dir+"/"+item)
    dst=[]
    for index,finger in enumerate(res[0]):
        print(finger)
        if finger[4]>0.5:
            dst=cv2.rectangle(src,(int(finger[0]),int(finger[1])),(int(finger[2]),int(finger[3])),(0,255,0),5)
    cv2.imwrite(r'/home/E/data/aoi/Connector/connector_detection_ds/test_res/'+item+"_.jpg",dst)
    # model.show_result(test_dir+"/"+item,res,out_file=test_dir+"_yolox_then/"+item)
    # src=cv2.imread(test_dir+"/"+item)
    # barcode_img=src[int(res[0][0][1]-10):int(res[0][0][3]+10),int(res[0][0][0]-10):int(res[0][0][2]+10)]
    # cv2.imwrite(r'/home/D/Item/datasheet/coffe_class/test/yolox/'+dir_name+"_yolox_then/"+item,barcode_img)
    # model.show_result(test_dir+"/"+item,res,score_thr=0.7,out_file="resImg/yolox_s_8x8_300e_coco_connector/"+item)


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