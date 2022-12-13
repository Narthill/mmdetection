from mmdet.apis import init_detector, inference_detector
import os
import time
import cv2
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

# config_file = 'configs/aoi/mirror_surface/faster_rcnn_r50_fpn_2x_coco_mirror_surface.py'
# # 从 model zoo 下载 checkpoint 并放在 `checkpoints/` 文件下
# # 网址为: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
# checkpoint_file = 'work_dirs/faster_rcnn_r50_fpn_2x_coco_mirror_surface/latest.pth'
# device = 'cuda'
# # 初始化检测器
# model = init_detector(config_file, checkpoint_file, device=device)
# # 推理演示图像
# test_dir=r'/home/E/data/aoi/2022-10-26/NG/镜面/img/'
# test_filelist=os.listdir(test_dir)
# for item in test_filelist:
#     res=inference_detector(model, test_dir+item)
#     model.show_result(test_dir+item,res,out_file="resImg/test_res/"+item)


# 测试
# python tools/test.py configs/aoi/mirror_surface/faster_rcnn_r50_fpn_1x_coco_mirror_surface.py work_dirs/faster_rcnn_r50_fpn_1x_coco_mirror_surface/latest.pth --eval bbox --eval-options 'iou_thrs=[0.5]' --show-dir resImg/mirror_surface_faster_rcnn




config_file = 'configs/coffe_class/fasterrcnn/faster_rcnn_r50_fpn_2x_coco_coffe_class.py'
# 从 model zoo 下载 checkpoint 并放在 `checkpoints/` 文件下
# 网址为: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = 'work_dirs/faster_rcnn_r50_fpn_2x_coco_coffe_class/latest.pth'
device = 'cuda'
# 初始化检测器
model = init_detector(config_file, checkpoint_file, device=device)
# 推理演示图像
test_dir1=r'/home/D/Item/datasheet/coffe_class/test/fast close'
test_dir2=r'/home/D/Item/datasheet/coffe_class/test/lighting 50lux'
test_dir3=r'/home/D/Item/datasheet/coffe_class/test/lighting 700lux'
test_dir4=r'/home/D/Item/datasheet/coffe_class/test/lighting 2000lux'
test_dir5=r'/home/D/Item/datasheet/coffe_class/test/SKU recongnition'

test_dir_list=[test_dir1,test_dir2,test_dir3,test_dir4,test_dir5]

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

for test_dir in test_dir_list:
 
    test_filelist=os.listdir(test_dir+"/")
    count=0
    for item in test_filelist:
        time_start=time.time()
        res=inference_detector(model, test_dir+"/"+item)
        time_then=time.time()
        model.show_result(test_dir+"/"+item,res,score_thr=0.6,text_color=(0,255,0),out_file=test_dir+"_then/"+item)


#     flag = 0
#     for findit in res:
#        flag = flag+1
#        if (len(findit) > 0):
#         if (findit[0][4] > 0.6):
#             print(flag)
#             count = count+1
#             # print(findit[0][4])
# print(count)

    # print('inference cost',time_then-time_start,'s')
    # src=cv2.imread(test_dir+item)
    # barcode_img=src[int(res[0][0][1]-10):int(res[0][0][3]+10),int(res[0][0][0]-10):int(res[0][0][2]+10)]
    # cv2.imwrite("resImg/faster_rcnn_coffe_class/"+item,barcode_img)
    # time_end=time.time()
    # print('show cost',time_end-time_then,'s')