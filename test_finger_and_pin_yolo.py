from mmdet.apis import init_detector, inference_detector
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 极大值抑制
def nms(dets, thresh):
    """Pure Python NMS baseline."""
    # x1、y1、x2、y2、以及score赋值
    x1 = dets[:, 0]  # xmin
    y1 = dets[:, 1]  # ymin
    x2 = dets[:, 2]  # xmax
    y2 = dets[:, 3]  # ymax
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # argsort()返回数组值从小到大的索引值
    order = scores.argsort()[::-1]    
    keep = []
    while order.size > 0:  # 还有数据
        # print("order:",order)
        i = order[0]
        keep.append(i)
        if order.size==1:break
        # 计算当前概率最大矩形框与其他矩形框的相交框的坐标
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算相交框的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
        IOU = inter / (areas[i] + areas[order[1:]] - inter)
     
        # 找到重叠度不高于阈值的矩形框索引
        # print("IOU:",IOU)
        left_index = (np.where(IOU <= thresh))[0]
        
        # 将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[left_index + 1]
        
    return keep

# 获取最小外接矩形
def get_min_area_rect(pts):
    cnt=np.array([pts]).transpose(1,0,2)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    return np.int0(box)


# 获取突包
def get_convex_Hull(pts):
    cnt=np.array([pts]).transpose(1,0,2)
    hull = cv2.convexHull(cnt)
    return hull

# config_file = 'configs/aoi/mirror_surface/faster_rcnn_r50_fpn_2x_coco_mirror_surface.py'
config_file = 'configs/aoi/connector/yolox_s_8x8_300e_coco_connector_with_pin.py'
# 从 model zoo 下载 checkpoint 并放在 `checkpoints/` 文件下
# 网址为: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
# checkpoint_file = 'work_dirs/faster_rcnn_r50_fpn_2x_coco_mirror_surface/latest.pth'
checkpoint_file = 'work_dirs/yolox_s_8x8_300e_coco_connector_with_pin/latest.pth'
device = 'cuda'
# 初始化检测器
model = init_detector(config_file, checkpoint_file, device=device)
# 推理演示图像
# test_dir=r'/home/E/data/aoi/Connector/fail1'
# test_dir=r'/home/E/data/aoi/Connector/fail2'
# test_dir=r'/home/E/data/aoi/Connector/fail3'
test_dir=r'/home/E/data/aoi/Connector/1_src_ds/all'
# test_dir=r'/home/E/data/aoi/Connector/connector_detection_ds/test'

test_filelist=os.listdir(test_dir+"/")

time_start=time.time()
for file_index,item in enumerate(test_filelist):
    # time_start=time.time()
    res=inference_detector(model, test_dir+"/"+item)
    
    # time_then=time.time()
    # print('inference cost',time_then-time_start,'s')
    src=cv2.imread(test_dir+"/"+item)
    dst=[]
    
    indexlist_finger=nms(res[0],0.05)
    indexlist_pin=(nms(res[1],0.05))
    
    pts_finger=[]
    pts_pin=[]
    for index,finger in enumerate(res[0]):
        if index in indexlist_finger:
            if finger[4]>0.5 and len(pts_finger)<24*2:
                pt1=(int(finger[0]),int(finger[1]))
                pt2=(int(finger[2]),int(finger[3]))
                # finger_center=(int((pt1[0]+pt2[0])/2),int((pt1[1]+pt2[1])/2))
                pts_finger.append(pt1)
                pts_finger.append(pt2)
                # pts_finger.append(finger_center)
                dst=cv2.rectangle(src,pt1,pt2,(0,255,0),2) 
                # dst=cv2.circle(dst,finger_center,2,(0,255,0),-1)
                cv2.putText(dst, str(round(finger[4],2)), (int(finger[0]),int(finger[1])), cv2.FONT_HERSHEY_PLAIN, 1.5, (127, 127, 255), 1)
    
    for index,pin in enumerate(res[1]):
        if index in indexlist_pin:
            if pin[4]>0.5 and len(pts_pin)<24*2:
                pt1=(int(pin[0]),int(pin[1]))
                pt2=(int(pin[2]),int(pin[3]))
                # pin_center=(int((pt1[0]+pt2[0])/2),int((pt1[1]+pt2[1])/2))
                pts_pin.append(pt1)
                pts_pin.append(pt2)
                # pts_pin.append(pin_center)
                dst=cv2.rectangle(dst,pt1,pt2,(255,127,0),2)
                # dst=cv2.circle(dst,pin_center,2,(255,127,0),-1)
                
    f_center = np.mean(np.array(pts_finger),axis=0)
    p_center = np.mean(np.array(pts_pin),axis=0)
    cv2.circle(dst,(int(f_center[0]),int(f_center[1])),2,(0,255,0),-1)
    cv2.circle(dst,(int(p_center[0]),int(p_center[1])),2,(255,127,0),-1)
    disx=(abs(f_center[0]-p_center[0]))**2
    disy=(abs(f_center[1]-p_center[1]))**2
    # disx=f_center[0]-p_center[0]
    # disy=f_center[1]-p_center[1]
    plt.scatter(disx, disy,s=10, marker='o',color = 'blue')
    
    # 找最小外接矩形
    box_pin=get_min_area_rect(pts_pin)
    cv2.drawContours(dst, [box_pin], 0, (0, 255, 255), 2)
    p_center2 =np.mean(np.array(box_pin),axis=0)
    cv2.circle(dst,(int(p_center2[0]),int(p_center2[1])),2,(0,255,255),-1)
    
    box_finger=get_min_area_rect(pts_finger)
    cv2.drawContours(dst, [box_finger], 0, (0, 0, 255), 2)
    f_center2 = np.mean(np.array(box_finger),axis=0)
    cv2.circle(dst,(int(f_center2[0]),int(f_center2[1])),2,(0,0,255),-1)
    
    disx2=(abs(f_center2[0]-p_center2[0]))**2
    disy2=(abs(f_center2[1]-p_center2[1]))**2
    plt.scatter(disx2, disy2,s=10, marker='o',color = 'red')
    
    if disx>300:
      print("disx1:"+item)
    
    if disx2>300:
      print("disx2:"+item)
    
    cv2.imwrite(r'/home/E/data/aoi/Connector/1_src_ds/find_finger_and_pin/'+item+"_.jpg",dst)
    # cv2.imwrite(r'/home/E/data/aoi/Connector/connector_detection_ds/test_res/'+item+"_.jpg",dst)
    

plt.show()
