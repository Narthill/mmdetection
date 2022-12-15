import onnx
import time
import cv2
import mmcv
import onnxruntime as ort
import numpy as np

image_hwc=cv2.imread("out_5.jpg")

img = cv2.cvtColor(image_hwc, cv2.COLOR_BGR2RGB)
img = mmcv.imresize(img, (960, 704))

img = img/255.0
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
# [123.675, 116.28, 103.53]
# [58.395, 57.12, 57.375]

img[..., 0] -= mean[0]
img[..., 1] -= mean[1]
img[..., 2] -= mean[2]

img[..., 0] /= std[0]
img[..., 1] /= std[1]
img[..., 2] /= std[2]
# cv2.imshow("test",img)
# cv2.waitKey(0)
chw=np.transpose(img, (2,0,1))
print(chw.shape)
np.expand_dims(chw,axis=0)
nchw=chw[np.newaxis,:]
ort_session = ort.InferenceSession('checkpoints/coffe_class_1215.onnx',providers=['CUDAExecutionProvider'])
print('开始修复......')
outs = ort_session.run(
    None, 
    {
        "input":  nchw.astype(np.float32),
    }
)
start = time.time()
outs = ort_session.run(
    None, 
    {
        "input":  nchw.astype(np.float32),
    }
)
end = time.time()
print('修复结束耗时'+str(end-start)+"s")
print(outs[0][0][0])
