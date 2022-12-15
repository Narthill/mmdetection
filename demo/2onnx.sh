python tools/deployment/pytorch2onnx.py \
    configs/coffe_class/retinanet/retinanet_r50_fpn_2x_coco_coffe_class.py \
    checkpoints/coffe_class.pth \
    --output-file checkpoints/coffe_class_1214.onnx \
    --input-img out_3.jpg \
    --test-img out_5.jpg \
    --shape 480 640 \
    --show \
    --verify \
    --dynamic-export \
    --cfg-options \
      model.test_cfg.deploy_nms_pre=-1 

       x: array([[2.396782e+02, 1.116820e+02, 7.077327e+02, 5.605599e+02,
        1.215236e-01],
       [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,...
 y: array([[2.396782e+02, 1.116821e+02, 7.077327e+02, 5.605599e+02,
        1.215239e-01]], dtype=float32)