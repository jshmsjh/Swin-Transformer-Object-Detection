from mmdet.apis import init_detector, inference_detector
import mmcv

# Specify the path to model config and checkpoint file
#网络配置文件
config_file = ‘configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py’
#预训练网络文件   http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth 

checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# build the model from a config file and a checkpoint file
#生产模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = ‘test.jpg’    # 待检测的照片
result = inference_detector(model, img)      #检测命令
# save the visualization results to image files
model.show_result(img, result, out_file=‘result.jpg’)   #输出结果图片
