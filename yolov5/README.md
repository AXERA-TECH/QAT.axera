# 准备工作
1. 准备好yolov5的代码库 clone from https://github.com/ultralytics/yolov5/tree/master
2. 复制QAT.axera下utils目录中的所有文件到yolov5/utils中
3. 复制QAT.axera下yolov5中的train.py，val.py和detect.py到准备好的yolov5代码库中覆盖原始文件
4. 注掉这一行 https://github.com/ultralytics/yolov5/blob/master/models/common.py#L337

# 训练
目前支持8w8f和4w8f两种训练
## 8w8f
执行`python train.py --data coco.yaml --epochs 1 --weights yolov5s.pt --cfg yolov5s.yaml  --batch-size 1 --qat_weight 8w8f`

## 4w8f
执行`python train.py --data coco.yaml --epochs 1 --weights yolov5s.pt --cfg yolov5s.yaml  --batch-size 1 --qat_weight 4w8f`
