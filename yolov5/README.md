仅供参考导出，剩下的东西去 https://github.com/ultralytics/yolov5/tree/master 搞

另外要注掉这一行 https://github.com/ultralytics/yolov5/blob/master/models/common.py#L337

执行
`python train.py --data coco.yaml --epochs 1 --weights yolov5s.pt --cfg yolov5s.yaml  --batch-size 1`