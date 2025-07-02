切子图多阶段推理方法

### 训练
运行
python3 -m resnet50.train
确保 ./resnet50/checkpoint/last_checkpoint.pth 存在

### 推理
运行
python3 -m multi_stage.multi_stage_demo
