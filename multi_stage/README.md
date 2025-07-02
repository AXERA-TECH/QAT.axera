
切子图多阶段推理方法

### 训练
运行
python3 -m resnet50.train
确保 ./resnet50/checkpoint/last_checkpoint.pth 存在

### 推理
运行
python3 -m multi_stage.multi_stage_demo

---
切子图方法与多个 forward 分别独立推理方法的比较

### 修改 quantizer
临时将权重量化由 per-channel 对称量化改为 per-tensor 对称量化，即
将 https://github.com/AXERA-TECH/QAT.axera/blob/main/utils/ax_quantizer.py#L144
从
weight_qscheme = torch.per_channel_symmetric
改为
weight_qscheme = torch.per_tensor_symmetric
否则无法复现错误分段推理方式。

通常来说错误分段推理方式，会由于不同段 weight 的 channel 数不同无法加载权重。但是也有可能由于巧合正好每个阶段对应 weight 的 chennel 都相同，也可以加载
由于 demo 使用的 ResNet50 不能加载 per-channel 量化参数，因此手改一下量化方法，来复现并证明错误分段推理方式的精度问题

### 训练
运行
python3 -m resnet50.train
确保 ./resnet50/checkpoint/last_checkpoint.pth 存在

### 推理
运行
python3 -m multi_stage.multi_stage_contrast_demo

### 预期结果
上述 demo会进行5次推理，分别是：

1. 原始完整浮点模型
2. multi stage 浮点模型
3. 完整量化模型
4. 由 multi stage 浮点模型每个 stage 分别独立加载参数的量化模型
5. 由完整量化模型切多个子图再进行分 stage 推理的量化模型

预期结果： 1 和 2 精度一致为高精度；3 和 5 精度一致接近高精度；4 精度明显劣化