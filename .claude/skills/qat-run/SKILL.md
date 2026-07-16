---
name: qat-run
description: 跑 QAT 训练与 demo(resnet50/minimum/multi_stage/reuse_conv)。统一 utils API,torch 2.6/2.10 同一份代码。要训练、跑示例、复现结果时用这个。
---

# qat-run:跑 QAT 训练与 demo

前置:工作目录为仓库根目录;当前 Python 环境满足 `requirements_2_10.txt`
(torch 2.10)或 `requirements.txt`(torch 2.6)——版本路由自动完成,同一份
脚本双版本可跑;依赖清单见 requirements_2_10.txt / requirements.txt。

```bash
PYTHONPATH=. [CUDA_VISIBLE_DEVICES=<卡号>] python -u <script> [args]
```

多卡机器先用 `nvidia-smi` 挑空闲卡;长任务建议 nohup 后台化并重定向日志。

## 入口速查

| 入口 | 用途/要点 |
|------|-----------|
| `resnet50/train.py --data fake\|cifar10 --config ./resnet50/config*.json --steps N --seed 42 [--eval-size N]` | 主训练。fake=结构验证零下载;cifar10=真数据(需 `dataset/cifar10/cifar-10-python.tar.gz`)。配置名决定产物后缀(config_4w4f→`_4w4f`) |
| `resnet50/test.py` / `resnet50/cross_export.py --checkpoint <pth>` | checkpoint 评测(torch+ORT 双路)/交叉导出 |
| `minimum/minimum_demo.py`、`minimum/yolov5_demo.py` | 最小示例(conv-bn-relu / 多输入 grid_sample) |
| `multi_stage/multi_stage_demo.py`、`multi_stage_contrast_demo.py` | 切子图分段推理(依赖 `resnet50/checkpoint/last_checkpoint_2_10.pth`,cifar 10 类头) |
| `reuse_conv/train.py→test.py`、`train_resnet.py→test_resnet.py` | 复用 BN 场景(test 首跑生成 fixture,二跑逐 bit 回归) |

量化配置(`resnet50/`):`config.json`=U8/S8;`config_4w4f_all`=全局 4bit;
`config_4w4f`/`config_16f`/`config_fp32`=U4/U16/FP32 混合 regional。

## 动态 shape(torch 2.10)

`capture` 的 `dynamic_shapes` 为 torch.export 原生语义透传;下采样网络的
H/W 整除性 guard 用派生维表达,报 ConstraintViolation 时**照抄报错里的
Suggested fixes**:

```python
from torch.export import Dim
_h = Dim("_h", min=2, max=32)   # resnet50 下采样 32 倍 → H=32*_h
gm = capture(m, ex, dynamic_shapes=({0: Dim("batch", min=1, max=256), 2: 32*_h, 3: 32*_w},))
```

## 防坑清单

1. **别用无后缀 config 快跑覆盖共享 checkpoint**:`config.json` 的产物文件名
   无后缀,会覆盖 `last_checkpoint_2_10.pth`(multi_stage 依赖的 cifar 10 类版);
   做结构验证请用带后缀配置(如 config_4w4f_all);
2. **共享产物的 demo 不要并行跑多份**(reuse_conv 的 checkpoint/fixture 会互踩);
3. fake 数据时 `fake_train_size`(默认 1024)须 > steps×batch,否则数据提前耗尽;
4. 产物(float 参考 onnx、epoch checkpoint)体积可观,批量实验后及时清理
   可再生文件(float 参考 onnx、epoch/配置后缀 checkpoint、cross/opt onnx 等,批量实验后及时清理)。
