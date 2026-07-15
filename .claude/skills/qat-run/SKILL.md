---
name: qat-run
description: 跑 QAT 训练与 demo(resnet50/minimum/multi_stage/reuse_conv/test_clamp)。统一 utils API,torch 2.6/2.10 同一份代码。要训练、跑示例、复现结果时用这个。
---

# qat-run:跑 QAT 训练与 demo

命令以仓库根目录为工作目录;本地编辑端操作时包一层 `ssh qat-dev '...'`。
先 `nvidia-smi` 挑空卡(4×A100-80G,常与他人共用):

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<空卡> \
  /home/heqi/miniforge3/envs/torch2.10/bin/python -u <script> [args] > /tmp/qax_<名>.log 2>&1
```

- 对照实验换 `/home/heqi/miniforge3/envs/torch2.6/bin/python`(同一份脚本双环境可跑);
- 长任务后台化(nohup 脱管 + 轮询日志);A100 上 resnet50 QAT 约 3 分钟/epoch。

## 入口速查

| 入口 | 用途/要点 |
|------|-----------|
| `resnet50/train.py --data fake\|cifar10 --config ./resnet50/config*.json --steps N --seed 42 [--eval-size N]` | 主训练。fake=结构验证零下载;cifar10=真数据(dataset/cifar10)。配置名决定产物后缀(config_4w4f→`_4w4f`) |
| `resnet50/test.py` / `resnet50/cross_export.py --checkpoint <pth>` | checkpoint 评测(torch+ORT 双路)/交叉导出 |
| `minimum/minimum_demo.py`、`minimum/yolov5_demo.py` | 最小示例(conv-bn-relu / 多输入 grid_sample) |
| `multi_stage/multi_stage_demo.py`、`multi_stage_contrast_demo.py` | 切子图分段推理(依赖 `resnet50/checkpoint/last_checkpoint_2_10.pth`,cifar 10 类) |
| `reuse_conv/train.py→test.py`、`train_resnet.py→test_resnet.py` | 复用 BN 场景(test 首跑生成 fixture,二跑逐 bit 回归) |
| `test_clamp/*_demo_axquant.py` | clamp/relu6 数值对照(CPU 即可) |

量化配置(resnet50/):`config.json`=U8/S8;`config_4w4f_all`=全局 4bit;
`config_4w4f`/`config_16f`/`config_fp32`=U4/U16/FP32 混合 regional。

## 动态 shape(torch 2.10)

`capture` 的 `dynamic_shapes` 为 torch 原生语义透传;整除性 guard 用派生维,
报 ConstraintViolation 时**照抄报错里的 Suggested fixes**:

```python
from torch.export import Dim
_h = Dim("_h", min=2, max=32)   # resnet50 下采样 32 倍 → H=32*_h
gm = capture(m, ex, dynamic_shapes=({0: Dim("batch", min=1, max=256), 2: 32*_h, 3: 32*_w},))
```

## 防坑清单(全部实战踩过)

1. **别用无后缀 config 快跑覆盖共享 checkpoint**:`config.json` 产物无后缀,
   会覆盖 `last_checkpoint_2_10.pth`(multi_stage 依赖 cifar 10 类版);
   快跑结构验证用带后缀配置(如 config_4w4f_all);
2. **共享产物的 demo 不要跨环境并行**(reuse_conv 的 tmp_ax.pth/fixture 互踩);
3. 验收命令别用 `grep -cE FAIL` 收尾——0 匹配返回码 1,任务被误报失败;
4. fake 数据时 `fake_train_size`(默认 1024)须 > steps×batch;
5. 跑完大批实验清理可再生产物(见 qat-env 第 3 节,NFS 配额易爆)。
