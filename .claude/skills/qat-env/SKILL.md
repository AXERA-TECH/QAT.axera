---
name: qat-env
description: QAT.axera 环境体检与修复——torch2.10/2.6 环境版本核对、清华源装包与 pip 挂死诊断、NFS 磁盘配额清理。环境异常、装包、跑不起来时先用这个。
---

# qat-env:环境体检与修复

命令以仓库根目录(`/home/heqi/project-qat/QAT.axera`)为工作目录;
若从本地编辑端(Mac 经 mac-mini 跳板)操作,包一层 `ssh qat-dev '...'`。

## 1. 版本体检(期望值与依据见 env.md)

```bash
/home/heqi/miniforge3/envs/torch2.10/bin/pip list | grep -iE "^(torch|torchao|onnx|onnxscript|onnx-ir|onnxruntime|onnxslim) "
```

期望:torch 2.10.0+cu126 / torchao 0.16.0+cu126 / onnx 1.19.1 /
**onnxscript 0.6.2**(修复 optimize 折叠权重 DQ)/
**onnx-ir 0.1.15(约束 <0.1.16,勿让 pip 连带升级)** / onnxruntime 1.23.2。
对照环境 torch2.6:onnx 1.17.0 / onnxscript 0.4.0。
版本路由为**严格判定**:仅 torch 2.6 与 2.10 受支持,其他版本 import 即告警。

## 2. 装包(必须清华源)

直连 PyPI 会**无限挂死**(进程零 socket 零磁盘 IO)。诊断:
`cat /proc/<pid>/io` 两次采样无变化 + `ss -tnp | grep pid=` 无连接 → 杀掉换源。
⚠️ `pkill -f "pip install"` 会自匹配杀掉调用方 shell,用字符类:`pkill -f "pip instal[l]"`。

```bash
pip install --no-input -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements_2_10.txt
```

## 3. 磁盘配额(NFS:df 显示卷有空闲但用户配额会先耗尽,git 报 No space left)

清理**可再生**产物:`resnet50/resnet50_float_*.onnx`(每个 130MB)、
`resnet50/checkpoint/checkpoint_*_0.pth` 与带配置后缀的 last_checkpoint、
cross/opt onnx、`reuse_conv/resnet50*.pth|*.onnx`、`env_check/out/`、`__pycache__`。
**必须保留**:`resnet50_pretrained_float.pth`、金标准 `resnet50_qat[_sim].onnx`、
`last_checkpoint_2_6.pth`/`last_checkpoint_2_10.pth`(cifar 等价性 + multi_stage 依赖)、
`*.eq50bak.pth`、`reuse_conv/input_ax.npy|gt_ax.npy`(回归锚点)、`dataset/`。

## 4. 快速功能冒烟

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<空卡> \
  /home/heqi/miniforge3/envs/torch2.10/bin/python -u env_check/gen_candidate_torch210.py
# 期望:fake-quant 插入数量 11,两个 onnx 导出成功;随后用 qat-check 体检产物
```
