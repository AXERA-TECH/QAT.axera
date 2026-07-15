---
name: qat-equivalence
description: torch 2.6 vs 2.10 双环境等价性对照——同种子训练锁步、sim 结构逐项对比、同 checkpoint 交叉导出的 qparam 逐 bit 对账。改了 utils/量化器/导出链后验证行为无回退时用这个。
---

# qat-equivalence:双环境等价性对照

**utils 合并实现的任何改动,双环境回归是必选项。** 历史基线数字见下,
偏离即回退信号。命令以仓库根目录为工作目录;本地编辑端包一层 `ssh qat-dev '...'`。

## 协议(四层证据,按需取用)

**① 训练行为锁步**:同一脚本、同 seed/数据/超参,两环境各跑一次(挑两张空卡,
GPU 错开可并行):

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<A> /home/heqi/miniforge3/envs/torch2.10/bin/python -u \
  resnet50/train.py --data cifar10 --steps 50 --seed 42 --eval-size 20 > /tmp/qax_eq_210.log 2>&1
# 另一侧换 torch2.6 的 python 与另一张卡
```

对账点:`[assert] fake-quant 插入数量` 两侧必须相等(config.json 为 128,
fp32 配置 124);逐 batch loss 逐点差 ≤0.05 无发散;历史基线:
50 步 eval 56.562/96.250(2.6) vs 53.750/93.281(2.10)(n=640 噪声内);
全量 1 epoch test top1 94.72 vs 94.97(Δ0.25pt 噪声级)。

**② sim 结构逐项对比**:

```bash
$P env_check/check_onnx_structure.py --model resnet50/resnet50_qat_2_6_sim.onnx --sim --save-profile /tmp/p26.json
$P env_check/check_onnx_structure.py --model resnet50/resnet50_qat_2_10_sim.onnx --sim --baseline /tmp/p26.json
# 期望 BASE-* 5 项全 PASS;全配置一键总表:env_check/cmp_sim_matrix.py
# (4w4f 混合配置 ✗ 属已知 2.6 老 bug 指纹:55/23 vs 8/5,见 qat-check)
```

**③ qparam 逐 bit 对账(排除训练随机性)**:同一 checkpoint 在两环境各自
convert+导出(`resnet50/cross_export.py --checkpoint resnet50/checkpoint/last_checkpoint_2_6.pth`),
Q/DQ 的 (scale, zero_point) 按值多重集合(md5)对账应 0 差异。
⚠️ 按拓扑位置对齐会出假差异(并行分支平序不同),必须按值匹配。
checkpoint 跨体系天然互通(1216 键 strict 直载,已验证)。

**④ 精度终验**(可选,~3 分钟/epoch):`--steps 1563 --epochs 1 --eval-size 400`
全量 1 epoch + 1 万张 test。

## 防坑

- 并行跑两环境时错开 GPU、**避开共享产物的 demo**(reuse_conv 的
  fixture/tmp_ax.pth 互踩;resnet50 产物带 `_2_6/_2_10` 环境标签天然隔离);
- 结构验证别动 `last_checkpoint_2_10.pth`(multi_stage 依赖),快跑用带后缀配置;
- "逐数字复现"是现实标准:重构验收时两环境 eval 应与历史逐位相同——
  只是"接近"就先查 seed/数据管线是否真的一致。
