---
name: qat-check
description: QAT 导出 ONNX 结构体检——13 条规则检查器、金标准基线对比、全配置矩阵。导出后验证结构、排查量化丢失/污染、对比两份 onnx 时用这个。
---

# qat-check:导出结构体检

工具在 `env_check/`(详档 env_check/README.md)。FAIL 时退出码非 0,可接 CI。
命令以仓库根目录为工作目录;本地编辑端操作时包一层 `ssh qat-dev '...'`。

```bash
P=/home/heqi/miniforge3/envs/torch2.10/bin/python
$P env_check/check_onnx_structure.py --model <raw>.onnx --ort           # raw 规则(R1~R9)
$P env_check/check_onnx_structure.py --model <sim>.onnx --sim           # sim 规则(跳 R9,Cast 升 FAIL)
$P env_check/check_onnx_structure.py --model <sim>.onnx --sim \
   --baseline env_check/baselines/<x>.profile.json                      # 与基线逐项对比(同网络才有意义)
$P env_check/check_onnx_structure.py --model <m>.onnx --save-profile p.json  # 固化新基线
$P env_check/cmp_sim_matrix.py                                          # resnet50 全配置 2.6vs2.10 总表
```

基线:`resnet50_qat[_sim].profile.json`(金标准,sim 为 4w4f 流 S4/U4)、
`minimum_qat[_sim].profile.json`。

## 结果解读(已知例外,勿误判)

| 现象 | 定性 |
|------|------|
| raw 有 R7-cast WARN(per-channel 零点 Cast) | 预期,sim 阶段折叠 |
| 2.10 raw 臃肿(Constant/Cast 数百个) | optimize=False 的代价;**看图/交付一律用 sim**(与 2.6 同样干净) |
| R4b 在 FP32 区域的 conv 上 FAIL | 两侧**对称误报**(conv 读共享 DQ 输出、权重故意不量化),人工确认即可 |
| 4bit sim 喂 ORT 报 MaxPool 不接受 uint4 | 金标准同款行为,4bit sim 只给 pulsar2;数值对齐在 raw 上做 |
| R4b 真 FAIL(量化 conv 权重无 DQ) | **严重**:权重量化被常量折叠丢失,检查是否绕过了 utils 的 dynamo_export(optimize 必须关) |
| R7-leftover 有 pkg 域函数节点(如 aten_hardtanh) | 函数型算子未内联;utils 的 dynamo_export 已内置 onnx.inliner,确认走的统一导出 |
| ir_version=12 / ORT 拒载 | 后处理没走统一 simplify(内置 ir=10 回写) |

## 跨版本/跨运行对比方法论

- 拓扑**平序**可不同(并行分支输出顺序):按位置对齐会出假差异,
  **按值/按消费者匹配**(参考 qparam 多重集合 md5 对账法);
- **2.6 老管线的"混合 4bit"产物不可作金标准**(optimize 去重 + 按名标 4bit
  的 zp 污染 bug:实测 47 激活+18 权重被误标;统一管线无此问题);
- convert 后 conv 被折叠重命名(conv2d_106 起):找 module_names 要在
  **prepare 前的捕获图**上;convert 后定位用拓扑序位置索引
  (参考 multi_stage/multi_stage_demo.py::find_stage_cuts)。
