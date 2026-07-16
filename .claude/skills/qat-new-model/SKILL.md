---
name: qat-new-model
description: 新模型接入 QAT 全流程——可捕获性预检、算子覆盖面核对、量化 config 编写、QAT 训练、导出与体检。第一次给一个新网络做量化时用这个。
---

# qat-new-model:新模型接入 QAT

目标产物:QDQ ONNX(`_sim.onnx`,交给 pulsar2)。
模板:`minimum/minimum_demo.py`(最小骨架,无训练)→ `resnet50/train.py`
(完整版:数据/训练循环/checkpoint/评测)。跑法见 qat-run,体检见 qat-check。

## 标准流程

```python
from utils import (AXQuantizer, capture, prepare_qat_pt2e, convert_pt2e,
                   move_exported_model_to_eval, dynamo_export,
                   export_float_reference, onnx_simplify,
                   simplify_and_fix_4bit_dtype)

export_float_reference(float_model, example_input, "float.onnx")   # ① float 参考
gm = capture(float_model.train(), (example_input,))                # ② 捕获(训练态)
print(gm.graph)                                                    # ③ 记下节点名 → config
quantizer = AXQuantizer("config.json")                             # ④
prepared = prepare_qat_pt2e(gm, quantizer)                         # ⑤
...QAT 训练循环...                                                  # ⑥ 观察者在训练中校准
quantized = convert_pt2e(prepared)                                 # ⑦
move_exported_model_to_eval(quantized)                             # (FP32 区域残留 BN 必需)
dynamo_export(quantized, example_input, "qat.onnx")                # ⑧

# ⑨ 后处理(选择条件详见下方"⑨ 后处理怎么选"),产物 qat_sim.onnx 交 pulsar2:
if config_含_4bit_或_FP32_混合区域:
    simplify_and_fix_4bit_dtype("qat.onnx", "qat_sim.onnx")
else:  # 纯 8/16bit 全量化
    onnx_simplify("qat.onnx", "qat_sim.onnx")
```

- ① 不可跳过:它是数值对齐与结构体检的对照物,且内置 eval 深拷贝
  (torch 2.10 训练态 BN 模型不能直接导 ONNX);
- ⑥ 无训练数据时可先喂几个随机 batch 前向(校准观察者)打通全流程再接真数据;
  训练辅助 `train_one_epoch/evaluate/evaluate_np` 可直接用(resnet50/train.py 示范);
- 动态 batch/H/W:capture 的 `dynamic_shapes` 原生透传,写法与坑见 qat-run。

**⑨ 后处理怎么选(`simplify_and_fix_4bit_dtype` vs `onnx_simplify`)**:

- **config 含 4bit(U4/S4)→ 必须用 `simplify_and_fix_4bit_dtype`**:导出器
  写不出原生 4bit(torch 侧以 uint8/int8 + 收窄 qmin/qmax 表达:U4=0..15、
  S4=-7..7),它靠 Q/DQ 节点的 `pkg.torch.onnx.fx_node` 元数据识别并改写成
  UINT4/INT4。此时输入必须是 ⑧ 直出的 raw,且它是唯一后处理——先过任何
  第三方 optimize/slim 会丢元数据或去重共享 zero_point,4bit 标记**静默
  丢失/污染**(2.6 老管线 zp 污染 bug 即此成因,见 README_2_10.md 告警);
- **含 FP32 混合区域 → 也用它**:FP32 区域会残留 zero-bias 的
  Expand(CastLike) 链,只有它内置的清理 pass 处理,`onnx_simplify` 不管;
- **纯 8/16bit 全量化 → `onnx_simplify` 即可**(yolov5_demo/reuse_conv/
  test_clamp 均此用法);用 `simplify_and_fix_4bit_dtype` 也无害——无 4bit
  时标记集为空,退化为普通 simplify(minimum/resnet50 即此,两条路径都
  经过回归验证);
- float 参考模型(无 Q/DQ)用 `onnx_simplify`;
- 产出的 4bit sim 只交 pulsar2,不喂 ORT(见"验收"第 3 条)。

## 第 0 步:可捕获性预检

模型必须能过 `torch.export`(数据依赖的控制流、动态列表操作等会失败)——
先单独跑一句 `capture(model.train(), (ex,))`,报错在这一步解决(改模型写法),
不要带着捕获问题进量化流程。

注意:这一步产出的**不是 ONNX**,而是 PyTorch 的 FX 图(aten 算子级的
`torch.fx.GraphModule`,仍可训练/前向)——注解、prepare、QAT 训练、convert
全在这张图上进行;ONNX 到第⑧步 `dynamo_export` 才出现。

## 算子覆盖面核对(新模型最容易踩空的一步)

AXQuantizer 只注解 `AXQuantizer.OPS` 列表内的算子(utils/ax_quantizer.py):
add / sub / mul / matmul / conv / convtranspose / linear / concat / split /
avgpool2d / layernorm / groupnorm / silu / gelu / glu / sigmoid / softmax /
leakyrelu / gridsample。要点:

- **conv/linear 按融合 pattern 注解**:conv[+bn][+relu/relu6/hardtanh] 是一个
  整体,配置只写核心算子(见 CONFIG.md);
- **不在列表内的算子静默保持浮点**——不报错。核对办法:数 prepare 后图里的
  fake_quant/观察者,或导出后用 qat-check 的 checker 看哪些算子两侧没有 Q/DQ;
- matmul 与 gridsample 有**内置 regional 默认**(S16 对称输入),不写 config
  也会生效,属有意设计;
- gru/mha 注解器在 torch 2.10 下不可用(显式 NotImplementedError,防静默漏注解);
- 新算子要支持:在 utils/ax_quantizer_utils.py 仿照 avgpool2d/layernorm 的
  **aten 直匹配**写法加注解器并注册进 OPS(2.10 下 get_source_partitions 已废,
  别参考 gru/mha 的旧写法)。

## 量化 config

格式详档 CONFIG.md:`global_config`(全局 U8/S8)+ `regional_configs`
(按 `module_type` + 可选 `module_names` 指定区域混合精度 U4/U16/FP32,
样例见 resnet50/config_4w4f.json 等 5 份)。两条铁律:

1. **module_names 必须在 prepare 之前的捕获图上找**(第③步 print);
   convert 会折叠重建 conv 并整体改名(conv2d_106 起),convert 后的名字对不上
   是正常现象。convert 后要定位结构(如切子图)用拓扑序位置索引,参考
   multi_stage/multi_stage_demo.py::find_stage_cuts;
2. bias 现状:仅 conv1d/2d 默认派生量化(int32,scale=Sa×Sw),Linear/
   ConvTranspose 不量化,config 无 bias 通道(现状与待定项见 README_2_10.md 告警)。

## 先打通链路,再投入训练(强烈建议)

QAT 工具接入完成后,**别急着开完整训练**。先用小批量数据训 1 个 epoch
(甚至几十步),立即走完 ⑦convert → ⑧导出 → ⑨simplify,把 `_sim.onnx`
送 pulsar2 编译、部署到 Axera NPU 跑通一次前向——**优先验证"QAT→导出→NPU
部署"整条链路畅通**。

理由:量化配置/算子覆盖面/导出结构的问题(某算子 NPU 不支持、pulsar2 编译
报错、shape 不被接受),在这条最短闭环里就会暴露,精度好不好此刻不重要。
链路一旦跑通,再投入完整训练调精度,避免训到收敛才发现卡在部署环节、
配置推倒重来。此时的 checkpoint/onnx 是临时验证品,可随时清理。

## 验收(每个新模型都做)

1. checker:raw 加 `--ort`(数值对齐),sim 用 `--sim`——命令与已知例外解读
   见 qat-check;首次通过后 `--save-profile` 固化基线,此后改动跑基线对比;
2. 精度:QAT 后 eval 对 float 基线,掉点异常时先查覆盖面(是不是关键算子
   没被注解、或该 FP32 的区域被量化);
3. 交付 pulsar2 的一律是 `_sim.onnx`;含 4bit 的 sim 喂 ORT 报 MaxPool 不接受
   uint4 属预期,数值对齐在 raw 上做。

## 新模型特有坑

- **多个层复用同一 BatchNorm 模块**:convert 崩时 capture 后先过
  `remove_reused_bn_param_hack(gm)`(版本无关实现在 utils/ax_quantizer.py;
  调用点位置见 reuse_conv/train.py 中的注释行);
- 捕获样例的动态维取值别用 1(0/1 特化;capture 会自动翻倍规避,但样例
  尽量给 ≥2);
- 训练/评测切换用 `move_exported_model_to_train/eval`(捕获图不再响应
  `.train()/.eval()`);
- checkpoint 跨 torch 2.6/2.10 可互载(state_dict 键名一致,已实测逐 bit 等价)。
