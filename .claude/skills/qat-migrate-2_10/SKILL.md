---
name: qat-migrate-2_10
description: 把一个已在 torch 2.6 跑通的 PT2E QAT 项目迁到 torch 2.10 的排雷清单——命名空间/元数据/BN patch/导出后处理四类坑的识别与修法。迁移别的 AXERA 系 QAT 项目(如 QAT.Ultralytics)时用这个。
---

# qat-migrate-2_10:torch 2.6 → 2.10 QAT 迁移排雷

适用:一个 PT2E QAT 项目(自定义 Quantizer + torch.ao PT2E)已在 2.6 跑通,
要迁到 torch 2.10。本仓库 `utils/` 就是迁移完成态,可作**参照实现**逐文件对拍。
根因:2.10 中 `torch.ao` 的 PT2E 已弃用且实际已坏,元数据换代,导出器行为变化。

## 第 0 步(最重要):先甄别活代码 vs 死代码,别改死代码

grep 到某个雷点符号 ≠ 它在跑。**先追调用链**:入口脚本(train/export/eval)
→ engine/trainer/exporter → 实际用的是哪个 Quantizer、哪些函数。只改活路径上的。

典型陷阱:一个项目常并存多个量化器(如 config 驱动版 + 老式 nn_module_stack 版
+ LSQ 版),入口只 import 其中一个,其余是遗留/预留。**死代码里的
`source_fn_stack`/`nn_module_stack`/`get_source_partitions` 不构成迁移障碍**,
改了纯属浪费且引入风险。判定:`grep -rn "import <模块>" 全仓 --include=*.py`,
无活引用(只有自引用或注释掉的 import)即死代码。

## 四类坑(每类:症状 → grep 识别 → 修法)

### ① PT2E 命名空间:torch.ao 必须整体迁 torchao(判决点)

- **症状**:`convert_pt2e` 在 conv-bn 折叠处报 `KeyError: 'source_fn_stack'`
  (`export_for_training`/`torch.export.export`/`strict=False` 三种捕获入口
  结果相同 → 非用法问题,是 torch.ao 弃用后 bitrot)。
- **grep**:`from torch.ao.quantization` / `torch.ao.quantization.quantize_pt2e`
  / `move_exported_model_to_eval` / `allow_exported_model_train_eval`。
- **修法**:全部换到 `torchao.quantization.pt2e.*`。⚠️ **不能只换 prepare/convert
  入口**:torch.ao 与 torchao 的 `QuantizationSpec`/`Quantizer`/observer/
  fake_quantize 是两套独立类型,混用直接 AssertionError——量化器、观察者、
  DerivedQuantizationSpec 等**整包**迁移。符号改名对照见本仓库 utils/_compat.py
  (WrapperModule / _annotate_*_qspec_map / _get_module_name_filter /
  _DerivedObserverOrFakeQuantize 等)。

### ② export 元数据换代:注解器静默失效(最隐蔽)

2.10 的 `torch.export` 不再产出 `source_fn_stack`/`nn_module_stack`,
`get_source_partitions` 随之返回空 → 依赖它们的注解器**不报错、但一个都不注解**,
量化悄悄消失,精度崩了才发现。

- **grep**:`source_fn_stack` / `nn_module_stack` / `get_source_partitions`。
- **修法**:凡活路径上依赖它们的注解器,改写为 **aten 算子直匹配**
  (SubgraphMatcher 或直接遍历 graph 匹配 aten target)。本仓库
  utils/ax_quantizer_utils.py 的 avgpool2d/layernorm/groupnorm/concat 即改写样例;
  conv/convtranspose 用 SubgraphMatcher 保融合(relu6/hardtanh)。改不动或用不到的
  (gru/mha)加 `NotImplementedError` 守卫,**宁可显式报错也不静默漏注解**。
- **`export_for_training` 捕获入口**:2.10 废弃 → `torch.export.export(...).module()`;
  封装 + 动态 shape 处理见 utils/capture.py。
- **`remove_reused_bn_param_hack` 一类读 source_fn_stack 的 hack**:改写为按
  buffer 名(`num_batches_tracked`)识别的版本无关实现(utils/ax_quantizer.py),
  仅复用 BN 场景需要;是否活雷取决于目标模型是否复用 BN 且是否调用它。

### ③ BN patch:2.6 专属的 monkey-patch 大概率失效

项目常有 `pt2e_bn_patch.py` 一类文件,`patch` 掉
`torch.ao.quantization.pt2e.export_utils._replace_batchnorm` 保留 BN 超参。
**它常在 utils/__init__ 里模块级激活(import 即打 patch)**。

- **grep**:`_replace_batchnorm` / `export_utils` / 谁在 `__init__` 里调 patch 函数。
- **修法**:迁 torchao 后 patch 目标路径变成 `torchao.quantization.pt2e.export_utils`,
  原 patch 静默失效或报 AttributeError。先判断 torchao 下**是否还需要**这个 patch
  (torchao 可能已修),需要则把 patch 目标改到 torchao 命名空间;不需要则移除激活。
  相关坑:2.10 不能直接导出训练态 BN 模型(buffer 突变报错),float 参考须
  eval 深拷贝导出(utils/train_utils.py::export_float_reference);convert 后
  FP32 区域残留训练态 BN → convert 后补 `move_exported_model_to_eval`。

### ④ 导出后处理:optimize 折叠 / 元数据格式 / ir_version

- **`torch.onnx.export(dynamo=True).optimize()`**:2.9+ 默认 optimize=True 会把
  「int8 权重 + DQ」链常量折叠成 float(权重量化丢失);且 initializer 去重会
  **污染混合 4bit 标记**(共享 zp 被合并 + 按名标 4bit → 8bit 激活被误标)。
  **修法**:导出 `optimize=False`;需要 slim 走专用后处理(见下)。注:onnxscript
  0.6.2 修了「权重折叠」但**没修 initializer 去重污染**,混合 4bit 仍须 optimize=False。
- **函数型 torchlib 算子**:optimize=False 下 relu6/hardtanh 以未内联的
  `aten_hardtanh` 函数节点存活(onnxslim 不内联函数,pulsar2 不认 ONNX functions)。
  **修法**:导出后 `onnx.inliner.inline_local_functions` + 清理未用 domain
  (utils/train_utils.py::dynamo_export)。
- **QDQ 元数据格式变了**:2.10 的 `pkg.torch.onnx.fx_node` / `namespace` 格式与
  2.6 不同,只按 `namespace.split(": ")[1]` 单格式解析会 assert 或误判。
  **修法**:优先从 fx_node regex 提 target,namespace 回退——双格式兼容
  (utils/quant_utils.py::simplify_and_fix_4bit_dtype)。
- **ir_version**:simplify 管线会把 ir_version 抬到 12,ORT/老 pulsar2 拒载。
  **修法**:回写 ir_version=10。
- FP32 混合区域残留 zero-bias 的 Expand(CastLike) 链 → `_castlike_to_cast` 清理。

## 迁移顺序与验证

1. **先跑最小闭环证伤**:torch2.10 env 里跑一次目标项目的 QAT(prepare→convert),
   十有八九停在 ①的 convert_pt2e KeyError——确认伤在 torch.ao 而非用法;
2. 按 ①→②→③→④ 修,每步小闭环回归(一个 minimal 网络跑通再上真模型);
3. **双环境等价性对照**:同 config、同 seed,2.6 与 2.10 各跑一遍,比注解计数、
   逐 batch loss、导出 sim 结构(参考本仓库四层证据:结构/qparam/训练行为/精度);
4. 结构体检用 qat-check(checker + 基线对比);跑通后先小批量训 1 epoch 直接
   导出部署到 NPU 验证全链路(见 qat-new-model),再投完整训练。

## 已印证案例

QAT.Ultralytics(YOLO11 QAT,克隆在 cache/,已 .gitignore)是本清单的现实印证:
它 utils 与本仓库迁移前同构,①~④ 的活雷全中(全 torch.ao、ax_quantizer 经
get_source_partitions 注解、pt2e_bn_patch 模块级激活、导出 optimize());
其 quantizer.py/quantizer_utils.py/ax_quantizer_lsq.py 为死代码(无活引用),
按第 0 步甄别可直接跳过。其 requirements 已是 onnx 1.19.1/onnxscript 0.6.2/
onnx-ir 0.1.15,onnx 生态零迁移成本。
