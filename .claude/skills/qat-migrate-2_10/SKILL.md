---
name: qat-migrate-2_10
description: 把一个已在 torch 2.6 跑通的 PT2E QAT 项目迁到 torch 2.10 的排雷清单——命名空间/元数据/BN patch/导出后处理四类坑的识别与修法。迁移任何基于 torch.ao PT2E + 自定义 Quantizer 的 QAT 项目时用这个。
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
量化悄悄消失,精度崩了才发现。这里有一个**响亮早炸**和一个**沉默晚炸**:

- **先炸(响亮,发生在 prepare 阶段,早于 convert):`gm_using_training_ir`**。
  conv/convtranspose 注解器里 `from torch._export import gm_using_training_ir`
  在 2.10 报 `ImportError: cannot import name 'gm_using_training_ir'`(该 helper
  2.10 已移除)。它把 `using_training_ir` 传给 `get_aten_graph_module_for_pattern`,
  而后者 2.10 签名也变了。**修法**:删该 import,改用只接
  `(pattern, example_inputs, is_cuda, gm)` 的 2.10 版
  `get_aten_graph_module_for_pattern` 包装(2.6 的 `using_training_ir=` 在包装内
  吸收),见本仓库 utils/_compat.py。grep:`gm_using_training_ir`。
- **后炸(沉默):`get_source_partitions` 返回空** → 注解器静默 no-op(下详)。
- **grep**:`gm_using_training_ir` / `source_fn_stack` / `nn_module_stack` / `get_source_partitions`。
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

### ③ BN patch:迁 torchao 时必须同步搬家(易漏)

项目常有 `pt2e_bn_patch.py` 一类文件,`patch` 掉
`torch.ao.quantization.pt2e.export_utils._replace_batchnorm` 保留 BN 超参。
**它常在 utils/__init__ 里模块级激活(import 即打 patch)**。

- **注意**:原地 2.10(还没迁 torchao 时),`torch.ao.quantization.pt2e.export_utils`
  **仍在**,patch 照常应用成功——所以它**不是**当前的报错点,容易被忽略。
- **grep**:`_replace_batchnorm` / `export_utils` / 谁在 `__init__` 里调 patch 函数。
- **修法**:一旦 ① 把 PT2E 调用迁到 torchao,BN 折叠走的就是
  `torchao.quantization.pt2e.export_utils`,而 patch 还盯着 torch.ao 的旧模块 →
  **静默不生效**(BN 超参又丢了)。迁 ① 时同步判断 torchao 下是否仍需此 patch
  (可能已修),需要则把 patch 目标改到 torchao 命名空间,不需要则移除激活。
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

## 实测错误签名参考(在 torch 2.10 上会遇到的两处典型报错)

在 torch 2.10 env 跑一个 torch.ao PT2E + 自定义 Quantizer 项目的最小 QAT 闭环,
典型会撞到下面两处(与 ①②逐一对应),按签名即可快速定位:

- **prepare 阶段先炸(响亮)**:`prepare_qat_pt2e → annotate → conv 注解器` 报
  `ImportError: cannot import name 'gm_using_training_ir' from 'torch._export'`
  ——**根本走不到 convert**(对应 ②「先炸」)。
- **convert 阶段炸(内核问题)**:把自定义量化器换成 torch 原厂
  `XNNPACKQuantizer` 做隔离验证——prepare 正常、有注解,但 `convert_pt2e` 报
  `KeyError: 'source_fn_stack'`。**用原厂量化器复现 = 证明是 torch.ao PT2E 内核
  本身在 2.10 已坏,非你的代码**(对应 ①)。torch 2.10 启动时也会打印官方横幅劝迁
  torchao。
- 附:`pt2e_bn_patch` 一类 BN patch 在原地 2.10 会激活成功(export_utils 仍在),
  对应 ③「原地不报错」。

**快速证伤手法(不必装目标项目的全部依赖)**:桩掉目标包的重 `__init__`
(用 `types.ModuleType` + `__path__` 指向真实目录,跳过与量化内核无关的重依赖),
只加载它的量化内核模块;再在 torch 2.10 env 用一个 conv-bn-relu 最小网络跑
prepare/convert,CPU 即可。既能证伤,又能在动手迁移前摸清活/死代码边界。
