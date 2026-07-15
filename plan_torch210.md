# QAT.axera torch 2.6 → 2.10 适配规划

> 注:历史文档(P0–P4 迁移知识库)。内文的 utils_2_10/resnet50_2_10/*_2_10
> 命名反映当时的双轨布局;2026-07-14 起已统一为 utils/ 并对齐上游目录
> (演进记录见 plan_unified_api.md)。

> 环境：`/home/heqi/miniforge3/envs/torch2.10`（配套依据见 `env.md`）
> 结构检查工具与金标准基线：`env_check/`（见其 README）
> 撰写：2026-07-09,基于远程机实测,非文档推断

## 一、目标与总体验收标准

把仓库的 PT2E QAT 全流程(注解 → QAT 训练 → convert → ONNX 导出 → simplify/4bit
后处理)从 torch 2.6 迁到 torch 2.10,产物结构与 torch2.6 金标准一致、pulsar2 可消费。

验收(全部满足):

1. `minimum/minimum_demo.py`、`resnet50/train.py` 在 torch2.10 env 跑通;
2. `env_check/check_onnx_structure.py` 对新导出 raw 模型 **0 FAIL**(允许 R7-cast WARN);
3. resnet50 新导出与金标准基线 `--baseline env_check/baselines/resnet50_qat.profile.json`
   结构逐项一致(op 直方图、QDQ 计数、权重/激活量化形态);
4. sim 后处理产物 ir_version==10,ORT 可加载,数值与 torch 侧对齐;
5. resnet50 QAT 短训精度与 torch2.6 基线相当(冒烟级:fake_data 收敛趋势一致;
   完整级:imagenet 精度差 < 0.1pt)。

## 二、2.10 相比 2.6 的 quant 相关改动(重点)

### 改动 1:PT2E 训练栈从 torch.ao 迁到 torchao(强制,非可选)

- **torch.ao 的 PT2E QAT 在 2.10 已实际损坏**:`convert_pt2e` 在 conv+BN 折叠处
  `KeyError: 'source_fn_stack'`(根因见改动 2),三种 export 入口均复现,无绕行空间。
- **两套类型不互通**:torch.ao 与 torchao 的 `QuantizationSpec/Quantizer` 是独立类,
  torchao 的 `prepare_qat_pt2e` 收到 torch.ao 的 spec 直接
  `AssertionError: Expected QuantizationSpec got ...` → **不能只换 prepare/convert
  入口,自定义量化器全家(spec、observer 构造、注解工具)必须整体换命名空间**。
- 符号迁移映射(torchao 0.16.0 实测可导入):

| torch 2.6(torch.ao) | torch 2.10(torchao) |
|---|---|
| `torch.ao.quantization.quantize_pt2e.prepare_qat_pt2e/convert_pt2e` | `torchao.quantization.pt2e.quantize_pt2e.*` |
| `torch.ao.quantization.quantizer.{Quantizer, QuantizationSpec, DerivedQuantizationSpec, SharedQuantizationSpec, QuantizationAnnotation}` | `torchao.quantization.pt2e.quantizer.*` |
| `torch.ao.quantization.fake_quantize.{FakeQuantize, FusedMovingAvgObsFakeQuantize}` | `torchao.quantization.pt2e.fake_quantize.*` |
| `torch.ao.quantization.observer.{HistogramObserver, MinMax/MovingAverage(PerChannel)MinMaxObserver, PlaceholderObserver}` | `torchao.quantization.pt2e.observer.*` |
| `torch.ao.quantization.{ObserverOrFakeQuantize, _DerivedObserverOrFakeQuantize}` | `torchao.quantization.pt2e.*`(私有名以实现为准) |
| `torch.ao.quantization.pt2e.utils.get_new_attr_name_with_prefix` 等 | `torchao.quantization.pt2e.utils.*` |
| `torch.ao.quantization.pt2e.export_utils._WrapperModule` | `torchao.quantization.pt2e.export_utils`(符号名待核对) |
| `torch.ao.quantization.quantizer.utils._get_module_name_filter` | `torchao.quantization.pt2e.quantizer.utils`(名字有变,实现时核对) |
| `torch.ao...xnnpack_quantizer_utils.{OP_TO_ANNOTATOR, QuantizationConfig, OperatorConfig}` | **不迁移**:torchao testing 版注解语义与仓库定制版不同,自持仓库版只换依赖(见改动 5,2026-07-09 增补) |
| `torch.ao.quantization.move_exported_model_to_eval/train` | `torchao.quantization.pt2e.move_exported_model_to_eval/train` |

- torchao 侧闭环已验证:prepare_qat → 训练 → convert → 推理 → ONNX 导出 →
  ORT 数值对齐 max|diff|=0.0(见 `env_check/qax_smoke2.py`)。

### 改动 2:torch.export 图元数据换代(影响注解与模块定位)

**2.10 的 `torch.export.export(...).module()` 节点 meta 不再携带
`source_fn_stack` 和 `nn_module_stack`,只有 `from_node`(新 provenance 机制);
`torch.fx.passes.utils.source_matcher_utils.get_source_partitions` 返回空。**(实测)

连锁影响:

- torch.ao `convert_pt2e` 崩溃的根因(它按 `source_fn_stack` 找 conv+BN 对);
- `utils/ax_quantizer_utils.py` 里所有走 `get_source_partitions` 的注解器
  在 2.10 下会**静默注解不到任何层**(不报错,直接漏注解)——迁移时必须改成
  torchao 版注解工具的匹配方式(aten 目标直匹配 / from_node);
- `_get_module_name_filter`(regional_configs 的 module_names 过滤)依赖
  nn_module_stack,须换 torchao 实现;
- `CONFIG.md` 里"打印 graph 找 module_names"的工作流,节点命名和定位方式
  会变化,配置文件里的 module_names(如 `conv2d_1`、`add__5`)需按新图重新核对;
- `export_for_training` 已弃用(2.10 仍在但走老路径),统一换 `torch.export.export`。

### 改动 3:torch.onnx 导出器行为变化(结构级风险)

- 2.10 中 `dynamo=True`、**`optimize=True` 均为默认**;
- **新版 onnxscript 的 optimize 常量折叠会把「int8 权重 initializer +
  DequantizeLinear」整链折叠成 float 权重**——图检查一切正常但权重量化信息全丢,
  pulsar2 拿到的是浮点权重。`utils/train_utils.dynamo_export` 里的
  `onnx_program.optimize()` 同样触发。(torch2.6 + onnxscript 0.2.2 无此行为)
  → **导出必须显式 `optimize=False`**,折叠交给下游
  `gs.fold_constants + onnxslim`(实测它们不折 QDQ);
  → `env_check` 已加 R4b 规则专防此回归(现行管线 FAIL,optimize=False 通过);
- QDQ 节点 `metadata_props` 里 namespace 的 target 格式从
  `quantized_decomposed.quantize_per_tensor.default` 变成
  `torch.export._trace...Wrapper/quantize_per_tensor` →
  `utils/quant_utils.simplify_and_fix_4bit_dtype` 的 `namespace.split(": ")[1]`
  解析直接 assert,需按新格式适配(建议取 `/` 后段 + 前缀白名单);
- 默认 opset 变为 20(仓库显式传 `opset_version=21`,此项只需保持显式传参);
- **训练态 BN 模型不能再直接导出**(P1 实测):float 模型(training=True,BN 带
  buffer 突变)走 dynamo 导出报
  `Key 'b_bn_running_mean' does not match the name of the value 'getitem_3'`,
  2.6 可以容忍 → float 参考模型须用 **eval 态深拷贝**导出(QAT 捕获仍用训练态
  原模型);分支上 WIP 的 resnet50/train.py 把该导出包 try/except pass 即是
  踩过此坑的痕迹,P2 迁移时应改为 eval 拷贝导出而非吞异常。

### 改动 4:onnxscript / onnx 生态换代(0.2.2 → 0.5.4)

- 自定义 torchlib 映射 `utils/quantized_decomposed_dequantize_per_channel.py`
  (`@torch_op("quantized_decomposed::dequantize_per_channel")`)**在 0.5.4 上
  仍然生效**(实测 per-channel DQ + zp Cast 正常出现在导出图中)——短期可沿用,
  中期建议改官方 `custom_translation_table` 参数(风险 2);
- 后处理链 `gs.export_onnx/make_model` 会按当前 onnx 库默认值**盖章 ir_version
  (=12)**,ORT 1.23(max 11)与老 pulsar2 拒载 → 后处理保存前回写
  `model.ir_version = 10`(现存 `resnet50_qat_sim.onnx` 即中招样本);
- `onnxruntime.quantization.quant_utils.pack_bytes_to_4bit` 在 ORT 1.23.2 仍在(实测)。

### 改动 5:torchao testing 注解器与仓库定制版语义不同(2026-07-09 实测增补)

对比三方(仓库 `utils/ax_quantizer_utils.py` 1701 行 / torch2.6 原版 1112 行 /
torchao 0.16 testing 版 1133 行)结论:

- **上游 T26→TAO 几乎没动语义**:diff 只有 import 换 torchao、符号去下划线
  (`_WrapperModule`→`WrapperModule`、`_annotate_input_qspec_map`→
  `annotate_input_qspec_map`、`_get_module_name_filter`→`get_module_name_filter`)
  和 typing 现代化;
- **仓库 fork 与两者语义差异大,不可用 torchao testing 版替代**,以 conv 为例:
  - 仓库把 {conv1d/2d} × {±bn} × {relu/relu_/relu6/relu6_inplace/无} 的**全部
    融合模式收进一个 `"conv"` 注解器**(SubgraphMatcher 子图匹配),整个
    conv-bn-act 视为一个量化单元,只在边界打 Q/DQ;还带 regional 混合精度
    的 re-annotate 路径(`_update_last_node_output_qspec`);
  - torchao testing 版 `"conv"` 只注解裸 conv,融合变体拆成 conv_relu/conv_bn/
    conv_bn_relu 独立注册,且**没有 relu6(hardtanh) 融合**——直接换用会导致
    conv→bn 间被插 Q/DQ、relu6 网络结构错乱、精度崩;
  - 仓库另有上游没有的注解器:sub/matmul/gridsample/silu/gelu/sigmoid/
    leakyrelu/glu/softmax/sdpa/split/convtranspose(统一版)等。
- **仓库 22 个注解器按匹配机制分三类**(决定 2.10 兼容性):
  - `SubgraphMatcher`(2.10 可用,仅换依赖):conv、convtranspose;
  - `aten 直匹配`(2.10 天然兼容):linear、add、sub、mul、matmul、gridsample、
    silu、gelu、sigmoid、leakyrelu、glu、softmax、sdpa、split;
  - `get_source_partitions`(**2.10 下静默空匹配,必须改写**):gru_io_only、
    avgpool2d、layernorm、groupnorm、concat、mha。
    其中 avgpool2d(resnet50 的 GAP/ReduceMean 用)、concat(yolov5 用)优先改写,
    其余四个先加"2.10 下显式报错"守卫,防静默漏注解。

### 未变/已验证可用(不需要动的部分)

- QDQ 表示形态不变:激活 per-tensor Q→DQ 成对、权重 int8 initializer +
  per-channel DQ(axis)、bias 不量化 —— 金标准结构在 2.10 可完整复现;
- raw 导出 ir_version 仍为 10、opset 可指定 21;
- `move_exported_model_to_eval/train` 语义不变(换 torchao 入口);
- 4×A100 训练环境、数据管线(torchvision 0.25)无 quant 相关阻碍。

## 三、仓库受影响面盘点

按 torch.ao/export_for_training 引用密度与迁移动作分类(全仓 grep 实测):

| 文件 | 迁移动作 |
|---|---|
| `utils/ax_quantizer.py`(6 处) | **P0** 命名空间整体迁移 + module_name filter 换 torchao 实现 |
| `utils/ax_quantizer_utils.py`(7 处) | **P0** 同上 + 注解器匹配机制从 source_partitions 改 torchao 方式 |
| `utils/quantizer.py` / `quantizer_utils.py`(8/6 处) | **P0 或弃用**:与 ax_* 是新旧两套,先确认是否仍被引用,弃用则不迁 |
| `utils/train_utils.py`(2 处) | **P0** `dynamo_export` 加 `optimize=False`;evaluate 里 move_to_eval 换 torchao |
| `utils/quant_utils.py`(2 处) | **P1** metadata namespace 解析适配 + ir_version 回写 + `_DerivedObserverOrFakeQuantize` 换源 |
| `minimum/minimum_demo.py`(4 处) | **P1** 首个端到端打通样例(兼回归入口) |
| `resnet50/train.py` / `test.py`(4/3 处) | **P2** 主验收模型;注意分支上已有 4w4f WIP 改动(fake_data 开关),迁移时保留 |
| `multi_stage/`(11+5 处)、`reuse_conv/`(4 文件)、`test_clamp/`(4 文件) | **P3** 批量替换,逐个跑通 |
| `minimum/yolov5_demo.py`、`yolov5/train.py` | **P3** 同上 |
| `requirements.txt` | **P4** 更新为 env.md 的版本组合 |

## 四、分阶段实施计划

### P0:utils 核心迁移(主战场)✅ 已完成(2026-07-09,验收全绿)

**双环境策略(2026-07-09 定)**:`utils/` 原样保留 = torch2.6 实现;新建
**`utils_2.10/`** 平行包 = torch2.10/torchao 实现,包内互引指向 utils_2.10,
与 2.6 版共存互不影响;demo 脚本后续按环境选 import(P1 起处理)。

1. `utils_2.10/ax_quantizer_utils.py`:从 utils 版移植——import 全部切 torchao
   (按改动 5 的改名表);删除 `gm_using_training_ir`(torch._export 已移除,
   2.10 只有 training IR 一条路);6 个 source_partitions 注解器中优先改写
   avgpool2d、concat 为 aten 直匹配,gru/layernorm/groupnorm/mha 加显式
   NotImplementedError 守卫(防静默漏注解);
2. `utils_2.10/ax_quantizer.py`:spec/observer/fake_quantize 换 torchao 类;
   `_get_module_name_filter` → torchao `get_module_name_filter`;
3. `utils_2.10/train_utils.py`:`dynamo_export` 显式 `optimize=False`;
   `evaluate` 的 move_exported_model_to_eval 换 torchao 入口;
4. `utils_2.10/quant_utils.py`:metadata namespace 新格式解析 + 保存前回写
   `ir_version=10`;
5. `utils_2.10/quantized_decomposed_dequantize_per_channel.py`:原样拷贝
   (onnxscript 0.5.4 实测兼容);
6. `utils/quantizer.py` 旧套(demo 均已注释改用 ax_*)不迁。

**验收**:`gen_candidate_torch210.py` 把 `build_quantizer()` 换成
`utils_2.10.AXQuantizer(minimum/config.json)` 后,TinyNet 全链路跑通,
checker 0 FAIL,且注解层数与 config 预期一致(防静默漏注解:断言
prepared_model 中 fake_quant 模块数量 > 0 且符合层数预期)。

### P1:最小样例 + 后处理链打通 ✅ 已完成(2026-07-09,验收全绿)

1. 新建 `minimum/minimum_demo_2_10.py`(2.6 原版不动,双轨并存),
   走 utils_2_10 + torchao + torch.export.export,float 参考用 eval 拷贝导出;
2. checker:raw 13 项 0 FAIL(1 预期 WARN),sim 12 项全 PASS;
3. **与 2.6 头对头**:torch2.6 env 重跑原版 demo 生成金标准
   (baselines/minimum_qat*.profile.json),`--baseline` 对比:
   **sim vs sim 16/16 全项一致**;raw vs raw 仅差 1 Cast + 5 Constant
   (2.10 关 optimize 的预期残留,sim 阶段被 gs.fold_constants 清掉)。

### P2:resnet50 主模型验收

**数据方案(2026-07-09 定,用户拍板)**:机器上没有 ImageNet 训练集
(`dataset/imagenet/` 仅 val tar 包 8.4GB),P2 不依赖 ImageNet:

- **结构回归 → fake_data**:与数据内容无关,用现成 `fake_data=True` 零下载;
- **2.6/2.10 等价性对照 → CIFAR-10**:两个环境同一脚本、同 seed 同数据同超参
  各短训 N 步,对比逐 batch loss 曲线与导出结构是否一致(迁移验证的本质是
  "行为一致",不是刷精度);10 类需换 fc 头(脚本已处理,fc 初始化重设种子);
- **绝对精度指标**:留待有 ImageNet 训练集后补充,或解 val.tar 做评测型对照;
- CIFAR-10 由用户自行下载(机器直连外网会挂):
  `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`(~163MB,
  md5 `c58f30108f718f92721af3b95e74349a`),放到
  `qat-dev:/home/heqi/project-qat/QAT.axera/dataset/cifar10/cifar-10-python.tar.gz`,
  torchvision 校验 md5 后自动解包,无需再下载。

**代码组织**:新建 **`resnet50_2_10/`**(`resnet50/` 原样保留 = 2.6 版):
`train.py`/`test.py` 为**双环境单脚本**——运行时按 torch 版本自动切
utils/torch.ao(2.6)或 utils_2_10/torchao(2.10),保证等价性对照两边跑同一份
数据管线代码;产物文件名带 `_2_6`/`_2_10` 后缀写入本目录;预训练权重仍引用
`resnet50/` 原位置;**5 份量化配置已复制到本目录**(2026-07-10 用户要求,
与原件逐字节一致,三个脚本默认路径已指向本目录副本);float 导出用 eval
深拷贝(修正 WIP 的 try/except pass)。

步骤:

1. ✅ `resnet50_2_10/train.py`/`test.py` 已建;
2. ✅ fake_data 短训(10 batch,loss 8.25→7.91)→ 导出 raw+sim,
   checker:raw 13 项 0 FAIL(54 个零点 Cast 预期 WARN),注解计数 128;
3. ✅ **结构回归通过**:config 用 `config_4w4f_all.json`(经核对,金标准 sim 全
   U4/S4 只与它对应;`config_4w4f.json` 是混合精度版,对不上),sim 产物对
   `resnet50_qat_sim.profile.json` **基线 5 项全一致**(op 直方图/S4×54 权重/
   U4×74 激活/counts/opset);
4. ✅ 数值路径:test.py checkpoint 加载 + torch/ORT 双路评测跑通(fake 冒烟);
5. ✅ **等价性对照通过**(2026-07-09,CIFAR-10 by 用户,md5 校验一致):
   两 env 各跑 `train.py --data cifar10 --steps 50 --seed 42 --eval-size 20`
   (2.6@GPU3 / 2.10@GPU2 并行):
   - 注解计数两侧同为 128;
   - 逐 batch loss 曲线锁步:2.6 2.388→2.032,2.10 2.432→2.046,
     逐点差 +0.002~+0.044、无发散趋势(数值核/观察者精度级差异);
   - 评测 top1 56.6(2.6) vs 53.8(2.10) @n=640,在该样本量统计噪声内;
   - **两侧导出 sim 结构对比 16/16 全项一致**(含基线 5 项);
   - **完整 1 epoch 精度对照(2026-07-10 补,全量 5 万训练 + 1 万 test 评测)**:
     末尾 running loss 0.362 vs 0.361;test top1 **94.72(2.6) vs 94.97(2.10)**,
     Δ0.25pt 为噪声量级且 2.10 略高;top5 双侧 99.96 完全相同 →
     50 步时的 2.8pt 差异定性为"欠训练 + 小评测集"双重噪声,
     **最终精度等价成立**。

**P2 结论:torch2.10 迁移在 resnet50 上结构与训练行为均与 2.6 等价。**

6. ✅ **深对比 + 交叉实验**(2026-07-10,详见 env_check/README.md):
   两侧 sim 拓扑序 324 节点逐位置一致;同一 checkpoint(2.6 训练,1216 键)
   strict 载入两体系,convert 后 **202 个 Q/DQ 的 scale+zero_point 按值
   逐 bit 一致** —— observer→qparam 计算跨体系等价、checkpoint 互通。
   注意:约 8 个权重 DQ 拓扑平序不同,跨版本比较须按值匹配而非按位置。
7. ✅ **全配置矩阵复测**(2026-07-10,resnet50 下 5 份配置全部在 2.10 复测,
   fake_data + 双环境对照,详见 env_check/README.md):

   | 配置 | 语义 | 2.10 | 与 2.6 sim 对比 |
   |------|------|------|----------------|
   | config.json | 全局 U8/S8 | ✅ | 16/16(等价性+交叉实验) |
   | config_4w4f_all.json | 全局 4bit | ✅ | 基线 5 项一致(vs 金标准) |
   | config_16f.json | U8 + U16 region | ✅ | 16/16 全项一致 |
   | config_4w4f.json | U8 + U4 region | ✅ | 除 act_q 外一致;差异源是 **2.6 管线 bug**(见下) |
   | config_fp32.json | U16 + FP32 region | ✅(修 2 处) | 基线 5 项一致 |

   本轮新发现/新修复:
   - **2.6 管线混合 4bit 配置有 zp 污染 bug**:2.6 dynamo_export 的 optimize()
     在导出时把相等 zero_point 常量去重合并,simplify 按名字标 4bit 时把共享
     zp 的 54 个 U8 激活一并误标成 U4(quant_utils docstring 早有"分不开"的
     警告);2.10(optimize=False,标记先于去重)精确命中配置意图的 8 条边,
     与 16f 的 U16 足迹完全重合 → **2.10 行为才是对的,历史 4w4f 混合精度
     产物需重新审视**;
   - FP32 区域两连修(都进 train.py/utils_2_10):①未量化 conv+BN 不被 convert
     折叠,残留训练态 BN → convert 后统一 move_exported_model_to_eval(两环境
     同改);②零 bias 以 Expand(CastLike(标量)) 残链形式出现且无法常量折叠 →
     utils_2_10/quant_utils 新增 _castlike_to_cast pass;
   - checker R4b 对 FP32 区域 conv(共享上游 DQ 输出、权重故意不量化)会
     两侧对称误报,属已知例外,人工确认即可;
   - regional module_names(conv2d_1..5/add__5/add__9)在 2.10 图中命名不变,
     regional 混合精度路径无需改配置。

**P2 新增迁移发现(fake 阶段,均已修进 resnet50_2_10/train.py 的 capture())**:

- **动态 batch 三连**:2.10 `torch.export.export` ①把 example 的 batch 烙成
  静态 guard(训练换 batch 报 `Guard failed: x.size()[0]==1`,2.6 不检查)→
  须 `dynamic_shapes=({0: Dim("batch")},)`;②0/1 特化:动态维 example 值必须
  >=2(batch=1 样例报 ConstraintViolation)→ 捕获样例用 batch=2;③上界必须
  显式:guard 推出 `batch < 2^31/单样本元素数`(int32 限制),无上界 Dim 冲突
  → `Dim("batch", min=1, max=1024)`;
- **ORT 拒载 4bit sim 非回归**(对照实验证实):金标准 2.6 sim 修好 IR 后喂
  ORT 报同样的 `MaxPool 不接受 tensor(uint4)`——4w4f sim 本就只有 pulsar2
  能消费,ORT 数值校验应在 raw(4bit-in-8bit 容器)上做,`--ort` 不适用于
  4bit sim 产物。

### P3:其余示例批量迁移(2026-07-10 主体完成,双轨 xxx_2_10 惯例)

| 目标 | 状态 | 说明 |
|------|------|------|
| multi_stage 两个 demo | ✅ 跑通 | CIFAR-10 + resnet50_2_10 全 epoch checkpoint;**切图点自动定位**(convert 折叠会把 conv 重命名为 conv2d_106 起,按拓扑序位置索引);contrast:1≈2(94.34 全等)、3≈5(94.50 全等),mode4 按原 docstring 预期优雅跳过(per-channel 跨段不可加载) |
| reuse_conv 小对(train/test) | ✅ 跑通 | fixture 改首跑自生成、二跑逐 bit 回归通过;hack 无操作路径验证 |
| reuse_conv resnet 对 | ✅ 跑通 | fake_data 冒烟;跨 capture checkpoint 加载对齐;torch/ORT 双路评测 |
| minimum/yolov5_demo | ✅ 跑通 | grid_sample/ConvTranspose/cat/linear 全注解,checker raw 0 FAIL |
| yolov5/train_2_10.py | ✅ 已生成 | 外部 yolov5 仓库参考补丁(本仓库内不运行),4 处 QAT 行替换+伴随文件说明 |
| test_clamp 四个 demo | ✅ 跑通(2026-07-10 补) | 脚本化移植;与 2.6 原版数值输出**逐字符一致**(4/4,含 clamp 共享观察器与 relu6 融合路径);发现并修复 **hardtanh 函数残留**(optimize=False 下函数型 torchlib 算子以未内联 aten_hardtanh 存活到 sim,pulsar2 不认识 ONNX functions → dynamo_export 增加 onnx.inliner 内联,sim 恢复标准 Clip,与 2.6 形态一致);pt2e_bn_patch.py 无引用不迁 |

utils_2_10 本阶段增补:`extract.py`(原样拷入,纯 fx 手术);
**`remove_reused_bn_param_hack` 2.10 实现**(source_fn_stack 没了 → 按
num_batches_tracked buffer 名穿透链式 add_ 识别,语义等价);
`train_one_epoch` 修掉 `top1.global_avg` 潜在 bug(2.6 同有,真 ImageNet
跑不完 epoch 故未触发,fake 小数据集会踩);`onnx_simplify` 回写 ir_version=10。

顺带发现的原版问题(移植中已修,原件未动):`AXQuantizer()` 无参调用在现行
签名(config_file 必填)下即报错(yolov5_demo/multi_stage_demo/reuse_conv
train/yolov5 train 四处,均为配置重构前的旧写法);train_resnet/resnet50
train 的 `dynamo_export(model, (example_inputs,))` 双层 tuple 笔误。

### P4:收尾(2026-07-10 完成)

1. ✅ 依赖清单:新建 **requirements_2_10.txt**(双轨:requirements.txt 保留为
   2.6 基线),含版本注释与清华源提示;env.md 复装命令同步更新;
2. ✅ **CONFIG.md 追加「torch 2.10 补充」**:捕获入口换 torch.export.export;
   捕获图节点命名与 2.6 一致、既有 module_names 无需修改;⚠️ 名字必须在
   prepare 前的捕获图上找(convert 后 conv 被折叠重命名);convert 后定位
   用拓扑序位置索引(参考 multi_stage find_stage_cuts);
3. ✅ env.md / env_check/README.md 已随各阶段持续补记;
4. ✖ 分支提交:按用户指示**不提交**(2026-07-10),全部改动以工作区形态保留;
   他人的 4w4f WIP(resnet50/train.py、utils/train_utils.py、test_clamp/、
   config_4w4f_all.json)未被本项目触碰。

遗留清单(全部为可选/待用户发话):
- test_clamp 四个 demo 的 2_10 移植(机械操作,torchao 开关 API 已验证);
- simplify 共享 zp 防御 → 开回 optimize=True(onnxscript 0.6.2 折叠已修);**设计已定稿并记录于 env_check/README.md 的 Backlog 小节**(实现步骤/完成判据/触发条件齐备,按需执行);
- ImageNet 训练集到位后的绝对精度验收(CIFAR-10 等价性已闭环)。

## 五、风险与预案

1. **torchao testing 注解器语义与仓库定制版不同,不可替代**(已实测确认,
   详见改动 5:conv 融合单元划分、relu6 支持、专属注解器、regional 改写路径
   均有差异)→ 已定案:`utils_2.10/` 自持仓库版注解器,只换类型依赖,
   不采用 torchao testing 实现;torchao 侧只依赖公开的 quantizer 基类、
   observer/fake_quantize 与 pt2e 工具函数;
2. **自定义 torchlib 注册是旧机制**,onnxscript 后续版本可能移除 →
   预案:迁到 `torch.onnx.export(..., custom_translation_table={...})`,
   P1 结束后择机做,checker 可即时验证等价性;
3. **注解静默失效**(改动 2)是最隐蔽的坑,精度掉了才发现就晚了 →
   预案:P0 验收里强制"注解计数断言",并把它加进 checker 或 gen 脚本;
4. **金标准 sim 基线是 4w4f 流**,与 8bit 主流程不同源 →
   对比时注意配对:8bit raw 对 `resnet50_qat.profile.json`,
   4w4f sim 对 `resnet50_qat_sim.profile.json`;
5. 远程 pip 直连 PyPI 挂死 → 一律清华源(见 env.md)。

## 六、参考

- `env.md`:环境版本与选型依据、PT2E 可用性实测;
- `env_check/README.md`:结构检查器用法与 3 个已知问题详情;
- `env_check/qax_probe*.py`:各 API 行为对照实验(可复跑);
- 姊妹项目 `~/engine/heqi/ultralytics`(torch2.6 版 AXQuantizer 经验)。
