# 统一量化 API 规划（plan_unified_api.md，v2 — 采纳方案 B' 合并实现，2026-07-14）

> 分支：`feat/unified-quant-api`（基于 torch2.10 分支 00debca）
> 前置：P0–P4 迁移已完成且等价性四层闭环（见 README_2_10.md），本规划是其上的体验重构。

## 一、问题陈述

当前双轨结构正确但不优雅，版本差异泄漏到了用户代码里：

1. **import 要挑版本**：`from utils.ax_quantizer import ...` vs `from utils_2_10.ax_quantizer import ...`；
2. **demo 要么双份**（`xxx.py` / `xxx_2_10.py`），要么单份但充斥 `IS_210` 分支
   （resnet50_2_10 三个脚本各有一段 if/else import 块）；
3. **版本差异的处理散落在各 demo**：capture 的动态 batch 三件套、float 导出的 eval
   深拷贝、move_to_eval 的来源……每个新 demo 都要重抄一遍。

## 二、目标

- **唯一公共入口**：`from axquant import AXQuantizer, capture, prepare_qat_pt2e, ...`，
  用户代码零版本分支、零 `utils_2_10` 字样；
- 版本差异 100% 收进适配层（capture/导出/开关 API 一站式封装）;
- **语义零回退**：2.6 实现层（`utils/`）保持零改动，适配层只做路由与薄封装；
  重构完成的判据是既有全套验证（checker 矩阵 + 等价性）原样通过。

## 三、方案对比

| 方案 | 做法 | 评价 |
|------|------|------|
| **A. 顶层适配包（推荐）** | 新建包作为唯一公共 API，import 时按 torch 版本把符号路由到 utils（2.6/torch.ao）或 utils_2_10（2.10/torchao），并新增 `capture()` 等统一封装 | 用户面最干净；实现层零改动、风险最低；差异封装有归处 |
| B. 合并两套 utils | utils 内部 try-import torchao 降级 torch.ao | 改动大；2.6 已验证语义可能被波及；两套类型体系混在一个文件里更难读 |
| C. 仅做转发 shim | 一个 utils_compat.py 转发符号 | 最小改动，但 capture/导出差异仍留在 demo 里，不解决问题 2/3 |

## 四、方案 A 设计草案

```
axquant/                    # 包名已定(D1)
  __init__.py               # 公共 API 出口(下表)
  _compat.py                # 版本探测 + 符号路由(import 时一次性完成)
  capture.py                # capture(model, example_inputs, dynamic_batch=False,
                            #         batch_max=1024)
                            #   2.6 → export_for_training
                            #   2.10 → torch.export.export;dynamic_batch=True 时
                            #          自动处理 0/1 特化与显式上界三件套
  export.py                 # dynamo_export / onnx_simplify /
                            #   simplify_and_fix_4bit_dtype / export_float_reference
                            #   (后者内置 eval 深拷贝,解决训练态 BN 导出问题)
```

公共 API 清单（草案）：

| 类别 | 符号 |
|------|------|
| 量化器 | `AXQuantizer`、`load_config` |
| PT2E 流程 | `prepare_qat_pt2e`、`convert_pt2e`、`move_exported_model_to_eval/train` |
| 开关 | `disable/enable_fake_quant`、`disable/enable_observer` |
| 捕获/导出 | **`capture`（新）**、`dynamo_export`、**`export_float_reference`（新）**、`onnx_simplify`、`simplify_and_fix_4bit_dtype` |
| 图工具 | `extract_subgraph`、`remove_reused_bn_param_hack` |
| 训练辅助 | `load_model`、数据 loaders、`train_one_epoch`、`evaluate`（D3 已定:进公共 API） |

版本路由规则：`torch >= 2.10` → utils_2_10/torchao，否则 → utils/torch.ao。
官方支持点仅 2.6 与 2.10（已验证）；2.7–2.9 未验证，import 时打一条告警。

## 五、阶段划分

- **R1 包骨架**：`_compat` 路由 + `capture`/`export` 封装；
  验收 = 新版 minimum demo **单文件**在两个环境跑通（同一份代码零分支），
  产物过 checker 且与既有 `_2_10`/2.6 产物结构一致；
- **R2 resnet50 收敛**：train/test/cross_export 切统一 API、删 `IS_210` 块；
  验收 = 全配置矩阵（`cmp_sim_matrix.py`）+ CIFAR-10 等价性冒烟原样通过；
- **R3 其余 demo 收敛**：multi_stage / reuse_conv / test_clamp / yolov5；
  demo 与 2.6 原件的合并/退役策略见决策点 D2；
- **R4 文档**：README_2_10 与 CONFIG.md 示例改统一 API；requirements 说明不变。

## 六、已定决策（2026-07-10 与用户共同拍板）

- **D1 包名 = `axquant`**；
- **D2 = 统一 demo 替代 `xxx_2_10.py`（删后缀版），2.6 原件继续原样保留**——
  最终目录形态：每个 demo 一份统一版（无后缀新名或顶替 _2_10 位置）+ 一份
  2.6 原件；R3 收敛完成后 `*_2_10.py` demo 全部删除（utils_2_10 作为实现层保留）；
- **D3 = 训练辅助进公共 API**（load_model/数据 loaders/train_one_epoch/evaluate
  一并从 axquant 出口，demo 零 utils_2_10 直接 import；这些函数本身无版本差异，
  承诺成本低）；
- **D4 = 不留逃生门**，import 时纯按 torch 版本一次性路由（torch.ao 的 PT2E 在
  2.10 已坏，强制切换无实际意义）。

## 七、风险与约束

- 适配层只 re-export + 薄封装，**不改 utils/utils_2_10 的任何实现**——等价性
  结论不失效，回归只需验"路由正确 + 封装正确"；
- `capture(dynamic_batch=...)` 的默认值取 False（与 2.6 行为一致），训练场景
  显式开——避免隐式行为差异；
- 回归成本集中在 R2（矩阵 + 冒烟，约半小时机器时间）。

## 八、v2 修订：采纳方案 B'（合并实现，2026-07-14 与用户确认）

**变更**：axquant 不再是"路由到两套 utils 的壳"（方案 A），而是**唯一的合并
实现**——版本分支下沉到实现内部的三个最小点位，其余全部单份代码：

1. `_compat.py`：唯一的条件 import 块（torch>=2.10 → torchao，否则 torch.ao；
   符号改名在此对齐，如 WrapperModule/annotate_*_qspec_map/DerivedObserver*；
   另提供 `get_aten_graph_module_for_pattern()` 包装吃掉 2.6 的
   using_training_ir 参数差异）；
2. `train_utils.dynamo_export` 内部 if（2.6：export+optimize()；
   2.10：optimize=False + onnx.inliner 内联 + 域清理）；
3. `capture()` 内部 if（2.6：export_for_training；2.10：export，
   dynamic_batch=True 时自动处理 0/1 特化与显式上界）。

依据（移植史盘点）：两套 utils 的差异 ≈95% 是 import；4 个重写注解器
（aten 直匹配）、hack（按 num_batches_tracked）、metadata 双格式解析、
ir 回写/CastLike/内联（2.6 下天然 no-op）均为**双版本通用**。

**终局**：R3 收敛后 `utils_2_10/` 整个删除；`utils/`（2.6 原件）保留但
退役为上游对照物，不再被任何 demo 引用。

**R1 重定义**：`_compat` + 核心文件（ax_quantizer / ax_quantizer_utils /
train_utils / quant_utils / extract / 自定义 per-channel 映射,单份合并版）
+ `capture.py` + `__init__.py`（公共 API 出口,包 import 即自动完成
per-channel 映射注册）;验收 = 统一版 minimum demo **同一份文件**在
torch2.6 与 torch2.10 两个环境跑通,产物过 checker 且与既有 2.6 金标准 /
_2_10 产物结构一致。

**新增约束**：merged 实现的任何改动,双环境回归为必选项(安全网 =
env_check 矩阵 + 等价性 harness)。

## 九、执行记录

- **R1 ✅**(b197375):axquant 包 9 文件;minimum 单文件双环境跑通,
  2.6 raw 13/13、sim vs sim 17/17;
- **R2 ✅**(c006f36):resnet50 三脚本收敛;双环境矩阵+CIFAR 冒烟,
  eval 与重构前逐数字一致(56.562/96.250 与 53.750/93.281);
- **R3 ✅**(1b200aa):12 个统一 demo(_axquant)落地,13 个 *_2_10.py 删除;
  2.10 全量 12 项 + 2.6 抽查 5 项通过(2 个失败裁定为 fixture 并发竞态,
  串行重跑干净通过);
- **R4 ✅**(2026-07-14):env_check/gen_candidate 收敛 axquant;
  **utils_2_10/ 删除(B' 终局达成)**;README_2_10/CONFIG/env_check README
  更新;删除后回归(gen_candidate+checker+minimum 双环境)通过。
  utils/(2.6 原件)保留为上游对照物,已无任何 demo 引用。
- **R5 ✅**(2026-07-14,用户指示):包更名 axquant → **utils**,与上游目录
  保持一致;旧 2.6 utils 全部清理(其 WIP fake_data 改动已含于合并实现)。
  附带收益:合并实现模块名与原 utils 同构,2.6 原版 demo 的
  `from utils.ax_quantizer import ...` 写法对新 utils 天然兼容(已实测);
  yolov5 外部补丁改用 qat_utils 拷贝名防与 yolov5 自带 utils 撞名。
- **R6 ✅**(2026-07-14,用户指示,目录全面对齐上游
  https://github.com/AXERA-TECH/QAT.axera):原始 resnet50/ 删除,
  resnet50_2_10 → resnet50(预训练 pth/金标准 onnx/他人 config_4w4f_all
  已迁入,旧 checkpoint 归档 checkpoint_legacy0623/);全部 *_axquant demo
  更名为上游原名并删除对应 2.6 原件(train_resnet.py 类定义内联);
  minimum/yolov5 产物名回归上游;test_clamp 例外(非上游内容,
  无后缀原件为未跟踪 WIP 未动,统一版保留 _axquant 名)。
  终态:上游文件名 × 双版本统一内容。
