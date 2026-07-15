# QAT.axera × torch 2.10 迁移总览（README_2_10）

> 状态：**迁移 P0–P4 + 统一 API R1–R4 全部完成**（2026-07-07 ~ 07-14）。
> 2.6 原件全部未动；实现层与 demo 已统一为 `axquant` 包 + `*_axquant.py`（历史 `*_2_10` 命名已退役）。
> 详档：迁移知识库 `plan_torch210.md` · 环境依据 `env.md` · 验证记录与工具 `env_check/README.md`
> · **统一 API 规划与执行记录 `plan_unified_api.md`**。
>
> **2026-07-14 起,公共入口统一为 `axquant` 包**(`from axquant import AXQuantizer, capture, ...`,
> torch 2.6/2.10 同一份代码,零版本分支;utils_2_10 已并入 axquant 后删除)。

## 一、结论先行

torch 2.6 → 2.10 的 PT2E QAT 全流程迁移完成，等价性**四层闭环**（resnet50 实测）：

| 层面 | 证据 |
|------|------|
| 图结构 | sim 产物拓扑序 324 节点逐位置一致；5 份量化配置矩阵复测全过 |
| 量化参数 | 同一 checkpoint 两体系导出，202 个 Q/DQ 的 scale+zero_point **按值逐 bit 一致** |
| 训练行为 | 同 seed 同数据逐 batch loss 锁步（差 ≤0.044 无发散）；checkpoint 双向互通（1216 键 strict） |
| 最终精度 | CIFAR-10 全量 1 epoch：top1 94.72(2.6) vs **94.97(2.10)**，Δ0.25pt 噪声级，top5 双侧 99.96 |

## 二、快速开始

```bash
# 环境(已建好):/home/heqi/miniforge3/envs/torch2.10
# 复装依赖(内网必须走清华源,直连 PyPI 会挂死):
pip install -r requirements_2_10.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 最小示例(conv+bn+relu 全链路:注解→QAT→convert→导出→simplify)
cd /home/heqi/project-qat/QAT.axera
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<空卡> \
  python minimum/minimum_demo_axquant.py   # axquant 统一 API,2.6/2.10 同一份代码

# resnet50(双环境单脚本,--data fake|cifar10,--config 选量化配置)
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<空卡> \
  python resnet50_2_10/train.py --data cifar10 --steps 50 --seed 42 --eval-size 20

# 导出结构体检(13 条规则 + 金标准基线对比,FAIL 时退出码非 0 可接 CI)
python env_check/check_onnx_structure.py --model xxx_qat_2_10.onnx --ort
python env_check/check_onnx_structure.py --model xxx_sim.onnx --sim \
  --baseline env_check/baselines/resnet50_qat_sim.profile.json
```

## 三、双轨文件地图

| 2.6 原件（未动） | 2.10 对应物 | 备注 |
|------------------|-------------|------|
| `utils/` | **`axquant/`** | 统一 API 兼唯一合并实现(方案 B'):版本分支仅 _compat/dynamo_export/capture 三点位;utils_2_10 已删除(R4) |
| `requirements.txt` | `requirements_2_10.txt` | 版本选型依据见 env.md |
| `minimum/minimum_demo.py` | `minimum/minimum_demo_axquant.py` | 双环境同文件;sim vs sim 17/17 一致 |
| `minimum/yolov5_demo.py` | `minimum/yolov5_demo_axquant.py` | grid_sample/ConvTranspose/cat/linear 全绿 |
| `resnet50/` | **`resnet50_2_10/`** | train/test/cross_export 均为**双环境单脚本**(按 torch 版本自动切实现);5 份量化配置副本 |
| `multi_stage/*.py` | `multi_stage/*_axquant.py` | 切子图点自动定位(见"必知差异"第 7 条);3≈5 精度一分不差 |
| `reuse_conv/*.py` | `reuse_conv/*_axquant.py` | 复用 BN hack 版本无关实现;含逐 bit fixture 回归 |
| `test_clamp/*_demo.py` | `test_clamp/*_demo_axquant.py` | 4/4 与 2.6 数值输出逐字符一致 |
| `yolov5/train.py` | `yolov5/train_axquant.py` | 外部 yolov5 仓库参考补丁(不在本仓库运行,需带 axquant/ 包) |
| `CONFIG.md` | 文末「torch 2.10 补充」 | module_names 无需改;找名字须在 prepare 前的捕获图上 |
| — | `env.md` / `plan_torch210.md` / `env_check/` | 本次迁移新增的文档与工具 |

## 四、2.6 → 2.10 必知差异（速查,详见 plan_torch210.md）

1. **PT2E 必须走 torchao**：torch.ao 的 `convert_pt2e` 在 2.10 已坏（`KeyError: 'source_fn_stack'`）；
   且 torch.ao 与 torchao 的 QuantizationSpec 类型不互通，量化器需整体换命名空间（axquant 已完成）。
2. **图捕获**：`export_for_training` 废弃 → `torch.export.export`；训练场景需**动态 batch 三件套**
   （声明 `Dim("batch", min=1, max=N)`、捕获样例 batch≥2、上界必须显式），见 `resnet50_2_10/train.py::capture()`。
3. **导出必须 `optimize=False`**：新版 onnxscript 的 optimize 会折权重 DQ 链/去重 zp 常量
   （onnxscript 0.6.2 修了前者，后者仍会污染混合 4bit 标记）。代价是 raw 中间产物臃肿——
   **看图/交付一律用 `*_sim.onnx`**（sim 与 2.6 同样干净，324 节点级）。
4. **训练态 BN 模型不能直接导出**：float 参考模型用 eval 深拷贝导出；FP32 区域残留 BN 需
   convert 后 `move_exported_model_to_eval` 再导。
5. **export 图元数据换代**：`source_fn_stack`/`nn_module_stack` 没了 → 依赖它们的注解器已改写
   （avgpool2d/layernorm/groupnorm/concat → aten 直匹配；reuse BN hack → 按 num_batches_tracked 识别）。
6. **QDQ metadata 格式变了**：simplify 的 4bit 标记已改从 fx_node 提取 target；后处理需回写
   `ir_version=10`（否则 ORT/老 pulsar2 拒载）——axquant 均已内置。
7. **convert 后 conv 会被折叠重命名**（conv2d_106 起）：配置里的 module_names 要在 **prepare 前
   的捕获图**上核对（命名与 2.6 完全一致，无需改配置）；convert 后定位结构用拓扑序位置索引
   （参考 `multi_stage/multi_stage_demo_axquant.py::find_stage_cuts`）。
8. **2.10 导出的 onnx 节点名保留 fx 目标名**（如 `node_dequantize_per_channel`），与 2.6 的
   `node_QuantizeLinear_1` 风格不同——按节点名定位的下游脚本需留意；跨版本对比 onnx 要
   **按值/按消费者匹配**，不能按拓扑位置（并行分支平序不同）。

## 五、已知事项与告警

- ⚠️ **2.6 老管线的"混合 4bit"产物不要作金标准**：其 optimize 去重 + 按名标 4bit 存在 zp 污染
  bug（实测 47 个 U8 激活、18 个权重被误标 4bit）；2.10 管线无此问题（8/5 精确命中配置意图）。
- checker R4b 对 FP32 区域 conv（共享上游 DQ、权重故意不量化）会两侧**对称误报**，属已知例外。
- 4bit sim 模型 ORT 不能跑（MaxPool 不接受 uint4，金标准同款行为）——数值对齐在 raw 上做。
- 环境 onnxscript 已定向升 0.6.2（optimize 折叠修复，本项目提报）、onnx-ir 压 `<0.1.16`。

## 六、Backlog（按需执行,均已文档化）

1. **simplify 共享 zp 防御**（解锁 optimize=True）：设计定稿于 `env_check/README.md` Backlog 小节；
2. ImageNet 训练集到位后的绝对精度验收（CIFAR-10 等价性已闭环）;
3. 分支状态：迁移成果在 `torch2.10` 分支（00debca），统一 API 在
   `feat/unified-quant-api` 分支（R1–R4 共 6 个提交）；均**未 push**，推送等用户指令。
