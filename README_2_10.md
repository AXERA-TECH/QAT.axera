# QAT.axera 量化 utils(torch 2.6 / 2.10)—— 使用指导

> 本仓库的量化 `utils` 是 **torch 2.6 / 2.10 双版本统一实现**:同一份代码两个版本都能跑,
> 零版本分支,版本差异收敛在 `utils/_compat.py` 内部。
> 公共入口:`from utils import AXQuantizer, capture, prepare_qat_pt2e, convert_pt2e, ...`
> (与上游目录同名;上游写法 `from utils.ax_quantizer import ...` 亦兼容)。
> 环境版本与选型依据见 `env.md`;结构检查工具见 `env_check/README.md`。

## 一、快速开始

### 安装依赖

torch 2.10 用 `requirements_2_10.txt`(torch 2.6 基线用 `requirements.txt`)。
内网机器直连 PyPI 会挂死,安装走清华源:

```bash
pip install -r requirements_2_10.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

requirements 只钉版本号、不绑定 CUDA 构建。需要指定 CUDA(如 cu126)时,加 PyTorch 官方源:

```bash
pip install -r requirements_2_10.txt \
  --index-url https://download.pytorch.org/whl/cu126 \
  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 跑示例

在仓库根目录执行(torch 2.6 / 2.10 任一环境,同一份代码):

```bash
# 最小示例:conv+bn+relu 全链路(注解 → QAT → convert → 导出 → simplify)
PYTHONPATH=. python minimum/minimum_demo.py

# resnet50 单脚本双环境(--data fake|cifar10,--config 选量化配置)
PYTHONPATH=. python resnet50/train.py --data cifar10 --steps 50 --seed 42

# 导出结构体检(13 条规则 + 金标准基线对比;FAIL 时退出码非 0,可接 CI)
python env_check/check_onnx_structure.py --model <raw>.onnx --ort
python env_check/check_onnx_structure.py --model <sim>.onnx --sim \
  --baseline env_check/baselines/resnet50_qat_sim.profile.json
```

多卡机器用 `CUDA_VISIBLE_DEVICES=<空卡>` 挑卡。更多入口、动态 shape 写法与防坑清单见
skill `qat-run`;第一次给新模型做量化见 skill `qat-new-model`。

## 二、文件地图

| 位置 | 说明 |
|------|------|
| `utils/` | 统一量化 API 兼实现层:`AXQuantizer`、`capture`、`prepare_qat_pt2e`、`convert_pt2e`、`dynamo_export`、`simplify_and_fix_4bit_dtype`、训练辅助等;版本差异只在 `_compat` / `dynamo_export` / `capture` 三个点位 |
| `minimum/`(minimum_demo.py、yolov5_demo.py) | 最小示例(conv-bn-relu / 多输入 grid_sample) |
| `resnet50/` | train / test / cross_export + 5 份量化配置 + 预训练 pth + 金标准 onnx(产物带 `_2_6/_2_10` 标签,用于双环境等价性对照) |
| `multi_stage/`、`reuse_conv/`、`yolov5/` | 其余示例 demo |
| `test_clamp/` | clamp / relu6 数值对照 demo(CPU 即可) |
| `env_check/` | 结构检查器 `check_onnx_structure.py` + 金标准基线 `baselines/` + 说明 |
| `.claude/skills/` | 项目技能:qat-new-model(新模型接入)/ qat-run(跑训练与 demo)/ qat-check(结构体检)/ qat-migrate-2_10(把 torch.ao PT2E 项目迁 2.10) |
| `requirements_2_10.txt` / `requirements.txt` | torch 2.10 / 2.6 依赖 |
| `env.md` | 环境版本与选型依据 |

## 三、2.6 → 2.10 必知差异(速查)

1. **PT2E 必须走 torchao**:torch.ao 的 `convert_pt2e` 在 2.10 已坏(`KeyError: 'source_fn_stack'`);
   且 torch.ao 与 torchao 的 `QuantizationSpec` 类型不互通,量化器需整体换命名空间(utils 已内置)。
2. **图捕获**:`export_for_training` 废弃 → `torch.export.export`。`utils/capture.py` 的
   `capture(model, example_inputs, dynamic_shapes=...)`:`dynamic_shapes` 为 torch.export 原生
   语义原样透传(batch/H/W/任意维;下采样整除性 guard 用派生维 `k*_dim` 表达,报
   ConstraintViolation 时照抄报错里的 Suggested fixes;0/1 特化 example 由 capture 自动处理)。
3. **导出必须 `optimize=False`**:新版 onnxscript 的 optimize 会折权重 DQ 链 / 去重 zp 常量
   (onnxscript 0.6.2 修了前者,后者仍会污染混合 4bit 标记)。代价是 raw 中间产物臃肿——
   **看图 / 交付一律用 `*_sim.onnx`**(sim 与 2.6 同样干净)。
4. **训练态 BN 模型不能直接导出**:float 参考模型用 eval 深拷贝导出(`export_float_reference`);
   FP32 区域残留 BN 需 convert 后 `move_exported_model_to_eval` 再导。
5. **export 图元数据换代**:`source_fn_stack` / `nn_module_stack` 没了 → 依赖它们的注解器已改写为
   aten 直匹配(avgpool2d/layernorm/groupnorm/concat);reuse BN hack 改按 `num_batches_tracked` 识别。
6. **QDQ metadata 格式变了**:simplify 的 4bit 标记改从 `fx_node` 提取 target;后处理需回写
   `ir_version=10`(否则 ORT / 老 pulsar2 拒载)——utils 均已内置。
7. **convert 后 conv 会被折叠重命名**(conv2d_106 起):配置里的 `module_names` 要在
   **prepare 之前的捕获图**上核对(命名与 2.6 完全一致,无需改配置);convert 后定位结构用
   拓扑序位置索引(参考 `multi_stage/multi_stage_demo.py::find_stage_cuts`)。
8. **2.10 导出的 onnx 节点名保留 fx 目标名**(如 `node_dequantize_per_channel`),与 2.6 的
   `node_QuantizeLinear_1` 风格不同——按节点名定位的下游脚本需留意;跨版本对比 onnx 要
   **按值 / 按消费者匹配**,不能按拓扑位置(并行分支平序不同)。

## 四、已知事项与告警

- ⚠️ **2.6 老管线的"混合 4bit"产物不要作金标准**:其 optimize 去重 + 按名标 4bit 存在 zp 污染
  bug(实测 47 个 U8 激活、18 个权重被误标 4bit);2.10 管线无此问题(精确命中配置意图)。
- **bias 量化覆盖面**:现状仅 conv1d/2d 默认 int32 派生量化(per-channel、scale=Sa×Sw),
  Linear / ConvTranspose 不量化,config 无 bias 通道——是否应全部量化以内部后端团队核对为准。
- checker R4b 对 FP32 区域 conv(共享上游 DQ、权重故意不量化)会两侧**对称误报**,属已知例外。
- 4bit sim 模型 ORT 不能跑(MaxPool 不接受 uint4,金标准同款行为)——数值对齐在 raw 上做。
- 环境 onnxscript 定向升 0.6.2(optimize 折叠修复,本项目提报)、onnx-ir 约束 `<0.1.16`,详见 env.md。

## 五、相关

- 把别的「torch.ao PT2E + 自定义 Quantizer」项目迁到 2.10:skill `qat-migrate-2_10`。
- 迁移过程记录、双环境等价性证据、Backlog 等归档于内部规划文档(不随项目上传)。
