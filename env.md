pi# torch 2.10 环境说明（env.md）

> 环境路径：`/home/heqi/miniforge3/envs/torch2.10`（Python 3.12.0）
> 用途：QAT.axera 升级到 torch 2.10 的开发/训练/导出环境（分支 `torch2.10`）
> 整理日期：2026-07-07

## 一、已安装版本

| 包 | 版本 | 来源 |
|----|------|------|
| torch | 2.10.0+cu126 | pytorch 官方 cu126 渠道（建环境时装好） |
| torchvision | 0.25.0+cu126 | 同上 |
| torchaudio | 2.10.0+cu126 | 同上 |
| torchao | 0.16.0+cu126 | 同上（与 torch 2.10 配套的 cu126 构建） |
| numpy | 2.4.4 | 随 torch 装入 |
| **onnx** | **1.19.1** | PyPI（清华源） |
| **onnxscript** | **0.6.2**（2026-07-10 自 0.5.4 升级，见下） | PyPI（清华源） |
| **onnx-ir** | **0.1.15**（约束 `<0.1.16`，用户指定；曾被 pip 连带升到 0.2.1 后压回，pip check 无冲突） | PyPI（清华源） |
| onnxruntime | 1.23.2 | PyPI（清华源） |
| onnx-graphsurgeon | 0.6.1 | PyPI（清华源） |
| onnxslim | 0.1.94 | PyPI（清华源） |
| tqdm / pyyaml / ipython | 4.68.4 / 6.0.3 / 9.15.0 | PyPI（清华源） |

## 二、版本选择依据

### onnx==1.19.1 / onnxscript==0.5.4 / onnx-ir==0.1.12（核心依据）

**与 torch 2.10 官方 CI 完全一致。** PyTorch `release/2.10` 分支的
`.ci/docker/requirements-ci.txt` 明确 pin：

```
onnx==1.19.1 ; python_version < "3.14"   ← 本环境 py3.12，命中此条
onnxscript==0.5.4
onnx-ir==0.1.12
```

即 torch 2.10 的 ONNX 导出器（dynamo exporter，2.10 起 `dynamo=True` 为默认）
就是在这组版本上测试发布的，跟随 CI pin 风险最小。
不追最新（onnx 1.22 / onnxscript 0.7.1）的原因：onnxscript/onnx-ir 是 dynamo
导出器的直接依赖，跨大版本组合未经 torch 2.10 CI 验证；旧仓库在 torch 2.6 时代
也吃过 onnx≥1.20 生态不兼容的亏（姊妹项目 ultralytics 锁 `onnx<1.20`）。

**2026-07-10 更新：onnxscript 定向升到 0.6.2 + onnx-ir 压在 0.1.15**（`<0.1.16`，
用户指定），偏离 CI pin 的理由：0.6.2 修复了 optimize 常量折叠吃掉「int8 权重
+ DQ」链的问题（该修复由本项目负责人上游提报）。该组合下三项实测全过：
①现行 optimize=False 管线回归全绿；②optimize=True 下 54/54 权重 DQ 保留；
③⚠️ 但 optimize 的 initializer 去重仍会触发混合 4bit 标记污染（U4=55 vs
本意 8，与 onnx-ir 版本无关），`dynamo_export` 维持 optimize=False，
决策记录见 env_check/README.md。

### onnxruntime==1.23.2

- torch 2.10 dynamo 导出器实际写出 **opset 20 / IR 10**（实测，见下），
  这是 2024 年水平的格式，近两年任何 ORT 都能读；
- 选 1.23.2 是因为它与 onnx 1.19 同期发布（2025Q4），官方支持矩阵覆盖 opset≤24，
  且为该系列末位补丁版，比追 1.27 最新版更稳；
- 装 CPU 版（与旧 requirements.txt 惯例一致）：ORT 只用来做导出模型的数值
  对齐校验，训练/验证都在 torch/CUDA 侧，CPU 版可避免与 torch 自带 cuDNN 的
  版本纠缠。

### torchao==0.16.0+cu126（关键，不要卸/换）

torch 2.10 中 `torch.ao.quantization` 的 PT2E 已弃用且**实际已坏**（见验证三），
QAT 流程必须走 torchao。0.16.0+cu126 是 pytorch 渠道与 torch 2.10 配套发布的
构建，优先于 PyPI 通用版（0.17.0）。

### 其余包

`onnx-graphsurgeon`、`onnxslim`、`tqdm`、`pyyaml`、`ipython` 为仓库脚本实际
import 的依赖（全仓 grep 确认），无强版本耦合，取安装时最新即可。

## 三、环境验证结果（2026-07-07 实测）

1. **CUDA**：`torch.cuda.is_available()=True`，A100-SXM4-80GB ×4，matmul 正常。
2. **float 模型 dynamo 导出**：`torch.onnx.export(..., dynamo=True)` →
   opset 20 / IR 10，`onnx.checker` 通过，ORT 推理与 torch 输出
   max|diff| ≈ 2.2e-07（float 舍入级，通过）。
3. **torch.ao PT2E QAT：已坏 ❌**
   `prepare_qat_pt2e` → `convert_pt2e` 在 conv+BN 折叠处报
   `KeyError: 'source_fn_stack'`（`torch/ao/quantization/pt2e/qat_utils.py`
   `_fold_conv_bn_qat`），且 `export_for_training` / `torch.export.export` /
   `strict=False` 三种导出入口结果相同 → 非用法问题，是 torch.ao 弃用后 bitrot。
4. **torchao PT2E QAT：可用 ✅**
   `torchao.quantization.pt2e.quantize_pt2e.prepare_qat_pt2e/convert_pt2e`
   完整闭环（prepare → 训练数步 → convert → 推理）通过，图中出现
   `quantize_per_tensor/dequantize_per_tensor`。
5. **QDQ 模型 ONNX 导出**：torchao convert 后的模型经
   `move_exported_model_to_eval` + dynamo 导出成功，图含
   `QuantizeLinear/DequantizeLinear`，ORT 与 torch 数值 **max|diff| = 0.0**。
6. **跨库不兼容（迁移要点）**：torch.ao 与 torchao 的 `QuantizationSpec` /
   `Quantizer` 是两套独立类型，混用直接 `AssertionError` →
   `utils/ax_quantizer.py` 等必须**整体**迁到 `torchao.quantization.pt2e.*`
   命名空间，不能只换 prepare/convert 入口。

## 四、安装渠道备注

- 直连默认 PyPI 时 pip 会**无限挂死**（进程 10 分钟无任何 socket/磁盘 IO），
  本机必须走清华源：
  ```bash
  /home/heqi/miniforge3/envs/torch2.10/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <pkg>
  ```
- 依赖清单已固化为仓库根目录 **requirements_2_10.txt**（双轨：requirements.txt 保留为 2.6 基线），安装记得走清华源；
- 复装一键命令（与 requirements_2_10.txt 同步，2026-07-10 更新为 onnxscript 0.6.2 组合）：
  ```bash
  /home/heqi/miniforge3/envs/torch2.10/bin/pip install --no-input \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    onnx==1.19.1 onnxscript==0.6.2 onnx-ir==0.1.15 onnxruntime==1.23.2 \
    onnx-graphsurgeon==0.6.1 onnxslim==0.1.94 tqdm pyyaml ipython
  ```
- **磁盘配额**（/home 为 NFS 配额盘，df 显示卷有空闲但用户配额会先耗尽，git 报
  No space left）：优先清理**可再生**产物——`resnet50/resnet50_float_*.onnx`
  （每个 130MB）、`resnet50/checkpoint/checkpoint_*_0.pth` 与带配置后缀的
  last_checkpoint、cross/opt onnx、`reuse_conv/resnet50*.pth|*.onnx`、
  `env_check/out/`、`__pycache__`。**必须保留**：`resnet50_pretrained_float.pth`、
  金标准 `resnet50_qat[_sim].onnx`、`last_checkpoint_2_6/2_10.pth`
  （等价性与 multi_stage 依赖）、`*.eq50bak.pth`、`reuse_conv/input_ax.npy|gt_ax.npy`
  （回归锚点）、`dataset/`。
- 验证脚本已归档到仓库 `env_check/`：`qax_smoke.py`（版本/CUDA/torch.ao 路径）、
  `qax_probe.py`（三种 export 入口对照）、`qax_probe2.py`（torchao vs torch.ao quantizer）、
  `qax_smoke2.py`（导出 + ORT 对齐）。
