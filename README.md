# QAT.axera

axera QAT demo
包含一个最小导出 demo 和一个 resnet50 训练 demo

## torch 2.10 环境

量化 `utils` 基于 **torch 2.10 + torchao**（torch 2.10 中 torch.ao 的 PT2E 已损坏，
QAT 必须走 torchao）。常用入口：`from utils.ax_quantizer import AXQuantizer`、
`from utils.train_utils import dynamo_export` 等（直连模块，无聚合包）。

### 安装依赖

**分两步装**：PyTorch 栈从官方源取（带 CUDA 构建），其余（ONNX 生态等）走清华源
——不能整份 `-i 清华源` 一把梭，那样 torch 会从清华 PyPI 镜像拉到非验证的构建。

```bash
# 1) PyTorch 栈：官方源装，选对应 CUDA（本项目验证用 cu126）
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 torchao==0.16.0 \
  --index-url https://download.pytorch.org/whl/cu126

# 2) 其余依赖：清华源（第 1 步已装的 torch 栈会自动跳过）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

requirements 里 torch 栈不绑定 `+cuXXX`，CUDA 由第 1 步的 `--index-url` 决定
（其它 CUDA 把 `cu126` 换成 `cu124` / `cpu` 等）。

## minimum export demo

```bash
python -m minimum.minimum_demo
```

## resnet50 train

```bash
# download imagenet dataset
cd QAT.axera
mkdir -p dataset/imagenet && cd dataset/imagenet
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar --no-check-certificate
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz --no-check-certificate

# download resnet50 pretrained model
cd QAT.axera
wget -O resnet50/resnet50_pretrained_float.pth https://download.pytorch.org/models/resnet50-0676ba61.pth

# train
cd QAT.axera
mkdir -p resnet50/checkpoint
python -m resnet50.train
# 4bit 量化位宽时参考 resnet50/config_4w4f 配置，并用 simplify_and_fix_4bit_dtype 替代 onnx_simplify

# test
cd QAT.axera
python -m resnet50.test
```

## Validate on board

[请点击查看上板测试文档。](pulsar2/README.md)

## 量化规范

[AXERA_QUANT_SPEC](https://axera-tech.github.io/AXERA_QUANT_SPEC) 描述 axera 支持的量化规范：规范文档、ONNX/QDQ 合规验证工具与在线算子规则站点（本仓库以 submodule 形式引入，见 `.gitmodules`）。

`AXERA_QUANT_SPEC/src/verify_qat_onnx.py` 提供检测导出的 QDQ 模型是否合规的工具：

```bash
# 克隆（含 submodule）
git clone --recurse-submodules git@github.com:AXERA-TECH/QAT.axera.git
# 或已克隆后补拉 submodule（需 SSH key）
git submodule update --init AXERA_QUANT_SPEC

# 安装验证工具依赖（onnx/rich/netron）
pip install -r AXERA_QUANT_SPEC/requirements.txt

# 结构合规检查
python AXERA_QUANT_SPEC/src/verify_qat_onnx.py -m quant_onnx.onnx
# 如需数值检查（QDQ pair / zero_point / bias scale），加 -c
python AXERA_QUANT_SPEC/src/verify_qat_onnx.py -m quant_onnx.onnx -c
```

> 与仓库自带 `utils/check_onnx_structure.py`（开发期内部体检）互补：交付前用官方 `verify_qat_onnx.py` 做对外合规验证。
