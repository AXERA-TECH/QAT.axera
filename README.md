# QAT.axera

axera QAT demo
包含一个最小导出 demo 和一个 resnet50 训练 demo

## 量化规范

[AXERA_QUANT_SPEC](https://github.com/AXERA-TECH/AXERA_QUANT_SPEC) 描述 axera 支持的量化规范，以 submodule 形式引入本仓库（`AXERA_QUANT_SPEC/`），包含规范文档、ONNX/QDQ 合规验证工具与在线算子规则站点。

克隆本仓库时请使用 `git clone --recurse-submodules`，或在克隆后执行 `git submodule update --init` 拉取该仓库。

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
