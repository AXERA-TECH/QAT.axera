# QAT.axera

axera QAT demo
包含一个最小导出 demo 和一个 resnet50 训练 demo

## torch 2.10 支持

量化 `utils` 已支持 **torch 2.6 / 2.10 双版本**——同一份代码两个版本都能跑,零版本分支
(版本差异收敛在 `utils/_compat.py` 内部)。

**torch 2.10 用户请先看 [README_2_10.md](README_2_10.md)**,内含:

- **环境安装**:两步装法(PyTorch 栈走官方源定 CUDA + 其余走镜像),清单 `requirements_2_10.txt`;
- **快速开始**:最小示例 / resnet50 / 导出结构体检的命令;
- **2.6 → 2.10 必知差异(8 条)**:PT2E 必须走 torchao等；
- **已知事项与告警**等。

torch 2.6 用户按下文原有流程即可(依赖见 `requirements.txt`)。

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
