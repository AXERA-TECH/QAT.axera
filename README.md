# QAT.axera

axera QAT demo
包含一个最小导出 demo 和一个 resnet50 训练 demo

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

[AXERA_QUANT_SPEC](https://axera-tech.github.io/AXERA_QUANT_SPEC) 描述 axera 支持的量化规范.

`AXERA_QUANT_SPEC/src/verify_qat_onnx.py` 提供了检测导出的 `quant_onnx` 是否合规的工具，使用方法如下

```
git clone --recurse-submodules
python AXERA_QUANT_SPEC/src/verify_qat_onnx.py -m quant_onnx.onnx
# 如果需要做数值检查，则加上-c
python AXERA_QUANT_SPEC/src/verify_qat_onnx.py -m quant_onnx.onnx -c 
```


