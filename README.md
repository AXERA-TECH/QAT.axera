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

# test
cd QAT.axera
python -m resnet50.test
```

# qat训练精度配置
修改resnet50下config.json，json包含global_config和regional_configs两个配置

- 1.单精度qat(如8w8f/4w8f)

只需配置global_config的weight字段的qmax和qmin即可(8w8f:qmax=127,qmin=-127 或者 4w8f:qmax=7,qmin=-7)，并将regional_configs中的module_names字段设置为空

- 2.混合精度qat(8w8f+4w8f)

首先通过global_config的weight字段设置全局精度(如4w8f),再通过regional_configs的module_names字段添加某些需要单独设置module的精度的node，并通过regional_configs的weight字段设置其精度(如8w8f),module_names需要手动打印查询,例如在[train](./resnet50/train.py)中的第64行exported_model = torch.export.export_for_training(float_model, example_inputs).module()后添加print(exported_model.graph)查找需要配置的算子的node.name

示例中的resnet50/config.json配置为(4w8f+8w8f)的混合精度训练，全局设置为4w8f，并将"conv2d_1", "conv2d_2", "conv2d_3", "conv2d_4", "conv2d_5"单独设置为8w8f


## Validate on board

[请点击查看上板测试文档。](pulsar2/README.md)
