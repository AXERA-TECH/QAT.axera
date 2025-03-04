# Validate on board

这个部分主要包含经过 QAT 训练出来的模型在 `pulsar2` 工具链上的使用和精度数据。

## Compiling with pulsar2

`pulsar2` 从 3.4 版本开始支持 QDQ ONNX 格式模型的编译，请确认您当前使用的版本等于或大于 3.4 版本。

```bash
root@ai-dev1:/data# pulsar2 version
version: 3.4
commit: 3dfd5692
```

使用如下命令编译：

```bash
pulsar2 build --config config.json --input resnet50_qat_sim.onnx --output_dir outputs
```

## 精度数据

现在我们编译出 compiled axmodel，路径位于 "./outputs/compiled.axmodel", 使用该模型，上板实测 Imagenet 验证集的精度数据如下：

| 模型 | 验证集规模 | top1 | top5 |
| --- | --- | --- | --- |
| 浮点模型 | 50000 | 72.638 | 91.378 |
| QAT 模型 | 50000 | 72.634 | 91.324 |
