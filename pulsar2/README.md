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
| QAT 模型 <br> ONNX Runtime 推理 | 50000 | 72.638 | 91.378 |
| Compiled AXModel 模型 <br> AXEngine 推理 | 50000 | 72.634 | 91.324 |

## ONNX (4w8f) 编译使用说明

考虑到 PyTorch 还不支持正式的 int4 weight 格式，用户从 QAT 训练并导出 ONNX 模型，实际上仍然使用 int8 表示，不过位宽已经限制在 int4 范围，首先需要执行如下命令将 ONNX 模型转成真正的 int4 的 ONNX 模型 (没错, ONNX 已经支持了 int4 格式. )

将 weights 修改成 int4 格式:

```bash
onnxslim model_qat_4w8f.onnx model_qat_4w8f_slim.onnx
python convert_onnx_to_4w8f.py --input model_qat_4w8f_slim.onnx --output model_qat_4w8f_opt.onnx
```

使用如下命令编译即可：

```bash
pulsar2 build --config config.json --input model_qat_4w8f_opt.onnx --output_dir outputs --target_hardware AX620E
```
