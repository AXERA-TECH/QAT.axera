# 支持复用权重的卷积算子

目前只支持 Conv-BN-Relu 结构，暂不支持单独 Conv 重复调用

需要手改源码
1. 在 https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/pt2e/qat_utils.py#L726 下添加 <br>
（torch 2.6.0 版本的话在 #747 行）
```
if original_node in all_original_to_replacement_nodes:
    continue
```

需要配置
1. AXQuantizer 的 annotate_bias 需要置为 False <br>
`quantizer = AXQuantizer(annotate_bias=False)`
2. convert_pt2e 之前需要先执行 remove_reused_bn_param_hack <br>
`remove_reused_bn_param_hack(prepared_model)`

---

运行 <br>
python3 -m reuse_conv.train <br>
python3 -m reuse_conv.test <br>
该 demo 实现权重复用模型结构的训练和导出，以及重新加载权重；加载权重后的推理结果和训练结束保存时的结果一致。 <br>


运行 <br>
python3 -m reuse_conv.train_resnet <br>
python3 -m reuse_conv.test_resnet <br>
该 demo 在 Resnet50 头上强行加了一组复用权重的 Conv 结构，证明复用权重结构在 QAT 流程下可训练，且保存的权重能够正常重新加载 <br>