import copy
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock
from typing import Any, Callable, List, Optional, Type, Union

from torch.ao.quantization.quantize_pt2e import (
    prepare_qat_pt2e,
    convert_pt2e,
)
from utils.ax_quantizer import(
    load_config,
    AXQuantizer,
)
from utils.train_utils import (
    load_model,
    imagenet_data_loaders,
    dynamo_export,
    onnx_simplify,
    evaluate,
)
from utils.extract import extract_subgraph
from IPython import embed


if __name__ == "__main__":
    # 预训练权重
    model_file = "./resnet50/resnet50_pretrained_float.pth"
    # 数据集
    data_loader, data_loader_test = imagenet_data_loaders("dataset/imagenet/")
    example_inputs = (torch.rand(1, 3, 224, 224).to("cuda"),)
    # quantizer
    global_config, regional_configs = load_config("./resnet50/config.json")
    quantizer = AXQuantizer()
    quantizer.set_global(global_config)
    quantizer.set_regional(regional_configs)

    # float model
    model = load_model(model_file, "resnet50").to("cuda")
    # quantized model
    example_inputs = (torch.rand(1, 3, 224, 224).to("cuda"),)
    exported_model = torch.export.export_for_training(model, example_inputs).module()
    prepared_model = prepare_qat_pt2e(exported_model, quantizer)

    prepared_model.load_state_dict(torch.load("./resnet50/checkpoint/last_checkpoint.pth"))
    quantized_model = convert_pt2e(prepared_model)
    # submodule
    # 这里可能需要修改 subgraph 起止 node name
    submodule_1 = extract_subgraph(quantized_model, ["quantize_per_tensor_default"], ["dequantize_per_tensor_default_80"])
    submodule_2 = extract_subgraph(quantized_model, ["quantize_per_tensor_default_15"], ["dequantize_per_tensor_default_100"])
    submodule_3 = extract_subgraph(quantized_model, ["quantize_per_tensor_default_57"], ["dequantize_per_tensor_default_73"])

    def model3s_submodule_forward(x):
        torch.ao.quantization.move_exported_model_to_eval(submodule_1)
        torch.ao.quantization.move_exported_model_to_eval(submodule_2)
        torch.ao.quantization.move_exported_model_to_eval(submodule_3)

        x = submodule_1(x)
        x = submodule_2(x)
        x = submodule_3(x)

        return x

    # 推理前 100 个数据，快速对比结果；要推理完整测试集设置 total_size=None
    top1, top5 = evaluate(model.eval(), data_loader_test, total_size=100)
    top1_q, top5_q = evaluate(quantized_model, data_loader_test, total_size=100)
    top1_3ss, top5_3ss = evaluate(model3s_submodule_forward, data_loader_test, total_size=100)

    # 打印
    def to_float(t):
        assert isinstance(t, torch.Tensor)
        return t.cpu().numpy().tolist()
    print(f"model1: top1:{to_float(top1.avg)}, top5:{to_float(top5.avg)}")
    print(f"model3: top1:{to_float(top1_q.avg)}, top5:{to_float(top5_q.avg)}")
    print(f"model5: top1:{to_float(top1_3ss.avg)}, top5:{to_float(top5_3ss.avg)}")

    