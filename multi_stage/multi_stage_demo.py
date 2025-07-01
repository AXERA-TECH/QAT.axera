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


class ThreeStageResNet(ResNet):
    """
    模仿 ResNet 定义一个推理分成三个阶段的 3S ResNet
    这里的 _forward_impl_n 组合起来与原本 ResNet 的 _forward_impl 等价
    """
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(ThreeStageResNet, self).__init__(block=block, layers=layers)

    def _forward_impl_stage1(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        return x
    
    def _forward_impl_stage2(self, x: Tensor) -> Tensor:
        x = self.layer2(x)
        x = self.layer3(x)

        return x
    
    def _forward_impl_stage3(self, x: Tensor) -> Tensor:
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward1(self, x: Tensor) -> Tensor:
        return self._forward_impl_stage1(x)

    def forward2(self, x: Tensor) -> Tensor:
        return self._forward_impl_stage2(x)

    def forward3(self, x: Tensor) -> Tensor:
        return self._forward_impl_stage3(x)


if __name__ == "__main__":
    """
    在主函数中，会分别进行五种模型的推理，分别是
    1. 原始完整浮点模型
    2. multi stage 浮点模型
    3. 完整量化模型
    4. 由 multi stage 浮点模型每个 stage 分别独立加载参数的量化模型
    5. 由完整量化模型切多个子图再进行分 stage 推理的量化模型
    """
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


    """
    接下来准备五种模型
    """
    # 准备 1. 原始浮点模型
    model = load_model(model_file, "resnet50").to("cuda")

    # 准备 2. multi stage 浮点模型
    model3s = ThreeStageResNet(Bottleneck, [3, 4, 6, 3])  # ResNet50
    model3s.load_state_dict(torch.load(model_file, weights_only=True))
    model3s.to("cuda")

    model3s.forward = model3s.forward1
    stage1 = copy.deepcopy(model3s)  # float stage1
    model3s.forward = model3s.forward2
    stage2 = copy.deepcopy(model3s)  # float stage2
    model3s.forward = model3s.forward3
    stage3 = copy.deepcopy(model3s)  # float stage3

    # 准备 3. 完整量化模型
    example_inputs = (torch.rand(1, 3, 224, 224).to("cuda"),)
    exported_model = torch.export.export_for_training(model, example_inputs).module()
    prepared_model = prepare_qat_pt2e(exported_model, quantizer)

    prepared_model.load_state_dict(torch.load("./resnet50/checkpoint/last_checkpoint.pth"))
    quantized_model = convert_pt2e(prepared_model)

    # 准备 4. 由 multi stage 浮点模型每个 stage 分别独立加载参数的量化模型
    example_inputs_s1 = (torch.rand(1, 3, 224, 224).to("cuda"),)
    exported_model_s1 = torch.export.export_for_training(stage1, example_inputs_s1).module()
    prepared_model_s1 = prepare_qat_pt2e(exported_model_s1, quantizer)

    prepared_model_s1.load_state_dict(torch.load("./resnet50/checkpoint/last_checkpoint.pth"), strict=False)
    quantized_model_s1 = convert_pt2e(prepared_model_s1)  # quant stage1

    example_inputs_s2 = (torch.rand(1, 256, 56, 56).to("cuda"),)
    exported_model_s2 = torch.export.export_for_training(stage2, example_inputs_s2).module()
    prepared_model_s2 = prepare_qat_pt2e(exported_model_s2, quantizer)

    prepared_model_s2.load_state_dict(torch.load("./resnet50/checkpoint/last_checkpoint.pth"), strict=False)
    quantized_model_s2 = convert_pt2e(prepared_model_s2)  # quant stage2

    example_inputs_s3 = (torch.rand(1, 1024, 14, 14).to("cuda"),)
    exported_model_s3 = torch.export.export_for_training(stage3, example_inputs_s3).module()
    prepared_model_s3 = prepare_qat_pt2e(exported_model_s3, quantizer)

    prepared_model_s3.load_state_dict(torch.load("./resnet50/checkpoint/last_checkpoint.pth"), strict=False)
    quantized_model_s3 = convert_pt2e(prepared_model_s3)  # quant stage3

    # 准备 5. 由完整量化模型切多个子图再进行分 stage 推理的量化模型
    """
    找到和 ThreeStageResNet 一致的切分位置，可以通过以下几个可视化方

    1. 导出浮点模型：
    dynamo_export(model, example_inputs, "./tmp.onnx")
    2. 导出量化模型：
    dynamo_export(quantized_model, example_inputs, "./tmp_q.onnx")
    3. 打印 gm.graph
    print(quantized_model.graph)
    4. 打印 readable 模型
    quantized_model.print_readable()

    起始和末尾都要有完整的量化节点，不能在 quant 和 dequant 之间分段
    """
    submodule_1 = extract_subgraph(quantized_model, ["quantize_per_tensor_default_1"], ["dequantize_per_tensor_default_134"])
    submodule_2 = extract_subgraph(quantized_model, ["quantize_per_tensor_default_27"], ["dequantize_per_tensor_default_154"])
    submodule_3 = extract_subgraph(quantized_model, ["quantize_per_tensor_default_101"], ["dequantize_per_tensor_default_127"])


    """
    接下来为多 multi stage 推理流程准备推理函数
    以便使用已有的推理脚本
    """
    # multi stage 浮点模型
    def model3s_forward(x):
        stage1.eval()
        stage2.eval()
        stage3.eval()

        x = stage1(x)
        x = stage2(x)
        x = stage3(x)

        return x

    # 由 multi stage 浮点模型每个 stage 分别独立加载参数的量化模型
    def model3s_quant_forward(x):
        torch.ao.quantization.move_exported_model_to_eval(quantized_model_s1)
        torch.ao.quantization.move_exported_model_to_eval(quantized_model_s2)
        torch.ao.quantization.move_exported_model_to_eval(quantized_model_s3)

        x = quantized_model_s1(x)
        x = quantized_model_s2(x)
        x = quantized_model_s3(x)

        return x

    # 由完整量化模型切多个子图再进行分 stage 推理的量化模型
    def model3s_submodule_forward(x):
        torch.ao.quantization.move_exported_model_to_eval(submodule_1)
        torch.ao.quantization.move_exported_model_to_eval(submodule_2)
        torch.ao.quantization.move_exported_model_to_eval(submodule_3)

        x = submodule_1(x)
        x = submodule_2(x)
        x = submodule_3(x)

        return x


    """
    最后推理并打印结果
    """
    # 推理前 100 个数据，快速对比结果；要推理完整测试集设置 total_size=None
    top1, top5 = evaluate(model.eval(), data_loader_test, total_size=100)
    top1_3s, top5_3s = evaluate(model3s_forward, data_loader_test, total_size=100)
    top1_q, top5_q = evaluate(quantized_model, data_loader_test, total_size=100)
    top1_3sq, top5_3sq = evaluate(model3s_quant_forward, data_loader_test, total_size=100)
    top1_3ss, top5_3ss = evaluate(model3s_submodule_forward, data_loader_test, total_size=100)

    # 打印
    def to_float(t):
        assert isinstance(t, torch.Tensor)
        return t.cpu().numpy().tolist()
    print(f"model1: top1:{to_float(top1.avg)}, top5:{to_float(top5.avg)}")
    print(f"model2: top1:{to_float(top1_3s.avg)}, top5:{to_float(top5_3s.avg)}")
    print(f"model3: top1:{to_float(top1_q.avg)}, top5:{to_float(top5_q.avg)}")
    print(f"model4: top1:{to_float(top1_3sq.avg)}, top5:{to_float(top5_3sq.avg)}")
    print(f"model5: top1:{to_float(top1_3ss.avg)}, top5:{to_float(top5_3ss.avg)}")

    # 可以保存模型查比较化参数的区别
    # dynamo_export(quantized_model, example_inputs, "./tmp_q.onnx")
    # dynamo_export(quantized_model_s2, example_inputs_s2, "./tmp_3sq_2.onnx")
    # dynamo_export(submodule_2, example_inputs_s2, "./tmp_3ss_2.onnx")
    