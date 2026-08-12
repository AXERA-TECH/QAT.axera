"""stage2 循环复用的分段 QAT 训练(torch 2.10,utils 统一 API)。

说明:
  1. 模型类(ResNetFloat/Stage1/2/3/MultiStage)无 torch.ao 依赖,直接从原模块 import 复用;
  2. 图捕获:torch.export.export + 动态 batch(训练 batch=32);
  3. float 参考导出与 QAT 导出同路径(dynamo_export);
  4. 数据:机器无 ImageNet 训练集 → imagenet_data_loaders(fake_data=True);
  5. 修正原版导出处的双层 tuple 笔误 dynamo_export(model, (example_inputs,));
  6. 产物写入 reuse_conv/ 目录。

运行:
  cd /home/heqi/project-qat/QAT.axera && PYTHONPATH=. CUDA_VISIBLE_DEVICES=<空卡> \
    <env>/bin/python reuse_conv/train_resnet.py
"""
import copy

import torch
from torch.export import Dim
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock
from typing import Callable, List, Optional, Type, Union
from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e, convert_pt2e
from torchao.quantization.pt2e import move_exported_model_to_eval

from utils.ax_quantizer import AXQuantizer, load_config
from utils.train_utils import (
    dynamo_export,
    load_model,
    train_one_epoch,
    imagenet_data_loaders,
    onnx_simplify,
    evaluate,
)
import utils.quantized_decomposed_dequantize_per_channel  # noqa: F401 注册 per-channel torchlib 映射

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning, module=r'.*')


class ResNetFloat(ResNet):
    """
    额外添加的需要循环调用的部分
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
        super(ResNetFloat, self).__init__(block=block, layers=layers)
        self.float_conv = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.float_bn = nn.BatchNorm2d(256)
        self.float_relu = nn.ReLU(inplace=True)

        self.tmp_conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True)
        self.tmp_bn1 = nn.BatchNorm2d(256)
        self.tmp_relu1 = nn.ReLU(inplace=True)

        self.tmp_conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.tmp_bn2 = nn.BatchNorm2d(256)
        self.tmp_relu2 = nn.ReLU(inplace=True)

        self.tmp_conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True)
        self.tmp_bn3 = nn.BatchNorm2d(256)
        self.tmp_relu3 = nn.ReLU(inplace=True)
    
    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        # stage1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # loop
        for i in range(2):
            # not QAT
            self.float_conv(x)
            self.float_bn(x)
            self.float_relu(x)

            # stage2
            identity = x

            out = self.tmp_conv1(x)
            out = self.tmp_bn1(out)
            out = self.tmp_relu1(out)

            out = self.tmp_conv2(out)
            out = self.tmp_bn2(out)
            out = self.tmp_relu2(out)

            out = self.tmp_conv3(out)
            out = self.tmp_bn3(out)

            out += identity
            out = self.tmp_relu3(out)
            x = out
        # stage3
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return out


class ResNetStage1(ResNet):
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
        super(ResNetStage1, self).__init__(block=block, layers=layers)
    
    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        return x

    
class ResNetStage2(ResNet):
    """
    额外添加的需要循环调用的部分
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
        super(ResNetStage2, self).__init__(block=block, layers=layers)
        self.tmp_conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True)
        self.tmp_bn1 = nn.BatchNorm2d(256)
        self.tmp_relu1 = nn.ReLU(inplace=True)

        self.tmp_conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.tmp_bn2 = nn.BatchNorm2d(256)
        self.tmp_relu2 = nn.ReLU(inplace=True)

        self.tmp_conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True)
        self.tmp_bn3 = nn.BatchNorm2d(256)
        self.tmp_relu3 = nn.ReLU(inplace=True)
    
    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        identity = x

        out = self.tmp_conv1(x)
        out = self.tmp_bn1(out)
        out = self.tmp_relu1(out)

        out = self.tmp_conv2(out)
        out = self.tmp_bn2(out)
        out = self.tmp_relu2(out)

        out = self.tmp_conv3(out)
        out = self.tmp_bn3(out)

        out += identity
        out = self.tmp_relu3(out)
        x = out

        return x


class ResNetStage3(ResNet):
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
        super(ResNetStage3, self).__init__(block=block, layers=layers)
    
    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class ResNetMultiStage(ResNet):
    """
    将子模型链接起来，需要循环调用的部分在这里循环
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
        stage1 = None,
        stage2 = None,
        stage3 = None,
    ) -> None:
        super(ResNetMultiStage, self).__init__(block=block, layers=layers)
        self.float_conv = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.float_bn = nn.BatchNorm2d(256)
        self.float_relu = nn.ReLU(inplace=True)

        self.stage1 = stage1
        self.stage2 = stage2
        self.stage3 = stage3
    
    def forward(self, x):
        x = self.stage1(x)
        for i in range(2):
            self.float_conv(x)
            self.float_bn(x)
            self.float_relu(x)
            x = self.stage2(x)
        x = self.stage3(x)

        return x
    
    def _float_forward(self, x):
        self.float_conv(x)
        self.float_bn(x)
        self.float_relu(x)
        return x


def train():
    # load data(fake_data:机器无 ImageNet 训练集;fake_train_size 需 > 每
    # epoch 步数×batch,否则 loader 提前耗尽)
    data_loader, data_loader_test = imagenet_data_loaders(
        "dataset/imagenet/", fake_data=True, fake_train_size=3200)
    example_inputs_stage1 = (torch.rand(2, 3, 224, 224).to("cuda"),)
    example_inputs_stage2 = (torch.rand(2, 256, 56, 56).to("cuda"),)
    example_inputs_stage3 = (torch.rand(2, 256, 56, 56).to("cuda"),)

    # set float model
    float_model = ResNetFloat(Bottleneck, [3, 4, 6, 3]).to("cuda")
    float_model_stage1 = ResNetStage1(Bottleneck, [3, 4, 6, 3]).to("cuda")
    float_model_stage2 = ResNetStage2(Bottleneck, [3, 4, 6, 3]).to("cuda")
    float_model_stage3 = ResNetStage3(Bottleneck, [3, 4, 6, 3]).to("cuda")
    state_dict = torch.load("./resnet50/resnet50_pretrained_float.pth", weights_only=True)
    float_model_stage1.load_state_dict(state_dict)
    # float_model_stage2.load_state_dict(state_dict)
    float_model_stage3.load_state_dict(state_dict)

    # float 参考导出(与 QAT 导出同路径 dynamo_export)
    dynamo_export(float_model, example_inputs_stage1,
                  "./reuse_conv/resnet50_float_ax.onnx")
    dynamo_export(float_model_stage1, example_inputs_stage1,
                  "./reuse_conv/resnet50_float_stage1_ax.onnx")
    dynamo_export(float_model_stage2, example_inputs_stage2,
                  "./reuse_conv/resnet50_float_stage2_ax.onnx")
    dynamo_export(float_model_stage3, example_inputs_stage3,
                  "./reuse_conv/resnet50_float_stage3_ax.onnx")

    # quantizer
    global_config, regional_configs = load_config("./reuse_conv/config.json")
    quantizer = AXQuantizer("./reuse_conv/config.json", annotate_bias=False)

    # 分段捕获(batch 动态,example 给 >=2 规避 0/1 特化)
    dynamic_shapes = ({0: Dim("batch", min=1, max=1024)},)
    exported_model_stage1 = torch.export.export(
        float_model_stage1.train(), example_inputs_stage1, dynamic_shapes=dynamic_shapes,
    ).module()
    exported_model_stage2 = torch.export.export(
        float_model_stage2.train(), example_inputs_stage2, dynamic_shapes=dynamic_shapes,
    ).module()
    exported_model_stage3 = torch.export.export(
        float_model_stage3.train(), example_inputs_stage3, dynamic_shapes=dynamic_shapes,
    ).module()
    prepared_model_stage1 = prepare_qat_pt2e(exported_model_stage1, quantizer)
    prepared_model_stage2 = prepare_qat_pt2e(exported_model_stage2, quantizer)
    prepared_model_stage3 = prepare_qat_pt2e(exported_model_stage3, quantizer)
    model = ResNetMultiStage(
        Bottleneck,
        [3, 4, 6, 3],
        stage1=prepared_model_stage1,
        stage2=prepared_model_stage2,
        stage3=prepared_model_stage3,
    ).to("cuda")

    num_epochs = 5
    num_train_batches = 50
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # 更小的学习率

    # train
    for nepoch in range(num_epochs):
        train_one_epoch(model, criterion, optimizer, data_loader, "cuda", num_train_batches)

    torch.save(model.state_dict(), "./reuse_conv/resnet50_ax.pth")
    torch.save(model.stage1.state_dict(), "./reuse_conv/resnet50_stage1_ax.pth")
    torch.save(model.stage2.state_dict(), "./reuse_conv/resnet50_stage2_ax.pth")
    torch.save(model.stage3.state_dict(), "./reuse_conv/resnet50_stage3_ax.pth")

    # evaluate
    float_stage = copy.deepcopy(model)
    float_stage.forward = float_stage._float_forward
    quantized_model_stage1 = convert_pt2e(model.stage1)
    quantized_model_stage2 = convert_pt2e(model.stage2)
    quantized_model_stage3 = convert_pt2e(model.stage3)

    def quantized_model_forward(x):
        float_stage.eval()
        move_exported_model_to_eval(quantized_model_stage1)
        move_exported_model_to_eval(quantized_model_stage2)
        move_exported_model_to_eval(quantized_model_stage3)

        x = quantized_model_stage1(x)
        for i in range(2):
            x = float_stage(x)
            x = quantized_model_stage2(x)
        x = quantized_model_stage3(x)

        return x
    top1, top5 = evaluate(quantized_model_forward, data_loader_test, total_size=100)
    print(f"[eval] 分段量化(fake data): top1={top1.avg:.3f} top5={top5.avg:.3f}")

    # export(部署产物:固定 shape,example batch=1 且不声明动态维)
    export_example_stage1 = (torch.rand(1, 3, 224, 224).to("cuda"),)
    export_example_stage2 = (torch.rand(1, 256, 56, 56).to("cuda"),)
    export_example_stage3 = (torch.rand(1, 256, 56, 56).to("cuda"),)
    qat_path_stage1 = "./reuse_conv/resnet50_qat_stage1_ax.onnx"
    dynamo_export(quantized_model_stage1, export_example_stage1, qat_path_stage1)
    qat_path_stage2 = "./reuse_conv/resnet50_qat_stage2_ax.onnx"
    dynamo_export(quantized_model_stage2, export_example_stage2, qat_path_stage2)
    qat_path_stage3 = "./reuse_conv/resnet50_qat_stage3_ax.onnx"
    dynamo_export(quantized_model_stage3, export_example_stage3, qat_path_stage3)

    # onnx simplify
    onnx_simplify(qat_path_stage1, "./reuse_conv/resnet50_qat_sim_stage1_ax.onnx")
    onnx_simplify(qat_path_stage2, "./reuse_conv/resnet50_qat_sim_stage2_ax.onnx")
    onnx_simplify(qat_path_stage3, "./reuse_conv/resnet50_qat_sim_stage3_ax.onnx")


if __name__ == "__main__":
    train()
