import copy
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock
from typing import Any, Callable, List, Optional, Type, Union
import numpy as np

from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from torch.ao.quantization.quantize_pt2e import (
    prepare_qat_pt2e,
    convert_pt2e,
)

# from utils.quantizer import (
#     AXQuantizer,
#     get_quantization_config,
# )
from utils.ax_quantizer import(
    load_config,
    AXQuantizer,
    remove_reused_bn_param_hack,
)
from utils.train_utils import (
    load_model,
    train_one_epoch,
    imagenet_data_loaders,
    dynamo_export,
    onnx_simplify,
    evaluate,
)
import utils.quantized_decomposed_dequantize_per_channel


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
    # load data
    data_loader, data_loader_test = imagenet_data_loaders("dataset/imagenet/")
    example_inputs_stage1 = (torch.rand(1, 3, 224, 224).to("cuda"),)
    example_inputs_stage2 = (torch.rand(1, 256, 56, 56).to("cuda"),)
    example_inputs_stage3 = (torch.rand(1, 256, 56, 56).to("cuda"),)

    # set float model
    float_model = ResNetFloat(Bottleneck, [3, 4, 6, 3]).to("cuda")
    float_model_stage1 = ResNetStage1(Bottleneck, [3, 4, 6, 3]).to("cuda")
    float_model_stage2 = ResNetStage2(Bottleneck, [3, 4, 6, 3]).to("cuda")
    float_model_stage3 = ResNetStage3(Bottleneck, [3, 4, 6, 3]).to("cuda")
    state_dict = torch.load("./resnet50/resnet50_pretrained_float.pth", weights_only=True)
    float_model_stage1.load_state_dict(state_dict)
    # float_model_stage2.load_state_dict(state_dict)
    float_model_stage3.load_state_dict(state_dict)

    float_path = "./reuse_conv/resnet50_float.onnx"
    dynamo_export(float_model, example_inputs_stage1, float_path)
    float_path_stage1 = "./reuse_conv/resnet50_float_stage1.onnx"
    dynamo_export(float_model_stage1, example_inputs_stage1, float_path_stage1)
    float_path_stage2 = "./reuse_conv/resnet50_float_stage2.onnx"
    dynamo_export(float_model_stage2, example_inputs_stage2, float_path_stage2)
    float_path_stage3 = "./reuse_conv/resnet50_float_stage3.onnx"
    dynamo_export(float_model_stage3, example_inputs_stage3, float_path_stage3)

    # quantizer
    global_config, regional_configs = load_config("./reuse_conv/config.json")
    quantizer = AXQuantizer(annotate_bias=False)
    quantizer.set_global(global_config)
    quantizer.set_regional(regional_configs)

    exported_model_stage1 = torch.export.export_for_training(float_model_stage1, example_inputs_stage1).module()
    exported_model_stage2 = torch.export.export_for_training(float_model_stage2, example_inputs_stage2).module()
    exported_model_stage3 = torch.export.export_for_training(float_model_stage3, example_inputs_stage3).module()
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
    num_epochs_between_evals = 2
    for nepoch in range(num_epochs):
        train_one_epoch(model, criterion, optimizer, data_loader, "cuda", num_train_batches)

        # checkpoint_path = "./reuse_conv/checkpoint/checkpoint_%s.pth" % nepoch
        # torch.save(prepared_model.state_dict(), checkpoint_path)

        # if (nepoch + 1) % num_epochs_between_evals == 0:
        #     prepared_model_copy = copy.deepcopy(prepared_model)
        #     quantized_model = convert_pt2e(prepared_model_copy)
        #     top1, top5 = evaluate(quantized_model, data_loader_test)
        #     print('Epoch %d: Evaluation accuracy, %2.2f' % (nepoch, top1.avg))

    torch.save(model.state_dict(), "./reuse_conv/resnet50.pth")
    torch.save(model.stage1.state_dict(), "./reuse_conv/resnet50_stage1.pth")
    torch.save(model.stage2.state_dict(), "./reuse_conv/resnet50_stage2.pth")
    torch.save(model.stage3.state_dict(), "./reuse_conv/resnet50_stage3.pth")

    # evaluate
    float_stage = copy.deepcopy(model)
    float_stage.forward = float_stage._float_forward
    quantized_model_stage1 = convert_pt2e(model.stage1)
    quantized_model_stage2 = convert_pt2e(model.stage2)
    quantized_model_stage3 = convert_pt2e(model.stage3)

    def quantized_model_forward(x):
        float_stage.eval()
        torch.ao.quantization.move_exported_model_to_eval(quantized_model_stage1)
        torch.ao.quantization.move_exported_model_to_eval(quantized_model_stage2)
        torch.ao.quantization.move_exported_model_to_eval(quantized_model_stage3)

        x = quantized_model_stage1(x)
        for i in range(2):
            x = float_stage(x)
            x = quantized_model_stage2(x)
        x = quantized_model_stage3(x)

        return x
    top1, top5 = evaluate(quantized_model_forward, data_loader_test, total_size=100)

    # export
    qat_path_stage1 = "./reuse_conv/resnet50_qat_stage1.onnx"
    dynamo_export(quantized_model_stage1, (example_inputs_stage1,), qat_path_stage1)
    qat_path_stage2 = "./reuse_conv/resnet50_qat_stage2.onnx"
    dynamo_export(quantized_model_stage2, (example_inputs_stage2,), qat_path_stage2)
    qat_path_stage3 = "./reuse_conv/resnet50_qat_stage3.onnx"
    dynamo_export(quantized_model_stage3, (example_inputs_stage3,), qat_path_stage3)

    # # onnx simplify
    sim_path_stage1 = "./reuse_conv/resnet50_qat_sim_stage1.onnx"
    onnx_simplify(qat_path_stage1, sim_path_stage1)
    sim_path_stage2 = "./reuse_conv/resnet50_qat_sim_stage2.onnx"
    onnx_simplify(qat_path_stage2, sim_path_stage2)
    sim_path_stage3 = "./reuse_conv/resnet50_qat_sim_stage3.onnx"
    onnx_simplify(qat_path_stage3, sim_path_stage3)


if __name__ == "__main__":
    train()
