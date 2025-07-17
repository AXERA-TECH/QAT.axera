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


class ReuseResNet(ResNet):
    """
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
        super(ReuseResNet, self).__init__(block=block, layers=layers)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.reused_conv = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1)
        self.reused_bn = norm_layer(self.inplanes)
        self.reused_relu = nn.ReLU(inplace=True)

    
    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # 强行复用
        x = self.reused_conv(x)
        x = self.reused_bn(x)  # 去掉 BN 不支持
        x = self.reused_relu(x)
        x = self.reused_conv(x)
        x = self.reused_bn(x)  # 去掉 BN 不支持
        x = self.reused_relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def train():
    # load data
    data_loader, data_loader_test = imagenet_data_loaders("dataset/imagenet/")
    example_inputs = (torch.rand(1, 3, 224, 224).to("cuda"),)

    # set float model
    float_model = ReuseResNet(Bottleneck, [3, 4, 6, 3]).to("cuda")
    state_dict = torch.load("./resnet50/resnet50_pretrained_float.pth", weights_only=True)
    float_model.load_state_dict(state_dict, strict=False)
    float_path = "./reuse_conv/resnet50_float.onnx"
    dynamo_export(float_model, example_inputs, float_path)

    # quantizer
    global_config, regional_configs = load_config("./reuse_conv/config.json")
    quantizer = AXQuantizer(annotate_bias=False)
    quantizer.set_global(global_config)
    quantizer.set_regional(regional_configs)

    exported_model = torch.export.export_for_training(float_model, example_inputs).module()
    prepared_model = prepare_qat_pt2e(exported_model, quantizer)

    num_epochs = 5
    num_train_batches = 50
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(prepared_model.parameters(), lr=0.001, momentum=0.9)  # 更小的学习率

    # train
    num_epochs_between_evals = 2
    for nepoch in range(num_epochs):
        train_one_epoch(prepared_model, criterion, optimizer, data_loader, "cuda", num_train_batches)

        # checkpoint_path = "./reuse_conv/checkpoint/checkpoint_%s.pth" % nepoch
        # torch.save(prepared_model.state_dict(), checkpoint_path)

        # if (nepoch + 1) % num_epochs_between_evals == 0:
        #     prepared_model_copy = copy.deepcopy(prepared_model)
        #     quantized_model = convert_pt2e(prepared_model_copy)
        #     top1, top5 = evaluate(quantized_model, data_loader_test)
        #     print('Epoch %d: Evaluation accuracy, %2.2f' % (nepoch, top1.avg))

    torch.save(prepared_model.state_dict(), "./reuse_conv/resnet50.pth")

    # evaluate
    remove_reused_bn_param_hack(prepared_model)
    quantized_model = convert_pt2e(prepared_model)
    top1, top5 = evaluate(quantized_model, data_loader_test, total_size=100)

    # export
    qat_path = "./reuse_conv/resnet50_qat.onnx"
    dynamo_export(quantized_model, (example_inputs,), qat_path)

    # onnx simplify
    sim_path = "./reuse_conv/resnet50_qat_sim.onnx"
    onnx_simplify(qat_path, sim_path)


if __name__ == "__main__":
    train()
