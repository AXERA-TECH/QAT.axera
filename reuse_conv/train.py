import torch  # Version: 2.6.0+cu118
import torch.nn as nn
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
from utils.train_utils import dynamo_export, onnx_simplify
import utils.quantized_decomposed_dequantize_per_channel
from utils.extract import extract_subgraph
from IPython import embed

# Set up warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.ao.quantization'
)


class Net_Loop_Module1(nn.Module):
    def __init__(self, module0, module1, module2):
        super().__init__()
        self.module0 = module0
        self.module1 = module1
        self.module2 = module2
    
    def forward(self, x):
        x = self.module0(x)
        for i in range(2):
            x = self.module1(x)
        x = self.module2(x)

        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(64),  # 去掉 BN 不支持
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


def train():
    # example inputs
    input = torch.rand(1, 64, 256, 768).to("cuda")

    # float_model
    float_model0 = Net().to("cuda")
    float_model1 = Net().to("cuda")
    float_model2 = Net().to("cuda")
    # float_path = "./tmp_float.onnx"
    # dynamo_export(float_model, input, float_path)

    # set quantizer
    global_config, regional_configs = load_config("./reuse_conv/config.json")
    quantizer = AXQuantizer(annotate_bias=False)
    quantizer.set_global(global_config)
    quantizer.set_regional(regional_configs)

    # export qat model
    exported_model0 = torch.export.export_for_training(float_model0, (input,)).module()
    exported_model1 = torch.export.export_for_training(float_model1, (input,)).module()
    exported_model2 = torch.export.export_for_training(float_model2, (input,)).module()

    prepared_model0 = prepare_qat_pt2e(exported_model0, quantizer)
    prepared_model1 = prepare_qat_pt2e(exported_model1, quantizer)
    prepared_model2 = prepare_qat_pt2e(exported_model2, quantizer)

    new_model = Net_Loop_Module1(prepared_model0, prepared_model1, prepared_model2)


    # train
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(new_model.parameters(), lr=0.001, momentum=0.9)  # 更小的学习率
    output = new_model(input)
    target = torch.rand(1, 64, 256, 768).to("cuda")  # 随机一个 gt 训一轮

    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # torch.save(prepared_model.state_dict(), "./tmp.pth")

    # convert
    # remove_reused_bn_param_hack(new_model.module1)
    quantized_model0 = convert_pt2e(new_model.module0)
    quantized_model1 = convert_pt2e(new_model.module1)
    quantized_model2 = convert_pt2e(new_model.module2)

    # export
    qat_path = "./tmp0_qat.onnx"
    dynamo_export(quantized_model0, input, qat_path)
    qat_path = "./tmp1_qat.onnx"
    dynamo_export(quantized_model1, input, qat_path)
    qat_path = "./tmp2_qat.onnx"
    dynamo_export(quantized_model2, input, qat_path)

    # onnx simplify
    # sim_path = "./tmp_qat_sim.onnx"
    # onnx_simplify(qat_path, sim_path)


if __name__ == "__main__":
    train()
