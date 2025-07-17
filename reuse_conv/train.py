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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.loop = 3
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(64),  # 去掉 BN 不支持
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        for i in range(self.loop):
            x = self.conv(x)  # RuntimeError: Tried to erase Node add_ but it still had 1 users in the graph: {add__1: None}!
        return x


def train():
    # example inputs
    input = torch.rand(1, 64, 256, 768).to("cuda")

    # float_model
    float_model = Net().to("cuda")
    float_path = "./reuse_conv/tmp_float.onnx"
    dynamo_export(float_model, input, float_path)

    # set quantizer
    global_config, regional_configs = load_config("./reuse_conv/config.json")
    quantizer = AXQuantizer(annotate_bias=False)
    quantizer.set_global(global_config)
    quantizer.set_regional(regional_configs)

    # export qat model
    exported_model = torch.export.export_for_training(float_model, (input,)).module()
    prepared_model = prepare_qat_pt2e(exported_model, quantizer)

    # train
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(prepared_model.parameters(), lr=0.001, momentum=0.9)  # 更小的学习率
    output = prepared_model(input)
    target = torch.rand(1, 64, 256, 768).to("cuda")  # 随机一个 gt 训一轮

    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    torch.save(prepared_model.state_dict(), "./reuse_conv/tmp.pth")

    # convert
    remove_reused_bn_param_hack(prepared_model)
    quantized_model = convert_pt2e(prepared_model)

    # save input & output for test
    gt = quantized_model(input)
    np.save("./reuse_conv/input.npy", input.cpu().numpy())
    np.save("./reuse_conv/gt.npy", gt.cpu().numpy())

    # export
    qat_path = "./reuse_conv/tmp_qat.onnx"
    dynamo_export(quantized_model, input, qat_path)

    # onnx simplify
    sim_path = "./reuse_conv/tmp_qat_sim.onnx"
    onnx_simplify(qat_path, sim_path)


if __name__ == "__main__":
    train()
