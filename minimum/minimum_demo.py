"""最小示例(torch 2.10,utils 统一 API):conv+BN+ReLU 小网络 QAT → 导出。

- 图捕获:显式 torch.export.export(静态 shape,无需 dynamic_shapes);
- dynamo_export():float 参考与 QAT 导出同路径;
- dynamo_export():optimize=False + 函数内联;
- per-channel torchlib 映射:脚本显式 import utils.quantized_decomposed_dequantize_per_channel 注册。

运行:
  cd /home/heqi/project-qat/QAT.axera && PYTHONPATH=. CUDA_VISIBLE_DEVICES=<空卡> \
    <env>/bin/python minimum/minimum_demo.py
"""
import torch
import torch.nn as nn
from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e, convert_pt2e

from utils.ax_quantizer import AXQuantizer
from utils.train_utils import dynamo_export
from utils.quant_utils import simplify_and_fix_4bit_dtype
import utils.quantized_decomposed_dequantize_per_channel  # noqa: F401 注册 per-channel torchlib 映射

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning, module=r'.*')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        return output


# example inputs
torch.manual_seed(42)
input = torch.rand(1, 32, 256, 768).to("cuda")

# float_model
float_model = Net().to("cuda")
float_path = "./minimum/minimum_float.onnx"
dynamo_export(float_model, input, float_path)

# set quantizer
quantizer = AXQuantizer("./minimum/config.json")

# export qat model
# 静态 shape 捕获(example 固定 batch=1,直接 torch.export.export)
exported_model = torch.export.export(float_model.train(), (input,)).module()
prepared_model = prepare_qat_pt2e(exported_model, quantizer)

quantized_model = convert_pt2e(prepared_model)

# export
qat_path = "./minimum/minimum_qat.onnx"
dynamo_export(quantized_model, input, qat_path)

# onnx simplify & fix dtype
sim_path = "./minimum/minimum_qat_sim.onnx"
simplify_and_fix_4bit_dtype(qat_path, sim_path)
