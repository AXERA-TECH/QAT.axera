"""统一 API 版最小示例:同一份代码在 torch 2.6 与 2.10 直接运行,零版本分支。

与 minimum_demo.py(2.6 原件)流程一一对应,所有版本差异由 axquant 内部消化:
  - capture():2.6 → export_for_training,2.10 → torch.export.export;
  - export_float_reference():内置 eval 深拷贝(2.10 不能直接导训练态 BN);
  - dynamo_export():2.6 → optimize(),2.10 → optimize=False + 函数内联;
  - per-channel torchlib 映射:import axquant 即自动注册。

运行(qat-dev,任一环境):
  cd /home/heqi/project-qat/QAT.axera && PYTHONPATH=. CUDA_VISIBLE_DEVICES=<空卡> \
    <env>/bin/python minimum/minimum_demo_axquant.py
"""
import torch
import torch.nn as nn

from axquant import (
    AXQuantizer,
    capture,
    prepare_qat_pt2e,
    convert_pt2e,
    dynamo_export,
    export_float_reference,
    simplify_and_fix_4bit_dtype,
)

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
float_path = "./minimum/minimum_float_ax.onnx"
export_float_reference(float_model, input, float_path)

# set quantizer
quantizer = AXQuantizer("./minimum/config.json")

# export qat model
exported_model = capture(float_model.train(), (input,))
prepared_model = prepare_qat_pt2e(exported_model, quantizer)

quantized_model = convert_pt2e(prepared_model)

# export
qat_path = "./minimum/minimum_qat_ax.onnx"
dynamo_export(quantized_model, input, qat_path)

# onnx simplify & fix dtype
sim_path = "./minimum/minimum_qat_ax_sim.onnx"
simplify_and_fix_4bit_dtype(qat_path, sim_path)
