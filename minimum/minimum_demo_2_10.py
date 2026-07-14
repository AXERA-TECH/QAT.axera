"""torch 2.10 版最小示例(与 minimum_demo.py 逐行对应,差异见行内注释)。

相对 2.6 版的改动(依据见仓库根 plan_torch210.md):
  1. PT2E 入口:torch.ao.quantization.quantize_pt2e → torchao.quantization.pt2e.quantize_pt2e
  2. 量化器:utils(torch.ao 类型体系) → utils_2_10(torchao 类型体系)
  3. 图捕获:export_for_training(已废弃) → torch.export.export
  4. 导出:dynamo_export 内部已显式 optimize=False(防权重 DQ 被常量折叠)
  5. float 参考模型改用 eval 态深拷贝导出:2.10 导出器不再容忍训练态 BN 的
     buffer 突变(报 "Key 'b_bn_running_mean' does not match ... 'getitem_3'");
     原模型保持训练态,供后续 QAT 图捕获
  6. 产物文件名带 _2_10 后缀,不覆盖 2.6 版产物

运行(qat-dev):
  cd /home/heqi/project-qat/QAT.axera && \
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=<空卡> \
    /home/heqi/miniforge3/envs/torch2.10/bin/python minimum/minimum_demo_2_10.py
"""
import copy

import torch  # Version: 2.10.0+cu126
import torch.nn as nn

from torchao.quantization.pt2e.quantize_pt2e import (
    prepare_qat_pt2e,
    convert_pt2e,
)

from utils_2_10.ax_quantizer import (
    load_config,
    AXQuantizer,
)
from utils_2_10.train_utils import dynamo_export
from utils_2_10.quant_utils import simplify_and_fix_4bit_dtype
import utils_2_10.quantized_decomposed_dequantize_per_channel

# Set up warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)


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
input = torch.rand(1, 32, 256, 768).to("cuda")

# float_model
float_model = Net().to("cuda")
float_path = "./minimum/minimum_float_2_10.onnx"
# 2.10:训练态 BN 的 buffer 突变会让 dynamo 导出失败,用 eval 态副本导出参考模型
dynamo_export(copy.deepcopy(float_model).eval(), input, float_path)

# set quantizer
quantizer = AXQuantizer("./minimum/config.json")

# export qat model(2.10:export_for_training 已废弃,统一走 torch.export.export)
exported_model = torch.export.export(float_model, (input,)).module()
prepared_model = prepare_qat_pt2e(exported_model, quantizer)

# # train
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(prepared_model.parameters(), lr=0.001, momentum=0.9)  # 更小的学习率
# output = prepared_model(input)
# target = torch.rand(1, 32, 256, 768).to("cuda")  # 随机一个 gt 训一轮

# loss = criterion(output, target)
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()

quantized_model = convert_pt2e(prepared_model)

# export
qat_path = "./minimum/minimum_qat_2_10.onnx"
dynamo_export(quantized_model, input, qat_path)

# onnx simplify & fix dtype
sim_path = "./minimum/minimum_qat_2_10_sim.onnx"
simplify_and_fix_4bit_dtype(qat_path, sim_path)
