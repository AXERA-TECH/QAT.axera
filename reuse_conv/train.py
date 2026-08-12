"""运行时循环复用 prepared 子模型 demo(torch 2.10,utils 统一 API)。

说明:
  1. 图捕获:torch.export.export(输入 batch 固定为 1,静态捕获即可);
  2. 存 tmp_ax.pth 供 test.py 使用;
  3. 产物写入 reuse_conv/ 目录。
备注:本示例的"复用"是运行时循环(module1 每个 forward 被调两次),图内无展开,
remove_reused_bn_param_hack 保持注释(版本无关实现见 utils/ax_quantizer.py)。

运行:
  cd /home/heqi/project-qat/QAT.axera && PYTHONPATH=. CUDA_VISIBLE_DEVICES=<空卡> \
    <env>/bin/python reuse_conv/train.py
"""
import torch
import torch.nn as nn
from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e, convert_pt2e

from utils.ax_quantizer import AXQuantizer, load_config, remove_reused_bn_param_hack  # noqa: F401 与原版保持一致(调用点见下方注释)
from utils.train_utils import dynamo_export, onnx_simplify
import utils.quantized_decomposed_dequantize_per_channel  # noqa: F401 注册 per-channel torchlib 映射

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning, module=r'.*')


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

    # set quantizer
    global_config, regional_configs = load_config("./reuse_conv/config.json")
    quantizer = AXQuantizer("./reuse_conv/config.json", annotate_bias=False)

    # export qat model(静态 shape,直接 torch.export.export)
    exported_model0 = torch.export.export(float_model0.train(), (input,)).module()
    exported_model1 = torch.export.export(float_model1.train(), (input,)).module()
    exported_model2 = torch.export.export(float_model2.train(), (input,)).module()

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

    torch.save(prepared_model1.state_dict(), "./reuse_conv/tmp_ax.pth")

    # convert
    # remove_reused_bn_param_hack(new_model.module1)
    quantized_model0 = convert_pt2e(new_model.module0)
    quantized_model1 = convert_pt2e(new_model.module1)
    quantized_model2 = convert_pt2e(new_model.module2)

    # export
    qat_path = "./reuse_conv/tmp0_qat_ax.onnx"
    dynamo_export(quantized_model0, input, qat_path)
    qat_path = "./reuse_conv/tmp1_qat_ax.onnx"
    dynamo_export(quantized_model1, input, qat_path)
    qat_path = "./reuse_conv/tmp2_qat_ax.onnx"
    dynamo_export(quantized_model2, input, qat_path)

    # onnx simplify
    sim_path = "./reuse_conv/tmp1_qat_ax_sim.onnx"
    onnx_simplify("./reuse_conv/tmp1_qat_ax.onnx", sim_path)


if __name__ == "__main__":
    train()
