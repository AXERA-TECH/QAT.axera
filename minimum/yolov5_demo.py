"""utils 统一 API 版(统一 utils API,torch 2.6/2.10 同一份代码):多输入 BEV 小网络 QAT 导出。

覆盖算子:ConvTranspose、cat、grid_sample、permute/reshape(共享观察器传播)、
多层 Linear——正好检验 axquant 里 convtranspose/concat/gridsample/linear
注解器与 propagate_annotation 的 2.10 适配。

相对 2.6 版的改动:
  1. utils 统一 API(内部自动路由 torch.ao/torchao);export_for_training → torch.export.export;
  2. float 导出用 eval 深拷贝(2.10 不能直接导训练态 BN);
  3. 原版 `AXQuantizer()` 无参调用在现行签名(config_file 必填)下本就报错,
     修正为带配置构造(原版笔误,2.6 上同样跑不了);
  4. 产物带 _2_10 后缀。

运行(qat-dev):
  cd /home/heqi/project-qat/QAT.axera && PYTHONPATH=. CUDA_VISIBLE_DEVICES=<空卡> \
    /home/heqi/miniforge3/envs/torch2.10/bin/python minimum/yolov5_demo_2_10.py
"""
import torch
import torch.nn as nn

from utils import (
    prepare_qat_pt2e,
    convert_pt2e,
    capture,
    export_float_reference,
    AXQuantizer,
    dynamo_export,
    onnx_simplify,
)

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning, module=r'.*')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.upscore1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 2, 2, bias=False),
            nn.ReLU(inplace=True)
        )

        self.conv4_1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        )

    def forward(self, input, grid0, grid1):

        # backbone
        conv1_1_out = self.conv1_1(input)
        conv1_2_out = self.conv1_2(conv1_1_out)

        pool1_out = self.pool1(conv1_2_out)
        conv2_1_out = self.conv2_1(pool1_out)
        conv2_2_out = self.conv2_2(conv2_1_out)

        pool2_out = self.pool2(conv2_2_out)
        conv3_1_out = self.conv3_1(pool2_out)
        conv3_2_out = self.conv3_2(conv3_1_out)
        upscore1_out = self.upscore1(conv3_2_out)

        concat_out = torch.cat([conv2_2_out, upscore1_out], dim=1)
        conv4_1_out = self.conv4_1(concat_out)
        backbone_out = self.conv4_2(conv4_1_out)  # (1, 64, h_, w_)

        # bev
        grid0_out = nn.functional.grid_sample(input=backbone_out, grid=grid0, mode="bilinear", padding_mode="zeros")
        grid1_out = nn.functional.grid_sample(input=backbone_out, grid=grid1, mode="bilinear", padding_mode="zeros")
        output = torch.cat([grid0_out, grid1_out], dim=0)  # (2, 64, h, w)

        # mlp
        output = output.permute(2, 3, 0, 1)  # (h, w, 2, 64)
        output = output.reshape(-1, 128)  # (h * w, 128), good for nhwc layout
        output = self.mlp(output)

        return output


# example inputs
input = torch.rand(1, 3, 256, 768).to("cuda")
grid0 = torch.rand(1, 120, 40, 2).to("cuda")
grid1 = torch.rand(1, 120, 40, 2).to("cuda")
inputs = (input, grid0, grid1)

# export float_model(eval 深拷贝,2.10 不能直接导训练态 BN)
float_model = Net().to("cuda")
float_path = "./minimum/yolov5_float.onnx"
export_float_reference(float_model, inputs, float_path)

# set quantizer(原版 AXQuantizer() 无参调用是笔误,现行签名 config_file 必填)
quantizer = AXQuantizer("./minimum/config.json")

# export qat model
exported_model = capture(float_model.train(), inputs)
prepared_model = prepare_qat_pt2e(exported_model, quantizer)
quantized_model = convert_pt2e(prepared_model)

# export
qat_path = "./minimum/yolov5_qat.onnx"
dynamo_export(quantized_model, inputs, qat_path)

# onnxsim
sim_path = "./minimum/yolov5_qat_sim.onnx"
onnx_simplify(qat_path, sim_path)
