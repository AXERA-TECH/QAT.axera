# torch 2.10 版(与同名去 _2_10 文件对应,脚本化移植)。改动:
#   1. torchao PT2E + utils_2_10;export_for_training → torch.export.export;
#   2. move_exported_model_to_eval / disable|enable_fake_quant / disable|enable_observer
#      改从 torchao.quantization.pt2e 取(接口同名,已验证存在);
#   3. 产物带 _2_10 后缀。模型均先 eval 再导出,无训练态 BN 问题。
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchao.quantization.pt2e.quantize_pt2e import (
    prepare_qat_pt2e,
    convert_pt2e,
)
import torchao.quantization.pt2e as tao_pt2e
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent  # 向上两级：test_clamp/ -> QAT.axera/
project_root_str = str(project_root)

if project_root_str not in sys.path:
    sys.path.append(project_root_str)

from utils_2_10.ax_quantizer import AXQuantizer
from utils_2_10.train_utils import dynamo_export, onnx_simplify
import utils_2_10.quantized_decomposed_dequantize_per_channel

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning, module=r'.*')
warnings.filterwarnings(action='default', module=r'torch.ao.quantization')


class ConvRelu6Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu6(x)
        return x


torch.manual_seed(42)
input = torch.rand(1, 3, 64, 64)

float_model = ConvRelu6Net()
float_model.eval()
with torch.no_grad():
    float_out = float_model(input)

float_path = "./test_clamp/conv_relu6_float_2_10.onnx"
dynamo_export(float_model, input, float_path)
print(f"float onnx exported to {float_path}")

quantizer = AXQuantizer("./test_clamp/config.json")
exported_model = torch.export.export(float_model, (input,)).module()
prepared_model = prepare_qat_pt2e(exported_model, quantizer)

tao_pt2e.move_exported_model_to_eval(prepared_model)

with torch.no_grad():
    qat_out = prepared_model(input)

prepared_model.apply(tao_pt2e.disable_fake_quant)
with torch.no_grad():
    qat_no_fq_out = prepared_model(input)

prepared_model.apply(tao_pt2e.enable_fake_quant)
quantized_model = convert_pt2e(prepared_model)

qat_path = "./test_clamp/conv_relu6_qat_2_10.onnx"
dynamo_export(quantized_model, input, qat_path)
print(f"qat onnx exported to {qat_path}")

sim_path = "./test_clamp/conv_relu6_qat_2_10_sim.onnx"
onnx_simplify(qat_path, sim_path)
print(f"simplified onnx exported to {sim_path}")

float_np = float_out.numpy()
qat_np = qat_out.detach().numpy()
qat_no_fq_np = qat_no_fq_out.detach().numpy()

print()
print("=" * 60)
print("PyTorch 模型对比")
print("-" * 30)
diff_fq = np.abs(float_np - qat_np)
diff_no_fq = np.abs(float_np - qat_no_fq_np)
print(f"  float                    min={float_np.min():.6f}  max={float_np.max():.6f}")
print(f"  qat  (fake quant ON)     min={qat_np.min():.6f}  max={qat_np.max():.6f}")
print(f"  qat  (fake quant OFF)    min={qat_no_fq_np.min():.6f}  max={qat_no_fq_np.max():.6f}")
print(f"  diff (fake quant ON)     max={diff_fq.max():.6f}  mean={diff_fq.mean():.6f}")
print(f"  diff (fake quant OFF)    max={diff_no_fq.max():.6f}  mean={diff_no_fq.mean():.6f}")
print(f"  allclose (fake quant OFF): {np.allclose(float_np, qat_no_fq_np, atol=1e-5)}")
print("=" * 60)
