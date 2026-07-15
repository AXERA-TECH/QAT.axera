import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.ao.quantization.quantize_pt2e import (
    prepare_qat_pt2e,
    convert_pt2e,
)
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent  # 向上两级：test_clamp/ -> QAT.axera/
project_root_str = str(project_root)

if project_root_str not in sys.path:
    sys.path.append(project_root_str)

from utils.ax_quantizer import AXQuantizer
from utils.train_utils import dynamo_export, onnx_simplify
import utils.quantized_decomposed_dequantize_per_channel

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

float_path = "./test_clamp/conv_relu6_float.onnx"
dynamo_export(float_model, input, float_path)
print(f"float onnx exported to {float_path}")

quantizer = AXQuantizer("./test_clamp/config.json")
exported_model = torch.export.export_for_training(float_model, (input,)).module()
prepared_model = prepare_qat_pt2e(exported_model, quantizer)

torch.ao.quantization.move_exported_model_to_eval(prepared_model)

with torch.no_grad():
    qat_out = prepared_model(input)

prepared_model.apply(torch.ao.quantization.disable_fake_quant)
with torch.no_grad():
    qat_no_fq_out = prepared_model(input)

prepared_model.apply(torch.ao.quantization.enable_fake_quant)
quantized_model = convert_pt2e(prepared_model)

qat_path = "./test_clamp/conv_relu6_qat.onnx"
dynamo_export(quantized_model, input, qat_path)
print(f"qat onnx exported to {qat_path}")

sim_path = "./test_clamp/conv_relu6_qat_sim.onnx"
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
