import torch
import torch.nn as nn
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


class AddClipNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        x = x1 + x2
        x = torch.clamp(x, min=0, max=1)
        return x


torch.manual_seed(42)
input1 = torch.rand(1, 3, 64, 64)
input2 = torch.rand(1, 3, 64, 64)

float_model = AddClipNet()
float_model.eval()
with torch.no_grad():
    float_out = float_model(input1, input2)

float_path = "./test_clamp/add_clip_float.onnx"
dynamo_export(float_model, (input1, input2), float_path)
print(f"float onnx exported to {float_path}")

quantizer = AXQuantizer("./test_clamp/config.json")
exported_model = torch.export.export_for_training(float_model, (input1, input2)).module()

print("\n=== FX Graph Nodes (before annotation) ===")
for node in exported_model.graph.nodes:
    if node.op == "call_function":
        print(f"  {node.name:30s} target={node.target}")
print()

prepared_model = prepare_qat_pt2e(exported_model, quantizer)

torch.ao.quantization.move_exported_model_to_eval(prepared_model)

prepared_model.apply(torch.ao.quantization.disable_fake_quant)
prepared_model.apply(torch.ao.quantization.disable_observer)
with torch.no_grad():
    prepared_float_like_out = prepared_model(input1, input2)

prepared_model.apply(torch.ao.quantization.enable_fake_quant)
prepared_model.apply(torch.ao.quantization.enable_observer)
with torch.no_grad():
    prepared_model(input1, input2)
prepared_model.apply(torch.ao.quantization.disable_observer)
with torch.no_grad():
    qat_out = prepared_model(input1, input2)

quantized_model = convert_pt2e(prepared_model)

qat_path = "./test_clamp/add_clip_qat.onnx"
dynamo_export(quantized_model, (input1, input2), qat_path)
print(f"qat onnx exported to {qat_path}")

sim_path = "./test_clamp/add_clip_qat_sim.onnx"
onnx_simplify(qat_path, sim_path)
print(f"simplified onnx exported to {sim_path}")

float_np = float_out.numpy()
prepared_float_like_np = prepared_float_like_out.detach().numpy()
qat_np = qat_out.detach().numpy()

print()
print("=" * 60)
print("PyTorch 模型对比")
print("-" * 30)
diff_prepared = np.abs(float_np - prepared_float_like_np)
diff_qat = np.abs(float_np - qat_np)
diff_quant_effect = np.abs(prepared_float_like_np - qat_np)
float_flat = float_np.reshape(-1)
qat_flat = qat_np.reshape(-1)
cos_sim = np.dot(float_flat, qat_flat) / (np.linalg.norm(float_flat) * np.linalg.norm(qat_flat) + 1e-12)
print(f"  float                    min={float_np.min():.6f}  max={float_np.max():.6f}")
print(f"  prepared (fake quant OFF) min={prepared_float_like_np.min():.6f}  max={prepared_float_like_np.max():.6f}")
print(f"  prepared (fake quant ON)  min={qat_np.min():.6f}  max={qat_np.max():.6f}")
print(f"  diff float vs prepared OFF max={diff_prepared.max():.6f}  mean={diff_prepared.mean():.6f}")
print(f"  diff float vs prepared ON  max={diff_qat.max():.6f}  mean={diff_qat.mean():.6f}")
print(f"  diff prepared OFF vs ON    max={diff_quant_effect.max():.6f}  mean={diff_quant_effect.mean():.6f}")
print(f"  float vs qat abs error     max={diff_qat.max():.6f}  mean={diff_qat.mean():.6f}")
print(f"  float vs qat cos sim       {cos_sim:.6f}")
print(f"  float vs qat allclose      {np.allclose(float_np, qat_np, atol=1e-5)}")
print(f"  allclose (prepared OFF): {np.allclose(float_np, prepared_float_like_np, atol=1e-5)}")
print("=" * 60)
