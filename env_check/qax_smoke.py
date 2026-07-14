"""torch2.10 环境冒烟：版本、CUDA、PT2E QAT 闭环、dynamo ONNX 导出 + ORT 数值对齐。"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

def ver(mod):
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "?")
    except Exception as e:
        return f"IMPORT FAIL: {e}"

print("== versions ==")
for m in ["torch", "torchvision", "numpy", "onnx", "onnxscript", "onnx_ir",
          "onnxruntime", "onnxslim", "onnx_graphsurgeon", "tqdm", "yaml"]:
    print(f"  {m:18s} {ver(m)}")

print("== cuda ==")
print("  available:", torch.cuda.is_available(), "| device:", torch.cuda.get_device_name(0))
x = torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")
print("  matmul on cuda OK, mean =", float(x.mean()))

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(8)
        self.act = torch.nn.ReLU()
        self.head = torch.nn.Conv2d(8, 4, 1)
    def forward(self, x):
        return self.head(self.act(self.bn(self.conv(x))))

ex = (torch.randn(2, 3, 32, 32),)

print("== PT2E QAT roundtrip (torch.ao) ==")
from torch.ao.quantization.quantize_pt2e import prepare_qat_pt2e, convert_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer, get_symmetric_quantization_config)
m = Tiny().train()
gm = torch.export.export_for_training(m, ex).module()
q = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_qat=True))
prep = prepare_qat_pt2e(gm, q)
opt = torch.optim.SGD([p for p in prep.parameters() if p.requires_grad], lr=1e-3)
for _ in range(3):
    opt.zero_grad(); prep(*ex).sum().backward(); opt.step()
conv = convert_pt2e(prep)
out = conv(*ex)
print("  prepare_qat_pt2e -> train 3 steps -> convert_pt2e OK, out", tuple(out.shape))

print("== repo AXQuantizer import ==")
import sys
sys.path.insert(0, "/home/heqi/project-qat/QAT.axera")
try:
    from utils import ax_quantizer  # noqa
    print("  utils.ax_quantizer import OK")
except Exception as e:
    print(f"  utils.ax_quantizer FAIL (预期需迁移): {type(e).__name__}: {e}")

print("== dynamo ONNX export (float) + ORT parity ==")
import onnx, onnxruntime as ort
fm = Tiny().eval()
ep = torch.onnx.export(fm, ex, dynamo=True)
ep.save("/tmp/qax_smoke_float.onnx")
mo = onnx.load("/tmp/qax_smoke_float.onnx")
onnx.checker.check_model(mo)
print("  opset:", [f"{o.domain or 'ai.onnx'}:{o.version}" for o in mo.opset_import],
      "| ir_version:", mo.ir_version)
sess = ort.InferenceSession("/tmp/qax_smoke_float.onnx", providers=["CPUExecutionProvider"])
ref = fm(*ex).detach().numpy()
got = sess.run(None, {sess.get_inputs()[0].name: ex[0].numpy()})[0]
print("  ORT parity max|diff| =", float(np.abs(ref - got).max()))

print("== dynamo ONNX export (PT2E converted QDQ) ==")
try:
    ep2 = torch.onnx.export(conv, ex, dynamo=True)
    ep2.save("/tmp/qax_smoke_qdq.onnx")
    mo2 = onnx.load("/tmp/qax_smoke_qdq.onnx")
    ops = {n.op_type for n in mo2.graph.node}
    print("  QDQ export OK, ops:", sorted(ops))
except Exception as e:
    print(f"  QDQ export FAIL (迁移重点): {type(e).__name__}: {e}")

print("== DONE ==")
