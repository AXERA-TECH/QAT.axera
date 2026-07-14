import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, onnx, onnxruntime as ort

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

print("== float dynamo export + ORT parity ==")
fm = Tiny().eval()
ep = torch.onnx.export(fm, ex, dynamo=True)
ep.save("/tmp/qax_smoke_float.onnx")
mo = onnx.load("/tmp/qax_smoke_float.onnx")
onnx.checker.check_model(mo)
print("  opset:", [f"{o.domain or 'ai.onnx'}:{o.version}" for o in mo.opset_import], "| ir:", mo.ir_version)
sess = ort.InferenceSession("/tmp/qax_smoke_float.onnx", providers=["CPUExecutionProvider"])
ref = fm(*ex).detach().numpy()
got = sess.run(None, {sess.get_inputs()[0].name: ex[0].numpy()})[0]
print("  ORT parity max|diff| =", float(np.abs(ref - got).max()))

print("== torchao QAT -> convert -> dynamo export QDQ ==")
from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e, convert_pt2e
from torchao.testing.pt2e._xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
gm = torch.export.export(Tiny().train(), ex).module()
prep = prepare_qat_pt2e(gm, XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_qat=True)))
opt = torch.optim.SGD([p for p in prep.parameters() if p.requires_grad], lr=1e-3)
for _ in range(2):
    opt.zero_grad(); prep(*ex).sum().backward(); opt.step()
conv = convert_pt2e(prep)
from torchao.quantization.pt2e import move_exported_model_to_eval
move_exported_model_to_eval(conv)
try:
    ep2 = torch.onnx.export(conv, ex, dynamo=True)
    ep2.save("/tmp/qax_smoke_qdq.onnx")
    mo2 = onnx.load("/tmp/qax_smoke_qdq.onnx")
    qdq = sorted({n.op_type for n in mo2.graph.node if "Linear" in n.op_type or "Quant" in n.op_type})
    print("  QDQ export OK, quant ops:", qdq)
    s2 = ort.InferenceSession("/tmp/qax_smoke_qdq.onnx", providers=["CPUExecutionProvider"])
    r2 = conv(*ex).detach().numpy()
    g2 = s2.run(None, {s2.get_inputs()[0].name: ex[0].numpy()})[0]
    print("  QDQ ORT parity max|diff| =", float(np.abs(r2 - g2).max()))
except Exception as e:
    print(f"  QDQ export FAIL (迁移重点): {type(e).__name__}: {e}")
print("== DONE ==")
