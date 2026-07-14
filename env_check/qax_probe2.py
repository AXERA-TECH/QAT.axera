import warnings; warnings.filterwarnings("ignore")
import torch, torchao
print("torchao", torchao.__version__)

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(8)
        self.act = torch.nn.ReLU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

ex = (torch.randn(2, 3, 32, 32),)

def run(name, prepare_fn, convert_fn, quantizer):
    try:
        gm = torch.export.export(Tiny().train(), ex).module()
        prep = prepare_fn(gm, quantizer)
        opt = torch.optim.SGD([p for p in prep.parameters() if p.requires_grad], lr=1e-3)
        for _ in range(2):
            opt.zero_grad(); prep(*ex).sum().backward(); opt.step()
        conv = convert_fn(prep)
        out = conv(*ex)
        ops = {n.target.__name__ for n in conv.graph.nodes if n.op == "call_function"}
        qops = sorted(o for o in ops if "quant" in o)
        print(f"[{name}] OK, out {tuple(out.shape)}, q-ops: {qops[:4]}")
    except Exception as e:
        print(f"[{name}] FAIL: {type(e).__name__}: {e}")

# torchao 自家 PT2E + torchao 版 quantizer 基类
from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e as ao_prep, convert_pt2e as ao_conv
try:
    from torchao.testing.pt2e._xnnpack_quantizer import (
        XNNPACKQuantizer as AOXQ, get_symmetric_quantization_config as ao_cfg)
    run("torchao pt2e + torchao XNNPACKQuantizer", ao_prep, ao_conv, AOXQ().set_global(ao_cfg(is_qat=True)))
except ImportError as e:
    print("torchao xnnpack quantizer not found:", e)

# torchao PT2E + torch.ao 老 XNNPACKQuantizer（看跨库兼容）
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer as TAXQ, get_symmetric_quantization_config as ta_cfg)
run("torchao pt2e + torch.ao XNNPACKQuantizer", ao_prep, ao_conv, TAXQ().set_global(ta_cfg(is_qat=True)))
