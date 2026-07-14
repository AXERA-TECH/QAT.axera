import warnings; warnings.filterwarnings("ignore")
import torch
from torch.ao.quantization.quantize_pt2e import prepare_qat_pt2e, convert_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer, get_symmetric_quantization_config)

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(8)
        self.act = torch.nn.ReLU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

ex = (torch.randn(2, 3, 32, 32),)

def try_flow(name, gm_fn):
    try:
        gm = gm_fn()
        q = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_qat=True))
        prep = prepare_qat_pt2e(gm, q)
        prep(*ex)
        conv = convert_pt2e(prep)
        conv(*ex)
        print(f"[{name}] OK")
    except Exception as e:
        print(f"[{name}] FAIL: {type(e).__name__}: {e}")

try_flow("export_for_training", lambda: torch.export.export_for_training(Tiny().train(), ex).module())
try_flow("torch.export.export", lambda: torch.export.export(Tiny().train(), ex).module())
try_flow("export strict=False", lambda: torch.export.export(Tiny().train(), ex, strict=False).module())
