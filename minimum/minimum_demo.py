import torch  # Version: 2.6.0+cu118
import torch.nn as nn

from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from torch.ao.quantization.quantize_pt2e import (
    prepare_qat_pt2e,
    convert_pt2e,
)

# from utils.quantizer import (
#     AXQuantizer,
#     get_quantization_config,
# )
from utils.ax_quantizer import(
    load_config,
    AXQuantizer,
)
from utils.train_utils import dynamo_export, onnx_simplify
import utils.quantized_decomposed_dequantize_per_channel

# Set up warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.ao.quantization'
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
float_path = "./minimum/minimum_float.onnx"
dynamo_export(float_model, input, float_path)

# set quantizer
global_config, regional_configs = load_config("./minimum/config.json")
quantizer = AXQuantizer()
quantizer.set_global(global_config)
quantizer.set_regional(regional_configs)

# export qat model
exported_model = torch.export.export_for_training(float_model, (input,)).module()
prepared_model = prepare_qat_pt2e(exported_model, quantizer)
quantized_model = convert_pt2e(prepared_model)

# export
qat_path = "./minimum/minimum_qat.onnx"
dynamo_export(quantized_model, input, qat_path)

# onnx simplify
sim_path = "./minimum/minimum_qat_sim.onnx"
onnx_simplify(qat_path, sim_path)
