import torch  # Version: 2.6.0+cu118
import torch.nn as nn
import numpy as np

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
    remove_reused_bn_param_hack,
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


def test():
    # example inputs
    from reuse_conv.train import Net 
    input = torch.rand(1, 64, 256, 768).to("cuda")

    # float_model
    float_model = Net().to("cuda")
    float_path = "./reuse_conv/tmp_float.onnx"
    dynamo_export(float_model, input, float_path)

    # set quantizer
    global_config, regional_configs = load_config("./reuse_conv/config.json")
    quantizer = AXQuantizer(annotate_bias=False)
    quantizer.set_global(global_config)
    quantizer.set_regional(regional_configs)

    # export qat model
    exported_model = torch.export.export_for_training(float_model, (input,)).module()
    prepared_model = prepare_qat_pt2e(exported_model, quantizer)
    prepared_model.load_state_dict(torch.load("./reuse_conv/tmp.pth"))
    
    # convert
    remove_reused_bn_param_hack(prepared_model)
    quantized_model = convert_pt2e(prepared_model)

    # test
    input = torch.tensor(np.load("./reuse_conv/input.npy")).to("cuda")
    gt = torch.tensor(np.load("./reuse_conv/gt.npy")).to("cuda")
    pd = quantized_model(input)
    np.testing.assert_equal(gt.cpu().numpy(), pd.cpu().numpy())
    print("gt & pd assert equal")


if __name__ == "__main__":
    test()