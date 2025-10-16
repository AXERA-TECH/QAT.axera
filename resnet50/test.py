import torch
import onnxruntime as ort

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
from utils.train_utils import (
  load_model,
  evaluate,
  evaluate_np,
  imagenet_data_loaders,
)


def test():
    # float model
    float_model = load_model("./resnet50/resnet50_pretrained_float.pth", "resnet50").to("cuda")

    # quantizer
    quantizer = AXQuantizer("./resnet50/config.json")

    # quant model
    example_inputs = (torch.rand(1, 3, 224, 224).to("cuda"),)
    exported_model = torch.export.export_for_training(float_model, example_inputs).module()
    prepared_model = prepare_qat_pt2e(exported_model, quantizer)

    prepared_model.load_state_dict(torch.load("./resnet50/checkpoint/last_checkpoint.pth"))
    quantized_model = convert_pt2e(prepared_model)

    # onnx session
    sess = ort.InferenceSession("./resnet50/resnet50_qat.onnx")

    # dataset
    data_loader, data_loader_test = imagenet_data_loaders("dataset/imagenet/")

    # evaluate
    top1, top5 = evaluate(quantized_model, data_loader_test)
    top1, top5 = evaluate_np(sess, data_loader_test)


if __name__ == "__main__":
    test()