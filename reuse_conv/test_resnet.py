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
    remove_reused_bn_param_hack,
)
from utils.train_utils import (
  load_model,
  evaluate,
  evaluate_np,
  imagenet_data_loaders,
)
from reuse_conv.train_resnet import(
    ResNet_stage1,
    ResNet_stage2,
    ResNet_stage3,
    Bottleneck,
)

def test():
    example_inputs_stage1 = (torch.rand(1, 3, 224, 224).to("cuda"),)
    example_inputs_stage2 = (torch.rand(1, 256, 56, 56).to("cuda"),)
    example_inputs_stage3 = (torch.rand(1, 256, 56, 56).to("cuda"),)
    # float model
    # float_model = load_model("./resnet50/resnet50_pretrained_float.pth", "resnet50").to("cuda")
    float_model_stage1 = ResNet_stage1(Bottleneck, [3, 4, 6, 3]).to("cuda")
    float_model_stage2 = ResNet_stage2(Bottleneck, [3, 4, 6, 3]).to("cuda")
    float_model_stage3 = ResNet_stage3(Bottleneck, [3, 4, 6, 3]).to("cuda")
    state_dict = torch.load("./resnet50/resnet50_pretrained_float.pth", weights_only=True)
    float_model_stage1.load_state_dict(state_dict)
    # float_model_stage2.load_state_dict(state_dict)
    float_model_stage3.load_state_dict(state_dict)

    # quantizer
    global_config, regional_configs = load_config("./reuse_conv/config.json")
    quantizer = AXQuantizer(annotate_bias=False)
    quantizer.set_global(global_config)
    quantizer.set_regional(regional_configs)

    # quant model
    exported_model_stage1 = torch.export.export_for_training(float_model_stage1, example_inputs_stage1).module()
    exported_model_stage2 = torch.export.export_for_training(float_model_stage2, example_inputs_stage2).module()
    exported_model_stage3 = torch.export.export_for_training(float_model_stage3, example_inputs_stage3).module()
    prepared_model_stage1 = prepare_qat_pt2e(exported_model_stage1, quantizer)
    prepared_model_stage2 = prepare_qat_pt2e(exported_model_stage2, quantizer)
    prepared_model_stage3 = prepare_qat_pt2e(exported_model_stage3, quantizer)
    prepared_model_stage1.load_state_dict(torch.load("./reuse_conv/resnet50_stage1.pth"))
    prepared_model_stage2.load_state_dict(torch.load("./reuse_conv/resnet50_stage2.pth"))
    prepared_model_stage3.load_state_dict(torch.load("./reuse_conv/resnet50_stage3.pth"))
    quantized_model_stage1 = convert_pt2e(prepared_model_stage1)
    quantized_model_stage2 = convert_pt2e(prepared_model_stage2)
    quantized_model_stage3 = convert_pt2e(prepared_model_stage3)

    # onnx session
    sess_stage1 = ort.InferenceSession("./reuse_conv/resnet50_qat_sim_stage1.onnx")
    sess_stage2 = ort.InferenceSession("./reuse_conv/resnet50_qat_sim_stage2.onnx")
    sess_stage3 = ort.InferenceSession("./reuse_conv/resnet50_qat_sim_stage3.onnx")

    # dataset
    data_loader, data_loader_test = imagenet_data_loaders("dataset/imagenet/")

    # evaluate
    def quantized_model_forward(x):
        torch.ao.quantization.move_exported_model_to_eval(quantized_model_stage1)
        torch.ao.quantization.move_exported_model_to_eval(quantized_model_stage2)
        torch.ao.quantization.move_exported_model_to_eval(quantized_model_stage3)

        x = quantized_model_stage1(x)
        for i in range(2):
            x = quantized_model_stage2(x)
        x = quantized_model_stage3(x)

        return x
    def sess_forward(x):
        x = sess_stage1.run(None, {"x_0": x})[0]
        for i in range(2):
            x = sess_stage2.run(None, {"x_0": x})[0]
        x = sess_stage3.run(None, {"x_0": x})[0]

        return x
    top1, top5 = evaluate(quantized_model_forward, data_loader_test, total_size=100)
    top1, top5 = evaluate_np(sess_forward, data_loader_test, total_size=100)


if __name__ == "__main__":
    test()