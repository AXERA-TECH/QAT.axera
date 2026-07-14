"""axquant 统一 API 版(与 test_resnet.py 对应):加载 train_resnet_axquant 产物做分段推理评测。

相对 2.6 版的改动:
  1. axquant 统一 API(内部自动路由 torch.ao/torchao);capture 与训练侧一致(动态 batch,保证
     prepared state_dict 键对齐);
  2. ORT 输入名动态获取(2.6 硬编码 "x_0",2.10 导出输入名不同);
  3. 数据 fake_data;评测结果补打印(原版收集后未打印);
  4. 加载 *_2_10 产物。

运行(qat-dev,先跑 train_resnet_2_10.py):
  cd /home/heqi/project-qat/QAT.axera && PYTHONPATH=. CUDA_VISIBLE_DEVICES=<空卡> \
    /home/heqi/miniforge3/envs/torch2.10/bin/python reuse_conv/test_resnet_2_10.py
"""
import copy

import torch
import onnxruntime as ort

from axquant import (
    prepare_qat_pt2e,
    convert_pt2e,
    move_exported_model_to_eval,
    capture,
    load_config,
    AXQuantizer,
    evaluate,
    evaluate_np,
    imagenet_data_loaders,
)
from reuse_conv.train_resnet import (
    ResNetStage1,
    ResNetStage2,
    ResNetStage3,
    ResNetMultiStage,
    Bottleneck,
)

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning, module=r'.*')


def test():
    example_inputs_stage1 = (torch.rand(1, 3, 224, 224).to("cuda"),)
    example_inputs_stage2 = (torch.rand(1, 256, 56, 56).to("cuda"),)
    example_inputs_stage3 = (torch.rand(1, 256, 56, 56).to("cuda"),)
    # float model
    float_model_stage1 = ResNetStage1(Bottleneck, [3, 4, 6, 3]).to("cuda")
    float_model_stage2 = ResNetStage2(Bottleneck, [3, 4, 6, 3]).to("cuda")
    float_model_stage3 = ResNetStage3(Bottleneck, [3, 4, 6, 3]).to("cuda")
    state_dict = torch.load("./resnet50/resnet50_pretrained_float.pth", weights_only=True)
    float_model_stage1.load_state_dict(state_dict)
    # float_model_stage2.load_state_dict(state_dict)
    float_model_stage3.load_state_dict(state_dict)

    # quantizer
    global_config, regional_configs = load_config("./reuse_conv/config.json")
    quantizer = AXQuantizer("./reuse_conv/config.json", annotate_bias=False)

    # quant model(capture 与训练一致 → state_dict 键对齐)
    prepared_model_stage1 = prepare_qat_pt2e(
        capture(float_model_stage1.train(), example_inputs_stage1, dynamic_batch=True), quantizer)
    prepared_model_stage2 = prepare_qat_pt2e(
        capture(float_model_stage2.train(), example_inputs_stage2, dynamic_batch=True), quantizer)
    prepared_model_stage3 = prepare_qat_pt2e(
        capture(float_model_stage3.train(), example_inputs_stage3, dynamic_batch=True), quantizer)
    prepared_model_stage1.load_state_dict(
        torch.load("./reuse_conv/resnet50_stage1_ax.pth", weights_only=True))
    prepared_model_stage2.load_state_dict(
        torch.load("./reuse_conv/resnet50_stage2_ax.pth", weights_only=True))
    prepared_model_stage3.load_state_dict(
        torch.load("./reuse_conv/resnet50_stage3_ax.pth", weights_only=True))
    model = ResNetMultiStage(
        Bottleneck,
        [3, 4, 6, 3],
        stage1=prepared_model_stage1,
        stage2=prepared_model_stage2,
        stage3=prepared_model_stage3,
    ).to("cuda")
    model.load_state_dict(torch.load("./reuse_conv/resnet50_ax.pth", weights_only=True))

    float_stage = copy.deepcopy(model)
    float_stage.forward = float_stage._float_forward
    quantized_model_stage1 = convert_pt2e(prepared_model_stage1)
    quantized_model_stage2 = convert_pt2e(prepared_model_stage2)
    quantized_model_stage3 = convert_pt2e(prepared_model_stage3)

    # onnx session(输入名动态获取)
    sess_stage1 = ort.InferenceSession("./reuse_conv/resnet50_qat_sim_stage1_ax.onnx",
                                       providers=["CPUExecutionProvider"])
    sess_stage2 = ort.InferenceSession("./reuse_conv/resnet50_qat_sim_stage2_ax.onnx",
                                       providers=["CPUExecutionProvider"])
    sess_stage3 = ort.InferenceSession("./reuse_conv/resnet50_qat_sim_stage3_ax.onnx",
                                       providers=["CPUExecutionProvider"])
    in1 = sess_stage1.get_inputs()[0].name
    in2 = sess_stage2.get_inputs()[0].name
    in3 = sess_stage3.get_inputs()[0].name

    # dataset(fake_data:机器无 ImageNet)
    data_loader, data_loader_test = imagenet_data_loaders("dataset/imagenet/", fake_data=True)

    # evaluate
    def quantized_model_forward(x):
        float_stage.eval()
        move_exported_model_to_eval(quantized_model_stage1)
        move_exported_model_to_eval(quantized_model_stage2)
        move_exported_model_to_eval(quantized_model_stage3)

        x = quantized_model_stage1(x)
        for i in range(2):
            x = float_stage(x)
            x = quantized_model_stage2(x)
        x = quantized_model_stage3(x)

        return x

    def sess_forward(x):
        float_stage.eval()
        x = sess_stage1.run(None, {in1: x})[0]
        for i in range(2):
            x = torch.tensor(x).to("cuda")
            x = float_stage(x)
            x = x.cpu().numpy()
            x = sess_stage2.run(None, {in2: x})[0]
        x = sess_stage3.run(None, {in3: x})[0]

        return x

    top1, top5 = evaluate(quantized_model_forward, data_loader_test, total_size=100)
    print(f"[eval torch] top1={top1.avg:.3f} top5={top5.avg:.3f}")
    top1, top5 = evaluate_np(sess_forward, data_loader_test, total_size=100)
    print(f"[eval onnx ] top1={top1.avg:.3f} top5={top5.avg:.3f}")


if __name__ == "__main__":
    test()
