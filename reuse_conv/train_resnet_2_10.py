"""torch 2.10 版(与 train_resnet.py 对应):stage2 运行时循环复用的分段 QAT 训练。

相对 2.6 版的改动:
  1. torchao PT2E + utils_2_10;模型类(ResNetFloat/Stage1/2/3/MultiStage)
     无 torch.ao 依赖,直接从原模块 import 复用;
  2. export_for_training → torch.export.export + 动态 batch(训练 batch=32);
  3. float 参考导出用 eval 深拷贝(2.10 不能直接导训练态 BN);
  4. 数据:机器无 ImageNet 训练集 → imagenet_data_loaders(fake_data=True);
  5. 修正原版导出处的双层 tuple 笔误 dynamo_export(model, (example_inputs,));
  6. 产物带 _2_10 后缀。

运行(qat-dev):
  cd /home/heqi/project-qat/QAT.axera && PYTHONPATH=. CUDA_VISIBLE_DEVICES=<空卡> \
    /home/heqi/miniforge3/envs/torch2.10/bin/python reuse_conv/train_resnet_2_10.py
"""
import copy

import torch

from torchao.quantization.pt2e.quantize_pt2e import (
    prepare_qat_pt2e,
    convert_pt2e,
)
from torchao.quantization.pt2e import move_exported_model_to_eval

from utils_2_10.ax_quantizer import (
    load_config,
    AXQuantizer,
)
from utils_2_10.train_utils import (
    load_model,
    train_one_epoch,
    imagenet_data_loaders,
    dynamo_export,
    onnx_simplify,
    evaluate,
)
import utils_2_10.quantized_decomposed_dequantize_per_channel  # noqa: F401
from reuse_conv.train_resnet import (
    ResNetFloat,
    ResNetStage1,
    ResNetStage2,
    ResNetStage3,
    ResNetMultiStage,
    Bottleneck,
)

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning, module=r'.*')


def capture(model, example_inputs):
    # 动态 batch 捕获,原因见 resnet50_2_10/train.py capture()
    x = example_inputs[0]
    capture_inputs = (torch.cat([x, x], dim=0) if x.shape[0] == 1 else x,)
    batch = torch.export.Dim("batch", min=1, max=1024)
    return torch.export.export(
        model, capture_inputs, dynamic_shapes=({0: batch},)).module()


def train():
    # load data(fake_data:机器无 ImageNet 训练集;fake_train_size 需 > 每
    # epoch 步数×batch,否则 loader 提前耗尽)
    data_loader, data_loader_test = imagenet_data_loaders(
        "dataset/imagenet/", fake_data=True, fake_train_size=3200)
    example_inputs_stage1 = (torch.rand(1, 3, 224, 224).to("cuda"),)
    example_inputs_stage2 = (torch.rand(1, 256, 56, 56).to("cuda"),)
    example_inputs_stage3 = (torch.rand(1, 256, 56, 56).to("cuda"),)

    # set float model
    float_model = ResNetFloat(Bottleneck, [3, 4, 6, 3]).to("cuda")
    float_model_stage1 = ResNetStage1(Bottleneck, [3, 4, 6, 3]).to("cuda")
    float_model_stage2 = ResNetStage2(Bottleneck, [3, 4, 6, 3]).to("cuda")
    float_model_stage3 = ResNetStage3(Bottleneck, [3, 4, 6, 3]).to("cuda")
    state_dict = torch.load("./resnet50/resnet50_pretrained_float.pth", weights_only=True)
    float_model_stage1.load_state_dict(state_dict)
    # float_model_stage2.load_state_dict(state_dict)
    float_model_stage3.load_state_dict(state_dict)

    # float 参考导出(eval 深拷贝,2.10 不能直接导训练态 BN)
    dynamo_export(copy.deepcopy(float_model).eval(), example_inputs_stage1,
                  "./reuse_conv/resnet50_float_2_10.onnx")
    dynamo_export(copy.deepcopy(float_model_stage1).eval(), example_inputs_stage1,
                  "./reuse_conv/resnet50_float_stage1_2_10.onnx")
    dynamo_export(copy.deepcopy(float_model_stage2).eval(), example_inputs_stage2,
                  "./reuse_conv/resnet50_float_stage2_2_10.onnx")
    dynamo_export(copy.deepcopy(float_model_stage3).eval(), example_inputs_stage3,
                  "./reuse_conv/resnet50_float_stage3_2_10.onnx")

    # quantizer
    global_config, regional_configs = load_config("./reuse_conv/config.json")
    quantizer = AXQuantizer("./reuse_conv/config.json", annotate_bias=False)

    exported_model_stage1 = capture(float_model_stage1.train(), example_inputs_stage1)
    exported_model_stage2 = capture(float_model_stage2.train(), example_inputs_stage2)
    exported_model_stage3 = capture(float_model_stage3.train(), example_inputs_stage3)
    prepared_model_stage1 = prepare_qat_pt2e(exported_model_stage1, quantizer)
    prepared_model_stage2 = prepare_qat_pt2e(exported_model_stage2, quantizer)
    prepared_model_stage3 = prepare_qat_pt2e(exported_model_stage3, quantizer)
    model = ResNetMultiStage(
        Bottleneck,
        [3, 4, 6, 3],
        stage1=prepared_model_stage1,
        stage2=prepared_model_stage2,
        stage3=prepared_model_stage3,
    ).to("cuda")

    num_epochs = 5
    num_train_batches = 50
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # 更小的学习率

    # train
    for nepoch in range(num_epochs):
        train_one_epoch(model, criterion, optimizer, data_loader, "cuda", num_train_batches)

    torch.save(model.state_dict(), "./reuse_conv/resnet50_2_10.pth")
    torch.save(model.stage1.state_dict(), "./reuse_conv/resnet50_stage1_2_10.pth")
    torch.save(model.stage2.state_dict(), "./reuse_conv/resnet50_stage2_2_10.pth")
    torch.save(model.stage3.state_dict(), "./reuse_conv/resnet50_stage3_2_10.pth")

    # evaluate
    float_stage = copy.deepcopy(model)
    float_stage.forward = float_stage._float_forward
    quantized_model_stage1 = convert_pt2e(model.stage1)
    quantized_model_stage2 = convert_pt2e(model.stage2)
    quantized_model_stage3 = convert_pt2e(model.stage3)

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
    top1, top5 = evaluate(quantized_model_forward, data_loader_test, total_size=100)
    print(f"[eval] 分段量化(fake data): top1={top1.avg:.3f} top5={top5.avg:.3f}")

    # export(修正原版双层 tuple 笔误)
    qat_path_stage1 = "./reuse_conv/resnet50_qat_stage1_2_10.onnx"
    dynamo_export(quantized_model_stage1, example_inputs_stage1, qat_path_stage1)
    qat_path_stage2 = "./reuse_conv/resnet50_qat_stage2_2_10.onnx"
    dynamo_export(quantized_model_stage2, example_inputs_stage2, qat_path_stage2)
    qat_path_stage3 = "./reuse_conv/resnet50_qat_stage3_2_10.onnx"
    dynamo_export(quantized_model_stage3, example_inputs_stage3, qat_path_stage3)

    # onnx simplify
    onnx_simplify(qat_path_stage1, "./reuse_conv/resnet50_qat_sim_stage1_2_10.onnx")
    onnx_simplify(qat_path_stage2, "./reuse_conv/resnet50_qat_sim_stage2_2_10.onnx")
    onnx_simplify(qat_path_stage3, "./reuse_conv/resnet50_qat_sim_stage3_2_10.onnx")


if __name__ == "__main__":
    train()
