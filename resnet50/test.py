"""resnet50 QAT 评测(torch 2.10,utils 统一 API)。

加载 train.py 产出的 checkpoint,convert 后分别用 torch 与 onnxruntime 评测,
用于确认「训练态 QAT 模型」与「导出 ONNX」数值/精度一致。
"""
import argparse

import torch
from torch.export import Dim
import onnxruntime as ort
from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e, convert_pt2e

from utils.ax_quantizer import AXQuantizer
from utils.train_utils import (
    load_model,
    evaluate,
    evaluate_np,
    imagenet_data_loaders,
    cifar10_data_loaders,
)
import utils.quantized_decomposed_dequantize_per_channel  # noqa: F401 注册 per-channel torchlib 映射

OUT_DIR = "./resnet50"


def test(args):
    torch.manual_seed(args.seed)

    # dataset
    if args.data == "cifar10":
        data_loader, data_loader_test = cifar10_data_loaders("dataset/cifar10")
        num_classes = 10
    else:
        data_loader, data_loader_test = imagenet_data_loaders("dataset/imagenet/", fake_data=True)
        num_classes = 1000

    # float model
    float_model = load_model("./resnet50/resnet50_pretrained_float.pth", "resnet50").to("cuda")
    if num_classes != 1000:
        torch.manual_seed(args.seed)
        float_model.fc = torch.nn.Linear(float_model.fc.in_features, num_classes).to("cuda")

    # quantizer
    quantizer = AXQuantizer(args.config)

    # capture 与训练一致 → state_dict 键对齐(batch 动态,example 给 >=2 规避 0/1 特化)
    example_inputs = (torch.rand(2, 3, 224, 224).to("cuda"),)
    dynamic_shapes = ({0: Dim("batch", min=1, max=1024)},)
    exported_model = torch.export.export(
        float_model.train(), example_inputs, dynamic_shapes=dynamic_shapes,
    ).module()
    prepared_model = prepare_qat_pt2e(exported_model, quantizer)

    prepared_model.load_state_dict(
        torch.load(f"{OUT_DIR}/checkpoint/last_checkpoint.pth", weights_only=True))
    quantized_model = convert_pt2e(prepared_model)

    # onnx session(输入名动态获取)
    sess = ort.InferenceSession(f"{OUT_DIR}/resnet50_qat.onnx",
                                providers=["CPUExecutionProvider"])

    # evaluate
    top1, top5 = evaluate(quantized_model, data_loader_test, total_size=args.eval_size or None)
    print(f"[torch ] top1={top1.avg:.3f} top5={top5.avg:.3f}")
    # ORT 评测:部署产物为固定 batch=1,喂 batch=1 数据
    # (cifar10 loader 默认 eval batch=32 与产物不匹配;imagenet fake 的 eval batch 本就是 1)
    if args.data == "cifar10":
        _, data_loader_ort = cifar10_data_loaders("dataset/cifar10", eval_batch_size=1)
    else:
        data_loader_ort = data_loader_test
    top1, top5 = evaluate_np(sess, data_loader_ort, total_size=args.eval_size or None)
    print(f"[onnx  ] top1={top1.avg:.3f} top5={top5.avg:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["fake", "cifar10"], default="fake")
    parser.add_argument("--config", type=str, default="./resnet50/config.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-size", type=int, default=20, help="验证 batch 数,0=全量")
    args = parser.parse_args()
    print(f"[env] torch {torch.__version__} | data={args.data}")
    test(args)
