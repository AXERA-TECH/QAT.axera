"""resnet50 QAT 评测 —— utils 统一 API 版(torch 2.6 / 2.10 同一份代码)。

加载 train.py 产出的 checkpoint,convert 后分别用 torch 与 onnxruntime 评测,
用于确认「训练态 QAT 模型」与「导出 ONNX」数值/精度一致。
"""
import argparse

import torch
import onnxruntime as ort

from utils import (
    IS_TORCH_210,
    AXQuantizer,
    capture,
    prepare_qat_pt2e,
    convert_pt2e,
    load_model,
    evaluate,
    evaluate_np,
    imagenet_data_loaders,
    cifar10_data_loaders,
)

TAG = "2_10" if IS_TORCH_210 else "2_6"  # 仅用于产物文件名
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

    # quant model(capture 与训练一致 → state_dict 键对齐)
    example_inputs = (torch.rand(1, 3, 224, 224).to("cuda"),)
    exported_model = capture(float_model.train(), example_inputs, dynamic_batch=True)
    prepared_model = prepare_qat_pt2e(exported_model, quantizer)

    prepared_model.load_state_dict(
        torch.load(f"{OUT_DIR}/checkpoint/last_checkpoint_{TAG}.pth", weights_only=True))
    quantized_model = convert_pt2e(prepared_model)

    # onnx session(输入名动态获取)
    sess = ort.InferenceSession(f"{OUT_DIR}/resnet50_qat_{TAG}.onnx",
                                providers=["CPUExecutionProvider"])

    # evaluate
    top1, top5 = evaluate(quantized_model, data_loader_test, total_size=args.eval_size or None)
    print(f"[torch ] top1={top1.avg:.3f} top5={top5.avg:.3f}")
    top1, top5 = evaluate_np(sess, data_loader_test, total_size=args.eval_size or None)
    print(f"[onnx  ] top1={top1.avg:.3f} top5={top5.avg:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["fake", "cifar10"], default="fake")
    parser.add_argument("--config", type=str, default="./resnet50/config.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-size", type=int, default=20, help="验证 batch 数,0=全量")
    args = parser.parse_args()
    print(f"[env] torch {torch.__version__} → {TAG} | data={args.data}")
    test(args)
