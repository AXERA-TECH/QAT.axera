"""resnet50 QAT 评测 —— torch2.6 / torch2.10 双环境等价脚本(P2)。

加载 train.py 产出的 checkpoint,convert 后分别用 torch 与 onnxruntime 评测,
用于确认「训练态 QAT 模型」与「导出 ONNX」数值/精度一致。
数据选项与 train.py 相同(fake / cifar10)。
"""
import argparse

import torch
import onnxruntime as ort

_ver = tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2])
IS_210 = _ver >= (2, 10)
TAG = "2_10" if IS_210 else "2_6"

if IS_210:
    from torchao.quantization.pt2e.quantize_pt2e import (
        prepare_qat_pt2e,
        convert_pt2e,
    )
    from utils_2_10.ax_quantizer import AXQuantizer
    from utils_2_10.train_utils import (
        load_model,
        evaluate,
        evaluate_np,
        imagenet_data_loaders,
        cifar10_data_loaders,
    )
    import utils_2_10.quantized_decomposed_dequantize_per_channel  # noqa: F401
else:
    from torch.ao.quantization.quantize_pt2e import (
        prepare_qat_pt2e,
        convert_pt2e,
    )
    from utils.ax_quantizer import AXQuantizer
    from utils.train_utils import (
        load_model,
        evaluate,
        evaluate_np,
        imagenet_data_loaders,
        cifar10_data_loaders,
    )
    import utils.quantized_decomposed_dequantize_per_channel  # noqa: F401


OUT_DIR = "./resnet50_2_10"


def capture(model: torch.nn.Module, example_inputs):
    # batch 维声明动态 + 0/1 特化规避(捕获样例 batch>=2),原因见 train.py capture()
    if IS_210:
        x = example_inputs[0]
        capture_inputs = (torch.cat([x, x], dim=0) if x.shape[0] == 1 else x,)
        # 上界须显式(guard 推出 batch < 2^31/单样本元素数),见 train.py
        batch = torch.export.Dim("batch", min=1, max=1024)
        return torch.export.export(
            model, capture_inputs, dynamic_shapes=({0: batch},)).module()
    return torch.export.export_for_training(model, example_inputs).module()


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

    # quant model
    example_inputs = (torch.rand(1, 3, 224, 224).to("cuda"),)
    exported_model = capture(float_model.train(), example_inputs)
    prepared_model = prepare_qat_pt2e(exported_model, quantizer)

    prepared_model.load_state_dict(torch.load(f"{OUT_DIR}/checkpoint/last_checkpoint_{TAG}.pth"))
    quantized_model = convert_pt2e(prepared_model)

    # onnx session
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
    parser.add_argument("--config", type=str, default="./resnet50_2_10/config.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-size", type=int, default=20, help="验证 batch 数,0=全量")
    args = parser.parse_args()
    print(f"[env] torch {torch.__version__} → {TAG} 实现 | data={args.data}")
    test(args)
