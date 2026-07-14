"""交叉实验 —— axquant 统一 API 版:同一份 QAT checkpoint 在两个环境各自
convert+导出,验证「observer 状态 → 量化参数 → 导出 QDQ」计算跨体系等价。

用法(两个环境各跑一次,--checkpoint 指向同一个文件):
  cd /home/heqi/project-qat/QAT.axera && PYTHONPATH=. \
    <env>/bin/python resnet50_2_10/cross_export.py \
    --checkpoint ./resnet50_2_10/checkpoint/last_checkpoint_2_6.pth
产物: resnet50_2_10/resnet50_qat_cross_{2_6|2_10}[_sim].onnx
"""
import argparse

import torch

from axquant import (
    IS_TORCH_210,
    AXQuantizer,
    capture,
    prepare_qat_pt2e,
    convert_pt2e,
    load_model,
    dynamo_export,
    simplify_and_fix_4bit_dtype,
)

TAG = "2_10" if IS_TORCH_210 else "2_6"  # 仅用于产物文件名
OUT_DIR = "./resnet50_2_10"


def main(args):
    torch.manual_seed(args.seed)

    float_model = load_model("./resnet50/resnet50_pretrained_float.pth", "resnet50").to("cuda")
    if args.num_classes != 1000:
        torch.manual_seed(args.seed)
        float_model.fc = torch.nn.Linear(float_model.fc.in_features, args.num_classes).to("cuda")

    quantizer = AXQuantizer(args.config)
    example_inputs = (torch.rand(1, 3, 224, 224).to("cuda"),)
    exported_model = capture(float_model.train(), example_inputs, dynamic_batch=True)
    prepared_model = prepare_qat_pt2e(exported_model, quantizer)

    # strict 加载:键不匹配会直接报错,本身就是跨体系兼容性的检验点
    state = torch.load(args.checkpoint, weights_only=True)
    prepared_model.load_state_dict(state)
    print(f"[cross] checkpoint 加载成功({len(state)} 个键,strict)")

    quantized_model = convert_pt2e(prepared_model)

    qat_path = f"{OUT_DIR}/resnet50_qat_cross_{TAG}.onnx"
    dynamo_export(quantized_model, example_inputs, qat_path)
    sim_path = f"{OUT_DIR}/resnet50_qat_cross_{TAG}_sim.onnx"
    simplify_and_fix_4bit_dtype(qat_path, sim_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="./resnet50_2_10/config.json")
    parser.add_argument("--num-classes", type=int, default=10, help="须与 checkpoint 训练时一致")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(f"[env] torch {torch.__version__} → {TAG} | ckpt={args.checkpoint}")
    main(args)
