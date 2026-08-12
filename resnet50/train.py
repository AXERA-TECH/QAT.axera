"""resnet50 QAT 训练(torch 2.10,utils 统一 API)。

数据:
  --data fake     FakeData,零下载,用于结构回归(默认)
  --data cifar10  CIFAR-10,数据放 dataset/cifar10/cifar-10-python.tar.gz
                  (md5 c58f30108f718f92721af3b95e74349a)

运行:
  cd /home/heqi/project-qat/QAT.axera && \
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=<空卡> \
    <env>/bin/python resnet50/train.py --data fake --config ./resnet50/config_4w4f.json
"""
import argparse
import os

import torch
from torch.export import Dim
from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e, convert_pt2e
from torchao.quantization.pt2e import move_exported_model_to_eval

from utils.ax_quantizer import AXQuantizer
from utils.train_utils import (
    load_model,
    train_one_epoch,
    imagenet_data_loaders,
    cifar10_data_loaders,
    dynamo_export,
    evaluate,
)
from utils.quant_utils import simplify_and_fix_4bit_dtype
import utils.quantized_decomposed_dequantize_per_channel  # noqa: F401 注册 per-channel torchlib 映射

OUT_DIR = "./resnet50"


def count_fake_quants(prepared) -> int:
    return sum(1 for node in prepared.graph.nodes
               if node.op == "call_module"
               and str(node.target).startswith("activation_post_process"))


def build_dataloaders(args):
    if args.data == "cifar10":
        data_loader, data_loader_test = cifar10_data_loaders(
            "dataset/cifar10", train_batch_size=args.batch_size)
        num_classes = 10
    else:
        data_loader, data_loader_test = imagenet_data_loaders(
            "dataset/imagenet/", train_batch_size=args.batch_size, fake_data=True)
        num_classes = 1000
    return data_loader, data_loader_test, num_classes


def config_suffix(config_path: str) -> str:
    """按配置名区分产物,如 config_4w4f.json → \"_4w4f\",config.json → \"\"。"""
    stem = os.path.splitext(os.path.basename(config_path))[0]
    if stem.startswith("config"):
        stem = stem[len("config"):].strip("_")
    return f"_{stem}" if stem else ""


def train(args):
    torch.manual_seed(args.seed)
    os.makedirs(os.path.join(OUT_DIR, "checkpoint"), exist_ok=True)
    cfg = config_suffix(args.config)

    data_loader, data_loader_test, num_classes = build_dataloaders(args)
    example_inputs = (torch.rand(2, 3, 224, 224).to("cuda"),)

    float_model = load_model("./resnet50/resnet50_pretrained_float.pth", "resnet50").to("cuda")
    if num_classes != 1000:
        # cifar10 换 10 类头;重设种子保证两个环境的 fc 初始化一致
        torch.manual_seed(args.seed)
        float_model.fc = torch.nn.Linear(float_model.fc.in_features, num_classes).to("cuda")

    # float 参考导出(与 QAT 导出同路径 dynamo_export,training 图结构)
    float_path = f"{OUT_DIR}/resnet50_float{cfg}.onnx"
    dynamo_export(float_model, example_inputs, float_path)

    # quantizer
    quantizer = AXQuantizer(args.config)

    # 训练/评测的 batch 与 example 不同 → dynamic_shapes 声明 batch 动态
    # (example batch 给 >=2,规避动态维 example=1 的 0/1 特化)
    dynamic_shapes = ({0: Dim("batch", min=1, max=1024)},)
    exported_model = torch.export.export(
        float_model.train(), example_inputs, dynamic_shapes=dynamic_shapes,
    ).module()
    prepared_model_qat = prepare_qat_pt2e(exported_model, quantizer)

    # 防静默漏注解(plan 风险 3):resnet50 预期插入 ~130 个观察点,阈值取宽松下限
    n_fq = count_fake_quants(prepared_model_qat)
    print(f"[assert] fake-quant 插入数量: {n_fq}")
    assert n_fq >= 100, f"注解疑似漏失: 仅插入 {n_fq} 个 fake-quant(预期 >= 100)"

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(prepared_model_qat.parameters(), lr=0.001, momentum=0.9)

    # train(train_one_epoch 逐 batch 打印 loss,等价性对照即比对这些曲线)
    for nepoch in range(args.epochs):
        train_one_epoch(prepared_model_qat, criterion, optimizer, data_loader, "cuda", args.steps)
        checkpoint_path = f"{OUT_DIR}/checkpoint/checkpoint{cfg}_{nepoch}.pth"
        torch.save(prepared_model_qat.state_dict(), checkpoint_path)

    torch.save(prepared_model_qat.state_dict(), f"{OUT_DIR}/checkpoint/last_checkpoint{cfg}.pth")

    # evaluate(可选,--eval-size 0 跳过)
    quantized_model = convert_pt2e(prepared_model_qat)
    # FP32 区域(未量化)的 conv+BN 不会被 convert 折叠,残留训练态 BN 需转 eval
    # 后再导出;对全量化配置本调用是无操作
    move_exported_model_to_eval(quantized_model)
    if args.eval_size > 0:
        top1, top5 = evaluate(quantized_model, data_loader_test, total_size=args.eval_size)
        print(f"[eval] quantized top1={top1.avg:.3f} top5={top5.avg:.3f} (n={args.eval_size})")

    # export(部署产物:固定 shape,example batch=1 且不声明动态维;
    qat_path = f"{OUT_DIR}/resnet50_qat{cfg}.onnx"
    export_example_inputs = (torch.rand(1, 3, 224, 224).to("cuda"),)
    dynamo_export(quantized_model, export_example_inputs, qat_path)

    # onnx simplify & fix dtype
    sim_path = f"{OUT_DIR}/resnet50_qat{cfg}_sim.onnx"
    simplify_and_fix_4bit_dtype(qat_path, sim_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["fake", "cifar10"], default="fake")
    parser.add_argument("--config", type=str, default="./resnet50/config.json")
    parser.add_argument("--steps", type=int, default=10, help="每个 epoch 训练 batch 数")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-size", type=int, default=0, help="验证 batch 数,0 跳过")
    args = parser.parse_args()
    print(f"[env] torch {torch.__version__} | data={args.data} config={args.config} "
          f"steps={args.steps} seed={args.seed}")
    train(args)
