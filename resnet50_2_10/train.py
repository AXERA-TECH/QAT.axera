"""resnet50 QAT 训练 —— axquant 统一 API 版(torch 2.6 / 2.10 同一份代码)。

版本差异全部由 axquant 内部消化(capture/dynamo_export/export_float_reference),
本文件零版本分支;TAG 仅用于区分两个环境的产物文件名(等价性对照需要)。

数据(见 plan_torch210.md P2 数据方案):
  --data fake     FakeData,零下载,用于结构回归(默认)
  --data cifar10  CIFAR-10,用于 2.6/2.10 等价性对照;数据放
                  dataset/cifar10/cifar-10-python.tar.gz(md5 c58f30108f718f92721af3b95e74349a)

运行(qat-dev):
  cd /home/heqi/project-qat/QAT.axera && \
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=<空卡> \
    <env>/bin/python resnet50_2_10/train.py --data fake --config ./resnet50_2_10/config_4w4f_all.json
"""
import argparse
import os

import torch

from axquant import (
    IS_TORCH_210,
    AXQuantizer,
    capture,
    prepare_qat_pt2e,
    convert_pt2e,
    move_exported_model_to_eval,
    load_model,
    train_one_epoch,
    imagenet_data_loaders,
    cifar10_data_loaders,
    dynamo_export,
    export_float_reference,
    evaluate,
    simplify_and_fix_4bit_dtype,
)

TAG = "2_10" if IS_TORCH_210 else "2_6"  # 仅用于产物文件名
OUT_DIR = "./resnet50_2_10"


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
    example_inputs = (torch.rand(1, 3, 224, 224).to("cuda"),)

    float_model = load_model("./resnet50/resnet50_pretrained_float.pth", "resnet50").to("cuda")
    if num_classes != 1000:
        # cifar10 换 10 类头;重设种子保证两个环境的 fc 初始化一致
        torch.manual_seed(args.seed)
        float_model.fc = torch.nn.Linear(float_model.fc.in_features, num_classes).to("cuda")

    # float 参考导出(export_float_reference 内置 eval 深拷贝)
    float_path = f"{OUT_DIR}/resnet50_float_{TAG}{cfg}.onnx"
    export_float_reference(float_model, example_inputs, float_path)

    # quantizer
    quantizer = AXQuantizer(args.config)

    # 训练/评测的 batch 与 example 不同 → dynamic_batch=True(2.6 下为无操作)
    exported_model = capture(float_model.train(), example_inputs, dynamic_batch=True)
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
        checkpoint_path = f"{OUT_DIR}/checkpoint/checkpoint_{TAG}{cfg}_{nepoch}.pth"
        torch.save(prepared_model_qat.state_dict(), checkpoint_path)

    torch.save(prepared_model_qat.state_dict(), f"{OUT_DIR}/checkpoint/last_checkpoint_{TAG}{cfg}.pth")

    # evaluate(可选,--eval-size 0 跳过)
    quantized_model = convert_pt2e(prepared_model_qat)
    # FP32 区域(未量化)的 conv+BN 不会被 convert 折叠,残留训练态 BN 需转 eval
    # 后再导出;对全量化配置本调用是无操作
    move_exported_model_to_eval(quantized_model)
    if args.eval_size > 0:
        top1, top5 = evaluate(quantized_model, data_loader_test, total_size=args.eval_size)
        print(f"[eval] quantized top1={top1.avg:.3f} top5={top5.avg:.3f} (n={args.eval_size})")

    # export
    qat_path = f"{OUT_DIR}/resnet50_qat_{TAG}{cfg}.onnx"
    dynamo_export(quantized_model, example_inputs, qat_path)

    # onnx simplify & fix dtype
    sim_path = f"{OUT_DIR}/resnet50_qat_{TAG}{cfg}_sim.onnx"
    simplify_and_fix_4bit_dtype(qat_path, sim_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["fake", "cifar10"], default="fake")
    parser.add_argument("--config", type=str, default="./resnet50_2_10/config.json")
    parser.add_argument("--steps", type=int, default=10, help="每个 epoch 训练 batch 数")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-size", type=int, default=0, help="验证 batch 数,0 跳过")
    args = parser.parse_args()
    print(f"[env] torch {torch.__version__} → {TAG} | data={args.data} config={args.config} "
          f"steps={args.steps} seed={args.seed}")
    train(args)
