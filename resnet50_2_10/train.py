"""resnet50 QAT 训练 —— torch2.6 / torch2.10 双环境等价脚本(P2)。

按运行时 torch 版本自动选择实现:
  torch >= 2.10 → utils_2_10(torchao 体系) + torch.export.export
  torch <  2.10 → utils(torch.ao 体系) + export_for_training
两个环境跑的是同一份数据管线/训练循环代码,便于做 2.6 vs 2.10 等价性对照
(同 seed 同数据同超参,对比逐 batch loss 曲线与最终导出结构)。

数据(P2 方案,见 plan_torch210.md):
  --data fake     FakeData,零下载,用于结构回归(默认)
  --data cifar10  CIFAR-10,用于 2.6/2.10 等价性对照;数据放
                  dataset/cifar10/cifar-10-python.tar.gz(md5 c58f30108f718f92721af3b95e74349a)

产物写入本目录,文件名带环境后缀(_2_6 / _2_10),不覆盖 resnet50/ 的 2.6 产物。

运行(qat-dev):
  cd /home/heqi/project-qat/QAT.axera && \
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=<空卡> \
    <env>/bin/python resnet50_2_10/train.py --data fake --config ./resnet50_2_10/config_4w4f_all.json
"""
import argparse
import copy
import os

import torch

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
        train_one_epoch,
        imagenet_data_loaders,
        cifar10_data_loaders,
        dynamo_export,
        evaluate,
    )
    from utils_2_10.quant_utils import simplify_and_fix_4bit_dtype
    from torchao.quantization.pt2e import move_exported_model_to_eval
    import utils_2_10.quantized_decomposed_dequantize_per_channel  # noqa: F401
else:
    from torch.ao.quantization.quantize_pt2e import (
        prepare_qat_pt2e,
        convert_pt2e,
    )
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
    from torch.ao.quantization import move_exported_model_to_eval
    import utils.quantized_decomposed_dequantize_per_channel  # noqa: F401


OUT_DIR = "./resnet50_2_10"


def capture(model: torch.nn.Module, example_inputs):
    """图捕获:2.10 用 torch.export.export(export_for_training 已废弃),2.6 保持原路径。

    2.10 的 export 会把 example_inputs 的 batch 烙成静态 guard,训练时换 batch
    直接 `Guard failed: x.size()[0] == 1`(2.6 不检查)→ 显式声明 batch 维动态;
    且 export 有 0/1 特化规则(动态维的 example 值必须 >= 2),捕获样例用 batch=2。
    """
    if IS_210:
        x = example_inputs[0]
        capture_inputs = (torch.cat([x, x], dim=0) if x.shape[0] == 1 else x,)
        # 上界必须显式给:guard 分析会推出 batch < 2^31/单样本元素数(int32 限制),
        # 无上界的 Dim 与之冲突;1024 对训练/评测都绰绰有余
        batch = torch.export.Dim("batch", min=1, max=1024)
        return torch.export.export(
            model, capture_inputs, dynamic_shapes=({0: batch},)).module()
    return torch.export.export_for_training(model, example_inputs).module()


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

    # float 参考导出:2.10 不能直接导出训练态 BN(buffer 突变,见 plan 改动 3),
    # 统一用 eval 深拷贝导出;原模型保持训练态供 QAT 捕获
    float_path = f"{OUT_DIR}/resnet50_float_{TAG}{cfg}.onnx"
    dynamo_export(copy.deepcopy(float_model).eval(), example_inputs, float_path)

    # quantizer
    quantizer = AXQuantizer(args.config)

    exported_model = capture(float_model.train(), example_inputs)
    prepared_model_qat = prepare_qat_pt2e(copy.deepcopy(exported_model), quantizer)

    # 防静默漏注解(plan 风险 3):resnet50 预期插入 ~180 个观察点,阈值取宽松下限
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
    # FP32 区域(未量化)的 conv+BN 不会被 convert 折叠,残留训练态 BN 会让
    # 2.10 导出报 buffer 突变(b_..._running_mean vs getitem);转 eval 后导出。
    # 对全量化配置本调用是无操作;两环境同改以保持等价
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
    print(f"[env] torch {torch.__version__} → {TAG} 实现 | data={args.data} config={args.config} "
          f"steps={args.steps} seed={args.seed}")
    train(args)
