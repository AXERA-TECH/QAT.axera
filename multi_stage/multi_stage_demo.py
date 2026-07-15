"""utils 统一 API 版多段推理 demo(与 multi_stage_demo.py 对应,torch 2.6/2.10 同一份代码)。

相对 2.6 版的改动:
  1. utils 统一 API(内部自动路由 torch.ao/torchao)(双轨惯例,原件不动);
  2. 图捕获:export_for_training → torch.export.export + 动态 batch
     (评测 batch=32,静态 guard 会炸,同 resnet50/train.py);
  3. 数据:机器上没有 ImageNet 训练集 → CIFAR-10(10 类 fc,种子与
     resnet50 训练一致),checkpoint 用全 epoch QAT 产物
     last_checkpoint_2_10.pth(test top1≈94.97);
  4. 切子图起止点:2.6 版硬编码节点名(quantize_per_tensor_default_80 等)
     在 2.10 图中不存在 → 改为按结构自动定位:以 layer 边界 conv
     (conv2d_11 = layer2.0.conv1,conv2d_43 = layer4.0.conv1)的输入向上
     追 DQ/Q,与原 demo 的三段划分(stem+layer1 / layer2+3 / layer4+head)
     一致;分段边界两侧各含完整 Q/DQ(与原 demo "起始和末尾都要有完整的
     量化节点"约定相同)。
  5. 浮点基线说明:torch.export 的 .module() 与原模型共享参数存储,
     prepared.load_state_dict(checkpoint) 会原地覆盖 → mode1 实际是
     「QAT 训练权重的浮点推理」(实测 94.34,非随机水平;2.6 的
     export_for_training 同样共享,行为一致)。本 demo 的验证重点是
     mode3(完整量化) ≈ mode5(切子图分段),实测二者一分不差(94.50)。

运行(qat-dev):
  cd /home/heqi/project-qat/QAT.axera && PYTHONPATH=. CUDA_VISIBLE_DEVICES=<空卡> \
    /home/heqi/miniforge3/envs/torch2.10/bin/python multi_stage/multi_stage_demo_2_10.py
"""
import torch

from utils import (
    prepare_qat_pt2e,
    convert_pt2e,
    move_exported_model_to_eval,
    capture,
    AXQuantizer,
    load_config,
    load_model,
    cifar10_data_loaders,
    evaluate,
    extract_subgraph,
)

SEED = 42


def find_stage_cuts(gm, boundary_conv_idx=(11, 43)):
    """按结构定位三段切点:对每个边界 conv,取其输入 DQ 与该 DQ 的 Q 生产者。

    边界 conv 用**拓扑序位置**定位(第 11 个 = layer2.0.conv1,第 43 个 =
    layer4.0.conv1):convert_pt2e 的 conv-bn 折叠会重建 conv 节点、名字整体
    偏移(实测 2.10 下变成 conv2d_106 起),按名字找不可靠,按 forward 顺序
    位置不变。
    返回 [(start_q_name, end_dq_name), ...] 三段;第一段起点 = 图中第一个
    激活 Q,最后一段终点 = 输出前最后一个 DQ。段边界两侧共享同一 Q/DQ 对
    (原 demo 同款重叠切法,数值上 Q(DQ(t)) 幂等)。
    """
    convs = [n for n in gm.graph.nodes
             if n.op == "call_function" and "conv" in str(n.target)]
    first_q = next(n for n in gm.graph.nodes
                   if n.op == "call_function" and "quantize_per_tensor" in str(n.target)
                   and "dequantize" not in str(n.target))
    output_node = next(n for n in gm.graph.nodes if n.op == "output")
    last_dq = output_node.args[0][0] if isinstance(output_node.args[0], (tuple, list)) \
        else output_node.args[0]

    cuts, starts = [], [first_q.name]
    for idx in boundary_conv_idx:
        conv = convs[idx]
        dq = conv.args[0]           # 边界 DQ(2.10 下共享输入可能有两个 DQ,任取喂此 conv 的)
        q = dq.args[0]              # 边界 Q
        assert "dequantize" in str(dq.target) and "quantize" in str(q.target), \
            f"conv[{idx}]({conv.name}) 上游不是 Q/DQ: {dq.target} / {q.target}"
        cuts.append((starts[-1], dq.name))
        starts.append(q.name)
    cuts.append((starts[-1], last_dq.name))
    return cuts


if __name__ == "__main__":
    torch.manual_seed(SEED)
    # 数据集(CIFAR-10,理由见文件头)
    data_loader, data_loader_test = cifar10_data_loaders("dataset/cifar10")
    example_inputs = (torch.rand(1, 3, 224, 224).to("cuda"),)
    # quantizer
    global_config, regional_configs = load_config("./resnet50/config.json")
    quantizer = AXQuantizer("./resnet50/config.json")

    # float model(10 类 fc,种子与 resnet50/train.py 一致以对上 checkpoint)
    model = load_model("./resnet50/resnet50_pretrained_float.pth", "resnet50").to("cuda")
    torch.manual_seed(SEED)
    model.fc = torch.nn.Linear(model.fc.in_features, 10).to("cuda")

    # quantized model
    exported_model = capture(model.train(), example_inputs, dynamic_batch=True)
    prepared_model = prepare_qat_pt2e(exported_model, quantizer)

    prepared_model.load_state_dict(
        torch.load("./resnet50/checkpoint/last_checkpoint_2_10.pth", weights_only=True))
    quantized_model = convert_pt2e(prepared_model)

    # submodule(切点按结构自动定位,不再硬编码节点名)
    cuts = find_stage_cuts(quantized_model)
    print(f"[cuts] {cuts}")
    submodule_1 = extract_subgraph(quantized_model, [cuts[0][0]], [cuts[0][1]])
    submodule_2 = extract_subgraph(quantized_model, [cuts[1][0]], [cuts[1][1]])
    submodule_3 = extract_subgraph(quantized_model, [cuts[2][0]], [cuts[2][1]])

    def model3s_submodule_forward(x):
        move_exported_model_to_eval(submodule_1)
        move_exported_model_to_eval(submodule_2)
        move_exported_model_to_eval(submodule_3)

        x = submodule_1(x)
        x = submodule_2(x)
        x = submodule_3(x)

        return x

    # 推理前 100 个 batch,快速对比结果;要推理完整测试集设置 total_size=None
    top1, top5 = evaluate(model.eval(), data_loader_test, total_size=100)
    top1_q, top5_q = evaluate(quantized_model, data_loader_test, total_size=100)
    top1_3ss, top5_3ss = evaluate(model3s_submodule_forward, data_loader_test, total_size=100)

    # 打印
    def to_float(t):
        assert isinstance(t, torch.Tensor)
        return t.cpu().numpy().tolist()
    print(f"model1(float, 参数与 prepared 共享已被 checkpoint 覆盖): top1:{to_float(top1.avg)}, top5:{to_float(top5.avg)}")
    print(f"model3(完整量化):                      top1:{to_float(top1_q.avg)}, top5:{to_float(top5_q.avg)}")
    print(f"model5(切子图分三段):                  top1:{to_float(top1_3ss.avg)}, top5:{to_float(top5_3ss.avg)}")
    assert abs(top1_q.avg - top1_3ss.avg) < 0.5, "切子图分段推理与完整量化模型精度不一致!"
    print("[OK] mode3 ≈ mode5,切子图分段推理与完整量化一致")
