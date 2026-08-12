"""切子图方法与多个 forward 分别独立推理方法的比较(torch 2.10,utils 统一 API)。

五种推理方式:
  1. 原始完整浮点模型(注:torch.export 的 .module() 与原模型共享参数,
     checkpoint 加载后原地覆盖 → mode1 = QAT 训练权重的浮点推理)
  2. multi stage 浮点模型(权重取自 mode1 被覆盖后的 state_dict,与 1 同源)
  3. 完整量化模型
  4. 由 multi stage 浮点模型每个 stage 分别独立加载参数的量化模型
     ⚠️ 原 demo docstring 已说明:per-channel 权重跨段 channel 数不同,
     无法直接分段加载(需手改 ax_quantizer weight_qscheme=per_tensor 复现
     其精度劣化);本脚本 try/except 优雅降级,加载失败即跳过 mode4
  5. 由完整量化模型切多个子图再分段推理的量化模型(切点按结构自动定位)

预期:1≈2(同源浮点),3≈5(高精度一致);4 若可运行则明显劣化。

运行:
  cd /home/heqi/project-qat/QAT.axera && PYTHONPATH=. CUDA_VISIBLE_DEVICES=<空卡> \
    <env>/bin/python multi_stage/multi_stage_contrast_demo.py
"""
import copy

import torch
from torch.export import Dim
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock
from typing import Callable, List, Optional, Type, Union
from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e, convert_pt2e
from torchao.quantization.pt2e import move_exported_model_to_eval

from utils.ax_quantizer import AXQuantizer
from utils.train_utils import load_model, cifar10_data_loaders, evaluate
from utils.extract import extract_subgraph
import utils.quantized_decomposed_dequantize_per_channel  # noqa: F401 注册 per-channel torchlib 映射

from multi_stage.multi_stage_demo import find_stage_cuts

SEED = 42


class ThreeStageResNet(ResNet):
    """模仿 ResNet 定义一个推理分成三个阶段的 3S ResNet。"""

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(ThreeStageResNet, self).__init__(block=block, layers=layers)

    def _forward_impl_stage1(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        return x

    def _forward_impl_stage2(self, x: Tensor) -> Tensor:
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def _forward_impl_stage3(self, x: Tensor) -> Tensor:
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward1(self, x: Tensor) -> Tensor:
        return self._forward_impl_stage1(x)

    def forward2(self, x: Tensor) -> Tensor:
        return self._forward_impl_stage2(x)

    def forward3(self, x: Tensor) -> Tensor:
        return self._forward_impl_stage3(x)


if __name__ == "__main__":
    torch.manual_seed(SEED)
    ckpt = "./resnet50/checkpoint/last_checkpoint.pth"
    data_loader, data_loader_test = cifar10_data_loaders("dataset/cifar10")
    example_inputs = (torch.rand(2, 3, 224, 224).to("cuda"),)
    quantizer = AXQuantizer("./resnet50/config.json")

    # 准备 1. 原始浮点模型(10 类 fc,种子对齐 checkpoint)
    model = load_model("./resnet50/resnet50_pretrained_float.pth", "resnet50").to("cuda")
    torch.manual_seed(SEED)
    model.fc = torch.nn.Linear(model.fc.in_features, 10).to("cuda")

    # 准备 3. 完整量化模型(load_state_dict 同时把共享参数覆盖进 model)
    dynamic_shapes = ({0: Dim("batch", min=1, max=1024)},)
    exported_model = torch.export.export(
        model.train(), example_inputs, dynamic_shapes=dynamic_shapes,
    ).module()
    prepared_model = prepare_qat_pt2e(exported_model, quantizer)
    prepared_model.load_state_dict(torch.load(ckpt, weights_only=True))
    quantized_model = convert_pt2e(prepared_model)

    # 准备 2. multi stage 浮点模型(权重 = 被覆盖后的 model,与 mode1 同源)
    model3s = ThreeStageResNet(Bottleneck, [3, 4, 6, 3])  # ResNet50
    model3s.fc = torch.nn.Linear(model3s.fc.in_features, 10)
    model3s.load_state_dict(model.state_dict())
    model3s.to("cuda")

    model3s.forward = model3s.forward1
    stage1 = copy.deepcopy(model3s)  # float stage1
    model3s.forward = model3s.forward2
    stage2 = copy.deepcopy(model3s)  # float stage2
    model3s.forward = model3s.forward3
    stage3 = copy.deepcopy(model3s)  # float stage3

    # 准备 4. 每个 stage 分别独立加载参数的量化模型(见文件头 ⚠️)
    mode4_ok = True
    try:
        quant_stages = []
        for stage, shape in ((stage1, (1, 3, 224, 224)),
                             (stage2, (1, 256, 56, 56)),
                             (stage3, (1, 1024, 14, 14))):
            ex_s = (torch.rand(2, *shape[1:]).to("cuda"),)
            gm_s = torch.export.export(
                stage.train(), ex_s, dynamic_shapes=dynamic_shapes,
            ).module()
            prep_s = prepare_qat_pt2e(gm_s, quantizer)
            prep_s.load_state_dict(torch.load(ckpt, weights_only=True), strict=False)
            quant_stages.append(convert_pt2e(prep_s))
        quantized_model_s1, quantized_model_s2, quantized_model_s3 = quant_stages
    except RuntimeError as e:
        mode4_ok = False
        print(f"[mode4] 分段独立加载失败(per-channel 权重跨段不匹配,与原 demo docstring 一致): "
              f"{str(e).splitlines()[0][:120]}")

    # 准备 5. 完整量化模型切子图(切点自动定位,同 multi_stage_demo)
    cuts = find_stage_cuts(quantized_model)
    print(f"[cuts] {cuts}")
    submodule_1 = extract_subgraph(quantized_model, [cuts[0][0]], [cuts[0][1]])
    submodule_2 = extract_subgraph(quantized_model, [cuts[1][0]], [cuts[1][1]])
    submodule_3 = extract_subgraph(quantized_model, [cuts[2][0]], [cuts[2][1]])

    # 推理函数
    def model3s_forward(x):
        stage1.eval(); stage2.eval(); stage3.eval()
        return stage3(stage2(stage1(x)))

    def model3s_quant_forward(x):
        move_exported_model_to_eval(quantized_model_s1)
        move_exported_model_to_eval(quantized_model_s2)
        move_exported_model_to_eval(quantized_model_s3)
        return quantized_model_s3(quantized_model_s2(quantized_model_s1(x)))

    def model3s_submodule_forward(x):
        move_exported_model_to_eval(submodule_1)
        move_exported_model_to_eval(submodule_2)
        move_exported_model_to_eval(submodule_3)
        return submodule_3(submodule_2(submodule_1(x)))

    # 推理前 100 个 batch;完整测试集设 total_size=None
    top1, top5 = evaluate(model.eval(), data_loader_test, total_size=100)
    top1_3s, top5_3s = evaluate(model3s_forward, data_loader_test, total_size=100)
    top1_q, top5_q = evaluate(quantized_model, data_loader_test, total_size=100)
    if mode4_ok:
        top1_3sq, top5_3sq = evaluate(model3s_quant_forward, data_loader_test, total_size=100)
    top1_3ss, top5_3ss = evaluate(model3s_submodule_forward, data_loader_test, total_size=100)

    def to_float(t):
        assert isinstance(t, torch.Tensor)
        return t.cpu().numpy().tolist()
    print(f"model1(完整浮点):        top1:{to_float(top1.avg)}, top5:{to_float(top5.avg)}")
    print(f"model2(3 段浮点):        top1:{to_float(top1_3s.avg)}, top5:{to_float(top5_3s.avg)}")
    print(f"model3(完整量化):        top1:{to_float(top1_q.avg)}, top5:{to_float(top5_q.avg)}")
    if mode4_ok:
        print(f"model4(分段独立加载量化): top1:{to_float(top1_3sq.avg)}, top5:{to_float(top5_3sq.avg)}")
    else:
        print("model4(分段独立加载量化): 跳过(见上方 [mode4] 说明)")
    print(f"model5(切子图分段量化):   top1:{to_float(top1_3ss.avg)}, top5:{to_float(top5_3ss.avg)}")

    assert abs(top1.avg - top1_3s.avg) < 0.5, "mode1 与 mode2 不一致!"
    assert abs(top1_q.avg - top1_3ss.avg) < 0.5, "mode3 与 mode5 不一致!"
    print("[OK] 1≈2(同源浮点)且 3≈5(切子图),与原 demo 预期一致")
