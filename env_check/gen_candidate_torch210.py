"""生成 QDQ 候选导出(utils 统一 API 版),供 check_onnx_structure.py 检查。

历史:P0 时代本脚本用 utils_2_10 直连;R4 起收敛到 axquant(同一份代码
torch2.6/2.10 均可运行,文件名保留 torch210 以延续文档引用)。
带注解计数断言,防注解器静默漏注解。

产出:
  env_check/out/tiny_qat_torch210.onnx        raw 导出(dynamo, opset 21, 权重 DQ 保留)
  env_check/out/tiny_qat_torch210_sim.onnx    simplify 后(ir_version 已回写 10)
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch

from utils import (
    AXQuantizer,
    capture,
    prepare_qat_pt2e,
    convert_pt2e,
    move_exported_model_to_eval,
    dynamo_export,
    simplify_and_fix_4bit_dtype,
)


class TinyNet(torch.nn.Module):
    """覆盖 conv+bn+relu / 残差 add / maxpool / gap / linear 的最小结构。"""

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(16, 16, 3, 1, 1)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.pool = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(16, 10)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = x + self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.mean(dim=(2, 3))
        return self.fc(x)


def count_fake_quants(prepared) -> int:
    return sum(1 for node in prepared.graph.nodes
               if node.op == "call_module"
               and str(node.target).startswith("activation_post_process"))


def main():
    out_dir = os.path.join(REPO_ROOT, "env_check", "out")
    os.makedirs(out_dir, exist_ok=True)

    example_inputs = (torch.randn(1, 3, 64, 64),)
    float_model = TinyNet().train()

    quantizer = AXQuantizer(os.path.join(REPO_ROOT, "minimum", "config.json"))

    exported_model = capture(float_model, example_inputs)
    prepared_model = prepare_qat_pt2e(exported_model, quantizer)

    # 防静默漏注解:conv1 单元 / conv2 单元 / add / fc 至少各引入观察点,
    # TinyNet 预期 fake-quant 插入数量 >= 8(输入/权重/输出各若干)
    n_fq = count_fake_quants(prepared_model)
    print(f"[assert] prepared 模型 fake-quant 插入数量: {n_fq}")
    assert n_fq >= 8, f"注解疑似漏失: 仅插入 {n_fq} 个 fake-quant(预期 >= 8)"

    optimizer = torch.optim.SGD(
        [p for p in prepared_model.parameters() if p.requires_grad], lr=1e-3)
    for _ in range(3):
        optimizer.zero_grad()
        prepared_model(*example_inputs).sum().backward()
        optimizer.step()

    quantized_model = convert_pt2e(prepared_model)
    move_exported_model_to_eval(quantized_model)

    qat_path = os.path.join(out_dir, "tiny_qat_torch210.onnx")
    dynamo_export(quantized_model, example_inputs, qat_path)

    sim_path = os.path.join(out_dir, "tiny_qat_torch210_sim.onnx")
    simplify_and_fix_4bit_dtype(qat_path, sim_path)


if __name__ == "__main__":
    main()
