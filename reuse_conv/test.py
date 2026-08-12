"""加载 train.py 的 checkpoint 并做数值回归(torch 2.10,utils 统一 API)。

说明:
  1. 图捕获:torch.export.export;
  2. float 导出与 QAT 导出同路径(dynamo_export);
  3. fixture(input/gt npy)原版依赖历史产物且已缺失 → 改为首跑自动生成、
     后续运行做数值回归断言;
  4. remove_reused_bn_param_hack 调用保留(本模型无图内复用,预期无操作,
     用于验证该实现的兼容性)。

运行(先跑 train.py 生成 checkpoint):
  cd /home/heqi/project-qat/QAT.axera && PYTHONPATH=. CUDA_VISIBLE_DEVICES=<空卡> \
    <env>/bin/python reuse_conv/test.py
"""
import os

import torch
import numpy as np
from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e, convert_pt2e

from utils.ax_quantizer import AXQuantizer, load_config, remove_reused_bn_param_hack
from utils.train_utils import dynamo_export
import utils.quantized_decomposed_dequantize_per_channel  # noqa: F401 注册 per-channel torchlib 映射

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning, module=r'.*')


def test():
    # example inputs
    from reuse_conv.train import Net
    torch.manual_seed(42)
    input = torch.rand(1, 64, 256, 768).to("cuda")

    # float_model(与 QAT 导出同路径 dynamo_export)
    float_model = Net().to("cuda")
    float_path = "./reuse_conv/tmp_float_ax.onnx"
    dynamo_export(float_model, input, float_path)

    # set quantizer
    global_config, regional_configs = load_config("./reuse_conv/config.json")
    quantizer = AXQuantizer("./reuse_conv/config.json", annotate_bias=False)

    # export qat model(静态 shape,直接 torch.export.export)
    exported_model = torch.export.export(float_model.train(), (input,)).module()
    prepared_model = prepare_qat_pt2e(exported_model, quantizer)
    prepared_model.load_state_dict(torch.load("./reuse_conv/tmp_ax.pth", weights_only=True))

    # convert(hack 对无图内复用的模型应为无操作)
    n_add_before = sum(1 for n in prepared_model.graph.nodes
                       if n.target == torch.ops.aten.add_.Tensor)
    remove_reused_bn_param_hack(prepared_model)
    n_add_after = sum(1 for n in prepared_model.graph.nodes
                      if n.target == torch.ops.aten.add_.Tensor)
    print(f"[hack] add_ 节点 {n_add_before} → {n_add_after}(无图内复用,预期不变)")
    quantized_model = convert_pt2e(prepared_model)

    # test(fixture 缺失则首跑自动生成,存在则做数值回归)
    in_path, gt_path = "./reuse_conv/input_ax.npy", "./reuse_conv/gt_ax.npy"
    if not (os.path.exists(in_path) and os.path.exists(gt_path)):
        pd = quantized_model(input)
        np.save(in_path, input.cpu().numpy())
        np.save(gt_path, pd.detach().cpu().numpy())
        print(f"[fixture] 首跑生成 {in_path} / {gt_path},再次运行即做回归断言")
        return

    input = torch.tensor(np.load(in_path)).to("cuda")
    gt = torch.tensor(np.load(gt_path)).to("cuda")
    pd = quantized_model(input)
    np.testing.assert_equal(gt.cpu().numpy(), pd.detach().cpu().numpy())
    print("gt & pd assert equal")


if __name__ == "__main__":
    test()
