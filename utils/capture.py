"""统一图捕获入口:吃掉 export_for_training vs torch.export.export 的版本差异。"""
import torch

from ._compat import IS_TORCH_210


def capture(model, example_inputs, dynamic_batch=False, batch_max=1024,
            dynamic_hw=False, hw_min=32, hw_max=4096, hw_multiple_of=1):
    """QAT 图捕获。

    - torch 2.6:走 export_for_training(不检查运行时形状,动态开关均为无操作);
    - torch 2.10:走 torch.export.export;运行时形状与 example_inputs 不一致的
      维度必须显式声明动态,否则命中静态 guard
      (`Guard failed: x.size()[i]==N`):
        * dynamic_batch=True → 第一个输入的第 0 维动态(上界 batch_max);
        * dynamic_hw=True    → 第一个输入的第 2/3 维动态(要求该输入为
          NCHW 4 维;下上界 hw_min/hw_max);
        * hw_multiple_of=k   → 带下采样的网络会产生 H/W 整除性 guard
          (如 MaxPool2d(2) 要求偶数),此时须声明派生维 H=k*_h
          (报 ConstraintViolation 时,export 的 Suggested fixes 会直接
          给出应设的倍数;TinyNet/MaxPool 级取 2,resnet50 全下采样取 32);
    - 自动处理 2.10 的两个坑:0/1 特化(动态维的 example 值必须 >= 2,
      batch=1 的样例内部翻倍;H/W 样例须 >= 2,由断言把关)与 guard 推导的
      int32 乘积上界(动态维必须给显式 max;若 batch_max × hw_max 组合仍触发
      ConstraintViolation,按报错信息收紧对应上界即可)。
    仅第一个输入声明动态(当前全部用例的形态);多输入的其余输入保持静态。
    """
    if not IS_TORCH_210:
        return torch.export.export_for_training(model, example_inputs).module()

    if not dynamic_batch and not dynamic_hw:
        return torch.export.export(model, example_inputs).module()

    x = example_inputs[0]
    dyn = {}
    if dynamic_batch:
        dyn[0] = torch.export.Dim("batch", min=1, max=batch_max)
    if dynamic_hw:
        assert x.dim() == 4, f"dynamic_hw 要求第一个输入为 NCHW 4 维,实际 {x.dim()} 维"
        assert x.shape[2] >= 2 and x.shape[3] >= 2, \
            f"0/1 特化限制:H/W 样例值必须 >= 2,实际 {tuple(x.shape[2:])}"
        assert hw_min >= 1 and hw_max >= max(x.shape[2], x.shape[3]), \
            f"hw_max({hw_max}) 需 >= 样例 H/W {tuple(x.shape[2:])}"
        m = int(hw_multiple_of)
        if m <= 1:
            dyn[2] = torch.export.Dim("height", min=hw_min, max=hw_max)
            dyn[3] = torch.export.Dim("width", min=hw_min, max=hw_max)
        else:
            assert x.shape[2] % m == 0 and x.shape[3] % m == 0, \
                f"样例 H/W {tuple(x.shape[2:])} 须为 hw_multiple_of({m}) 的整数倍"
            _h = torch.export.Dim("_height", min=max(1, hw_min // m), max=hw_max // m)
            _w = torch.export.Dim("_width", min=max(1, hw_min // m), max=hw_max // m)
            dyn[2] = m * _h
            dyn[3] = m * _w

    first = torch.cat([x, x], dim=0) if (dynamic_batch and x.shape[0] == 1) else x
    capture_inputs = (first,) + tuple(example_inputs[1:])
    dynamic_shapes = (dyn,) + tuple(None for _ in example_inputs[1:])
    return torch.export.export(
        model, capture_inputs, dynamic_shapes=dynamic_shapes).module()
