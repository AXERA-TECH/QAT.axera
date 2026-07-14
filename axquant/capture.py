"""统一图捕获入口:吃掉 export_for_training vs torch.export.export 的版本差异。"""
import torch

from ._compat import IS_TORCH_210


def capture(model, example_inputs, dynamic_batch=False, batch_max=1024):
    """QAT 图捕获。

    - torch 2.6:走 export_for_training(不检查运行时 batch,dynamic_batch 无需);
    - torch 2.10:走 torch.export.export;若训练/评测的 batch 与 example_inputs
      不一致,必须 dynamic_batch=True——2.10 会把 example 的 batch 烙成静态
      guard(`Guard failed: x.size()[0]==N`);
    - dynamic_batch=True 时自动处理 2.10 的两个坑:0/1 特化(动态维的
      example 值必须 >= 2,内部把 batch=1 的样例翻倍)与 guard 推导的
      int32 上界(Dim 必须给显式 max,默认 1024)。
    仅第一个输入的第 0 维声明动态(当前全部用例的形态);多输入的其余
    输入保持静态。
    """
    if not IS_TORCH_210:
        return torch.export.export_for_training(model, example_inputs).module()

    if not dynamic_batch:
        return torch.export.export(model, example_inputs).module()

    x = example_inputs[0]
    first = torch.cat([x, x], dim=0) if x.shape[0] == 1 else x
    capture_inputs = (first,) + tuple(example_inputs[1:])
    batch = torch.export.Dim("batch", min=1, max=batch_max)
    dynamic_shapes = ({0: batch},) + tuple(None for _ in example_inputs[1:])
    return torch.export.export(
        model, capture_inputs, dynamic_shapes=dynamic_shapes).module()
