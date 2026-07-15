"""统一图捕获入口:吃掉 export_for_training vs torch.export.export 的版本差异。"""
import torch

from ._compat import IS_TORCH_210


def capture(model, example_inputs, dynamic_shapes=None):
    """QAT 图捕获。

    - torch 2.6:走 export_for_training(不检查运行时形状,dynamic_shapes 忽略);
    - torch 2.10:走 torch.export.export,`dynamic_shapes` **原样透传**
      (torch 原生语义,详见 torch.export.Dim);运行时会变化的维度必须在此
      声明,否则命中静态 guard(`Guard failed: x.size()[i]==N`)。

    示例(与 torch.export 写法完全一致):

        from torch.export import Dim
        # 仅 batch 动态
        gm = capture(m, ex, dynamic_shapes=({0: Dim("batch", min=1, max=1024)},))
        # batch + H/W 动态;下采样网络会产生 H/W 整除性 guard,用派生维表达
        # (如 32 倍下采样的 resnet50:H=32*_h)
        _h = Dim("_h", min=2, max=32)
        _w = Dim("_w", min=2, max=32)
        gm = capture(m, ex, dynamic_shapes=({0: Dim("batch", min=1, max=256),
                                             2: 32 * _h, 3: 32 * _w},))

    2.10 已知坑位(报错时对照):
    1. 动态维必须给显式 max(guard 会推导 int32 乘积上界,无界即冲突);
    2. 整除性约束用派生维(k*_dim);遇 ConstraintViolation 时,报错中的
       Suggested fixes 会直接给出应设写法,照抄即可;
    3. 0/1 特化:动态维的 example 值必须 >= 2——当 dynamic_shapes 为按位置的
       tuple/list 形式时,本函数会把"example 在动态维上为 1"的输入自动沿该维
       翻倍;其他形式(如按参数名的 dict)请自行保证 example >= 2。
    """
    if not IS_TORCH_210:
        return torch.export.export_for_training(model, example_inputs).module()

    if dynamic_shapes is None:
        return torch.export.export(model, example_inputs).module()

    # 0/1 特化自动处理(仅按位置 tuple/list 形式;翻倍仅用于捕获,不影响语义)
    inputs = list(example_inputs)
    if isinstance(dynamic_shapes, (tuple, list)):
        for i, spec in enumerate(dynamic_shapes):
            if not isinstance(spec, dict) or i >= len(inputs) \
                    or not isinstance(inputs[i], torch.Tensor):
                continue
            x = inputs[i]
            for d, v in spec.items():
                if v is not None and isinstance(d, int) and x.shape[d] == 1:
                    x = torch.cat([x, x], dim=d)
            inputs[i] = x
    return torch.export.export(
        model, tuple(inputs), dynamic_shapes=dynamic_shapes).module()
