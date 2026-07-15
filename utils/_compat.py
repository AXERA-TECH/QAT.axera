"""utils 版本兼容层:全包唯一的条件 import 块。

torch >= 2.10 → torchao 体系(torch.ao 的 PT2E 在 2.10 已坏,见 plan_torch210.md);
torch <  2.10 → torch.ao 体系(以 2.6 为验证基线)。
官方支持点:2.6 与 >=2.10;2.7–2.9 未经验证,import 时告警。

两套体系的符号改名差异在此统一对齐为 2.6 时代的内部命名
(_WrapperModule/_annotate_*_qspec_map/_get_module_name_filter/
_DerivedObserverOrFakeQuantize),实现层代码零感知;
另提供 get_aten_graph_module_for_pattern() 吃掉 2.6 的 using_training_ir
参数差异(2.10 已移除该参数)。
"""
import warnings

import torch

_ver = tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2])
IS_TORCH_210 = _ver >= (2, 10)
BACKEND = "torchao" if IS_TORCH_210 else "torch.ao"

if not IS_TORCH_210 and _ver != (2, 6):
    warnings.warn(
        f"QAT.axera utils 官方支持 torch 2.6 与 >=2.10,当前 {torch.__version__} 未经验证",
        stacklevel=2,
    )

if IS_TORCH_210:
    from torchao.quantization.pt2e.quantize_pt2e import (  # noqa: F401
        prepare_qat_pt2e,
        convert_pt2e,
    )
    from torchao.quantization.pt2e import (  # noqa: F401
        move_exported_model_to_eval,
        move_exported_model_to_train,
        disable_fake_quant,
        enable_fake_quant,
        disable_observer,
        enable_observer,
        observer,
        ObserverOrFakeQuantize,
        DerivedObserverOrFakeQuantize as _DerivedObserverOrFakeQuantize,
    )
    from torchao.quantization.pt2e.fake_quantize import (  # noqa: F401
        FakeQuantize,
        FusedMovingAvgObsFakeQuantize,
    )
    from torchao.quantization.pt2e.observer import (  # noqa: F401
        HistogramObserver,
        MinMaxObserver,
        MovingAverageMinMaxObserver,
        MovingAveragePerChannelMinMaxObserver,
        PerChannelMinMaxObserver,
        PlaceholderObserver,
    )
    from torchao.quantization.pt2e.quantizer import (  # noqa: F401
        QuantizationAnnotation,
        QuantizationSpec,
        QuantizationSpecBase,
        SharedQuantizationSpec,
        DerivedQuantizationSpec,
        Quantizer,
    )
    from torchao.quantization.pt2e.quantizer.utils import (  # noqa: F401
        annotate_input_qspec_map as _annotate_input_qspec_map,
        annotate_output_qspec as _annotate_output_qspec,
        get_module_name_filter as _get_module_name_filter,
    )
    from torchao.quantization.pt2e.export_utils import (  # noqa: F401
        WrapperModule as _WrapperModule,
    )
    from torchao.quantization.pt2e.utils import (  # noqa: F401
        _get_aten_graph_module_for_pattern as _pattern_impl,
        _is_conv_node,
        _is_conv_transpose_node,
        get_new_attr_name_with_prefix,
    )

    def get_aten_graph_module_for_pattern(pattern, example_inputs, is_cuda, gm):
        # 2.10/torchao 版签名已无 using_training_ir(export 只有 training IR 一条路)
        return _pattern_impl(pattern, example_inputs, is_cuda)

else:
    from torch.ao.quantization.quantize_pt2e import (  # noqa: F401
        prepare_qat_pt2e,
        convert_pt2e,
    )
    from torch.ao.quantization import (  # noqa: F401
        move_exported_model_to_eval,
        move_exported_model_to_train,
        disable_fake_quant,
        enable_fake_quant,
        disable_observer,
        enable_observer,
        observer,
        ObserverOrFakeQuantize,
        _DerivedObserverOrFakeQuantize,
    )
    from torch.ao.quantization.fake_quantize import (  # noqa: F401
        FakeQuantize,
        FusedMovingAvgObsFakeQuantize,
    )
    from torch.ao.quantization.observer import (  # noqa: F401
        HistogramObserver,
        MinMaxObserver,
        MovingAverageMinMaxObserver,
        MovingAveragePerChannelMinMaxObserver,
        PerChannelMinMaxObserver,
        PlaceholderObserver,
    )
    from torch.ao.quantization.quantizer import (  # noqa: F401
        QuantizationAnnotation,
        QuantizationSpec,
        QuantizationSpecBase,
        SharedQuantizationSpec,
        DerivedQuantizationSpec,
        Quantizer,
    )
    from torch.ao.quantization.quantizer.utils import (  # noqa: F401
        _annotate_input_qspec_map,
        _annotate_output_qspec,
        _get_module_name_filter,
    )
    from torch.ao.quantization.pt2e.export_utils import (  # noqa: F401
        _WrapperModule,
    )
    from torch.ao.quantization.pt2e.utils import (  # noqa: F401
        _get_aten_graph_module_for_pattern as _pattern_impl,
        _is_conv_node,
        _is_conv_transpose_node,
    )
    from torch.ao.quantization.fx.utils import (  # noqa: F401
        get_new_attr_name_with_prefix,
    )

    def get_aten_graph_module_for_pattern(pattern, example_inputs, is_cuda, gm):
        # 2.6:pattern 构建需与目标 gm 的 IR 形态一致(training IR 与否)
        from torch._export import gm_using_training_ir

        return _pattern_impl(
            pattern, example_inputs, is_cuda,
            using_training_ir=gm_using_training_ir(gm),
        )
