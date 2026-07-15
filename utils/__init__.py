"""utils —— QAT.axera 的统一量化 API(torch 2.6 / 2.10 双版本,单一入口,
唯一的合并实现层)。

用法(用户代码零版本分支):

    from utils import (
        AXQuantizer, capture, prepare_qat_pt2e, convert_pt2e,
        dynamo_export, export_float_reference, simplify_and_fix_4bit_dtype,
    )

版本差异全部收敛在 _compat.py(条件 import)、capture()、dynamo_export()
三个点位,其余为单份实现。设计与决策见 plan_unified_api.md(v2,方案 B';包原名 axquant,后更名 utils 与上游对齐)。
import 本包即自动完成 quantized_decomposed per-channel 的 torchlib 映射注册。
"""
from ._compat import (  # noqa: F401
    IS_TORCH_210,
    BACKEND,
    prepare_qat_pt2e,
    convert_pt2e,
    move_exported_model_to_eval,
    move_exported_model_to_train,
    disable_fake_quant,
    enable_fake_quant,
    disable_observer,
    enable_observer,
)
from .ax_quantizer import (  # noqa: F401
    AXQuantizer,
    load_config,
    get_quantization_config,
    remove_reused_bn_param_hack,
)
from .capture import capture  # noqa: F401
from .train_utils import (  # noqa: F401
    load_model,
    cifar10_data_loaders,
    imagenet_data_loaders,
    train_one_epoch,
    evaluate,
    evaluate_np,
    dynamo_export,
    onnx_simplify,
    export_float_reference,
)
from .quant_utils import (  # noqa: F401
    simplify_and_fix_4bit_dtype,
    load_ptq_calibration_to_qat,
)
from .extract import extract_subgraph  # noqa: F401

# torchlib 自定义映射(quantized_decomposed::quantize/dequantize_per_channel),
# import 即注册,业务侧无需再显式 import 该模块
from . import quantized_decomposed_dequantize_per_channel  # noqa: F401
